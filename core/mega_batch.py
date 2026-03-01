"""
Mega-Batch процессор для инкрементальной обработки 500K PDF.

Возобновляемая обработка с прогресс-трекингом через SQLite,
поддержка множества Colab LLM серверов с round-robin,
автоматическое перемещение файлов в -sort директории.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ProgressDB — SQLite трекер прогресса
# ─────────────────────────────────────────────────────────────────────────────

class ProgressDB:
    """
    SQLite база данных для отслеживания прогресса обработки.

    Отдельная БД от template cache — хранит какие файлы обработаны,
    статистику запусков, ошибки. Позволяет возобновление после рестарта.
    """

    def __init__(self, db_path: str | Path = "data/progress/progress.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row

        self._create_tables()
        self._migrate_tables()

    def _create_tables(self):
        """Создаёт таблицы если не существуют."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_root TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                cache_hits INTEGER DEFAULT 0,
                llm_calls INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_run_id INTEGER,
                file_path TEXT NOT NULL,
                file_path_hash TEXT NOT NULL,
                folder TEXT,
                file_size INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                country TEXT,
                doc_type TEXT,
                company TEXT,
                year INTEGER,
                confidence REAL DEFAULT 0.0,
                source TEXT,
                dest_path TEXT,
                error_msg TEXT,
                processing_time_ms INTEGER DEFAULT 0,
                processed_at TEXT,
                FOREIGN KEY (batch_run_id) REFERENCES batch_runs(id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_file_path_hash
                ON processed_files(file_path_hash);

            CREATE INDEX IF NOT EXISTS idx_status
                ON processed_files(status);

            CREATE INDEX IF NOT EXISTS idx_folder_status
                ON processed_files(folder, status);
        """)
        self.conn.commit()

    def _migrate_tables(self):
        """Миграция: добавление doc_category если отсутствует."""
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(processed_files)").fetchall()]
        if 'doc_category' not in columns:
            logger.info("Migrating progress DB: adding doc_category column")
            self.conn.execute("ALTER TABLE processed_files ADD COLUMN doc_category TEXT")
            self.conn.commit()

    def _hash_path(self, file_path: str | Path) -> str:
        """SHA256 хэш нормализованного пути для O(1) lookup."""
        normalized = str(file_path).replace("\\", "/").lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get_or_create_run(self, source_root: str) -> int:
        """Находит активный run или создаёт новый."""
        row = self.conn.execute(
            "SELECT id FROM batch_runs WHERE source_root = ? AND status = 'running' "
            "ORDER BY started_at DESC LIMIT 1",
            (source_root,)
        ).fetchone()

        if row:
            return row['id']

        cursor = self.conn.execute(
            "INSERT INTO batch_runs (source_root, status) VALUES (?, 'running')",
            (source_root,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def register_files(self, files: list[Path], run_id: int, folder: str) -> int:
        """
        Регистрирует файлы для обработки (idempotent).
        Возвращает количество новых файлов.
        """
        new_count = 0
        batch = []

        for f in files:
            path_hash = self._hash_path(f)
            batch.append((
                run_id,
                str(f),
                path_hash,
                folder,
                f.stat().st_size if f.exists() else 0,
            ))

        # Batch insert с IGNORE для идемпотентности
        for item in batch:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO processed_files "
                    "(batch_run_id, file_path, file_path_hash, folder, file_size, status) "
                    "VALUES (?, ?, ?, ?, ?, 'pending')",
                    item,
                )
                if self.conn.total_changes:
                    new_count += 1
            except sqlite3.IntegrityError:
                pass

        self.conn.commit()

        # Обновляем total_files в run
        total = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM processed_files WHERE batch_run_id = ?",
            (run_id,)
        ).fetchone()['cnt']
        self.conn.execute(
            "UPDATE batch_runs SET total_files = ? WHERE id = ?",
            (total, run_id)
        )
        self.conn.commit()

        return new_count

    def get_pending(self, folder: str = None) -> list[dict]:
        """Возвращает список необработанных файлов."""
        if folder:
            rows = self.conn.execute(
                "SELECT id, file_path, folder FROM processed_files "
                "WHERE folder = ? AND status IN ('pending', 'pending_llm') "
                "ORDER BY id",
                (folder,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, file_path, folder FROM processed_files "
                "WHERE status IN ('pending', 'pending_llm') "
                "ORDER BY id"
            ).fetchall()

        return [dict(r) for r in rows]

    def record_success(
        self,
        file_path: str | Path,
        country: str = None,
        doc_type: str = None,
        doc_category: str = None,
        company: str = None,
        year: int = None,
        confidence: float = 0.0,
        source: str = "",
        dest_path: str = None,
        processing_time_ms: int = 0,
    ):
        """Записывает успешную обработку файла."""
        path_hash = self._hash_path(file_path)
        self.conn.execute(
            "UPDATE processed_files SET "
            "status = 'ok', country = ?, doc_type = ?, doc_category = ?, "
            "company = ?, year = ?, "
            "confidence = ?, source = ?, dest_path = ?, "
            "processing_time_ms = ?, processed_at = datetime('now') "
            "WHERE file_path_hash = ?",
            (country, doc_type, doc_category, company, year, confidence, source,
             dest_path, processing_time_ms, path_hash)
        )
        self.conn.commit()

    def record_error(self, file_path: str | Path, error_msg: str, status: str = 'error'):
        """Записывает ошибку обработки."""
        path_hash = self._hash_path(file_path)
        self.conn.execute(
            "UPDATE processed_files SET status = ?, error_msg = ?, "
            "processed_at = datetime('now') WHERE file_path_hash = ?",
            (status, error_msg, path_hash)
        )
        self.conn.commit()

    def record_pending_llm(self, file_path: str | Path, error_msg: str = ""):
        """Помечает файл как ожидающий LLM (Colab disconnected)."""
        self.record_error(file_path, error_msg, status='pending_llm')

    def get_stats(self, run_id: int = None) -> dict:
        """Возвращает статистику обработки."""
        where = f"WHERE batch_run_id = {run_id}" if run_id else ""

        row = self.conn.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) as ok,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'pending_llm' THEN 1 ELSE 0 END) as pending_llm,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN source = 'template_cache' THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN source = 'llm' THEN 1 ELSE 0 END) as llm_calls,
                AVG(CASE WHEN processing_time_ms > 0 THEN processing_time_ms END) as avg_time_ms
            FROM processed_files {where}
        """).fetchone()

        return dict(row)

    def get_folder_stats(self) -> list[dict]:
        """Статистика по папкам."""
        rows = self.conn.execute("""
            SELECT
                folder,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) as ok,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'pending_llm' THEN 1 ELSE 0 END) as pending_llm,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN source = 'template_cache' THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN source = 'llm' THEN 1 ELSE 0 END) as llm_calls
            FROM processed_files
            GROUP BY folder
            ORDER BY folder
        """).fetchall()

        return [dict(r) for r in rows]

    def reset_folder(self, folder: str) -> int:
        """Сбрасывает прогресс для папки. Возвращает кол-во сброшенных."""
        cursor = self.conn.execute(
            "DELETE FROM processed_files WHERE folder = ?",
            (folder,)
        )
        self.conn.commit()
        return cursor.rowcount

    def reset_errors(self, folder: str = None) -> int:
        """Сбрасывает ошибочные файлы в pending."""
        if folder:
            cursor = self.conn.execute(
                "UPDATE processed_files SET status = 'pending', error_msg = NULL "
                "WHERE folder = ? AND status IN ('error', 'pending_llm')",
                (folder,)
            )
        else:
            cursor = self.conn.execute(
                "UPDATE processed_files SET status = 'pending', error_msg = NULL "
                "WHERE status IN ('error', 'pending_llm')"
            )
        self.conn.commit()
        return cursor.rowcount

    def finish_run(self, run_id: int):
        """Помечает run как завершённый."""
        stats = self.get_stats(run_id)
        self.conn.execute(
            "UPDATE batch_runs SET finished_at = datetime('now'), status = 'finished', "
            "processed_files = ?, cache_hits = ?, llm_calls = ?, errors = ? "
            "WHERE id = ?",
            (stats['ok'], stats['cache_hits'], stats['llm_calls'],
             stats['errors'], run_id)
        )
        self.conn.commit()

    def get_runs(self) -> list[dict]:
        """Возвращает список всех запусков."""
        rows = self.conn.execute(
            "SELECT * FROM batch_runs ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        """Закрывает соединение."""
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# MegaBatchProcessor — оркестратор
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MegaBatchConfig:
    """Конфигурация mega-batch."""
    output_suffix: str = "-sort"
    min_confidence: float = 0.7
    cache_hit_threshold: float = 0.95
    llm_timeout: int = 60
    progress_db: str = "data/progress/progress.db"
    max_errors_before_pause: int = 50
    health_check_interval: int = 30
    move_files: bool = True
    dry_run: bool = False
    cache_only: bool = False
    target_folder: str = None


class MegaBatchProcessor:
    """
    Оркестратор для инкрементальной обработки сотен тысяч PDF.

    Особенности:
    - Возобновление после перезапуска (SQLite прогресс)
    - Поддержка нескольких Colab URL с round-robin
    - Автоматическое ожидание при disconnect
    - Перемещение в -sort директории
    - Graceful shutdown по Ctrl+C
    """

    def __init__(
        self,
        source_root: str | Path,
        config: dict = None,
        config_path: str | Path = None,
        colab_urls: list[str] = None,
        stop_event: threading.Event = None,
    ):
        self.source_root = Path(source_root)
        self._shutdown_requested = False
        self._stop_event = stop_event

        # Загрузка конфигурации
        if config is None:
            config = {}
            if config_path:
                import yaml
                p = Path(config_path)
                if p.exists():
                    with p.open('r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}

        self.config = config
        mb_config = config.get('mega_batch', {})

        self.mb = MegaBatchConfig(
            output_suffix=mb_config.get('output_suffix', '-sort'),
            min_confidence=mb_config.get('min_confidence', 0.7),
            cache_hit_threshold=mb_config.get('cache_hit_threshold', 0.95),
            llm_timeout=mb_config.get('llm_timeout', 60),
            progress_db=mb_config.get('progress_db', 'data/progress/progress.db'),
            max_errors_before_pause=mb_config.get('max_errors_before_pause', 50),
            health_check_interval=mb_config.get('health_check_interval', 30),
            move_files=mb_config.get('move_files', True),
        )

        # Progress DB
        self.progress = ProgressDB(self.mb.progress_db)

        # Инициализация классификатора
        # Если переданы colab_urls — добавляем их как серверы
        if colab_urls:
            servers = config.get('servers', [])
            for i, url in enumerate(colab_urls):
                url = url.strip().rstrip('/')
                # Проверяем что этот URL ещё не добавлен
                existing_urls = [s.get('url', '') for s in servers]
                if url not in existing_urls:
                    servers.insert(0, {
                        'name': f'Colab {i + 1}',
                        'url': url,
                        'priority': 0,
                    })
            config['servers'] = servers

        # Увеличиваем timeout для tunnel
        if 'llm' not in config:
            config['llm'] = {}
        config['llm']['timeout'] = self.mb.llm_timeout

        from .classifier import InvoiceLLMClassifier
        self.classifier = InvoiceLLMClassifier(config=config)

        # Статистика текущей сессии
        self._session_start = None
        self._session_processed = 0
        self._session_cache_hits = 0
        self._session_llm_calls = 0
        self._session_errors = 0
        self._consecutive_errors = 0

    def run(
        self,
        target_folder: str = None,
        dry_run: bool = False,
        cache_only: bool = False,
    ):
        """
        Основной цикл обработки.

        Args:
            target_folder: Обработать только конкретную папку
            dry_run: Только показать план без обработки
            cache_only: Использовать только кэш (без LLM)
        """
        self.mb.dry_run = dry_run
        self.mb.cache_only = cache_only
        self.mb.target_folder = target_folder

        self._session_start = time.time()
        self._shutdown_requested = False

        # Регистрация обработчика Ctrl+C (только из main thread;
        # в GUI остановка идёт через stop_event)
        use_signal = (self._stop_event is None) and (threading.current_thread() is threading.main_thread())
        if use_signal:
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_shutdown)

        try:
            self._run_impl()
        finally:
            if use_signal:
                signal.signal(signal.SIGINT, original_sigint)
            self._print_session_summary()

    def _run_impl(self):
        """Внутренняя реализация run."""
        # 1. Обнаружение папок
        folders = self._discover_folders()

        if not folders:
            print(f"No folders found in {self.source_root}")
            return

        print(f"\n{'='*70}")
        print(f"MEGA-BATCH PROCESSOR")
        print(f"{'='*70}")
        print(f"Source:    {self.source_root}")
        print(f"Folders:   {len(folders)}")
        print(f"Mode:      {'DRY RUN' if self.mb.dry_run else 'CACHE ONLY' if self.mb.cache_only else 'FULL (cache + LLM)'}")
        print(f"Move:      {'yes' if self.mb.move_files and not self.mb.dry_run else 'no'}")
        print(f"{'='*70}\n")

        # 2. Получаем/создаём run
        run_id = self.progress.get_or_create_run(str(self.source_root))

        # 3. Обрабатываем каждую папку
        for folder_path, folder_name in folders:
            if self._is_stopped():
                break

            if self.mb.target_folder and folder_name != self.mb.target_folder:
                continue

            self._process_folder(folder_path, folder_name, run_id)

        # 4. Завершаем run
        if not self._is_stopped():
            self.progress.finish_run(run_id)

    def _discover_folders(self) -> list[tuple[Path, str]]:
        """
        Обнаруживает папки для обработки.

        Ищет директории в source_root, содержащие PDF.
        Если source_root сам содержит PDF — обрабатывает его как одну папку.
        """
        folders = []

        # Проверяем есть ли PDF прямо в source_root
        direct_pdfs = list(self.source_root.glob("*.pdf"))
        if direct_pdfs:
            folders.append((self.source_root, self.source_root.name))

        # Проверяем подпапки
        try:
            for entry in sorted(self.source_root.iterdir()):
                if entry.is_dir() and not entry.name.endswith(self.mb.output_suffix):
                    # Проверяем есть ли PDF в этой папке
                    has_pdf = any(entry.glob("*.pdf"))
                    if has_pdf:
                        folders.append((entry, entry.name))
        except PermissionError as e:
            logger.warning(f"Permission denied listing {self.source_root}: {e}")

        return folders

    def _process_folder(self, folder_path: Path, folder_name: str, run_id: int):
        """Обрабатывает одну папку."""
        print(f"\n{'-'*60}")
        print(f"Folder: {folder_name}")
        print(f"{'-'*60}")

        # 1. Сканируем PDF файлы
        pdf_files = sorted(folder_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")

        if not pdf_files:
            return

        # 2. Регистрируем файлы (idempotent)
        new_count = self.progress.register_files(pdf_files, run_id, folder_name)
        if new_count > 0:
            print(f"Registered {new_count} new files")

        # 3. Получаем необработанные
        pending = self.progress.get_pending(folder_name)
        print(f"Pending: {len(pending)} files")

        if not pending:
            print(f"All files in '{folder_name}' already processed!")
            return

        if self.mb.dry_run:
            print(f"[DRY RUN] Would process {len(pending)} files")
            # Показать sort dir
            sort_dir = self._get_sort_dir(folder_path)
            print(f"[DRY RUN] Output: {sort_dir}")
            return

        # 4. Определяем sort директорию
        sort_dir = self._get_sort_dir(folder_path)
        sort_dir.mkdir(parents=True, exist_ok=True)

        # 5. Обрабатываем файлы
        from extractors import extract_pdf_text, PDFOpenError, PDFBlockedError

        total_pending = len(pending)

        for idx, item in enumerate(pending):
            if self._is_stopped():
                print(f"\nShutdown requested. Progress saved.")
                break

            file_path = Path(item['file_path'])

            # Проверяем существование файла (мог быть уже перемещён)
            if not file_path.exists():
                self.progress.record_error(file_path, "File not found (already moved?)")
                continue

            start_time = time.time()

            try:
                # Извлечение текста
                text = extract_pdf_text(file_path)

                if not text or len(text.strip()) < 20:
                    self.progress.record_error(file_path, "Empty or too short text")
                    self._session_errors += 1
                    continue

                # Классификация
                if self.mb.cache_only:
                    result = self.classifier.classify(text, file_path.name, force_llm=False)
                    # В cache_only режиме: если кэш не дал результат — пропускаем
                    if result.source == 'no_match':
                        self.progress.record_error(
                            file_path, "No cache match (cache-only mode)",
                            status='pending_llm'
                        )
                        continue
                else:
                    result = self.classifier.classify(text, file_path.name, force_llm=True)

                proc_time_ms = int((time.time() - start_time) * 1000)

                # Проверяем all_dead (classify вернёт невалидный результат)
                if (not result.is_valid
                        and result.validation_errors
                        and "All LLM servers are DEAD" in result.validation_errors[0]):
                    print(f"\n[!] All servers are DEAD. Auto-stopping.")
                    if self._stop_event:
                        self._stop_event.set()
                    self.progress.record_pending_llm(file_path, "All servers dead")
                    break

                # Обработка результата
                if result.is_valid and result.confidence >= self.mb.min_confidence:
                    # Успех — перемещаем файл
                    dest = self._build_sort_path(sort_dir, file_path, result)

                    if self.mb.move_files:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(dest))

                    self.progress.record_success(
                        file_path,
                        country=result.country,
                        doc_type=result.doc_type,
                        doc_category=result.doc_category,
                        company=result.company,
                        year=result.year,
                        confidence=result.confidence,
                        source=result.source,
                        dest_path=str(dest),
                        processing_time_ms=proc_time_ms,
                    )

                    if result.source == 'template_cache':
                        self._session_cache_hits += 1
                    else:
                        self._session_llm_calls += 1

                    self._consecutive_errors = 0
                else:
                    # Низкий confidence или невалидный
                    errors = "; ".join(result.validation_errors) if result.validation_errors else "Low confidence"
                    self.progress.record_error(file_path, f"{errors} (conf={result.confidence:.2f})")
                    self._session_errors += 1

                self._session_processed += 1

            except (PDFOpenError, PDFBlockedError) as e:
                self.progress.record_error(file_path, str(e))
                self._session_errors += 1
                self._session_processed += 1

            except (ConnectionError, OSError) as e:
                # Сетевая ошибка — возможно Colab отвалился
                err_str = str(e)
                self.progress.record_pending_llm(file_path, err_str)
                self._session_errors += 1
                self._consecutive_errors += 1

                # Проверяем: все серверы мертвы?
                if self.classifier.llm_cluster.all_dead:
                    dead_n = self.classifier.llm_cluster.dead_count
                    total_n = len(self.classifier.llm_cluster.endpoints)
                    print(f"\n[!] All {dead_n}/{total_n} servers are DEAD. Auto-stopping.")
                    if self._stop_event:
                        self._stop_event.set()
                    break

                if self._consecutive_errors >= 10:
                    # Вероятно Colab отвалился — ждём восстановления
                    if not self._wait_for_colab():
                        print("\nColab not recovered. Saving progress and exiting.")
                        break
                    self._consecutive_errors = 0

            except Exception as e:
                self.progress.record_error(file_path, str(e))
                self._session_errors += 1
                self._session_processed += 1
                self._consecutive_errors += 1
                logger.error(f"Error processing {file_path.name}: {e}")

                # Слишком много ошибок подряд
                if self._consecutive_errors >= self.mb.max_errors_before_pause:
                    print(f"\n[!] {self._consecutive_errors} consecutive errors. Pausing.")
                    if not self._wait_for_colab():
                        break
                    self._consecutive_errors = 0

            # Прогресс-бар
            self._print_progress(idx + 1, total_pending, folder_name)

        print()  # Новая строка после прогресса

    def _get_sort_dir(self, folder_path: Path) -> Path:
        """Возвращает путь к -sort директории."""
        return folder_path.parent / f"{folder_path.name}{self.mb.output_suffix}"

    def _build_sort_path(
        self,
        sort_dir: Path,
        source: Path,
        result,
    ) -> Path:
        """
        Строит путь в -sort директории:
        {sort_dir}/{Country}/{DocCategory}/{DocType}/{Company}/{Year}/filename.pdf
        """
        country = self._safe_path(result.country or "unknown")
        doc_category = self._safe_path(getattr(result, 'doc_category', None) or "other")
        doc_type = self._safe_path(result.doc_type or "unknown")
        company = self._safe_path(result.company or "unknown")
        year = str(result.year) if result.year else "unknown"

        dest_dir = sort_dir / country / doc_category / doc_type / company / year
        dest_path = dest_dir / source.name

        # Разрешаем конфликты имен
        if dest_path.exists():
            stem = source.stem
            suffix = source.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
                if counter > 10000:
                    break

        return dest_path

    def _safe_path(self, name: str) -> str:
        """Нормализует компонент пути для файловой системы."""
        if not name:
            return "unknown"

        # Удаляем запрещённые символы Windows
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip().replace(" ", "_").strip("_")

        # Обрезаем до 50 символов
        if len(name) > 50:
            name = name[:50].rstrip("_")

        return name or "unknown"

    def _wait_for_colab(self) -> bool:
        """
        Ожидает восстановления Colab сервера.

        Поллит /health каждые N секунд.
        Возвращает True если сервер восстановился, False если пользователь прервал.
        """
        print(f"\n{'='*60}")
        print("LLM server disconnected!")
        print("Options:")
        print("  1. Restart Colab notebook and wait")
        print("  2. Press Ctrl+C to save progress and exit")
        print(f"{'='*60}")
        print(f"Checking health every {self.mb.health_check_interval}s...")

        check_count = 0
        while not self._is_stopped():
            time.sleep(self.mb.health_check_interval)
            check_count += 1

            # Проверяем доступность LLM
            if self.classifier.llm_cluster.is_available(force_check=True):
                available = len(self.classifier.llm_cluster._available_endpoints)
                print(f"\nLLM server recovered! ({available} endpoints available)")
                print("Resuming processing...\n")
                return True

            elapsed = check_count * self.mb.health_check_interval
            mins = elapsed // 60
            secs = elapsed % 60
            print(f"  [{mins:02d}:{secs:02d}] Still waiting... (Ctrl+C to exit)", end='\r')

        return False

    def _is_stopped(self) -> bool:
        """Проверяет запрос на остановку (Ctrl+C или внешний stop_event)."""
        if self._shutdown_requested:
            return True
        if self._stop_event and self._stop_event.is_set():
            return True
        return False

    def _handle_shutdown(self, signum, frame):
        """Обработчик Ctrl+C — graceful shutdown."""
        if self._shutdown_requested:
            # Второй Ctrl+C — немедленный выход
            print("\nForce exit!")
            sys.exit(1)

        print("\n\nShutdown requested. Finishing current file...")
        self._shutdown_requested = True

    def _print_progress(self, current: int, total: int, folder: str):
        """Выводит строку прогресса."""
        pct = (current / total * 100) if total > 0 else 0

        # ETA
        elapsed = time.time() - self._session_start
        if self._session_processed > 0:
            rate = self._session_processed / elapsed
            remaining = total - current
            eta_secs = int(remaining / rate) if rate > 0 else 0
            eta_h = eta_secs // 3600
            eta_m = (eta_secs % 3600) // 60
            if eta_h > 0:
                eta_str = f"{eta_h}h{eta_m:02d}m"
            else:
                eta_str = f"{eta_m}m"
        else:
            eta_str = "..."

        # Cache hit rate
        total_done = self._session_cache_hits + self._session_llm_calls
        cache_pct = (self._session_cache_hits / total_done * 100) if total_done > 0 else 0

        # Dead count
        dead_str = ""
        try:
            dead_n = self.classifier.llm_cluster.dead_count
            if dead_n > 0:
                dead_str = f"| Dead: {dead_n} "
        except Exception:
            pass

        # Строка прогресса
        line = (
            f"\r[{current:,}/{total:,}] {pct:.0f}% "
            f"| Cache: {cache_pct:.0f}% "
            f"| LLM: {self._session_llm_calls} "
            f"| Err: {self._session_errors} "
            f"{dead_str}"
            f"| ETA: {eta_str} "
            f"| {folder}"
        )

        # Обрезаем до ширины терминала
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 120
        line = line[:term_width]

        print(line, end='', flush=True)

    def _print_session_summary(self):
        """Выводит итоги сессии."""
        if self._session_start is None:
            return

        elapsed = time.time() - self._session_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        total_done = self._session_cache_hits + self._session_llm_calls
        cache_pct = (self._session_cache_hits / total_done * 100) if total_done > 0 else 0

        print(f"\n\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration:     {mins}m {secs}s")
        print(f"Processed:    {self._session_processed:,}")
        print(f"Cache hits:   {self._session_cache_hits:,} ({cache_pct:.0f}%)")
        print(f"LLM calls:    {self._session_llm_calls:,}")
        print(f"Errors:       {self._session_errors:,}")

        if self._session_processed > 0 and elapsed > 0:
            rate = self._session_processed / elapsed
            print(f"Speed:        {rate:.1f} files/sec")

        # Общий прогресс
        overall = self.progress.get_stats()
        if overall['total'] > 0:
            done_pct = (overall['ok'] or 0) / overall['total'] * 100
            print(f"\nOverall progress: {overall['ok'] or 0:,}/{overall['total']:,} ({done_pct:.1f}%)")
            if overall['pending']:
                print(f"Remaining:    {overall['pending']:,} pending")
            if overall['pending_llm']:
                print(f"Pending LLM:  {overall['pending_llm']:,} (retry with mega-batch retry)")
            if overall['errors']:
                print(f"Errors:       {overall['errors']:,}")

        print(f"{'='*60}\n")

    def get_status(self) -> dict:
        """Возвращает полный статус для CLI."""
        overall = self.progress.get_stats()
        folders = self.progress.get_folder_stats()
        runs = self.progress.get_runs()

        return {
            'overall': overall,
            'folders': folders,
            'runs': runs,
        }
