"""
Организатор PDF файлов по директориям.

Создает структуру:
    {output_dir}/{Страна}/{Категория}/{Компания}/{Год}/файл.pdf

С фильтрацией мусорных файлов (невалидные или низкий confidence).
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Tuple

from .classifier import InvoiceLLMClassifier, ClassificationResult

logger = logging.getLogger(__name__)

# Символы, запрещенные в именах файлов/директорий Windows
INVALID_PATH_CHARS = r'<>:"/\|?*'
MAX_PATH_COMPONENT_LENGTH = 50


@dataclass
class OrganizeResult:
    """Результат организации одного файла."""
    source_path: Path
    dest_path: Optional[Path] = None
    status: str = ""  # "success", "skipped", "error", "trash"
    classification: Optional[ClassificationResult] = None
    error: Optional[str] = None


@dataclass
class OrganizeReport:
    """Отчет об организации директории."""
    total_files: int = 0
    successful: int = 0
    skipped: int = 0
    errors: int = 0
    moved_to_trash: int = 0

    by_country: dict = field(default_factory=dict)
    by_doc_type: dict = field(default_factory=dict)

    results: list = field(default_factory=list)

    def add_result(self, result: OrganizeResult):
        """Добавляет результат и обновляет статистику."""
        self.results.append(result)
        self.total_files += 1

        if result.status == "success":
            self.successful += 1
            if result.classification:
                country = result.classification.country or "unknown"
                doc_type = result.classification.doc_type or "unknown"
                self.by_country[country] = self.by_country.get(country, 0) + 1
                self.by_doc_type[doc_type] = self.by_doc_type.get(doc_type, 0) + 1
        elif result.status == "skipped":
            self.skipped += 1
        elif result.status == "error":
            self.errors += 1
        elif result.status == "trash":
            self.moved_to_trash += 1


class PDFOrganizer:
    """
    Организатор PDF файлов по структуре директорий.

    Создает структуру:
        {output_dir}/{Страна}/{Категория}/{Компания}/{Год}/файл.pdf

    Использование:
        organizer = PDFOrganizer(output_dir="/sorted", min_confidence=0.7)
        report = organizer.organize(source_dir="/incoming", move=True)

        # Предпросмотр без изменений
        report = organizer.organize(source_dir="/incoming", dry_run=True)
    """

    def __init__(
        self,
        output_dir: str | Path,
        min_confidence: float = 0.7,
        trash_dir: Optional[str | Path] = None,
        config_path: str | Path = None,
    ):
        """
        Инициализация организатора.

        Args:
            output_dir: Целевая директория для организованных файлов
            min_confidence: Минимальный порог уверенности (файлы ниже → trash)
            trash_dir: Директория для мусора (по умолчанию: {output_dir}/unclassified)
            config_path: Путь к config.yaml
        """
        self.output_dir = Path(output_dir)
        self.min_confidence = min_confidence

        if trash_dir:
            self.trash_dir = Path(trash_dir)
        else:
            self.trash_dir = self.output_dir / "unclassified"

        self.classifier = InvoiceLLMClassifier(config_path=config_path)

        logger.info(
            f"PDFOrganizer initialized: output={self.output_dir}, "
            f"min_confidence={min_confidence}, trash={self.trash_dir}"
        )

    def organize(
        self,
        source_dir: str | Path,
        move: bool = True,
        dry_run: bool = False,
        recursive: bool = True,
        force_llm: bool = False,
        use_trash: bool = True,
    ) -> OrganizeReport:
        """
        Организует PDF файлы из source_dir.

        Args:
            source_dir: Исходная директория с PDF
            move: True=перемещать, False=копировать
            dry_run: Только показать что будет сделано (без изменений)
            recursive: Искать рекурсивно в поддиректориях
            force_llm: Принудительно использовать LLM
            use_trash: Перемещать невалидные файлы в trash_dir

        Returns:
            OrganizeReport с результатами
        """
        source_dir = Path(source_dir)
        report = OrganizeReport()

        # Находим PDF файлы
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(source_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {source_dir}")
            return report

        logger.info(f"Found {len(pdf_files)} PDF files to organize")

        # Создаем директории если не dry_run
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if use_trash:
                self.trash_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline: фоновое извлечение текста + последовательная классификация
        from extractors import extract_pdf_text, PDFOpenError, PDFBlockedError

        # Очередь с ограничением размера (не грузить всё в память)
        queue: Queue[Optional[Tuple[Path, Optional[str], Optional[str]]]] = Queue(maxsize=5)

        def _extract_worker():
            """Фоновый worker: извлекает текст из PDF и кладёт в очередь."""
            for pdf_file in pdf_files:
                try:
                    text = extract_pdf_text(pdf_file)
                    queue.put((pdf_file, text, None))
                except PDFBlockedError as e:
                    queue.put((pdf_file, None, f"PDF protected: {e}"))
                except PDFOpenError as e:
                    queue.put((pdf_file, None, f"Cannot open PDF: {e}"))
                except Exception as e:
                    queue.put((pdf_file, None, str(e)))
            queue.put(None)  # Sentinel: конец данных

        # Запускаем extractor в фоновом потоке
        extractor_thread = Thread(target=_extract_worker, daemon=True)
        extractor_thread.start()

        # Обрабатываем файлы по мере готовности (LLM запросы последовательно)
        while True:
            item = queue.get()
            if item is None:  # Sentinel - все файлы обработаны
                break

            pdf_file, text, extract_error = item
            result = OrganizeResult(source_path=pdf_file)

            try:
                # 1. Проверяем результат извлечения
                if extract_error:
                    result.status = "error"
                    result.error = extract_error
                    report.add_result(result)
                    logger.warning(f"Extraction failed for {pdf_file}: {extract_error}")
                    continue

                # 2. Классифицируем (один запрос к LLM за раз)
                classification = self.classifier.classify(
                    text,
                    pdf_file.name,
                    force_llm=force_llm
                )
                result.classification = classification

                # 3. Проверяем качество
                if not classification.is_valid:
                    if use_trash:
                        result.status = "trash"
                        result.dest_path = self._build_trash_path(pdf_file)
                        if not dry_run:
                            self._copy_or_move(pdf_file, result.dest_path, move)
                    else:
                        result.status = "skipped"
                        result.error = "Invalid classification"
                    report.add_result(result)
                    continue

                if classification.confidence < self.min_confidence:
                    if use_trash:
                        result.status = "trash"
                        result.dest_path = self._build_trash_path(pdf_file)
                        if not dry_run:
                            self._copy_or_move(pdf_file, result.dest_path, move)
                    else:
                        result.status = "skipped"
                        result.error = f"Confidence too low: {classification.confidence:.0%}"
                    report.add_result(result)
                    continue

                # 4. Строим путь назначения
                dest_path = self._build_dest_path(pdf_file, classification)
                result.dest_path = dest_path

                # 5. Копируем/перемещаем
                if not dry_run:
                    self._copy_or_move(pdf_file, dest_path, move)

                result.status = "success"
                report.add_result(result)

                logger.debug(f"Organized: {pdf_file.name} -> {dest_path}")

            except Exception as e:
                result.status = "error"
                result.error = str(e)
                report.add_result(result)
                logger.error(f"Error processing {pdf_file}: {e}")

        extractor_thread.join()

        logger.info(
            f"Organization complete: {report.successful} success, "
            f"{report.moved_to_trash} trash, {report.errors} errors"
        )

        return report

    def _build_dest_path(
        self,
        source: Path,
        classification: ClassificationResult
    ) -> Path:
        """
        Строит путь назначения: {output}/{Страна}/{Категория}/{Компания}/{Год}/файл.pdf
        """
        # Извлекаем компоненты
        country = self._normalize_path_component(classification.country or "unknown")
        doc_type = self._normalize_path_component(classification.doc_type or "unknown")
        company = self._normalize_path_component(classification.company or "unknown")
        year = str(classification.year) if classification.year else "unknown"

        # Строим путь
        dest_dir = self.output_dir / country / doc_type / company / year
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Разрешаем конфликты имен
        dest_path = self._resolve_conflict(dest_dir / source.name)

        return dest_path

    def _build_trash_path(self, source: Path) -> Path:
        """Строит путь в trash директории."""
        dest_path = self._resolve_conflict(self.trash_dir / source.name)
        return dest_path

    def _normalize_path_component(self, name: str) -> str:
        """
        Нормализует компонент пути:
        - Удаляет запрещенные символы
        - Обрезает до максимальной длины
        - Заменяет пробелы на _
        """
        if not name:
            return "unknown"

        # Удаляем запрещенные символы
        for char in INVALID_PATH_CHARS:
            name = name.replace(char, "_")

        # Удаляем множественные подчеркивания
        name = re.sub(r'_+', '_', name)

        # Удаляем пробелы в начале и конце, заменяем внутренние на _
        name = name.strip().replace(" ", "_")

        # Убираем _ в начале и конце
        name = name.strip("_")

        # Обрезаем до максимальной длины
        if len(name) > MAX_PATH_COMPONENT_LENGTH:
            name = name[:MAX_PATH_COMPONENT_LENGTH].rstrip("_")

        return name or "unknown"

    def _resolve_conflict(self, path: Path) -> Path:
        """
        Разрешает конфликты имен добавлением суффикса _1, _2, ...
        """
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

            # Защита от бесконечного цикла
            if counter > 10000:
                raise RuntimeError(f"Too many conflicts for {path}")

    def _copy_or_move(self, source: Path, dest: Path, move: bool):
        """Копирует или перемещает файл."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        if move:
            shutil.move(str(source), str(dest))
            logger.debug(f"Moved: {source} -> {dest}")
        else:
            shutil.copy2(str(source), str(dest))
            logger.debug(f"Copied: {source} -> {dest}")


def organize_pdfs(
    source_dir: str | Path,
    output_dir: str | Path,
    min_confidence: float = 0.7,
    move: bool = True,
    dry_run: bool = False,
    force_llm: bool = False,
    use_trash: bool = True,
    trash_dir: Optional[str | Path] = None,
) -> OrganizeReport:
    """
    Удобная функция для организации PDF файлов.

    Args:
        source_dir: Исходная директория с PDF
        output_dir: Целевая директория
        min_confidence: Минимальный порог уверенности
        move: True=перемещать, False=копировать
        dry_run: Только показать что будет сделано
        force_llm: Принудительно использовать LLM
        use_trash: Перемещать невалидные файлы в trash
        trash_dir: Директория для мусора

    Returns:
        OrganizeReport с результатами
    """
    organizer = PDFOrganizer(
        output_dir=output_dir,
        min_confidence=min_confidence,
        trash_dir=trash_dir,
    )

    return organizer.organize(
        source_dir=source_dir,
        move=move,
        dry_run=dry_run,
        force_llm=force_llm,
        use_trash=use_trash,
    )
