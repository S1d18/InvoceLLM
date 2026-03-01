"""
Template Cache с self-learning для Invoice LLM.

Авто-накопление шаблонов с fingerprinting для мгновенной классификации.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """Результат из кэша шаблонов."""
    country: Optional[str] = None
    doc_type: Optional[str] = None
    doc_category: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    confidence: float = 0.0
    fingerprint: str = ""
    template_id: int = 0
    hit_count: int = 0
    last_hit: Optional[datetime] = None

    @property
    def data(self) -> dict:
        """Возвращает данные как словарь."""
        return {
            'country': self.country,
            'doc_type': self.doc_type,
            'doc_category': self.doc_category,
            'company': self.company,
            'year': self.year,
        }


@dataclass
class TemplateStats:
    """Статистика кэша шаблонов."""
    total_templates: int = 0
    total_hits: int = 0
    countries: int = 0
    companies: int = 0
    hit_rate_24h: float = 0.0
    oldest_template: Optional[datetime] = None
    newest_template: Optional[datetime] = None


class TemplateCache:
    """
    Авто-накопление шаблонов с fingerprinting.

    Fingerprint = hash(normalized_company + структура_документа)

    Использование:
        cache = TemplateCache()

        # Поиск в кэше
        result = cache.match(text, filename)
        if result and result.confidence > 0.95:
            return result

        # Сохранение нового шаблона (self-learning)
        cache.learn(text, classification_result)
    """

    # SQL схема
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fingerprint TEXT UNIQUE NOT NULL,
        header_hash TEXT NOT NULL,
        company_hint TEXT,
        country TEXT,
        doc_type TEXT,
        company TEXT,
        confidence REAL DEFAULT 0.0,
        hit_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_hit_at TIMESTAMP,
        metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_fingerprint ON templates(fingerprint);
    CREATE INDEX IF NOT EXISTS idx_header_hash ON templates(header_hash);
    CREATE INDEX IF NOT EXISTS idx_company_hint ON templates(company_hint);
    CREATE INDEX IF NOT EXISTS idx_country ON templates(country);

    CREATE TABLE IF NOT EXISTS stats (
        id INTEGER PRIMARY KEY,
        total_queries INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        last_query_at TIMESTAMP
    );

    INSERT OR IGNORE INTO stats (id, total_queries, total_hits) VALUES (1, 0, 0);
    """

    def __init__(
        self,
        db_path: str | Path = "data/templates/fingerprints.db",
        hit_threshold: float = 0.95,
        learn_threshold: float = 0.85,
        max_age_days: int = 365,
    ):
        """
        Инициализация кэша шаблонов.

        Args:
            db_path: Путь к SQLite базе
            hit_threshold: Минимальный confidence для cache hit
            learn_threshold: Минимальный confidence для сохранения
            max_age_days: Максимальный возраст шаблона в днях
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.hit_threshold = hit_threshold
        self.learn_threshold = learn_threshold
        self.max_age_days = max_age_days

        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Возвращает connection для текущего потока."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Инициализирует базу данных."""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()
        self._migrate_db(conn)

    def _migrate_db(self, conn: sqlite3.Connection):
        """Миграция: добавление doc_category если отсутствует + backfill."""
        # Проверяем есть ли столбец doc_category
        columns = [row[1] for row in conn.execute("PRAGMA table_info(templates)").fetchall()]
        if 'doc_category' not in columns:
            logger.info("Migrating template cache: adding doc_category column")
            conn.execute("ALTER TABLE templates ADD COLUMN doc_category TEXT")

            # Backfill: вычисляем category из doc_type
            from .llm_client import DOC_TYPE_TO_CATEGORY
            rows = conn.execute("SELECT id, doc_type FROM templates WHERE doc_type IS NOT NULL").fetchall()
            for row in rows:
                category = DOC_TYPE_TO_CATEGORY.get(row['doc_type'], 'other')
                conn.execute(
                    "UPDATE templates SET doc_category = ? WHERE id = ?",
                    (category, row['id'])
                )
            conn.commit()
            logger.info(f"Backfilled doc_category for {len(rows)} templates")

    def match(self, text: str, filename: str = "") -> Optional[CachedResult]:
        """
        Поиск в кэше по fingerprint.

        Args:
            text: Текст документа
            filename: Имя файла (опционально)

        Returns:
            CachedResult если найден, иначе None
        """
        if not text or len(text) < 50:
            return None

        conn = self._get_conn()

        # Обновляем статистику запросов
        conn.execute(
            "UPDATE stats SET total_queries = total_queries + 1, last_query_at = ? WHERE id = 1",
            (datetime.now(),)
        )

        # 1. Точное совпадение по fingerprint
        fingerprint = self._compute_fingerprint(text)
        row = conn.execute(
            "SELECT * FROM templates WHERE fingerprint = ?",
            (fingerprint,)
        ).fetchone()

        if row and row['confidence'] >= self.hit_threshold:
            self._record_hit(conn, row['id'])
            return self._row_to_result(row, fingerprint)

        # 2. Поиск по header_hash (первые 500 символов)
        header_hash = self._compute_header_hash(text)
        row = conn.execute(
            "SELECT * FROM templates WHERE header_hash = ? ORDER BY hit_count DESC LIMIT 1",
            (header_hash,)
        ).fetchone()

        if row and row['confidence'] >= self.hit_threshold:
            self._record_hit(conn, row['id'])
            result = self._row_to_result(row, fingerprint)
            result.confidence *= 0.98  # Немного снижаем для header match
            return result

        # 3. Fuzzy match по company_hint
        company_hint = self._extract_company_hint(text)
        if company_hint and len(company_hint) >= 3:
            row = conn.execute(
                """SELECT * FROM templates
                WHERE company_hint LIKE ?
                ORDER BY hit_count DESC, confidence DESC
                LIMIT 1""",
                (f"%{company_hint}%",)
            ).fetchone()

            if row and row['confidence'] >= 0.90:
                self._record_hit(conn, row['id'])
                result = self._row_to_result(row, fingerprint)
                result.confidence *= 0.90  # Снижаем для fuzzy match
                return result

        conn.commit()
        return None

    def learn(self, text: str, result: Any, filename: str = ""):
        """
        Сохранение нового шаблона (self-learning).

        Args:
            text: Текст документа
            result: ClassificationResult с результатами классификации
            filename: Имя файла
        """
        # Проверяем confidence
        confidence = getattr(result, 'country_confidence', 0.0)
        if confidence < self.learn_threshold:
            logger.debug(f"Skipping learn: confidence {confidence:.2f} < {self.learn_threshold}")
            return

        fingerprint = self._compute_fingerprint(text)
        header_hash = self._compute_header_hash(text)
        company_hint = self._extract_company_hint(text)

        conn = self._get_conn()

        # Проверяем существует ли уже
        existing = conn.execute(
            "SELECT id, hit_count, confidence FROM templates WHERE fingerprint = ?",
            (fingerprint,)
        ).fetchone()

        doc_category = getattr(result, 'doc_category', None)

        if existing:
            # Confidence boost: каждое повторное подтверждение поднимает confidence
            # Формула: new = max(old, base) + boost, но не выше 0.99
            # boost = 0.02 за каждое подтверждение (5 повторов: 0.85 -> 0.95)
            old_conf = existing['confidence'] or 0.0
            base = max(old_conf, confidence)
            boosted = min(base + 0.02, 0.99)

            conn.execute(
                """UPDATE templates SET
                    hit_count = hit_count + 1,
                    confidence = ?,
                    doc_category = COALESCE(?, doc_category),
                    last_hit_at = ?
                WHERE id = ?""",
                (boosted, doc_category, datetime.now(), existing['id'])
            )
        else:
            # Создаём новый
            metadata = json.dumps({
                'filename': filename,
                'source': getattr(result, 'country_source', 'unknown'),
                'created_by': 'self_learning',
            })

            conn.execute(
                """INSERT INTO templates
                (fingerprint, header_hash, company_hint, country, doc_type, doc_category, company, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fingerprint,
                    header_hash,
                    company_hint,
                    getattr(result, 'country', None),
                    getattr(result, 'doc_type', None),
                    doc_category,
                    getattr(result, 'company', None),
                    confidence,
                    metadata,
                )
            )
            logger.info(f"Learned new template: {company_hint or fingerprint[:16]}")

        conn.commit()

    def _compute_fingerprint(self, text: str) -> str:
        """
        Вычисляет fingerprint документа.

        Fingerprint = hash of:
        - Normalized header (first 500 chars)
        - Document structure (positions of VAT, dates, amounts)
        """
        header = self._normalize(text[:500])
        structure = self._extract_structure(text)

        content = f"{header}|{structure}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

    def _compute_header_hash(self, text: str) -> str:
        """Вычисляет hash заголовка документа."""
        header = self._normalize(text[:500])
        return hashlib.md5(header.encode('utf-8')).hexdigest()[:16]

    def _normalize(self, text: str) -> str:
        """Нормализует текст для сравнения."""
        # Убираем лишние пробелы и переводы строк
        text = re.sub(r'\s+', ' ', text)
        # Убираем цифры (могут меняться: даты, суммы)
        text = re.sub(r'\d+', '#', text)
        # Приводим к нижнему регистру
        text = text.lower().strip()
        return text

    def _extract_structure(self, text: str) -> str:
        """
        Извлекает структуру документа.

        Возвращает позиции ключевых элементов.
        """
        elements = []

        # VAT номер
        vat_match = re.search(r'[A-Z]{2}\d{8,12}', text)
        if vat_match:
            elements.append(f"vat:{vat_match.start() // 100}")

        # IBAN
        iban_match = re.search(r'[A-Z]{2}\d{2}[A-Z0-9]{10,30}', text)
        if iban_match:
            elements.append(f"iban:{iban_match.start() // 100}")

        # Дата
        date_match = re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)
        if date_match:
            elements.append(f"date:{date_match.start() // 100}")

        # Сумма с валютой
        amount_match = re.search(r'\d+[.,]\d{2}\s*(EUR|USD|GBP|PLN|CHF)', text)
        if amount_match:
            elements.append(f"amount:{amount_match.start() // 100}")

        return '|'.join(sorted(elements))

    def _extract_company_hint(self, text: str) -> Optional[str]:
        """
        Извлекает подсказку компании из текста.

        Ищет в первых 500 символах название компании.
        """
        header = text[:500].upper()

        # Паттерны для компаний
        company_patterns = [
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:GMBH|AG|SA|SAS|SRL|LTD|PLC|INC|BV|NV)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:ENERGIE|ENERGY|TELECOM|MOBILE)',
        ]

        for pattern in company_patterns:
            match = re.search(pattern, header)
            if match:
                company = match.group(1).strip()
                if len(company) >= 3 and len(company) <= 50:
                    return company.lower()

        # Fallback: первые значимые слова
        words = re.findall(r'[A-Z][a-z]+', header[:200])
        if words:
            return ' '.join(words[:3]).lower()

        return None

    def _record_hit(self, conn: sqlite3.Connection, template_id: int):
        """Записывает cache hit."""
        conn.execute(
            "UPDATE templates SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (datetime.now(), template_id)
        )
        conn.execute(
            "UPDATE stats SET total_hits = total_hits + 1 WHERE id = 1"
        )

    def _row_to_result(self, row: sqlite3.Row, fingerprint: str) -> CachedResult:
        """Конвертирует row в CachedResult."""
        # doc_category может отсутствовать в старых БД до миграции
        try:
            doc_category = row['doc_category']
        except (IndexError, KeyError):
            doc_category = None

        return CachedResult(
            country=row['country'],
            doc_type=row['doc_type'],
            doc_category=doc_category,
            company=row['company'],
            confidence=row['confidence'],
            fingerprint=fingerprint,
            template_id=row['id'],
            hit_count=row['hit_count'],
            last_hit=row['last_hit_at'],
        )

    def get_stats(self) -> TemplateStats:
        """Возвращает статистику кэша."""
        conn = self._get_conn()

        stats = TemplateStats()

        # Общее количество шаблонов
        row = conn.execute("SELECT COUNT(*) as cnt FROM templates").fetchone()
        stats.total_templates = row['cnt']

        # Количество стран
        row = conn.execute("SELECT COUNT(DISTINCT country) as cnt FROM templates").fetchone()
        stats.countries = row['cnt']

        # Количество компаний
        row = conn.execute("SELECT COUNT(DISTINCT company) as cnt FROM templates WHERE company IS NOT NULL").fetchone()
        stats.companies = row['cnt']

        # Общее количество hits
        row = conn.execute("SELECT total_queries, total_hits FROM stats WHERE id = 1").fetchone()
        if row:
            stats.total_hits = row['total_hits']
            if row['total_queries'] > 0:
                stats.hit_rate_24h = row['total_hits'] / row['total_queries']

        # Даты
        row = conn.execute("SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM templates").fetchone()
        if row['oldest']:
            stats.oldest_template = row['oldest']
            stats.newest_template = row['newest']

        return stats

    def clear_old(self, days: int = None):
        """Удаляет старые шаблоны."""
        if days is None:
            days = self.max_age_days

        conn = self._get_conn()
        conn.execute(
            "DELETE FROM templates WHERE created_at < datetime('now', ?)",
            (f'-{days} days',)
        )
        conn.commit()

        # VACUUM для освобождения места
        conn.execute("VACUUM")

    def export_templates(self, output_path: str | Path):
        """Экспортирует шаблоны в JSON."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM templates ORDER BY hit_count DESC").fetchall()

        templates = []
        for row in rows:
            tpl = {
                'fingerprint': row['fingerprint'],
                'company_hint': row['company_hint'],
                'country': row['country'],
                'doc_type': row['doc_type'],
                'company': row['company'],
                'confidence': row['confidence'],
                'hit_count': row['hit_count'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
            }
            try:
                tpl['doc_category'] = row['doc_category']
            except (IndexError, KeyError):
                pass
            templates.append(tpl)

        output_path = Path(output_path)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(templates, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(templates)} templates to {output_path}")

    def import_templates(self, input_path: str | Path):
        """Импортирует шаблоны из JSON."""
        input_path = Path(input_path)
        with input_path.open('r', encoding='utf-8') as f:
            templates = json.load(f)

        conn = self._get_conn()
        imported = 0

        for tpl in templates:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO templates
                    (fingerprint, header_hash, company_hint, country, doc_type, doc_category, company, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tpl['fingerprint'],
                        tpl.get('header_hash', ''),
                        tpl.get('company_hint'),
                        tpl.get('country'),
                        tpl.get('doc_type'),
                        tpl.get('doc_category'),
                        tpl.get('company'),
                        tpl.get('confidence', 0.0),
                        json.dumps(tpl.get('metadata', {})),
                    )
                )
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import template: {e}")

        conn.commit()
        logger.info(f"Imported {imported} templates from {input_path}")


# Singleton instance
_cache: Optional[TemplateCache] = None


def get_template_cache(config: dict = None) -> TemplateCache:
    """Возвращает singleton экземпляр TemplateCache."""
    global _cache
    if _cache is None:
        if config:
            _cache = TemplateCache(
                db_path=config.get('cache', {}).get('db_path', 'data/templates/fingerprints.db'),
                hit_threshold=config.get('cache', {}).get('hit_threshold', 0.95),
                learn_threshold=config.get('cache', {}).get('learn_threshold', 0.85),
            )
        else:
            _cache = TemplateCache()
    return _cache
