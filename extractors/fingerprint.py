"""
Document Fingerprinting для Invoice LLM.

Создание уникальных fingerprints документов для кэширования.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentFingerprint:
    """Fingerprint документа."""
    full_hash: str          # Полный hash документа
    header_hash: str        # Hash заголовка (первые 500 символов)
    structure_hash: str     # Hash структуры (позиции VAT, IBAN, дат)
    company_hint: str       # Подсказка названия компании

    @property
    def fingerprint(self) -> str:
        """Основной fingerprint для кэша."""
        return self.full_hash

    def __str__(self) -> str:
        return f"Fingerprint({self.full_hash[:16]}..., company={self.company_hint})"


class FingerprintGenerator:
    """
    Генератор fingerprints для документов.

    Использование:
        generator = FingerprintGenerator()
        fp = generator.generate(text)

        print(f"Fingerprint: {fp.fingerprint}")
        print(f"Company hint: {fp.company_hint}")
    """

    def generate(self, text: str) -> DocumentFingerprint:
        """
        Генерирует fingerprint для документа.

        Args:
            text: Текст документа

        Returns:
            DocumentFingerprint
        """
        if not text:
            return DocumentFingerprint(
                full_hash='empty',
                header_hash='empty',
                structure_hash='empty',
                company_hint='',
            )

        # Нормализуем текст
        normalized = self._normalize(text)

        # Полный hash
        header = normalized[:500]
        structure = self._extract_structure(text)
        full_content = f"{header}|{structure}"
        full_hash = hashlib.sha256(full_content.encode('utf-8')).hexdigest()[:32]

        # Header hash
        header_hash = hashlib.md5(header.encode('utf-8')).hexdigest()[:16]

        # Structure hash
        structure_hash = hashlib.md5(structure.encode('utf-8')).hexdigest()[:16]

        # Company hint
        company_hint = self._extract_company_hint(text)

        return DocumentFingerprint(
            full_hash=full_hash,
            header_hash=header_hash,
            structure_hash=structure_hash,
            company_hint=company_hint or '',
        )

    def _normalize(self, text: str) -> str:
        """Нормализует текст для сравнения."""
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        # Заменяем цифры на placeholder
        text = re.sub(r'\d+', '#', text)
        # Lowercase
        text = text.lower().strip()
        return text

    def _extract_structure(self, text: str) -> str:
        """
        Извлекает структуру документа.

        Возвращает строку с позициями ключевых элементов.
        """
        elements = []

        # VAT номер (позиция в документе / 100)
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

        # Email
        email_match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', text)
        if email_match:
            elements.append(f"email:{email_match.start() // 100}")

        return '|'.join(sorted(elements))

    def _extract_company_hint(self, text: str) -> Optional[str]:
        """
        Извлекает подсказку названия компании.

        Args:
            text: Текст документа

        Returns:
            Название компании или None
        """
        header = text[:500].upper()

        # Паттерны компаний (с юридической формой)
        company_patterns = [
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:GMBH|AG|SA|SAS|SRL|LTD|PLC|INC|BV|NV|SP\.?\s*Z\.?\s*O\.?\s*O)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:ENERGIE|ENERGY|TELECOM|MOBILE|BANK)',
        ]

        for pattern in company_patterns:
            match = re.search(pattern, header)
            if match:
                company = match.group(1).strip()
                if 3 <= len(company) <= 50:
                    return company.lower()

        # Fallback: первые слова с заглавной буквы
        words = re.findall(r'[A-Z][a-z]+', header[:200])
        if words:
            return ' '.join(words[:3]).lower()

        return None

    def similarity(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint) -> float:
        """
        Вычисляет similarity между двумя fingerprints.

        Args:
            fp1: Первый fingerprint
            fp2: Второй fingerprint

        Returns:
            Similarity от 0.0 до 1.0
        """
        score = 0.0

        # Полное совпадение full_hash
        if fp1.full_hash == fp2.full_hash:
            return 1.0

        # Header совпадение
        if fp1.header_hash == fp2.header_hash:
            score += 0.6

        # Structure совпадение
        if fp1.structure_hash == fp2.structure_hash:
            score += 0.3

        # Company hint совпадение
        if fp1.company_hint and fp2.company_hint:
            if fp1.company_hint == fp2.company_hint:
                score += 0.1
            elif fp1.company_hint in fp2.company_hint or fp2.company_hint in fp1.company_hint:
                score += 0.05

        return min(score, 1.0)


# Singleton
_generator: Optional[FingerprintGenerator] = None


def get_fingerprint_generator() -> FingerprintGenerator:
    """Возвращает singleton экземпляр."""
    global _generator
    if _generator is None:
        _generator = FingerprintGenerator()
    return _generator


def generate_fingerprint(text: str) -> DocumentFingerprint:
    """Удобная функция для генерации fingerprint."""
    return get_fingerprint_generator().generate(text)
