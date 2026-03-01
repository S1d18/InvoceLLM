"""
Hallucination Guard для Invoice LLM.

Защита от галлюцинаций LLM:
- Проверка что год присутствует в тексте
- Проверка что компания похожа на что-то в тексте
- Валидация VAT/IBAN checksum
- Проверка соответствия страны и VAT prefix
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Any

from .vat_validator import validate_vat
from .iban_validator import validate_iban


@dataclass
class ValidationResult:
    """Результат валидации."""
    is_valid: bool = True
    confidence_penalty: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class HallucinationGuard:
    """
    Защита от галлюцинаций LLM.

    Проверки:
    1. Год должен быть в тексте
    2. Компания должна быть похожа на что-то в тексте
    3. VAT/IBAN должны проходить checksum
    4. Страна должна соответствовать VAT prefix

    Использование:
        guard = HallucinationGuard()
        result = guard.validate(classification_result, original_text)

        if not result.is_valid:
            print(f"Errors: {result.errors}")
    """

    def __init__(
        self,
        config: dict = None,
        check_vat: bool = True,
        check_iban: bool = True,
        check_year: bool = True,
        check_company: bool = True,
        min_company_similarity: float = 0.5,
    ):
        """
        Инициализация guard.

        Args:
            config: Конфигурация
            check_vat: Проверять VAT
            check_iban: Проверять IBAN
            check_year: Проверять год в тексте
            check_company: Проверять компанию
            min_company_similarity: Минимальное сходство компании
        """
        if config:
            validation = config.get('validation', {})
            check_vat = validation.get('check_vat', check_vat)
            check_iban = validation.get('check_iban', check_iban)
            check_year = validation.get('check_year_in_text', check_year)
            check_company = validation.get('check_company_similarity', check_company)
            min_company_similarity = validation.get('min_company_similarity', min_company_similarity)

        self.check_vat = check_vat
        self.check_iban = check_iban
        self.check_year = check_year
        self.check_company = check_company
        self.min_company_similarity = min_company_similarity

    def validate(self, result: Any, original_text: str) -> ValidationResult:
        """
        Валидирует результат классификации.

        Args:
            result: ClassificationResult или LLMResult
            original_text: Оригинальный текст документа

        Returns:
            ValidationResult с ошибками и warnings
        """
        validation = ValidationResult()

        if not original_text:
            return validation

        text_lower = original_text.lower()

        # 1. Проверка года
        if self.check_year:
            year = getattr(result, 'year', None)
            if year:
                self._check_year(year, original_text, validation)

        # 2. Проверка компании
        if self.check_company:
            company = getattr(result, 'company', None)
            if company:
                self._check_company(company, original_text, validation)

        # 3. Проверка VAT
        if self.check_vat:
            # Пробуем разные атрибуты
            vat = (getattr(result, 'vat', None) or
                   getattr(result, 'vat_number', None) or
                   getattr(result, 'extracted_vat', None))
            country = getattr(result, 'country', None)
            if vat:
                self._check_vat(vat, country, validation)

        # 4. Проверка IBAN
        if self.check_iban:
            iban = (getattr(result, 'iban', None) or
                    getattr(result, 'extracted_iban', None))
            if iban:
                self._check_iban(iban, validation)

        # 5. Проверка страны в тексте
        country = getattr(result, 'country', None)
        if country:
            self._check_country_evidence(country, original_text, validation)

        return validation

    def _check_year(self, year: int, text: str, validation: ValidationResult):
        """Проверяет что год присутствует в тексте."""
        year_str = str(year)

        if year_str not in text:
            # Пробуем короткий формат (24 вместо 2024)
            short_year = year_str[-2:]
            if short_year not in text:
                validation.errors.append(f"Year {year} not found in text (possible hallucination)")
                validation.confidence_penalty += 0.3
                validation.is_valid = False
            else:
                validation.warnings.append(f"Year {year} found only as '{short_year}'")
                validation.confidence_penalty += 0.05

    def _check_company(self, company: str, text: str, validation: ValidationResult):
        """Проверяет что компания похожа на что-то в тексте."""
        similarity = self._find_company_in_text(company, text)

        if similarity < self.min_company_similarity:
            validation.warnings.append(f"Company '{company}' may be hallucinated (similarity: {similarity:.0%})")
            validation.confidence_penalty += 0.1
        elif similarity < 0.7:
            validation.warnings.append(f"Company '{company}' has low confidence (similarity: {similarity:.0%})")
            validation.confidence_penalty += 0.05

    def _find_company_in_text(self, company: str, text: str) -> float:
        """
        Ищет компанию в тексте и возвращает similarity.

        Args:
            company: Название компании
            text: Текст документа

        Returns:
            Similarity от 0.0 до 1.0
        """
        company_lower = company.lower()
        text_lower = text.lower()

        # Точное совпадение
        if company_lower in text_lower:
            return 1.0

        # Проверяем отдельные слова
        company_words = set(re.findall(r'\b\w{3,}\b', company_lower))
        if not company_words:
            return 0.0

        found_words = sum(1 for word in company_words if word in text_lower)
        word_similarity = found_words / len(company_words)

        # Проверяем начало названия (первые 2-3 слова)
        first_words = ' '.join(company_lower.split()[:2])
        if len(first_words) >= 4 and first_words in text_lower:
            word_similarity = max(word_similarity, 0.8)

        return word_similarity

    def _check_vat(self, vat: str, country: Optional[str], validation: ValidationResult):
        """Проверяет VAT номер."""
        is_valid, vat_country, errors = validate_vat(vat)

        if not is_valid:
            validation.errors.append(f"Invalid VAT checksum: {vat}")
            validation.confidence_penalty += 0.2
            validation.is_valid = False
        elif country and vat_country and country.lower() != vat_country.lower():
            validation.warnings.append(f"VAT {vat} ({vat_country}) doesn't match country {country}")
            validation.confidence_penalty += 0.1

    def _check_iban(self, iban: str, validation: ValidationResult):
        """Проверяет IBAN."""
        is_valid, _, errors = validate_iban(iban)

        if not is_valid:
            validation.errors.append(f"Invalid IBAN: {iban}")
            validation.confidence_penalty += 0.15
            validation.is_valid = False

    def _check_country_evidence(self, country: str, text: str, validation: ValidationResult):
        """Проверяет есть ли evidence страны в тексте."""
        country_lower = country.lower()
        text_lower = text.lower()

        # Паттерны для стран
        country_indicators = {
            'germany': ['deutschland', 'germany', 'de-', 'berlin', 'münchen', 'hamburg', 'gmbh'],
            'france': ['france', 'francia', 'paris', 'lyon', 'marseille', 'sarl', 'sas'],
            'italy': ['italia', 'italy', 'rome', 'roma', 'milano', 'srl', 'spa'],
            'spain': ['españa', 'spain', 'madrid', 'barcelona', 's.l.', 's.a.'],
            'uk': ['united kingdom', 'england', 'london', 'manchester', 'ltd', 'plc'],
            'poland': ['polska', 'poland', 'warszawa', 'krakow', 'sp. z o.o.'],
            'netherlands': ['nederland', 'netherlands', 'amsterdam', 'rotterdam', 'b.v.'],
            'belgium': ['belgique', 'belgium', 'bruxelles', 'brussels', 'antwerp'],
            'austria': ['österreich', 'austria', 'wien', 'vienna', 'graz'],
            'switzerland': ['schweiz', 'suisse', 'switzerland', 'zürich', 'genève'],
        }

        if country_lower in country_indicators:
            indicators = country_indicators[country_lower]
            found = any(ind in text_lower for ind in indicators)

            if not found:
                # Проверяем VAT prefix
                vat_match = re.search(r'\b([A-Z]{2})\d{8,12}', text)
                if vat_match:
                    vat_prefix = vat_match.group(1)
                    # Маппинг VAT prefix на страну
                    vat_country_map = {
                        'DE': 'germany', 'FR': 'france', 'IT': 'italy',
                        'ES': 'spain', 'GB': 'uk', 'PL': 'poland',
                        'NL': 'netherlands', 'BE': 'belgium', 'AT': 'austria',
                        'CH': 'switzerland',
                    }
                    if vat_country_map.get(vat_prefix) == country_lower:
                        found = True

            if not found:
                validation.warnings.append(f"No strong evidence for country '{country}' in text")
                validation.confidence_penalty += 0.05


# Singleton
_guard: Optional[HallucinationGuard] = None


def get_hallucination_guard(config: dict = None) -> HallucinationGuard:
    """Возвращает singleton экземпляр."""
    global _guard
    if _guard is None:
        _guard = HallucinationGuard(config=config)
    return _guard
