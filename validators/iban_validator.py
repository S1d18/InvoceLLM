"""
IBAN Validator для Invoice LLM.

Валидация IBAN с проверкой checksum (MOD 97).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# IBAN длины по странам
IBAN_LENGTHS = {
    'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28,
    'BE': 16, 'BA': 20, 'BR': 29, 'BG': 22, 'CR': 22, 'HR': 21,
    'CY': 28, 'CZ': 24, 'DK': 18, 'DO': 28, 'TL': 23, 'EE': 20,
    'FO': 18, 'FI': 18, 'FR': 27, 'GE': 22, 'DE': 22, 'GI': 23,
    'GR': 27, 'GL': 18, 'GT': 28, 'HU': 28, 'IS': 26, 'IQ': 23,
    'IE': 22, 'IL': 23, 'IT': 27, 'JO': 30, 'KZ': 20, 'XK': 20,
    'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21, 'LT': 20, 'LU': 20,
    'MK': 19, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24,
    'ME': 22, 'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28,
    'PT': 25, 'QA': 29, 'RO': 24, 'LC': 32, 'SM': 27, 'ST': 25,
    'SA': 24, 'RS': 22, 'SC': 31, 'SK': 24, 'SI': 19, 'ES': 24,
    'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26, 'UA': 29, 'AE': 23,
    'GB': 22, 'VA': 22, 'VG': 24,
}

# Маппинг кода страны на название
IBAN_COUNTRIES = {
    'AL': 'albania', 'AD': 'andorra', 'AT': 'austria', 'AZ': 'azerbaijan',
    'BH': 'bahrain', 'BY': 'belarus', 'BE': 'belgium', 'BA': 'bosnia',
    'BR': 'brazil', 'BG': 'bulgaria', 'CR': 'costa rica', 'HR': 'croatia',
    'CY': 'cyprus', 'CZ': 'czech republic', 'DK': 'denmark',
    'EE': 'estonia', 'FI': 'finland', 'FR': 'france', 'GE': 'georgia',
    'DE': 'germany', 'GI': 'gibraltar', 'GR': 'greece', 'HU': 'hungary',
    'IS': 'iceland', 'IE': 'ireland', 'IL': 'israel', 'IT': 'italy',
    'JO': 'jordan', 'KZ': 'kazakhstan', 'KW': 'kuwait', 'LV': 'latvia',
    'LB': 'lebanon', 'LI': 'liechtenstein', 'LT': 'lithuania',
    'LU': 'luxembourg', 'MT': 'malta', 'MC': 'monaco', 'MD': 'moldova',
    'ME': 'montenegro', 'NL': 'netherlands', 'NO': 'norway', 'PK': 'pakistan',
    'PL': 'poland', 'PT': 'portugal', 'QA': 'qatar', 'RO': 'romania',
    'SM': 'san marino', 'SA': 'saudi arabia', 'RS': 'serbia', 'SK': 'slovakia',
    'SI': 'slovenia', 'ES': 'spain', 'SE': 'sweden', 'CH': 'switzerland',
    'TN': 'tunisia', 'TR': 'turkey', 'UA': 'ukraine', 'AE': 'uae', 'GB': 'uk',
}


class IBANValidator:
    """
    Валидатор IBAN.

    Использование:
        validator = IBANValidator()
        is_valid, country = validator.validate("DE89370400440532013000")
    """

    def validate(self, iban: str) -> Tuple[bool, Optional[str], list[str]]:
        """
        Валидирует IBAN.

        Args:
            iban: IBAN номер

        Returns:
            (is_valid, country, errors)
        """
        errors = []

        if not iban:
            return False, None, ["Empty IBAN"]

        # Нормализация
        iban = iban.upper().replace(' ', '').replace('-', '')

        # Проверка базового формата
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', iban):
            return False, None, ["Invalid IBAN format"]

        # Код страны
        country_code = iban[:2]

        if country_code not in IBAN_LENGTHS:
            return False, None, [f"Unknown IBAN country code: {country_code}"]

        country = IBAN_COUNTRIES.get(country_code)

        # Проверка длины
        expected_length = IBAN_LENGTHS[country_code]
        if len(iban) != expected_length:
            errors.append(f"Invalid IBAN length: expected {expected_length}, got {len(iban)}")

        # Checksum валидация (MOD 97)
        if not self._validate_checksum(iban):
            errors.append("Invalid IBAN checksum")

        is_valid = len(errors) == 0

        return is_valid, country, errors

    def _validate_checksum(self, iban: str) -> bool:
        """
        Проверяет IBAN checksum (MOD 97 алгоритм).

        Алгоритм:
        1. Перемещаем первые 4 символа в конец
        2. Заменяем буквы на числа (A=10, B=11, ..., Z=35)
        3. Делим на 97, остаток должен быть 1

        Args:
            iban: IBAN номер

        Returns:
            True если checksum валиден
        """
        # Перемещаем первые 4 символа в конец
        rearranged = iban[4:] + iban[:4]

        # Заменяем буквы на числа
        numeric = ''
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                # A=10, B=11, ..., Z=35
                numeric += str(ord(char) - ord('A') + 10)

        # MOD 97
        try:
            return int(numeric) % 97 == 1
        except ValueError:
            return False

    def get_country(self, iban: str) -> Optional[str]:
        """
        Возвращает страну по IBAN.

        Args:
            iban: IBAN номер

        Returns:
            Название страны или None
        """
        _, country, _ = self.validate(iban)
        return country

    def format_iban(self, iban: str) -> str:
        """
        Форматирует IBAN с пробелами (группы по 4 символа).

        Args:
            iban: IBAN номер

        Returns:
            Отформатированный IBAN
        """
        iban = iban.upper().replace(' ', '').replace('-', '')
        return ' '.join(iban[i:i+4] for i in range(0, len(iban), 4))

    def extract_bban(self, iban: str) -> str:
        """
        Извлекает BBAN (Basic Bank Account Number) из IBAN.

        Args:
            iban: IBAN номер

        Returns:
            BBAN часть
        """
        iban = iban.upper().replace(' ', '').replace('-', '')
        return iban[4:]  # Убираем код страны и check digits


# Singleton
_validator: Optional[IBANValidator] = None


def get_iban_validator() -> IBANValidator:
    """Возвращает singleton экземпляр."""
    global _validator
    if _validator is None:
        _validator = IBANValidator()
    return _validator


def validate_iban(iban: str) -> Tuple[bool, Optional[str], list[str]]:
    """Удобная функция для валидации IBAN."""
    return get_iban_validator().validate(iban)
