"""
VAT Validator для Invoice LLM.

Валидация VAT номеров с проверкой checksum.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# VAT форматы и checksum алгоритмы по странам
VAT_FORMATS = {
    'AT': (r'^ATU\d{8}$', 'austria'),
    'BE': (r'^BE[01]\d{9}$', 'belgium'),
    'BG': (r'^BG\d{9,10}$', 'bulgaria'),
    'HR': (r'^HR\d{11}$', 'croatia'),
    'CY': (r'^CY\d{8}[A-Z]$', 'cyprus'),
    'CZ': (r'^CZ\d{8,10}$', 'czech republic'),
    'DK': (r'^DK\d{8}$', 'denmark'),
    'EE': (r'^EE\d{9}$', 'estonia'),
    'FI': (r'^FI\d{8}$', 'finland'),
    'FR': (r'^FR[A-Z0-9]{2}\d{9}$', 'france'),
    'DE': (r'^DE\d{9}$', 'germany'),
    'EL': (r'^EL\d{9}$', 'greece'),
    'GR': (r'^GR\d{9}$', 'greece'),
    'HU': (r'^HU\d{8}$', 'hungary'),
    'IE': (r'^IE\d[A-Z0-9+*]\d{5}[A-Z]{1,2}$', 'ireland'),
    'IT': (r'^IT\d{11}$', 'italy'),
    'LV': (r'^LV\d{11}$', 'latvia'),
    'LT': (r'^LT(\d{9}|\d{12})$', 'lithuania'),
    'LU': (r'^LU\d{8}$', 'luxembourg'),
    'MT': (r'^MT\d{8}$', 'malta'),
    'NL': (r'^NL\d{9}B\d{2}$', 'netherlands'),
    'PL': (r'^PL\d{10}$', 'poland'),
    'PT': (r'^PT\d{9}$', 'portugal'),
    'RO': (r'^RO\d{2,10}$', 'romania'),
    'SK': (r'^SK\d{10}$', 'slovakia'),
    'SI': (r'^SI\d{8}$', 'slovenia'),
    'ES': (r'^ES[A-Z0-9]\d{7}[A-Z0-9]$', 'spain'),
    'SE': (r'^SE\d{12}$', 'sweden'),
    'GB': (r'^GB(\d{9}|\d{12}|GD\d{3}|HA\d{3})$', 'uk'),
    'CH': (r'^CHE\d{9}(MWST|TVA|IVA)?$', 'switzerland'),
    'NO': (r'^NO\d{9}MVA$', 'norway'),
}


class VATValidator:
    """
    Валидатор VAT номеров.

    Использование:
        validator = VATValidator()
        is_valid, country = validator.validate("DE123456789")
    """

    def validate(self, vat: str) -> Tuple[bool, Optional[str], list[str]]:
        """
        Валидирует VAT номер.

        Args:
            vat: VAT номер

        Returns:
            (is_valid, country, errors)
        """
        errors = []

        if not vat:
            return False, None, ["Empty VAT number"]

        # Нормализация
        vat = vat.upper().replace(' ', '').replace('.', '').replace('-', '')

        # Определение страны
        prefix = vat[:2]
        if prefix == 'EL':
            prefix = 'GR'  # Греция использует EL и GR

        if prefix not in VAT_FORMATS:
            # Пробуем 3-буквенный префикс (CHE, ATU)
            prefix3 = vat[:3]
            if prefix3 == 'CHE':
                prefix = 'CH'
            elif prefix3 == 'ATU':
                prefix = 'AT'
            else:
                return False, None, [f"Unknown VAT prefix: {prefix}"]

        pattern, country = VAT_FORMATS[prefix]

        # Проверка формата
        if not re.match(pattern, vat):
            errors.append(f"Invalid VAT format for {country}")

        # Checksum валидация
        checksum_valid = self._validate_checksum(vat, prefix)
        if not checksum_valid:
            errors.append(f"Invalid VAT checksum")

        is_valid = len(errors) == 0

        return is_valid, country, errors

    def _validate_checksum(self, vat: str, prefix: str) -> bool:
        """
        Проверяет checksum VAT номера.

        Args:
            vat: VAT номер
            prefix: Код страны

        Returns:
            True если checksum валиден
        """
        try:
            if prefix == 'DE':
                return self._check_de(vat)
            elif prefix == 'FR':
                return self._check_fr(vat)
            elif prefix == 'IT':
                return self._check_it(vat)
            elif prefix == 'PL':
                return self._check_pl(vat)
            elif prefix == 'NL':
                return self._check_nl(vat)
            elif prefix == 'BE':
                return self._check_be(vat)
            elif prefix == 'ES':
                return self._check_es(vat)
            elif prefix == 'PT':
                return self._check_pt(vat)
            elif prefix == 'AT':
                return self._check_at(vat)
            elif prefix == 'GB':
                return self._check_gb(vat)
            # Для остальных стран - только формат
            return True
        except Exception:
            return False

    def _check_de(self, vat: str) -> bool:
        """Проверка немецкого VAT (DE + 9 цифр)."""
        if len(vat) != 11:
            return False

        digits = vat[2:]
        if not digits.isdigit():
            return False

        # MOD 11 алгоритм
        product = 10
        for digit in digits[:-1]:
            sum_val = (int(digit) + product) % 10
            if sum_val == 0:
                sum_val = 10
            product = (2 * sum_val) % 11

        check = 11 - product
        if check == 10:
            check = 0

        return check == int(digits[-1])

    def _check_fr(self, vat: str) -> bool:
        """Проверка французского VAT."""
        if len(vat) != 13:
            return False

        siren = vat[4:]
        if not siren.isdigit():
            return False

        # Проверка SIREN
        check = int(vat[2:4]) if vat[2:4].isdigit() else -1
        expected = (12 + 3 * (int(siren) % 97)) % 97

        return check == expected

    def _check_it(self, vat: str) -> bool:
        """Проверка итальянского VAT."""
        if len(vat) != 13:
            return False

        digits = vat[2:]
        if not digits.isdigit():
            return False

        # Luhn-like алгоритм
        total = 0
        for i, digit in enumerate(digits[:-1]):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        check = (10 - (total % 10)) % 10
        return check == int(digits[-1])

    def _check_pl(self, vat: str) -> bool:
        """Проверка польского VAT (NIP)."""
        if len(vat) != 12:
            return False

        digits = vat[2:]
        if not digits.isdigit():
            return False

        weights = [6, 5, 7, 2, 3, 4, 5, 6, 7]
        total = sum(int(d) * w for d, w in zip(digits[:-1], weights))
        check = total % 11

        if check == 10:
            return False

        return check == int(digits[-1])

    def _check_nl(self, vat: str) -> bool:
        """Проверка нидерландского VAT."""
        if len(vat) != 14 or vat[-3] != 'B':
            return False

        digits = vat[2:11]
        if not digits.isdigit():
            return False

        # MOD 11 алгоритм
        weights = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        total = sum(int(d) * w for d, w in zip(digits, weights))

        return total % 11 == 0

    def _check_be(self, vat: str) -> bool:
        """Проверка бельгийского VAT."""
        if len(vat) != 12:
            return False

        digits = vat[2:]
        if not digits.isdigit():
            return False

        # MOD 97 алгоритм
        number = int(digits[:8])
        check = int(digits[8:])

        return 97 - (number % 97) == check

    def _check_es(self, vat: str) -> bool:
        """Проверка испанского VAT (NIF/CIF)."""
        if len(vat) != 11:
            return False

        # Упрощённая проверка формата
        return True

    def _check_pt(self, vat: str) -> bool:
        """Проверка португальского VAT."""
        if len(vat) != 11:
            return False

        digits = vat[2:]
        if not digits.isdigit():
            return False

        weights = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        total = sum(int(d) * w for d, w in zip(digits, weights))

        return total % 11 == 0

    def _check_at(self, vat: str) -> bool:
        """Проверка австрийского VAT."""
        if len(vat) != 11 or vat[2] != 'U':
            return False

        digits = vat[3:]
        if not digits.isdigit():
            return False

        # Luhn-like алгоритм
        total = 0
        for i, digit in enumerate(digits[:-1]):
            d = int(digit)
            if i % 2 == 0:
                d = d * 2
                if d > 9:
                    d = d // 10 + d % 10
            total += d

        check = (10 - (total + 4) % 10) % 10
        return check == int(digits[-1])

    def _check_gb(self, vat: str) -> bool:
        """Проверка британского VAT."""
        if len(vat) < 11:
            return False

        digits = vat[2:]

        # GD или HA prefix (government departments)
        if digits.startswith('GD') or digits.startswith('HA'):
            return len(digits) == 5

        if not digits.isdigit():
            return False

        if len(digits) == 9:
            # MOD 97 алгоритм
            weights = [8, 7, 6, 5, 4, 3, 2]
            total = sum(int(d) * w for d, w in zip(digits[:7], weights))
            check = int(digits[7:9])

            # Два варианта check digit
            return (total + check) % 97 == 0 or (total + check + 55) % 97 == 0

        return True

    def get_country(self, vat: str) -> Optional[str]:
        """
        Возвращает страну по VAT номеру.

        Args:
            vat: VAT номер

        Returns:
            Название страны или None
        """
        _, country, _ = self.validate(vat)
        return country


# Singleton
_validator: Optional[VATValidator] = None


def get_vat_validator() -> VATValidator:
    """Возвращает singleton экземпляр."""
    global _validator
    if _validator is None:
        _validator = VATValidator()
    return _validator


def validate_vat(vat: str) -> Tuple[bool, Optional[str], list[str]]:
    """Удобная функция для валидации VAT."""
    return get_vat_validator().validate(vat)
