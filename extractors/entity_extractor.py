"""
Entity Extractor для Invoice LLM.

Извлечение структурированных данных из текста:
- VAT номера
- IBAN
- BIC/SWIFT
- Телефоны
- Email
- Валюты
- Даты

Адаптировано из New_sort/ml/recipient_extractor.py.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# === VAT PREFIXES ===
VAT_PREFIXES = {
    'AT': 'austria', 'BE': 'belgium', 'BG': 'bulgaria', 'HR': 'croatia',
    'CY': 'cyprus', 'CZ': 'czech republic', 'DK': 'denmark', 'EE': 'estonia',
    'FI': 'finland', 'FR': 'france', 'DE': 'germany', 'EL': 'greece',
    'GR': 'greece', 'HU': 'hungary', 'IE': 'ireland', 'IT': 'italy',
    'LV': 'latvia', 'LT': 'lithuania', 'LU': 'luxembourg', 'MT': 'malta',
    'NL': 'netherlands', 'PL': 'poland', 'PT': 'portugal', 'RO': 'romania',
    'SK': 'slovakia', 'SI': 'slovenia', 'ES': 'spain', 'SE': 'sweden',
    'GB': 'uk', 'CH': 'switzerland', 'NO': 'norway', 'UA': 'ukraine',
    'RU': 'russia', 'RS': 'serbia', 'TR': 'turkey', 'AU': 'australia',
    'NZ': 'new zealand', 'CA': 'canada', 'US': 'usa',
}

# === IBAN COUNTRIES ===
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

# === BIC COUNTRIES ===
BIC_COUNTRIES = IBAN_COUNTRIES.copy()
BIC_COUNTRIES.update({
    'US': 'usa', 'CA': 'canada', 'AU': 'australia', 'NZ': 'new zealand',
    'JP': 'japan', 'CN': 'china', 'HK': 'hong kong', 'SG': 'singapore',
    'KR': 'south korea', 'IN': 'india', 'ZA': 'south africa', 'MX': 'mexico',
})

# === CURRENCY TO COUNTRY ===
CURRENCY_TO_COUNTRY = {
    'GBP': ('uk', 0.90),
    'PLN': ('poland', 0.95),
    'CZK': ('czech republic', 0.95),
    'HUF': ('hungary', 0.95),
    'RON': ('romania', 0.95),
    'BGN': ('bulgaria', 0.95),
    'HRK': ('croatia', 0.90),
    'RSD': ('serbia', 0.95),
    'UAH': ('ukraine', 0.95),
    'RUB': ('russia', 0.95),
    'CHF': ('switzerland', 0.85),
    'SEK': ('sweden', 0.95),
    'NOK': ('norway', 0.95),
    'DKK': ('denmark', 0.95),
    'ISK': ('iceland', 0.95),
    'TRY': ('turkey', 0.95),
    'JPY': ('japan', 0.95),
    'CNY': ('china', 0.95),
    'KRW': ('south korea', 0.95),
    'INR': ('india', 0.95),
    'BRL': ('brazil', 0.95),
    'MXN': ('mexico', 0.95),
    'ZAR': ('south africa', 0.95),
    'AED': ('uae', 0.95),
    'SAR': ('saudi arabia', 0.95),
    'ILS': ('israel', 0.95),
    'NZD': ('new zealand', 0.85),
    'EUR': (None, 0.30),  # 20+ стран еврозоны
    'USD': ('usa', 0.50),
    'CAD': ('canada', 0.80),
    'AUD': ('australia', 0.80),
}

# BIC blacklist (слова похожие на BIC)
BIC_BLACKLIST = {
    'KOOKHUIS', 'GASTHUIS', 'RAADHUIS', 'CLUBHUIS',
    'POSTHUIS', 'LANDHUIS', 'PAKHUIS', 'TOLHUIS',
    'PAIEMENT', 'DOCUMENT', 'MOVEMENT', 'PAVEMENT', 'STATEMENT',
}


@dataclass
class ExtractedEntities:
    """Результат извлечения сущностей из документа."""

    # VAT
    vat_numbers: list[str] = field(default_factory=list)
    vat_country: Optional[str] = None

    # IBAN
    ibans: list[str] = field(default_factory=list)
    iban_country: Optional[str] = None

    # BIC
    bics: list[str] = field(default_factory=list)
    bic_country: Optional[str] = None

    # Контакты
    phones: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    email_domains: list[str] = field(default_factory=list)

    # Валюты
    currencies: list[str] = field(default_factory=list)
    currency_country: Optional[str] = None

    # Даты
    dates: list[str] = field(default_factory=list)
    year: Optional[int] = None

    # Суммы
    amounts: list[str] = field(default_factory=list)

    # Номер документа
    invoice_number: Optional[str] = None

    # Определённая страна
    country: Optional[str] = None
    country_confidence: float = 0.0
    country_source: str = ""


class EntityExtractor:
    """
    Извлекает структурированные данные из текста документа.

    Использование:
        extractor = EntityExtractor()
        entities = extractor.extract(text)

        print(f"VAT: {entities.vat_numbers}")
        print(f"Country: {entities.country}")
    """

    # Regex паттерны
    VAT_PATTERN = re.compile(
        r'\b(?:VAT|BTW|USt-ID|USt\.?-?IdNr|NIF|TVA|P\.?IVA|IVA|'
        r'ИНН|КПП|ЄДРПОУ|ABN|GST|EIN|Tax ID|Steuernummer|'
        r'Numéro de TVA|Partita IVA|NIP|DIČ|UID)[:\s.-]*'
        r'([A-Z]{2}[\dA-Z]{6,14}|\d{8,14})',
        re.IGNORECASE
    )

    VAT_STANDALONE = re.compile(r'\b([A-Z]{2}\d{8,12}[A-Z]?\d{0,2})\b')

    IBAN_PATTERN = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{4}[A-Z0-9]{4,26})\b')

    BIC_PATTERN = re.compile(r'\b([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b')

    PHONE_PATTERN = re.compile(r'\+(\d{1,3})[\s.\-()]*\d')

    EMAIL_PATTERN = re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')

    CURRENCY_PATTERN = re.compile(
        r'\b(EUR|USD|GBP|PLN|CHF|SEK|NOK|DKK|CZK|HUF|RON|BGN|'
        r'HRK|RSD|UAH|RUB|TRY|JPY|CNY|KRW|INR|BRL|MXN|ZAR|'
        r'AED|SAR|ILS|NZD|CAD|AUD|ISK)\b(?!\s*:)'
    )

    DATE_PATTERN = re.compile(r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b')

    INVOICE_NUMBER_PATTERN = re.compile(
        r'(?:Invoice|Rechnung|Facture|Faktura|Fattura|Račun|Счёт|Счет)'
        r'\s*(?:No\.?|Nr\.?|#|№|Number|Nummer)?[:\s]*'
        r'([A-Z0-9][\w\-/]{2,20})',
        re.IGNORECASE
    )

    AMOUNT_PATTERN = re.compile(
        r'(?:Total|Gesamt|Summe|Montant|Totale|Razem|Итого|Всього)[:\s]*'
        r'([\d\s.,]+)\s*(?:EUR|USD|GBP|PLN)?',
        re.IGNORECASE
    )

    def extract(self, text: str) -> ExtractedEntities:
        """
        Извлекает все сущности из текста.

        Args:
            text: Текст документа

        Returns:
            ExtractedEntities с найденными данными
        """
        entities = ExtractedEntities()

        if not text:
            return entities

        # Извлечение сущностей
        self._extract_vat(text, entities)
        self._extract_iban(text, entities)
        self._extract_bic(text, entities)
        self._extract_phones(text, entities)
        self._extract_emails(text, entities)
        self._extract_currency(text, entities)
        self._extract_dates(text, entities)
        self._extract_invoice_number(text, entities)
        self._extract_amounts(text, entities)

        # Определение страны
        self._determine_country(entities)

        return entities

    def _extract_vat(self, text: str, entities: ExtractedEntities):
        """Извлекает VAT номера."""
        # С меткой
        for match in self.VAT_PATTERN.finditer(text):
            vat = match.group(1).upper().replace(' ', '').replace('.', '')
            prefix = vat[:2]
            if prefix in VAT_PREFIXES and len(vat) >= 8:
                if vat not in entities.vat_numbers:
                    entities.vat_numbers.append(vat)

        # Без метки
        for match in self.VAT_STANDALONE.finditer(text):
            vat = match.group(1)
            prefix = vat[:2]
            if prefix in VAT_PREFIXES and len(vat) >= 8:
                if any(c.isdigit() for c in vat[2:]) and vat not in entities.vat_numbers:
                    entities.vat_numbers.append(vat)

        # Определяем страну по первому VAT
        if entities.vat_numbers:
            prefix = entities.vat_numbers[0][:2]
            entities.vat_country = VAT_PREFIXES.get(prefix)

    def _extract_iban(self, text: str, entities: ExtractedEntities):
        """Извлекает IBAN."""
        for match in self.IBAN_PATTERN.finditer(text):
            iban = match.group(1)
            if len(iban) >= 15 and iban[:2].isalpha() and iban[2:4].isdigit():
                if iban not in entities.ibans:
                    entities.ibans.append(iban)

        if entities.ibans:
            prefix = entities.ibans[0][:2]
            entities.iban_country = IBAN_COUNTRIES.get(prefix)

    def _extract_bic(self, text: str, entities: ExtractedEntities):
        """Извлекает BIC/SWIFT."""
        for match in self.BIC_PATTERN.finditer(text):
            bic = match.group(1)
            if len(bic) in (8, 11) and bic[4:6] in BIC_COUNTRIES:
                if bic not in BIC_BLACKLIST and bic not in entities.bics:
                    entities.bics.append(bic)

        if entities.bics:
            country_code = entities.bics[0][4:6]
            entities.bic_country = BIC_COUNTRIES.get(country_code)

    def _extract_phones(self, text: str, entities: ExtractedEntities):
        """Извлекает телефоны."""
        for match in self.PHONE_PATTERN.finditer(text):
            code = match.group(1)
            if code not in entities.phones:
                entities.phones.append(code)

    def _extract_emails(self, text: str, entities: ExtractedEntities):
        """Извлекает email."""
        for match in self.EMAIL_PATTERN.finditer(text):
            email = match.group(1).lower()
            if email not in entities.emails:
                entities.emails.append(email)
                domain = email.split('@')[1]
                if domain not in entities.email_domains:
                    entities.email_domains.append(domain)

    def _extract_currency(self, text: str, entities: ExtractedEntities):
        """Извлекает валюты."""
        for match in self.CURRENCY_PATTERN.finditer(text):
            currency = match.group(1).upper()
            if currency not in entities.currencies:
                entities.currencies.append(currency)

        # Определяем страну по валюте
        for currency in entities.currencies:
            if currency in CURRENCY_TO_COUNTRY:
                country, conf = CURRENCY_TO_COUNTRY[currency]
                if country and conf > 0.7:
                    entities.currency_country = country
                    break

    def _extract_dates(self, text: str, entities: ExtractedEntities):
        """Извлекает даты."""
        years = []

        for match in self.DATE_PATTERN.finditer(text):
            day, month, year = match.groups()
            date_str = f"{day}.{month}.{year}"

            if date_str not in entities.dates:
                entities.dates.append(date_str)

            year_int = int(year)
            if year_int < 100:
                year_int += 2000 if year_int < 50 else 1900

            if 2000 <= year_int <= 2030:
                years.append(year_int)

        if years:
            year_counts = Counter(years)
            entities.year = year_counts.most_common(1)[0][0]

    def _extract_invoice_number(self, text: str, entities: ExtractedEntities):
        """Извлекает номер счёта."""
        match = self.INVOICE_NUMBER_PATTERN.search(text)
        if match:
            entities.invoice_number = match.group(1)

    def _extract_amounts(self, text: str, entities: ExtractedEntities):
        """Извлекает суммы."""
        for match in self.AMOUNT_PATTERN.finditer(text):
            amount = match.group(1).strip()
            if amount not in entities.amounts:
                entities.amounts.append(amount)

    def _determine_country(self, entities: ExtractedEntities):
        """Определяет страну на основе сущностей."""
        evidences = []

        # VAT (высший приоритет)
        if entities.vat_country:
            evidences.append(('vat', entities.vat_country, 0.95))

        # IBAN
        if entities.iban_country:
            evidences.append(('iban', entities.iban_country, 0.90))

        # BIC
        if entities.bic_country:
            evidences.append(('bic', entities.bic_country, 0.85))

        # Валюта
        if entities.currency_country:
            for currency in entities.currencies:
                if currency in CURRENCY_TO_COUNTRY:
                    country, conf = CURRENCY_TO_COUNTRY[currency]
                    if country:
                        evidences.append(('currency', country, conf))

        if not evidences:
            return

        # Комбинируем evidence
        country_scores: dict[str, float] = {}

        for source, country, confidence in evidences:
            if country not in country_scores:
                country_scores[country] = 0.0

            # Накопление: 1 - (1 - p1) * (1 - p2)
            current = country_scores[country]
            country_scores[country] = 1 - (1 - current) * (1 - confidence)

        # Лучшая страна
        best_country = max(country_scores, key=country_scores.get)
        entities.country = best_country
        entities.country_confidence = country_scores[best_country]

        # Источник
        for source, country, _ in evidences:
            if country == best_country:
                entities.country_source = source
                break


# Singleton
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Возвращает singleton экземпляр."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_entities(text: str) -> ExtractedEntities:
    """Удобная функция для извлечения сущностей."""
    return get_entity_extractor().extract(text)
