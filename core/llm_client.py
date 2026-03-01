"""
LLM Cluster Client для Invoice LLM.

Round-robin клиент для кластера llama.cpp серверов.
Перенесено и улучшено из New_sort/ml/llm_classifier.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Any

import requests

logger = logging.getLogger(__name__)

# URL серверов можно задать через переменную окружения
DEFAULT_LLM_URLS = os.environ.get(
    'LLM_SERVER_URL',
    'http://192.168.50.20:8080,http://192.168.50.20:8081,http://192.168.50.20:8082'
).split(',')


# Допустимые значения — 8 категорий
VALID_DOC_CATEGORIES = {
    'financial', 'bank', 'utility', 'telecom', 'tax', 'insurance', 'legal', 'other',
}

# 35 подтипов
VALID_DOC_TYPES = {
    # financial
    'invoice', 'proforma_invoice', 'credit_note', 'debit_note',
    'receipt', 'payment_confirmation', 'dunning_letter', 'purchase_order',
    # bank
    'bank_statement', 'swift_message', 'payment_advice', 'bank_confirmation', 'bank_guarantee',
    # utility
    'electricity', 'gas', 'water', 'heating', 'waste',
    # telecom
    'mobile', 'landline', 'internet', 'tv',
    # tax
    'tax_notice', 'tax_return', 'tax_certificate', 'vat_report',
    # insurance
    'insurance_policy', 'insurance_invoice', 'insurance_claim',
    # legal
    'contract', 'agreement', 'certificate', 'power_of_attorney',
    # other
    'delivery_note', 'report', 'letter', 'unknown',
}

# Маппинг подтип -> категория
DOC_TYPE_TO_CATEGORY = {
    'invoice': 'financial', 'proforma_invoice': 'financial', 'credit_note': 'financial',
    'debit_note': 'financial', 'receipt': 'financial', 'payment_confirmation': 'financial',
    'dunning_letter': 'financial', 'purchase_order': 'financial',
    'bank_statement': 'bank', 'swift_message': 'bank', 'payment_advice': 'bank',
    'bank_confirmation': 'bank', 'bank_guarantee': 'bank',
    'electricity': 'utility', 'gas': 'utility', 'water': 'utility',
    'heating': 'utility', 'waste': 'utility',
    'mobile': 'telecom', 'landline': 'telecom', 'internet': 'telecom', 'tv': 'telecom',
    'tax_notice': 'tax', 'tax_return': 'tax', 'tax_certificate': 'tax', 'vat_report': 'tax',
    'insurance_policy': 'insurance', 'insurance_invoice': 'insurance', 'insurance_claim': 'insurance',
    'contract': 'legal', 'agreement': 'legal', 'certificate': 'legal',
    'power_of_attorney': 'legal',
    'delivery_note': 'other', 'report': 'other', 'letter': 'other', 'unknown': 'other',
}

VALID_COUNTRIES = {
    'germany', 'france', 'italy', 'spain', 'uk', 'poland', 'netherlands',
    'belgium', 'austria', 'switzerland', 'portugal', 'greece', 'czech republic',
    'romania', 'hungary', 'sweden', 'denmark', 'norway', 'finland', 'ireland',
    'usa', 'canada', 'australia', 'new zealand', 'japan', 'south korea',
    'china', 'india', 'brazil', 'mexico', 'argentina', 'chile', 'colombia',
    'south africa', 'uae', 'saudi arabia', 'israel', 'turkey', 'russia',
    'ukraine', 'slovakia', 'slovenia', 'croatia', 'serbia', 'bulgaria',
    'lithuania', 'latvia', 'estonia', 'luxembourg', 'malta', 'cyprus',
}

# Нормализация названий стран
COUNTRY_ALIASES = {
    'united kingdom': 'uk',
    'great britain': 'uk',
    'england': 'uk',
    'scotland': 'uk',
    'wales': 'uk',
    'united states': 'usa',
    'united states of america': 'usa',
    'america': 'usa',
    'the netherlands': 'netherlands',
    'holland': 'netherlands',
    'czech': 'czech republic',
    'czechia': 'czech republic',
    'republic of ireland': 'ireland',
    'deutschland': 'germany',
    'allemagne': 'germany',
    'alemania': 'germany',
    'francia': 'france',
    'frankreich': 'france',
    'espana': 'spain',
    'españa': 'spain',
    'spanien': 'spain',
    'italia': 'italy',
    'italien': 'italy',
    'polska': 'poland',
    'polen': 'poland',
    'osterreich': 'austria',
    'österreich': 'austria',
    'schweiz': 'switzerland',
    'suisse': 'switzerland',
    'svizzera': 'switzerland',
    'belgien': 'belgium',
    'belgique': 'belgium',
    'niederlande': 'netherlands',
    'pays-bas': 'netherlands',
    'portugal': 'portugal',
    'griechenland': 'greece',
    'grèce': 'greece',
}

# Маппинг нестандартных типов документов (backward-compatible алиасы)
DOC_TYPE_MAPPING = {
    # Old generic types -> default subtypes
    'bank': 'bank_statement',
    'telecom': 'mobile',
    'tax': 'tax_notice',
    'other': 'unknown',
    # Aliases -> subtypes
    'electric': 'electricity',
    'power': 'electricity',
    'energy': 'electricity',
    'utility': 'electricity',
    'phone': 'mobile',
    'telephone': 'landline',
    'telecommunications': 'mobile',
    'banking': 'bank_statement',
    'financial': 'invoice',
    'statement': 'bank_statement',
    'bill': 'invoice',
    'insurance': 'insurance_invoice',
    'proforma': 'proforma_invoice',
    'credit note': 'credit_note',
    'debit note': 'debit_note',
    'dunning': 'dunning_letter',
    'purchase order': 'purchase_order',
    'po': 'purchase_order',
    'swift': 'swift_message',
    'guarantee': 'bank_guarantee',
    'policy': 'insurance_policy',
    'claim': 'insurance_claim',
    'delivery': 'delivery_note',
    'poa': 'power_of_attorney',
}


@dataclass
class LLMResult:
    """Результат классификации от LLM."""
    country: Optional[str] = None
    country_confidence: float = 0.0
    doc_type: Optional[str] = None
    doc_type_confidence: float = 0.0
    doc_category: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    raw_response: str = ""
    is_valid: bool = False
    validation_errors: list = field(default_factory=list)
    server_used: str = ""
    processing_time: float = 0.0


class LLMCluster:
    """
    Round-robin клиент для кластера LLM серверов.

    Использование:
        cluster = LLMCluster(config)
        result = cluster.classify(text, filename)
    """

    # Системный промпт
    SYSTEM_PROMPT = """You are a document classification assistant. Your task is to extract structured information from invoice/bill documents.

IMPORTANT RULES:
1. Extract ONLY information that is explicitly present in the text
2. For country: identify the country of the ISSUING company (who sent the invoice), not the recipient
3. For doc_type: classify using one of these specific subtypes:
   - financial: invoice, proforma_invoice, credit_note, debit_note, receipt, payment_confirmation, dunning_letter, purchase_order
   - bank: bank_statement, swift_message, payment_advice, bank_confirmation, bank_guarantee
   - utility: electricity, gas, water, heating, waste
   - telecom: mobile, landline, internet, tv
   - tax: tax_notice, tax_return, tax_certificate, vat_report
   - insurance: insurance_policy, insurance_invoice, insurance_claim
   - legal: contract, agreement, certificate, power_of_attorney
   - other: delivery_note, report, letter, unknown
4. For company: extract the name of the company that ISSUED the document
5. For year: use the INVOICE DATE year, NOT copyright year (ignore "© 2017" in footers)
6. If information is not clearly present, use null

OUTPUT FORMAT (valid JSON only):
{"country": "Country Name", "doc_type": "invoice|electricity|bank_statement|...", "company": "Company Name", "year": 2024}"""

    # Few-shot примеры (нейтральные, без bias на конкретные страны)
    FEW_SHOT_EXAMPLES = [
        # Пример 1: Инвойс — полные данные
        {
            "text": "Company ABC GmbH\nStreet 123\n1010 Vienna, Austria\nInvoice #12345\nDate: 15.03.2024\nVAT: ATU12345678\nTotal: 150.00 EUR",
            "output": '{"country": "Austria", "doc_type": "invoice", "company": "Company ABC GmbH", "year": 2024}'
        },
        # Пример 2: Коммунальный счёт — электричество
        {
            "text": "Energie AG\nStromrechnung\nKundennr: 98765\nAbrechnungszeitraum: Januar 2024\nVerbrauch: 350 kWh\nBetrag: 49.99 EUR",
            "output": '{"country": "Austria", "doc_type": "electricity", "company": "Energie AG", "year": 2024}'
        },
        # Пример 3: Минимум данных — почти всё null
        {
            "text": "Document scan\nPage 1 of 3\nConfidential\n\u00a9 2019",
            "output": '{"country": null, "doc_type": null, "company": null, "year": null}'
        },
    ]

    def __init__(
        self,
        config: dict = None,
        servers: list[dict] = None,
        timeout: float = 30.0,
        max_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_text_length: int = 3000,
        use_few_shot: bool = True,
        workers_per_server: int = 3,
        colab_url: str = None,
    ):
        """
        Инициализация LLM кластера.

        Args:
            config: Конфигурация (если задана, используется вместо отдельных параметров)
            servers: Список серверов [{'host': '...', 'ports': [...]}]
            timeout: Таймаут запроса в секундах
            max_tokens: Максимум токенов в ответе
            temperature: Температура генерации
            top_p: Top-p sampling
            max_text_length: Максимальная длина текста
            use_few_shot: Использовать few-shot примеры
            workers_per_server: Количество workers на сервер
            colab_url: Override URL для Colab (или несколько через запятую)
        """
        # Если передан config, извлекаем параметры
        if config:
            llm_config = config.get('llm', {})
            timeout = llm_config.get('timeout', timeout)
            max_tokens = llm_config.get('max_tokens', max_tokens)
            temperature = llm_config.get('temperature', temperature)
            top_p = llm_config.get('top_p', top_p)
            max_text_length = llm_config.get('max_text_length', max_text_length)
            workers_per_server = llm_config.get('workers_per_server', workers_per_server)
            servers = config.get('servers', servers)

        # Если передан colab_url — добавляем как высокоприоритетный сервер
        if colab_url:
            if servers is None:
                servers = []
            urls = [u.strip() for u in colab_url.split(',')]
            for i, url in enumerate(urls):
                url = url.rstrip('/')
                existing_urls = [s.get('url', '') for s in servers]
                if url not in existing_urls:
                    servers.insert(0, {'name': f'Colab {i + 1}', 'url': url, 'priority': 0})

        # Строим список endpoints
        self.endpoints = self._build_endpoints(servers)

        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_text_length = max_text_length
        self.use_few_shot = use_few_shot
        self.workers_per_server = workers_per_server

        # Round-robin state
        self._index = 0
        self._lock = threading.Lock()
        self._available_endpoints: list[str] = []
        self._failed_endpoints: set[str] = set()  # Временно недоступные
        self._dead_endpoints: set[str] = set()  # Мёртвые (>= dead_threshold ошибок)
        self._endpoint_fail_count: dict[str, int] = {}  # Счётчик ошибок по endpoint
        self._dead_threshold: int = 10  # Порог: 10 фейлов = мёртвый
        self._last_health_check: float = 0
        self._health_check_interval: float = 600.0  # 10 минут

        # Начальная проверка доступности
        self._check_endpoints()

        logger.info(f"LLM Cluster initialized with {len(self.endpoints)} endpoints, {len(self._available_endpoints)} available")

    def _build_endpoints(self, servers: list[dict] = None) -> list[str]:
        """Строит плоский список всех endpoints."""
        endpoints = []

        if servers:
            for server in servers:
                # Поддержка прямого URL (например, для Colab tunnel)
                if 'url' in server:
                    endpoints.append(server['url'].rstrip('/'))
                    continue

                host = server.get('host', 'localhost')
                ports = server.get('ports', [8080])
                ssl = server.get('ssl', False)
                scheme = 'https' if ssl else 'http'
                for port in ports:
                    endpoints.append(f"{scheme}://{host}:{port}")
        else:
            # Fallback на DEFAULT_LLM_URLS
            endpoints = [url.strip().rstrip('/') for url in DEFAULT_LLM_URLS]

        return endpoints

    def _check_endpoints(self, force: bool = False) -> None:
        """
        Проверяет доступность всех endpoints.
        Вызывается при старте и периодически (раз в 10 минут).
        Dead endpoints тоже проверяются — могут восстановиться.
        """
        now = time.time()

        # Проверяем не чаще чем раз в 10 минут (если не force)
        if not force and (now - self._last_health_check) < self._health_check_interval:
            return

        self._last_health_check = now
        new_available = []

        for endpoint in self.endpoints:
            try:
                # Cloudflare tunnel и https endpoints — увеличенный timeout
                is_tunnel = 'trycloudflare.com' in endpoint or endpoint.startswith('https://')
                health_timeout = 10.0 if is_tunnel else 3.0

                response = requests.get(
                    f"{endpoint}/health",
                    timeout=health_timeout,
                    headers={"Connection": "close"},
                )
                if response.status_code == 200:
                    new_available.append(endpoint)
                    # Если был в failed/dead - восстановился
                    if endpoint in self._failed_endpoints:
                        self._failed_endpoints.discard(endpoint)
                        logger.info(f"Endpoint recovered: {endpoint}")
                    if endpoint in self._dead_endpoints:
                        self._dead_endpoints.discard(endpoint)
                        self._endpoint_fail_count.pop(endpoint, None)
                        logger.info(f"Dead endpoint resurrected: {endpoint}")
            except requests.RequestException:
                pass

        # Обновляем список
        old_count = len(self._available_endpoints)
        self._available_endpoints = new_available

        if len(new_available) != old_count:
            logger.info(f"Available LLM servers: {len(new_available)}/{len(self.endpoints)}")

        if not new_available:
            logger.warning(f"No LLM servers available!")

    def _mark_endpoint_failed(self, endpoint: str) -> None:
        """
        Помечает endpoint как временно недоступный.
        При >= dead_threshold последовательных ошибок — помечает как dead.
        """
        with self._lock:
            # Инкремент счётчика ошибок
            self._endpoint_fail_count[endpoint] = self._endpoint_fail_count.get(endpoint, 0) + 1
            fail_count = self._endpoint_fail_count[endpoint]

            self._failed_endpoints.add(endpoint)
            if endpoint in self._available_endpoints:
                self._available_endpoints.remove(endpoint)

            # Проверяем порог dead
            if fail_count >= self._dead_threshold and endpoint not in self._dead_endpoints:
                self._dead_endpoints.add(endpoint)
                logger.error(
                    f"Endpoint DEAD after {fail_count} failures: {endpoint} "
                    f"(dead: {len(self._dead_endpoints)}/{len(self.endpoints)})"
                )
            else:
                logger.warning(
                    f"Endpoint failed ({fail_count}/{self._dead_threshold}): {endpoint} "
                    f"(remaining: {len(self._available_endpoints)})"
                )

    @property
    def all_dead(self) -> bool:
        """True если все endpoints помечены как dead."""
        return len(self._dead_endpoints) >= len(self.endpoints) and len(self.endpoints) > 0

    @property
    def dead_count(self) -> int:
        """Количество dead endpoints."""
        return len(self._dead_endpoints)

    def is_available(self, force_check: bool = False) -> bool:
        """
        Проверяет доступность хотя бы одного LLM сервера.

        Args:
            force_check: Принудительная проверка

        Returns:
            True если хотя бы один сервер доступен
        """
        # Периодическая проверка
        self._check_endpoints(force=force_check)

        return len(self._available_endpoints) > 0

    def _get_next_endpoint(self) -> Optional[str]:
        """Возвращает следующий доступный endpoint по round-robin."""
        # Периодическая проверка в фоне
        self._check_endpoints()

        if not self._available_endpoints:
            return None

        with self._lock:
            endpoint = self._available_endpoints[self._index % len(self._available_endpoints)]
            self._index += 1

        return endpoint

    def classify(
        self,
        text: str,
        filename: str = "",
        max_retries: int = 3,
    ) -> LLMResult:
        """
        Классифицирует документ с помощью LLM.

        Args:
            text: Текст документа
            filename: Имя файла
            max_retries: Максимум попыток при ошибках

        Returns:
            LLMResult с извлечёнными данными
        """
        start_time = time.time()

        if self.all_dead:
            return LLMResult(
                is_valid=False,
                validation_errors=["All LLM servers are DEAD"],
            )

        if not self.is_available():
            return LLMResult(
                is_valid=False,
                validation_errors=["LLM server not available"],
            )

        # Подготовка текста
        prepared_text = self._prepare_text(text, filename)

        # Формирование промпта
        messages = self._build_messages(prepared_text)

        # Retry loop - пробуем разные серверы при ошибках
        last_error = None
        tried_endpoints = set()

        for attempt in range(max_retries):
            endpoint = self._get_next_endpoint()

            if endpoint is None:
                break

            # Пропускаем уже попробованные
            if endpoint in tried_endpoints:
                continue
            tried_endpoints.add(endpoint)

            try:
                response = self._call_llm(endpoint, messages)
                raw_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Успех - парсим и возвращаем
                result = self._parse_response(raw_response, text)
                result.server_used = endpoint
                result.processing_time = time.time() - start_time

                return result

            except requests.exceptions.ConnectionError as e:
                # Возможно keep-alive сброс или tunnel reset
                err_str = str(e)
                logger.warning(f"Connection error to {endpoint}: {e}")

                is_reset = ('RemoteDisconnected' in err_str or
                            'ConnectionResetError' in err_str or
                            'ConnectionReset' in err_str)

                if is_reset and attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 10)
                    logger.info(f"Retrying {endpoint} in {backoff}s (connection reset)...")
                    time.sleep(backoff)
                    tried_endpoints.discard(endpoint)  # Разрешаем повторную попытку
                else:
                    self._mark_endpoint_failed(endpoint)
                last_error = e

            except requests.exceptions.Timeout as e:
                # Таймаут - помечаем и пробуем другой
                logger.warning(f"Timeout from {endpoint}: {e}")
                self._mark_endpoint_failed(endpoint)
                last_error = e

            except requests.exceptions.HTTPError as e:
                # HTTP ошибка (500, 400 и т.д.)
                logger.error(f"HTTP error from {endpoint}: {e}")
                if e.response and e.response.status_code >= 500:
                    # Серверная ошибка — exponential backoff перед пометкой
                    if attempt < max_retries - 1:
                        backoff = min(2 ** attempt, 10)
                        time.sleep(backoff)
                        tried_endpoints.discard(endpoint)
                    else:
                        self._mark_endpoint_failed(endpoint)
                last_error = e

            except (ConnectionResetError, BrokenPipeError) as e:
                # Низкоуровневые сетевые ошибки
                logger.warning(f"Network error to {endpoint}: {e}")
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 10)
                    time.sleep(backoff)
                    tried_endpoints.discard(endpoint)
                else:
                    self._mark_endpoint_failed(endpoint)
                last_error = e

            except Exception as e:
                logger.error(f"LLM request failed on {endpoint}: {e}")
                last_error = e

        # Все попытки исчерпаны
        return LLMResult(
            is_valid=False,
            validation_errors=[f"All LLM servers failed. Last error: {str(last_error)}"],
            server_used=", ".join(tried_endpoints),
        )

    def classify_batch(
        self,
        documents: list[tuple[str, str]],
        max_workers: int = None,
    ) -> list[LLMResult]:
        """
        Параллельная классификация нескольких документов.

        Args:
            documents: Список кортежей (text, filename)
            max_workers: Макс. число потоков

        Returns:
            Список результатов
        """
        if max_workers is None:
            available = len(self._available_endpoints) if self._available_endpoints else len(self.endpoints)
            max_workers = available * self.workers_per_server

        # Создаём задачи с индексами
        tasks = [(i, text, filename) for i, (text, filename) in enumerate(documents)]
        results = [None] * len(documents)

        def process_task(task):
            idx, text, filename = task
            return idx, self.classify(text, filename)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_task, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Task failed: {e}")

        return results

    def _prepare_text(self, text: str, filename: str = "") -> str:
        """
        Подготавливает текст для отправки в LLM.

        Использует semantic truncation вместо positional:
        - Сохраняет header (первые N строк)
        - Ищет строки с ключевыми словами (date, address, VAT, etc.)
        - НЕ обрезает середину документа
        """
        # Очистка контрольных символов (кроме \n, \r, \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Разбиваем на строки, сохраняя структуру
        lines = text.split("\n")

        if len(text) <= self.max_text_length:
            # Текст достаточно короткий — убираем только лишние пробелы внутри строк
            cleaned_lines = [re.sub(r' +', ' ', line).strip() for line in lines]
            result = "\n".join(line for line in cleaned_lines if line)
        else:
            # Semantic truncation
            result = self._smart_truncate(lines, self.max_text_length)

        # НЕ добавляем filename — это может быть data leakage
        return result

    def _smart_truncate(self, lines: list[str], max_length: int) -> str:
        """
        Умная обрезка с приоритетом важных зон документа.

        Приоритеты:
        1. Header (первые 30 строк) — company, logo, address
        2. Строки с ключевыми словами — date, VAT, total, invoice
        3. Остаток header если есть место
        """
        # Ключевые слова для поиска важных строк (многоязычные)
        keywords = [
            # Даты
            "datum", "date", "дата", "data",
            # Счёт/инвойс
            "rechnung", "invoice", "facture", "fattura", "счёт", "счет",
            # VAT/налоги
            "atu", "vat", "ust", "mwst", "iva", "tva", "ндс", "inn", "инн",
            # Суммы
            "total", "summe", "betrag", "amount", "итого", "сумма",
            # Адрес
            "address", "anschrift", "adresse", "адрес",
            # Банк
            "iban", "bic", "swift", "bank",
            # Компания
            "gmbh", "ag", "ltd", "inc", "llc", "ооо", "оао", "зао",
        ]

        # Зона 1: Header (первые 30 строк)
        header_size = min(30, len(lines))
        header_lines = lines[:header_size]

        # Зона 2: Ищем важные строки в остальном тексте
        important_lines = []
        seen_content = set()  # Избегаем дубликатов

        for line in lines[header_size:]:
            line_clean = re.sub(r' +', ' ', line).strip()
            if not line_clean or line_clean in seen_content:
                continue

            line_lower = line_clean.lower()

            # Проверяем наличие ключевых слов
            if any(kw in line_lower for kw in keywords):
                important_lines.append(line_clean)
                seen_content.add(line_clean)

                # Берём также следующую строку (часто значение после метки)
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    next_line = re.sub(r' +', ' ', lines[idx + 1]).strip()
                    if next_line and next_line not in seen_content:
                        important_lines.append(next_line)
                        seen_content.add(next_line)

        # Собираем результат
        header_clean = [re.sub(r' +', ' ', line).strip() for line in header_lines]
        header_clean = [line for line in header_clean if line]

        result_lines = header_clean + ["", "---", ""] + important_lines
        result = "\n".join(result_lines)

        # Если всё равно слишком длинно — обрезаем конец (НЕ середину)
        if len(result) > max_length:
            result = result[:max_length - 20] + "\n...[trimmed]"

        return result

    def _build_messages(self, text: str) -> list[dict]:
        """Формирует сообщения для chat API."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Few-shot примеры
        if self.use_few_shot:
            for example in self.FEW_SHOT_EXAMPLES:
                messages.append({
                    "role": "user",
                    "content": f"Extract information from:\n{example['text']}"
                })
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })

        # Основной запрос
        messages.append({
            "role": "user",
            "content": f"Extract information from:\n{text}"
        })

        return messages

    def _call_llm(self, endpoint: str, messages: list[dict]) -> dict:
        """Отправляет запрос к LLM."""
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        # Connection: close для совместимости с Cloudflare tunnel
        headers = {"Connection": "close"}

        # Увеличенный timeout для tunnel endpoints
        is_tunnel = 'trycloudflare.com' in endpoint or endpoint.startswith('https://')
        call_timeout = max(self.timeout, 60.0) if is_tunnel else self.timeout

        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=call_timeout,
        )
        response.raise_for_status()

        return response.json()

    def _parse_response(self, raw_response: str, original_text: str) -> LLMResult:
        """Парсит и валидирует ответ LLM."""
        result = LLMResult(raw_response=raw_response)

        # Извлечение JSON
        json_data = self._extract_json(raw_response)

        if not json_data:
            result.validation_errors.append("Failed to parse JSON from response")
            return result

        # Извлечение полей
        country = json_data.get("country")
        doc_type = json_data.get("doc_type")
        company = json_data.get("company")
        year = json_data.get("year")

        # Валидация country
        if country:
            normalized = self._normalize_country(country)
            if normalized:
                result.country = normalized
                result.country_confidence = 0.85
            else:
                result.validation_errors.append(f"Unknown country: {country}")

        # Валидация doc_type
        if doc_type:
            doc_type_lower = doc_type.lower().strip().replace(' ', '_')
            if doc_type_lower in VALID_DOC_TYPES:
                result.doc_type = doc_type_lower
                result.doc_type_confidence = 0.85
            else:
                mapped = DOC_TYPE_MAPPING.get(doc_type_lower)
                if mapped:
                    result.doc_type = mapped
                    result.doc_type_confidence = 0.75
                else:
                    result.doc_type = "unknown"
                    result.doc_type_confidence = 0.5
                    result.validation_errors.append(f"Unknown doc_type '{doc_type}', mapped to 'unknown'")

            # Автоматическое вычисление категории из подтипа
            if result.doc_type:
                result.doc_category = DOC_TYPE_TO_CATEGORY.get(result.doc_type, 'other')

        # Валидация company
        if company and isinstance(company, str) and len(company) > 1:
            result.company = company.strip()

        # Валидация year (anti-hallucination)
        if year:
            try:
                year_int = int(year)
                if str(year_int) in original_text:
                    if 1990 <= year_int <= 2030:
                        result.year = year_int
                    else:
                        result.validation_errors.append(f"Year {year_int} out of valid range")
                else:
                    result.validation_errors.append(f"Year {year_int} not found in text (hallucination?)")
            except (ValueError, TypeError):
                result.validation_errors.append(f"Invalid year format: {year}")

        result.is_valid = bool(result.country or result.doc_type or result.company)

        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """Извлекает JSON из текста ответа."""
        # Попытка 1: Весь текст как JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Попытка 2: JSON в markdown блоке
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Попытка 3: Первый JSON объект
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Попытка 4: JSON с вложенными объектами
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _normalize_country(self, country: str) -> Optional[str]:
        """Нормализует название страны."""
        if not country:
            return None

        country_lower = country.lower().strip()

        # Прямое совпадение
        if country_lower in VALID_COUNTRIES:
            return country_lower

        # Через алиасы
        if country_lower in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[country_lower]

        # Частичное совпадение
        for valid_country in VALID_COUNTRIES:
            if valid_country in country_lower or country_lower in valid_country:
                return valid_country

        return None

    def get_server_stats(self) -> dict:
        """Возвращает статистику серверов."""
        stats = {
            'total_endpoints': len(self.endpoints),
            'available_endpoints': len(self._available_endpoints),
            'dead_endpoints': len(self._dead_endpoints),
            'endpoints': self.endpoints,
            'available': self._available_endpoints,
            'dead': list(self._dead_endpoints),
            'fail_counts': dict(self._endpoint_fail_count),
            'all_dead': self.all_dead,
        }
        return stats


# Singleton instance
_cluster: Optional[LLMCluster] = None


def get_llm_cluster(config: dict = None) -> LLMCluster:
    """Возвращает singleton экземпляр LLM кластера."""
    global _cluster
    if _cluster is None:
        _cluster = LLMCluster(config=config)
    return _cluster


def classify_with_llm(text: str, filename: str = "") -> LLMResult:
    """Удобная функция для классификации с LLM."""
    return get_llm_cluster().classify(text, filename)
