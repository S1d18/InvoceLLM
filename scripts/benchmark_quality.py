#!/usr/bin/env python
"""
Benchmark качества ответов LLM серверов.

Проверяет консистентность ответов между серверами на одинаковых данных.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import requests

# Добавляем корень проекта
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from extractors import extract_pdf_text


@dataclass
class LLMResponse:
    """Ответ от LLM."""
    endpoint: str
    pdf_file: str
    raw_response: str = ""
    country: Optional[str] = None
    doc_type: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    elapsed: float = 0.0
    tokens: int = 0
    error: Optional[str] = None


def extract_json(text: str) -> Optional[dict]:
    """Извлекает JSON из ответа."""
    # Попытка 1: Весь текст как JSON
    try:
        return json.loads(text.strip())
    except:
        pass

    # Попытка 2: JSON в markdown
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Попытка 3: Первый JSON объект
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


class QualityBenchmark:
    """Бенчмарк качества ответов."""

    SYSTEM_PROMPT = """You are a document classification assistant. Extract structured information from invoice/bill documents.

OUTPUT FORMAT (valid JSON only):
{"country": "Country Name", "doc_type": "electricity|telecom|bank|water|gas|tax|other", "company": "Company Name", "year": 2024}"""

    def __init__(
        self,
        hosts: list[str],
        ports: list[int],
        timeout: float = 60.0,
        max_tokens: int = 200,
    ):
        self.hosts = hosts
        self.ports = ports
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Создаём список всех endpoints
        self.endpoints = []
        for host in hosts:
            for port in ports:
                self.endpoints.append(f"http://{host}:{port}")

        # Результаты: {pdf_file: {endpoint: LLMResponse}}
        self.results: dict[str, dict[str, LLMResponse]] = defaultdict(dict)

    def call_llm(self, endpoint: str, text: str, pdf_name: str) -> LLMResponse:
        """Вызывает LLM и возвращает структурированный ответ."""
        response = LLMResponse(endpoint=endpoint, pdf_file=pdf_name)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract information from:\n{text[:2000]}"}
        ]

        payload = {
            "model": "local",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            start = time.perf_counter()

            r = requests.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()

            response.elapsed = time.perf_counter() - start

            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            response.raw_response = content
            response.tokens = data.get("usage", {}).get("completion_tokens", 0)

            # Парсим JSON
            json_data = extract_json(content)
            if json_data:
                response.country = json_data.get("country")
                response.doc_type = json_data.get("doc_type")
                response.company = json_data.get("company")
                year = json_data.get("year")
                if year:
                    try:
                        response.year = int(year)
                    except:
                        pass

        except Exception as e:
            response.error = str(e)

        return response

    def run_benchmark(self, pdf_dir: Path, num_files: int = 5):
        """Запускает бенчмарк качества."""
        # Собираем PDF
        pdf_files = list(pdf_dir.glob("*.pdf"))[:num_files]

        print("=" * 70)
        print("BENCHMARK КАЧЕСТВА ОТВЕТОВ")
        print("=" * 70)
        print(f"Endpoints: {len(self.endpoints)}")
        print(f"PDF файлов: {len(pdf_files)}")
        print(f"Всего запросов: {len(self.endpoints) * len(pdf_files)}")

        # Извлекаем текст
        print("\nИзвлечение текста из PDF...")
        test_data = []
        for pdf_file in pdf_files:
            try:
                text = extract_pdf_text(pdf_file)
                test_data.append((pdf_file, text))
                print(f"  [OK] {pdf_file.name} ({len(text)} chars)")
            except Exception as e:
                print(f"  [ERR] {pdf_file.name}: {e}")

        if not test_data:
            print("Нет данных для тестирования!")
            return

        # Тестируем каждый endpoint на каждом файле
        print("\n" + "=" * 70)
        print("ТЕСТИРОВАНИЕ")
        print("=" * 70)

        total = len(self.endpoints) * len(test_data)
        current = 0

        for pdf_file, text in test_data:
            pdf_name = pdf_file.name
            print(f"\n--- {pdf_name} ---")

            for endpoint in self.endpoints:
                current += 1
                print(f"  [{current}/{total}] {endpoint}...", end=" ", flush=True)

                response = self.call_llm(endpoint, text, pdf_name)
                self.results[pdf_name][endpoint] = response

                if response.error:
                    print(f"ERROR: {response.error}")
                else:
                    print(f"{response.elapsed:.2f}s - {response.country}/{response.doc_type}/{response.company}/{response.year}")

        # Анализ результатов
        self.analyze_results()

    def analyze_results(self):
        """Анализирует и выводит результаты."""
        print("\n" + "=" * 70)
        print("АНАЛИЗ КОНСИСТЕНТНОСТИ ОТВЕТОВ")
        print("=" * 70)

        total_files = len(self.results)
        consistent_files = 0
        inconsistencies = []

        for pdf_name, responses in self.results.items():
            print(f"\n{'='*70}")
            print(f"Файл: {pdf_name}")
            print(f"{'='*70}")

            # Собираем уникальные значения по каждому полю
            countries = defaultdict(list)
            doc_types = defaultdict(list)
            companies = defaultdict(list)
            years = defaultdict(list)

            for endpoint, resp in responses.items():
                if resp.error:
                    continue

                # Нормализуем для сравнения
                country = (resp.country or "").lower().strip()
                doc_type = (resp.doc_type or "").lower().strip()
                company = (resp.company or "").lower().strip()
                year = resp.year

                countries[country].append(endpoint)
                doc_types[doc_type].append(endpoint)
                companies[company].append(endpoint)
                years[year].append(endpoint)

            # Проверяем консистентность
            is_consistent = True

            # Country
            print(f"\n  COUNTRY:")
            if len(countries) == 1:
                val = list(countries.keys())[0]
                print(f"    [OK] Все согласны: '{val or 'null'}'")
            else:
                is_consistent = False
                print(f"    [DIFF] Расхождение!")
                for val, eps in sorted(countries.items(), key=lambda x: -len(x[1])):
                    print(f"      '{val or 'null'}': {len(eps)} серверов")
                    for ep in eps:
                        host = ep.split("//")[1].split(":")[0]
                        port = ep.split(":")[-1]
                        print(f"        - {host}:{port}")

            # Doc Type
            print(f"\n  DOC_TYPE:")
            if len(doc_types) == 1:
                val = list(doc_types.keys())[0]
                print(f"    [OK] Все согласны: '{val or 'null'}'")
            else:
                is_consistent = False
                print(f"    [DIFF] Расхождение!")
                for val, eps in sorted(doc_types.items(), key=lambda x: -len(x[1])):
                    print(f"      '{val or 'null'}': {len(eps)} серверов")
                    for ep in eps:
                        host = ep.split("//")[1].split(":")[0]
                        port = ep.split(":")[-1]
                        print(f"        - {host}:{port}")

            # Company
            print(f"\n  COMPANY:")
            if len(companies) == 1:
                val = list(companies.keys())[0]
                print(f"    [OK] Все согласны: '{val or 'null'}'")
            else:
                # Проверяем схожесть (могут быть небольшие различия)
                unique_companies = list(companies.keys())
                if len(unique_companies) <= 2:
                    # Проверяем похожесть
                    similar = False
                    if len(unique_companies) == 2:
                        c1, c2 = unique_companies
                        if c1 in c2 or c2 in c1:
                            similar = True
                    if similar:
                        print(f"    [~OK] Похожие значения:")
                    else:
                        is_consistent = False
                        print(f"    [DIFF] Расхождение!")
                else:
                    is_consistent = False
                    print(f"    [DIFF] Расхождение!")

                for val, eps in sorted(companies.items(), key=lambda x: -len(x[1])):
                    print(f"      '{val or 'null'}': {len(eps)} серверов")

            # Year
            print(f"\n  YEAR:")
            if len(years) == 1:
                val = list(years.keys())[0]
                print(f"    [OK] Все согласны: '{val or 'null'}'")
            else:
                is_consistent = False
                print(f"    [DIFF] Расхождение!")
                for val, eps in sorted(years.items(), key=lambda x: -len(x[1])):
                    print(f"      '{val}': {len(eps)} серверов")

            if is_consistent:
                consistent_files += 1
            else:
                inconsistencies.append(pdf_name)

        # Итоговая статистика
        print("\n" + "=" * 70)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 70)

        print(f"\nФайлов с полной консистентностью: {consistent_files}/{total_files}")
        print(f"Файлов с расхождениями: {len(inconsistencies)}/{total_files}")

        if inconsistencies:
            print(f"\nФайлы с расхождениями:")
            for f in inconsistencies:
                print(f"  - {f}")

        # Статистика по скорости
        print("\n" + "=" * 70)
        print("СТАТИСТИКА ПО СКОРОСТИ")
        print("=" * 70)

        endpoint_stats = defaultdict(lambda: {"times": [], "tokens": []})

        for pdf_name, responses in self.results.items():
            for endpoint, resp in responses.items():
                if not resp.error:
                    endpoint_stats[endpoint]["times"].append(resp.elapsed)
                    endpoint_stats[endpoint]["tokens"].append(resp.tokens)

        # Группируем по хостам
        host_stats = defaultdict(lambda: {"times": [], "tokens": []})
        for endpoint, stats in endpoint_stats.items():
            host = endpoint.split("//")[1].split(":")[0]
            host_stats[host]["times"].extend(stats["times"])
            host_stats[host]["tokens"].extend(stats["tokens"])

        print(f"\nПо хостам:")
        for host in sorted(host_stats.keys()):
            stats = host_stats[host]
            avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
            total_tokens = sum(stats["tokens"])
            total_time = sum(stats["times"])
            tps = total_tokens / total_time if total_time > 0 else 0
            print(f"  {host}: avg={avg_time:.2f}s, {tps:.1f} tok/s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark качества LLM')
    parser.add_argument(
        '--hosts', nargs='+',
        default=['192.168.50.15', '192.168.50.20', '192.168.50.3'],
    )
    parser.add_argument(
        '--ports', nargs='+', type=int,
        default=[8080, 8081, 8082],
    )
    parser.add_argument(
        '--pdf-dir', type=Path,
        default=ROOT / 'pdf_test',
    )
    parser.add_argument(
        '--files', type=int, default=5,
        help='Количество PDF файлов для теста'
    )
    parser.add_argument(
        '--timeout', type=float, default=60.0,
    )

    args = parser.parse_args()

    benchmark = QualityBenchmark(
        hosts=args.hosts,
        ports=args.ports,
        timeout=args.timeout,
    )

    benchmark.run_benchmark(
        pdf_dir=args.pdf_dir,
        num_files=args.files,
    )


if __name__ == '__main__':
    main()