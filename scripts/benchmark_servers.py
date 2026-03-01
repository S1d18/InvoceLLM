#!/usr/bin/env python
"""
Benchmark скрипт для тестирования LLM серверов.

Тестирует скорость классификации PDF на каждом сервере отдельно.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import requests

# Добавляем корень проекта
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from extractors import extract_pdf_text


@dataclass
class EndpointStats:
    """Статистика по endpoint."""
    endpoint: str
    host: str
    port: int
    is_available: bool = False
    model_name: str = ""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    tokens_generated: int = 0
    errors: list = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_time / self.successful_requests

    @property
    def tokens_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.tokens_generated / self.total_time

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


@dataclass
class BenchmarkResult:
    """Результат бенчмарка."""
    total_pdfs: int
    total_tests: int
    endpoints: list
    by_host: dict
    duration: float


class ServerBenchmark:
    """Бенчмарк LLM серверов."""

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
                self.endpoints.append({
                    'host': host,
                    'port': port,
                    'url': f"http://{host}:{port}",
                })

        # Статистика
        self.stats: dict[str, EndpointStats] = {}
        for ep in self.endpoints:
            key = ep['url']
            self.stats[key] = EndpointStats(
                endpoint=key,
                host=ep['host'],
                port=ep['port'],
            )

    def check_availability(self) -> dict[str, bool]:
        """Проверяет доступность всех endpoints."""
        print("\n" + "="*60)
        print("Проверка доступности серверов")
        print("="*60)

        results = {}

        for ep in self.endpoints:
            url = ep['url']
            stats = self.stats[url]

            try:
                # Проверяем health endpoint
                r = requests.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    stats.is_available = True
                    results[url] = True

                    # Пробуем получить информацию о модели
                    try:
                        r2 = requests.get(f"{url}/v1/models", timeout=5)
                        if r2.status_code == 200:
                            models = r2.json().get('data', [])
                            if models:
                                stats.model_name = models[0].get('id', 'unknown')
                    except:
                        pass

                    print(f"  [OK]   {url} - {stats.model_name or 'model info unavailable'}")
                else:
                    stats.is_available = False
                    results[url] = False
                    print(f"  [FAIL] {url} - HTTP {r.status_code}")

            except requests.exceptions.ConnectionError:
                stats.is_available = False
                results[url] = False
                print(f"  [DOWN] {url} - Connection refused")
            except requests.exceptions.Timeout:
                stats.is_available = False
                results[url] = False
                print(f"  [DOWN] {url} - Timeout")
            except Exception as e:
                stats.is_available = False
                results[url] = False
                print(f"  [ERR]  {url} - {e}")

        available = sum(1 for v in results.values() if v)
        print(f"\nДоступно: {available}/{len(self.endpoints)} endpoints")

        return results

    def _call_llm(self, endpoint: str, text: str) -> tuple[str, float, int]:
        """
        Вызывает LLM на указанном endpoint.

        Returns:
            (response_text, elapsed_time, tokens_generated)
        """
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

        start = time.perf_counter()

        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        elapsed = time.perf_counter() - start

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))

        return content, elapsed, tokens

    def test_endpoint(
        self,
        endpoint: str,
        test_data: list[tuple[Path, str]],
    ) -> EndpointStats:
        """Тестирует один endpoint на подготовленных данных."""
        stats = self.stats[endpoint]

        if not stats.is_available:
            return stats

        for pdf_file, text in test_data:
            stats.total_requests += 1

            try:
                # Вызываем LLM (текст уже извлечён)
                response, elapsed, tokens = self._call_llm(endpoint, text)

                # Обновляем статистику
                stats.successful_requests += 1
                stats.total_time += elapsed
                stats.tokens_generated += tokens
                stats.min_time = min(stats.min_time, elapsed)
                stats.max_time = max(stats.max_time, elapsed)

            except Exception as e:
                stats.failed_requests += 1
                stats.errors.append(str(e))

        return stats

    def run_benchmark(
        self,
        pdf_dir: Path,
        tests_per_endpoint: int = 5,
        parallel: bool = False,
    ) -> BenchmarkResult:
        """
        Запускает полный бенчмарк.

        Args:
            pdf_dir: Директория с PDF файлами
            tests_per_endpoint: Количество тестов на каждый endpoint
            parallel: Запускать тесты параллельно
        """
        start_time = time.time()

        # Собираем PDF файлы
        pdf_files = list(pdf_dir.glob("*.pdf"))[:50]  # Максимум 50 файлов

        if not pdf_files:
            print(f"Не найдено PDF файлов в {pdf_dir}")
            return None

        # Выбираем фиксированные файлы для тестов
        test_files = pdf_files[:tests_per_endpoint]

        print(f"\n" + "="*60)
        print(f"Запуск бенчмарка")
        print(f"="*60)
        print(f"PDF файлов в папке: {len(pdf_files)}")
        print(f"Тестовых файлов: {len(test_files)}")
        print(f"Всего endpoints: {len(self.endpoints)}")

        # Предварительно извлекаем текст из тестовых PDF
        print(f"\nИзвлечение текста из PDF...")
        test_data = []
        for pdf_file in test_files:
            try:
                text = extract_pdf_text(pdf_file)
                test_data.append((pdf_file, text))
                print(f"  [OK] {pdf_file.name} ({len(text)} chars)")
            except Exception as e:
                print(f"  [ERR] {pdf_file.name}: {e}")

        if not test_data:
            print("Не удалось извлечь текст ни из одного PDF!")
            return None

        print(f"\nПодготовлено {len(test_data)} файлов для тестирования")
        print(f"Каждый endpoint получит ОДИНАКОВЫЕ {len(test_data)} файлов")

        # Проверяем доступность
        self.check_availability()

        available_endpoints = [ep['url'] for ep in self.endpoints if self.stats[ep['url']].is_available]

        if not available_endpoints:
            print("\nНет доступных серверов для тестирования!")
            return None

        print(f"\n" + "="*60)
        print(f"Тестирование скорости (одинаковые файлы)")
        print(f"="*60)

        if parallel:
            # Параллельное тестирование
            with ThreadPoolExecutor(max_workers=len(available_endpoints)) as executor:
                futures = {
                    executor.submit(
                        self.test_endpoint, ep, test_data
                    ): ep for ep in available_endpoints
                }

                for future in as_completed(futures):
                    ep = futures[future]
                    try:
                        stats = future.result()
                        self._print_endpoint_stats(stats)
                    except Exception as e:
                        print(f"  [ERR] {ep}: {e}")
        else:
            # Последовательное тестирование
            for endpoint in available_endpoints:
                print(f"\nТестирование {endpoint}...")
                stats = self.test_endpoint(endpoint, test_data)
                self._print_endpoint_stats(stats)

        duration = time.time() - start_time

        # Группировка по хостам
        by_host = {}
        for ep in self.endpoints:
            host = ep['host']
            if host not in by_host:
                by_host[host] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'total_time': 0.0,
                    'tokens_generated': 0,
                    'ports': [],
                }

            stats = self.stats[ep['url']]
            by_host[host]['total_requests'] += stats.total_requests
            by_host[host]['successful_requests'] += stats.successful_requests
            by_host[host]['total_time'] += stats.total_time
            by_host[host]['tokens_generated'] += stats.tokens_generated
            by_host[host]['ports'].append({
                'port': ep['port'],
                'available': stats.is_available,
                'avg_time': stats.avg_time,
                'tokens_per_second': stats.tokens_per_second,
            })

        return BenchmarkResult(
            total_pdfs=len(pdf_files),
            total_tests=tests_per_endpoint * len(available_endpoints),
            endpoints=[self.stats[ep['url']] for ep in self.endpoints],
            by_host=by_host,
            duration=duration,
        )

    def _print_endpoint_stats(self, stats: EndpointStats):
        """Выводит статистику endpoint."""
        if not stats.is_available:
            print(f"  {stats.endpoint}: недоступен")
            return

        print(f"  {stats.endpoint}:")
        print(f"    Запросов: {stats.successful_requests}/{stats.total_requests}")
        print(f"    Среднее время: {stats.avg_time:.2f}s")
        print(f"    Мин/Макс: {stats.min_time:.2f}s / {stats.max_time:.2f}s")
        print(f"    Токенов/сек: {stats.tokens_per_second:.1f}")

        if stats.errors:
            print(f"    Ошибки: {len(stats.errors)}")

    def print_summary(self, result: BenchmarkResult):
        """Выводит итоговую статистику."""
        print(f"\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)

        print(f"\nОбщее время бенчмарка: {result.duration:.1f}s")
        print(f"PDF файлов использовано: {result.total_pdfs}")
        print(f"Всего тестов: {result.total_tests}")

        # Статистика по хостам
        print(f"\n{'='*60}")
        print("СТАТИСТИКА ПО МАШИНАМ")
        print(f"{'='*60}")

        for host, data in sorted(result.by_host.items()):
            print(f"\n{host}:")

            if data['total_requests'] == 0:
                print("  Нет данных (все порты недоступны)")
                continue

            avg_time = data['total_time'] / data['successful_requests'] if data['successful_requests'] > 0 else 0
            tps = data['tokens_generated'] / data['total_time'] if data['total_time'] > 0 else 0

            print(f"  Успешных запросов: {data['successful_requests']}/{data['total_requests']}")
            print(f"  Общее время: {data['total_time']:.2f}s")
            print(f"  Среднее время: {avg_time:.2f}s")
            print(f"  Токенов/сек: {tps:.1f}")

            print(f"  Порты:")
            for port_info in data['ports']:
                status = "OK" if port_info['available'] else "DOWN"
                if port_info['available'] and port_info['avg_time'] > 0:
                    print(f"    :{port_info['port']} [{status}] - {port_info['avg_time']:.2f}s, {port_info['tokens_per_second']:.1f} tok/s")
                else:
                    print(f"    :{port_info['port']} [{status}]")

        # Рейтинг по скорости
        print(f"\n{'='*60}")
        print("РЕЙТИНГ ПО СКОРОСТИ (tokens/sec)")
        print(f"{'='*60}")

        sorted_hosts = sorted(
            [(h, d['tokens_generated'] / d['total_time'] if d['total_time'] > 0 else 0)
             for h, d in result.by_host.items()],
            key=lambda x: -x[1]
        )

        for i, (host, tps) in enumerate(sorted_hosts, 1):
            if tps > 0:
                print(f"  {i}. {host}: {tps:.1f} tokens/sec")
            else:
                print(f"  {i}. {host}: N/A")


def main():
    """Главная функция."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark LLM серверов')
    parser.add_argument(
        '--hosts',
        nargs='+',
        default=['192.168.50.15', '192.168.50.20', '192.168.50.3'],
        help='IP адреса хостов'
    )
    parser.add_argument(
        '--ports',
        nargs='+',
        type=int,
        default=[8080, 8081, 8082],
        help='Порты'
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default=ROOT / 'pdf_test',
        help='Директория с PDF файлами'
    )
    parser.add_argument(
        '--tests',
        type=int,
        default=5,
        help='Количество тестов на endpoint'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Таймаут запроса в секундах'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Параллельное тестирование'
    )

    args = parser.parse_args()

    print("="*60)
    print("LLM Server Benchmark")
    print("="*60)
    print(f"Хосты: {args.hosts}")
    print(f"Порты: {args.ports}")
    print(f"PDF директория: {args.pdf_dir}")

    benchmark = ServerBenchmark(
        hosts=args.hosts,
        ports=args.ports,
        timeout=args.timeout,
    )

    result = benchmark.run_benchmark(
        pdf_dir=args.pdf_dir,
        tests_per_endpoint=args.tests,
        parallel=args.parallel,
    )

    if result:
        benchmark.print_summary(result)


if __name__ == '__main__':
    main()