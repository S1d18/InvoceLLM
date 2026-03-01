#!/usr/bin/env python3
"""
InvoiceLLM — тест подключения к Colab LLM-бэкенду.

Использование:
    python client_test.py --url https://YOUR-URL.trycloudflare.com
    python client_test.py --url http://localhost:8080
"""

import argparse
import json
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("Установите openai: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Установите requests: pip install requests")
    sys.exit(1)


SAMPLE_INVOICE = """RECHNUNG
Wiener Netze GmbH
Erdbergstraße 236, 1110 Wien
Rechnungsdatum: 15.03.2024
Stromrechnung für den Zeitraum 01.01.2024 - 31.03.2024
Gesamtbetrag: EUR 245,67 inkl. MwSt
UID: ATU12345678
IBAN: AT611904300234573201"""

SYSTEM_PROMPT = """Extract from the invoice text: country, doc_type (electricity/telecom/bank/water/gas/tax/other), company, year.
Respond ONLY with JSON: {"country": "...", "doc_type": "...", "company": "...", "year": ...}"""

EXPECTED = {
    "country": "Austria",
    "doc_type": "electricity",
    "company": "Wiener Netze",
    "year": 2024,
}


def check_health(base_url: str) -> bool:
    """Check /health endpoint."""
    url = f"{base_url}/health"
    print(f"\n[1/3] Health check: {url}")

    try:
        start = time.time()
        r = requests.get(url, timeout=15)
        elapsed = time.time() - start

        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Status: {data.get('status', 'ok')}")
            print(f"  ✓ Model: {data.get('model', 'n/a')}")
            print(f"  ✓ Backend: {data.get('backend', 'n/a')}")
            print(f"  ✓ Latency: {elapsed*1000:.0f}ms")
            return True
        else:
            print(f"  ✗ HTTP {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Не удалось подключиться к {base_url}")
        print(f"    Проверьте что туннель активен в Colab")
        return False
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False


def test_completion(base_url: str) -> tuple:
    """Test /v1/chat/completions endpoint."""
    print(f"\n[2/3] Тест классификации инвойса...")

    client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")

    try:
        start = time.time()
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SAMPLE_INVOICE},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content
        print(f"  ✓ Ответ получен за {elapsed*1000:.0f}ms")
        print(f"  ✓ Raw: {content[:300]}")

        # Parse JSON
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        print(f"  ✓ JSON валиден")
        return True, result, elapsed

    except json.JSONDecodeError:
        print(f"  ⚠ Ответ не является валидным JSON")
        print(f"    {content[:200]}")
        return False, None, 0
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False, None, 0


def validate_result(result: dict) -> None:
    """Validate extracted fields against expected values."""
    print(f"\n[3/3] Валидация результата...")

    if result is None:
        print("  ✗ Нет результата для валидации")
        return

    checks = 0
    passed = 0

    for field, expected in EXPECTED.items():
        checks += 1
        actual = result.get(field)

        if field == "company":
            # Fuzzy match for company name
            ok = actual and expected.lower() in actual.lower()
        elif field == "year":
            ok = actual == expected or str(actual) == str(expected)
        else:
            ok = actual and actual.lower() == str(expected).lower()

        if ok:
            passed += 1
            print(f"  ✓ {field}: {actual}")
        else:
            print(f"  ✗ {field}: {actual} (ожидалось: {expected})")

    print(f"\n  Результат: {passed}/{checks} полей корректны")


def main():
    parser = argparse.ArgumentParser(description="InvoiceLLM Colab endpoint tester")
    parser.add_argument("--url", required=True, help="Base URL (e.g. https://xxx.trycloudflare.com)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"InvoiceLLM — тест Colab-бэкенда")
    print(f"URL: {base_url}")
    print("=" * 60)

    # Step 1: Health check
    if not check_health(base_url):
        print("\n✗ Health check не пройден. Завершение.")
        sys.exit(1)

    # Step 2: Test completion
    ok, result, latency = test_completion(base_url)

    # Step 3: Validate
    if ok:
        validate_result(result)

    # Summary
    print(f"\n{'=' * 60}")
    if ok:
        print(f"✓ Бэкенд работает! Latency: {latency*1000:.0f}ms")
        print(f"\nДля InvoiceLLM config.yaml:")
        host = base_url.replace("https://", "").replace("http://", "")
        if ":" in host.split("/")[0]:
            h, p = host.split("/")[0].split(":")
            print(f'  host: "{h}"')
            print(f'  port: {p}')
            print(f'  ssl: false')
        else:
            print(f'  host: "{host}"')
            print(f'  port: 443')
            print(f'  ssl: true')
    else:
        print(f"✗ Тест не пройден")
        sys.exit(1)


if __name__ == "__main__":
    main()
