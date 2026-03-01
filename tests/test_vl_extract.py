"""
VL Document Extractor - извлечение данных из документов через Qwen2-VL
Поддержка нескольких серверов + сортировка по папкам country/document_type
"""
from PIL import Image, ImageOps
import io
import base64
import requests
import json
import re
import shutil
from pathlib import Path
from queue import Queue
import threading


# Список доступных серверов (добавь свои)
SERVERS = [
    "http://192.168.50.3:8080",
    "http://192.168.50.15:8080",
    "http://192.168.50.20:8080",
    "http://127.0.0.1:8080",
]

MODEL = "Qwen2-VL-2B-Instruct"

# Папки по умолчанию (измени под себя)
INPUT_DIR = Path("D:/python/InvoceLLM/tests/img")       # откуда брать
OUTPUT_DIR = Path("D:/python/InvoceLLM/tests/sorted")   # куда перемещать

# Thread-safe round-robin (для единичных запросов)
_server_lock = threading.Lock()
_server_index = 0


def prepare_image(image_path: str, max_side: int = 512) -> str:
    """Подготовка изображения: resize + autocontrast -> base64 JPEG"""
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.BICUBIC
        )

    img = ImageOps.autocontrast(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> dict | None:
    """Извлекает JSON из ответа модели (с или без ```json блока)"""
    # Пробуем найти JSON в ```json ... ``` блоке
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Пробуем найти просто JSON объект
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


PROMPT_FRONT = """Extract from this document:
- document_type (passport, id_card, driver_license, other)
- country
- full_name
- birth_date
- document_number

Reply ONLY with JSON:
{"document_type": "...", "side": "front", "country": "...", "full_name": "...", "birth_date": "...", "document_number": "..."}"""

PROMPT_BACK = """Extract from this document back side:
- document_type (passport, id_card, driver_license, other)
- country
- address
- mrz

Reply ONLY with JSON:
{"document_type": "...", "side": "back", "country": "...", "address": "...", "mrz": "..."}"""

PROMPT_AUTO = """Look at this image. Is it an identity document (passport, ID card, driver license)?

If YES - extract: document_type, country, full_name, birth_date, document_number
If NOT a document (receipt, random photo, etc.) - set document_type to "not_a_document"

document_type options: passport, id_card, driver_license, not_a_document

Reply ONLY with JSON:
{"document_type": "...", "country": "...", "full_name": "...", "birth_date": "...", "document_number": "..."}"""


# Подозрительные паттерны (фейковые данные)
SUSPICIOUS_PATTERNS = [
    "john doe", "jane doe", "test", "example", "sample",
    "1234567890", "0123456789", "123456789", "000000",
    "1990-01-01", "01.01.1990", "1980-01-01", "01.01.1980",
]


def validate_result(result: dict) -> dict:
    """Проверяет результат на подозрительные/фейковые данные"""
    if "error" in result:
        return result

    # Проверяем на подозрительные паттерны
    suspicious = False
    for field in ["full_name", "document_number", "birth_date"]:
        value = str(result.get(field, "")).lower()
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in value:
                suspicious = True
                break

    if suspicious:
        result["_warning"] = "suspicious_data"

    return result


def get_next_server() -> str:
    """Получает следующий сервер по round-robin (thread-safe)"""
    global _server_index
    with _server_lock:
        server = SERVERS[_server_index % len(SERVERS)]
        _server_index += 1
    return server


def check_servers() -> list[str]:
    """Проверяет доступность серверов, возвращает список рабочих"""
    available = []
    for server in SERVERS:
        try:
            r = requests.get(f"{server}/health", timeout=5)
            if r.ok:
                available.append(server)
                print(f"  ✓ {server}")
            else:
                print(f"  ✗ {server} (status {r.status_code})")
        except Exception as e:
            print(f"  ✗ {server} ({e})")
    return available


def vl_extract(image_path: str, prompt: str = None, side: str = "auto",
               retries: int = 2, server: str = None) -> dict:
    """
    Извлекает данные из документа через VL модель.

    Args:
        image_path: путь к изображению
        prompt: кастомный промпт (опционально)
        side: "auto" (определить автоматически), "front", "back"
        retries: количество повторных попыток при ошибке
        server: конкретный сервер (или авто-выбор)

    Returns:
        dict с полями в зависимости от стороны документа
        или {"error": "..."} при ошибке
    """
    if prompt is None:
        if side == "front":
            prompt = PROMPT_FRONT
        elif side == "back":
            prompt = PROMPT_BACK
        else:
            prompt = PROMPT_AUTO

    image_b64 = prepare_image(image_path)

    if server is None:
        server = get_next_server()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stop": ["```\n", "\n\n\n"]
    }

    api_url = f"{server}/v1/chat/completions"

    for attempt in range(retries + 1):
        try:
            r = requests.post(api_url, json=payload, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < retries:
                import time
                time.sleep(1)
                # Попробуем другой сервер
                server = get_next_server()
                api_url = f"{server}/v1/chat/completions"
                continue
            return {"error": str(e)}

    response = r.json()
    content = response["choices"][0]["message"]["content"]

    extracted = extract_json(content)
    if extracted is None:
        return {"error": "Failed to parse JSON", "raw": content}

    # Валидация на подозрительные данные
    extracted = validate_result(extracted)

    return extracted


def normalize_name(name: str, is_country: bool = False) -> str:
    """Нормализует имя для использования в пути (убирает спецсимволы)"""
    name = name.strip()

    # Нормализуем регистр для стран
    if is_country:
        name = name.title()  # "france" -> "France", "GERMANY" -> "Germany"

    # Заменяем пробелы
    name = name.replace(' ', '_')
    # Убираем недопустимые символы для файловой системы
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name or "unknown"


def organize_file(img_path: Path, result: dict, output_dir: Path, move: bool = False) -> Path | None:
    """
    Перемещает/копирует файл в папку country/document_type/

    Args:
        img_path: исходный путь к файлу
        result: результат распознавания
        output_dir: корневая папка для сортировки
        move: True = переместить, False = копировать

    Returns:
        новый путь к файлу или None при ошибке
    """
    if "error" in result:
        # Файлы с ошибками в отдельную папку
        target_dir = output_dir / "_errors"
    elif result.get("document_type") == "not_a_document" or result.get("_warning"):
        # Не документы и подозрительные - в _unknown
        target_dir = output_dir / "_unknown"
    else:
        country = normalize_name(result.get("country", "unknown"), is_country=True)
        doc_type = normalize_name(result.get("document_type", "unknown"))
        target_dir = output_dir / country / doc_type

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / img_path.name

    # Если файл уже существует, добавляем суффикс
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    try:
        if move:
            shutil.move(str(img_path), str(target_path))
        else:
            shutil.copy2(str(img_path), str(target_path))
        return target_path
    except Exception as e:
        print(f"Error moving {img_path.name}: {e}")
        return None


def process_single(img_path: Path, side: str = "auto", server: str = None) -> dict:
    """Обрабатывает один файл (для параллельной обработки)"""
    result = vl_extract(str(img_path), side=side, server=server)
    result["_file"] = img_path.name
    result["_path"] = str(img_path)
    result["_server"] = server or "auto"
    return result


def _server_worker(server: str, file_queue: Queue, results: list,
                   results_lock: threading.Lock, side: str, print_lock: threading.Lock,
                   output_dir: Path = None, move: bool = False):
    """
    Воркер привязанный к конкретному серверу.
    Берёт файлы из общей очереди, обрабатывает на своём сервере.
    Сразу перемещает файл после обработки (если output_dir задан).
    """
    server_short = server.split("//")[1].split(":")[0].split(".")[-1]  # последний октет IP

    while True:
        try:
            img_path = file_queue.get_nowait()
        except Exception:
            break  # очередь пуста — выходим

        try:
            result = process_single(img_path, side=side, server=server)

            # Сразу перемещаем файл после обработки
            if output_dir:
                new_path = organize_file(img_path, result, output_dir, move=move)
                if new_path:
                    result["_sorted_to"] = str(new_path)

            with results_lock:
                results.append(result)

            with print_lock:
                if "error" in result:
                    print(f"  ✗ [{server_short}] {img_path.name}: {result['error']}")
                else:
                    name = result.get('full_name') or result.get('address') or 'N/A'
                    doc_type = result.get('document_type', '?')
                    country = result.get('country', '?')
                    dest = f" -> {country}/{doc_type}/" if output_dir else ""
                    print(f"  ✓ [{server_short}] {img_path.name}: {doc_type}/{country} - {name}{dest}")
        except Exception as e:
            with print_lock:
                print(f"  ✗ [{server_short}] {img_path.name}: Exception {e}")
            with results_lock:
                results.append({"error": str(e), "_file": img_path.name, "_path": str(img_path)})
        finally:
            file_queue.task_done()


def process_directory(img_dir: Path, side: str = "auto", workers: int = None,
                      servers: list[str] = None,
                      output_dir: Path = None, move: bool = False) -> list[dict]:
    """
    Обрабатывает все изображения в директории.
    Каждый сервер = отдельный воркер с общей очередью файлов.
    Если output_dir задан — сразу перемещает каждый файл после обработки.

    Args:
        img_dir: папка с изображениями
        side: сторона документа
        workers: не используется (оставлен для совместимости)
        servers: список серверов (по умолчанию все доступные)
        output_dir: папка для сортировки (None = не перемещать)
        move: True = переместить, False = копировать
    """
    results = []
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))

    if not images:
        print(f"Нет изображений в {img_dir}")
        return results

    if servers is None:
        # Проверяем доступность серверов
        print("Проверка серверов...")
        servers = check_servers()
        if not servers:
            print("Нет доступных серверов!")
            return results
        print()

    action = "Обработка + перемещение" if output_dir and move else \
             "Обработка + копирование" if output_dir else "Обработка"
    print(f"{action}: {len(images)} файлов на {len(servers)} серверах...\n")

    if len(servers) > 1 and len(images) > 1:
        # Параллельная обработка: каждый сервер = свой поток
        file_queue = Queue()
        for img in images:
            file_queue.put(img)

        results_lock = threading.Lock()
        print_lock = threading.Lock()
        threads = []

        for server in servers:
            t = threading.Thread(
                target=_server_worker,
                args=(server, file_queue, results, results_lock, side, print_lock,
                      output_dir, move),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Ждём пока все файлы обработаны
        file_queue.join()
    else:
        # Один сервер — последовательная обработка
        server = servers[0] if servers else None
        for img_path in images:
            print(f"  Processing: {img_path.name}...", end=" ", flush=True)
            result = process_single(img_path, side, server=server)

            # Сразу перемещаем
            if output_dir:
                new_path = organize_file(img_path, result, output_dir, move=move)
                if new_path:
                    result["_sorted_to"] = str(new_path)

            results.append(result)

            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                name = result.get('full_name') or result.get('address') or 'N/A'
                doc_type = result.get('document_type', '?')
                country = result.get('country', '?')
                dest = f" -> {country}/{doc_type}/" if output_dir else ""
                print(f"OK - {name}{dest}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VL Document Extractor")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--side", choices=["auto", "front", "back"], default="auto",
                        help="Document side: auto, front, back")
    parser.add_argument("--input", "-i", help=f"Input directory (default: {INPUT_DIR})")
    parser.add_argument("--output", "-o", help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of move")
    parser.add_argument("--no-organize", action="store_true", help="Only process, don't move/copy files")
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser.add_argument("--check-servers", action="store_true", help="Check server availability")
    args = parser.parse_args()

    if args.check_servers:
        print("Проверка серверов:")
        available = check_servers()
        print(f"\nДоступно: {len(available)}/{len(SERVERS)}")
        exit(0)

    if args.image:
        # Один файл
        result = vl_extract(args.image, side=args.side)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Используем пути по умолчанию или из аргументов
        img_dir = Path(args.input) if args.input else INPUT_DIR
        output_dir = Path(args.output) if args.output else OUTPUT_DIR

        if args.no_organize:
            # Только обработка без перемещения
            results = process_directory(img_dir, side=args.side)
        else:
            # Обработка + сразу перемещение каждого файла
            results = process_directory(
                img_dir, side=args.side,
                output_dir=output_dir,
                move=not args.copy,  # по умолчанию перемещаем
            )

        print("\n" + "="*50)
        print(f"Обработано: {len(results)} файлов")

        # Статистика по странам и типам
        stats = {}
        for r in results:
            if "error" not in r:
                key = f"{r.get('country', '?')}/{r.get('document_type', '?')}"
                stats[key] = stats.get(key, 0) + 1

        if stats:
            print("\nСтатистика:")
            for key, count in sorted(stats.items()):
                print(f"  {key}: {count}")

        #print("\n" + json.dumps(results, indent=2, ensure_ascii=False))

# Использование:

# По умолчанию: img/ -> sorted/ (перемещение)
# python tests/test_vl_extract.py

# Копировать вместо перемещения
# python tests/test_vl_extract.py --copy

# Свои папки
# python tests/test_vl_extract.py -i /path/to/input -o /path/to/output

# Только обработка без перемещения
# python tests/test_vl_extract.py --no-organize

# С указанием воркеров
# python tests/test_vl_extract.py -w 4

# Проверить доступность серверов
# python tests/test_vl_extract.py --check-servers
