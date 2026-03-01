from PIL import Image, ImageOps
import io
import base64

def prepare_image(
    image_path: str,
    max_side: int = 1024
) -> str:
    """
    Подготавливает изображение под VL:
    - resize (длинная сторона <= max_side)
    - auto-contrast
    Возвращает base64 (JPEG)
    """
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


import requests
import json

API_URL = "http://192.168.50.3:8080/v1/chat/completions"
MODEL = "Qwen2-VL-2B-Instruct"

def vl_request(image_b64: str, prompt: str):
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
        "max_tokens": 512,
        "stop": ["```\n", "\n\n\n"]
    }

    r = requests.post(API_URL, json=payload, timeout=120)
    if not r.ok:
        print(f"\nОшибка {r.status_code}: {r.text}\n", flush=True)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    IMG_DIR = Path(__file__).parent / "img"

    # Можно передать путь как аргумент или взять первый файл из img/
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.jpeg"))
        if not images:
            print("Нет изображений в", IMG_DIR)
            sys.exit(1)
        img_path = images[0]
        print(f"Используем: {img_path.name}")

    img_b64 = prepare_image(str(img_path), max_side=512)  # меньше для 6GB VRAM

    prompt = """Extract from this document:
- document_type (passport, id_card, driver_license, other)
- country
- full_name
- birth_date
- document_number

Reply ONLY with JSON:
{"document_type": "...", "country": "...", "full_name": "...", "birth_date": "...", "document_number": "..."}"""

    result = vl_request(img_b64, prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))