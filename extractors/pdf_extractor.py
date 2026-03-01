"""
PDF Text Extractor для Invoice LLM.

Извлечение текста из PDF с обработкой проблемных файлов.
Адаптировано из New_sort/pdf_text.py.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Конфигурация
MAX_PAGES_FOR_TEXT = int(os.environ.get('PDF_MAX_PAGES', '5'))
PDF_TIMEOUT_SEC = int(os.environ.get('PDF_TIMEOUT', '30'))
PDF_NO_TIMEOUT = os.environ.get('PDF_NO_TIMEOUT', '').lower() in ('1', 'true', 'yes')

BADLIST_FILE = Path(__file__).parent.parent / "data" / "pdf_badlist.json"
SMALL_SIZE_BYTES = 500_000  # 500 KB

# Import PDF libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.warning("PyMuPDF (fitz) not installed, using pdfminer fallback")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    try:
        from pdfminer.pdfdocument import PDFEncryptionError
    except ImportError:
        class PDFEncryptionError(Exception):
            pass
except ImportError:
    pdfminer_extract = None
    PDFPasswordIncorrect = Exception
    PDFEncryptionError = Exception
    logger.warning("pdfminer.six not installed")


class PDFOpenError(Exception):
    """Ошибка открытия PDF."""
    pass


class PDFBlockedError(Exception):
    """PDF защищён паролем."""
    pass


# Кэш badlist
_badlist_cache: Optional[set[str]] = None
_badlist_mtime: float = 0.0


def _sig_for(fp: Path) -> str:
    """Создаёт сигнатуру файла."""
    try:
        st = fp.stat()
        return f"{fp.resolve()};{int(st.st_size)};{int(st.st_mtime)}"
    except Exception:
        return str(fp.resolve())


def _load_badlist() -> set[str]:
    """Загружает список проблемных PDF."""
    global _badlist_cache, _badlist_mtime

    try:
        if _badlist_cache is not None:
            if BADLIST_FILE.exists():
                current_mtime = BADLIST_FILE.stat().st_mtime
                if current_mtime == _badlist_mtime:
                    return _badlist_cache
            else:
                return _badlist_cache

        if BADLIST_FILE.exists():
            with BADLIST_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _badlist_cache = set(data if isinstance(data, list) else [])
            _badlist_mtime = BADLIST_FILE.stat().st_mtime
        else:
            _badlist_cache = set()

        return _badlist_cache
    except Exception:
        if _badlist_cache is None:
            _badlist_cache = set()
        return _badlist_cache


def _save_badlist(sigs: set[str]) -> None:
    """Сохраняет список проблемных PDF."""
    global _badlist_cache, _badlist_mtime

    try:
        BADLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BADLIST_FILE.open("w", encoding="utf-8") as f:
            json.dump(sorted(sigs), f, ensure_ascii=False, indent=2)
        _badlist_cache = sigs
        _badlist_mtime = BADLIST_FILE.stat().st_mtime
    except Exception as e:
        logger.warning(f"Failed to save badlist: {e}")


def _extract_structured(fp: Path, max_pages: int) -> str:
    """
    Layout-aware извлечение текста с сохранением структуры.

    Использует get_text("dict") для получения блоков с координатами,
    затем сортирует по позиции (page → y → x) для правильного порядка чтения.

    Args:
        fp: Путь к PDF
        max_pages: Максимум страниц

    Returns:
        Извлечённый текст с сохранённой структурой
    """
    if fitz is None:
        raise PDFOpenError("PyMuPDF required for structured extraction")

    all_blocks = []

    with fitz.open(str(fp)) as doc:
        for page_num, page in enumerate(doc):
            if page_num >= max_pages:
                break

            # Получаем блоки с координатами
            data = page.get_text("dict")

            for block in data.get("blocks", []):
                if block.get("type") != 0:  # только text blocks
                    continue

                bbox = block.get("bbox", [0, 0, 0, 0])
                y_pos = bbox[1]  # верхняя граница
                x_pos = bbox[0]  # левая граница

                # Собираем текст из lines/spans
                block_text = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        block_text.append(line_text.strip())

                text = "\n".join(block_text)
                if text.strip():
                    all_blocks.append({
                        "page": page_num,
                        "y": round(y_pos, 1),  # округляем для стабильной сортировки
                        "x": round(x_pos, 1),
                        "text": text.strip()
                    })

    # Сортируем: по странице, потом по Y (сверху вниз), потом по X (слева направо)
    all_blocks.sort(key=lambda b: (b["page"], b["y"], b["x"]))

    # Собираем текст с разделителями между блоками
    return "\n\n".join(b["text"] for b in all_blocks)


def _fast_extract(fp: Path, max_pages: int) -> str:
    """
    Быстрое извлечение текста без subprocess (fallback).

    Args:
        fp: Путь к PDF
        max_pages: Максимум страниц

    Returns:
        Извлечённый текст
    """
    if fitz is not None:
        text_parts = []
        with fitz.open(str(fp)) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)

    elif pdfminer_extract is not None:
        return pdfminer_extract(str(fp), maxpages=max_pages) or ""

    else:
        raise PDFOpenError("No PDF library available (install PyMuPDF or pdfminer.six)")


def _worker_extract(fp_str: str, max_pages: int, out_q: mp.Queue) -> None:
    """Worker для извлечения в отдельном процессе."""
    fp = Path(fp_str)

    try:
        if fitz is not None:
            text_parts = []
            with fitz.open(str(fp)) as doc:
                for i, page in enumerate(doc):
                    if i >= max_pages:
                        break
                    text_parts.append(page.get_text("text"))
            out_q.put(("ok", "\n".join(text_parts)))

        elif pdfminer_extract is not None:
            out_q.put(("ok", pdfminer_extract(str(fp), maxpages=max_pages) or ""))

        else:
            out_q.put(("error", "No PDF library"))

    except (PDFPasswordIncorrect, PDFEncryptionError):
        out_q.put(("blocked", ""))
    except UnicodeDecodeError:
        out_q.put(("error", "decode-fail"))
    except Exception as e:
        err_msg = str(e).lower()
        if "password" in err_msg or "encrypt" in err_msg:
            out_q.put(("blocked", ""))
        else:
            out_q.put(("error", str(e)))


def _safe_extract_with_timeout(fp: Path, max_pages: int, timeout_sec: int) -> str:
    """
    Безопасное извлечение с таймаутом через subprocess.

    Args:
        fp: Путь к PDF
        max_pages: Максимум страниц
        timeout_sec: Таймаут в секундах

    Returns:
        Извлечённый текст
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_extract, args=(str(fp), max_pages, q), daemon=True)
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(2)
        raise PDFOpenError("timeout")

    try:
        status, payload = q.get_nowait()
    except Exception:
        raise PDFOpenError("no-result")

    if status == "ok":
        return payload
    if status == "blocked":
        raise PDFBlockedError("PDF is password-protected")
    raise PDFOpenError(str(payload or "extract-failed"))


def _is_small_simple(fp: Path) -> bool:
    """Проверяет является ли файл маленьким."""
    try:
        return fp.stat().st_size <= SMALL_SIZE_BYTES
    except Exception:
        return False


def extract_pdf_text(
    fp: Path | str,
    max_pages: int = None,
    timeout: int = None,
    structured: bool = True,
) -> str:
    """
    Извлечение текста из PDF.

    Args:
        fp: Путь к PDF файлу
        max_pages: Максимум страниц (по умолчанию из env)
        timeout: Таймаут в секундах (по умолчанию из env)
        structured: Использовать layout-aware извлечение (по умолчанию True)

    Returns:
        Извлечённый текст

    Raises:
        PDFOpenError: Ошибка открытия
        PDFBlockedError: PDF защищён паролем
    """
    fp = Path(fp)
    pages = max_pages or MAX_PAGES_FOR_TEXT
    timeout_sec = timeout or PDF_TIMEOUT_SEC

    if not fp.exists():
        raise PDFOpenError(f"File not found: {fp}")

    bad = _load_badlist()
    sig = _sig_for(fp)

    # Выбираем метод извлечения
    if structured and fitz is not None:
        extract_func = _extract_structured
    else:
        extract_func = _fast_extract

    # Если PDF_NO_TIMEOUT=1, всегда используем прямой путь
    if PDF_NO_TIMEOUT:
        try:
            return extract_func(fp, pages)
        except Exception as e:
            if "password" in str(e).lower() or "encrypt" in str(e).lower():
                raise PDFBlockedError("PDF is password-protected")
            raise PDFOpenError(str(e))

    use_safe = (sig in bad) or not _is_small_simple(fp)

    if not use_safe:
        # Пробуем прямой путь
        try:
            return extract_func(fp, pages)
        except PDFBlockedError:
            raise
        except Exception:
            # Добавляем в badlist и пробуем безопасный путь
            bad.add(sig)
            _save_badlist(bad)

    # Безопасный путь с таймаутом (fallback на _fast_extract в subprocess)
    try:
        return _safe_extract_with_timeout(fp, pages, timeout_sec)
    except PDFBlockedError:
        raise
    except PDFOpenError:
        bad = _load_badlist()
        bad.add(sig)
        _save_badlist(bad)
        raise


def extract_text_from_bytes(
    pdf_bytes: bytes,
    max_pages: int = None,
) -> str:
    """
    Извлечение текста из PDF байтов.

    Args:
        pdf_bytes: PDF как bytes
        max_pages: Максимум страниц

    Returns:
        Извлечённый текст
    """
    pages = max_pages or MAX_PAGES_FOR_TEXT

    if fitz is not None:
        text_parts = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                if i >= pages:
                    break
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)

    else:
        # Для pdfminer нужно сохранить во временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            return extract_pdf_text(temp_path, max_pages=pages)
        finally:
            Path(temp_path).unlink(missing_ok=True)


def is_pdf_readable(fp: Path | str) -> bool:
    """
    Проверяет можно ли прочитать PDF.

    Args:
        fp: Путь к файлу

    Returns:
        True если PDF можно прочитать
    """
    try:
        text = extract_pdf_text(fp, max_pages=1, timeout=5)
        return len(text.strip()) > 0
    except Exception:
        return False
