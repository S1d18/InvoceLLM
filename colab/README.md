# InvoiceLLM — Google Colab как LLM-бэкенд

## Обзор

Набор готовых notebook-ов для запуска LLM-бэкенда в Google Colab. Каждый notebook поднимает OpenAI-совместимый сервер (`/v1/chat/completions`) + публичный туннель, к которому можно подключить InvoiceLLM.

## Какой notebook выбрать?

| # | Notebook | GPU | API Key | Сложность | Скорость | Надёжность |
|---|----------|-----|---------|-----------|----------|------------|
| 1 | `01_gemini_api_proxy` | ❌ Нет | ✅ Нужен | ⭐ Легко | ⚡⚡⚡ Быстро | ⭐⭐⭐ Высокая |
| 2 | `02_colab_ai_wrapper` | ❌ Нет | ❌ Не нужен | ⭐ Легко | ⚡⚡ Средне | ⭐⭐ Средняя |
| 3 | `03_ollama_server` | ✅ T4/V100 | ❌ Не нужен | ⭐⭐ Средне | ⚡⚡ Средне | ⭐⭐ Средняя |
| 4 | `04_llamacpp_server` | ✅ T4/V100 | ❌ Не нужен | ⭐⭐⭐ Сложно | ⚡⚡ Средне | ⭐⭐⭐ Высокая |
| 5 | `05_batch_gdrive` | ❌ Нет | ✅ Нужен | ⭐ Легко | ⚡ Пакетно | ⭐⭐⭐ Высокая |

### Рекомендации

- **Быстрый старт без GPU**: `01_gemini_api_proxy` — нужен только API ключ Google AI Studio
- **Без ключей, без GPU**: `02_colab_ai_wrapper` — ноль настройки, но ограниченные возможности
- **Своя модель на GPU**: `03_ollama_server` — проще всего запустить Gemma/Llama
- **Максимум контроля**: `04_llamacpp_server` — для тех кто хочет тонко настроить параметры
- **Массовая обработка**: `05_batch_gdrive` — без туннеля, всё через Google Drive

## Быстрый старт

### 1. Запустите notebook в Colab

1. Откройте нужный `.ipynb` в Google Colab
2. Для GPU-notebook-ов: `Runtime → Change runtime type → T4 GPU`
3. Выполните все ячейки по порядку
4. Скопируйте URL туннеля из вывода (кроме `05_batch_gdrive`)

### 2. Проверьте подключение

```bash
python colab/client_test.py --url https://YOUR-TUNNEL-URL.trycloudflare.com
```

### 3. Подключите к InvoiceLLM

В `config.yaml` добавьте Colab-сервер:

```yaml
llm:
  servers:
    - name: "Google Colab"
      host: "YOUR-TUNNEL-URL.trycloudflare.com"
      port: 443
      ssl: true
      priority: 1
```

Или замените все серверы на один Colab endpoint.

## Туннели

Все notebook-и (кроме batch) используют **Cloudflare Quick Tunnel** по умолчанию:
- Бесплатный, без регистрации
- URL меняется при каждом перезапуске
- Альтернатива: ngrok (нужен токен, но стабильный URL)

## Ограничения Google Colab

- **Free tier**: ~12 часов сессии, T4 GPU при наличии
- **Colab Pro**: дольше, лучше GPU (V100/A100)
- Сессия отключается при неактивности (~30 мин)
- Cloudflare URL меняется при перезапуске

## Структура файлов

```
colab/
├── README.md                    ← Вы здесь
├── 01_gemini_api_proxy.ipynb    # Gemini API + FastAPI прокси
├── 02_colab_ai_wrapper.ipynb    # google.colab.ai обёртка
├── 03_ollama_server.ipynb       # Ollama + Gemma на GPU
├── 04_llamacpp_server.ipynb     # llama.cpp + GGUF на GPU
├── 05_batch_gdrive.ipynb        # Пакетная обработка через Drive
└── client_test.py               # Тестовый скрипт
```

## Устранение проблем

### Туннель не работает
- Проверьте что сервер запущен: в notebook должен быть вывод "Server running on port 8080"
- Подождите 10-15 секунд после запуска cloudflared
- Попробуйте ngrok как альтернативу

### GPU не доступен
- `Runtime → Change runtime type → T4 GPU`
- Free tier GPU может быть недоступен при высокой нагрузке — попробуйте позже
- Notebook-и 01, 02, 05 не требуют GPU

### Сессия отключается
- Colab отключает неактивные сессии через ~30 мин
- Держите вкладку открытой
- Colab Pro увеличивает лимит

### Ошибка "Model not found"
- Для Ollama: убедитесь что `ollama pull` завершился
- Для llama.cpp: проверьте что GGUF скачался полностью
- Для Gemini: проверьте API ключ
