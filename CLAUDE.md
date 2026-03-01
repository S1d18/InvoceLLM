# Invoice LLM - Руководство для разработки

## Обзор проекта

LLM-first классификатор инвойсов с self-learning шаблонов. Автоматически извлекает данные из PDF-счетов используя LLM кластер с кэшированием шаблонов.

## Команды разработки

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск CLI
python run.py <command>

# Основные команды
python run.py classify <pdf_path>    # Классификация одного PDF
python run.py batch <directory>      # Пакетная обработка директории
python run.py status                 # Статус системы и серверов
python run.py servers                # Управление LLM серверами
python run.py cache                  # Управление кэшем шаблонов

# Тесты (требует раскомментировать pytest в requirements.txt)
pytest
```

## Архитектура

### Поток данных
```
PDF → Template Cache → [HIT] → Result
                    → [MISS] → LLM Cluster → Validator → Learn → Result
```

### Режимы работы
- **NIGHT**: Полная LLM мощность (автоматический Wake-on-LAN серверов)
- **DAY**: Только кэш шаблонов (энергосбережение)
- **FORCE**: Ручной режим с принудительным LLM

### Core модули (`src/core/`)
- `classifier.py` - Основная логика классификации
- `llm_client.py` - Клиент для llama.cpp OpenAI-compatible API
- `template_cache.py` - SQLite кэш с auto-learning
- `scheduler.py` - Планировщик режимов DAY/NIGHT

### Extractors (`src/extractors/`)
- `pdf_extractor.py` - Извлечение текста/таблиц из PDF
- `entity_extractor.py` - NER для извлечения сущностей
- `fingerprint.py` - Генерация fingerprint документа

### Validators (`src/validators/`)
- `vat_validator.py` - Валидация VAT номеров
- `iban_validator.py` - Валидация IBAN
- `hallucination_guard.py` - Защита от галлюцинаций LLM

## Ключевые концепции

### Template Cache
SQLite-based кэш шаблонов. При `confidence >= 0.85` автоматически сохраняет новый шаблон. Fingerprint документа используется для быстрого поиска похожих.

### Document Fingerprinting
Хэширование структуры документа (позиции блоков, соотношения) для быстрого сравнения без полного анализа.

### Wake-on-LAN
Автоматическое включение LLM серверов в NIGHT режиме, отключение в DAY для экономии энергии.

### HallucinationGuard
Валидация LLM ответов: проверка что извлечённые данные присутствуют в исходном тексте, кросс-валидация форматов (VAT, IBAN, даты).

## Конфигурация

Основные настройки в `config/settings.yaml`:
- LLM серверы и их MAC-адреса
- Пороги confidence для auto-learning
- Расписание DAY/NIGHT режимов

## LLM Backend

Использует llama.cpp server с OpenAI-compatible API. Поддерживает кластер из нескольких серверов с load balancing.
