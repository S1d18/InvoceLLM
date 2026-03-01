# Invoice LLM
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

LLM-first классификатор инвойсов с self-learning шаблонов.
Система классификации PDF-документов для сортировки счетов и инвойсов по стране, типу документа и компании. Использует каскадный гибридный подход: правила, извлечение сущностей, similarity-matching и ML-модели.

**Допустимые типы документов:** electricity, telecom, bank, water, gas, tax, other (по рекомендации накопление)

## Архитектура

```
PDF → Template Cache → [HIT] → Instant Result (0.01s)
          │
          ▼ [MISS]
     LLM Cluster → Validator → Result
          │                       │
          └─────── Learn ◄────────┘
                      │
                      ▼
               Template Cache (save)
```

### Режимы работы

- **NIGHT** (23:00-06:00): Полная мощность LLM, batch processing
- **DAY** (06:00-23:00): Только cache, серверы в sleep
- **FORCE**: Ручной запуск LLM по требованию

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### CLI

```bash
# Классификация одного файла
python run.py classify invoice.pdf

# Принудительный LLM (даже днём)
python run.py classify invoice.pdf --force-llm

# Batch обработка (только ночью или с --force)
python run.py batch M:/incoming --output results.csv --force

# Статус системы
python run.py status

# Управление серверами
python run.py servers wake      # Wake-on-LAN
python run.py servers sleep     # Suspend
python run.py servers status    # Health check

# Статистика кэша
python run.py cache stats
python run.py cache clear --older-than 90
python run.py cache export templates.json
python run.py cache import templates.json
```

### Python API

```python
from invoice_llm import classify_document, extract_pdf_text

# Извлечение текста из PDF
text = extract_pdf_text("invoice.pdf")

# Классификация
result = classify_document(text, "invoice.pdf")

print(f"Country: {result.country} ({result.country_confidence:.0%})")
print(f"Doc Type: {result.doc_type}")
print(f"Company: {result.company}")
print(f"Year: {result.year}")
print(f"Source: {result.source}")  # template_cache, llm, no_match
```

## Конфигурация

Редактируйте `config.yaml`:

```yaml
scheduler:
  night_start: "23:00"
  night_end: "06:00"
  wake_on_lan: true

servers:
  - name: "Main PC"
    host: "192.168.50.20"
    ports: [8080, 8081, 8082]
    mac: "AA:BB:CC:DD:EE:F1"

llm:
  temperature: 0.0
  max_tokens: 200
  timeout: 30

cache:
  hit_threshold: 0.95
  learn_threshold: 0.85
```

## Компоненты

### Core
- `core/classifier.py` - Главный классификатор
- `core/llm_client.py` - LLM cluster client (round-robin)
- `core/template_cache.py` - Авто-шаблоны с self-learning
- `core/scheduler.py` - Режимы работы (night/day/force)

### Extractors
- `extractors/pdf_extractor.py` - Извлечение текста из PDF
- `extractors/entity_extractor.py` - VAT, IBAN, телефоны
- `extractors/fingerprint.py` - Document fingerprinting

### Validators
- `validators/vat_validator.py` - VAT checksum validation
- `validators/iban_validator.py` - IBAN validation
- `validators/hallucination_guard.py` - Anti-hallucination checks

## LLM Серверы

Требуется llama.cpp server с OpenAI-compatible API:

```bash
llama-server -m model.gguf -c 4096 --port 8080
```

Рекомендуемые модели: Qwen2.5-3B, Llama-3.2-3B (Q4_K_M квантизация)

## Self-Learning

Система автоматически сохраняет шаблоны для документов с высоким confidence:

1. Документ классифицируется через LLM
2. Если confidence >= 0.85, создаётся fingerprint
3. Fingerprint сохраняется в SQLite кэш
4. При следующей встрече похожего документа → instant result из кэша

## Структура проекта

```
  D:/python/InvoceLLM/                                                                                                                                           
  ├── __init__.py                 # Главный модуль                                                                                                               
  ├── run.py                      # Entry point                                                                                                                  
  ├── config.yaml                 # Конфигурация                                                                                                                 
  ├── requirements.txt            # Зависимости                                                                                                                  
  ├── README.md                   # Документация                                                                                                                 
  │                                                                                                                                                              
  ├── core/                                                                                                                                                      
  │   ├── __init__.py                                                                                                                                            
  │   ├── classifier.py           # Главный классификатор (~200 строк)                                                                                           
  │   ├── llm_client.py           # LLM cluster client (~350 строк)                                                                                              
  │   ├── template_cache.py       # Self-learning кэш (~350 строк)                                                                                               
  │   └── scheduler.py            # Режимы работы (~200 строк)                                                                                                   
  │                                                                                                                                                              
  ├── extractors/                                                                                                                                                
  │   ├── __init__.py                                                                                                                                            
  │   ├── pdf_extractor.py        # PDF извлечение (~200 строк)                                                                                                  
  │   ├── entity_extractor.py     # VAT/IBAN/phones (~350 строк)                                                                                                 
  │   └── fingerprint.py          # Document fingerprint (~150 строк)                                                                                            
  │                                                                                                                                                              
  ├── validators/                                                                                                                                                
  │   ├── __init__.py                                                                                                                                            
  │   ├── vat_validator.py        # VAT checksum (~300 строк)                                                                                                    
  │   ├── iban_validator.py       # IBAN validation (~150 строк)                                                                                                 
  │   └── hallucination_guard.py  # Anti-hallucination (~200 строк)                                                                                              
  │                                                                                                                                                              
  ├── cli/                                                                                                                                                       
  │   ├── __init__.py                                                                                                                                            
  │   └── main.py                 # CLI интерфейс (~350 строк)                                                                                                   
  │                                                                                                                                                              
  ├── data/                                                                                                                                                      
  │   ├── companies.json          # ~288 KB (из New_sort)                                                                                                        
  │   ├── company_aliases.json    # ~15 KB (из New_sort)                                                                                                         
  │   └── templates/              # SQLite кэш шаблонов                                                                                                          
  │                                                                                                                                                              
  ├── servers/                                                                                                                                                   
  ├── api/                                                                                                                                                       
  └── tests/                                                                                                                                                     
                                                                                                                                                                 
  Использование                                                                                                                                                  
                                                                                                                                                                 
  # Установка зависимостей                                                                                                                                       
  cd D:\python\InvoceLLM                                                                                                                                         
  pip install -r requirements.txt                                                                                                                                
                                                                                                                                                                 
  # Классификация                                                                                                                                                
  python run.py classify invoice.pdf                                                                                                                             
  python run.py classify invoice.pdf --force-llm                                                                                                                 
                                                                                                                                                                 
  # Batch обработка                                                                                                                                              
  python run.py batch M:/incoming --output results.csv --force                                                                                                   
                                                                                                                                                                 
  # Статус                                                                                                                                                       
  python run.py status                                                                                                                                           
                                                                                                                                                                 
  # Управление серверами                                                                                                                                         
  python run.py servers wake                                                                                                                                     
  python run.py servers status                                                                                                                                   
                                                                                                                                                                 
  # Кэш                                                                                                                                                          
  python run.py cache stats                                                                                                                                      
                                                                                                                                                                 
  Ключевые особенности                                                                                                                                           
                                                                                                                                                                 
  1. Self-learning: Автоматическое накопление шаблонов для мгновенной классификации                                                                              
  2. Scheduler: Режимы NIGHT/DAY/FORCE для экономии энергии                                                                                                      
  3. Wake-on-LAN: Автоматическое пробуждение серверов                                                                                                            
  4. Валидация: VAT/IBAN checksum + защита от галлюцинаций LLM                                                                                                   
  5. Round-robin: Балансировка между несколькими LLM серверами    
```
