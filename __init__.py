"""
Invoice LLM - LLM-first классификатор инвойсов с self-learning.

Архитектура:
    PDF → Template Cache → [HIT] → Instant Result (0.01s)
              │
              ▼ [MISS]
         LLM Cluster → Validator → Result
              │                       │
              └─────── Learn ◄────────┘
                          │
                          ▼
                   Template Cache (save)

Режимы работы:
    - NIGHT (23:00-06:00): Полная мощность LLM, batch processing
    - DAY (06:00-23:00): Только cache, серверы в sleep
    - FORCE: Ручной запуск LLM по требованию

Использование:
    from invoice_llm import classify_document

    result = classify_document(text, filename)
    print(f"Country: {result.country}")
    print(f"Doc Type: {result.doc_type}")

CLI:
    invoice-llm classify invoice.pdf
    invoice-llm batch M:/incoming --output results.csv
    invoice-llm status
"""

__version__ = "0.1.0"
__author__ = "Invoice LLM Team"

from .core import (
    InvoiceLLMClassifier,
    ClassificationResult,
    LLMCluster,
    TemplateCache,
    WorkScheduler,
    get_classifier,
    classify_document,
)

from .extractors import (
    extract_pdf_text,
    EntityExtractor,
    ExtractedEntities,
    DocumentFingerprint,
)

from .validators import (
    VATValidator,
    IBANValidator,
    HallucinationGuard,
    validate_vat,
    validate_iban,
)

__all__ = [
    # Core
    'InvoiceLLMClassifier',
    'ClassificationResult',
    'LLMCluster',
    'TemplateCache',
    'WorkScheduler',
    'get_classifier',
    'classify_document',

    # Extractors
    'extract_pdf_text',
    'EntityExtractor',
    'ExtractedEntities',
    'DocumentFingerprint',

    # Validators
    'VATValidator',
    'IBANValidator',
    'HallucinationGuard',
    'validate_vat',
    'validate_iban',
]
