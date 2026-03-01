"""
Core модули Invoice LLM.

- classifier: Главный классификатор
- llm_client: Клиент для LLM кластера
- template_cache: Кэш шаблонов с self-learning
- scheduler: Управление режимами работы
- organizer: Организатор PDF по директориям
- mega_batch: Инкрементальная обработка 500K+ PDF
"""

from .classifier import InvoiceLLMClassifier, ClassificationResult, SchedulerError
from .llm_client import LLMCluster, get_llm_cluster
from .template_cache import TemplateCache, get_template_cache
from .scheduler import WorkScheduler, get_scheduler
from .organizer import PDFOrganizer, OrganizeResult, OrganizeReport, organize_pdfs
from .mega_batch import ProgressDB, MegaBatchProcessor

__all__ = [
    # classifier
    'InvoiceLLMClassifier',
    'ClassificationResult',
    'SchedulerError',
    # llm_client
    'LLMCluster',
    'get_llm_cluster',
    # template_cache
    'TemplateCache',
    'get_template_cache',
    # scheduler
    'WorkScheduler',
    'get_scheduler',
    # organizer
    'PDFOrganizer',
    'OrganizeResult',
    'OrganizeReport',
    'organize_pdfs',
    # mega_batch
    'ProgressDB',
    'MegaBatchProcessor',
]
