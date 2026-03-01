"""
Главный классификатор Invoice LLM.

LLM-first архитектура с self-learning шаблонов:
1. Проверяем кэш шаблонов (instant)
2. Если нет в кэше и режим позволяет → LLM
3. Валидация результата
4. Сохранение в кэш (self-learning)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import yaml

from .template_cache import TemplateCache, get_template_cache
from .llm_client import LLMCluster, LLMResult, get_llm_cluster
from .scheduler import WorkScheduler, WorkMode, get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Результат классификации документа."""
    # Основные поля
    country: Optional[str] = None
    country_confidence: float = 0.0
    country_source: str = ""  # "template_cache", "llm", "no_match"

    doc_type: Optional[str] = None
    doc_type_confidence: float = 0.0
    doc_category: Optional[str] = None

    company: Optional[str] = None
    year: Optional[int] = None

    # Метаданные
    source: str = ""  # "template_cache", "template_cache_partial", "llm", "no_match"
    processing_time: float = 0.0
    needs_llm: bool = False  # True если нужна обработка ночью

    # Валидация
    is_valid: bool = False
    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)

    # LLM данные
    raw_llm_response: str = ""
    server_used: str = ""

    @property
    def confidence(self) -> float:
        """Общий confidence (минимум из country и doc_type)."""
        confidences = [c for c in [self.country_confidence, self.doc_type_confidence] if c > 0]
        return min(confidences) if confidences else 0.0


class SchedulerError(Exception):
    """Ошибка scheduler (например, batch в дневное время)."""
    pass


class InvoiceLLMClassifier:
    """
    Главный классификатор Invoice LLM.

    Алгоритм:
    1. Проверяем кэш шаблонов (instant)
    2. Если нет в кэше и режим позволяет → LLM
    3. Валидация результата
    4. Сохранение в кэш (self-learning)

    Использование:
        classifier = InvoiceLLMClassifier()
        result = classifier.classify(text, filename)

        # Принудительное использование LLM
        result = classifier.classify(text, filename, force_llm=True)
    """

    def __init__(
        self,
        config_path: str | Path = None,
        config: dict = None,
    ):
        """
        Инициализация классификатора.

        Args:
            config_path: Путь к config.yaml
            config: Конфигурация как dict (приоритет над config_path)
        """
        # Загрузка конфигурации
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            # Ищем config.yaml в разных местах
            for path in ['config.yaml', 'data/config.yaml', '../config.yaml']:
                if Path(path).exists():
                    self.config = self._load_config(path)
                    break
            else:
                self.config = {}

        # Инициализация компонентов
        self.template_cache = get_template_cache(self.config)
        self.llm_cluster = get_llm_cluster(self.config)
        self.scheduler = get_scheduler(self.config)

        # Валидатор (lazy init)
        self._validator = None

        # Пороги
        cache_config = self.config.get('cache', {})
        self.cache_hit_threshold = cache_config.get('hit_threshold', 0.95)
        self.learn_threshold = cache_config.get('learn_threshold', 0.85)

        logger.info("InvoiceLLMClassifier initialized")

    def _load_config(self, config_path: str | Path) -> dict:
        """Загружает конфигурацию из YAML."""
        config_path = Path(config_path)
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def validator(self):
        """Lazy-загрузка валидатора."""
        if self._validator is None:
            from ..validators import HallucinationGuard
            self._validator = HallucinationGuard(self.config)
        return self._validator

    def classify(
        self,
        text: str,
        filename: str = "",
        force_llm: bool = False,
    ) -> ClassificationResult:
        """
        Классификация документа.

        Args:
            text: Текст документа
            filename: Имя файла
            force_llm: Принудительно использовать LLM (даже днём)

        Returns:
            ClassificationResult с результатами
        """
        start_time = time.time()

        if not text or len(text.strip()) < 20:
            return ClassificationResult(
                source='no_match',
                is_valid=False,
                validation_errors=['Text too short'],
                processing_time=time.time() - start_time,
            )

        # 1. Проверяем кэш
        cached = self.template_cache.match(text, filename)

        if cached and cached.confidence >= self.cache_hit_threshold:
            logger.debug(f"Cache hit: {cached.company or cached.fingerprint[:16]}")
            return ClassificationResult(
                country=cached.country,
                country_confidence=cached.confidence,
                country_source='template_cache',
                doc_type=cached.doc_type,
                doc_type_confidence=cached.confidence,
                doc_category=cached.doc_category,
                company=cached.company,
                source='template_cache',
                is_valid=True,
                processing_time=time.time() - start_time,
            )

        # 2. Проверяем можно ли использовать LLM
        if not self.scheduler.can_use_llm(force=force_llm):
            # Днём без LLM
            if cached:
                # Возвращаем частичный результат из кэша
                logger.debug(f"Partial cache hit (day mode): {cached.fingerprint[:16]}")
                return ClassificationResult(
                    country=cached.country,
                    country_confidence=cached.confidence * 0.9,
                    country_source='template_cache_partial',
                    doc_type=cached.doc_type,
                    doc_type_confidence=cached.confidence * 0.9,
                    doc_category=cached.doc_category,
                    company=cached.company,
                    source='template_cache_partial',
                    is_valid=True,
                    processing_time=time.time() - start_time,
                )

            # Нет в кэше и LLM недоступен
            return ClassificationResult(
                source='no_match',
                is_valid=False,
                needs_llm=True,  # Пометка для batch обработки ночью
                processing_time=time.time() - start_time,
            )

        # 3. LLM классификация
        llm_result = self.llm_cluster.classify(text, filename)

        # 4. Валидация результата
        result = self._convert_llm_result(llm_result, text)
        result = self._validate_result(result, text)

        # 5. Self-learning: сохраняем в кэш
        if result.confidence >= self.learn_threshold and result.is_valid:
            self.template_cache.learn(text, result, filename)

        result.processing_time = time.time() - start_time

        return result

    def classify_batch(
        self,
        documents: list[tuple[str, str]],
        force: bool = False,
        parallel: bool = True,
    ) -> list[ClassificationResult]:
        """
        Batch классификация документов.

        Args:
            documents: Список (text, filename)
            force: Принудительный режим (игнорировать scheduler)
            parallel: Параллельная обработка

        Returns:
            Список результатов
        """
        # Проверяем режим
        if not self.scheduler.can_use_llm(force=force):
            raise SchedulerError(
                f"Batch processing only available in NIGHT mode. "
                f"Current mode: {self.scheduler.current_mode.value}. "
                f"Use force=True to override."
            )

        results = []

        # Сначала проверяем кэш
        cache_results = []
        llm_needed = []

        for i, (text, filename) in enumerate(documents):
            cached = self.template_cache.match(text, filename)
            if cached and cached.confidence >= self.cache_hit_threshold:
                cache_results.append((i, ClassificationResult(
                    country=cached.country,
                    country_confidence=cached.confidence,
                    country_source='template_cache',
                    doc_type=cached.doc_type,
                    doc_type_confidence=cached.confidence,
                    doc_category=cached.doc_category,
                    company=cached.company,
                    source='template_cache',
                    is_valid=True,
                )))
            else:
                llm_needed.append((i, text, filename))

        logger.info(f"Batch: {len(cache_results)} cache hits, {len(llm_needed)} need LLM")

        # LLM классификация для остальных
        if llm_needed and parallel:
            llm_docs = [(text, filename) for _, text, filename in llm_needed]
            llm_results = self.llm_cluster.classify_batch(llm_docs)

            for (i, text, filename), llm_result in zip(llm_needed, llm_results):
                result = self._convert_llm_result(llm_result, text)
                result = self._validate_result(result, text)

                # Self-learning
                if result.confidence >= self.learn_threshold and result.is_valid:
                    self.template_cache.learn(text, result, filename)

                cache_results.append((i, result))

        elif llm_needed:
            # Последовательная обработка
            for i, text, filename in llm_needed:
                llm_result = self.llm_cluster.classify(text, filename)
                result = self._convert_llm_result(llm_result, text)
                result = self._validate_result(result, text)

                if result.confidence >= self.learn_threshold and result.is_valid:
                    self.template_cache.learn(text, result, filename)

                cache_results.append((i, result))

        # Сортируем по оригинальному порядку
        cache_results.sort(key=lambda x: x[0])
        results = [r for _, r in cache_results]

        return results

    def _convert_llm_result(self, llm_result: LLMResult, text: str) -> ClassificationResult:
        """Конвертирует LLMResult в ClassificationResult."""
        return ClassificationResult(
            country=llm_result.country,
            country_confidence=llm_result.country_confidence,
            country_source='llm',
            doc_type=llm_result.doc_type,
            doc_type_confidence=llm_result.doc_type_confidence,
            doc_category=llm_result.doc_category,
            company=llm_result.company,
            year=llm_result.year,
            source='llm',
            is_valid=llm_result.is_valid,
            validation_errors=llm_result.validation_errors.copy(),
            raw_llm_response=llm_result.raw_response,
            server_used=llm_result.server_used,
        )

    def _validate_result(self, result: ClassificationResult, text: str) -> ClassificationResult:
        """Валидирует результат с помощью HallucinationGuard."""
        try:
            validation = self.validator.validate(result, text)

            if validation.errors:
                result.validation_errors.extend(validation.errors)
                result.is_valid = False

            if validation.warnings:
                result.validation_warnings.extend(validation.warnings)

            # Применяем penalty к confidence
            if validation.confidence_penalty > 0:
                result.country_confidence = max(0, result.country_confidence - validation.confidence_penalty)
                result.doc_type_confidence = max(0, result.doc_type_confidence - validation.confidence_penalty)

        except Exception as e:
            logger.warning(f"Validation error: {e}")

        return result

    def get_status(self) -> dict:
        """Возвращает статус системы."""
        cache_stats = self.template_cache.get_stats()
        scheduler_status = self.scheduler.get_status()
        server_stats = self.llm_cluster.get_server_stats()

        return {
            'scheduler': scheduler_status,
            'cache': {
                'total_templates': cache_stats.total_templates,
                'countries': cache_stats.countries,
                'companies': cache_stats.companies,
                'hit_rate': f"{cache_stats.hit_rate_24h:.1%}",
            },
            'llm': {
                'available_servers': server_stats['available_endpoints'],
                'total_servers': server_stats['total_endpoints'],
                'dead_servers': server_stats['dead_endpoints'],
                'all_dead': server_stats['all_dead'],
            },
        }


# Singleton instance
_classifier: Optional[InvoiceLLMClassifier] = None


def get_classifier(config_path: str = None, config: dict = None) -> InvoiceLLMClassifier:
    """Возвращает singleton экземпляр классификатора."""
    global _classifier
    if _classifier is None:
        _classifier = InvoiceLLMClassifier(config_path=config_path, config=config)
    return _classifier


def classify_document(text: str, filename: str = "", force_llm: bool = False) -> ClassificationResult:
    """Удобная функция для классификации."""
    return get_classifier().classify(text, filename, force_llm=force_llm)
