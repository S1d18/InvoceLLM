"""
CLI интерфейс для Invoice LLM.

Использование:
    invoice-llm classify invoice.pdf
    invoice-llm classify invoice.pdf --force-llm
    invoice-llm batch M:/incoming --output M:/sorted --force
    invoice-llm status
    invoice-llm servers wake
    invoice-llm servers sleep
    invoice-llm cache stats
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

# Настройка logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_path():
    """Добавляет корневую директорию в path."""
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def cmd_classify(args):
    """Классификация одного файла."""
    setup_path()

    from core import InvoiceLLMClassifier
    from extractors import extract_pdf_text

    classifier = InvoiceLLMClassifier(config_path=args.config)

    # Извлечение текста
    try:
        text = extract_pdf_text(args.file)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 1

    # Классификация
    result = classifier.classify(text, Path(args.file).name, force_llm=args.force_llm)

    # Вывод результата
    print(f"\n{'='*60}")
    print(f"File: {args.file}")
    print(f"{'='*60}")
    print(f"Country:    {result.country or 'Unknown'} ({result.country_confidence:.0%})")
    print(f"Category:   {result.doc_category or 'Unknown'}")
    print(f"Doc Type:   {result.doc_type or 'Unknown'} ({result.doc_type_confidence:.0%})")
    print(f"Company:    {result.company or 'Unknown'}")
    print(f"Year:       {result.year or 'Unknown'}")
    print(f"Source:     {result.source}")
    print(f"Valid:      {result.is_valid}")

    if result.validation_errors:
        print(f"\nErrors:")
        for err in result.validation_errors:
            print(f"  - {err}")

    if result.validation_warnings:
        print(f"\nWarnings:")
        for warn in result.validation_warnings:
            print(f"  - {warn}")

    if args.verbose and result.raw_llm_response:
        print(f"\nLLM Response:")
        print(f"  {result.raw_llm_response}")

    print(f"\nProcessing time: {result.processing_time:.3f}s")

    return 0


def cmd_batch(args):
    """Batch классификация директории."""
    setup_path()

    from core import InvoiceLLMClassifier, SchedulerError
    from extractors import extract_pdf_text

    classifier = InvoiceLLMClassifier(config_path=args.config)

    # Проверка режима
    status = classifier.scheduler.get_status()
    if not classifier.scheduler.can_use_llm(force=args.force):
        print(f"Error: Batch processing requires NIGHT mode or --force flag")
        print(f"Current mode: {status['mode']}")
        if 'time_until_night' in status:
            print(f"Next batch: {status['time_until_night']}")
        return 1

    # Сбор PDF файлов
    input_dir = Path(args.input_dir)
    pdf_files = list(input_dir.glob('**/*.pdf'))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return 1

    print(f"Found {len(pdf_files)} PDF files")

    # Извлечение текста
    documents = []
    for pdf_file in pdf_files:
        try:
            text = extract_pdf_text(pdf_file)
            documents.append((text, pdf_file.name, pdf_file))
        except Exception as e:
            logger.warning(f"Failed to read {pdf_file}: {e}")

    print(f"Successfully read {len(documents)} files")

    # Классификация
    results = []
    try:
        llm_results = classifier.classify_batch(
            [(text, name) for text, name, _ in documents],
            force=args.force,
        )

        for (text, name, path), result in zip(documents, llm_results):
            results.append({
                'file': str(path),
                'filename': name,
                'country': result.country,
                'country_confidence': result.country_confidence,
                'doc_category': result.doc_category,
                'doc_type': result.doc_type,
                'doc_type_confidence': result.doc_type_confidence,
                'company': result.company,
                'year': result.year,
                'source': result.source,
                'is_valid': result.is_valid,
            })

    except SchedulerError as e:
        print(f"Error: {e}")
        return 1

    # Сохранение результатов
    if args.output:
        output_path = Path(args.output)
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_path}")

    # Статистика
    countries = {}
    doc_types = {}
    for r in results:
        if r['country']:
            countries[r['country']] = countries.get(r['country'], 0) + 1
        if r['doc_type']:
            doc_types[r['doc_type']] = doc_types.get(r['doc_type'], 0) + 1

    print(f"\n{'='*40}")
    print("Results Summary")
    print(f"{'='*40}")
    print(f"Total processed: {len(results)}")
    print(f"\nCountries:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1]):
        print(f"  {country}: {count}")
    print(f"\nDoc Types:")
    for dtype, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}")

    return 0


def cmd_status(args):
    """Статус системы."""
    setup_path()

    from core import InvoiceLLMClassifier

    classifier = InvoiceLLMClassifier(config_path=args.config)
    status = classifier.get_status()

    print(f"\n{'='*40}")
    print("Invoice LLM Status")
    print(f"{'='*40}")

    # Scheduler
    sched = status['scheduler']
    print(f"\nScheduler:")
    print(f"  Mode: {sched['mode'].upper()}")
    print(f"  Can use LLM: {sched['can_use_llm']}")
    print(f"  Night hours: {sched['night_start']} - {sched['night_end']}")
    if 'time_until_night' in sched:
        print(f"  Next batch: in {sched['time_until_night']}")
    if 'time_until_day' in sched:
        print(f"  Night ends: in {sched['time_until_day']}")

    # Cache
    cache = status['cache']
    print(f"\nCache:")
    print(f"  Templates: {cache['total_templates']}")
    print(f"  Countries: {cache['countries']}")
    print(f"  Companies: {cache['companies']}")
    print(f"  Hit rate: {cache['hit_rate']}")

    # LLM
    llm = status['llm']
    print(f"\nLLM Servers:")
    print(f"  Available: {llm['available_servers']}/{llm['total_servers']}")
    if llm.get('dead_servers', 0) > 0:
        print(f"  Dead: {llm['dead_servers']}")
    if llm.get('all_dead'):
        print(f"  WARNING: All servers are DEAD!")

    return 0


def cmd_servers_wake(args):
    """Wake-on-LAN для серверов."""
    setup_path()

    from core import get_scheduler

    scheduler = get_scheduler()
    woken = scheduler.wake_servers()

    if woken:
        print(f"Sent WoL packets to: {', '.join(woken)}")
    else:
        print("No servers to wake (check config or WoL settings)")

    return 0


def cmd_servers_sleep(args):
    """Отправка серверов в sleep."""
    setup_path()

    from core import get_scheduler

    scheduler = get_scheduler()
    slept = scheduler.sleep_servers()

    if slept:
        print(f"Sent sleep command to: {', '.join(slept)}")
    else:
        print("No servers to sleep (check config or SSH settings)")

    return 0


def cmd_servers_status(args):
    """Статус серверов."""
    setup_path()

    from core import get_llm_cluster

    cluster = get_llm_cluster()
    cluster.is_available(force_check=True)

    stats = cluster.get_server_stats()

    print(f"\nLLM Servers Status")
    print(f"{'='*40}")
    print(f"Total endpoints: {stats['total_endpoints']}")
    print(f"Available: {stats['available_endpoints']}")

    dead_set = set(stats.get('dead', []))
    fail_counts = stats.get('fail_counts', {})

    print(f"\nEndpoints:")
    for ep in stats['endpoints']:
        if ep in dead_set:
            status = "DEAD"
        elif ep in stats['available']:
            status = "OK"
        else:
            status = "DOWN"
        fails = fail_counts.get(ep, 0)
        fail_str = f" (fails: {fails})" if fails > 0 else ""
        print(f"  {ep}: {status}{fail_str}")

    if stats.get('all_dead'):
        print(f"\n  WARNING: All servers are DEAD!")

    return 0


def cmd_cache_stats(args):
    """Статистика кэша."""
    setup_path()

    from core import get_template_cache

    cache = get_template_cache()
    stats = cache.get_stats()

    print(f"\nTemplate Cache Statistics")
    print(f"{'='*40}")
    print(f"Total templates: {stats.total_templates}")
    print(f"Countries: {stats.countries}")
    print(f"Companies: {stats.companies}")
    print(f"Total hits: {stats.total_hits}")
    print(f"Hit rate: {stats.hit_rate_24h:.1%}")

    if stats.oldest_template:
        print(f"Oldest template: {stats.oldest_template}")
    if stats.newest_template:
        print(f"Newest template: {stats.newest_template}")

    return 0


def cmd_cache_clear(args):
    """Очистка старых шаблонов."""
    setup_path()

    from core import get_template_cache

    cache = get_template_cache()

    before = cache.get_stats().total_templates
    cache.clear_old(days=args.older_than)
    after = cache.get_stats().total_templates

    print(f"Cleared {before - after} templates older than {args.older_than} days")
    print(f"Remaining: {after} templates")

    return 0


def cmd_cache_export(args):
    """Экспорт шаблонов."""
    setup_path()

    from core import get_template_cache

    cache = get_template_cache()
    cache.export_templates(args.output)

    print(f"Templates exported to {args.output}")

    return 0


def cmd_cache_import(args):
    """Импорт шаблонов."""
    setup_path()

    from core import get_template_cache

    cache = get_template_cache()
    cache.import_templates(args.input)

    print(f"Templates imported from {args.input}")

    return 0


def cmd_organize(args):
    """Организация PDF файлов по директориям."""
    setup_path()

    from core import PDFOrganizer

    # Создаем организатор
    organizer = PDFOrganizer(
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        trash_dir=args.trash_dir,
        config_path=args.config,
    )

    # Определяем режим: перемещение или копирование
    move = not args.copy

    # Определяем использование trash директории
    use_trash = not args.no_trash

    print(f"\n{'='*60}")
    print("PDF Organizer")
    print(f"{'='*60}")
    print(f"Source:         {args.source_dir}")
    print(f"Output:         {args.output_dir}")
    print(f"Min confidence: {args.min_confidence:.0%}")
    print(f"Mode:           {'MOVE' if move else 'COPY'}")
    print(f"Trash:          {organizer.trash_dir if use_trash else 'disabled'}")
    if args.dry_run:
        print(f"DRY RUN:        No files will be modified")
    print(f"{'='*60}\n")

    # Запускаем организацию
    report = organizer.organize(
        source_dir=args.source_dir,
        move=move,
        dry_run=args.dry_run,
        force_llm=args.force,
        use_trash=use_trash,
    )

    # Вывод результатов
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Total files:    {report.total_files}")
    print(f"Successful:     {report.successful}")
    print(f"Moved to trash: {report.moved_to_trash}")
    print(f"Skipped:        {report.skipped}")
    print(f"Errors:         {report.errors}")

    if report.by_country:
        print(f"\nBy Country:")
        for country, count in sorted(report.by_country.items(), key=lambda x: -x[1]):
            print(f"  {country}: {count}")

    if report.by_doc_type:
        print(f"\nBy Doc Type:")
        for dtype, count in sorted(report.by_doc_type.items(), key=lambda x: -x[1]):
            print(f"  {dtype}: {count}")

    # Сохранение отчета
    if args.report:
        report_path = Path(args.report)
        if report_path.suffix.lower() == '.csv':
            _save_report_csv(report, report_path)
        else:
            _save_report_json(report, report_path)
        print(f"\nReport saved to: {report_path}")

    # Подробный вывод при verbose или dry_run
    if args.verbose or args.dry_run:
        print(f"\n{'='*60}")
        print("Detailed Results")
        print(f"{'='*60}")
        for r in report.results:
            status_icon = {
                "success": "+",
                "trash": "T",
                "skipped": "-",
                "error": "!",
            }.get(r.status, "?")

            if r.dest_path:
                print(f"[{status_icon}] {r.source_path.name}")
                print(f"    -> {r.dest_path}")
            else:
                print(f"[{status_icon}] {r.source_path.name}: {r.error or r.status}")

    return 0 if report.errors == 0 else 1


def _save_report_csv(report, path: Path):
    """Сохраняет отчет в CSV."""
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'source', 'destination', 'status', 'country', 'doc_type',
            'company', 'year', 'confidence', 'error'
        ])
        writer.writeheader()

        for r in report.results:
            row = {
                'source': str(r.source_path),
                'destination': str(r.dest_path) if r.dest_path else '',
                'status': r.status,
                'country': r.classification.country if r.classification else '',
                'doc_type': r.classification.doc_type if r.classification else '',
                'company': r.classification.company if r.classification else '',
                'year': r.classification.year if r.classification else '',
                'confidence': f"{r.classification.confidence:.2f}" if r.classification else '',
                'error': r.error or '',
            }
            writer.writerow(row)


def _save_report_json(report, path: Path):
    """Сохраняет отчет в JSON."""
    import json

    data = {
        'summary': {
            'total_files': report.total_files,
            'successful': report.successful,
            'moved_to_trash': report.moved_to_trash,
            'skipped': report.skipped,
            'errors': report.errors,
            'by_country': report.by_country,
            'by_doc_type': report.by_doc_type,
        },
        'results': [
            {
                'source': str(r.source_path),
                'destination': str(r.dest_path) if r.dest_path else None,
                'status': r.status,
                'classification': {
                    'country': r.classification.country,
                    'doc_type': r.classification.doc_type,
                    'company': r.classification.company,
                    'year': r.classification.year,
                    'confidence': r.classification.confidence,
                } if r.classification else None,
                'error': r.error,
            }
            for r in report.results
        ],
    }

    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog='invoice-llm',
        description='Invoice LLM - LLM-first классификатор инвойсов'
    )
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to config file')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # classify
    p_classify = subparsers.add_parser('classify', help='Classify a single PDF')
    p_classify.add_argument('file', help='PDF file to classify')
    p_classify.add_argument('--force-llm', '-f', action='store_true', help='Force LLM even in DAY mode')
    p_classify.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    p_classify.set_defaults(func=cmd_classify)

    # batch
    p_batch = subparsers.add_parser('batch', help='Batch classify directory')
    p_batch.add_argument('input_dir', help='Input directory with PDFs')
    p_batch.add_argument('--output', '-o', help='Output CSV file')
    p_batch.add_argument('--force', '-f', action='store_true', help='Force processing in DAY mode')
    p_batch.set_defaults(func=cmd_batch)

    # organize
    p_organize = subparsers.add_parser('organize', help='Organize PDFs into directories')
    p_organize.add_argument('source_dir', help='Source directory with PDFs')
    p_organize.add_argument('output_dir', help='Output directory for organized files')
    p_organize.add_argument(
        '--min-confidence', type=float, default=0.7,
        help='Minimum confidence threshold (default: 0.7)'
    )
    p_organize.add_argument(
        '--copy', action='store_true',
        help='Copy files instead of moving (default: move)'
    )
    p_organize.add_argument(
        '--no-trash', action='store_true',
        help='Do not create unclassified folder for invalid files'
    )
    p_organize.add_argument(
        '--trash-dir',
        help='Custom directory for unclassified files (default: {output}/unclassified)'
    )
    p_organize.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be done without making changes'
    )
    p_organize.add_argument(
        '--force', '-f', action='store_true',
        help='Force LLM usage even in DAY mode'
    )
    p_organize.add_argument(
        '--report',
        help='Save report to file (CSV or JSON based on extension)'
    )
    p_organize.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output'
    )
    p_organize.set_defaults(func=cmd_organize)

    # mega-batch
    from cli.cmd_mega_batch import (
        cmd_mega_batch, cmd_mega_batch_status,
        cmd_mega_batch_retry, cmd_mega_batch_reset,
    )

    p_mega = subparsers.add_parser(
        'mega-batch',
        help='Incremental batch processing with resume (500K+ PDFs)'
    )
    mega_sub = p_mega.add_subparsers(dest='mega_cmd')

    # mega-batch run
    p_mega_run = mega_sub.add_parser('run', help='Start/resume processing')
    p_mega_run.add_argument('source_dir', help='Root directory with PDF folders')
    p_mega_run.add_argument(
        '--colab-url', '-u',
        help='Colab LLM URL(s), comma-separated for multiple servers'
    )
    p_mega_run.add_argument(
        '--folder',
        help='Process only this specific folder'
    )
    p_mega_run.add_argument(
        '--cache-only', action='store_true',
        help='Use only template cache, no LLM calls'
    )
    p_mega_run.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be done without processing'
    )
    p_mega_run.set_defaults(func=cmd_mega_batch)

    # mega-batch status
    p_mega_status = mega_sub.add_parser('status', help='Show processing progress')
    p_mega_status.set_defaults(func=cmd_mega_batch_status)

    # mega-batch retry
    p_mega_retry = mega_sub.add_parser('retry', help='Retry failed/pending_llm files')
    p_mega_retry.add_argument('source_dir', help='Root directory')
    p_mega_retry.add_argument('--colab-url', '-u', help='Colab LLM URL(s)')
    p_mega_retry.add_argument('--folder', help='Retry only this folder')
    p_mega_retry.set_defaults(func=cmd_mega_batch_retry)

    # mega-batch reset
    p_mega_reset = mega_sub.add_parser('reset', help='Reset progress for a folder')
    p_mega_reset.add_argument('source_dir', nargs='?', help='Root directory')
    p_mega_reset.add_argument('--folder', help='Reset only this folder')
    p_mega_reset.add_argument(
        '--confirm', action='store_true',
        help='Confirm reset (required)'
    )
    p_mega_reset.set_defaults(func=cmd_mega_batch_reset)

    # status
    p_status = subparsers.add_parser('status', help='Show system status')
    p_status.set_defaults(func=cmd_status)

    # servers
    p_servers = subparsers.add_parser('servers', help='Manage LLM servers')
    servers_sub = p_servers.add_subparsers(dest='servers_cmd')

    p_wake = servers_sub.add_parser('wake', help='Wake servers via WoL')
    p_wake.set_defaults(func=cmd_servers_wake)

    p_sleep = servers_sub.add_parser('sleep', help='Send servers to sleep')
    p_sleep.set_defaults(func=cmd_servers_sleep)

    p_srv_status = servers_sub.add_parser('status', help='Check server status')
    p_srv_status.set_defaults(func=cmd_servers_status)

    # cache
    p_cache = subparsers.add_parser('cache', help='Manage template cache')
    cache_sub = p_cache.add_subparsers(dest='cache_cmd')

    p_stats = cache_sub.add_parser('stats', help='Cache statistics')
    p_stats.set_defaults(func=cmd_cache_stats)

    p_clear = cache_sub.add_parser('clear', help='Clear old templates')
    p_clear.add_argument('--older-than', type=int, default=90, help='Days (default: 90)')
    p_clear.set_defaults(func=cmd_cache_clear)

    p_export = cache_sub.add_parser('export', help='Export templates to JSON')
    p_export.add_argument('output', help='Output JSON file')
    p_export.set_defaults(func=cmd_cache_export)

    p_import = cache_sub.add_parser('import', help='Import templates from JSON')
    p_import.add_argument('input', help='Input JSON file')
    p_import.set_defaults(func=cmd_cache_import)

    # Shorthand: mega-batch <path> -> mega-batch run <path>
    argv = sys.argv[1:]
    mega_subcmds = {'run', 'status', 'retry', 'reset', '-h', '--help'}
    if (len(argv) >= 2 and argv[0] == 'mega-batch'
            and argv[1] not in mega_subcmds):
        argv = ['mega-batch', 'run'] + argv[1:]

    # Parse args
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if hasattr(args, 'func'):
        return args.func(args)

    # Подкоманды servers и cache
    if args.command == 'servers' and not args.servers_cmd:
        p_servers.print_help()
        return 1

    if args.command == 'cache' and not args.cache_cmd:
        p_cache.print_help()
        return 1

    if args.command == 'mega-batch' and not args.mega_cmd:
        p_mega.print_help()
        return 1

    return 0


def cli():
    """Entry point для console_scripts."""
    sys.exit(main())


if __name__ == '__main__':
    cli()
