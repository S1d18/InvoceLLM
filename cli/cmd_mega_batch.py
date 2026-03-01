"""
CLI обработчики для mega-batch команды.

Подкоманды:
    mega-batch run <path>       — запуск/возобновление
    mega-batch status           — статистика
    mega-batch retry <path>     — повтор ошибок
    mega-batch reset <path>     — сброс прогресса
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup():
    """Подготовка окружения."""
    import sys
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def cmd_mega_batch(args):
    """Запуск/возобновление mega-batch обработки."""
    _setup()

    from core.mega_batch import MegaBatchProcessor

    # Собираем colab URLs
    colab_urls = []
    if hasattr(args, 'colab_url') and args.colab_url:
        colab_urls = [u.strip() for u in args.colab_url.split(',')]

    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_path}")
        return 1

    processor = MegaBatchProcessor(
        source_root=source_path,
        config_path=args.config,
        colab_urls=colab_urls,
    )

    processor.run(
        target_folder=getattr(args, 'folder', None),
        dry_run=getattr(args, 'dry_run', False),
        cache_only=getattr(args, 'cache_only', False),
    )

    return 0


def cmd_mega_batch_status(args):
    """Статистика mega-batch прогресса."""
    _setup()

    from core.mega_batch import ProgressDB

    db = ProgressDB()
    overall = db.get_stats()
    folders = db.get_folder_stats()
    runs = db.get_runs()

    print(f"\n{'='*70}")
    print("MEGA-BATCH STATUS")
    print(f"{'='*70}")

    # Общая статистика
    total = overall['total'] or 0
    ok = overall['ok'] or 0
    pending = overall['pending'] or 0
    pending_llm = overall['pending_llm'] or 0
    errors = overall['errors'] or 0
    cache_hits = overall['cache_hits'] or 0
    llm_calls = overall['llm_calls'] or 0

    pct = (ok / total * 100) if total > 0 else 0
    total_done = cache_hits + llm_calls
    cache_pct = (cache_hits / total_done * 100) if total_done > 0 else 0

    print(f"\nOverall:")
    print(f"  Total files:    {total:,}")
    print(f"  Processed:      {ok:,} ({pct:.1f}%)")
    print(f"  Pending:        {pending:,}")
    print(f"  Pending LLM:    {pending_llm:,}")
    print(f"  Errors:         {errors:,}")
    print(f"  Cache hits:     {cache_hits:,} ({cache_pct:.0f}%)")
    print(f"  LLM calls:      {llm_calls:,}")

    if overall['avg_time_ms']:
        print(f"  Avg time:       {overall['avg_time_ms']:.0f}ms")

    # По папкам
    if folders:
        print(f"\nBy Folder:")
        print(f"  {'Folder':<40} {'Done':>7} {'Total':>7} {'%':>5} {'Cache%':>6}")
        print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*5} {'-'*6}")

        for f in folders:
            f_total = f['total'] or 0
            f_ok = f['ok'] or 0
            f_pct = (f_ok / f_total * 100) if f_total > 0 else 0
            f_cache = f['cache_hits'] or 0
            f_llm = f['llm_calls'] or 0
            f_done = f_cache + f_llm
            f_cache_pct = (f_cache / f_done * 100) if f_done > 0 else 0

            name = f['folder'] or '?'
            if len(name) > 38:
                name = name[:35] + "..."

            print(f"  {name:<40} {f_ok:>7,} {f_total:>7,} {f_pct:>4.0f}% {f_cache_pct:>5.0f}%")

    # Последние запуски
    if runs:
        print(f"\nRecent Runs:")
        for run in runs[:5]:
            status_icon = "+" if run['status'] == 'finished' else "*"
            started = run['started_at'] or '?'
            finished = run['finished_at'] or 'running'
            print(f"  [{status_icon}] {started} -> {finished}  "
                  f"({run['processed_files'] or 0:,} files)")

    print()
    db.close()
    return 0


def cmd_mega_batch_retry(args):
    """Повтор ошибок и pending_llm."""
    _setup()

    from core.mega_batch import ProgressDB, MegaBatchProcessor

    db = ProgressDB()

    folder = getattr(args, 'folder', None)
    reset_count = db.reset_errors(folder)

    print(f"Reset {reset_count} files to pending")

    if reset_count == 0:
        print("No files to retry")
        db.close()
        return 0

    db.close()

    # Запускаем обработку
    if hasattr(args, 'source_dir') and args.source_dir:
        colab_urls = []
        if hasattr(args, 'colab_url') and args.colab_url:
            colab_urls = [u.strip() for u in args.colab_url.split(',')]

        processor = MegaBatchProcessor(
            source_root=args.source_dir,
            config_path=args.config,
            colab_urls=colab_urls,
        )
        processor.run(target_folder=folder)

    return 0


def cmd_mega_batch_reset(args):
    """Сброс прогресса для папки."""
    _setup()

    from core.mega_batch import ProgressDB

    if not getattr(args, 'confirm', False):
        print("Error: Use --confirm to confirm reset (irreversible)")
        return 1

    db = ProgressDB()

    folder = getattr(args, 'folder', None)
    if folder:
        count = db.reset_folder(folder)
        print(f"Reset {count} files for folder '{folder}'")
    else:
        # Сброс всего
        stats = db.get_stats()
        total = stats['total'] or 0

        db.conn.execute("DELETE FROM processed_files")
        db.conn.execute("DELETE FROM batch_runs")
        db.conn.commit()

        print(f"Reset all progress ({total:,} files)")

    db.close()
    return 0
