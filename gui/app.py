"""
Tkinter GUI для Invoice LLM.

Три вкладки:
  1. Mega-Batch — запуск обработки, прогресс, логи
  2. Cache — статистика шаблонов
  3. LLM Servers — статус серверов
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path

# Добавляем корень проекта в path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Logging -> Queue bridge
# ---------------------------------------------------------------------------

class QueueHandler(logging.Handler):
    """Перенаправляет logging записи в queue.Queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> dict:
    """Загружает config.yaml."""
    import yaml
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class InvoiceLLMApp:
    """Главное окно приложения."""

    PROGRESS_POLL_MS = 2000  # обновление прогресс-бара
    LOG_POLL_MS = 100        # чтение логов из queue

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Invoice LLM")
        self.root.geometry("900x660")
        self.root.minsize(750, 500)

        self.config = _load_config()

        # --- state ---
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_queue: queue.Queue = queue.Queue()
        self._processor = None  # MegaBatchProcessor (для hot-add серверов)

        # --- logging bridge ---
        self._setup_logging()

        # --- notebook (tabs) ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_tab_mega_batch()
        self._build_tab_cache()
        self._build_tab_servers()

        # --- periodic polling ---
        self._poll_logs()
        self._poll_progress()

    # -----------------------------------------------------------------------
    # Logging setup
    # -----------------------------------------------------------------------

    def _setup_logging(self):
        handler = QueueHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    # -----------------------------------------------------------------------
    # Tab 1: Mega-Batch
    # -----------------------------------------------------------------------

    def _build_tab_mega_batch(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Mega-Batch  ")

        # -- Top controls frame --
        ctl = ttk.LabelFrame(tab, text="Settings")
        ctl.pack(fill=tk.X, padx=6, pady=(6, 3))

        # Row 0 — folder
        ttk.Label(ctl, text="Source folder:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.var_source = tk.StringVar(value=r"W:\baza\bill_pdf")
        ttk.Entry(ctl, textvariable=self.var_source, width=60).grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=4)
        ttk.Button(ctl, text="Browse...", command=self._browse_folder).grid(row=0, column=4, padx=4)

        # Row 1-3 — Colab URLs
        self.var_colab = [tk.StringVar() for _ in range(3)]
        for i in range(3):
            ttk.Label(ctl, text=f"Colab {i+1}:").grid(row=1+i, column=0, sticky=tk.W, padx=4, pady=2)
            ttk.Entry(ctl, textvariable=self.var_colab[i], width=72).grid(row=1+i, column=1, columnspan=4, sticky=tk.EW, padx=4)

        # Row 4 — options
        self.var_cache_only = tk.BooleanVar()
        self.var_dry_run = tk.BooleanVar()
        self.var_target_folder = tk.StringVar()

        opts = ttk.Frame(ctl)
        opts.grid(row=4, column=0, columnspan=5, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(opts, text="Cache Only", variable=self.var_cache_only).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(opts, text="Dry Run", variable=self.var_dry_run).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(opts, text="Folder:").pack(side=tk.LEFT)
        ttk.Entry(opts, textvariable=self.var_target_folder, width=24).pack(side=tk.LEFT, padx=4)

        ctl.columnconfigure(1, weight=1)

        # -- Buttons --
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=6, pady=3)

        self.btn_start = ttk.Button(btn_frame, text="Start", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_retry = ttk.Button(btn_frame, text="Retry Errors", command=self._on_retry)
        self.btn_retry.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_add_srv = ttk.Button(btn_frame, text="Add Servers", command=self._on_add_servers, state=tk.DISABLED)
        self.btn_add_srv.pack(side=tk.LEFT, padx=(0, 4))

        # -- Progress bar --
        prog_frame = ttk.Frame(tab)
        prog_frame.pack(fill=tk.X, padx=6, pady=3)

        self.progress_bar = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.lbl_progress = ttk.Label(prog_frame, text="Idle", width=20, anchor=tk.E)
        self.lbl_progress.pack(side=tk.RIGHT, padx=(6, 0))

        # -- Stats line --
        self.lbl_stats = ttk.Label(tab, text="", anchor=tk.W)
        self.lbl_stats.pack(fill=tk.X, padx=6)

        # -- Log window --
        self.log_text = scrolledtext.ScrolledText(tab, height=14, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 6))

    # -----------------------------------------------------------------------
    # Tab 2: Cache
    # -----------------------------------------------------------------------

    def _build_tab_cache(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Cache  ")

        # Stats frame
        sf = ttk.LabelFrame(tab, text="Template Cache Stats")
        sf.pack(fill=tk.X, padx=6, pady=(6, 3))

        labels = ["Templates:", "Countries:", "Companies:", "Hit rate:"]
        self.cache_labels: dict[str, ttk.Label] = {}
        for i, lbl_text in enumerate(labels):
            ttk.Label(sf, text=lbl_text).grid(row=0, column=i*2, sticky=tk.W, padx=(8, 2), pady=6)
            val = ttk.Label(sf, text="-", width=10, font=("Consolas", 10, "bold"))
            val.grid(row=0, column=i*2+1, sticky=tk.W, padx=(0, 16), pady=6)
            key = lbl_text.rstrip(":").lower().replace(" ", "_")
            self.cache_labels[key] = val

        # Buttons
        bf = ttk.Frame(tab)
        bf.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(bf, text="Refresh", command=self._refresh_cache_stats).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bf, text="Export...", command=self._cache_export).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bf, text="Import...", command=self._cache_import).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bf, text="Clear Old (90d)", command=self._cache_clear_old).pack(side=tk.LEFT, padx=(0, 4))

        # Treeview: top companies
        tf = ttk.LabelFrame(tab, text="Top Companies by Templates")
        tf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 6))

        cols = ("company", "country", "doc_category", "doc_type", "hits", "confidence")
        self.cache_tree = ttk.Treeview(tf, columns=cols, show="headings", height=12)
        for c, w, a in [("company", 200, tk.W), ("country", 90, tk.W),
                         ("doc_category", 80, tk.W), ("doc_type", 100, tk.W),
                         ("hits", 60, tk.E), ("confidence", 80, tk.E)]:
            self.cache_tree.heading(c, text=c.replace("_", " ").title())
            self.cache_tree.column(c, width=w, anchor=a)

        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.cache_tree.yview)
        self.cache_tree.configure(yscrollcommand=sb.set)
        self.cache_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # auto-load
        self.root.after(500, self._refresh_cache_stats)

    # -----------------------------------------------------------------------
    # Tab 3: LLM Servers
    # -----------------------------------------------------------------------

    def _build_tab_servers(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  LLM Servers  ")

        # Server list
        sf = ttk.LabelFrame(tab, text="Servers")
        sf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 3))

        cols = ("name", "url", "status")
        self.srv_tree = ttk.Treeview(sf, columns=cols, show="headings", height=8)
        for c, w in [("name", 140), ("url", 480), ("status", 80)]:
            self.srv_tree.heading(c, text=c.title())
            self.srv_tree.column(c, width=w)
        self.srv_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._populate_server_list()

        # Buttons
        bf = ttk.Frame(tab)
        bf.pack(fill=tk.X, padx=6, pady=(3, 6))
        ttk.Button(bf, text="Check Health", command=self._check_server_health).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bf, text="Wake Servers (WoL)", command=self._wake_servers).pack(side=tk.LEFT, padx=(0, 4))

    # ===================================================================
    # Mega-Batch actions
    # ===================================================================

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.var_source.get())
        if folder:
            self.var_source.set(folder)

    def _collect_colab_urls(self) -> list[str]:
        urls = []
        for v in self.var_colab:
            u = v.get().strip()
            if u:
                urls.append(u)
        return urls

    def _on_start(self):
        source = self.var_source.get().strip()
        if not source:
            self._log("ERROR: Source folder is empty")
            return
        if not Path(source).exists():
            self._log(f"ERROR: Folder not found: {source}")
            return

        colab_urls = self._collect_colab_urls()
        cache_only = self.var_cache_only.get()
        dry_run = self.var_dry_run.get()
        target_folder = self.var_target_folder.get().strip() or None

        self._stop_event.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_retry.configure(state=tk.DISABLED)
        self.btn_add_srv.configure(state=tk.NORMAL)

        self._log(f"Starting mega-batch: {source}")
        if colab_urls:
            self._log(f"Colab URLs: {', '.join(colab_urls)}")

        self._worker_thread = threading.Thread(
            target=self._worker_run,
            args=(source, colab_urls, cache_only, dry_run, target_folder, False),
            daemon=True,
        )
        self._worker_thread.start()

    def _on_add_servers(self):
        """Добавляет новые Colab URL в работающий процессор на лету."""
        proc = self._processor
        if proc is None:
            self._log("No active processor")
            return

        new_urls = self._collect_colab_urls()
        if not new_urls:
            self._log("No URLs to add -- fill Colab fields first")
            return

        cluster = proc.classifier.llm_cluster
        added = 0
        for url in new_urls:
            url = url.strip().rstrip("/")
            if url not in cluster.endpoints:
                cluster.endpoints.append(url)
                added += 1

        if added > 0:
            # Force health check to discover new endpoints
            cluster._check_endpoints(force=True)
            available = len(cluster._available_endpoints)
            self._log(f"Added {added} server(s), available now: {available}/{len(cluster.endpoints)}")
        else:
            self._log("All URLs already in cluster")

    def _on_stop(self):
        self._log("Stop requested -- finishing current file...")
        self._stop_event.set()
        self.btn_stop.configure(state=tk.DISABLED)

    def _on_retry(self):
        source = self.var_source.get().strip()
        if not source:
            self._log("ERROR: Source folder is empty")
            return

        colab_urls = self._collect_colab_urls()
        target_folder = self.var_target_folder.get().strip() or None

        self._stop_event.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_retry.configure(state=tk.DISABLED)

        self._log("Retrying errors...")

        self._worker_thread = threading.Thread(
            target=self._worker_run,
            args=(source, colab_urls, False, False, target_folder, True),
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_run(
        self,
        source: str,
        colab_urls: list[str],
        cache_only: bool,
        dry_run: bool,
        target_folder: str | None,
        retry: bool,
    ):
        """Рабочий поток для mega-batch (или retry)."""
        try:
            from core.mega_batch import MegaBatchProcessor, ProgressDB

            if retry:
                db = ProgressDB()
                reset_count = db.reset_errors(target_folder)
                self._log(f"Reset {reset_count} files to pending")
                db.close()
                if reset_count == 0:
                    self._log("No files to retry")
                    return

            processor = MegaBatchProcessor(
                source_root=source,
                config_path="config.yaml",
                colab_urls=colab_urls,
                stop_event=self._stop_event,
            )
            self._processor = processor
            processor.run(
                target_folder=target_folder,
                dry_run=dry_run,
                cache_only=cache_only,
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Worker error: {e}", exc_info=True)
        finally:
            # Вернуть кнопки в main thread
            self.root.after(0, self._worker_finished)

    def _worker_finished(self):
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_retry.configure(state=tk.NORMAL)
        self.btn_add_srv.configure(state=tk.DISABLED)
        self._processor = None
        self._log("--- Processing finished ---")

    # ===================================================================
    # Cache actions
    # ===================================================================

    def _refresh_cache_stats(self):
        try:
            from core.template_cache import TemplateCache
            cache = TemplateCache(
                db_path=self.config.get("cache", {}).get("db_path", "data/templates/fingerprints.db"),
            )
            stats = cache.get_stats()

            self.cache_labels["templates"].configure(text=str(stats.total_templates))
            self.cache_labels["countries"].configure(text=str(stats.countries))
            self.cache_labels["companies"].configure(text=str(stats.companies))
            self.cache_labels["hit_rate"].configure(text=f"{stats.hit_rate_24h:.1%}")

            # Populate treeview
            self.cache_tree.delete(*self.cache_tree.get_children())
            conn = cache._get_conn()
            # doc_category may not exist yet in old DBs
            try:
                rows = conn.execute(
                    "SELECT company, country, doc_category, doc_type, hit_count, confidence "
                    "FROM templates ORDER BY hit_count DESC LIMIT 100"
                ).fetchall()
                has_category = True
            except Exception:
                rows = conn.execute(
                    "SELECT company, country, doc_type, hit_count, confidence "
                    "FROM templates ORDER BY hit_count DESC LIMIT 100"
                ).fetchall()
                has_category = False
            for r in rows:
                if has_category:
                    self.cache_tree.insert("", tk.END, values=(
                        r["company"] or "-",
                        r["country"] or "-",
                        r["doc_category"] or "-",
                        r["doc_type"] or "-",
                        r["hit_count"],
                        f"{r['confidence']:.2f}",
                    ))
                else:
                    self.cache_tree.insert("", tk.END, values=(
                        r["company"] or "-",
                        r["country"] or "-",
                        "-",
                        r["doc_type"] or "-",
                        r["hit_count"],
                        f"{r['confidence']:.2f}",
                    ))
        except Exception as e:
            self._log(f"Cache stats error: {e}")

    def _cache_export(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            from core.template_cache import TemplateCache
            cache = TemplateCache(
                db_path=self.config.get("cache", {}).get("db_path", "data/templates/fingerprints.db"),
            )
            cache.export_templates(path)
            self._log(f"Exported templates to {path}")
        except Exception as e:
            self._log(f"Export error: {e}")

    def _cache_import(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            from core.template_cache import TemplateCache
            cache = TemplateCache(
                db_path=self.config.get("cache", {}).get("db_path", "data/templates/fingerprints.db"),
            )
            cache.import_templates(path)
            self._log(f"Imported templates from {path}")
            self._refresh_cache_stats()
        except Exception as e:
            self._log(f"Import error: {e}")

    def _cache_clear_old(self):
        try:
            from core.template_cache import TemplateCache
            cache = TemplateCache(
                db_path=self.config.get("cache", {}).get("db_path", "data/templates/fingerprints.db"),
            )
            before = cache.get_stats().total_templates
            cache.clear_old(days=90)
            after = cache.get_stats().total_templates
            self._log(f"Cleared {before - after} templates older than 90 days (remaining: {after})")
            self._refresh_cache_stats()
        except Exception as e:
            self._log(f"Clear error: {e}")

    # ===================================================================
    # Server actions
    # ===================================================================

    def _populate_server_list(self):
        """Заполняет дерево серверов из config."""
        self.srv_tree.delete(*self.srv_tree.get_children())
        servers = self.config.get("servers", [])
        for srv in servers:
            name = srv.get("name", "?")
            if "url" in srv:
                url = srv["url"]
            else:
                host = srv.get("host", "?")
                ports = srv.get("ports", [8080])
                url = ", ".join(f"http://{host}:{p}" for p in ports)
            self.srv_tree.insert("", tk.END, values=(name, url, "?"), iid=name)

    def _check_server_health(self):
        """Проверяет /health для каждого сервера (в фоновом потоке)."""
        self._log("Checking server health...")
        threading.Thread(target=self._health_check_worker, daemon=True).start()

    def _health_check_worker(self):
        import requests
        servers = self.config.get("servers", [])

        # Собираем dead endpoints из кластера (если есть)
        dead_eps = set()
        proc = self._processor
        if proc:
            try:
                dead_eps = set(proc.classifier.llm_cluster._dead_endpoints)
            except Exception:
                pass

        for srv in servers:
            name = srv.get("name", "?")
            endpoints = []
            if "url" in srv:
                endpoints.append(srv["url"].rstrip("/"))
            else:
                host = srv.get("host", "localhost")
                ports = srv.get("ports", [8080])
                for p in ports:
                    endpoints.append(f"http://{host}:{p}")

            # Если endpoint в dead — сразу помечаем
            is_dead = any(ep in dead_eps for ep in endpoints)
            if is_dead:
                status = "DEAD"
            else:
                status = "DOWN"
                for ep in endpoints:
                    try:
                        is_tunnel = "trycloudflare.com" in ep or ep.startswith("https://")
                        timeout = 10.0 if is_tunnel else 3.0
                        r = requests.get(f"{ep}/health", timeout=timeout, headers={"Connection": "close"})
                        if r.status_code == 200:
                            status = "OK"
                            break
                    except Exception:
                        pass

            self.root.after(0, lambda n=name, s=status: self._update_server_status(n, s))

        self.root.after(0, lambda: self._log("Health check complete"))

    def _update_server_status(self, name: str, status: str):
        try:
            vals = self.srv_tree.item(name, "values")
            self.srv_tree.item(name, values=(vals[0], vals[1], status))
        except tk.TclError:
            pass

    def _wake_servers(self):
        try:
            from core.scheduler import get_scheduler
            scheduler = get_scheduler(self.config)
            woken = scheduler.wake_servers()
            if woken:
                self._log(f"WoL sent to: {', '.join(woken)}")
            else:
                self._log("No servers to wake (check config)")
        except Exception as e:
            self._log(f"WoL error: {e}")

    # ===================================================================
    # Polling: logs & progress
    # ===================================================================

    def _poll_logs(self):
        """Читает log queue и пишет в Text widget (каждые LOG_POLL_MS мс)."""
        batch = []
        try:
            while True:
                batch.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if batch:
            self.log_text.configure(state=tk.NORMAL)
            for msg in batch:
                self.log_text.insert(tk.END, msg + "\n")
            # auto-scroll
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.root.after(self.LOG_POLL_MS, self._poll_logs)

    def _poll_progress(self):
        """Читает ProgressDB и обновляет прогресс-бар (каждые PROGRESS_POLL_MS мс)."""
        try:
            db_path = self.config.get("mega_batch", {}).get("progress_db", "data/progress/progress.db")
            if Path(db_path).exists():
                import sqlite3
                conn = sqlite3.connect(str(db_path), timeout=5)
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT "
                    "  COUNT(*) as total, "
                    "  SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) as ok, "
                    "  SUM(CASE WHEN status IN ('pending','pending_llm') THEN 1 ELSE 0 END) as pending, "
                    "  SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors, "
                    "  SUM(CASE WHEN source='template_cache' THEN 1 ELSE 0 END) as cache_hits, "
                    "  SUM(CASE WHEN source='llm' THEN 1 ELSE 0 END) as llm_calls "
                    "FROM processed_files"
                ).fetchone()
                conn.close()

                total = row["total"] or 0
                ok = row["ok"] or 0
                pending = row["pending"] or 0
                errors = row["errors"] or 0
                cache_hits = row["cache_hits"] or 0
                llm_calls = row["llm_calls"] or 0

                if total > 0:
                    pct = ok / total * 100
                    self.progress_bar["maximum"] = total
                    self.progress_bar["value"] = ok
                    self.lbl_progress.configure(text=f"{ok:,}/{total:,} ({pct:.1f}%)")

                    done = cache_hits + llm_calls
                    cache_pct = (cache_hits / done * 100) if done > 0 else 0

                    # Dead server info
                    dead_str = ""
                    proc = self._processor
                    if proc:
                        try:
                            dead_n = proc.classifier.llm_cluster.dead_count
                            if dead_n > 0:
                                dead_str = f"  |  DEAD: {dead_n}"
                        except Exception:
                            pass

                    self.lbl_stats.configure(
                        text=f"Cache hit: {cache_pct:.0f}%  |  LLM: {llm_calls:,}  |  "
                             f"Errors: {errors:,}  |  Pending: {pending:,}{dead_str}"
                    )
        except Exception:
            pass

        self.root.after(self.PROGRESS_POLL_MS, self._poll_progress)

    # ===================================================================
    # Helpers
    # ===================================================================

    def _log(self, msg: str):
        """Добавляет сообщение прямо в лог-окно (из main thread)."""
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{ts}  {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    InvoiceLLMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
