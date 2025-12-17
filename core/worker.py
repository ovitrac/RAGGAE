"""
RAGGAE/core/worker.py

Background indexing worker with SQLite job queue persistence.

Provides:
- IndexingWorker: Background thread for batch document indexing
- JobQueue: SQLite-backed persistent job queue
- Progress tracking with callbacks
- Crash recovery (resume interrupted jobs)

Designed for processing thousands of files efficiently.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
License: MIT
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job Status Enum
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    """Job lifecycle states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileStatus(str, enum.Enum):
    """Individual file processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class JobProgress:
    """Real-time job progress."""
    job_id: str
    status: JobStatus
    files_total: int
    files_processed: int
    files_failed: int
    files_skipped: int
    chunks_indexed: int
    current_file: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0

    @property
    def progress_percent(self) -> float:
        if self.files_total == 0:
            return 0.0
        return round((self.files_processed + self.files_failed + self.files_skipped)
                     / self.files_total * 100, 1)

    @property
    def is_complete(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "files_total": self.files_total,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "chunks_indexed": self.chunks_indexed,
            "current_file": self.current_file,
            "progress_percent": self.progress_percent,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class IndexJobConfig:
    """Configuration for an indexing job."""
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    min_chars: int = 40
    max_workers: int = 4
    prefer_pandoc: bool = True
    batch_size: int = 100  # Embeddings batch size


# ---------------------------------------------------------------------------
# SQLite Job Queue
# ---------------------------------------------------------------------------

class JobQueue:
    """
    SQLite-backed persistent job queue.

    Supports:
    - Job creation with file list
    - Progress tracking
    - Crash recovery
    - Job history
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize job queue.

        Args:
            db_path: Path to SQLite database (default: RAGGAE/data/jobs.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "jobs.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    index_path TEXT NOT NULL,
                    config TEXT NOT NULL,
                    files_total INTEGER DEFAULT 0,
                    files_processed INTEGER DEFAULT 0,
                    files_failed INTEGER DEFAULT 0,
                    files_skipped INTEGER DEFAULT 0,
                    chunks_indexed INTEGER DEFAULT 0,
                    current_file TEXT DEFAULT '',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS job_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunks_count INTEGER DEFAULT 0,
                    error TEXT,
                    processed_at TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );

                CREATE INDEX IF NOT EXISTS idx_job_files_job_id ON job_files(job_id);
                CREATE INDEX IF NOT EXISTS idx_job_files_status ON job_files(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            """)
            conn.commit()
        finally:
            conn.close()

    def create_job(
        self,
        files: List[str],
        index_path: str,
        config: IndexJobConfig
    ) -> str:
        """
        Create a new indexing job.

        Args:
            files: List of file paths to index
            index_path: Output index path
            config: Job configuration

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO jobs (job_id, status, index_path, config, files_total, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (job_id, JobStatus.PENDING.value, index_path, json.dumps(asdict(config)),
                  len(files), now))

            # Insert files
            conn.executemany("""
                INSERT INTO job_files (job_id, file_path, status)
                VALUES (?, ?, ?)
            """, [(job_id, f, FileStatus.PENDING.value) for f in files])

            conn.commit()
            logger.info(f"Created job {job_id} with {len(files)} files")
            return job_id
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current job progress."""
        job = self.get_job(job_id)
        if not job:
            return None

        elapsed = 0.0
        if job["started_at"]:
            start = datetime.fromisoformat(job["started_at"])
            end = datetime.fromisoformat(job["completed_at"]) if job["completed_at"] \
                else datetime.now()
            elapsed = (end - start).total_seconds()

        return JobProgress(
            job_id=job_id,
            status=JobStatus(job["status"]),
            files_total=job["files_total"],
            files_processed=job["files_processed"],
            files_failed=job["files_failed"],
            files_skipped=job["files_skipped"],
            chunks_indexed=job["chunks_indexed"],
            current_file=job["current_file"] or "",
            error=job["error"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            elapsed_seconds=elapsed
        )

    def get_pending_files(self, job_id: str, limit: int = 100) -> List[str]:
        """Get pending files for a job."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT file_path FROM job_files
                WHERE job_id = ? AND status = ?
                LIMIT ?
            """, (job_id, FileStatus.PENDING.value, limit)).fetchall()
            return [r["file_path"] for r in rows]
        finally:
            conn.close()

    def update_job_status(self, job_id: str, status: JobStatus, error: str = None):
        """Update job status."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            if status == JobStatus.RUNNING:
                conn.execute("""
                    UPDATE jobs SET status = ?, started_at = ? WHERE job_id = ?
                """, (status.value, now, job_id))
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                conn.execute("""
                    UPDATE jobs SET status = ?, completed_at = ?, error = ? WHERE job_id = ?
                """, (status.value, now, error, job_id))
            else:
                conn.execute("UPDATE jobs SET status = ? WHERE job_id = ?",
                             (status.value, job_id))
            conn.commit()
        finally:
            conn.close()

    def update_job_progress(
        self,
        job_id: str,
        files_processed: int = None,
        files_failed: int = None,
        files_skipped: int = None,
        chunks_indexed: int = None,
        current_file: str = None
    ):
        """Update job progress counters."""
        conn = self._get_conn()
        try:
            updates = []
            params = []

            if files_processed is not None:
                updates.append("files_processed = ?")
                params.append(files_processed)
            if files_failed is not None:
                updates.append("files_failed = ?")
                params.append(files_failed)
            if files_skipped is not None:
                updates.append("files_skipped = ?")
                params.append(files_skipped)
            if chunks_indexed is not None:
                updates.append("chunks_indexed = ?")
                params.append(chunks_indexed)
            if current_file is not None:
                updates.append("current_file = ?")
                params.append(current_file)

            if updates:
                params.append(job_id)
                conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", params)
                conn.commit()
        finally:
            conn.close()

    def update_file_status(
        self,
        job_id: str,
        file_path: str,
        status: FileStatus,
        chunks_count: int = 0,
        error: str = None
    ):
        """Update individual file status."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            conn.execute("""
                UPDATE job_files
                SET status = ?, chunks_count = ?, error = ?, processed_at = ?
                WHERE job_id = ? AND file_path = ?
            """, (status.value, chunks_count, error, now, job_id, file_path))
            conn.commit()
        finally:
            conn.close()

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute("""
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (status.value, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_resumable_jobs(self) -> List[str]:
        """Get jobs that were interrupted and can be resumed."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT job_id FROM jobs WHERE status = ?
            """, (JobStatus.RUNNING.value,)).fetchall()
            return [r["job_id"] for r in rows]
        finally:
            conn.close()

    def cleanup_old_jobs(self, days: int = 30):
        """Remove completed jobs older than N days."""
        conn = self._get_conn()
        try:
            cutoff = datetime.now().isoformat()[:10]  # Simplified
            conn.execute("""
                DELETE FROM job_files WHERE job_id IN (
                    SELECT job_id FROM jobs
                    WHERE status IN (?, ?, ?)
                    AND completed_at < date(?, '-' || ? || ' days')
                )
            """, (JobStatus.COMPLETED.value, JobStatus.FAILED.value,
                  JobStatus.CANCELLED.value, cutoff, days))
            conn.execute("""
                DELETE FROM jobs
                WHERE status IN (?, ?, ?)
                AND completed_at < date(?, '-' || ? || ' days')
            """, (JobStatus.COMPLETED.value, JobStatus.FAILED.value,
                  JobStatus.CANCELLED.value, cutoff, days))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Indexing Worker
# ---------------------------------------------------------------------------

class IndexingWorker:
    """
    Background worker for batch document indexing.

    Features:
    - Multi-threaded file processing
    - Batched embedding computation
    - Progress callbacks
    - Graceful cancellation
    - Crash recovery via SQLite queue
    """

    def __init__(
        self,
        queue: JobQueue,
        on_progress: Optional[Callable[[JobProgress], None]] = None,
        on_complete: Optional[Callable[[str, JobProgress], None]] = None
    ):
        """
        Initialize worker.

        Args:
            queue: Job queue for persistence
            on_progress: Callback for progress updates
            on_complete: Callback when job completes
        """
        self.queue = queue
        self.on_progress = on_progress
        self.on_complete = on_complete

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_job_id: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if worker is currently processing."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_job(self) -> Optional[str]:
        """Get currently running job ID."""
        return self._current_job_id

    def start(self, job_id: str) -> bool:
        """
        Start processing a job.

        Args:
            job_id: Job ID to process

        Returns:
            True if started, False if already running
        """
        with self._lock:
            if self.is_running:
                logger.warning(f"Worker already running job {self._current_job_id}")
                return False

            self._stop_event.clear()
            self._current_job_id = job_id
            self._thread = threading.Thread(
                target=self._run,
                args=(job_id,),
                daemon=True,
                name=f"IndexWorker-{job_id}"
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 30.0) -> bool:
        """
        Request graceful stop.

        Args:
            timeout: Max seconds to wait

        Returns:
            True if stopped, False if timeout
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def _run(self, job_id: str):
        """Main worker loop."""
        try:
            self._process_job(job_id)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            self.queue.update_job_status(job_id, JobStatus.FAILED, str(e))
            if self.on_complete:
                progress = self.queue.get_job_progress(job_id)
                self.on_complete(job_id, progress)
        finally:
            with self._lock:
                self._current_job_id = None

    def _process_job(self, job_id: str):
        """Process a single job."""
        # Import here to avoid circular imports
        from RAGGAE.io.ingest import ingest_file
        from RAGGAE.core.embeddings import STBiEncoder
        from RAGGAE.core.index_faiss import FaissIndex

        # Get job info
        job = self.queue.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        config = IndexJobConfig(**json.loads(job["config"]))

        # Mark as running
        self.queue.update_job_status(job_id, JobStatus.RUNNING)

        # Initialize encoder
        prefix_mode = "e5" if config.e5 else None
        encoder = STBiEncoder(config.model, prefix_mode=prefix_mode)

        # Collect all blocks
        all_texts = []
        all_metas = []
        files_processed = 0
        files_failed = 0
        files_skipped = 0
        chunks_indexed = 0

        # Process files in batches
        while True:
            if self._stop_event.is_set():
                logger.info(f"Job {job_id} cancelled")
                self.queue.update_job_status(job_id, JobStatus.CANCELLED)
                return

            # Get next batch of files
            pending = self.queue.get_pending_files(job_id, limit=config.max_workers * 2)
            if not pending:
                break

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(
                        ingest_file, f, config.min_chars, config.prefer_pandoc
                    ): f for f in pending
                }

                for future in as_completed(futures):
                    if self._stop_event.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.queue.update_job_status(job_id, JobStatus.CANCELLED)
                        return

                    file_path = futures[future]
                    try:
                        result = future.result()

                        if result.status == "ok":
                            # Add blocks
                            for block in result.blocks:
                                all_texts.append(block.text)
                                all_metas.append(block.as_metadata())

                            files_processed += 1
                            chunks_indexed += result.num_blocks
                            self.queue.update_file_status(
                                job_id, file_path, FileStatus.COMPLETED,
                                result.num_blocks
                            )

                        elif result.status == "skipped":
                            files_skipped += 1
                            self.queue.update_file_status(
                                job_id, file_path, FileStatus.SKIPPED,
                                error=result.error
                            )

                        else:
                            files_failed += 1
                            self.queue.update_file_status(
                                job_id, file_path, FileStatus.FAILED,
                                error=result.error
                            )

                    except Exception as e:
                        files_failed += 1
                        self.queue.update_file_status(
                            job_id, file_path, FileStatus.FAILED, error=str(e)
                        )

                    # Update progress
                    self.queue.update_job_progress(
                        job_id,
                        files_processed=files_processed,
                        files_failed=files_failed,
                        files_skipped=files_skipped,
                        chunks_indexed=chunks_indexed,
                        current_file=Path(file_path).name
                    )

                    if self.on_progress:
                        progress = self.queue.get_job_progress(job_id)
                        self.on_progress(progress)

        # Build and save index
        if all_texts:
            logger.info(f"Building index with {len(all_texts)} chunks...")

            # Batch embed
            embeddings = encoder.embed_texts(all_texts).astype("float32")

            # Create and save index
            idx = FaissIndex(dim=embeddings.shape[1])
            idx.add(embeddings, all_texts, all_metas)
            idx.save(config.index_path)

            logger.info(f"Index saved to {config.index_path}")

        # Mark complete
        self.queue.update_job_status(job_id, JobStatus.COMPLETED)

        if self.on_complete:
            progress = self.queue.get_job_progress(job_id)
            self.on_complete(job_id, progress)


# ---------------------------------------------------------------------------
# Global Worker Registry
# ---------------------------------------------------------------------------

_workers: Dict[str, IndexingWorker] = {}
_global_queue: Optional[JobQueue] = None


def get_queue() -> JobQueue:
    """Get or create global job queue."""
    global _global_queue
    if _global_queue is None:
        _global_queue = JobQueue()
    return _global_queue


def get_worker(
    on_progress: Optional[Callable[[JobProgress], None]] = None,
    on_complete: Optional[Callable[[str, JobProgress], None]] = None
) -> IndexingWorker:
    """
    Get or create global worker.

    Args:
        on_progress: Progress callback
        on_complete: Completion callback

    Returns:
        IndexingWorker instance
    """
    queue = get_queue()
    worker_id = "default"

    if worker_id not in _workers:
        _workers[worker_id] = IndexingWorker(queue, on_progress, on_complete)

    return _workers[worker_id]


def create_indexing_job(
    files: List[str],
    index_path: str,
    config: Optional[IndexJobConfig] = None,
    start_immediately: bool = True
) -> Tuple[str, Optional[IndexingWorker]]:
    """
    Create and optionally start an indexing job.

    Args:
        files: Files to index
        index_path: Output index path
        config: Job configuration (uses defaults if None)
        start_immediately: Start processing right away

    Returns:
        (job_id, worker) tuple
    """
    queue = get_queue()

    if config is None:
        config = IndexJobConfig(index_path=index_path)
    else:
        config.index_path = index_path

    job_id = queue.create_job(files, index_path, config)

    worker = None
    if start_immediately:
        worker = get_worker()
        worker.start(job_id)

    return job_id, worker


def get_job_progress(job_id: str) -> Optional[JobProgress]:
    """Get progress for a job."""
    return get_queue().get_job_progress(job_id)


def cancel_job(job_id: str) -> bool:
    """
    Cancel a running job.

    Returns:
        True if cancelled, False if not found/not running
    """
    queue = get_queue()
    job = queue.get_job(job_id)

    if not job:
        return False

    if job["status"] == JobStatus.RUNNING.value:
        worker = get_worker()
        if worker.current_job == job_id:
            worker.stop()
            queue.update_job_status(job_id, JobStatus.CANCELLED)
            return True

    return False


def list_jobs(status: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """List jobs with optional status filter."""
    queue = get_queue()
    status_enum = JobStatus(status) if status else None
    return queue.list_jobs(status_enum, limit)
