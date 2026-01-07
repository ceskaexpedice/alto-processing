from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExportJobState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    aborted = "aborted"


@dataclass
class ExportJobParams:
    source: str
    export_format: str
    range_mode: str
    range_value: Optional[str]
    book_uuid: str
    book_title: Optional[str]
    current_page_uuid: str
    pages: List[Dict[str, Any]]
    api_base: Optional[str]
    ignore_images: bool = False
    authors: List[str] = field(default_factory=list)
    cover_uuid: Optional[str] = None
    joiner: Dict[str, Any] = field(default_factory=dict)
    llm_agent: Dict[str, Any] = field(default_factory=dict)
    ocr_agent: Dict[str, Any] = field(default_factory=dict)
    output_filename: Optional[str] = None
    language_hint: str = "cs"
    omit_small_text: bool = False
    omit_note_text: bool = False


@dataclass
class ExportJob:
    id: str
    params: ExportJobParams
    state: ExportJobState = ExportJobState.pending
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    processed_pages: int = 0
    total_pages: int = 0
    message: str = ""
    eta_seconds: Optional[float] = None
    result_path: Optional[str] = None
    result_filename: Optional[str] = None
    error: Optional[str] = None
    abort_requested: bool = False
    future: Optional[Future] = None

    def to_dict(self) -> Dict[str, Any]:
        percent = 0.0
        if self.total_pages:
            percent = min(100.0, max(0.0, (self.processed_pages / self.total_pages) * 100.0))
        progress = {
            "processed": self.processed_pages,
            "total": self.total_pages,
            "message": self.message,
            "percent": percent,
            "eta_seconds": self.eta_seconds,
        }
        return {
            "job_id": self.id,
            "state": self.state.value,
            "progress": progress,
            "error": self.error,
            "download_url": f"/exports/{self.id}/download" if self.state == ExportJobState.completed else None,
            "filename": self.result_filename,
        }

    def update_progress(self, processed: int, total: int, message: str) -> None:
        self.processed_pages = processed
        self.total_pages = total
        self.message = message
        now = time.time()
        self.updated_at = now
        if not self.started_at:
            self.started_at = now
        elapsed = max(0.001, now - self.started_at)
        per_page = elapsed / processed if processed else None
        if per_page and total > processed:
            remaining = (total - processed) * per_page
            self.eta_seconds = max(0.0, remaining)
        else:
            self.eta_seconds = None


class AbortRequested(Exception):
    """Raised inside a job runner when user cancels the export."""


class ExportJobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: Dict[str, ExportJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="export-worker")

    def _cleanup_file(self, job: ExportJob) -> None:
        if job.result_path and os.path.exists(job.result_path):
            try:
                os.remove(job.result_path)
            except OSError:
                pass
        job.result_path = None

    def create_job(self, params: ExportJobParams, runner) -> ExportJob:
        job_id = uuid.uuid4().hex
        job = ExportJob(id=job_id, params=params)
        with self._lock:
            self._jobs[job_id] = job
        future = self._executor.submit(self._run_job, job, runner)
        job.future = future
        return job

    def _run_job(self, job: ExportJob, runner) -> None:
        job.state = ExportJobState.running
        job.started_at = time.time()
        job.updated_at = job.started_at
        try:
            result_path, filename = runner(job)
            job.result_path = result_path
            job.result_filename = filename
            job.state = ExportJobState.completed
            job.updated_at = time.time()
        except AbortRequested:
            job.state = ExportJobState.aborted
            job.error = "Stažení bylo zrušeno."
            job.updated_at = time.time()
            self._cleanup_file(job)
        except Exception as exc:  # noqa: BLE001
            job.state = ExportJobState.failed
            job.error = str(exc)
            job.updated_at = time.time()
            self._cleanup_file(job)

    def get_job(self, job_id: str) -> Optional[ExportJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def abort_job(self, job_id: str) -> Optional[ExportJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.abort_requested = True
        future = job.future
        if future and future.done():
            return job
        return job

    def remove_job(self, job_id: str) -> Optional[ExportJob]:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job:
            self._cleanup_file(job)
        return job


_EXPORT_MANAGER = ExportJobManager()


def get_export_manager() -> ExportJobManager:
    return _EXPORT_MANAGER
