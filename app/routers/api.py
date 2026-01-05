from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Body, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from xml.dom import minidom
import time

from ..core.agent_runner import AgentRunnerError, run_agent as run_agent_via_responses
from ..core.comparison_legacy import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    build_agent_diff,
    build_html_diff,
    delete_agent_file,
    describe_library,
    list_agents_files,
    normalize_agent_collection,
    read_agent_file,
    simulate_typescript_processing,
    write_agent_file,
)
from ..core.main_processor import AltoProcessor, DEFAULT_API_BASES
from ..core.export_jobs import ExportJobParams, ExportJobState, get_export_manager
from ..services.export_builder import run_export_job

router = APIRouter()


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


class ExportRequest(BaseModel):
    source: str = Field(..., description="algorithmic|llm|ocr")
    export_format: str = Field(..., alias="format", description="html|txt|md")
    range_mode: str = Field(..., alias="rangeMode")
    range_value: Optional[str] = Field(None, alias="rangeValue")
    book_uuid: str = Field(..., alias="bookUuid")
    book_title: Optional[str] = Field(None, alias="bookTitle")
    current_page_uuid: str = Field(..., alias="currentPageUuid")
    api_base: Optional[str] = Field(None, alias="apiBase")
    pages: list[dict[str, Any]]
    joiner: dict[str, Any] = Field(default_factory=dict)
    llm_agent: dict[str, Any] = Field(default_factory=dict, alias="llmAgent")
    ocr_agent: dict[str, Any] = Field(default_factory=dict, alias="ocrAgent")
    output_filename: Optional[str] = Field(None, alias="outputFilename")
    language_hint: str = Field("cs", alias="languageHint")
    strip_small_text: bool = Field(True, alias="stripSmallText")
    strip_note_text: bool = Field(True, alias="stripNoteText")


class DownloadRequest(BaseModel):
    uuid: str = Field(..., description="UUID knihy nebo stránky")
    export_format: Optional[str] = Field("txt", alias="format", description="html|txt|md|epub")
    range_value: Optional[str] = Field(None, alias="range")
    llm_agent: dict[str, Any] = Field(default_factory=dict, alias="llmAgent")
    drop_small: bool = Field(False, alias="dropSmall")
    output_filename: Optional[str] = Field(None, alias="outputName")
    api_base: Optional[str] = Field(None, alias="apiBase")
    language_hint: str = Field("cs", alias="languageHint")


def _build_export_params(request: ExportRequest) -> ExportJobParams:
    allowed_sources = {"algorithmic", "llm", "ocr"}
    if request.source not in allowed_sources:
        raise HTTPException(status_code=400, detail="Nepodporovaný typ exportu.")
    allowed_formats = {"html", "txt", "md", "epub"}
    if request.export_format not in allowed_formats:
        raise HTTPException(status_code=400, detail="Nepodporovaný formát exportu.")
    allowed_modes = {"all", "current", "custom"}
    if request.range_mode not in allowed_modes:
        raise HTTPException(status_code=400, detail="Nepodporovaný rozsah stránek.")
    if not request.pages:
        raise HTTPException(status_code=400, detail="Export neobsahuje žádné stránky.")
    params = ExportJobParams(
        source=request.source,
        export_format=request.export_format,
        range_mode=request.range_mode,
        range_value=request.range_value,
        book_uuid=request.book_uuid,
        book_title=request.book_title,
        current_page_uuid=request.current_page_uuid,
        pages=request.pages,
        api_base=request.api_base,
        joiner=request.joiner,
        llm_agent=request.llm_agent,
        ocr_agent=request.ocr_agent,
        output_filename=request.output_filename,
        language_hint=request.language_hint,
        omit_small_text=request.strip_small_text,
        omit_note_text=request.strip_note_text,
    )
    return params


@router.post("/download")
def start_download(payload: DownloadRequest) -> JSONResponse:
    fmt = (payload.export_format or "txt").lower()
    if fmt not in {"html", "txt", "md", "epub"}:
        raise HTTPException(status_code=400, detail="Nepodporovaný formát exportu.")

    processor = AltoProcessor(api_base_url=payload.api_base)
    context = processor.get_book_context(payload.uuid)
    if not context:
        raise HTTPException(status_code=404, detail="Nepodařilo se načíst metadata pro zadané UUID.")

    pages = context.get("pages") or []
    if not pages:
        raise HTTPException(status_code=400, detail="Zadaná kniha neobsahuje žádné stránky.")

    normalized_uuid = processor._strip_uuid_prefix(payload.uuid)  # type: ignore[attr-defined]
    page_uuid = processor._strip_uuid_prefix(context.get("page_uuid") or "")  # type: ignore[attr-defined]
    is_page_uuid = bool(normalized_uuid) and normalized_uuid == page_uuid

    if payload.range_value:
        token = str(payload.range_value).strip().lower()
        if token in {"all", "*", "book", "kniha"}:
            range_mode = "all"
            range_value = None
        else:
            range_mode = "custom"
            range_value = payload.range_value
    else:
        range_mode = "current" if is_page_uuid else "all"
        range_value = None

    book_data = context.get("book") or {}
    params = ExportJobParams(
        source="llm" if payload.llm_agent else "algorithmic",
        export_format=fmt,
        range_mode=range_mode,
        range_value=range_value,
        book_uuid=context.get("book_uuid") or "",
        book_title=book_data.get("title"),
        current_page_uuid=page_uuid or normalized_uuid,
        pages=pages,
        api_base=context.get("api_base") or payload.api_base,
        joiner={"manual": True},
        llm_agent=payload.llm_agent or {},
        ocr_agent={},
        output_filename=payload.output_filename,
        language_hint=payload.language_hint or "cs",
        omit_small_text=bool(payload.drop_small),
        omit_note_text=True,
    )

    manager = get_export_manager()
    job = manager.create_job(params, run_export_job)
    return JSONResponse(job.to_dict())


@router.get("/process")
def process_page(uuid: str = Query(...), api_base: Optional[str] = Query(None)) -> Response:
    if not uuid:
        return _json_error("UUID je povinný", status_code=400)

    try:
        t_start = time.perf_counter()
        processor = AltoProcessor(api_base_url=api_base)
        processor.reset_request_stats()
        context = processor.get_book_context(uuid)
        try:
            processor.log_request_stats("after_get_book_context")
        except Exception:
            pass
        t_after_context = time.perf_counter()
        if not context:
            return _json_error("Nepodařilo se načíst metadata pro zadané UUID", status_code=404)

        page_uuid = context.get("page_uuid")
        if not page_uuid:
            return _json_error("Nepodařilo se určit konkrétní stránku pro zadané UUID", status_code=404)

        active_api_base = context.get("api_base") or processor.api_base_url
        library_info = describe_library(active_api_base)
        handle_base = library_info.get("handle_base") or ""
        book_uuid = context.get("book_uuid")

        alto_xml = processor.get_alto_data(page_uuid)
        t_after_alto = time.perf_counter()
        if not alto_xml:
            return _json_error("Nepodařilo se stáhnout ALTO data", status_code=502)

        pretty_alto = minidom.parseString(alto_xml).toprettyxml(indent="  ")
        python_result = processor.get_formatted_text(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        typescript_result = simulate_typescript_processing(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        try:
            processor.log_request_stats("after_process_page")
        except Exception:
            pass
        t_after_processing = time.perf_counter()

        pages = context.get("pages") or []
        current_index = context.get("current_index", -1)
        total_pages = len(pages)
        has_prev = current_index > 0
        has_next = 0 <= current_index < total_pages - 1
        prev_uuid = pages[current_index - 1]["uuid"] if has_prev else None
        next_uuid = pages[current_index + 1]["uuid"] if has_next else None

        book_data = context.get("book") or {}
        mods_metadata = context.get("mods") or []

        def clean(value: Optional[str]) -> str:
            if not value:
                return ""
            return " ".join(value.replace("\xa0", " ").split())

        page_summary = context.get("page") or {}
        page_item = context.get("page_item") or {}
        book_handle = f"{handle_base}/handle/uuid:{book_uuid}" if handle_base and book_uuid else ""
        page_handle = f"{handle_base}/handle/uuid:{page_uuid}" if handle_base and page_uuid else ""
        details = page_item.get("details") or {}

        page_info = {
            "uuid": page_uuid,
            "title": clean(page_summary.get("title") or page_item.get("title")),
            "pageNumber": clean(page_summary.get("pageNumber") or details.get("pagenumber")),
            "pageType": clean(page_summary.get("pageType") or details.get("type")),
            "pageSide": clean(
                page_summary.get("pageSide")
                or details.get("pageposition")
                or details.get("pagePosition")
                or details.get("pagerole")
            ),
            "index": current_index,
            "iiif": page_item.get("iiif"),
            "handle": page_handle,
            "library": library_info,
        }

        book_info = {
            "uuid": book_uuid,
            "title": clean(book_data.get("title")),
            "model": book_data.get("model"),
            "handle": book_handle,
            "mods": mods_metadata,
            "constants": context.get("book_constants") or {},
            "library": library_info,
        }

        navigation = {
            "hasPrev": has_prev,
            "hasNext": has_next,
            "prevUuid": prev_uuid,
            "nextUuid": next_uuid,
            "total": total_pages,
        }

        response_data = {
            "python": python_result,
            "typescript": typescript_result,
            "book": book_info,
            "pages": pages,
            "currentPage": page_info,
            "navigation": navigation,
            "alto_xml": pretty_alto,
            "library": library_info,
        }

        try:
            stats = processor.get_request_stats()
            total_requests = (
                stats.get("info_k7", 0)
                + stats.get("info_k5", 0)
                + stats.get("children_k7", 0)
                + stats.get("children_k5", 0)
                + stats.get("mods_k7", 0)
                + stats.get("mods_k5", 0)
                + stats.get("alto_k7", 0)
                + stats.get("alto_k5", 0)
                + stats.get("iiif_manifest", 0)
            )
            summary_parts = [
                f"total={total_requests}",
                f"timing_context={t_after_context - t_start:.3f}s",
                f"timing_alto={t_after_alto - t_after_context:.3f}s",
                f"timing_process={t_after_processing - t_after_alto:.3f}s",
                f"timing_total={t_after_processing - t_start:.3f}s",
                f"iiif_manifest={stats.get('iiif_manifest', 0)}",
                f"iiif_manifest_fail={stats.get('iiif_manifest_fail', 0)}",
                f"info_k7={stats.get('info_k7', 0)}",
                f"info_k5={stats.get('info_k5', 0)}",
                f"info_cache_hit={stats.get('info_cache_hit', 0)}",
                f"info_cache_miss={stats.get('info_cache_miss', 0)}",
                f"children_k7={stats.get('children_k7', 0)}",
                f"children_k5={stats.get('children_k5', 0)}",
                f"children_cache_hit={stats.get('children_cache_hit', 0)}",
                f"children_cache_miss={stats.get('children_cache_miss', 0)}",
                f"mods_k7={stats.get('mods_k7', 0)}",
                f"mods_k5={stats.get('mods_k5', 0)}",
                f"mods_cache_hit={stats.get('mods_cache_hit', 0)}",
                f"mods_cache_miss={stats.get('mods_cache_miss', 0)}",
                f"alto_k7={stats.get('alto_k7', 0)}",
                f"alto_k5={stats.get('alto_k5', 0)}",
                f"page_num_child={stats.get('page_number_from_child', 0)}",
                f"page_num_info={stats.get('page_number_from_page_info', 0)}",
                f"page_num_mods={stats.get('page_number_from_page_mods', 0)}",
                f"page_num_missing_child={stats.get('page_number_missing_child', 0)}",
                f"fallback_manifest_to_api={stats.get('fallback_manifest_to_api', 0)}",
            ]
            print("[kramerius-stats] " + " ".join(summary_parts))
        except Exception:
            pass

        return JSONResponse(response_data)
    except Exception as exc:
        return _json_error(str(exc), status_code=500)


@router.get("/preview")
def preview_image(uuid: str = Query(...), stream: str = Query("IMG_PREVIEW"), api_base: Optional[str] = Query(None)) -> Response:
    if not uuid:
        return _json_error("UUID je povinný", status_code=400)

    allowed_streams = {"IMG_THUMB", "IMG_PREVIEW", "IMG_FULL", "AUTO"}
    if stream not in allowed_streams:
        stream = "IMG_PREVIEW"

    candidate_streams = [stream]
    if stream == "AUTO":
        candidate_streams = ["IMG_FULL", "IMG_PREVIEW", "IMG_THUMB"]

    candidate_bases = list(dict.fromkeys([base for base in [api_base] + DEFAULT_API_BASES if base]))
    last_error = None

    for candidate in candidate_streams:
        for base in candidate_bases:
            version = AltoProcessor._detect_api_version(base)
            pid = AltoProcessor._format_pid_for_version(uuid, version)
            if not pid:
                continue
            if version == "k7":
                path = "image"
                if candidate == "IMG_THUMB":
                    path = "image/thumb"
                elif candidate == "IMG_PREVIEW":
                    path = "image/preview"
                upstream_url = f"{base}/items/{pid}/{path}"
            else:
                upstream_url = f"{base}/item/uuid:{pid}/streams/{candidate}"
            try:
                response = requests.get(upstream_url, timeout=20)
            except Exception as exc:
                last_error = str(exc)
                continue

            if response.status_code != 200 or not response.content:
                last_error = f"Nepodařilo se načíst náhled (status {response.status_code} pro {candidate} z {base})"
                response.close()
                continue

            content_type = response.headers.get("Content-Type", "image/jpeg")
            if "jp2" in content_type.lower():
                last_error = f"Stream {candidate} vrací nepodporovaný formát {content_type}"
                response.close()
                continue

            headers = {
                "Cache-Control": "no-store",
                "Content-Length": str(len(response.content)),
                "X-Preview-Stream": candidate,
            }
            data = response.content
            response.close()
            return Response(content=data, media_type=content_type, headers=headers)

    message = last_error or "Nepodařilo se načíst náhled."
    return _json_error(message, status_code=502)


@router.get("/agents")
@router.get("/agents/list")
def list_agents(collection: Optional[str] = Query(None)) -> Dict[str, Any]:
    items = list_agents_files(collection or "")
    return {"agents": items}


@router.get("/agents/get")
def get_agent(name: str = Query(...), collection: Optional[str] = Query(None)) -> Response:
    agent = read_agent_file(name, collection)
    if agent is None:
        return _json_error("not_found", status_code=404)
    return JSONResponse(agent)


@router.post("/diff")
def diff(payload: Dict[str, Any] = Body(...)) -> Response:
    python_html = str(payload.get("python") or payload.get("python_html") or "")
    ts_html = str(payload.get("typescript") or payload.get("typescript_html") or "")
    mode = payload.get("mode")
    if not isinstance(mode, str):
        mode = "word"
    try:
        diff_result = build_html_diff(python_html, ts_html, mode)
        return JSONResponse({"ok": True, "diff": diff_result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@router.post("/agents/diff")
def agent_diff(payload: Dict[str, Any] = Body(...)) -> Response:
    original_html = str(payload.get("original") or payload.get("original_html") or "")
    corrected_html = str(payload.get("corrected") or payload.get("corrected_html") or "")
    mode = payload.get("mode")
    if not isinstance(mode, str):
        mode = "word"
    try:
        diff_result = build_agent_diff(original_html, corrected_html, mode)
        return JSONResponse({"ok": True, "diff": diff_result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@router.post("/agents/save")
def save_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    collection = payload.get("collection")
    stored = write_agent_file(payload if isinstance(payload, dict) else {}, collection)
    if not stored:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid"})
    data = read_agent_file(stored, collection) or {}
    normalized_collection = normalize_agent_collection(collection)
    body = {
        "ok": True,
        "stored_name": stored,
        "collection": normalized_collection,
        "agent": data,
    }
    return JSONResponse(body)


@router.post("/exports")
def create_export_job(payload: ExportRequest) -> JSONResponse:
    params = _build_export_params(payload)
    manager = get_export_manager()
    job = manager.create_job(params, run_export_job)
    return JSONResponse(job.to_dict())


@router.get("/exports/{job_id}")
def get_export_job(job_id: str) -> JSONResponse:
    manager = get_export_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Export nenalezen.")
    return JSONResponse(job.to_dict())


@router.delete("/exports/{job_id}")
def abort_export_job(job_id: str) -> JSONResponse:
    manager = get_export_manager()
    job = manager.abort_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Export nenalezen.")
    return JSONResponse(job.to_dict())


@router.get("/exports/{job_id}/download")
def download_export_job(job_id: str) -> Response:
    manager = get_export_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Export nenalezen.")
    if job.state != ExportJobState.completed or not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=400, detail="Výsledek ještě není k dispozici.")
    filename = job.result_filename or f"export-{job.id}.{job.params.export_format}"
    return FileResponse(job.result_path, media_type="application/octet-stream", filename=filename)


@router.post("/agents/delete")
def delete_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    name = payload.get("name") if isinstance(payload, dict) else None
    collection = payload.get("collection") if isinstance(payload, dict) else None
    ok = delete_agent_file(name or "", collection)
    if ok:
        return JSONResponse({"ok": True})
    return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})


@router.post("/agents/run")
def run_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    request_payload = payload if isinstance(payload, dict) else {}
    collection = request_payload.get("collection")
    agent_name = request_payload.get("name")
    agent = read_agent_file(agent_name or "", collection)
    if not agent:
        return JSONResponse(status_code=404, content={"ok": False, "error": "agent_not_found"})

    model_override = str(request_payload.get("model_override") or request_payload.get("model") or "").strip()
    reasoning_override = str(request_payload.get("reasoning_effort") or "").strip().lower()
    snapshot = request_payload.get("agent_snapshot")

    if isinstance(snapshot, dict):
        agent_for_run = dict(agent)
        agent_for_run.update(snapshot)
    else:
        agent_for_run = dict(agent)

    agent_for_run.setdefault("name", agent_name or "")
    if model_override:
        agent_for_run["model"] = model_override
    if reasoning_override in {"low", "medium", "high"}:
        agent_for_run["reasoning_effort"] = reasoning_override

    try:
        result = run_agent_via_responses(agent_for_run, request_payload)
    except AgentRunnerError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - bubble unexpected errors
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    body = {
        "ok": True,
        "result": result,
        "auto_correct": bool(request_payload.get("auto_correct")),
    }
    return JSONResponse(body)
