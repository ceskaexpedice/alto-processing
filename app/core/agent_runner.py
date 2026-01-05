#!/usr/bin/env python3
"""Helpers for executing stored agents via the OpenRouter/OpenAI Responses API."""

from __future__ import annotations

import base64
import copy
import inspect
import io
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:  # Pillow is optional but recommended for scan resizing
    from PIL import Image
except ImportError:  # pragma: no cover - fallback when Pillow missing
    Image = None  # type: ignore

try:
    from openai import OpenAI, BadRequestError
except ImportError as exc:  # pragma: no cover
    OpenAI = None  # type: ignore
    BadRequestError = None  # type: ignore
    _import_error = exc  # type: ignore
else:
    _import_error = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from openai.types.responses import Response  # type: ignore
else:  # pragma: no cover
    Response = Any  # type: ignore


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = ROOT_DIR / "config" / "models.json"
PROVIDER_REGISTRY_PATH = ROOT_DIR / "config" / "providers.json"


def _load_model_registry() -> Dict[str, Any]:
    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise AgentRunnerError(f"Soubor s modely je neplatný JSON: {MODEL_REGISTRY_PATH}") from exc


def _load_provider_registry() -> Dict[str, Any]:
    try:
        with PROVIDER_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise AgentRunnerError(f"Soubor s providery je neplatný JSON: {PROVIDER_REGISTRY_PATH}") from exc


MODEL_REGISTRY = _load_model_registry()
PROVIDER_REGISTRY = _load_provider_registry()
MODEL_DEFINITION_MAP: Dict[str, Dict[str, Any]] = {}
MODEL_CAPABILITY_MAP: Dict[str, Dict[str, bool]] = {}
MODEL_DEFAULTS_MAP: Dict[str, Dict[str, Any]] = {}
MODEL_UPSTREAM_MAP: Dict[str, str] = {}
PROVIDER_DEFINITION_MAP: Dict[str, Dict[str, Any]] = {}
ENABLE_RESPONSE_FORMAT = False  # sleeper feature – enable once OpenRouter enforces response_format

DEFAULT_API_BASES = [
    "https://api.kramerius.mzk.cz/search/api/client/v7.0",
    "https://kramerius.mzk.cz/search/api/v5.0",
    "https://kramerius5.nkp.cz/search/api/v5.0",
]
READER_DEFAULT_STREAM = "IMG_FULL"
READER_STREAM_FALLBACK = ("IMG_FULL", "IMG_PREVIEW", "IMG_THUMB")
READER_MAX_IMAGE_DIMENSION = 2000

if ENABLE_RESPONSE_FORMAT:
    SUPPORTED_RESPONSE_FORMAT_TYPES = {"json_object", "json_schema"}
    _BLOCK_ID_PATTERN = re.compile(r"^b\d+$")

    class _DictResponse:
        """Minimal facade mimicking the Responses API object."""

        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload
            self.id = payload.get("id")
            self.model = payload.get("model")
            self.output = payload.get("output") or payload.get("outputs")

        def model_dump(self) -> Dict[str, Any]:
            return self._payload

        @property
        def output_text(self) -> str:
            data = self._payload.get("output") or self._payload.get("outputs")
            if isinstance(data, list):
                chunks = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content") or []
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                            text_value = block.get("text")
                            if isinstance(text_value, dict):
                                inner = text_value.get("value") or text_value.get("text")
                                if inner:
                                    chunks.append(str(inner))
                            elif isinstance(text_value, str):
                                chunks.append(text_value)
                return "\n".join(chunks)
            return ""

    def _load_json_if_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            return None
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            return trimmed

    def _normalize_response_format(value: Any) -> Optional[Dict[str, Any]]:
        candidate = _load_json_if_string(value)
        if candidate is None:
            return None
        if isinstance(candidate, dict):
            type_value = candidate.get("type")
            if type_value is None and "json_schema" in candidate:
                type_value = "json_schema"
            if type_value is None and "schema" in candidate:
                type_value = "json_schema"
            if isinstance(type_value, str):
                normalized_type = type_value.strip().lower()
            else:
                normalized_type = ""
            if normalized_type == "json_object":
                return {"type": "json_object"}
            if normalized_type == "json_schema":
                schema_payload = candidate.get("json_schema")
                if schema_payload is None:
                    schema_payload = {
                        key: candidate.get(key)
                        for key in ("name", "schema", "strict")
                        if key in candidate
                    }
                schema_payload = _load_json_if_string(schema_payload)
                if isinstance(schema_payload, dict):
                    name = schema_payload.get("name")
                    schema = schema_payload.get("schema")
                    if isinstance(schema, str):
                        schema = _load_json_if_string(schema)
                    if isinstance(name, str) and isinstance(schema, dict):
                        sanitized: Dict[str, Any] = {"name": name, "schema": schema}
                        if isinstance(schema_payload.get("strict"), bool):
                            sanitized["strict"] = schema_payload["strict"]
                        for key, extra_value in schema_payload.items():
                            if key in {"name", "schema", "strict"}:
                                continue
                            sanitized[key] = extra_value
                        return {"type": "json_schema", "json_schema": sanitized}
        if isinstance(candidate, str):
            lowered = candidate.strip().lower()
            if lowered in SUPPORTED_RESPONSE_FORMAT_TYPES:
                return {"type": lowered}
        return None

    def _client_supports_response_format(client: OpenAI) -> bool:
        try:
            signature = inspect.signature(client.responses.create)
        except (AttributeError, ValueError, TypeError):
            return False
        return "response_format" in signature.parameters

    def _perform_responses_request(
        client: OpenAI,
        request_kwargs: Dict[str, Any],
        native_response_format_support: bool,
        provider_config: Dict[str, Any],
        api_key: str,
    ) -> Response:
        if native_response_format_support:
            return client.responses.create(**request_kwargs)

        payload = copy.deepcopy(request_kwargs)
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        api_base = str(provider_config.get("api_base") or "").rstrip("/")
        if not api_base:
            raise AgentRunnerError("Provider nemá nastavené 'api_base' pro Responses API.")

        url = api_base + "/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        default_headers = provider_config.get("default_headers")
        if isinstance(default_headers, dict):
            headers.update(default_headers)

        response = requests.post(url, headers=headers, json=payload, timeout=90)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise AgentRunnerError(
                f"OpenRouter responses API vrátil chybu {response.status_code}: {response.text}"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise AgentRunnerError("OpenRouter vrátil neplatný JSON.") from exc
        return _DictResponse(payload)

    def _validate_text_block_corrector(payload: Any) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "odpověď není JSON objekt"
        if "changes" not in payload:
            return False, "chybí pole 'changes'"
        changes = payload.get("changes")
        if not isinstance(changes, list):
            return False, "pole 'changes' není list"
        for index, item in enumerate(changes):
            if not isinstance(item, dict):
                return False, f"položka changes[{index}] není objekt"
            if set(item.keys()) - {"id", "text"}:
                return False, f"položka changes[{index}] obsahuje nepovolené klíče"
            if "id" not in item or "text" not in item:
                return False, f"položka changes[{index}] postrádá 'id' nebo 'text'"
            if not isinstance(item["id"], str) or not _BLOCK_ID_PATTERN.match(item["id"]):
                return False, f"changes[{index}].id není ve formátu 'b<number>'"
            if not isinstance(item["text"], str):
                return False, f"changes[{index}].text není řetězec"
        return True, ""
else:
    def _normalize_response_format(value: Any) -> Optional[Dict[str, Any]]:
        return None

    def _client_supports_response_format(_: Any) -> bool:
        return False

    def _perform_responses_request(
        client: OpenAI,
        request_kwargs: Dict[str, Any],
        native_response_format_support: bool,
        _: Optional[Dict[str, Any]] = None,
        __: Optional[str] = None,
    ) -> Response:
        return client.responses.create(**request_kwargs)

    def _validate_text_block_corrector(_: Any) -> Tuple[bool, str]:
        return True, ""
_CLIENT_SUPPORTS_NATIVE_RESPONSE_FORMAT: Dict[str, bool] = {}


def _normalize_model_id(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _normalize_provider_id(value: Optional[str]) -> str:
    return (value or "").strip().lower()


for entry in MODEL_REGISTRY.get("models", []) or []:
    if not isinstance(entry, dict):
        continue
    raw_id = entry.get("id")
    normalized = _normalize_model_id(raw_id)
    if not normalized:
        continue
    MODEL_DEFINITION_MAP[normalized] = entry
    raw_caps = entry.get("capabilities") or {}
    caps = {
        "temperature": bool(raw_caps.get("temperature", True)),
        "top_p": bool(raw_caps.get("top_p", True)),
        "reasoning": bool(raw_caps.get("reasoning", False)),
    }
    if ENABLE_RESPONSE_FORMAT:
        caps["response_format"] = bool(raw_caps.get("response_format", True))
    MODEL_CAPABILITY_MAP[normalized] = caps
    raw_defaults = entry.get("defaults") or {}
    if isinstance(raw_defaults, dict):
        MODEL_DEFAULTS_MAP[normalized] = raw_defaults
    upstream_id = entry.get("upstream_id") or raw_id
    if isinstance(upstream_id, str):
        MODEL_UPSTREAM_MAP[normalized] = upstream_id

for entry in PROVIDER_REGISTRY.get("providers", []) or []:
    if not isinstance(entry, dict):
        continue
    normalized = _normalize_provider_id(entry.get("name"))
    if not normalized:
        continue
    PROVIDER_DEFINITION_MAP[normalized] = entry


def _get_provider_config(provider_name: str) -> Dict[str, Any]:
    normalized = _normalize_provider_id(provider_name)
    if not normalized or normalized not in PROVIDER_DEFINITION_MAP:
        raise AgentRunnerError(f"Nenalezen provider '{provider_name}' v konfiguraci.")
    return PROVIDER_DEFINITION_MAP[normalized]


def _require_api_key(provider_config: Dict[str, Any]) -> str:
    env_var = str(provider_config.get("api_key_env") or "").strip()
    if not env_var:
        raise AgentRunnerError("Provider nemá definovanou proměnnou s API klíčem (api_key_env).")
    api_key = os.getenv(env_var)
    if not api_key:
        raise AgentRunnerError(
            f"Chybí proměnná prostředí {env_var}. Uložte ji do .env nebo prostředí."
        )
    return api_key


def _model_supports_scan(model_id: str) -> bool:
    entry = MODEL_DEFINITION_MAP.get(_normalize_model_id(model_id))
    if not entry:
        return False
    if "supports_scan" in entry:
        return bool(entry["supports_scan"])
    return False


def _model_supports_text(model_id: str) -> bool:
    entry = MODEL_DEFINITION_MAP.get(_normalize_model_id(model_id))
    if not entry:
        return True
    if "supports_text" in entry:
        return bool(entry["supports_text"])
    return True


def _iter_reader_api_bases(override: Optional[str] = None):
    seen: set[str] = set()
    if override:
        normalized = override.rstrip("/")
        if normalized:
            seen.add(normalized)
            yield normalized
    for base in DEFAULT_API_BASES:
        normalized = base.rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def _fetch_scan_image_bytes(
    uuid: str,
    preferred_stream: str,
    api_base_override: Optional[str],
) -> Tuple[bytes, str, str]:
    if not uuid:
        raise AgentRunnerError("Reader agent vyžaduje 'scan_uuid'.")
    normalized_stream = (preferred_stream or READER_DEFAULT_STREAM).upper()
    if normalized_stream not in READER_STREAM_FALLBACK:
        normalized_stream = READER_DEFAULT_STREAM
    candidate_streams = []
    for stream in (normalized_stream,) + tuple(s for s in READER_STREAM_FALLBACK if s != normalized_stream):
        if stream not in candidate_streams:
            candidate_streams.append(stream)
    last_error = None
    for stream in candidate_streams:
        for base in _iter_reader_api_bases(api_base_override):
            upstream_url = f"{base}/item/uuid:{uuid}/streams/{stream}"
            try:
                response = requests.get(upstream_url, timeout=25)
            except requests.RequestException as exc:
                last_error = f"Chyba při stahování skenu: {exc}"
                continue
            if response.status_code != 200 or not response.content:
                last_error = f"Stream {stream} ({base}) vrátil {response.status_code}"
                response.close()
                continue
            content_type = response.headers.get("Content-Type", "image/jpeg")
            if "jp2" in content_type.lower():
                last_error = f"Stream {stream} vrací nepodporovaný formát {content_type}"
                response.close()
                continue
            payload = response.content
            response.close()
            return payload, content_type or "image/jpeg", stream
    raise AgentRunnerError(last_error or "Nepodařilo se stáhnout obrázek stránky pro reader agenta.")


def _downscale_reader_image(image_bytes: bytes, content_type: Optional[str]) -> Tuple[bytes, str]:
    media_type = (content_type or "image/jpeg").lower()
    if media_type == "image/jp2":
        media_type = "image/jpeg"
    if Image is None:
        return image_bytes, media_type or "image/jpeg"
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            max_dim = max(width, height)
            target_media_type = media_type or f"image/{(img.format or 'jpeg').lower()}"
            if max_dim <= READER_MAX_IMAGE_DIMENSION:
                if target_media_type == "image/jp2":
                    target_media_type = "image/jpeg"
                return image_bytes, target_media_type or "image/jpeg"
            scale = READER_MAX_IMAGE_DIMENSION / float(max_dim)
            new_width = max(64, int(width * scale))
            new_height = max(64, int(height * scale))
            resized = img.convert("RGB").resize((new_width, new_height), Image.LANCZOS)
            output = io.BytesIO()
            resized.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue(), "image/jpeg"
    except Exception as exc:
        print(f"[AgentDebug] Downscale obrázku selhal: {exc}")
    return image_bytes, media_type or "image/jpeg"


def _prepare_reader_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    scan_uuid = str(payload.get("scan_uuid") or payload.get("page_uuid") or "").strip()
    if not scan_uuid:
        raise AgentRunnerError("Reader agent vyžaduje 'scan_uuid'.")
    preferred_stream = str(payload.get("scan_stream") or READER_DEFAULT_STREAM).strip().upper()
    api_base_override = str(payload.get("api_base") or "").strip() or None
    image_bytes, content_type, used_stream = _fetch_scan_image_bytes(scan_uuid, preferred_stream, api_base_override)
    processed_bytes, media_type = _downscale_reader_image(image_bytes, content_type)
    image_b64 = base64.b64encode(processed_bytes).decode("ascii")
    language_hint = payload.get("language_hint") or DEFAULT_LANGUAGE_HINT
    page_number = payload.get("page_number") or ""
    const_parts = [f"Jazyková nápověda: {language_hint}"]
    if page_number:
        const_parts.append(f"Číslo strany: {page_number}")
    const_parts.append("")
    const_parts.extend([
        "Úkol: Přepiš text z přiloženého skenu do čistého HTML (stejné bloky jako vstupní ALTO).",
        "Používej pouze bloky <h1>-<h3>, <p>, <blockquote>, <small>, <div class=\"note\">.",
        "Nepřidávej vlastní poznámky ani Markdown.",
        "Pokud je obsah nečitelný, vrať pouze <p>[nečitelná strana]</p>.",
    ])
    instruction_text = "\n".join(const_parts).strip()
    document_payload = {
        "language_hint": language_hint,
        "page_uuid": scan_uuid,
        "page_number": page_number,
        "page_index": payload.get("page_index"),
        "book_uuid": payload.get("book_uuid") or "",
        "api_base": api_base_override or "",
        "image_stream": used_stream,
        "image_media_type": media_type,
        "source": "scan_reader",
    }
    data_url = f"data:{media_type or 'image/jpeg'};base64,{image_b64}"
    chat_content = [
        {"type": "text", "text": instruction_text.strip()},
        {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": "auto",
            },
        },
    ]
    return {
        "document_payload": document_payload,
        "chat_content": chat_content,
    }


def _extract_chat_output_text(response: Any) -> str:
    # Debug: inspect response shape to handle both object and dict forms
    try:
        print("[LLMDebug][_extract_chat_output_text] response type:", type(response))
    except Exception:
        pass
    content = None
    try:
        choice = response.choices[0]
        try:
            content = getattr(choice.message, "content", None)
            print("[LLMDebug][_extract_chat_output_text] content (attr) type:", type(content), "preview:", repr(str(content)))
        except Exception as e:  # pragma: no cover - defensive
            print("[LLMDebug][_extract_chat_output_text] error accessing choice.message.content:", e)
            content = None
    except (AttributeError, IndexError) as e:
        # Fallback: try dict-style access from model_dump
        print("[LLMDebug][_extract_chat_output_text] choice access failed:", e)
        try:
            resp_dict = response.model_dump() if hasattr(response, "model_dump") else getattr(response, "__dict__", {})
        except Exception as e2:
            resp_dict = {}
            print("[LLMDebug][_extract_chat_output_text] response.model_dump failed:", e2)
        choices = resp_dict.get("choices") or resp_dict.get("output") or []
        print("[LLMDebug][_extract_chat_output_text] choices (dict) type:", type(choices))
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            print("[LLMDebug][_extract_chat_output_text] first choice type:", type(ch0))
            if isinstance(ch0, dict):
                msg = ch0.get("message") or {}
                print("[LLMDebug][_extract_chat_output_text] first choice.message (dict) keys:", list(msg.keys()) if isinstance(msg, dict) else type(msg))
                if isinstance(msg, dict):
                    content = msg.get("content")
                    print("[LLMDebug][_extract_chat_output_text] content (dict) preview:", repr(str(content)))
    # Normalize content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    # Last resort: try to stringify
    try:
        return str(content or "")
    except Exception:
        return ""


def _run_reader_chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    chat_content: list,
    temperature: Optional[float],
    top_p: Optional[float],
    max_output_tokens: Optional[int],
) -> Any:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": chat_content},
    ]
    chat_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if max_output_tokens is not None:
        chat_kwargs["max_tokens"] = int(max_output_tokens)

    print("\n=== [AgentDebug] Chat Completion Request ===")
    print(f"Model: {model}")
    if "temperature" in chat_kwargs:
        print(f"Temperature: {chat_kwargs['temperature']}")
    if "top_p" in chat_kwargs:
        print(f"Top P: {chat_kwargs['top_p']}")
    if "max_tokens" in chat_kwargs:
        print(f"Max tokens: {chat_kwargs['max_tokens']}")
    for msg in messages:
        role = msg.get("role")
        for item in msg.get("content", []):
            item_type = item.get("type")
            if item_type == "text":
                print(f"[{role}] {item.get('text')}")
            elif item_type == "image_url":
                image_data = item.get("image_url") or {}
                url_value = image_data.get("url")
                descriptor = f"url length={len(url_value) if isinstance(url_value, str) else 0}"
                print(f"[{role}] <image_url {descriptor}>")
    print("=== [AgentDebug] End Chat Request ===\n")

    response = client.chat.completions.create(**chat_kwargs)
    return response


DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 1.0
DEFAULT_REASONING_EFFORT = "medium"
MIN_OUTPUT_TOKENS = 1
MAX_OUTPUT_TOKENS = 128000

DEFAULT_MODEL = (
    os.getenv("OPENAI_DEFAULT_MODEL")
    or os.getenv("OPENAI_MODEL")
    or MODEL_REGISTRY.get("default_model")
    or "openai/gpt-4o-mini"
)
DEFAULT_LANGUAGE_HINT = os.getenv("OPENAI_LANGUAGE_HINT") or "cs"

REASONING_EFFORT_VALUES = {"low", "medium", "high"}
FALLBACK_REASONING_PREFIXES = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4", "gpt-5", "o1", "o3", "o4")
# Backwards compatibility for modules importing REASONING_PREFIXES directly.
REASONING_PREFIXES = FALLBACK_REASONING_PREFIXES
SUPPORTED_RESPONSE_FORMAT_TYPES = {"json_object", "json_schema"}

_BLOCK_ID_PATTERN = re.compile(r"^b\d+$")


class _DictResponse:
    """Minimal facade mimicking the Responses API object."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.id = payload.get("id")
        self.model = payload.get("model")
        self.output = payload.get("output") or payload.get("outputs")

    def model_dump(self) -> Dict[str, Any]:
        return self._payload

    @property
    def output_text(self) -> str:
        data = self._payload.get("output") or self._payload.get("outputs")
        if isinstance(data, list):
            chunks = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or []
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                        text_value = block.get("text")
                        if isinstance(text_value, dict):
                            inner = text_value.get("value") or text_value.get("text")
                            if inner:
                                chunks.append(str(inner))
                        elif isinstance(text_value, str):
                            chunks.append(text_value)
            return "\n".join(chunks)
        return ""


def _validate_text_block_corrector(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "odpověď není JSON objekt"
    if "changes" not in payload:
        return False, "chybí pole 'changes'"
    changes = payload.get("changes")
    if not isinstance(changes, list):
        return False, "pole 'changes' není list"
    for index, item in enumerate(changes):
        if not isinstance(item, dict):
            return False, f"položka changes[{index}] není objekt"
        if set(item.keys()) - {"id", "text"}:
            return False, f"položka changes[{index}] obsahuje nepovolené klíče"
        if "id" not in item or "text" not in item:
            return False, f"položka changes[{index}] postrádá 'id' nebo 'text'"
        if not isinstance(item["id"], str) or not _BLOCK_ID_PATTERN.match(item["id"]):
            return False, f"changes[{index}].id není ve formátu 'b<number>'"
        if not isinstance(item["text"], str):
            return False, f"changes[{index}].text není řetězec"
    return True, ""

def _get_model_capabilities(model: str) -> Dict[str, bool]:
    normalized = _normalize_model_id(model)
    if normalized in MODEL_CAPABILITY_MAP:
        return MODEL_CAPABILITY_MAP[normalized]
    upstream = _normalize_model_id(MODEL_UPSTREAM_MAP.get(normalized))
    for prefix in FALLBACK_REASONING_PREFIXES:
        lowered = prefix.lower()
        if upstream == lowered or upstream.startswith(f"{lowered}-"):
            caps = {"temperature": False, "top_p": False, "reasoning": True}
            if ENABLE_RESPONSE_FORMAT:
                caps["response_format"] = True
            return caps
    caps = {"temperature": True, "top_p": True, "reasoning": False}
    if ENABLE_RESPONSE_FORMAT:
        caps["response_format"] = True
    return caps


def _normalize_reasoning_effort(value: Optional[Any]) -> str:
    if value is None:
        return DEFAULT_REASONING_EFFORT
    normalized = str(value).strip().lower()
    return normalized if normalized in REASONING_EFFORT_VALUES else DEFAULT_REASONING_EFFORT


def _get_upstream_model_id(model_id: str) -> str:
    normalized = _normalize_model_id(model_id)
    upstream = MODEL_UPSTREAM_MAP.get(normalized)
    return upstream or model_id


def _load_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return trimmed


def _normalize_response_format(value: Any) -> Optional[Dict[str, Any]]:
    candidate = _load_json_if_string(value)
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        type_value = candidate.get("type")
        if type_value is None and "json_schema" in candidate:
            type_value = "json_schema"
        if type_value is None and "schema" in candidate:
            type_value = "json_schema"
        if isinstance(type_value, str):
            normalized_type = type_value.strip().lower()
        else:
            normalized_type = ""
        if normalized_type == "json_object":
            return {"type": "json_object"}
        if normalized_type == "json_schema":
            schema_payload = candidate.get("json_schema")
            if schema_payload is None:
                schema_payload = {
                    key: candidate.get(key)
                    for key in ("name", "schema", "strict")
                    if key in candidate
                }
            schema_payload = _load_json_if_string(schema_payload)
            if isinstance(schema_payload, dict):
                name = schema_payload.get("name")
                schema = schema_payload.get("schema")
                if isinstance(schema, str):
                    schema = _load_json_if_string(schema)
                if isinstance(name, str) and isinstance(schema, dict):
                    sanitized: Dict[str, Any] = {"name": name, "schema": schema}
                    if isinstance(schema_payload.get("strict"), bool):
                        sanitized["strict"] = schema_payload["strict"]
                    # Preserve any other JSON-schema fields that are valid JSON objects.
                    for key, extra_value in schema_payload.items():
                        if key in {"name", "schema", "strict"}:
                            continue
                        sanitized[key] = extra_value
                    return {"type": "json_schema", "json_schema": sanitized}
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in SUPPORTED_RESPONSE_FORMAT_TYPES:
            return {"type": lowered}
    return None


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _extract_settings(agent: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    settings = agent.get("settings")
    defaults: Dict[str, Any] = {}
    per_model: Dict[str, Dict[str, Any]] = {}
    if isinstance(settings, dict):
        raw_defaults = settings.get("defaults")
        if isinstance(raw_defaults, dict):
            defaults = raw_defaults
        raw_per_model = settings.get("per_model")
        if isinstance(raw_per_model, dict):
            per_model = raw_per_model  # type: ignore[assignment]
    return defaults, per_model


def _get_effective_settings(agent: Dict[str, Any], model: str) -> Dict[str, Any]:
    defaults, per_model = _extract_settings(agent)
    model_settings = per_model.get(model) if isinstance(per_model.get(model), dict) else {}
    capabilities = _get_model_capabilities(model)
    normalized_model = _normalize_model_id(model)
    registry_defaults = MODEL_DEFAULTS_MAP.get(normalized_model, {})
    result: Dict[str, Any] = {}
    if capabilities.get("temperature"):
        if isinstance(model_settings, dict) and "temperature" in model_settings:
            result["temperature"] = _clamp_float(model_settings["temperature"], 0.0, 2.0, DEFAULT_TEMPERATURE)
        elif "temperature" in defaults:
            result["temperature"] = _clamp_float(defaults["temperature"], 0.0, 2.0, DEFAULT_TEMPERATURE)
        elif "temperature" in registry_defaults:
            result["temperature"] = _clamp_float(registry_defaults["temperature"], 0.0, 2.0, DEFAULT_TEMPERATURE)
        elif "temperature" in agent:
            result["temperature"] = _clamp_float(agent["temperature"], 0.0, 2.0, DEFAULT_TEMPERATURE)
        else:
            result["temperature"] = DEFAULT_TEMPERATURE
    if capabilities.get("top_p"):
        if isinstance(model_settings, dict) and "top_p" in model_settings:
            result["top_p"] = _clamp_float(model_settings["top_p"], 0.0, 1.0, DEFAULT_TOP_P)
        elif "top_p" in defaults:
            result["top_p"] = _clamp_float(defaults["top_p"], 0.0, 1.0, DEFAULT_TOP_P)
        elif "top_p" in registry_defaults:
            result["top_p"] = _clamp_float(registry_defaults["top_p"], 0.0, 1.0, DEFAULT_TOP_P)
        elif "top_p" in agent:
            result["top_p"] = _clamp_float(agent["top_p"], 0.0, 1.0, DEFAULT_TOP_P)
        else:
            result["top_p"] = DEFAULT_TOP_P
    if capabilities.get("reasoning"):
        reasoning_source = None
        if isinstance(model_settings, dict) and "reasoning_effort" in model_settings:
            reasoning_source = model_settings["reasoning_effort"]
        elif "reasoning_effort" in defaults:
            reasoning_source = defaults["reasoning_effort"]
        elif "reasoning_effort" in registry_defaults:
            reasoning_source = registry_defaults["reasoning_effort"]
        elif "reasoning_effort" in agent:
            reasoning_source = agent["reasoning_effort"]
        result["reasoning_effort"] = _normalize_reasoning_effort(reasoning_source)
    if ENABLE_RESPONSE_FORMAT and capabilities.get("response_format"):
        response_candidates = (
            model_settings,
            defaults,
            registry_defaults,
            agent,
        )
        for source in response_candidates:
            if not isinstance(source, dict):
                continue
            normalized_format = _normalize_response_format(source.get("response_format"))
            if normalized_format is not None:
                result["response_format"] = normalized_format
                break
    max_tokens_source = (
        model_settings,
        defaults,
        registry_defaults,
        agent,
    )
    for source in max_tokens_source:
        if not isinstance(source, dict):
            continue
        if "max_output_tokens" in source and source["max_output_tokens"] is not None:
            result["max_output_tokens"] = _clamp_int(
                source["max_output_tokens"],
                MIN_OUTPUT_TOKENS,
                MAX_OUTPUT_TOKENS,
                MAX_OUTPUT_TOKENS,
            )
            break
    return result

BLOCK_TAGS = ("h1", "h2", "h3", "p", "div", "small", "note", "blockquote", "li")


class AgentRunnerError(RuntimeError):
    """Raised when an agent cannot be executed due to configuration issues."""


class AgentDiffApplicationError(RuntimeError):
    """Raised when diff-based agent output cannot be applied."""


_clients: Dict[str, OpenAI] = {}


def _get_client(provider_name: str) -> OpenAI:
    """Return a cached OpenAI client for the given provider."""
    normalized = _normalize_provider_id(provider_name)
    if not normalized:
        raise AgentRunnerError("Není zadaný provider pro model.")
    if normalized in _clients:
        return _clients[normalized]
    if OpenAI is None:
        reason = f"ImportError: {_import_error}" if _import_error else ""
        message = "Knihovna 'openai' není nainstalovaná. Přidejte ji do prostředí."
        if reason:
            message = f"{message} ({reason})"
        raise AgentRunnerError(message)

    provider_config = _get_provider_config(normalized)
    api_key = _require_api_key(provider_config)
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    api_base = provider_config.get("api_base")
    if isinstance(api_base, str) and api_base.strip():
        client_kwargs["base_url"] = api_base.strip()
    headers = provider_config.get("default_headers")
    if isinstance(headers, dict) and headers:
        client_kwargs["default_headers"] = headers
    client = OpenAI(**client_kwargs)
    _clients[normalized] = client
    return client


def _client_supports_response_format(client: OpenAI, provider_name: str) -> bool:
    normalized = _normalize_provider_id(provider_name)
    if not normalized:
        return False
    if normalized in _CLIENT_SUPPORTS_NATIVE_RESPONSE_FORMAT:
        return _CLIENT_SUPPORTS_NATIVE_RESPONSE_FORMAT[normalized]
    try:
        signature = inspect.signature(client.responses.create)
        supports = "response_format" in signature.parameters
    except (AttributeError, ValueError, TypeError):
        supports = False
    _CLIENT_SUPPORTS_NATIVE_RESPONSE_FORMAT[normalized] = supports
    return supports


def _perform_responses_request(
    client: OpenAI,
    request_kwargs: Dict[str, Any],
    native_response_format_support: bool,
    provider_config: Dict[str, Any],
    api_key: str,
) -> Response:
    if native_response_format_support:
        return client.responses.create(**request_kwargs)

    payload = copy.deepcopy(request_kwargs)
    extra_body = payload.pop("extra_body", None)
    if isinstance(extra_body, dict):
        payload.update(extra_body)

    api_base = str(provider_config.get("api_base") or "").rstrip("/")
    if not api_base:
        raise AgentRunnerError("Provider nemá nastavené 'api_base' pro Responses API.")

    url = api_base + "/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    default_headers = provider_config.get("default_headers")
    if isinstance(default_headers, dict):
        headers.update(default_headers)

    response = requests.post(url, headers=headers, json=payload, timeout=90)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise AgentRunnerError(
            f"Responses API vrátilo chybu {response.status_code}: {response.text}"
        ) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise AgentRunnerError("Responses API vrátilo neplatný JSON.") from exc
    return _DictResponse(payload)


def _extract_output_text(response: Response) -> str:
    """Safely extract concatenated text output from a Responses API reply."""
    try:
        print("[LLMDebug][_extract_output_text] response type:", type(response))
        has_out = hasattr(response, "output_text")
        print("[LLMDebug][_extract_output_text] has output_text attr:", has_out)
    except Exception:
        pass
    if hasattr(response, "output_text") and response.output_text:
        try:
            return response.output_text.strip()
        except Exception:
            print("[LLMDebug][_extract_output_text] failed to read response.output_text")

    data: Dict[str, Any]
    try:
        data = response.model_dump()
    except AttributeError:
        data = getattr(response, "__dict__", {})

    output_chunks = []
    output = data.get("output") or data.get("outputs") or []
    try:
        print("[LLMDebug][_extract_output_text] data keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        print("[LLMDebug][_extract_output_text] output type:", type(output))
    except Exception:
        pass
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            try:
                print("[LLMDebug][_extract_output_text] item type:", type(item), "content type:", type(content))
            except Exception:
                pass
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                try:
                    print("[LLMDebug][_extract_output_text] block type:", block_type)
                except Exception:
                    pass
                if block_type not in {"text", "output_text"}:
                    continue
                text_payload = block.get("text")
                if isinstance(text_payload, dict):
                    value = text_payload.get("value") or text_payload.get("text")
                    if value:
                        output_chunks.append(str(value))
                elif isinstance(text_payload, str):
                    output_chunks.append(text_payload)
    return "\n".join(output_chunks).strip()


def _extract_stop_reason(response_dict: Dict[str, Any]) -> Optional[str]:
    output = response_dict.get("output") or response_dict.get("outputs") or []
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                reason = item.get("stop_reason")
                if reason:
                    return str(reason)
    return None


def _extract_usage(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    usage = response_dict.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    allowed = ("input_tokens", "output_tokens", "total_tokens")
    return {key: usage[key] for key in allowed if key in usage}


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences to simplify JSON parsing."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.split("\n", 1)
        if len(inner) == 2:
            payload = inner[1]
            closing_index = payload.rfind("\n```")
            if closing_index != -1:
                return payload[:closing_index].strip()
        return stripped.strip("`").strip()
    return stripped


def _safe_json_loads(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse JSON if possible, returning None on failure."""
    if not candidate or not isinstance(candidate, str):
        return None
    text = _strip_code_fences(candidate)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            print("[LLMDebug][_safe_json_loads] JSON parse failed; preview:\n", repr(text)[:1000])
        except Exception:
            pass
        return None
    return parsed if isinstance(parsed, dict) else None


def _log_diff_warning(payload: Dict[str, Any], message: str) -> None:
    """Print a warning about diff processing failure including page context."""
    if not isinstance(payload, dict):
        payload = {}
    details = []
    page_uuid = payload.get("page_uuid") or payload.get("page_id")
    if page_uuid not in (None, "", []):
        details.append(f"page_uuid={page_uuid}")
    page_label = payload.get("page_number") or payload.get("page_index")
    if page_label not in (None, "", []):
        details.append(f"page_ref={page_label}")
    context = ", ".join(details) if details else "page_context=unknown"
    print(f"[AgentDiff] {message} ({context})")


def _log_raw_response(text: Any) -> None:
    """Debug helper to show raw LLM response exactly as dorazila."""
    try:
        preview = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
    except Exception:
        preview = str(text)
    print("===[RawResponse]===")
    try:
        print(preview)
    except Exception:
        print("[RawResponse] Nelze vypsat obsah.")
    print("===[RawResponse End]===")


def _apply_diff_to_document(
    document: Dict[str, Any],
    diff_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a diff `{changes: [...]}` to the original document payload."""
    if not isinstance(document, dict):
        raise AgentDiffApplicationError("Vstupní dokument má neočekávaný formát.")
    original_blocks = document.get("blocks")
    if not isinstance(original_blocks, list):
        raise AgentDiffApplicationError("Vstupní dokument neobsahuje pole 'blocks'.")
    changes = diff_payload.get("changes")
    if not isinstance(changes, list):
        raise AgentDiffApplicationError("Diff výstup postrádá pole 'changes'.")

    # Deep copy input blocks so we don't mutate cached originals.
    cloned_blocks = []
    block_index_map: Dict[str, Dict[str, Any]] = {}
    for block in original_blocks:
        cloned = copy.deepcopy(block)
        cloned_blocks.append(cloned)
        block_id = cloned.get("id")
        if isinstance(block_id, str):
            block_index_map[block_id] = cloned

    removed_ids: set[str] = set()

    for change in changes:
        if not isinstance(change, dict):
            raise AgentDiffApplicationError("Položka v 'changes' není objekt.")
        block_id = change.get("id")
        if not isinstance(block_id, str) or not block_id.strip():
            raise AgentDiffApplicationError("Položka diffu postrádá platné 'id'.")
        target = block_index_map.get(block_id)
        if target is None:
            # Podporujeme i nové bloky (např. při splitu) – přidáme je na konec jako paragraf.
            target = {"id": block_id, "type": "p", "text": ""}
            cloned_blocks.append(target)
            block_index_map[block_id] = target
        block_type = str(target.get("type") or "").lower()
        if block_type == "note":
            # Notes slouží jen jako kontext – ignorujeme změny i smazání.
            continue
        if "text" not in change:
            # Bez textu není co aplikovat; přeskočíme tichou úpravou.
            continue
        raw_text = change.get("text")
        text_value = "" if raw_text is None else str(raw_text)
        if not text_value.strip():
            removed_ids.add(block_id)
            continue
        target["text"] = text_value

    if not removed_ids and all(block.get("text") == original.get("text") for block, original in zip(cloned_blocks, original_blocks)):
        # No effective changes – return original clone to keep structure uniform.
        result_document = copy.deepcopy(document)
        result_document["blocks"] = cloned_blocks
        return result_document

    result_blocks = [
        block for block in cloned_blocks
        if not isinstance(block.get("id"), str) or block.get("id") not in removed_ids
    ]

    result_document = copy.deepcopy(document)
    result_document["blocks"] = result_blocks
    return result_document


def _infer_block_type(element) -> str:
    name = (element.name or "").lower()
    if name in {"h1", "h2", "h3"}:
        return name
    if name == "p":
        has_small_child = False
        for child in element.contents:
            if getattr(child, "name", None):
                child_name = child.name.lower()
                if child_name == "small":
                    has_small_child = True
                    continue
                if child_name == "br":
                    continue
                return "p"
            else:
                if isinstance(child, str) and child.strip():
                    return "p"
        if has_small_child:
            return "small"
        return "p"
    if name == "small":
        return "small"
    if name == "note":
        return "note"
    if name == "blockquote":
        return "blockquote"
    if name == "li":
        return "li"
    if name == "div":
        classes = element.get("class") or []
        normalized = [cls.lower() for cls in classes]
        if "centered" in normalized:
            return "centered"
        if "note" in normalized:
            return "note"
        return "p"
    return "p"


def _html_to_blocks(html_text: str) -> list[Dict[str, str]]:
    soup = BeautifulSoup(html_text or "", "html.parser")
    if soup is None:
        return []
    blocks: list[Dict[str, str]] = []
    seen = set()

    for element in soup.find_all(BLOCK_TAGS):
        parent = element.find_parent(BLOCK_TAGS)
        if parent:
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        block_type = _infer_block_type(element)
        token = id(element)
        if token in seen:
            continue
        seen.add(token)
        blocks.append({
            "type": block_type,
            "text": text,
        })

    if not blocks:
        fallback_text = soup.get_text(" ", strip=True)
        if fallback_text:
            blocks.append({"type": "p", "text": fallback_text})

    return [
        {"id": f"b{index}", "type": block["type"], "text": block["text"]}
        for index, block in enumerate(blocks, start=1)
    ]


def _build_document_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    collection = str(payload.get("collection") or "").strip().lower()
    if collection == "joiners":
        context = payload.get("stitch_context")
        if not context:
            raise AgentRunnerError("Chybí data pro napojení stran.")
        try:
            serialized = json.loads(json.dumps(context, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise AgentRunnerError(f"Kontext pro napojení stran není validní JSON: {exc}") from exc
        if isinstance(serialized, dict) and "language_hint" not in serialized:
            serialized["language_hint"] = payload.get("language_hint") or DEFAULT_LANGUAGE_HINT
        return serialized

    python_html = payload.get("python_html")
    if not python_html or not str(python_html).strip():
        raise AgentRunnerError("Python výstup pro agenta je prázdný.")

    language_hint = payload.get("language_hint") or DEFAULT_LANGUAGE_HINT
    ignore_format = collection in {"custom_lmm", "correctors"} and bool(payload.get("ignore_format"))

    blocks = _html_to_blocks(str(python_html))
    if not blocks:
        raise AgentRunnerError("Python výstup se nepodařilo převést na bloky.")

    if ignore_format:
        filtered_blocks = [block for block in blocks if str(block.get("type") or "").lower() != "note"]
        joined_text = "\n".join(block.get("text", "") for block in filtered_blocks if block.get("text"))
        return {
            "language_hint": language_hint,
            "text": joined_text,
            "ignore_format": True,
        }

    return {
        "language_hint": language_hint,
        "blocks": blocks,
    }


def run_agent(agent: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a stored agent configuration via the Responses API."""
    if not isinstance(agent, dict):
        raise AgentRunnerError("Agent nemá očekávaný formát.")

    prompt = str(agent.get("prompt") or "").strip()
    if not prompt:
        raise AgentRunnerError("Agent nemá vyplněný prompt.")

    collection = str(payload.get("collection") or "").strip().lower()
    ignore_format = collection in {"custom_lmm", "correctors"} and bool(payload.get("ignore_format"))
    reader_inputs: Optional[Dict[str, Any]] = None
    user_payload = ''
    user_message_blocks: Optional[list] = None
    if collection == "readers":
        reader_inputs = _prepare_reader_inputs(payload or {})
        document_payload = reader_inputs["document_payload"]
    else:
        document_payload = _build_document_payload(payload or {})
        if ignore_format:
            user_payload = str(document_payload.get("text") or "")
            user_message_blocks = [{"type": "input_text", "text": user_payload}]
        else:
            user_payload = json.dumps(document_payload, ensure_ascii=False, indent=2)
            user_message_blocks = [{"type": "input_text", "text": user_payload}]

    model = (
        str(agent.get("model")).strip()
        if agent.get("model")
        else DEFAULT_MODEL
    )
    upstream_model_id = _get_upstream_model_id(model)
    if collection == "readers" and not _model_supports_scan(model):
        raise AgentRunnerError(f"Model {model} nepodporuje čtení ze skenu – vyberte jiný model.")
    if collection != "readers" and not _model_supports_text(model):
        raise AgentRunnerError(f"Model {model} není povolený pro textové agenty.")

    model_definition = MODEL_DEFINITION_MAP.get(_normalize_model_id(model)) or {}
    provider_name = model_definition.get("provider") or "openrouter"
    provider_config = _get_provider_config(provider_name)
    provider_supports_responses = bool(provider_config.get("supports_responses", False))
    provider_supports_chat = bool(provider_config.get("supports_chat", False))
    if collection == "readers" and not provider_supports_chat:
        raise AgentRunnerError(f"Provider {provider_name} nepodporuje chat completions – vyberte jiný model.")
    if collection != "readers" and not (provider_supports_responses or provider_supports_chat):
        raise AgentRunnerError(f"Provider {provider_name} nepodporuje ani Responses ani Chat API pro textové agenty.")

    api_key = _require_api_key(provider_config)
    client = _get_client(provider_name)
    native_response_format_support = (
        _client_supports_response_format(client, provider_name) if provider_supports_responses else False
    )

    capabilities = _get_model_capabilities(model)
    effective_settings = _get_effective_settings(agent, model)
    payload_temperature = payload.get("temperature") if isinstance(payload, dict) else None
    payload_top_p = payload.get("top_p") if isinstance(payload, dict) else None
    payload_reasoning = payload.get("reasoning_effort") if isinstance(payload, dict) else None
    payload_response_format = (
        payload.get("response_format") if ENABLE_RESPONSE_FORMAT and isinstance(payload, dict) else None
    )

    max_output_tokens = effective_settings.get("max_output_tokens")
    if isinstance(max_output_tokens, (int, float)):
        max_output_tokens = _clamp_int(
            max_output_tokens,
            MIN_OUTPUT_TOKENS,
            MAX_OUTPUT_TOKENS,
            MAX_OUTPUT_TOKENS,
        )
    else:
        max_output_tokens = None

    temperature = None
    if capabilities.get("temperature"):
        if payload_temperature is not None:
            temperature = _clamp_float(payload_temperature, 0.0, 2.0, effective_settings.get("temperature", 0.0))
        else:
            temperature = effective_settings.get("temperature")

    top_p = None
    if capabilities.get("top_p"):
        if payload_top_p is not None:
            top_p = _clamp_float(payload_top_p, 0.0, 1.0, effective_settings.get("top_p", 1.0))
        else:
            top_p = effective_settings.get("top_p")

    reasoning_source = payload_reasoning or effective_settings.get("reasoning_effort")
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_source)
    supports_temperature = capabilities.get("temperature", True)
    supports_top_p = capabilities.get("top_p", True)
    supports_reasoning = capabilities.get("reasoning", False)
    supports_response_format = ENABLE_RESPONSE_FORMAT and capabilities.get("response_format", True)

    response_format = None
    expected_response_format = None
    if ENABLE_RESPONSE_FORMAT and supports_response_format:
        response_format = _normalize_response_format(payload_response_format)
        if response_format is None:
            response_format = _normalize_response_format(effective_settings.get("response_format"))
        expected_response_format = response_format if response_format is not None else None

    use_responses_api = collection != "readers" and provider_supports_responses
    use_chat_api = collection == "readers" or not use_responses_api

    request_kwargs: Dict[str, Any] = {}
    if use_responses_api:
        request_kwargs = {
            "model": upstream_model_id,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": user_message_blocks or [],
                },
            ],
        }
        if supports_temperature and temperature is not None:
            request_kwargs["temperature"] = _clamp_float(
                temperature,
                0.0,
                2.0,
                effective_settings.get("temperature", 0.0),
            )
        elif not supports_temperature and payload_temperature is not None:
            print(f"[AgentDebug] Model {model} ignoruje parametr temperature – nebude odeslán.")
        if supports_top_p and top_p is not None:
            request_kwargs["top_p"] = _clamp_float(
                top_p,
                0.0,
                1.0,
                effective_settings.get("top_p", 1.0),
            )
        elif not supports_top_p and payload_top_p is not None:
            print(f"[AgentDebug] Model {model} ignoruje parametr top_p – nebude odeslán.")
        if supports_reasoning:
            request_kwargs["reasoning"] = {"effort": normalized_reasoning_effort}
            agent["reasoning_effort"] = normalized_reasoning_effort
        if ENABLE_RESPONSE_FORMAT and response_format is not None:
            if supports_response_format:
                request_kwargs["response_format"] = response_format
                if not native_response_format_support:
                    print("[AgentDebug] Klient nepodporuje response_format nativně – požadavek pošlu přímo přes REST.")
            else:
                print(f"[AgentDebug] Model {model} ignoruje parametr response_format – nebude odeslán.")
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = int(max_output_tokens)

        debug_agent = agent.get("name") or agent.get("display_name") or "unknown"
        print("\n=== [AgentDebug] Responses API Request ===")
        print(f"Provider: {provider_name}")
        print(f"Agent: {debug_agent}")
        print(f"Model: {request_kwargs.get('model')}")
        if "temperature" in request_kwargs:
            print(f"Temperature: {request_kwargs.get('temperature')}")
        if "top_p" in request_kwargs:
            print(f"Top P: {request_kwargs.get('top_p')}")
        if "reasoning" in request_kwargs:
            print(f"Reasoning effort: {request_kwargs['reasoning'].get('effort')}")
        if "max_output_tokens" in request_kwargs:
            print(f"Max output tokens: {request_kwargs.get('max_output_tokens')}")
        if ENABLE_RESPONSE_FORMAT and "response_format" in request_kwargs:
            try:
                formatted_response_format = json.dumps(request_kwargs["response_format"], ensure_ascii=False)
            except Exception:
                formatted_response_format = str(request_kwargs["response_format"])
            print(f"Response format: {formatted_response_format}")
        print("--- Prompt ---")
        for part in request_kwargs.get("input", []):
            role = part.get("role")
            contents = part.get("content") or []
            for block in contents:
                if block.get("type") == "input_text":
                    print(f"[{role}] {block.get('text')}")
        print("=== [AgentDebug] End Request ===\n")

        def _retry_without_unsupported(error: Exception) -> Optional[Response]:
            removed_any = False
            message = " ".join(str(part) for part in getattr(error, "args", []) if part)
            message_lower = message.lower()
            if ENABLE_RESPONSE_FORMAT and isinstance(error, TypeError) and "response_format" in message_lower and "unexpected keyword" in message_lower:
                print("[AgentDebug] Klient nepodporuje parametr response_format – opakuji bez něj.")
                request_kwargs.pop("response_format", None)
                removed_any = True
                _CLIENT_SUPPORTS_NATIVE_RESPONSE_FORMAT[provider_name] = False
            if BadRequestError is None or not isinstance(error, BadRequestError):
                if not removed_any:
                    return None
            else:
                if "temperature" in request_kwargs and "temperature" in message_lower:
                    print("[AgentDebug] Model nepodporuje parametr temperature – opakuji bez něj.")
                    request_kwargs.pop("temperature", None)
                    removed_any = True
                if "top_p" in request_kwargs and "top_p" in message_lower:
                    print("[AgentDebug] Model nepodporuje parametr top_p – opakuji bez něj.")
                    request_kwargs.pop("top_p", None)
                    removed_any = True
                if "max_output_tokens" in request_kwargs and (
                    "max_output_tokens" in message_lower or "max tokens" in message_lower
                ):
                    print("[AgentDebug] Model nepodporuje parametr max_output_tokens – opakuji bez něj.")
                    request_kwargs.pop("max_output_tokens", None)
                    removed_any = True
                if "reasoning" in request_kwargs and "reasoning" in message_lower:
                    print("[AgentDebug] Model nepodporuje pole reasoning – opakuji bez něj.")
                    request_kwargs.pop("reasoning", None)
                    removed_any = True
                if ENABLE_RESPONSE_FORMAT and "response_format" in request_kwargs and "response_format" in message_lower:
                    print("[AgentDebug] Model nepodporuje pole response_format – opakuji bez něj.")
                    request_kwargs.pop("response_format", None)
                    removed_any = True
            if not removed_any:
                return None
            updated_support = _client_supports_response_format(client, provider_name) if ENABLE_RESPONSE_FORMAT else False
            return _perform_responses_request(
                client,
                request_kwargs,
                updated_support,
                provider_config,
                api_key,
            )

        try:
            response = _perform_responses_request(
                client,
                request_kwargs,
                native_response_format_support,
                provider_config,
                api_key,
            )
        except Exception as exc:
            retry = _retry_without_unsupported(exc)
            if retry is None:
                raise
            response = retry

        try:
            response_dict = response.model_dump()
        except AttributeError:
            response_dict = getattr(response, "__dict__", {})

        print("=== [AgentDebug] Responses API Response ===")
        try:
            print(response.output_text)
        except Exception:
            try:
                print(json.dumps(response_dict, ensure_ascii=False, indent=2))
            except Exception:
                print("[AgentDebug] Nelze zobrazit odpověď.")
        print("=== [AgentDebug] End Response ===\n")

        text = _extract_output_text(response)
        raw_model_output = text
        stop_reason = _extract_stop_reason(response_dict)
        usage = _extract_usage(response_dict)
    else:
        if collection == "readers":
            chat_content = (reader_inputs or {}).get("chat_content") or []
            response = _run_reader_chat_completion(
                client,
                upstream_model_id,
                prompt,
                chat_content,
                temperature if supports_temperature else None,
                top_p if supports_top_p else None,
                max_output_tokens,
            )
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_payload or json.dumps(document_payload, ensure_ascii=False)},
            ]
            chat_kwargs: Dict[str, Any] = {"model": upstream_model_id, "messages": messages}
            if supports_temperature and temperature is not None:
                chat_kwargs["temperature"] = _clamp_float(
                    temperature,
                    0.0,
                    2.0,
                    effective_settings.get("temperature", 0.0),
                )
            if supports_top_p and top_p is not None:
                chat_kwargs["top_p"] = _clamp_float(
                    top_p,
                    0.0,
                    1.0,
                    effective_settings.get("top_p", 1.0),
                )
            if max_output_tokens is not None:
                chat_kwargs["max_tokens"] = int(max_output_tokens)
            if supports_reasoning:
                chat_kwargs["reasoning"] = {"effort": normalized_reasoning_effort}
                agent["reasoning_effort"] = normalized_reasoning_effort
            print("\n=== [AgentDebug] Chat Completions Request ===")
            print(f"Provider: {provider_name}")
            print(f"Model: {model}")
            if "temperature" in chat_kwargs:
                print(f"Temperature: {chat_kwargs['temperature']}")
            if "top_p" in chat_kwargs:
                print(f"Top P: {chat_kwargs['top_p']}")
            if "reasoning" in chat_kwargs:
                print(f"Reasoning effort: {chat_kwargs['reasoning'].get('effort')}")
            if "max_tokens" in chat_kwargs:
                print(f"Max tokens: {chat_kwargs['max_tokens']}")
            print("--- Prompt ---")
            print(f"[system] {prompt}")
            print(f"[user] {chat_kwargs['messages'][1].get('content')}")
            print("=== [AgentDebug] End Chat Request ===\n")
            response = client.chat.completions.create(**chat_kwargs)

        try:
            response_dict = response.model_dump()
        except AttributeError:
            response_dict = getattr(response, "__dict__", {})
        print("=== [AgentDebug] Chat Completion Response ===")
        try:
            print(_extract_chat_output_text(response))
        except Exception:
            try:
                print(json.dumps(response_dict, ensure_ascii=False, indent=2))
            except Exception:
                print("[AgentDebug] Nelze zobrazit odpověď.")
        print("=== [AgentDebug] End Chat Response ===\n")
        text = _extract_chat_output_text(response)
        raw_model_output = text
        stop_reason = _extract_stop_reason(response_dict) if isinstance(response_dict, dict) else None
        usage = response_dict.get("usage") if isinstance(response_dict, dict) else None
    diff_applied = False
    diff_changes = None
    diff_error: Optional[str] = None
    output_document: Optional[Dict[str, Any]] = None

    parsed_output = None if ignore_format else _safe_json_loads(text)
    _log_raw_response(raw_model_output)
    try:
        print("[LLMDebug] parsed text preview:\n", repr(str(text)))
        print("[LLMDebug] parsed_output type:", type(parsed_output))
    except Exception:
        pass
    if ENABLE_RESPONSE_FORMAT and expected_response_format:
        if parsed_output is None:
            raise AgentRunnerError("Model vrátil výstup, který není platný JSON podle očekávaného schématu.")
        fmt_type = expected_response_format.get("type")
        if fmt_type == "json_object":
            if not isinstance(parsed_output, dict):
                raise AgentRunnerError("Model vrátil výstup v jiném formátu než JSON objekt.")
        elif fmt_type == "json_schema":
            schema_payload = expected_response_format.get("json_schema") or {}
            schema_name = schema_payload.get("name")
            if schema_name == "text_block_corrector":
                ok, reason = _validate_text_block_corrector(parsed_output)
                if not ok:
                    raise AgentRunnerError(f"Výstup modelu neodpovídá očekávanému schématu: {reason}.")
    blocks_present = isinstance(document_payload.get("blocks"), list)
    if blocks_present and parsed_output:
        if isinstance(parsed_output.get("changes"), list):
            try:
                applied_document = _apply_diff_to_document(document_payload, parsed_output)
            except AgentDiffApplicationError as exc:
                diff_error = str(exc)
                _log_diff_warning(document_payload, f"Diff aplikace selhala: {exc}")
                if parsed_output:
                    try:
                        print("=== [AgentDiff][RawResponse] ===")
                        print(json.dumps(parsed_output, ensure_ascii=False, indent=2))
                        print("=== [AgentDiff][RawResponse End] ===")
                    except Exception:
                        print("[AgentDiff] Nepodařilo se vypsat raw response.")
                text = json.dumps(document_payload, ensure_ascii=False, indent=2)
            else:
                text = json.dumps(applied_document, ensure_ascii=False, indent=2)
                diff_applied = True
                diff_changes = copy.deepcopy(parsed_output.get("changes"))
                output_document = applied_document
                try:
                    print("[LLMDebug] Diff applied, changes count:", len(diff_changes))
                except Exception:
                    pass
        elif isinstance(parsed_output.get("blocks"), list):
            # Agent vrátil plný dokument – znormalizujeme formátování.
            text = json.dumps(parsed_output, ensure_ascii=False, indent=2)
            output_document = parsed_output
        else:
            text = text.strip()
    elif parsed_output and isinstance(parsed_output.get("blocks"), list):
        # I když nemáme 'blocks' v document_payload (např. joiner snapshot),
        # uchováme výsledek pro případné další zpracování na klientovi.
        output_document = parsed_output
        text = json.dumps(parsed_output, ensure_ascii=False, indent=2)

    return {
        "text": text,
        "response_id": response_dict.get("id") or getattr(response, "id", None),
        "model": response_dict.get("model") or upstream_model_id,
        "stop_reason": stop_reason,
        "usage": usage,
        "input_document": document_payload,
        "reasoning_effort": agent.get("reasoning_effort"),
        "diff_applied": diff_applied,
        "diff_changes": diff_changes if diff_applied else None,
        "output_document": output_document,
        "diff_error": diff_error,
        "format_ignored": ignore_format,
    }
