"""Background export job runner with export orchestration."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from ..core.agent_runner import AgentRunnerError, run_agent as run_agent_via_responses
from ..core.export_jobs import AbortRequested, ExportJob, ExportJobParams
from ..core.main_processor import AltoProcessor
from ..core.comparison_legacy import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    normalize_agent_collection,
    read_agent_file,
)


BLOCK_TAGS = {
    "p",
    "div",
    "blockquote",
    "li",
    "h1",
    "h2",
    "h3",
    "small",
    "note",
}
HYPHENLIKE = "-–—‑‒−‐"
SENTENCE_END_RE = re.compile(r"[.!?…][\"'„“”«»‚‘’‹›)\]\s]*$")
SENTENCE_START_RE = re.compile(r"^[\s\"'„“”«»‚‘’‹›(\[{\-—–]*[A-ZÀ-Ž]")
LEADING_PUNCT_RE = re.compile(r"^[,.;:!?)]")


@dataclass
class PagePlan:
    uuid: str
    index: int
    page_number: Optional[str]
    title: Optional[str]
    cached_python: Optional[str] = None
    cached_llm: Optional[str] = None
    cached_reader: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Snippet:
    text: str
    html: str
    tag: str


@dataclass
class PageContent:
    plan: PagePlan
    blocks: List[str]
    snippets: List[Snippet]

    def first_snippet(self) -> Optional[Snippet]:
        return self.snippets[0] if self.snippets else None

    def last_snippet(self) -> Optional[Snippet]:
        return self.snippets[-1] if self.snippets else None

    def first_non_note_snippet(self) -> Optional[Snippet]:
        return next((snippet for snippet in self.snippets if snippet and snippet.tag != "note"), None)

    def last_non_note_snippet(self) -> Optional[Snippet]:
        for snippet in reversed(self.snippets):
            if snippet and snippet.tag != "note":
                return snippet
        return None

    def render_html(self) -> str:
        if not self.blocks:
            return ""
        return "\n".join(block for block in self.blocks if block is not None)


class ExportBuilder:
    def __init__(self, job: ExportJob) -> None:
        self.job = job
        self.params: ExportJobParams = job.params
        self.processor = AltoProcessor(api_base_url=self.params.api_base)
        self._debug_enabled = True
        self._epub_log_tag = "[EPUB_META]"

    def build(self) -> tuple[str, str]:
        pages = self._select_pages()
        if not pages:
            raise ValueError("Export neobsahuje žádné stránky.")
        self.job.total_pages = len(pages)

        page_contents: List[PageContent] = []
        for idx, plan in enumerate(pages, start=1):
            self._check_abort()
            html = self._render_source_html(plan)
            content = self._prepare_page_content(plan, html)
            if not content:
                self._log_debug(f"Stránka {plan.uuid} byla přeskočena (žádný použitelný obsah).")
            else:
                page_contents.append(content)
            label = plan.page_number or f"Strana {plan.index + 1}"
            self.job.update_progress(idx, len(pages), f"Zpracování {label} ({idx}/{len(pages)})")

        if not page_contents:
            raise RuntimeError("Žádná ze stránek neobsahuje použitelná data pro export.")

        self._apply_joiner(page_contents)
        combined_html = self._compose_document(page_contents)
        if self.params.export_format == "epub":
            path = self._build_epub(combined_html)
            filename = self._build_filename()
            return path, filename

        output_text = self._convert_format(combined_html)
        suffix = f".{self.params.export_format}"
        fd, path = tempfile.mkstemp(prefix="alto_export_", suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(output_text)
        filename = self._build_filename()
        return path, filename

    def _check_abort(self) -> None:
        if self.job.abort_requested:
            raise AbortRequested()

    def _log_debug(self, message: str) -> None:
        if not self._debug_enabled:
            return
        try:
            print(f"[ExportDebug] {message}")
        except Exception:
            pass

    def _render_source_html(self, plan: PagePlan) -> str:
        source = self.params.source
        self._log_debug(f"Zpracovávám stránku {plan.uuid} (index {plan.index}, source={source})")
        if source == "algorithmic":
            return self._ensure_algorithmic_html(plan)
        if source == "llm":
            return self._ensure_llm_html(plan)
        if source == "ocr":
            return self._ensure_reader_html(plan)
        raise ValueError(f"Nepodporovaný typ exportu: {source}")

    def _ensure_algorithmic_html(self, plan: PagePlan) -> str:
        if plan.cached_python:
            has_illustration = ("class=\"illustration\"" in plan.cached_python) or ("class='illustration'" in plan.cached_python)
            if self.params.export_format == "epub" and not self.params.ignore_images:
                if not has_illustration:
                    self._log_debug(f"Cache python HTML pro {plan.uuid} neobsahuje ilustrace – obnovuji z ALTO")
                else:
                    self._log_debug(f"Používám cache python HTML pro {plan.uuid} (obsahuje ilustrace)")
                    return plan.cached_python
            else:
                self._log_debug(f"Používám cache python HTML pro {plan.uuid}")
                return plan.cached_python
        self._log_debug(f"Stahuji ALTO data pro {plan.uuid}")
        alto_xml = self.processor.get_alto_data(plan.uuid)
        if not alto_xml:
            self._log_debug(f"ALTO data prázdná pro {plan.uuid}")
            raise RuntimeError(f"Nepodařilo se stáhnout ALTO pro stránku {plan.uuid}.")
        html = self.processor.get_formatted_text(alto_xml, plan.uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        if not html or not html.strip():
            self._log_debug(f"Formátovaný výstup prázdný pro {plan.uuid} – stránka bude přeskočena.")
            plan.cached_python = ""
            return ""
        self._log_debug(f"ALTO převod dokončen pro {plan.uuid} – délka {len(html)} znaků")
        plan.cached_python = html
        return html

    def _ensure_llm_html(self, plan: PagePlan) -> str:
        if plan.cached_llm:
            self._log_debug(f"Používám cache LLM HTML pro {plan.uuid}")
            return plan.cached_llm
        agent_options = self.params.llm_agent or {}
        agent_config = self._prepare_agent_config(agent_options)
        if not agent_config:
            raise RuntimeError("LLM agent není vybrán.")
        python_html = self._ensure_algorithmic_html(plan)
        if not python_html or not python_html.strip():
            self._log_debug(f"LLM export přeskočen – algoritmický výstup je prázdný ({plan.uuid}).")
            plan.cached_llm = ""
            return ""
        payload = self._build_agent_payload(agent_options, plan, collection="correctors")
        payload["python_html"] = python_html
        payload["language_hint"] = self.params.language_hint
        self._log_debug(f"Spouštím LLM agenta {payload.get('name') or '(bez názvu)'} pro {plan.uuid}")
        result = self._run_agent(agent_config, payload, "LLM korekcí")
        html = self._extract_agent_html(result)
        if not html:
            self._log_debug(f"LLM agent nevrátil výsledek pro {plan.uuid}")
            raise RuntimeError("LLM agent nevrátil žádný použitelný výsledek.")
        self._log_debug(f"LLM agent dokončil stránku {plan.uuid}, délka {len(html)} znaků")
        plan.cached_llm = html
        return html

    def _ensure_reader_html(self, plan: PagePlan) -> str:
        if plan.cached_reader:
            self._log_debug(f"Používám cache OCR HTML pro {plan.uuid}")
            return plan.cached_reader
        agent_options = self.params.ocr_agent or {}
        agent_config = self._prepare_agent_config(agent_options)
        if not agent_config:
            raise RuntimeError("Agent pro OCR není vybrán.")
        payload = self._build_agent_payload(agent_options, plan, collection="readers")
        payload["scan_uuid"] = plan.uuid
        payload["scan_stream"] = agent_options.get("scan_stream") or "IMG_FULL"
        self._log_debug(f"Spouštím OCR agenta {payload.get('name') or '(bez názvu)'} pro {plan.uuid}")
        result = self._run_agent(agent_config, payload, "OCR čtení")
        html = self._extract_agent_html(result)
        if not html:
            self._log_debug(f"OCR agent nevrátil výsledek pro {plan.uuid}")
            raise RuntimeError("OCR agent nevrátil žádný použitelný výsledek.")
        self._log_debug(f"OCR agent dokončil stránku {plan.uuid}, délka {len(html)} znaků")
        plan.cached_reader = html
        return html

    def _prepare_agent_config(self, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        snapshot = options.get("snapshot")
        if isinstance(snapshot, dict) and snapshot:
            return json.loads(json.dumps(snapshot))
        config = options.get("config")
        if isinstance(config, dict) and config:
            return json.loads(json.dumps(config))
        name = options.get("name")
        if isinstance(name, str) and name.strip():
            collection = normalize_agent_collection(options.get("collection") or "correctors")
            agent = read_agent_file(name.strip(), collection)
            if agent:
                return json.loads(json.dumps(agent))
        return None

    def _build_agent_payload(self, options: Dict[str, Any], plan: PagePlan, collection: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "collection": collection,
            "name": options.get("name") or "",
            "model": options.get("model") or (options.get("snapshot") or {}).get("model"),
            "model_override": options.get("model") or (options.get("snapshot") or {}).get("model"),
            "book_uuid": self.params.book_uuid,
            "book_title": self.params.book_title or "",
            "page_uuid": plan.uuid,
            "page_number": plan.page_number or "",
            "page_index": plan.index,
            "page_title": plan.title or "",
            "language_hint": self.params.language_hint,
            "agent_snapshot": self._prepare_agent_config(options),
        }
        api_base = options.get("api_base") or self.params.api_base
        if api_base:
            payload["api_base"] = api_base
        for field in ("temperature", "top_p", "reasoning_effort", "response_format"):
            if field in options and options[field] is not None:
                payload[field] = options[field]
        return payload

    def _run_agent(self, agent_config: Dict[str, Any], payload: Dict[str, Any], label: str) -> Dict[str, Any]:
        if not agent_config:
            raise RuntimeError(f"Chybí konfigurace agenta pro {label}.")
        try:
            self._log_debug(f"Spouštím agent {label} ({payload.get('name')}) s modelem {payload.get('model')}")
            return run_agent_via_responses(agent_config, payload)
        except AgentRunnerError as exc:  # pragma: no cover - komunikace s API
            self._log_debug(f"Agent {label} selhal: {exc}")
            raise RuntimeError(f"{label}: {exc}") from exc

    def _extract_agent_html(self, result: Dict[str, Any]) -> str:
        text = str(result.get("text") or "").strip()
        document = self._parse_agent_document(text)
        if document:
            html = self._document_blocks_to_html(document)
            if html:
                return html
        return text

    def _apply_joiner(self, pages: Sequence[PageContent]) -> None:
        if len(pages) < 2:
            return
        joiner_options = self.params.joiner or {}
        if joiner_options.get("disabled"):
            return
        manual_mode = bool(joiner_options.get("manual"))
        agent_config = self._prepare_agent_config(joiner_options)
        for idx in range(len(pages) - 1):
            self._check_abort()
            current = pages[idx]
            nxt = pages[idx + 1]
            if nxt.plan.index - current.plan.index != 1:
                continue
            left_snippet = current.last_non_note_snippet()
            right_snippet = nxt.first_non_note_snippet()
            if not left_snippet or not right_snippet:
                continue
            joiner_label = f"[JOINER_DEBUG] pair {current.plan.index}->{nxt.plan.index}"
            if manual_mode or not agent_config:
                decision = self._manual_join_decision(left_snippet, right_snippet)
                self._log_debug(f"{joiner_label} manual decision={decision}")
            else:
                context = self._build_joiner_context(current, nxt, left_snippet, right_snippet)
                if not context:
                    continue
                payload = self._build_agent_payload(joiner_options, nxt.plan, collection="joiners")
                payload["stitch_context"] = context
                result = self._run_agent(agent_config, payload, "napojování stran")
                decision = self._parse_joiner_decision(str(result.get("text") or ""))
                self._log_debug(f"{joiner_label} agent decision={decision}")
            if decision in {"join", "merge"}:
                self._merge_page_pair(current, nxt, left_snippet, right_snippet)

    def _manual_join_decision(self, first: Snippet, second: Snippet) -> str:
        tag_a = (first.tag or "").strip().lower()
        tag_b = (second.tag or "").strip().lower()
        if tag_a and tag_b and tag_a != tag_b:
            return "split"
        text_a = first.text.strip()
        text_b = second.text.strip()
        if not text_a or not text_b:
            return "split"
        if SENTENCE_END_RE.search(text_a) and SENTENCE_START_RE.search(text_b):
            return "split"
        if text_a and text_a[-1] in HYPHENLIKE:
            return "merge"
        return "join"

    def _build_joiner_context(
        self,
        previous: PageContent,
        current: PageContent,
        left: Snippet,
        right: Snippet,
    ) -> Optional[Dict[str, Any]]:
        if not left.text.strip() or not right.text.strip():
            return None
        return {
            "meta": {
                "previous_uuid": previous.plan.uuid,
                "current_uuid": current.plan.uuid,
            },
            "pages": {
                "previous": self._build_page_info(previous.plan),
                "current": self._build_page_info(current.plan),
            },
            "sections": {
                "start": {
                    "previous": self._serialize_snippet(left, previous.plan),
                    "current": self._serialize_snippet(right, current.plan),
                    "merged": None,
                }
            },
        }

    def _merge_page_pair(self, previous: PageContent, current: PageContent, left: Snippet, right: Snippet) -> None:
        merged_text = self._merge_snippet_text(left.text, right.text)
        if not merged_text:
            return
        merged_html = self._render_snippet_html(left, right, merged_text)
        merged = Snippet(text=merged_text, html=merged_html, tag=left.tag or right.tag or "p")
        if previous.snippets:
            previous.snippets[-1] = merged
        if previous.blocks:
            previous.blocks[-1] = merged.html
        # Odeber právě ten blok, který byl použit pro merge (může to být až za note)
        rm_index = None
        for idx, snippet in enumerate(current.snippets):
            if snippet is right:
                rm_index = idx
                break
        if rm_index is None:
            # fallback: použij první nenote index
            for idx, snippet in enumerate(current.snippets):
                if snippet.tag != "note":
                    rm_index = idx
                    break
        if rm_index is None and current.snippets:
            rm_index = 0
        if rm_index is not None and rm_index < len(current.snippets):
            current.snippets.pop(rm_index)
            if rm_index < len(current.blocks):
                current.blocks.pop(rm_index)
        # Pokud nic nezůstalo, přidej prázdný blok/snippet, aby downstream kód nespadl
        if not current.blocks:
            current.blocks.append("")
        if not current.snippets:
            current.snippets.append(Snippet(text="", html="", tag="p"))

    def _merge_snippet_text(self, text_a: str, text_b: str) -> str:
        left = text_a.strip()
        right = text_b.strip()
        if not left:
            return right
        if not right:
            return left
        if left[-1] in HYPHENLIKE:
            trimmed = left.rstrip(HYPHENLIKE).rstrip()
            return f"{trimmed}{right.lstrip()}"
        if LEADING_PUNCT_RE.match(right):
            return f"{left.rstrip()}{right.rstrip()}"
        return f"{left.rstrip()} {right.lstrip()}"

    def _render_snippet_html(self, first: Snippet, second: Snippet, text: str) -> str:
        candidates = [first.html, second.html]
        for candidate in candidates:
            markup = candidate.strip()
            if not markup:
                continue
            match = re.match(r"<([a-zA-Z0-9:-]+)([^>]*)>", markup)
            if not match:
                continue
            tag = match.group(1)
            attrs = match.group(2).strip()
            attr_string = f" {attrs}" if attrs else ""
            return f"<{tag}{attr_string}>{escape(text)}</{tag}>"
        tag = first.tag or second.tag or "p"
        return f"<{tag}>{escape(text)}</{tag}>"

    def _serialize_snippet(self, snippet: Snippet, plan: PagePlan) -> Dict[str, Any]:
        return {
            "text": snippet.text,
            "html": snippet.html or escape(snippet.text),
            "tag": snippet.tag,
            "page_uuid": plan.uuid,
            "page_number": plan.page_number or "",
            "page_index": plan.index,
            "page_title": plan.title or "",
        }

    def _build_page_info(self, plan: PagePlan) -> Dict[str, Any]:
        return {
            "uuid": plan.uuid,
            "index": plan.index,
            "pageNumber": plan.page_number or "",
            "title": plan.title or "",
        }

    def _compose_document(self, pages: Sequence[PageContent]) -> str:
        sections: List[str] = []
        for content in pages:
            body = content.render_html().strip()
            label = content.plan.page_number or f"Strana {content.plan.index + 1}"
            sections.append(self._wrap_section(label, body))
        title = self.params.book_title or "Download"
        joined = "\n".join(sections)
        return f"""<!DOCTYPE html>
<html lang=\"cs\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
  </head>
  <body>
{joined}
  </body>
</html>
"""

    def _wrap_section(self, title: str, body: str) -> str:
        safe_body = body if body else ""
        return f'<section data-page="{escape(title)}">\n{safe_body}\n</section>'

    def _prepare_page_content(self, plan: PagePlan, html_text: str) -> Optional[PageContent]:
        normalized = (html_text or "").strip()
        if not normalized:
            self._log_debug(f"Stránka {plan.uuid} nemá žádný HTML obsah – ignoruji ji.")
            return None
        wrapper = BeautifulSoup(f"<div>{normalized}</div>", "html.parser").div
        blocks: List[str] = []
        snippets: List[Snippet] = []
        omit_small = bool(self.params.omit_small_text)
        omit_note = bool(self.params.omit_note_text)
        ignore_images = bool(getattr(self.params, "ignore_images", False))
        export_format = (self.params.export_format or "").lower()
        if wrapper:
            for child in wrapper.children:
                if not isinstance(child, Tag):
                    continue
                tag_name = (child.name or "").lower()
                if tag_name not in BLOCK_TAGS:
                    continue
                block_html = child.decode()
                if ignore_images and self._is_illustration_note(child):
                    continue
                if omit_small and self._is_small_markup(block_html, tag_name):
                    continue
                if omit_note and self._is_note_markup(block_html, tag_name):
                    # Pro EPUB necháme ilustrační note projít (převádí se na obrázek),
                    # pro ostatní formáty je odstranit.
                    if export_format != "epub" or not self._is_illustration_note(child):
                        continue
                text = child.get_text(" ", strip=True)
                if not text:
                    continue
                blocks.append(block_html)
                snippets.append(Snippet(text=text, html=block_html, tag=tag_name))
        if not blocks:
            if omit_small or omit_note:
                self._log_debug(f"Stránka {plan.uuid} po odfiltrování small/note nemá žádné bloky.")
            else:
                self._log_debug(f"Stránka {plan.uuid} po parsování nemá žádné bloky.")
            return None
            text = wrapper.get_text(" ", strip=True) if wrapper else normalized
            markup = normalized or (f"<p>{escape(text)}</p>" if text else "<p></p>")
            blocks = [markup]
            snippets = [Snippet(text=text or "", html=markup, tag="p")]
        return PageContent(plan=plan, blocks=blocks, snippets=snippets)

    @staticmethod
    def _is_small_markup(markup: str, tag_name: str) -> bool:
        if tag_name == "small":
            return True
        lower = markup.lower()
        if "<small" in lower:
            return True
        return False

    @staticmethod
    def _is_note_markup(markup: str, tag_name: str) -> bool:
        if tag_name == "note":
            return True
        lower = markup.lower()
        if "<note" in lower:
            return True
        if 'class="note"' in lower or "class='note'" in lower:
            return True
        return False

    @staticmethod
    def _is_illustration_note(tag: Tag) -> bool:
        if not tag or not isinstance(tag, Tag):
            return False
        if (tag.name or "").lower() != "note":
            return False
        cls = " ".join(tag.get("class") or [])
        if "illustration" in cls.lower():
            return True
        data_kind = (tag.get("data-kind") or tag.get("data-type") or "").lower()
        if data_kind == "illustration":
            return True
        return False

    def _convert_format(self, html_document: str) -> str:
        format_kind = self.params.export_format
        if format_kind == "html":
            return html_document
        if format_kind == "epub":
            raise RuntimeError("EPUB by měl být zpracován přes _build_epub.")
        soup = BeautifulSoup(html_document, "html.parser")
        body = soup.body or soup
        blocks: List[str] = []
        allowed = {"h1", "h2", "h3", "p", "blockquote", "small", "div", "note"}
        for element in body.find_all(allowed):
            # přeskočit vnořené povolené tagy (např. <small> uvnitř <p>)
            parent = element.parent
            skip = False
            while parent and parent != body:
                if parent.name and parent.name.lower() in allowed:
                    skip = True
                    break
                parent = parent.parent
            if skip:
                continue
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if format_kind == "md":
                blocks.append(self._block_to_markdown(element.name, text))
            else:
                blocks.append(text)
        plain = "\n\n".join(blocks) if blocks else soup.get_text("\n", strip=True)
        return plain.strip()

    @staticmethod
    def _block_to_markdown(tag: Optional[str], text: str) -> str:
        tag = (tag or "p").lower()
        if tag == "h1":
            return f"# {text}"
        if tag == "h2":
            return f"## {text}"
        if tag == "h3":
            return f"### {text}"
        if tag == "blockquote":
            return "> " + text
        return text

    def _epub_log(self, message: str) -> None:
        self._log_debug(f"{self._epub_log_tag} {message}")

    def _resolve_cover_uuid(self) -> Optional[str]:
        if self.params.cover_uuid:
            return self.params.cover_uuid
        priority = ["frontcover", "titlepage", "frontjacket", "cover", "frontendsheet"]
        for page in self.params.pages or []:
            page_type = (page.get("pageType") or "").strip().lower()
            if page_type in priority and page.get("uuid"):
                self._epub_log(f"Cover candidate selected by pageType '{page_type}': {page.get('uuid')}")
                return str(page.get("uuid"))
        if self.params.pages:
            first_uuid = self.params.pages[0].get("uuid")
            if first_uuid:
                self._epub_log(f"Cover fallback to first page: {first_uuid}")
                return str(first_uuid)
        return None

    def _fetch_cover_bytes(self, cover_uuid: str) -> Optional[bytes]:
        streams = ["IMG_FULL", "IMG_PREVIEW", "IMG_THUMB"]
        for base in self.processor._iter_api_bases(self.params.api_base):
            version = AltoProcessor._detect_api_version(base)
            pid = AltoProcessor._format_pid_for_version(cover_uuid, version)
            for stream in streams:
                self._check_abort()
                if not pid:
                    continue
                if version == "k7":
                    path = "image"
                    if stream == "IMG_THUMB":
                        path = "image/thumb"
                    elif stream == "IMG_PREVIEW":
                        path = "image/preview"
                    url = f"{base}/items/{pid}/{path}"
                else:
                    url = f"{base}/item/uuid:{pid}/streams/{stream}"
                try:
                    self._epub_log(f"Fetching cover stream {stream} from {base}")
                    response = self.processor.session.get(url, timeout=20)
                except Exception as exc:  # pragma: no cover - network defensive
                    self._epub_log(f"Cover fetch failed for {cover_uuid} stream {stream} base {base}: {exc}")
                    continue

                if response.status_code != 200 or not response.content:
                    response.close()
                    continue
                content_type = response.headers.get("Content-Type", "").lower()
                if "jp2" in content_type:
                    response.close()
                    continue
                data = response.content
                response.close()
                if data:
                    self._epub_log(
                        f"Cover stream {stream} from {base} OK "
                        f"(bytes={len(data)}, content_type={content_type or 'unknown'})"
                    )
                    return data
        return None

    def _attach_cover(self, book: epub.EpubBook) -> None:
        cover_uuid = self._resolve_cover_uuid()
        if not cover_uuid:
            return
        self._epub_log(f"Downloading cover for uuid={cover_uuid}")
        cover_bytes = self._fetch_cover_bytes(cover_uuid)
        if not cover_bytes:
            self._epub_log(f"Nepodařilo se stáhnout obálku pro {cover_uuid}")
            return
        try:
            book.set_cover("cover.jpg", cover_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            self._epub_log(f"Přidání obálky do EPUB selhalo: {exc}")

    def _fetch_image_bytes(self, image_uuid: str) -> Optional[bytes]:
        streams = ["IMG_FULL", "IMG_PREVIEW", "IMG_THUMB"]
        for base in self.processor._iter_api_bases(self.params.api_base):
            version = AltoProcessor._detect_api_version(base)
            pid = AltoProcessor._format_pid_for_version(image_uuid, version)
            for stream in streams:
                self._check_abort()
                if not pid:
                    continue
                if version == "k7":
                    path = "image"
                    if stream == "IMG_THUMB":
                        path = "image/thumb"
                    elif stream == "IMG_PREVIEW":
                        path = "image/preview"
                    url = f"{base}/items/{pid}/{path}"
                else:
                    url = f"{base}/item/uuid:{pid}/streams/{stream}"
                try:
                    self._epub_log(f"Fetching illustration {image_uuid} stream {stream} from {base}")
                    response = self.processor.session.get(url, timeout=20)
                except Exception as exc:  # pragma: no cover - network defensive
                    self._epub_log(f"Illustration fetch failed {image_uuid} stream {stream} base {base}: {exc}")
                    continue
                if response.status_code != 200 or not response.content:
                    response.close()
                    continue
                content_type = response.headers.get("Content-Type", "").lower()
                if "jp2" in content_type:
                    response.close()
                    continue
                data = response.content
                response.close()
                if data:
                    self._epub_log(
                        f"Illustration stream {stream} from {base} OK "
                        f"(bytes={len(data)}, content_type={content_type or 'unknown'})"
                    )
                    return data
        return None

    def _fetch_iiif_crop(
        self,
        page_uuid: str,
        hpos: float,
        vpos: float,
        width: float,
        height: float,
        page_width: int,
        page_height: int,
    ) -> Optional[bytes]:
        if not page_uuid or not page_width or not page_height:
            return None
        # Ověřit, že IIIF dává smysl (typicky k7)
        for base in self.processor._iter_api_bases(self.params.api_base):
            version = AltoProcessor._detect_api_version(base)
            if version != "k7":
                continue
            iiif_info_url = f"{self.processor.iiif_base_url}/uuid:{page_uuid}/info.json"
            try:
                self._epub_log(f"IIIF info {iiif_info_url}")
                resp = self.processor.session.get(iiif_info_url, timeout=15)
                resp.raise_for_status()
                info = resp.json()
            except Exception as exc:  # pragma: no cover - network defensive
                self._epub_log(f"IIIF info failed {page_uuid}: {exc}")
                continue
            img_w = float(info.get("width") or 0)
            img_h = float(info.get("height") or 0)
            if not img_w or not img_h:
                continue
            scale_x = img_w / page_width
            scale_y = img_h / page_height
            x = int(round(max(0.0, hpos) * scale_x))
            y = int(round(max(0.0, vpos) * scale_y))
            w = int(round(max(1.0, width) * scale_x))
            h = int(round(max(1.0, height) * scale_y))
            # clamp to image bounds
            if x + w > img_w:
                w = max(1, int(img_w) - x)
            if y + h > img_h:
                h = max(1, int(img_h) - y)
            max_side = max(w, h)
            size_part = "max"
            if max_side > 1600:
                size_part = f",{1600}"
            region = f"{x},{y},{w},{h}"
            iiif_image_url = f"{self.processor.iiif_base_url}/uuid:{page_uuid}/{region}/{size_part}/0/default.jpg"
            try:
                self._epub_log(f"IIIF crop {iiif_image_url}")
                img_resp = self.processor.session.get(iiif_image_url, timeout=20)
                img_resp.raise_for_status()
                data = img_resp.content
                if data:
                    return data
            except Exception as exc:  # pragma: no cover
                self._epub_log(f"IIIF crop failed {page_uuid}: {exc}")
                continue
        return None

    def _fetch_and_crop_image(
        self,
        page_uuid: str,
        hpos: float,
        vpos: float,
        width: float,
        height: float,
        page_width: int,
        page_height: int,
    ) -> Optional[bytes]:
        if not page_uuid or not page_width or not page_height:
            return None
        data = self._fetch_image_bytes(page_uuid)
        if not data:
            return None
        try:
            from PIL import Image
            import io

            with Image.open(io.BytesIO(data)) as img:
                scale_x = img.width / page_width if page_width else 1.0
                scale_y = img.height / page_height if page_height else 1.0
                x = int(round(max(0.0, hpos) * scale_x))
                y = int(round(max(0.0, vpos) * scale_y))
                w = int(round(max(1.0, width) * scale_x))
                h = int(round(max(1.0, height) * scale_y))
                # prevent overflow outside image bounds
                if x + w > img.width:
                    w = max(1, img.width - x)
                if y + h > img.height:
                    h = max(1, img.height - y)
                box = (x, y, x + w, y + h)
                cropped = img.crop(box)
                # Omezit velikost pro EPUB
                max_side = max(cropped.width, cropped.height)
                if max_side > 1600:
                    scale = 1600 / float(max_side)
                    new_size = (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale)))
                    cropped = cropped.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=85, optimize=True)
                return buf.getvalue()
        except Exception as exc:  # pragma: no cover - defensive
            self._epub_log(f"Local crop failed {page_uuid}: {exc}")
            return None
    def _process_illustrations(self, soup: BeautifulSoup, book: epub.EpubBook) -> None:
        if self.params.ignore_images:
            self._epub_log("Illustrations disabled by ignore_images flag.")
            return
        if not soup:
            return
        illustration_notes = []
        for note in soup.find_all("note"):
            if self._is_illustration_note(note):
                illustration_notes.append(note)
        if not illustration_notes:
            self._epub_log("No illustration notes found in document.")
            return
        else:
            self._epub_log(f"Nalezeno ilustrací: {len(illustration_notes)}")
        cache: Dict[tuple, bytes] = {}
        counter = 0
        for note in illustration_notes:
            image_uuid = (
                note.get("data-uuid")
                or note.get("data-image")
                or note.get("data-illustration")
                or note.get("data-id")
                or ""
            )
            image_uuid = str(image_uuid).strip()
            bbox_raw = str(note.get("data-bbox") or "").strip()
            page_width = int(note.get("data-page-width") or 0)
            page_height = int(note.get("data-page-height") or 0)
            if not page_width or not page_height:
                self._epub_log(f"Skipping illustration {image_uuid} – missing page dimensions w={page_width} h={page_height}")
                continue
            bbox_parts = bbox_raw.split(",") if bbox_raw else []
            if len(bbox_parts) != 4:
                self._epub_log(f"Skipping illustration without valid bbox: uuid={image_uuid} bbox='{bbox_raw}'")
                continue
            try:
                hpos, vpos, width_val, height_val = [float(part) for part in bbox_parts]
            except ValueError:
                self._epub_log(f"Skipping illustration due to bbox parse error: '{bbox_raw}'")
                continue
            if not image_uuid:
                self._epub_log("Skipping illustration note without uuid.")
                continue
            cache_key = (image_uuid, hpos, vpos, width_val, height_val, page_width, page_height)
            if cache_key in cache:
                data = cache[cache_key]
            else:
                data = self._fetch_iiif_crop(image_uuid, hpos, vpos, width_val, height_val, page_width, page_height)
                if data is None:
                    data = self._fetch_and_crop_image(image_uuid, hpos, vpos, width_val, height_val, page_width, page_height)
                if data:
                    cache[cache_key] = data
            if not data:
                if self.params.omit_note_text:
                    note.decompose()
                else:
                    note.string = "Ilustraci se nepodařilo stáhnout."
                    note["data-error"] = "fetch_failed"
                self._epub_log(f"Nepodařilo se stáhnout ilustraci uuid={image_uuid}")
                continue
            counter += 1
            filename = f"images/ill_{counter}.jpg"
            self._epub_log(f"Přidávám ilustraci {image_uuid} jako {filename}")
            image_item = epub.EpubItem(file_name=filename, media_type="image/jpeg", content=data)
            book.add_item(image_item)

            figure = soup.new_tag("figure")
            figure["class"] = ["illustration"]
            img = soup.new_tag("img", src=filename)
            alt_text = "Ilustrace"
            img["alt"] = alt_text
            figure.append(img)
            note.replace_with(figure)

    def _build_epub(self, html_document: str) -> str:
        soup = BeautifulSoup(html_document or "", "html.parser")
        body = soup.body or soup
        chapters: List[tuple[str, List[str]]] = []
        current_title: Optional[str] = None
        current_parts: List[str] = []

        def flush():
            if not current_parts:
                return
            title = current_title or f"Kapitola {len(chapters) + 1}"
            chapters.append((title, list(current_parts)))

        book = epub.EpubBook()
        book.set_title(self.params.book_title or "Export")
        book.set_language((self.params.language_hint or "cs")[:5])
        identifier = self.params.book_uuid or self.params.current_page_uuid or uuid.uuid4().hex
        book.set_identifier(identifier)
        if self.params.authors:
            self._epub_log(f"EPUB authors: {', '.join(self.params.authors)}")
        for author in self.params.authors or []:
            book.add_author(author)
            book.add_metadata("DC", "creator", author)

        self._attach_cover(book)
        allowed_tags = {"h1", "h2", "h3", "p", "blockquote", "small", "div", "note", "figure", "img"}
        self._process_illustrations(soup, book)
        for element in body.find_all(allowed_tags, recursive=True):
            tag_name = (element.name or "").lower()
            # Přeskočit obrázek, který je uvnitř figure, aby nevznikly duplikáty
            if tag_name == "img" and element.find_parent("figure"):
                continue
            # Přeskočit vnořené povolené tagy (např. <small> uvnitř <p>)
            parent = element.parent
            skip = False
            while parent and parent != body:
                if parent.name and parent.name.lower() in allowed_tags and parent.name.lower() != "section":
                    skip = True
                    break
                parent = parent.parent
            if skip:
                continue
            html_chunk = element.decode()
            if tag_name in {"h1", "h2", "h3"}:
                flush()
                current_title = element.get_text(" ", strip=True) or f"Kapitola {len(chapters) + 1}"
                current_parts = [html_chunk]
            else:
                current_parts.append(html_chunk)
        flush()

        if not chapters:
            content = body.decode() if body else ""
            chapters = [("Kapitola 1", [content])]

        epub_chapters = []
        for idx, (title, parts) in enumerate(chapters, start=1):
            chapter = epub.EpubHtml(
                title=title or f"Kapitola {idx}",
                file_name=f"chap_{idx}.xhtml",
                lang=(self.params.language_hint or "cs")[:5],
            )
            content = "\n".join(parts)
            chapter.set_content(f"<html><head><meta charset='utf-8'></head><body>{content}</body></html>")
            book.add_item(chapter)
            epub_chapters.append(chapter)

        if epub_chapters:
            book.toc = epub_chapters
            book.spine = ["nav"] + epub_chapters
        else:
            book.spine = ["nav"]

        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        fd, path = tempfile.mkstemp(prefix="alto_export_", suffix=".epub")
        os.close(fd)
        epub.write_epub(path, book)
        return path

    def _parse_agent_document(self, text: str) -> Optional[Dict[str, Any]]:
        trimmed = text.strip()
        if not trimmed:
            return None
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            repaired = self._repair_json_newlines(trimmed)
            if not repaired:
                return None
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    def _document_blocks_to_html(self, document: Dict[str, Any]) -> Optional[str]:
        blocks = document.get("blocks") if isinstance(document, dict) else None
        if not isinstance(blocks, list):
            return None
        parts: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            normalized = text.strip()
            if not normalized:
                continue
            block_type = (block.get("type") or "p").lower()
            tag = "p"
            attrs = ""
            if block_type in {"h1", "h2", "h3"}:
                tag = block_type
            elif block_type == "small":
                parts.append(f"<p><small>{escape(normalized)}</small></p>")
                continue
            elif block_type == "note":
                tag = "note"
                attrs = ' style="display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;"'
            elif block_type == "centered":
                tag = "div"
                attrs = ' class="centered"'
            elif block_type == "blockquote":
                tag = "blockquote"
            parts.append(f"<{tag}{attrs}>{escape(normalized)}</{tag}>")
        return "".join(parts) if parts else None

    def _repair_json_newlines(self, raw: str) -> str:
        fixed_chars: List[str] = []
        in_string = False
        escape_next = False
        for ch in raw:
            code = ord(ch)
            if escape_next:
                fixed_chars.append(ch)
                escape_next = False
                continue
            if ch == "\\":
                fixed_chars.append(ch)
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                fixed_chars.append(ch)
                continue
            if in_string and code in {10, 13, 0x2028, 0x2029}:
                fixed_chars.append("\\n")
                continue
            fixed_chars.append(ch)
        return "".join(fixed_chars)

    def _parse_joiner_decision(self, raw_text: str) -> str:
        normalized = raw_text.strip().lower()
        if normalized in {"join", "merge", "split"}:
            return normalized
        if normalized in {"0", "1", "2"}:
            return {"0": "split", "1": "join", "2": "merge"}[normalized]
        payload = self._parse_agent_document(raw_text)
        if not payload:
            return "split"
        pairs = payload.get("pairs") if isinstance(payload, dict) else None
        if isinstance(pairs, dict):
            start = pairs.get("start")
        else:
            start = payload.get("start") if isinstance(payload, dict) else None
        if isinstance(start, dict):
            action = start.get("action") or start.get("decision")
            if isinstance(action, str):
                mapped = action.strip().lower()
                if mapped in {"join", "merge", "split"}:
                    return mapped
                if mapped in {"0", "1", "2"}:
                    return {"0": "split", "1": "join", "2": "merge"}[mapped]
        decision = payload.get("decision") if isinstance(payload, dict) else None
        if isinstance(decision, dict):
            start_value = decision.get("start")
            if isinstance(start_value, str) and start_value in {"0", "1", "2"}:
                return {"0": "split", "1": "join", "2": "merge"}[start_value]
        return "split"

    def _select_pages(self) -> List[PagePlan]:
        source_pages = self.params.pages or []
        plans: List[PagePlan] = []
        for idx, entry in enumerate(source_pages):
            page = entry or {}
            uuid = page.get("uuid")
            if not uuid:
                continue
            page_index = page.get("index")
            if not isinstance(page_index, int):
                page_index = idx
            plans.append(
                PagePlan(
                    uuid=str(uuid),
                    index=page_index,
                    page_number=page.get("pageNumber") or page.get("pagenumber"),
                    title=page.get("title"),
                    cached_python=self._read_cached_text(page, ["python", "pythonHtml", "cachedPython"]),
                    cached_llm=self._read_cached_text(page, ["llm", "corrected", "cachedLlm"]),
                    cached_reader=self._read_cached_text(page, ["ocr", "reader", "cachedReader"]),
                    data=dict(page),
                )
            )
        if not plans:
            return []
        if self.params.range_mode == "current":
            selected = [plan for plan in plans if plan.uuid == self.params.current_page_uuid]
            return selected or [plans[0]]
        if self.params.range_mode == "custom":
            indices = self._parse_custom_range(plans)
            if not indices:
                raise ValueError("Zadaný rozsah stránek není platný.")
            return [plans[i] for i in indices]
        return plans

    def _read_cached_text(self, page: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            value = page.get(key)
            if isinstance(value, str) and value.strip():
                return value
        cached = page.get("cached") if isinstance(page, dict) else None
        if isinstance(cached, dict):
            for key in keys:
                value = cached.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _parse_custom_range(self, plans: Sequence[PagePlan]) -> List[int]:
        total = len(plans)
        expr = (self.params.range_value or "").strip()
        if not expr:
            return []
        label_map = self._build_page_label_map(plans)
        indexes: List[int] = []
        for part in expr.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start_idx = self._resolve_page_selector(left.strip(), plans, label_map, default_index=0)
                end_idx = self._resolve_page_selector(right.strip(), plans, label_map, default_index=max(total - 1, 0), prefer_after=start_idx)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                indexes.extend(range(start_idx, end_idx + 1))
            else:
                idx = self._resolve_page_selector(part, plans, label_map, default_index=0)
                indexes.append(idx)
        seen = set()
        ordered: List[int] = []
        for idx in indexes:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered

    def _build_filename(self) -> str:
        if self.params.output_filename:
            return self.params.output_filename
        title = (self.params.book_title or "export").strip() or "export"
        safe_title = "_".join(title.split())
        return f"{safe_title}-{self.params.source}.{self.params.export_format}"

    def _build_page_label_map(self, plans: Sequence[PagePlan]) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for idx, plan in enumerate(plans):
            normalized = self._normalize_page_label(plan.page_number)
            if normalized:
                mapping.setdefault(normalized, []).append(idx)
        return mapping

    def _normalize_page_label(self, value: Optional[str]) -> str:
        if not value:
            return ""
        cleaned = value.strip().strip("[](){}").strip()
        cleaned = cleaned.replace(" ", "").lower()
        return cleaned

    def _resolve_page_selector(
        self,
        token: str,
        plans: Sequence[PagePlan],
        label_map: Dict[str, List[int]],
        default_index: int,
        prefer_after: Optional[int] = None,
    ) -> int:
        total = len(plans)
        if total == 0:
            raise ValueError("Výběr stránek není k dispozici.")
        if not token:
            return max(0, min(default_index, total - 1))
        normalized = self._normalize_page_label(token)
        if normalized:
            candidates = label_map.get(normalized)
            if candidates:
                if prefer_after is not None:
                    for idx in candidates:
                        if idx >= prefer_after:
                            return idx
                return candidates[-1] if prefer_after is not None else candidates[0]
        try:
            order = int(token)
        except ValueError:
            pass
        else:
            idx = order - 1
            if 0 <= idx < total:
                return idx
        raise ValueError(f"Stránku '{token or ' '}' se nepodařilo najít.")


def run_export_job(job: ExportJob) -> tuple[str, str]:
    builder = ExportBuilder(job)
    return builder.build()
