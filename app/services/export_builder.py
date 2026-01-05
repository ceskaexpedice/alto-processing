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
SENTENCE_END_RE = re.compile(r"[.!?…»\]\)]\s*$")
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
            left_snippet = current.last_snippet()
            right_snippet = nxt.first_snippet()
            if not left_snippet or not right_snippet:
                continue
            if manual_mode or not agent_config:
                decision = self._manual_join_decision(left_snippet, right_snippet)
            else:
                context = self._build_joiner_context(current, nxt, left_snippet, right_snippet)
                if not context:
                    continue
                payload = self._build_agent_payload(joiner_options, nxt.plan, collection="joiners")
                payload["stitch_context"] = context
                result = self._run_agent(agent_config, payload, "napojování stran")
                decision = self._parse_joiner_decision(str(result.get("text") or ""))
            if decision in {"join", "merge"}:
                self._merge_page_pair(current, nxt, left_snippet, right_snippet)

    def _manual_join_decision(self, first: Snippet, second: Snippet) -> str:
        text_a = first.text.strip()
        text_b = second.text.strip()
        if not text_a or not text_b:
            return "split"
        if SENTENCE_END_RE.search(text_a):
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
        if current.snippets:
            current.snippets.pop(0)
        if current.blocks:
            current.blocks.pop(0)
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
        if wrapper:
            for child in wrapper.children:
                if not isinstance(child, Tag):
                    continue
                tag_name = (child.name or "").lower()
                if tag_name not in BLOCK_TAGS:
                    continue
                block_html = child.decode()
                if omit_small and self._is_small_markup(block_html, tag_name):
                    continue
                if omit_note and self._is_note_markup(block_html, tag_name):
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
        if tag_name in {"small", "note"}:
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

    def _convert_format(self, html_document: str) -> str:
        format_kind = self.params.export_format
        if format_kind == "html":
            return html_document
        if format_kind == "epub":
            raise RuntimeError("EPUB by měl být zpracován přes _build_epub.")
        soup = BeautifulSoup(html_document, "html.parser")
        blocks = []
        for element in soup.find_all(["h1", "h2", "h3", "p", "blockquote", "small", "div", "note"]):
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

        allowed_tags = {"h1", "h2", "h3", "p", "blockquote", "small", "div", "note"}
        for element in body.find_all(allowed_tags, recursive=True):
            tag_name = (element.name or "").lower()
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

        book = epub.EpubBook()
        book.set_title(self.params.book_title or "Export")
        book.set_language((self.params.language_hint or "cs")[:5])
        identifier = self.params.book_uuid or self.params.current_page_uuid or uuid.uuid4().hex
        book.set_identifier(identifier)

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
