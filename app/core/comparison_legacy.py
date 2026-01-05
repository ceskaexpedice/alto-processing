#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webový server pro porovnání původního TypeScript a nového Python zpracování ALTO
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import json
import requests
import subprocess
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import html
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Import původního procesoru
from .main_processor import AltoProcessor, DEFAULT_API_BASES
from .agent_runner import (
    AgentRunnerError,
    run_agent as run_agent_via_responses,
    DEFAULT_MODEL,
    REASONING_PREFIXES,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TS_DIST_PATH = ROOT_DIR / 'dist' / 'run_original.js'
MODEL_REGISTRY_PATH = ROOT_DIR / 'config' / 'models.json'
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 1200
DEFAULT_AGENT_PROMPT_TEXT = (
    "Jsi pečlivý korektor češtiny. Dostaneš JSON s klíči "
    "\"language_hint\" a \"blocks\". Blocks je pole objektů {id, type, text}. "
    "Oprav překlepy a zjevné OCR chyby pouze v hodnotách \"text\". Nesjednocuj styl, "
    "neměň typy bloků ani jejich pořadí. Zachovej diakritiku, pokud lze. "
    "Odpovídej pouze validním JSON se stejnou strukturou a klíči jako vstup."
)

# Style value reused for <note> blocks in agent diff rendering.
NOTE_STYLE_ATTR = 'display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;'

try:
    _model_registry_raw = json.loads(MODEL_REGISTRY_PATH.read_text(encoding='utf-8'))
    MODEL_REGISTRY = _model_registry_raw if isinstance(_model_registry_raw, dict) else {}
except FileNotFoundError:
    MODEL_REGISTRY = {}
except json.JSONDecodeError as exc:  # pragma: no cover - invalid config should surface loudly
    print(f"[ModelRegistry] Nelze načíst soubor {MODEL_REGISTRY_PATH}: {exc}")
    MODEL_REGISTRY = {}
if 'models' not in MODEL_REGISTRY or not isinstance(MODEL_REGISTRY['models'], list):
    MODEL_REGISTRY['models'] = []
if 'default_model' not in MODEL_REGISTRY or not isinstance(MODEL_REGISTRY['default_model'], str):
    MODEL_REGISTRY['default_model'] = DEFAULT_MODEL
MODEL_REGISTRY_JSON = json.dumps(MODEL_REGISTRY, ensure_ascii=False)

KNOWN_LIBRARY_OVERRIDES: Dict[str, Dict[str, str]] = {
    "https://api.kramerius.mzk.cz/search/api/client/v7.0": {
        "code": "mzk",
        "label": "Moravská zemská knihovna v Brně",
        "handle_base": "https://kramerius.mzk.cz/search",
    },
    "https://kramerius.mzk.cz/search/api/v5.0": {
        "code": "mzk",
        "label": "Moravská zemská knihovna v Brně",
        "handle_base": "https://kramerius.mzk.cz/search",
    },
    "https://kramerius5.nkp.cz/search/api/v5.0": {
        "code": "nkp",
        "label": "Národní knihovna České republiky",
        "handle_base": "https://kramerius5.nkp.cz/search",
    },
}

DIFF_MODE_WORD = 'word'
DIFF_MODE_CHAR = 'char'
DIFF_MODE_NONE = 'none'

_VOID_HTML_ELEMENTS = {
    'br', 'hr', 'img', 'input', 'meta', 'link', 'source', 'track',
    'area', 'col', 'embed', 'param', 'wbr', 'base'
}
_TAG_REGEX = re.compile(r'<[^>]+?>')
_TAG_NAME_REGEX = re.compile(r'^<\s*/?\s*([a-zA-Z0-9:-]+)')
_WORD_SPLIT_REGEX = re.compile(r'(\s+|[^\w]+)', re.UNICODE)
_BLOCK_TEXT_STRIPPER = re.compile(r'<[^>]+>')


@dataclass(frozen=True)
class HtmlToken:
    kind: str  # 'tag' or 'text'
    value: str
    is_whitespace: bool = False


def _normalize_diff_mode(mode: Optional[str]) -> str:
    if mode == DIFF_MODE_CHAR:
        return DIFF_MODE_CHAR
    return DIFF_MODE_WORD


def _extract_tag_name(tag_text: str) -> str:
    match = _TAG_NAME_REGEX.match(tag_text)
    return match.group(1).lower() if match else ''


def split_html_blocks(html_content: str) -> List[str]:
    if not html_content:
        return []

    blocks: List[str] = []
    depth = 0
    block_start: Optional[int] = None
    last_index = 0

    for match in _TAG_REGEX.finditer(html_content):
        start, end = match.span()
        tag_text = match.group(0)

        if depth == 0 and start > last_index:
            segment = html_content[last_index:start]
            if segment:
                blocks.append(segment)

        tag_name = _extract_tag_name(tag_text)
        is_closing = tag_text.startswith('</')
        is_self_closing = tag_text.endswith('/>') or tag_name in _VOID_HTML_ELEMENTS

        if not is_closing and not is_self_closing:
            if depth == 0:
                block_start = start
            depth += 1
        elif is_self_closing:
            if depth == 0:
                blocks.append(tag_text)
        else:
            depth = max(depth - 1, 0)
            if depth == 0 and block_start is not None:
                blocks.append(html_content[block_start:end])
                block_start = None

        last_index = end

    if block_start is not None and block_start < len(html_content):
        blocks.append(html_content[block_start:])
    elif last_index < len(html_content):
        remainder = html_content[last_index:]
        if remainder:
            blocks.append(remainder)

    if not blocks and html_content:
        return [html_content]
    return blocks


def split_text_tokens(text: str, mode: str) -> List[HtmlToken]:
    if not text:
        return []

    if mode == DIFF_MODE_CHAR:
        return [
            HtmlToken('text', ch, ch.isspace())
            for ch in text
        ]

    tokens: List[HtmlToken] = []
    last_index = 0
    for match in _WORD_SPLIT_REGEX.finditer(text):
        start, end = match.span()
        if start > last_index:
            chunk = text[last_index:start]
            tokens.append(HtmlToken('text', chunk, chunk.isspace()))
        chunk = match.group(0)
        if chunk:
            tokens.append(HtmlToken('text', chunk, chunk.isspace()))
        last_index = end
    if last_index < len(text):
        chunk = text[last_index:]
        tokens.append(HtmlToken('text', chunk, chunk.isspace()))
    return tokens


def tokenize_block(block_html: str, mode: str) -> List[HtmlToken]:
    if not block_html:
        return []

    tokens: List[HtmlToken] = []
    last_index = 0
    for match in _TAG_REGEX.finditer(block_html):
        start, end = match.span()
        if start > last_index:
            tokens.extend(split_text_tokens(block_html[last_index:start], mode))
        tokens.append(HtmlToken('tag', match.group(0), False))
        last_index = end
    if last_index < len(block_html):
        tokens.extend(split_text_tokens(block_html[last_index:], mode))
    return tokens


def render_tokens(tokens: List[HtmlToken], highlight: Optional[str] = None) -> str:
    parts: List[str] = []
    for token in tokens:
        if token.kind == 'tag':
            parts.append(token.value)
            continue
        if highlight in ('added', 'removed') and not token.is_whitespace and token.value:
            parts.append(f'<span class="diff-{highlight}">{token.value}</span>')
        else:
            parts.append(token.value)
    return ''.join(parts)


def diff_block_content(left_block: str, right_block: str, mode: str) -> Tuple[str, str]:
    mode_normalized = _normalize_diff_mode(mode)
    left_tokens = tokenize_block(left_block, mode_normalized)
    right_tokens = tokenize_block(right_block, mode_normalized)
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)

    left_parts: List[str] = []
    right_parts: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_slice = left_tokens[i1:i2]
        right_slice = right_tokens[j1:j2]

        if tag == 'equal':
            left_parts.append(render_tokens(left_slice))
            right_parts.append(render_tokens(right_slice))
        elif tag == 'delete':
            left_parts.append(render_tokens(left_slice, 'removed'))
        elif tag == 'insert':
            right_parts.append(render_tokens(right_slice, 'added'))
        elif tag == 'replace':
            left_parts.append(render_tokens(left_slice, 'removed'))
            right_parts.append(render_tokens(right_slice, 'added'))

    return ''.join(left_parts), ''.join(right_parts)


def wrap_block(content: str, change: str) -> str:
    classes = ['diff-block']
    if change == 'added':
        classes.append('diff-block--added')
    elif change == 'removed':
        classes.append('diff-block--removed')
    elif change == 'changed':
        classes.append('diff-block--changed')
    class_attr = ' '.join(classes)
    data_attr = f' data-diff-change="{change}"' if change != 'equal' else ''
    return f'<div class="{class_attr}"{data_attr}>{content}</div>'


def _normalize_block_text(block_html: str) -> str:
    if not block_html:
        return ''
    stripped = _BLOCK_TEXT_STRIPPER.sub(' ', block_html)
    stripped = re.sub(r'\s+', ' ', stripped).strip().lower()
    return stripped

def _block_similarity(a_html: str, b_html: str) -> float:
    """Similarity of normalized text content (0..1)."""
    a = _normalize_block_text(a_html)
    b = _normalize_block_text(b_html)
    if not a and not b:
        return 1.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b, autojunk=False).ratio()

def _is_noise_block(block_html: str) -> bool:
    """
    Heuristika: krátké útržky / převážně interpunkce / římské číslice / odrážky / čísla stran.
    Příklad: '• • • I V /V 55'
    """
    t = _normalize_block_text(block_html)
    if not t:
        return True
    if len(t) <= 6:
        return True
    if re.fullmatch(r'[•\sIVXLCDM/0-9.,\-]+', t):
        return True
    punct = sum(ch in '•.,/ -' for ch in t)
    if punct / max(1, len(t)) > 0.7:
        return True
    return False

def _merge_html(block_list: List[str]) -> str:
    """Join adjacent blocks for matching (used in split/merge)."""
    return ''.join(block_list)

# Helper to build a stable key from tag+normalized text for alignment
def _block_key(block_html: str) -> str:
    """Build a stable alignment key from the first tag name + normalized text."""
    if not block_html:
        return ''
    tag = _extract_tag_name(block_html)
    text_key = _normalize_block_text(block_html)
    return f"{tag}|{text_key}"


def _align_block_slices(left_blocks: List[str], right_blocks: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    if not left_blocks and not right_blocks:
        return []

    i, j = 0, 0
    out: List[Tuple[Optional[str], Optional[str]]] = []

    # prahy laditelné podle dat
    THRESH_STRICT = 0.60
    THRESH_LOOSE = 0.50
    MAX_MERGE = 2  # zkoušíme sloučit až 2 bloky na každé straně

    nL, nR = len(left_blocks), len(right_blocks)

    def best_candidate(i0: int, j0: int) -> Tuple[float, int, int, str]:
        """
        Vrátí (score, li, rj, level) pro nejlepší z kombinací
        li in {1..MAX_MERGE}, rj in {1..MAX_MERGE}.
        level je 'strict' nebo 'loose' podle prahu.
        """
        best = (0.0, 1, 1, 'loose')
        for li in range(1, min(MAX_MERGE, nL - i0) + 1):
            left_merged = _merge_html(left_blocks[i0:i0+li])
            # noise na levé straně nepáruj agresivně
            if li == 1 and _is_noise_block(left_merged):
                continue
            for rj in range(1, min(MAX_MERGE, nR - j0) + 1):
                right_merged = _merge_html(right_blocks[j0:j0+rj])
                if rj == 1 and _is_noise_block(right_merged):
                    continue
                s = _block_similarity(left_merged, right_merged)
                level = 'strict' if s >= THRESH_STRICT else ('loose' if s >= THRESH_LOOSE else '')
                if level and s > best[0]:
                    best = (s, li, rj, level)
        return best

    while i < nL or j < nR:
        # vyčerpání jedné strany
        if i >= nL:
            out.append((None, right_blocks[j]))
            j += 1
            continue
        if j >= nR:
            out.append((left_blocks[i], None))
            i += 1
            continue

        # odpad (šum) – rovnou zahodit na své straně
        if _is_noise_block(left_blocks[i]):
            out.append((left_blocks[i], None))
            i += 1
            continue
        if _is_noise_block(right_blocks[j]):
            out.append((None, right_blocks[j]))
            j += 1
            continue

        score, li, rj, level = best_candidate(i, j)

        if level:  # máme match
            left_merged = _merge_html(left_blocks[i:i+li])
            right_merged = _merge_html(right_blocks[j:j+rj])
            out.append((left_merged, right_merged))
            i += li
            j += rj
            continue

        # žádný rozumný match – rozhodni, co odhodit (lookahead do 2)
        # 1) jak dobře se aktuální left[i] hodí k některému z right[j:j+2]
        best_right = 0.0
        for rj2 in range(1, min(2, nR - j) + 1):
            s = _block_similarity(left_blocks[i], _merge_html(right_blocks[j:j+rj2]))
            if s > best_right:
                best_right = s
        # 2) jak dobře se aktuální right[j] hodí k některému z left[i:i+2]
        best_left = 0.0
        for li2 in range(1, min(2, nL - i) + 1):
            s = _block_similarity(_merge_html(left_blocks[i:i+li2]), right_blocks[j])
            if s > best_left:
                best_left = s

        # odhoď stranu s horší vyhlídkou, šum preferenčně odhazuj
        if best_right < best_left:
            out.append((left_blocks[i], None))
            i += 1
        elif best_left < best_right:
            out.append((None, right_blocks[j]))
            j += 1
        else:
            # pat – odhoď kratší/„šumovější“ blok
            li_txt = _normalize_block_text(left_blocks[i])
            rj_txt = _normalize_block_text(right_blocks[j])
            if _is_noise_block(left_blocks[i]) or len(li_txt) <= len(rj_txt):
                out.append((left_blocks[i], None))
                i += 1
            else:
                out.append((None, right_blocks[j]))
                j += 1

    return out


def wrap_diff_content(blocks: List[str], mode: str) -> str:
    mode_attr = f' data-diff-mode="{mode}"' if mode in {DIFF_MODE_WORD, DIFF_MODE_CHAR} else ''
    inner = ''.join(blocks)
    if not inner:
        inner = '<div class="diff-placeholder">Bez obsahu</div>'
    return f'<div class="diff-content diff-html"{mode_attr}>{inner}</div>'


def _annotate_block_classes(block_html: str, change: str) -> str:
    if not block_html or not change or change == 'equal':
        return block_html
    class_map = {
        'added': 'diff-block diff-block--added',
        'removed': 'diff-block diff-block--removed',
        'changed': 'diff-block diff-block--changed',
    }
    class_name = class_map.get(change)
    if not class_name:
        return block_html
    stripped = block_html.lstrip()
    prefix_len = len(block_html) - len(stripped)
    prefix = block_html[:prefix_len]
    target = stripped
    leading_inline = ''
    while target.lower().startswith('<br'):
        end_idx = target.find('>')
        if end_idx == -1:
            break
        leading_inline += target[:end_idx + 1]
        target = target[end_idx + 1:]
    if not target:
        return prefix + leading_inline
    match = re.match(r'<([a-zA-Z0-9:-]+)([^>]*)>', target)
    if match:
        tag_name = match.group(1)
        attrs = match.group(2) or ''
        if 'class=' in attrs:
            updated = re.sub(r'class="([^"]*)"', lambda m: f'class="{m.group(1)} {class_name}"', target, count=1)
            return prefix + leading_inline + updated
        else:
            separator = ''
            if not attrs or not attrs.endswith(' '):
                separator = ' '
            insertion = f'<{tag_name}{attrs}{separator}class="{class_name}">'
            updated = target.replace(match.group(0), insertion, 1)
            return prefix + leading_inline + updated
    remaining = block_html[prefix_len + len(leading_inline):]


def build_html_diff(python_html: str, ts_html: str, mode: str) -> Dict[str, str]:
    mode_normalized = _normalize_diff_mode(mode)
    python_blocks = split_html_blocks(python_html or '')
    ts_blocks = split_html_blocks(ts_html or '')

    matcher = SequenceMatcher(None, python_blocks, ts_blocks, autojunk=False)

    python_render: List[str] = []
    ts_render: List[str] = []
    changes_detected = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        py_slice = python_blocks[i1:i2]
        ts_slice = ts_blocks[j1:j2]

        if tag == 'equal':
            for block in py_slice:
                python_render.append(wrap_block(block, 'equal'))
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'equal'))
            continue

        changes_detected = True

        if tag == 'delete':
            for block in py_slice:
                python_render.append(wrap_block(block, 'removed'))
                continue
        if tag == 'insert':
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'added'))
            continue

        if tag == 'replace':
            aligned_pairs = _align_block_slices(py_slice, ts_slice)
            for left_block, right_block in aligned_pairs:
                if left_block and right_block:
                    left_markup, right_markup = diff_block_content(left_block, right_block, mode_normalized)
                    python_render.append(wrap_block(left_markup, 'changed'))
                    ts_render.append(wrap_block(right_markup, 'changed'))
                elif left_block:
                    python_render.append(wrap_block(left_block, 'removed'))
                elif right_block:
                    ts_render.append(wrap_block(right_block, 'added'))
        else:
            for block in py_slice:
                python_render.append(wrap_block(block, 'removed'))
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'added'))

    return {
        'python': wrap_diff_content(python_render, mode_normalized),
        'typescript': wrap_diff_content(ts_render, mode_normalized),
        'has_changes': changes_detected,
        'mode': mode_normalized,
    }


def block_to_html_from_dict(block: dict) -> Optional[str]:
    if not isinstance(block, dict):
        return None
    text = block.get('text')
    if text is None:
        return None
    text_str = str(text).strip()
    if not text_str:
        return None

    block_type = str(block.get('type') or '').lower()
    escaped_text = html.escape(text_str, quote=False)

    tag = 'p'
    attrs = {}
    classes: List[str] = []
    style_attr = None
    wrap_with_small = False
    prefix = ''

    block_id = block.get('id')
    if block_id not in (None, ''):
        attrs['data-block-id'] = str(block_id)

    if block_type in {'h1', 'h2', 'h3'}:
        tag = block_type
    elif block_type == 'small':
        wrap_with_small = True
    elif block_type == 'note':
        tag = 'note'
        style_attr = NOTE_STYLE_ATTR
        prefix = '<br>'
    elif block_type == 'centered':
        tag = 'div'
        classes.append('centered')
    elif block_type == 'blockquote':
        tag = 'blockquote'
    elif block_type == 'li':
        tag = 'p'
        attrs['data-block-type'] = 'li'
    else:
        tag = 'p'
        if block_type and block_type != 'p':
            attrs['data-block-type'] = block_type

    if classes:
        attrs['class'] = ' '.join(classes)
    if style_attr:
        attrs['style'] = style_attr

    attr_parts = []
    for key, value in attrs.items():
        attr_parts.append(f'{key}="{html.escape(str(value), quote=False)}"')
    attr_str = f" {' '.join(attr_parts)}" if attr_parts else ''

    if wrap_with_small:
        markup = f'<p{attr_str}><small>{escaped_text}</small></p>'
    else:
        markup = f'<{tag}{attr_str}>{escaped_text}</{tag}>'
        if tag == 'note':
            markup = prefix + markup
    return markup


def build_agent_diff(original_html: str, corrected_html: str, mode: str) -> Dict[str, str]:
    mode_normalized = _normalize_diff_mode(mode)

    def parse_blocks(html_text: str) -> List[str]:
        try:
            data = json.loads(html_text)
            if isinstance(data, dict) and isinstance(data.get('blocks'), list):
                output = []
                for block in data['blocks']:
                    markup = block_to_html_from_dict(block)
                    if markup:
                        output.append((block.get('id'), markup))
                if output:
                    return output
        except Exception:
            pass
        tokens = split_html_blocks(html_text or '')
        fallback: List[Tuple[Optional[str], str]] = []
        for token in tokens:
            if not token:
                continue
            block_id = None
            match = re.search(r'data-block-id="([^"]+)"', token)
            if match:
                block_id = match.group(1)
            fallback.append((block_id, token))
        return fallback

    original_pairs = parse_blocks(original_html or '')
    corrected_pairs = parse_blocks(corrected_html or '')

    original_map = {bid: markup for bid, markup in original_pairs if bid}
    corrected_map = {bid: markup for bid, markup in corrected_pairs if bid}

    processed_ids = set()
    original_render: List[str] = []
    corrected_render: List[str] = []
    changes_detected = False

    for bid, markup in original_pairs:
        if bid and bid in corrected_map:
            corrected_markup = corrected_map[bid]
            if markup == corrected_markup:
                original_render.append(markup)
                corrected_render.append(corrected_markup)
            else:
                changes_detected = True
                left_markup, right_markup = diff_block_content(markup, corrected_markup, mode_normalized)
                original_render.append(_annotate_block_classes(left_markup, 'changed'))
                corrected_render.append(_annotate_block_classes(right_markup, 'changed'))
            processed_ids.add(bid)
        else:
            changes_detected = True
            original_render.append(_annotate_block_classes(markup, 'removed'))

    for bid, markup in corrected_pairs:
        if bid and bid in processed_ids:
            continue
        if bid and bid in original_map:
            continue
        changes_detected = True
        corrected_render.append(_annotate_block_classes(markup, 'added'))

    return {
        'original': ''.join(original_render),
        'corrected': ''.join(corrected_render),
        'has_changes': changes_detected,
        'mode': mode_normalized,
    }


# Agents storage/helpers
AGENTS_DIR = ROOT_DIR / 'agents'
DEFAULT_AGENT_COLLECTION = 'correctors'
AGENT_COLLECTIONS = {
    'correctors': AGENTS_DIR / 'correctors',
    'joiners': AGENTS_DIR / 'joiners',
    'readers': AGENTS_DIR / 'readers',
    'custom_lmm': AGENTS_DIR / 'custom_lmm',
}
AGENT_NAME_RE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')

def ensure_agents_dir():
    try:
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        for path in AGENT_COLLECTIONS.values():
            path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def normalize_agent_collection(collection: Optional[str]) -> str:
    if not collection:
        return DEFAULT_AGENT_COLLECTION
    normalized = str(collection).strip().lower()
    if normalized in AGENT_COLLECTIONS:
        return normalized
    return DEFAULT_AGENT_COLLECTION

def ensure_agent_collection_dir(collection: Optional[str]) -> Path:
    ensure_agents_dir()
    normalized = normalize_agent_collection(collection)
    path = AGENT_COLLECTIONS[normalized]
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def safe_agent_name(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    nm = name.strip()
    if AGENT_NAME_RE.match(nm):
        return nm
    return None


def sanitize_agent_name(name: str) -> Optional[str]:
    """Produce a filesystem-safe agent filename from arbitrary display name.
    Returns None if result would be empty."""
    if not name or not isinstance(name, str):
        return None
    # replace spaces and disallowed chars with -
    s = name.strip()
    # normalize: replace long sequences of non-allowed chars with '-'
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s)
    s = re.sub(r'-{2,}', '-', s)
    s = s.strip('-')
    if not s:
        return None
    # limit length
    if len(s) > 64:
        s = s[:64]
    # ensure it matches the allowed pattern
    if AGENT_NAME_RE.match(s):
        return s
    return None

def _is_reasoning_model_id(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    for prefix in REASONING_PREFIXES:
        lowered = prefix.lower()
        if normalized == lowered or normalized.startswith(f"{lowered}-"):
            return True
    return False

def _clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number

def _normalize_reasoning_effort_value(value, default: str = "medium") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else default

def _supports_temperature(model: str) -> bool:
    return not _is_reasoning_model_id(model)

def _supports_top_p(model: str) -> bool:
    return not _is_reasoning_model_id(model)

def _supports_reasoning(model: str) -> bool:
    return _is_reasoning_model_id(model)


def _load_json_if_string(value):
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return trimmed


def _sanitize_response_format(value):
    candidate = _load_json_if_string(value)
    if candidate is None:
        return None
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered == 'json_object':
            return {'type': 'json_object'}
        return None
    if isinstance(candidate, dict):
        type_value = candidate.get('type')
        if not type_value and (candidate.get('json_schema') or candidate.get('schema') or candidate.get('name')):
            type_value = 'json_schema'
        if isinstance(type_value, str):
            normalized_type = type_value.strip().lower()
        else:
            normalized_type = ''
        if normalized_type == 'json_object':
            return {'type': 'json_object'}
        if normalized_type == 'json_schema':
            schema_payload = candidate.get('json_schema')
            if schema_payload is None:
                schema_payload = {
                    key: candidate.get(key)
                    for key in ('name', 'schema', 'strict')
                    if key in candidate
                }
            schema_payload = _load_json_if_string(schema_payload)
            if not isinstance(schema_payload, dict):
                return None
            name_value = schema_payload.get('name')
            schema_value = schema_payload.get('schema')
            schema_value = _load_json_if_string(schema_value)
            if isinstance(schema_value, str):
                try:
                    schema_value = json.loads(schema_value)
                except json.JSONDecodeError:
                    schema_value = None
            if not isinstance(name_value, str) or not name_value.strip() or not isinstance(schema_value, dict):
                return None
            sanitized_schema = {'name': name_value.strip(), 'schema': schema_value}
            if isinstance(schema_payload.get('strict'), bool):
                sanitized_schema['strict'] = schema_payload['strict']
            for extra_key, extra_value in schema_payload.items():
                if extra_key in {'name', 'schema', 'strict'}:
                    continue
                sanitized_schema[extra_key] = extra_value
            return {'type': 'json_schema', 'json_schema': sanitized_schema}
    return None


def agent_filepath(name: str, collection: Optional[str] = None) -> Path:
    collection_dir = ensure_agent_collection_dir(collection)
    return collection_dir / f"{name}.json"

def list_agents_files(collection: Optional[str] = None):
    collection_key = normalize_agent_collection(collection)
    collection_dir = ensure_agent_collection_dir(collection_key)
    out = []
    for p in sorted(collection_dir.glob('*.json')):
        try:
            stat = p.stat()
            parsed = None
            display_name = p.stem
            try:
                raw = p.read_text(encoding='utf-8')
                parsed = json.loads(raw)
                display_name = parsed.get('display_name') or parsed.get('name') or p.stem
            except Exception:
                parsed = None
                display_name = p.stem
            out.append({
                'name': p.stem,
                'collection': collection_key,
                'display_name': display_name,
                'agent': parsed,
                'path': str(p),
                'updated_at': stat.st_mtime,
                'size': stat.st_size,
            })
        except Exception:
            continue
    return out

def read_agent_file(name: str, collection: Optional[str] = None) -> Optional[dict]:
    nm = safe_agent_name(name)
    if not nm:
        return None
    p = agent_filepath(nm, collection)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'collection' not in data:
            data['collection'] = normalize_agent_collection(collection)
        return data
    except Exception:
        return None

def write_agent_file(data: dict, collection: Optional[str] = None) -> Optional[str]:
    name = data.get('name') if isinstance(data, dict) else None
    nm = safe_agent_name(name)
    display_name = name if isinstance(name, str) else ''
    if not nm:
        nm = sanitize_agent_name(display_name)
    if not nm:
        return None
    collection_key = normalize_agent_collection(collection)
    ensure_agent_collection_dir(collection_key)
    p = agent_filepath(nm, collection_key)
    # sanitize and limit
    raw_temperature = data.get('temperature') if isinstance(data, dict) else None
    raw_top_p = data.get('top_p') if isinstance(data, dict) else None
    temperature = _clamp_float(raw_temperature, 0.0, 2.0, 0.0)
    top_p = _clamp_float(raw_top_p, 0.0, 1.0, 1.0)
    try:
        raw_model = str(data.get('model') or '').strip()
    except Exception:
        raw_model = ''
    model_value = raw_model or DEFAULT_MODEL
    base_reasoning = _normalize_reasoning_effort_value(
        data.get('reasoning_effort') if isinstance(data, dict) else None,
        ''
    )
    reasoning_value = base_reasoning if _supports_reasoning(model_value) else ''

    settings_payload = data.get('settings') if isinstance(data, dict) else {}
    defaults_input = settings_payload.get('defaults') if isinstance(settings_payload, dict) else {}
    per_model_input = settings_payload.get('per_model') if isinstance(settings_payload, dict) else {}

    defaults_temperature = _clamp_float(
        defaults_input.get('temperature') if isinstance(defaults_input, dict) else None,
        0.0,
        2.0,
        temperature,
    )
    defaults_top_p = _clamp_float(
        defaults_input.get('top_p') if isinstance(defaults_input, dict) else None,
        0.0,
        1.0,
        top_p,
    )
    defaults_reasoning = _normalize_reasoning_effort_value(
        defaults_input.get('reasoning_effort') if isinstance(defaults_input, dict) else None,
        reasoning_value or 'medium',
    )
    defaults_settings = {
        'temperature': defaults_temperature,
        'top_p': defaults_top_p,
        'reasoning_effort': defaults_reasoning,
    }
    defaults_response_format = _sanitize_response_format(
        defaults_input.get('response_format') if isinstance(defaults_input, dict) else None
    )
    if defaults_response_format:
        try:
            defaults_settings['response_format'] = json.loads(json.dumps(defaults_response_format))
        except Exception:
            defaults_settings['response_format'] = defaults_response_format

    per_model_settings: Dict[str, Dict[str, object]] = {}
    if isinstance(per_model_input, dict):
        for model_key_raw, settings in per_model_input.items():
            model_key = str(model_key_raw or '').strip()
            if not model_key:
                continue
            if not isinstance(settings, dict):
                settings = {}
            sanitized_entry: Dict[str, object] = {}
            if _supports_temperature(model_key) and 'temperature' in settings:
                sanitized_entry['temperature'] = _clamp_float(
                    settings.get('temperature'),
                    0.0,
                    2.0,
                    defaults_temperature,
                )
            if _supports_top_p(model_key) and 'top_p' in settings:
                sanitized_entry['top_p'] = _clamp_float(
                    settings.get('top_p'),
                    0.0,
                    1.0,
                    defaults_top_p,
                )
            if _supports_reasoning(model_key) and 'reasoning_effort' in settings:
                sanitized_entry['reasoning_effort'] = _normalize_reasoning_effort_value(
                    settings.get('reasoning_effort'),
                    defaults_reasoning,
                )
            if 'response_format' in settings:
                normalized_format = _sanitize_response_format(settings.get('response_format'))
                if normalized_format:
                    try:
                        sanitized_entry['response_format'] = json.loads(json.dumps(normalized_format))
                    except Exception:
                        sanitized_entry['response_format'] = normalized_format
            per_model_settings[model_key] = sanitized_entry

    default_entry = per_model_settings.setdefault(model_value, {})
    if _supports_temperature(model_value):
        default_entry['temperature'] = _clamp_float(
            default_entry.get('temperature', temperature),
            0.0,
            2.0,
            temperature,
        )
    else:
        default_entry.pop('temperature', None)
    if _supports_top_p(model_value):
        default_entry['top_p'] = _clamp_float(
            default_entry.get('top_p', top_p),
            0.0,
            1.0,
            top_p,
        )
    else:
        default_entry.pop('top_p', None)
    if _supports_reasoning(model_value):
        default_entry['reasoning_effort'] = _normalize_reasoning_effort_value(
            default_entry.get('reasoning_effort') or reasoning_value or defaults_reasoning,
            defaults_reasoning,
        )
    else:
        default_entry.pop('reasoning_effort', None)
    default_response_format = _sanitize_response_format(
        default_entry.get('response_format')
        or data.get('response_format')
        or defaults_settings.get('response_format')
    )
    if default_response_format:
        try:
            default_entry['response_format'] = json.loads(json.dumps(default_response_format))
        except Exception:
            default_entry['response_format'] = default_response_format
    else:
        default_entry.pop('response_format', None)

    safe = {
        'name': nm,
        'display_name': display_name,
        'prompt': str(data.get('prompt') or '')[:200000],
        'temperature': default_entry.get('temperature', temperature),
        'top_p': default_entry.get('top_p', top_p),
        'model': model_value,
        'reasoning_effort': default_entry.get('reasoning_effort', '') if _supports_reasoning(model_value) else '',
        'settings': {
            'defaults': defaults_settings,
            'per_model': per_model_settings,
        },
        'collection': collection_key,
        'updated_at': time.time(),
    }
    if default_entry.get('response_format'):
        try:
            safe['response_format'] = json.loads(json.dumps(default_entry['response_format']))
        except Exception:
            safe['response_format'] = default_entry['response_format']
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(p)
        try:
            p.chmod(0o600)
        except Exception:
            pass
        # return the canonical stored agent name
        return nm
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass
        return None

def delete_agent_file(name: str, collection: Optional[str] = None) -> bool:
    nm = safe_agent_name(name)
    if not nm:
        return False
    p = agent_filepath(nm, collection)
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception:
        return False
    return False


def _api_base_to_handle_base(api_base: str) -> str:
    if not api_base:
        return ""
    normalized = api_base.rstrip('/')
    if '/api/' in normalized:
        return normalized.split('/api/', 1)[0]
    if normalized.endswith('/api'):
        return normalized[:-4]
    return normalized


def describe_library(api_base: Optional[str]) -> Dict[str, str]:
    normalized = (api_base or '').rstrip('/')
    if not normalized:
        return {
            'label': '',
            'code': '',
            'api_base': '',
            'handle_base': '',
            'netloc': '',
            'version': '',
        }

    override = KNOWN_LIBRARY_OVERRIDES.get(normalized, {})
    handle_base = override.get('handle_base') or _api_base_to_handle_base(normalized)
    parsed = urlparse(handle_base or normalized)
    label = override.get('label') or (parsed.netloc or normalized)
    code = override.get('code') or (parsed.netloc.split('.', 1)[0] if parsed.netloc else '')
    version = ''
    low = normalized.lower()
    if 'api/client/v7.0' in low:
        version = 'K7'
    elif 'search/api/v5.0' in low:
        version = 'K5'

    return {
        'label': label,
        'code': code,
        'api_base': normalized,
        'handle_base': handle_base,
        'netloc': parsed.netloc or '',
        'version': version,
    }

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ComparisonHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        collection = (qs.get('collection') or [''])[0]
        # Agents API endpoints
        if path == '/agents' or path == '/agents/list':
            items = list_agents_files(collection)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'agents': items}, ensure_ascii=False).encode('utf-8'))
            return
        if path == '/agents/get':
            name = (qs.get('name') or [''])[0]
            data = read_agent_file(name, collection)
            if data is None:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'not_found'}).encode('utf-8'))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            return

        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            html = '''<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALTO Processing Comparison</title>
    <style>
:root {
            --thumbnail-drawer-width: clamp(260px, 35vw, 460px);
            --thumbnail-toggle-size: 32px;
            --primary-max-width: 1200px;
            --preview-drawer-gap: 12px;
            --left-drawer-space: var(--thumbnail-drawer-width);
            --right-drawer-space: var(--thumbnail-drawer-width);
        }
        body.thumbnail-drawer-collapsed {
            --left-drawer-space: 0px;
        }
        body.preview-drawer-collapsed {
            --right-drawer-space: 0px;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .page-shell {
            position: relative;
            z-index: 50;
            max-width: var(--primary-max-width);
            width: 100%;
            margin: 0 auto;
            padding-left: 0;
            overflow: visible;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: relative;
            max-width: var(--primary-max-width);
            margin-left: auto;
            margin-right: auto;
            z-index: 20;
        }
        .main-content {
            max-width: var(--primary-max-width);
            margin: 0 auto;
        }
        @media (min-width: 1101px) {
            .page-shell,
            .container,
            .main-content {
                width: min(
                    var(--primary-max-width),
                    max(360px, calc(100vw - var(--left-drawer-space) - var(--right-drawer-space) - 32px))
                );
            }
        }
        #thumbnail-drawer {
            position: absolute;
            top: 0;
            left: 0;
            width: var(--thumbnail-drawer-width);
            transform: translateX(-100%);
            transition: transform 0.3s ease, opacity 0.3s ease;
            pointer-events: auto;
        }
        .thumbnail-panel {
            background: white;
            border: 1px solid #dbe4f0;
            border-right: none;
            border-radius: 8px 0 0 8px;
            box-shadow: none;
            padding: 16px 16px 16px 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 100%;
            max-height: none;
            overflow: hidden;
        }
        body.thumbnail-drawer-collapsed #thumbnail-drawer {
            transform: translateX(0);
            opacity: 0;
            pointer-events: none;
        }
        body.thumbnail-drawer-collapsed .thumbnail-panel {
            pointer-events: none;
        }
        .thumbnail-toggle {
            position: absolute;
            top: 18px;
            left: 0;
            transform: translateX(-50%);
            width: var(--thumbnail-toggle-size);
            height: var(--thumbnail-toggle-size);
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            color: #1f2933;
            font-weight: 600;
            border: 1px solid #d0d7e2;
            border-radius: 10px;
            box-shadow: none;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease, box-shadow 0.25s ease;
            z-index: 21;
        }
        body.page-is-loading .thumbnail-toggle {
            z-index: 5;
            pointer-events: none;
            opacity: 0;
        }
        .thumbnail-toggle:hover {
            background: #1f78ff;
            color: #ffffff;
            box-shadow: none;
        }
        #preview-drawer {
            position: fixed;
            top: 20px;
            width: var(--thumbnail-drawer-width);
            max-width: var(--thumbnail-drawer-width);
            left: auto;
            right: auto;
            z-index: 5; /* keep behind .container (20) */
            transition: opacity 0.3s ease;
            transform: none; /* avoid creating new stacking context on the host */
            pointer-events: auto;
        }
        body.preview-drawer-collapsed #preview-drawer {
            pointer-events: none;
        }
        #preview-drawer .preview-drawer-panel {
            background: white;
            border: 1px solid #dbe4f0;
            border-left: none;
            border-radius: 0 8px 8px 0;
            box-shadow: none; /* remove elevation so it visually sits behind primary */
            z-index: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: calc(100vh - 40px);
            overflow: auto;
            pointer-events: auto;
            transition: transform 0.3s ease, opacity 0.3s ease;
            transform: none;
        }
        .preview-drawer-image {
            width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.15);
            object-fit: contain;
            display: none;
        }
        .preview-drawer-empty {
            font-size: 13px;
            color: #5f6b7c;
            text-align: center;
            padding: 20px 8px;
        }
        .preview-drawer-toggle {
            position: fixed;
            top: 38px;
            left: auto;
            right: 20px;
            width: var(--thumbnail-toggle-size);
            height: var(--thumbnail-toggle-size);
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            color: #1f2933;
            font-weight: 600;
            border: 1px solid #d0d7e2;
            border-radius: 10px;
            box-shadow: none;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease, box-shadow 0.25s ease;
            z-index: 120;
        }
        .preview-drawer-toggle:hover {
            background: #1f78ff;
            color: #ffffff;
            box-shadow: none;
        }
        body.page-is-loading .preview-drawer-toggle {
            z-index: 5;
            pointer-events: none;
            opacity: 0;
        }
        body.preview-drawer-collapsed #preview-drawer .preview-drawer-panel {
            transform: translateX(calc(-100% - var(--preview-drawer-gap)));
            opacity: 0;
            pointer-events: none;
        }
        .page-jump {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .page-jump label {
            margin: 0;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            color: #1f2933;
        }
        #page-number-input {
            width: 72px;
            padding: 6px 8px;
            border: 1px solid #cfd4dc;
            border-radius: 4px;
            font-size: 14px;
            background: #fff;
        }
        #page-number-input:disabled {
            background-color: #f1f3f5;
            color: #98a0ab;
        }
        .page-jump-total {
            color: #52606d;
            font-size: 13px;
        }
        .thumbnail-scroll {
            flex: 1 1 auto;
            overflow-y: auto;
            padding: 8px;
            border: 1px solid #e0e6ef;
            border-radius: 8px;
            background: #f9fbff;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.08);
            max-height: inherit;
            min-height: 0;
        }
        .thumbnail-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }
        .page-thumbnail {
            position: relative;
            border: none;
            background: #ffffff;
            color: inherit;
            border-radius: 6px;
            padding: 0;
            cursor: pointer;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.1);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            font-family: inherit;
            aspect-ratio: 3 / 4;
        }
        .thumbnail-placeholder {
            position: absolute;
            inset: 8px;
            border-radius: 6px;
            background: linear-gradient(135deg, #e7edf5 0%, #f1f5fb 100%);
            transition: opacity 0.2s ease;
        }
        .page-thumbnail:hover {
            background: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18);
        }
        .page-thumbnail:disabled {
            cursor: default;
            opacity: 0.6;
            box-shadow: none;
        }
        .page-thumbnail:disabled:hover {
            transform: none;
            box-shadow: none;
        }
        .page-thumbnail img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #f0f2f7;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .page-thumbnail.is-loaded .thumbnail-placeholder {
            opacity: 0;
            visibility: hidden;
        }
        .page-thumbnail.is-loaded img {
            opacity: 1;
        }
        .page-thumbnail-label {
            position: absolute;
            top: 6px;
            right: 6px;
            background: rgba(15, 23, 42, 0.75);
            color: #ffffff;
            font-size: 11px;
            padding: 3px 5px;
            border-radius: 4px;
            line-height: 1;
        }
        .page-thumbnail.is-active {
            outline: 3px solid #1f78ff;
            outline-offset: 0;
            box-shadow: 0 0 0 2px rgba(31, 120, 255, 0.25);
        }
        .page-thumbnail:focus-visible {
            outline: 3px solid #1f78ff;
            outline-offset: 0;
        }
        .thumbnail-empty {
            font-size: 13px;
            color: #5f6b7c;
            padding: 20px 8px;
            text-align: center;
        }
        .main-content {
            min-width: 0;
        }
        @media (max-width: 2150px) {
            :root {
                --thumbnail-drawer-width: clamp(220px, 32vw, 340px);
            }
            .thumbnail-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 1880px) {
            :root {
                --thumbnail-drawer-width: clamp(220px, 38vw, 300px);
            }
        }
        @media (max-width: 1100px) {
            body {
                padding: 12px;
                --primary-max-width: 100%;
            }
            .page-shell {
                max-width: none;
            }
            .container {
                padding: 16px;
                padding-top: 20px;
                padding-bottom: 20px;
            }
            #thumbnail-drawer {
                position: static;
                width: 100%;
                transform: none;
                margin-bottom: 12px;
            }
            body.thumbnail-drawer-collapsed #thumbnail-drawer {
                transform: none;
            }
            .thumbnail-panel {
                max-height: none;
                width: 100%;
                border-right: 1px solid #dbe4f0;
                border-radius: 8px;
                pointer-events: auto;
            }
            .thumbnail-scroll {
                max-height: none;
            }
            .thumbnail-grid {
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            }
            .thumbnail-toggle {
                position: static;
                margin-left: auto;
                box-shadow: none;
            }
            #preview-drawer {
                position: static;
                width: 100%;
                max-width: none;
                transform: none;
                margin-top: 12px;
            }
            body.preview-drawer-collapsed #preview-drawer {
                transform: none;
                opacity: 0;
                pointer-events: none;
            }
            .preview-drawer-toggle {
                position: static;
                transform: none;
                margin-left: auto;
                margin-top: 8px;
            }
            body.thumbnail-drawer-collapsed #thumbnail-scroll {
                visibility: visible;
            }
        }
        @media (max-height: 700px) {
            .thumbnail-panel {
                max-height: calc(100vh - 120px);
            }
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            box-sizing: border-box;
            resize: none;
            overflow: hidden;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle):not(.preview-drawer-toggle) {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle):not(.preview-drawer-toggle):hover {
            background-color: #0056b3;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle):not(.preview-drawer-toggle):disabled {
            background-color: #9aa0a6;
            cursor: not-allowed;
        }
        .form-group.uuid-group {
            width: 100%;
        }
        .uuid-input-group {
            display: flex;
            align-items: stretch;
            gap: 8px;
        }
        .inline-tooltip-anchor {
            position: relative;
            display: inline-flex;
            align-items: center;
        }
        .uuid-input-group input {
            flex: 1;
            min-width: 0;
        }
        .uuid-input-group button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #cfd4dc;
            background: #f9fafb;
            color: #4b5563;
            border-radius: 4px;
            padding: 0;
            font-size: 16px;
            line-height: 1;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
            width: 40px;
            height: 38px;
        }
        .uuid-input-group button:hover {
            background: #e5e7eb;
            color: #111827;
        }
        .uuid-input-group button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 2px rgba(31, 120, 255, 0.25);
        }
        .uuid-input-group button:disabled {
            border-color: #cfd4dc;
            color: #9aa0a6;
            background: #f5f6f8;
            cursor: not-allowed;
        }
        .inline-tooltip {
            position: absolute;
            left: 50%;
            top: calc(100% + 6px);
            transform: translate(-50%, 0);
            background: rgba(229, 241, 255, 0.96);
            color: #0b3d8a;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(183, 212, 255, 0.9);
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
            opacity: 0;
            pointer-events: none;
            white-space: nowrap;
            transition: opacity 0.15s ease, transform 0.15s ease;
            z-index: 30;
        }
        .inline-tooltip.is-visible {
            opacity: 1;
            transform: translate(-50%, -4px);
        }
        .agent-diff-section {
            margin-top: 16px;
            display: none;
            flex-direction: column;
            gap: 16px;
        }
        .agent-diff-section.is-visible {
            display: flex;
        }
        .agent-diff-controls {
            display: inline-flex;
            align-items: stretch;
            border: 1px solid #cdd5e0;
            border-radius: 999px;
            overflow: hidden;
            background: #f8f9fc;
        }
        .agent-diff-toggle {
            border: none;
            background: transparent;
            padding: 6px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
        }
        .agent-diff-toggle + .agent-diff-toggle {
            border-left: 1px solid #cdd5e0;
        }
        .agent-diff-toggle.is-active {
            background: #1f78ff;
            color: #ffffff;
        }
        .agent-diff-toggle:focus-visible {
            outline: none;
            box-shadow: inset 0 0 0 2px rgba(31, 120, 255, 0.4);
        }
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .action-row {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 20px;
            flex-wrap: wrap;
        }
        #load-button {
            min-width: calc(3 * 40px + 16px);
        }
        .tools-row {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
            width: 100%;
            justify-content: center;
            position: relative;
        }
        .navigation-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin: 0 auto;
        }
        .navigation-controls button {
            padding: 8px 12px;
        }
        .navigation-controls span {
            min-width: 60px;
            text-align: center;
            font-weight: bold;
            color: #333;
        }
        .page-info-layout {
            display: flex;
            align-items: stretch;
            gap: 16px;
            position: relative;
        }
        .page-details {
            flex: 1 1 auto;
            min-width: 0;
        }
        .page-preview {
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: flex-start;
            gap: 8px;
            overflow: visible;
            min-width: 140px;
            margin-left: auto;
            align-self: flex-start;
            height: auto;
            min-height: 120px;
            min-height: 200px;
        }
        .page-preview.preview-visible {
            display: flex;
        }
        .page-preview img {
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            max-height: 100%;
        }
        #preview-image-thumb {
            display: block;
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 180px;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .preview-error #preview-image-thumb {
            display: none;
        }
        .preview-error #preview-status {
            color: red;
            font-weight: bold;
        }
        .preview-large {
            pointer-events: auto;
            position: absolute;
            top: 0;
            right: 0;
            transform-origin: top right;
            transform: translate(16px, -16px) scale(0.35);
            opacity: 0;
            visibility: hidden;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 30;
            transition: transform 0.5s ease, opacity 0.5s ease;
        }
        .page-preview.preview-loaded:hover ~ .preview-large,
        .preview-large.preview-large-visible {
            transform: translate(16px, -16px) scale(1);
            opacity: 1;
            visibility: visible;
        }
        .preview-large img {
            display: block;
            border-radius: 4px;
        }
        #preview-status {
            width: 100%;
            text-align: right;
            max-width: 220px;
        }
        .result-box {
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
        }
        .result-box h3 {
            margin-top: 0;
            color: #333;
        }
        .scan-result-frame {
            position: relative;
            border: 1px solid #dbe4f0;
            border-radius: 6px;
            min-height: 280px;
            background: linear-gradient(135deg, #f8fafc 0%, #edf2fb 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .scan-result-frame.has-image {
            background: #ffffff;
        }
        .scan-result-image {
            width: 100%;
            height: auto;
            display: none;
            border-radius: 4px;
            object-fit: contain;
            background: #ffffff;
        }
        .comparison-status {
            margin-top: 8px;
            display: none;
        }
        .comparison-status.is-error {
            color: #b91c1c;
        }
        .comparison-status.is-pending {
            font-weight: 600;
        }
        #reader-result-text pre {
            margin: 0;
        }
        #reader-result-text pre p {
            margin: 0;
        }
        #reader-result-text pre p + p {
            margin-top: 12px;
        }
        .scan-result-placeholder {
            color: #4b5563;
            font-size: 14px;
            text-align: center;
            padding: 16px;
            line-height: 1.5;
        }
        .scan-result-frame.has-image .scan-result-placeholder {
            display: none;
        }
        .diff-section {
            margin-top: 28px;
            display: none;
            flex-direction: column;
            gap: 18px;
        }
        .diff-section.is-visible {
            display: flex;
        }
        .diff-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }
        .diff-heading {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            color: #1f2933;
            letter-spacing: -0.01em;
        }
        .diff-controls {
            display: inline-flex;
            align-items: stretch;
            border: 1px solid #cdd5e0;
            border-radius: 999px;
            overflow: hidden;
            background: #f8f9fc;
        }
        .diff-toggle {
            border: none;
            background: transparent;
            padding: 6px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
        }
        .diff-toggle + .diff-toggle {
            border-left: 1px solid #cdd5e0;
        }
        .diff-toggle.is-active {
            background: #1f78ff;
            color: #ffffff;
        }
        .diff-toggle:focus-visible {
            outline: none;
            box-shadow: inset 0 0 0 2px rgba(31, 120, 255, 0.4);
        }
        .result-rendered {
            line-height: 1.65;
        }
        .result-html-section {
            margin-top: 16px;
        }
        .diff-content {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 16px;
            white-space: normal;
            word-break: break-word;
            overflow-x: auto;
            min-height: 64px;
            font-family: inherit;
            line-height: 1.6;
        }
        .diff-html {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin: 0;
        }
        .diff-block {
            background: #ffffff;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 10px 12px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02);
        }
        .diff-block--added {
            background: rgba(46, 160, 67, 0.12);
            border-color: rgba(46, 160, 67, 0.35);
        }
        .diff-block--removed {
            background: rgba(219, 68, 55, 0.12);
            border-color: rgba(219, 68, 55, 0.35);
        }
        .diff-block--changed {
            background: rgba(255, 196, 0, 0.18);
            border-color: rgba(255, 196, 0, 0.5);
        }
        .diff-added {
            background: rgba(46, 160, 67, 0.38);
            border-radius: 3px;
            padding: 0 2px;
        }
        .diff-removed {
            background: rgba(219, 68, 55, 0.38);
            border-radius: 3px;
            padding: 0 2px;
        }
        .diff-loading {
            color: #52606d;
            font-style: italic;
        }
        .diff-error {
            color: #b91c1c;
            font-weight: 600;
        }
        .diff-placeholder {
            color: #6b7280;
            font-style: italic;
        }
        .loading {
            position: fixed;
            inset: 0;
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(3px);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 12px;
            text-align: center;
            z-index: 9999;
        }
        .container.is-loading .loading {
            display: flex;
        }
        .loading-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }
        .loading-spinner {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 4px solid rgba(0, 123, 255, 0.25);
            border-top-color: #007bff;
            animation: spin 0.8s linear infinite;
        }
        /* small inline spinner used next to dropdowns */
        .inline-spinner {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid rgba(0,0,0,0.08);
            border-top-color: #007bff;
            animation: spin 0.8s linear infinite;
            display: inline-block;
            vertical-align: middle;
        }
        .loading p {
            margin: 0;
            color: #333;
            font-weight: 600;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            color: red;
            padding: 10px;
            background: #ffe6e6;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success {
            color: green;
            padding: 10px;
            background: #e6ffe6;
            border-radius: 4px;
            margin: 10px 0;
        }
        .info-section {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #fafafa;
            position: relative;
        }

        .info-section h2 {
            margin-top: 0;
            color: #333;
        }
        .stitch-preview-row {
            display: flex;
            flex-wrap: wrap;
            gap: 64px;
            justify-content: center;
            overflow: visible;
            padding: 4px 0;
        }
        .stitch-preview-tile {
            flex: 0 0 auto;
            min-width: calc(var(--stitch-thumb-width, 180) * 1px);
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: center;
            position: relative;
        }
        .stitch-preview-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #52606d;
        }
        .stitch-preview-frame {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            min-height: calc(var(--stitch-thumb-min-height, 220) * 1px);
            overflow: visible;
            z-index: 1;
        }
        .stitch-preview-box {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            outline: none;
            transform-origin: bottom center;
            transition: transform 0.28s ease-in-out, box-shadow 0.22s ease-in-out;
            background: transparent;
            border: none;
            padding: 0;
            cursor: default;
        }
        .stitch-preview-box.is-empty {
            min-width: calc(var(--stitch-thumb-width, 180) * 1px);
            min-height: calc(var(--stitch-thumb-min-height, 220) * 1px);
            cursor: default;
        }
        .stitch-preview-box:not(.is-empty) {
            cursor: pointer;
        }
        .stitch-preview-thumb-img {
            max-width: 100%;
            max-height: calc(var(--stitch-thumb-min-height, 220) * 1px - 12px);
            border-radius: 6px;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.18);
            transition: box-shadow 0.22s ease-in-out;
        }
        .stitch-preview-box.is-empty .stitch-preview-thumb-img {
            display: none;
        }
        .stitch-preview-placeholder {
            font-size: 13px;
            color: #9aa5b1;
        }
        .stitch-preview-frame.is-active {
            z-index: 32;
        }
        .stitch-preview-frame.is-active .stitch-preview-box {
            transform: scale(var(--stitch-expand-scale, 3));
            box-shadow: none;
        }
        .stitch-preview-frame.is-active .stitch-preview-thumb-img {
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.35);
            border-radius: 6px;
        }
        .stitch-merge-container {
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .stitch-merge-block {
            border: 1px solid #d4dae4;
            border-radius: 6px;
            background: #ffffff;
            padding: 14px;
        }
        .stitch-merge-block h3 {
            margin: 0 0 12px 0;
            font-size: 16px;
            color: #1f2933;
        }
        .stitch-merge-columns {
            display: flex;
            flex-direction: row;
            gap: 16px;
        }
        .stitch-merge-column {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .stitch-merge-note {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #52606d;
        }
        .stitch-merge-snippet {
            min-height: 72px;
            padding: 10px;
            border-radius: 6px;
            border: 1px dashed #cbd2d9;
            background: #f8fafc;
            font-family: monospace;
            font-size: 13px;
            color: #1f2933;
            white-space: pre-wrap;
        }
        @media (max-width: 860px) {
            .stitch-preview-row {
                flex-direction: column;
            }
            .stitch-merge-columns {
                flex-direction: column;
            }
            .stitch-preview-frame {
                max-width: 100%;
            }
            .stitch-preview-frame.is-active .stitch-preview-box {
                --stitch-expand-scale: min(calc(90vw / (var(--stitch-thumb-width, 180) * 1px)), 4.0);
            }
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 8px 16px;
            margin: 0;
        }
        .metadata-grid dt {
            font-weight: bold;
            color: #333;
        }
        .metadata-grid dd {
            margin: 0 0 8px 0;
            color: #555;
        }
        .book-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px 12px;
            margin-bottom: 12px;
        }
        .book-chip {
            background: #eef3ff;
            border: 1px solid #d9e2ff;
            border-radius: 6px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .book-chip strong {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #2a3cb5;
        }
        .book-chip-value {
            font-size: 14px;
            color: #1f2933;
        }
        .book-chip-meta {
            font-size: 12px;
            color: #52606d;
        }
        .muted {
            color: #555;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgb(0,0,0);
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: auto;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 0;
            border: 1px solid #888;
            width: 80%;
            max-width: 1200px;
            max-height: 70%;
            overflow-y: auto;
        }
        .modal-header {
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background-color: #fefefe;
            border-bottom: 1px solid #ddd;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .modal-content pre {
            padding: 20px;
            margin: 0;
        }
        .close {
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="page-shell">
        <aside id="thumbnail-drawer" aria-label="Náhledy stránek">
            <div class="thumbnail-panel">
                <div class="page-jump">
                    <label for="page-number-input">Strana</label>
                    <input type="number" id="page-number-input" min="1" step="1" inputmode="numeric" aria-label="Zadat číslo strany">
                    <span id="page-number-total" class="page-jump-total"></span>
                </div>
                <div id="thumbnail-scroll" class="thumbnail-scroll">
                    <div id="thumbnail-grid" class="thumbnail-grid">
                        <div class="thumbnail-empty">Náhledy budou k dispozici po načtení knihy.</div>
                    </div>
                </div>
            </div>
        </aside>
        <button id="thumbnail-toggle" class="thumbnail-toggle" type="button" aria-expanded="true" aria-controls="thumbnail-grid" aria-label="Skrýt náhledy">&gt;</button>
        <div class="container">
            <div class="main-content">
                <h1>ALTO Processing Comparison</h1>
                <p>Ukázka způsobů zpracování formátovaného textu ze stránek knih.</p>

                <div class="form-group uuid-group">
                    <label for="uuid">UUID stránky nebo dokumentu:</label>
                    <div class="uuid-input-group">
                        <input type="text" id="uuid" placeholder="Zadejte UUID (např. 49c6424a-c820-4224-9475-4aa0d8a9d844)" value="49c6424a-c820-4224-9475-4aa0d8a9d844" autocomplete="off" autocorrect="off" autocapitalize="none" spellcheck="false">
                        <span class="inline-tooltip-anchor">
                            <button type="button" id="uuid-copy" title="Zkopírovat UUID" aria-label="Zkopírovat UUID">📋</button>
                            <div id="uuid-copy-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                        </span>
                        <span class="inline-tooltip-anchor">
                            <button type="button" id="uuid-paste" title="Vložit UUID ze schránky" aria-label="Vložit UUID ze schránky">📥</button>
                            <div id="uuid-paste-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                        </span>
                        <button type="button" id="uuid-clear" title="Vymazat UUID" aria-label="Vymazat UUID">✕</button>
                    </div>
                </div>

                <div class="action-row">
                    <button id="load-button" type="button" onclick="handleLoadClick()">Načíst stránku</button>
                </div>

                <div id="book-info" class="info-section" style="display: none;">
                    <h2 id="book-title">Informace o knize</h2>
                    <p id="book-handle" class="muted"></p>
                    <p id="book-library" class="muted"></p>
                    <div id="book-constants" class="book-summary-grid" style="display: none;"></div>
                    <div id="book-metadata-empty" class="muted" style="display: none;">Metadata se nepodařilo načíst.</div>
                    <dl id="book-metadata" class="metadata-grid"></dl>
                </div>

                <div id="page-info" class="info-section" style="display: none;">
                    <div class="page-info-layout">
                        <div class="page-details">
                            <h2>Informace o straně</h2>
                            <p id="page-summary" class="muted"></p>
                            <p id="page-side" class="muted"></p>
                            <p id="page-uuid" class="muted"></p>
                            <p id="page-handle" class="muted"></p>
                        </div>
                        <div class="page-alto-btn">
                            <span id="alto-preview-btn" style="display: none;">Zobrazit ALTO</span>
                        </div>
                        <div id="page-preview" class="page-preview" tabindex="0">
                            <div id="preview-status" class="muted"></div>
                            <img id="preview-image-thumb" alt="Náhled stránky">
                        </div>
                        <div id="preview-large" class="preview-large" aria-hidden="true">
                            <img id="preview-image-large" alt="Náhled stránky ve větší velikosti">
                        </div>
                    </div>
                </div>

                <div id="page-tools" class="tools-row" style="display: none;">
                    <div class="navigation-controls">
                        <button id="prev-page" type="button" aria-label="Předchozí stránka">◀</button>
                        <span id="page-position">-</span>
                        <button id="next-page" type="button" aria-label="Další stránka">▶</button>
                    </div>
                </div>
                <div id="results" class="results" style="display: none;">
                    <div class="result-box">
                        <h3>Aktuálně nasazené zpracování ALTO</h3>
                        <div id="typescript-result" class="result-rendered"></div>
                    </div>
                    <div class="result-box">
                        <h3>Nový přístup zpracování ALTO</h3>
                        <div id="python-result" class="result-rendered"></div>
                    </div>
                </div>
                <section id="diff-section" class="diff-section">
                    <div class="diff-header">
                        <h2 class="diff-heading">Porovnání formátovaného HTML</h2>
                        <div id="diff-mode-controls" class="diff-controls" role="group" aria-label="Režim zvýraznění">
                            <button type="button" class="diff-toggle" data-diff-mode="word">Slova</button>
                            <button type="button" class="diff-toggle" data-diff-mode="char">Znaky</button>
                        </div>
                    </div>
                    <div id="html-diff" class="results">
                        <div class="result-box">
                            <h3>Python diff</h3>
                            <div id="python-html" class="diff-html"></div>
                        </div>
                        <div class="result-box">
                            <h3>TypeScript diff</h3>
                            <div id="typescript-html" class="diff-html"></div>
                        </div>
                    </div>
                </section>

                <!-- LLM agent settings UI -->
                <div id="agent-row" class="info-section" style="margin-top:18px;">
                    <h2 style="margin-bottom:12px;">Oprava zpracovaného textu pomocí LLM</h2>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <label for="agent-select" style="margin:0;font-weight:600;">Agent:</label>
                        <select id="agent-select" aria-label="Vyberte agenta"></select>
                        <div id="agent-select-spinner" title="Načítám agenty" style="margin-left:8px;display:none;">
                            <span class="inline-spinner" aria-hidden="true"></span>
                        </div>
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="agent-auto-correct" type="checkbox">
                                <span style="font-size:13px;">Automaticky opravovat</span>
                            </label>
                            <button id="agent-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Oprav</button>
                            <button id="agent-expand-toggle" type="button" aria-expanded="false" title="Zobrazit nastavení agenta" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 10px;">⚙️</button>
                        </div>
                    </div>

                    <div id="agent-settings" style="display:none;margin-top:12px;border-top:1px solid #e6e9ef;padding-top:12px;">
                        <div class="form-group">
                            <label for="agent-name">Název agenta</label>
                            <input type="text" id="agent-name" placeholder="Např. default-editor">
                        </div>
                        <div class="form-group">
                            <label for="agent-prompt">Prompt</label>
                            <textarea id="agent-prompt" rows="6" style="width:100%;font-family:monospace;">Zadejte prompt...</textarea>
                        </div>
                        <div class="form-group">
                            <label for="agent-default-model">Model</label>
                            <select id="agent-default-model" aria-label="Vyberte model pro agenta" style="min-width:170px;"></select>
                        </div>
                        <div id="agent-parameter-fields" class="form-group" style="margin-top:8px;"></div>
                        <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;align-items:center;">
                            <span class="inline-tooltip-anchor">
                                <button id="agent-save" type="button">Uložit agenta</button>
                                <div id="agent-save-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                            </span>
                            <button id="agent-delete" type="button" style="background:#e53e3e;">Smazat</button>
                            <div style="margin-left:auto;">
                                <button id="agent-run-inline" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Oprav</button>
                            </div>
                        </div>
                    </div>
                    <div id="agent-output" style="display:none;margin-top:12px;">
                        <div id="agent-output-status" class="muted" style="margin-bottom:6px;"></div>
                        <pre id="agent-output-text" style="white-space:pre-wrap;"></pre>
                    </div>
                    <div id="agent-results" class="results" style="display:none;margin-top:16px;">
                        <div class="result-box">
                            <h3>Nový přístup zpracování ALTO</h3>
                            <div id="agent-result-original" class="result-rendered"></div>
                        </div>
                        <div class="result-box">
                            <h3>Zpracování ALTO - LLM</h3>
                            <div id="agent-result-corrected" class="result-rendered"></div>
                        </div>
                    </div>
                    <div style="margin-top:8px;display:flex;justify-content:flex-end;">
                        <div id="agent-diff-mode-controls" class="agent-diff-controls" role="group" aria-label="Režim zvýraznění agent diffu" style="display:none;">
                            <button type="button" class="agent-diff-toggle" data-diff-mode="word">Slova</button>
                            <button type="button" class="agent-diff-toggle" data-diff-mode="char">Znaky</button>
                        </div>
                    </div>
                </div>
                <div id="reader-row" class="info-section" style="margin-top:18px;">
                    <h2 style="margin-bottom:12px;">Čtení přímo ze skenu (OCR)</h2>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <label for="reader-agent-select" style="margin:0;font-weight:600;">Agent:</label>
                        <select id="reader-agent-select" aria-label="Vyberte agenta pro čtení"></select>
                        <div id="reader-agent-select-spinner" title="Načítám agenty" style="margin-left:8px;display:none;">
                            <span class="inline-spinner" aria-hidden="true"></span>
                        </div>
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="reader-agent-auto-read" type="checkbox">
                                <span style="font-size:13px;">Automaticky vyčítat</span>
                            </label>
                            <button id="reader-agent-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Vyčti</button>
                            <button id="reader-agent-expand-toggle" type="button" aria-expanded="false" title="Zobrazit nastavení agenta" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 10px;">⚙️</button>
                        </div>
                    </div>

                    <div id="reader-agent-settings" style="display:none;margin-top:12px;border-top:1px solid #e6e9ef;padding-top:12px;">
                        <div class="form-group">
                            <label for="reader-agent-name">Název agenta</label>
                            <input type="text" id="reader-agent-name" placeholder="Např. default-reader">
                        </div>
                        <div class="form-group">
                            <label for="reader-agent-prompt">Prompt</label>
                            <textarea id="reader-agent-prompt" rows="6" style="width:100%;font-family:monospace;">Zadejte prompt...</textarea>
                        </div>
                        <div class="form-group">
                            <label for="reader-agent-default-model">Model</label>
                            <select id="reader-agent-default-model" aria-label="Vyberte model pro čtení" style="min-width:170px;"></select>
                        </div>
                        <div id="reader-agent-parameter-fields" class="form-group" style="margin-top:8px;"></div>
                        <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;align-items:center;">
                            <span class="inline-tooltip-anchor">
                                <button id="reader-agent-save" type="button">Uložit agenta</button>
                                <div id="reader-agent-save-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                            </span>
                            <button id="reader-agent-delete" type="button" style="background:#e53e3e;">Smazat</button>
                            <div style="margin-left:auto;">
                                <button id="reader-agent-run-inline" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Vyčti</button>
                            </div>
                        </div>
                    </div>
                    <div id="reader-agent-output" style="display:none;margin-top:12px;">
                        <div id="reader-agent-output-status" class="muted" style="margin-bottom:6px;"></div>
                        <pre id="reader-agent-output-text" style="white-space:pre-wrap;"></pre>
                    </div>
                    <div id="reader-results" class="results" style="display:none;margin-top:16px;">
                        <div class="result-box">
                            <h3>Sken strany</h3>
                            <div id="reader-result-scan" class="scan-result-frame">
                                <img id="reader-result-scan-img" class="scan-result-image" alt="Aktuální sken strany" src="">
                                <div class="scan-result-placeholder">Náhled aktuální strany zatím není k dispozici.</div>
                            </div>
                        </div>
                        <div class="result-box">
                            <h3>Nové OCR</h3>
                            <div id="reader-result-text" class="result-rendered">
                                <div class="muted">Výsledek se zobrazí po spuštění „Vyčti“.</div>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- Additional comparison: processed ALTO (Python) vs new OCR -->
                <div id="comparison2-row" class="info-section" style="margin-top:18px;">
                    <h2 style="margin-bottom:12px;">Porovnání: Zpracované ALTO x Nové OCR</h2>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="comparison2-auto-run" type="checkbox">
                                <span style="font-size:13px;">Automaticky porovnávat</span>
                            </label>
                            <button id="comparison2-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Porovnej</button>
                        </div>
                    </div>
                    <div id="comparison2-status" class="comparison-status muted" aria-live="polite"></div>
                    <div id="comparison2-results" class="results" style="display:none;margin-top:16px;">
                        <div class="result-box">
                            <h3>Nový přístup zpracování ALTO</h3>
                            <div id="comparison2-result-left" class="result-rendered"></div>
                        </div>
                        <div class="result-box">
                            <h3>Nové OCR</h3>
                            <div id="comparison2-result-right" class="result-rendered"></div>
                        </div>
                    </div>
                    <div style="margin-top:8px;display:flex;justify-content:flex-end;">
                        <div id="comparison2-diff-mode-controls" class="agent-diff-controls" role="group" aria-label="Režim zvýraznění porovnání" style="display:none;">
                            <button type="button" class="agent-diff-toggle" data-mode="word">Slova</button>
                            <button type="button" class="agent-diff-toggle" data-mode="char">Znaky</button>
                        </div>
                    </div>
                </div>
                <div id="comparison-row" class="info-section" style="margin-top:18px;">
                    <h2 style="margin-bottom:12px;">Porovnání: LLM opravené zpracování ALTO x Nové OCR</h2>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="comparison-auto-run" type="checkbox">
                                <span style="font-size:13px;">Automaticky porovnávat</span>
                            </label>
                            <button id="comparison-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Porovnej</button>
                        </div>
                    </div>
                    <div id="comparison-status" class="comparison-status muted" aria-live="polite"></div>
                    <div id="comparison-results" class="results" style="display:none;margin-top:16px;">
                        <div class="result-box">
                            <h3>Zpracování ALTO - LLM</h3>
                            <div id="comparison-result-left" class="result-rendered"></div>
                        </div>
                        <div class="result-box">
                            <h3>Nové OCR</h3>
                            <div id="comparison-result-right" class="result-rendered"></div>
                        </div>
                    </div>
                    <div style="margin-top:8px;display:flex;justify-content:flex-end;">
                        <div id="comparison-diff-mode-controls" class="agent-diff-controls" role="group" aria-label="Režim zvýraznění porovnání" style="display:none;">
                            <button type="button" class="agent-diff-toggle" data-mode="word">Slova</button>
                            <button type="button" class="agent-diff-toggle" data-mode="char">Znaky</button>
                        </div>
                    </div>
                </div>
                <div id="stitch-row" class="info-section" style="margin-top:18px;">
                    <h2 style="margin-bottom:12px;">Napojení stran</h2>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <label for="stitch-agent-select" style="margin:0;font-weight:600;">Agent:</label>
                        <select id="stitch-agent-select" aria-label="Vyberte agenta pro napojování"></select>
                        <div id="stitch-agent-select-spinner" title="Načítám agenty" style="margin-left:8px;display:none;">
                            <span class="inline-spinner" aria-hidden="true"></span>
                        </div>
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="stitch-agent-auto-link" type="checkbox">
                                <span style="font-size:13px;">Automaticky napojovat</span>
                            </label>
                            <button id="stitch-agent-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Napoj</button>
                            <button id="stitch-agent-expand-toggle" type="button" aria-expanded="false" title="Zobrazit nastavení agenta" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 10px;">⚙️</button>
                        </div>
                    </div>

                    <div id="stitch-agent-settings" style="display:none;margin-top:12px;border-top:1px solid #e6e9ef;padding-top:12px;">
                        <div class="form-group">
                            <label for="stitch-agent-name">Název agenta</label>
                            <input type="text" id="stitch-agent-name" placeholder="Např. default-stitcher">
                        </div>
                        <div class="form-group">
                            <label for="stitch-agent-prompt">Prompt</label>
                            <textarea id="stitch-agent-prompt" rows="6" style="width:100%;font-family:monospace;">Zadejte prompt...</textarea>
                        </div>
                        <div class="form-group">
                            <label for="stitch-agent-default-model">Model</label>
                            <select id="stitch-agent-default-model" aria-label="Vyberte model pro napojovacího agenta" style="min-width:170px;"></select>
                        </div>
                        <div id="stitch-agent-parameter-fields" class="form-group" style="margin-top:8px;"></div>
                        <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;align-items:center;">
                            <span class="inline-tooltip-anchor">
                                <button id="stitch-agent-save" type="button">Uložit agenta</button>
                                <div id="stitch-agent-save-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                            </span>
                            <button id="stitch-agent-delete" type="button" style="background:#e53e3e;">Smazat</button>
                            <div style="margin-left:auto;">
                                <button id="stitch-agent-run-inline" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Napoj</button>
                            </div>
                        </div>
                    </div>

                    <div id="stitch-agent-output" style="display:none;margin-top:12px;">
                        <div id="stitch-agent-output-status" class="muted" style="margin-bottom:6px;"></div>
                        <pre id="stitch-agent-output-text" style="white-space:pre-wrap;"></pre>
                    </div>

                    <div id="stitch-preview-row" class="stitch-preview-row" style="display:none;margin-top:16px;">
                        <div class="stitch-preview-tile" data-role="previous">
                            <div class="stitch-preview-label">Předchozí strana</div>
                            <div class="stitch-preview-frame" data-role="previous">
                                <div class="stitch-preview-box is-empty" tabindex="-1" aria-label="Předchozí strana">
                                    <img id="stitch-preview-previous-thumb" class="stitch-preview-thumb-img" alt="Předchozí strana" src="" />
                                    <span class="stitch-preview-placeholder">Bez náhledu</span>
                                </div>
                            </div>
                        </div>
                        <div class="stitch-preview-tile" data-role="current">
                            <div class="stitch-preview-label">Aktuální strana</div>
                            <div class="stitch-preview-frame" data-role="current">
                                <div class="stitch-preview-box is-empty" tabindex="-1" aria-label="Aktuální strana">
                                    <img id="stitch-preview-current-thumb" class="stitch-preview-thumb-img" alt="Aktuální strana" src="" />
                                    <span class="stitch-preview-placeholder">Bez náhledu</span>
                                </div>
                            </div>
                        </div>
                        <div class="stitch-preview-tile" data-role="next">
                            <div class="stitch-preview-label">Následující strana</div>
                            <div class="stitch-preview-frame" data-role="next">
                                <div class="stitch-preview-box is-empty" tabindex="-1" aria-label="Následující strana">
                                    <img id="stitch-preview-next-thumb" class="stitch-preview-thumb-img" alt="Následující strana" src="" />
                                    <span class="stitch-preview-placeholder">Bez náhledu</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div id="stitch-merge-container" class="stitch-merge-container" style="display:none;">
                        <section class="stitch-merge-block" id="stitch-merge-start">
                            <h3>Začátek strany</h3>
                            <div class="stitch-merge-columns">
                                <div class="stitch-merge-column">
                                    <div class="stitch-merge-note">Poslední odstavec předchozí strany</div>
                                    <div id="stitch-start-previous" class="stitch-merge-snippet" aria-live="polite"></div>
                                    <div class="stitch-merge-note">První odstavec aktuální strany</div>
                                    <div id="stitch-start-current" class="stitch-merge-snippet" aria-live="polite"></div>
                                </div>
                                <div class="stitch-merge-column">
                                    <div class="stitch-merge-note">Spojená verze</div>
                                    <div id="stitch-start-merged" class="stitch-merge-snippet" aria-live="polite"></div>
                                </div>
                            </div>
                        </section>
                        <section class="stitch-merge-block" id="stitch-merge-end">
                            <h3>Konec strany</h3>
                            <div class="stitch-merge-columns">
                                <div class="stitch-merge-column">
                                    <div class="stitch-merge-note">Poslední odstavec aktuální strany</div>
                                    <div id="stitch-end-current" class="stitch-merge-snippet" aria-live="polite"></div>
                                    <div class="stitch-merge-note">První odstavec následující strany</div>
                                    <div id="stitch-end-next" class="stitch-merge-snippet" aria-live="polite"></div>
                                </div>
                                <div class="stitch-merge-column">
                                    <div class="stitch-merge-note">Spojená verze</div>
                                    <div id="stitch-end-merged" class="stitch-merge-snippet" aria-live="polite"></div>
                                </div>
                            </div>
                        </section>
                    </div>
                </div>
        <button id="preview-drawer-toggle" class="preview-drawer-toggle" type="button" aria-expanded="true" aria-controls="preview-drawer-panel" aria-label="Skrýt pevný náhled">&lt;</button>
        <aside id="preview-drawer" class="preview-drawer" aria-label="Pevný náhled aktuální stránky" aria-hidden="false">
            <div id="preview-drawer-panel" class="preview-drawer-panel">
                <div id="preview-pinned-placeholder" class="preview-drawer-empty">Náhled bude k dispozici po načtení.</div>
                <img id="preview-pinned-image" class="preview-drawer-image" alt="Stálý náhled aktuální stránky" aria-hidden="true">
            </div>
        </aside>
        </div>
        <div id="loading" class="loading" aria-live="polite" aria-hidden="true">
            <div class="loading-content">
                <div class="loading-spinner" role="presentation"></div>
                <p>Zpracovávám ALTO data...</p>
            </div>
            </div>
        </div>
    </div>

    <script type="application/json" id="model-registry-data">__MODEL_REGISTRY_DATA__</script>
    <script type="application/json" id="default-agent-model">__DEFAULT_AGENT_MODEL__</script>
    <script type="application/json" id="default-agent-prompt-data">__DEFAULT_AGENT_PROMPT__</script>
    <script>
        let previewObjectUrl = null;
        let previewFetchToken = null;
        let previewImageUuid = null;

        let currentBook = null;
        let currentPage = null;
        let currentLibrary = null;
        let navigationState = null;
        let processRequestToken = 0;

        const pageCache = new Map();
        const inflightProcessRequests = new Map();
        const previewCache = new Map();
        const inflightPreviewRequests = new Map();
        let cacheWindowUuids = new Set();
        let currentAltoXml = "";
        let bookPages = [];
        let lastRenderedBookUuid = null;
        let lastActiveThumbnailUuid = null;
        let thumbnailDrawerCollapsed = false;
        let previewDrawerCollapsed = false;
        let previewDrawerPositionFrame = null;
        let thumbnailObserver = null;
        const DIFF_MODES = {
            NONE: 'none',
            WORD: 'word',
            CHAR: 'char',
        };
        const DIFF_MODE_STORAGE_KEY = 'altoDiffMode';
        let diffMode = DIFF_MODES.NONE;
        const diffCache = new Map();
        const AGENT_DIFF_MODES = {
            WORD: 'word',
            CHAR: 'char',
        };
        const AGENT_DIFF_MODE_STORAGE_KEY = 'altoAgentDiffMode';
        const COMPARISON_DIFF_MODES = {
            WORD: 'word',
            CHAR: 'char',
        };
        const COMPARISON_DIFF_MODE_STORAGE_KEY = 'altoComparisonDiffMode';
        let comparisonDiffMode = COMPARISON_DIFF_MODES.WORD;
        let agentDiffMode = AGENT_DIFF_MODES.WORD;
        let lastAgentOriginalHtml = '';
        let lastAgentCorrectedHtml = '';
        let lastAgentCorrectedIsHtml = false;
        let lastAgentOriginalDocumentJson = '';
        let lastAgentCorrectedDocumentJson = '';
        let lastAgentCacheBaseKey = null;
        let agentDiffRequestToken = 0;
        let lastReaderResultHtml = '';
        let lastReaderResultIsHtml = false;
        const comparisonOutputs = {
            leftHtml: '',
            leftIsHtml: false,
            leftDisplay: '',
            rightHtml: '',
            rightIsHtml: false,
            rightDisplay: '',
            diffCache: {
                word: null,
                char: null,
            },
        };
        // Secondary comparison state (processed ALTO (Python) vs OCR)
        const comparison2Outputs = {
            leftHtml: '',
            leftIsHtml: false,
            leftDisplay: '',
            rightHtml: '',
            rightIsHtml: false,
            rightDisplay: '',
            diffCache: {
                word: null,
                char: null,
            },
        };
        let comparison2DiffRequestToken = 0;
        const COMPARISON2_DIFF_MODE_STORAGE_KEY = 'altoComparison2DiffMode';
        let comparison2DiffMode = COMPARISON_DIFF_MODES.WORD;
        const comparison2State = {
            running: false,
            autoRunScheduled: false,
        };
        const comparisonResultWaiters = {
            corrector: [],
            reader: [],
        };
        let comparisonDiffRequestToken = 0;
        const comparisonState = {
            running: false,
            autoRunScheduled: false,
            correctorRunPending: false,
            readerRunPending: false,
        };
        let comparisonRunButtonLabel = '';
        let stitchScaleResizeBound = false;
        let thumbnailHeightSyncPending = false;
        let currentResults = {
            python: "",
            typescript: "",
            baseKey: "",
        };
        const tooltipTimers = new Map();
        const NOTE_STYLE_ATTR = ' style=\"display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;\"';
        // --- LLM agent management ---
        function createAgentContext(config) {
            return {
                collection: config.collection,
                selectId: config.selectId,
                selectSpinnerId: config.selectSpinnerId,
                autoCheckboxId: config.autoCheckboxId,
                runButtonId: config.runButtonId,
                runButtonExtraIds: config.runButtonExtraIds || [],
                expandToggleId: config.expandToggleId,
                settingsId: config.settingsId,
                nameInputId: config.nameInputId,
                promptTextareaId: config.promptTextareaId,
                modelSelectId: config.modelSelectId,
                parameterFieldsId: config.parameterFieldsId,
                saveButtonId: config.saveButtonId,
                deleteButtonId: config.deleteButtonId,
                saveTooltipId: config.saveTooltipId,
                outputContainerId: config.outputContainerId,
                outputStatusId: config.outputStatusId,
                outputTextId: config.outputTextId,
                resultsContainerId: config.resultsContainerId,
                resultOriginalId: config.resultOriginalId,
                resultCorrectedId: config.resultCorrectedId,
                diffControlsId: config.diffControlsId,
                diffToggleSelector: config.diffToggleSelector,
                autoStorageKey: config.autoStorageKey,
                selectedStorageKey: config.selectedStorageKey,
                hasOutputPanel: Boolean(config.hasOutputPanel),
                supportsDiff: Boolean(config.supportsDiff),
                requiresPythonHtml: config.requiresPythonHtml !== false,
                runButtonLabel: config.runButtonLabel || 'Spustit',
                agentsCache: {},
                agentFingerprintCache: new Map(),
                agentResultCache: new Map(),
                currentAgentName: '',
                currentAgentDraft: null,
                agentDraftDirty: false,
                autoRunScheduled: false,
                stitchContext: null,
                lastJoinerResult: null,
            };
        }

        const agentContexts = {
            correctors: createAgentContext({
                collection: 'correctors',
                selectId: 'agent-select',
                selectSpinnerId: 'agent-select-spinner',
                autoCheckboxId: 'agent-auto-correct',
                runButtonId: 'agent-run',
                runButtonExtraIds: ['agent-run-inline'],
                expandToggleId: 'agent-expand-toggle',
                settingsId: 'agent-settings',
                nameInputId: 'agent-name',
                promptTextareaId: 'agent-prompt',
                modelSelectId: 'agent-default-model',
                parameterFieldsId: 'agent-parameter-fields',
                saveButtonId: 'agent-save',
                deleteButtonId: 'agent-delete',
                saveTooltipId: 'agent-save-tooltip',
                outputContainerId: 'agent-output',
                outputStatusId: 'agent-output-status',
                outputTextId: 'agent-output-text',
                resultsContainerId: 'agent-results',
                resultOriginalId: 'agent-result-original',
                resultCorrectedId: 'agent-result-corrected',
                diffControlsId: 'agent-diff-mode-controls',
                diffToggleSelector: '.agent-diff-toggle',
                autoStorageKey: 'altoAgentAutoCorrect_v1',
                selectedStorageKey: 'altoAgentSelected_v1',
                hasOutputPanel: true,
                supportsDiff: true,
                requiresPythonHtml: true,
                runButtonLabel: 'Oprav',
            }),
            readers: createAgentContext({
                collection: 'readers',
                selectId: 'reader-agent-select',
                selectSpinnerId: 'reader-agent-select-spinner',
                autoCheckboxId: 'reader-agent-auto-read',
                runButtonId: 'reader-agent-run',
                runButtonExtraIds: ['reader-agent-run-inline'],
                expandToggleId: 'reader-agent-expand-toggle',
                settingsId: 'reader-agent-settings',
                nameInputId: 'reader-agent-name',
                promptTextareaId: 'reader-agent-prompt',
                modelSelectId: 'reader-agent-default-model',
                parameterFieldsId: 'reader-agent-parameter-fields',
                saveButtonId: 'reader-agent-save',
                deleteButtonId: 'reader-agent-delete',
                saveTooltipId: 'reader-agent-save-tooltip',
                outputContainerId: 'reader-agent-output',
                outputStatusId: 'reader-agent-output-status',
                outputTextId: 'reader-agent-output-text',
                autoStorageKey: 'altoReaderAuto_v1',
                selectedStorageKey: 'altoReaderSelected_v1',
                hasOutputPanel: true,
                supportsDiff: false,
                requiresPythonHtml: false,
                runButtonLabel: 'Vyčti',
            }),
            joiners: createAgentContext({
                collection: 'joiners',
                selectId: 'stitch-agent-select',
                selectSpinnerId: 'stitch-agent-select-spinner',
                autoCheckboxId: 'stitch-agent-auto-link',
                runButtonId: 'stitch-agent-run',
                runButtonExtraIds: ['stitch-agent-run-inline'],
                expandToggleId: 'stitch-agent-expand-toggle',
                settingsId: 'stitch-agent-settings',
                nameInputId: 'stitch-agent-name',
                promptTextareaId: 'stitch-agent-prompt',
                modelSelectId: 'stitch-agent-default-model',
                parameterFieldsId: 'stitch-agent-parameter-fields',
                saveButtonId: 'stitch-agent-save',
                deleteButtonId: 'stitch-agent-delete',
                saveTooltipId: 'stitch-agent-save-tooltip',
                outputContainerId: 'stitch-agent-output',
                outputStatusId: 'stitch-agent-output-status',
                outputTextId: 'stitch-agent-output-text',
                autoStorageKey: 'altoStitchAutoLink_v1',
                selectedStorageKey: 'altoStitchSelected_v1',
                hasOutputPanel: true,
                supportsDiff: false,
                requiresPythonHtml: true,
                runButtonLabel: 'Napoj',
            }),
        };

        const stitchState = {
            context: null,
            renderToken: 0,
        };

        function getContextElement(ctx, key) {
            const id = ctx[key];
            if (!id) {
                return null;
            }
            return document.getElementById(id);
        }

        function persistContextSelectedAgent(ctx, name) {
            const key = ctx.selectedStorageKey;
            if (!key) return;
            try { localStorage.setItem(key, name || ''); } catch (e) {}
        }

        function loadContextSelectedAgent(ctx) {
            const key = ctx.selectedStorageKey;
            if (!key) return '';
            try { return localStorage.getItem(key) || ''; } catch (e) { return ''; }
        }

        function persistContextAutoTrigger(ctx, value) {
            const key = ctx.autoStorageKey;
            if (!key) return;
            try { localStorage.setItem(key, value ? '1' : '0'); } catch (e) {}
        }

        function loadContextAutoTrigger(ctx) {
            const key = ctx.autoStorageKey;
            if (!key) return false;
            try { return localStorage.getItem(key) === '1'; } catch (e) { return false; }
        }

        function buildAgentRequestUrl(path, ctx, extraParams = {}) {
            const params = new URLSearchParams();
            params.set('collection', ctx.collection || 'correctors');
            Object.keys(extraParams || {}).forEach((key) => {
                if (extraParams[key] !== undefined && extraParams[key] !== null) {
                    params.set(key, extraParams[key]);
                }
            });
            return `${path}?${params.toString()}`;
        }

        function buildAgentPayload(ctx, payload = {}) {
            return {
                ...payload,
                collection: ctx.collection || 'correctors',
            };
        }

        function findPageInBook(uuid) {
            if (!uuid || !Array.isArray(bookPages) || !bookPages.length) {
                return null;
            }
            return bookPages.find((page) => page && page.uuid === uuid) || null;
        }

        function mergePageInfo(primary, fallback) {
            if (!primary && !fallback) {
                return null;
            }
            const result = Object.assign({}, fallback || {});
            if (primary) {
                Object.assign(result, primary);
            }
            if (typeof result.index !== 'number' || Number.isNaN(result.index)) {
                const primaryIndex = primary && typeof primary.index === 'number' ? primary.index : null;
                const fallbackIndex = fallback && typeof fallback.index === 'number' ? fallback.index : null;
                result.index = primaryIndex !== null ? primaryIndex : fallbackIndex;
            }
            if (!result.uuid && primary && primary.uuid) {
                result.uuid = primary.uuid;
            }
            if (!result.uuid && fallback && fallback.uuid) {
                result.uuid = fallback.uuid;
            }
            result.pageNumber = result.pageNumber || result.page_number || result.pagenumber || '';
            result.title = result.title || '';
            result.thumbnail = result.thumbnail || '';
            if (!result.thumbnail && result.uuid) {
                result.thumbnail = `/preview?uuid=${encodeURIComponent(result.uuid)}&stream=IMG_THUMB`;
            }
            return result;
        }

        function extractParagraphSnippetsFromHtml(html) {
            if (!html || typeof html !== 'string') {
                return [];
            }
            if (typeof DOMParser === 'undefined') {
                return [];
            }
            let doc;
            try {
                const parser = new DOMParser();
                doc = parser.parseFromString(html, 'text/html');
            } catch (error) {
                console.warn('Nepodařilo se parsovat HTML pro napojování stran:', error);
                return [];
            }
            if (!doc || !doc.body) {
                return [];
            }
            const snippets = [];
            const pushElements = (selector) => {
                doc.body.querySelectorAll(selector).forEach((el) => {
                    if (!(el instanceof HTMLElement)) {
                        return;
                    }
                    if (el.closest('note')) {
                        return;
                    }
                    const text = (el.textContent || '').replace(/\s+/g, ' ').trim();
                    if (!text) {
                        return;
                    }
                    snippets.push({
                        text,
                        html: el.outerHTML || escapeHtml(text),
                        tag: (el.tagName || '').toLowerCase(),
                    });
                });
            };
            pushElements('p');
            if (!snippets.length) {
                pushElements('div');
                pushElements('blockquote');
                pushElements('li');
                pushElements('h1, h2, h3, h4, h5, h6');
            }
            return snippets;
        }

        function normalizeSnippetWithPage(snippet, pageInfo) {
            if (!snippet) {
                return null;
            }
            const normalized = {
                text: snippet.text || '',
                html: snippet.html || (snippet.text ? escapeHtml(snippet.text) : ''),
                tag: snippet.tag || '',
                page_uuid: pageInfo && pageInfo.uuid ? pageInfo.uuid : '',
                page_number: pageInfo && pageInfo.pageNumber ? pageInfo.pageNumber : '',
                page_index: pageInfo && typeof pageInfo.index === 'number' ? pageInfo.index : null,
                page_title: pageInfo && pageInfo.title ? pageInfo.title : '',
            };
            if (!normalized.text.trim()) {
                return null;
            }
            return normalized;
        }

        function renderStitchSnippet(elementId, snippet, options = {}) {
            const el = document.getElementById(elementId);
            if (!el) {
                return;
            }
            const settings = Object.assign({
                allowHtml: true,
                placeholder: '<div class="muted">Není k dispozici.</div>',
            }, options || {});
            if (!snippet) {
                el.innerHTML = settings.placeholder;
                return;
            }
            if (settings.allowHtml && snippet.html) {
                el.innerHTML = snippet.html;
            } else if (snippet.text) {
                el.textContent = snippet.text;
            } else {
                el.innerHTML = settings.placeholder;
            }
        }

        function extractSnippetTag(snippet) {
            if (!snippet || typeof snippet !== 'object') {
                return '';
            }
            if (snippet.tag && typeof snippet.tag === 'string' && snippet.tag.trim()) {
                return snippet.tag.trim().toLowerCase();
            }
            if (snippet.html && typeof snippet.html === 'string') {
                const match = snippet.html.match(/^<\s*([a-zA-Z0-9:-]+)/);
                if (match) {
                    return match[1].toLowerCase();
                }
            }
            return '';
        }

        function buildSplitSnippetPreview(section, sectionName) {
            if (!section) {
                return null;
            }
            const firstKey = sectionName === 'start' ? 'previous' : 'current';
            const secondKey = sectionName === 'start' ? 'current' : 'next';
            const first = section[firstKey];
            const second = section[secondKey];
            if (!first && !second) {
                return null;
            }
            const htmlParts = [];
            const textParts = [];
            const pushSnippet = (snippet) => {
                if (!snippet) {
                    return;
                }
                const rawText = typeof snippet.text === 'string' ? snippet.text : '';
                const normalizedText = rawText.replace(/\s+/g, ' ').trim();
                if (normalizedText) {
            textParts.push(normalizedText);
        }
        if (typeof snippet.html === 'string' && snippet.html.trim()) {
            htmlParts.push(snippet.html);
        } else if (normalizedText) {
            htmlParts.push(`<p>${escapeHtml(normalizedText)}</p>`);
                }
            };
            pushSnippet(first);
            pushSnippet(second);
    if (!htmlParts.length && !textParts.length) {
        return null;
    }
    const base = first || second || {};
    return {
        html: htmlParts.join(''),
        text: textParts.join('\\n\\n'),
                tag: '',
                page_uuid: base.page_uuid || '',
                page_number: base.page_number || '',
                page_index: base.page_index,
                page_title: base.page_title || '',
                split: true,
            };
        }

        function computeJoinedSnippetFromContext(stitchContext, sectionName) {
            if (!stitchContext || !stitchContext.sections || !stitchContext.sections[sectionName]) {
                return null;
            }
            const section = stitchContext.sections[sectionName];
            const firstKey = sectionName === 'start' ? 'previous' : 'current';
            const secondKey = sectionName === 'start' ? 'current' : 'next';
            const first = section[firstKey];
            const second = section[secondKey];
            if (!first || !second) {
                return null;
            }
            const normalizeText = (value) => (value || '').replace(/\s+/g, ' ').trim();
            const textA = normalizeText(first.text);
            const textB = normalizeText(second.text);
            if (!textA && !textB) {
                return null;
            }
            let mergedText = '';
            if (textA && textB) {
                const hyphenSplitMatch = textA.match(/\s*[\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]\s*$/);
                if (hyphenSplitMatch) {
                    const trimmedFirst = textA.slice(0, textA.length - hyphenSplitMatch[0].length);
                    mergedText = `${trimmedFirst}${textB}`;
                } else if (HYPHEN_LIKE_REGEX.test(textA.slice(-1))) {
                    mergedText = `${textA.slice(0, -1).trimEnd()}${textB}`;
                } else if (/^[,.;:!?)]/.test(textB)) {
                    mergedText = `${textA.trimEnd()}${textB}`;
                } else {
                    mergedText = `${textA.trimEnd()} ${textB}`.trim();
                }
            } else {
                mergedText = textA || textB;
            }
            if (!mergedText) {
                return null;
            }
            const baseSnippet = first.html ? first : (second.html ? second : null);
            let mergedHtml = '';
            if (baseSnippet && baseSnippet.html) {
                const openTagMatch = baseSnippet.html.match(/^<([a-zA-Z0-9:-]+)([^>]*)>/);
                const tagName = openTagMatch ? openTagMatch[1].toLowerCase() : (baseSnippet.tag || '');
                const attrPart = openTagMatch && openTagMatch[2] ? openTagMatch[2].trim() : '';
                const normalizedTag = (tagName || baseSnippet.tag || '').trim();
                if (normalizedTag) {
                    const safeAttrs = attrPart ? ` ${attrPart}` : '';
                    mergedHtml = `<${normalizedTag}${safeAttrs}>${escapeHtml(mergedText)}</${normalizedTag}>`;
                } else {
                    mergedHtml = escapeHtml(mergedText);
                }
            } else {
                mergedHtml = escapeHtml(mergedText);
            }
            return {
                text: mergedText,
                html: mergedHtml,
                tag: extractSnippetTag(first) || extractSnippetTag(second),
                page_uuid: first.page_uuid || second.page_uuid || '',
                page_number: first.page_number || second.page_number || '',
                page_index: typeof second.page_index === 'number'
                    ? second.page_index
                    : (typeof first.page_index === 'number' ? first.page_index : null),
                page_title: second.page_title || first.page_title || '',
            };
        }

        function applyJoinerActionToSection(ctx, sectionName, action, options = {}) {
            if (!ctx || !ctx.stitchContext || !ctx.stitchContext.sections) {
                return false;
            }
            const section = ctx.stitchContext.sections[sectionName];
            if (!section) {
                return false;
            }
            const elementId = sectionName === 'start' ? 'stitch-start-merged' : 'stitch-end-merged';
            if (action === 'join' || action === 'merge') {
                const mergedSnippet = computeJoinedSnippetFromContext(ctx.stitchContext, sectionName);
                if (!mergedSnippet) {
                    section.merged = null;
                    renderStitchSnippet(elementId, null, { allowHtml: false, placeholder: '<div class="muted">Bez výsledku.</div>' });
                    return false;
                }
                section.merged = mergedSnippet;
                renderStitchSnippet(elementId, mergedSnippet, { allowHtml: Boolean(mergedSnippet.html), placeholder: '<div class="muted">Bez výsledku.</div>' });
                return true;
            }
            if (action === 'split') {
                const preview = buildSplitSnippetPreview(section, sectionName);
                if (preview) {
                    section.merged = preview;
                    renderStitchSnippet(elementId, preview, { allowHtml: Boolean(preview.html), placeholder: '<div class="muted">Bez výsledku.</div>' });
                } else {
                    section.merged = null;
                    const placeholderMessage = (options && typeof options.placeholder === 'string' && options.placeholder.trim())
                        ? options.placeholder.trim()
                        : 'Bez výsledku.';
                    renderStitchSnippet(elementId, null, { allowHtml: false, placeholder: `<div class="muted">${escapeHtml(placeholderMessage)}</div>` });
                }
                return true;
            }
            return false;
        }

        function finalizeJoinerUiAfterActions(ctx, rawText, info = {}) {
            if (!ctx) {
                return;
            }
            const mergeContainer = document.getElementById('stitch-merge-container');
            if (mergeContainer && mergeContainer.style.display === 'none') {
                mergeContainer.style.display = 'flex';
            }
            const focusTarget = document.getElementById('stitch-merge-start');
            if (focusTarget && typeof focusTarget.scrollIntoView === 'function') {
                focusTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            scheduleThumbnailDrawerHeightSync();
            ctx.lastJoinerResult = {
                rawText,
                decision: info.decision || null,
                source: info.source || 'agent',
                timestamp: Date.now(),
            };
        }

        function computeManualPairDecision(first, second) {
            if (!first || !second) {
                return '0';
            }
            const tagA = extractSnippetTag(first);
            const tagB = extractSnippetTag(second);
            if (tagA && tagB && tagA !== tagB) {
                return '0';
            }
            const textA = (first.text || '').trim();
            const textB = (second.text || '').trim();
            if (!textA || !textB) {
                return '0';
            }
            if (SENTENCE_END_REGEX.test(textA) && SENTENCE_START_REGEX.test(textB)) {
                return '0';
            }
            if (TRAILING_HYPHEN_SEQUENCE_REGEX.test(textA)) {
                return '2';
            }
            return '1';
        }

        function runManualJoiner(ctx, stitchContext) {
            if (!ctx || !stitchContext || !stitchContext.sections) {
                return null;
            }
            const decisions = { start: null, end: null };
            const startSection = stitchContext.sections.start;
            if (startSection && startSection.previous && startSection.current) {
                const decision = computeManualPairDecision(startSection.previous, startSection.current);
                decisions.start = decision;
                const action = decision === '2' ? 'merge' : (decision === '1' ? 'join' : (decision === '0' ? 'split' : null));
                if (action) {
                    const message = action === 'split' ? 'Manual joiner: ponecháno oddělené.' : '';
                    applyJoinerActionToSection(ctx, 'start', action, { message });
                }
            } else if (startSection) {
                applyJoinerActionToSection(ctx, 'start', 'split', { message: 'Manual joiner: chybí data pro spojení.' });
                decisions.start = '0';
            }
            const endSection = stitchContext.sections.end;
            if (endSection && endSection.current && endSection.next) {
                const decision = computeManualPairDecision(endSection.current, endSection.next);
                decisions.end = decision;
                const action = decision === '2' ? 'merge' : (decision === '1' ? 'join' : (decision === '0' ? 'split' : null));
                if (action) {
                    const message = action === 'split' ? 'Manual joiner: ponecháno oddělené.' : '';
                    applyJoinerActionToSection(ctx, 'end', action, { message });
                }
            } else if (endSection) {
                applyJoinerActionToSection(ctx, 'end', 'split', { message: 'Manual joiner: chybí data pro spojení.' });
                decisions.end = '0';
            }
            finalizeJoinerUiAfterActions(ctx, 'manual-joiner', {
                source: 'manual',
                decision: decisions,
            });
            return decisions;
        }

        function buildJoinerAgentPayloadFromContext(context) {
            if (!context || !context.sections) {
                return null;
            }

            const serializeSnippet = (snippet, blockId, role) => {
                if (!snippet || typeof snippet.text !== 'string') {
                    return null;
                }
                const text = snippet.text.replace(/\s+/g, ' ').trim();
                if (!text) {
                    return null;
                }
                const tag = (snippet.tag || '').trim().toLowerCase();
                return {
                    id: blockId,
                    role,
                    tag,
                    text,
                };
            };

            const startSection = context.sections.start || {};
            const endSection = context.sections.end || {};

            const payloadPairs = {};

            const startPrevious = serializeSnippet(startSection.previous, 'start-previous', 'previous');
            const startCurrent = serializeSnippet(startSection.current, 'start-current', 'current');
            if (startPrevious && startCurrent) {
                payloadPairs.start = {
                    previous: startPrevious,
                    current: startCurrent,
                };
            }

            const endCurrent = serializeSnippet(endSection.current, 'end-current', 'current');
            const endNext = serializeSnippet(endSection.next, 'end-next', 'next');
            if (endCurrent && endNext) {
                payloadPairs.end = {
                    current: endCurrent,
                    next: endNext,
                };
            }

            if (!Object.keys(payloadPairs).length) {
                return null;
            }

            return {
                language_hint: DEFAULT_LANGUAGE_HINT,
                pairs: payloadPairs,
            };
        }


        async function buildStitchContext() {
            if (!currentPage || !currentPage.uuid) {
                return null;
            }
            const prevUuid = navigationState && navigationState.prevUuid ? navigationState.prevUuid : null;
            const nextUuid = navigationState && navigationState.nextUuid ? navigationState.nextUuid : null;

            const [prevData, nextData] = await Promise.all([
                prevUuid ? ensureProcessData(prevUuid).catch((error) => {
                    console.warn('Nepodařilo se načíst předchozí stranu pro napojení:', error);
                    return null;
                }) : Promise.resolve(null),
                nextUuid ? ensureProcessData(nextUuid).catch((error) => {
                    console.warn('Nepodařilo se načíst následující stranu pro napojení:', error);
                    return null;
                }) : Promise.resolve(null),
            ]);

            const currentHtml = currentResults && typeof currentResults.python === 'string' ? currentResults.python : '';
            const prevHtml = prevData && typeof prevData.python === 'string' ? prevData.python : '';
            const nextHtml = nextData && typeof nextData.python === 'string' ? nextData.python : '';

            const prevPageInfo = mergePageInfo(prevData && prevData.currentPage, findPageInBook(prevUuid));
            const currentPageInfo = mergePageInfo(currentPage, findPageInBook(currentPage.uuid));
            const nextPageInfo = mergePageInfo(nextData && nextData.currentPage, findPageInBook(nextUuid));

            const prevSnippets = extractParagraphSnippetsFromHtml(prevHtml);
            const currentSnippets = extractParagraphSnippetsFromHtml(currentHtml);
            const nextSnippets = extractParagraphSnippetsFromHtml(nextHtml);

            const startPrevious = prevSnippets.length ? prevSnippets[prevSnippets.length - 1] : null;
            const startCurrent = currentSnippets.length ? currentSnippets[0] : null;
            const endCurrent = currentSnippets.length ? currentSnippets[currentSnippets.length - 1] : null;
            const endNext = nextSnippets.length ? nextSnippets[0] : null;

            return {
                meta: {
                    current_uuid: currentPage && currentPage.uuid ? currentPage.uuid : '',
                    previous_uuid: prevUuid || '',
                    next_uuid: nextUuid || '',
                },
                pages: {
                    previous: prevPageInfo,
                    current: currentPageInfo,
                    next: nextPageInfo,
                },
                sections: {
                    start: {
                        previous: normalizeSnippetWithPage(startPrevious, prevPageInfo),
                        current: normalizeSnippetWithPage(startCurrent, currentPageInfo),
                        merged: null,
                    },
                    end: {
                        current: normalizeSnippetWithPage(endCurrent, currentPageInfo),
                        next: normalizeSnippetWithPage(endNext, nextPageInfo),
                        merged: null,
                    },
                },
            };
        }

        async function updateStitchPreviewTiles(context) {
            const roles = ['previous', 'current', 'next'];
            await Promise.all(roles.map(async (role) => {
                const tile = document.querySelector(`.stitch-preview-tile[data-role="${role}"]`);
                if (!tile) {
                    return;
                }
                const page = context && context.pages ? context.pages[role] : null;
                const frame = tile.querySelector('.stitch-preview-frame');
                const box = tile.querySelector('.stitch-preview-box');
                const thumb = tile.querySelector('.stitch-preview-thumb-img');
                const placeholder = tile.querySelector('.stitch-preview-placeholder');
                if (!page || !page.uuid) {
                    tile.style.display = 'none';
                    if (frame) {
                        frame.dataset.uuid = '';
                        frame.classList.remove('is-active');
                    }
                    if (box) {
                        box.classList.add('is-empty');
                        box.setAttribute('tabindex', '-1');
                        box.dataset.uuid = '';
                        box.onclick = null;
                        box.onkeydown = null;
                    }
                    if (thumb) {
                        thumb.src = '';
                        thumb.style.display = 'none';
                    }
                    if (large) {
                        large.src = '';
                    }
                    if (placeholder) {
                        placeholder.style.display = 'block';
                    }
                    return;
                }
                tile.style.display = 'flex';
                if (frame) {
                    frame.dataset.uuid = page.uuid;
                }
                if (placeholder) {
                    placeholder.style.display = 'none';
                }
                if (box) {
                    box.dataset.uuid = page.uuid;
                    box.classList.remove('is-empty');
                    box.setAttribute('tabindex', '0');
                    const ariaLabel = page.pageNumber ? `Strana ${page.pageNumber}` : 'Náhled stránky';
                    box.setAttribute('aria-label', ariaLabel);
                    box.onclick = (event) => {
                        event.preventDefault();
                        navigateToUuid(page.uuid);
                    };
                    box.onkeydown = (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            navigateToUuid(page.uuid);
                        }
                    };
                }
                let src = page.thumbnail || '';
                try {
                    const entry = await ensurePreviewEntry(page.uuid);
                    if (entry && entry.objectUrl) {
                        src = entry.objectUrl;
                    }
                } catch (error) {
                    if (!src) {
                        src = `/preview?uuid=${encodeURIComponent(page.uuid)}&stream=IMG_THUMB`;
                    }
                }
                if (!src && page.uuid) {
                    src = `/preview?uuid=${encodeURIComponent(page.uuid)}&stream=IMG_THUMB`;
                }
                if (thumb) {
                    if (src) {
                        if (thumb.src !== src) {
                            thumb.src = src;
                        }
                        thumb.style.display = 'block';
                    } else {
                        thumb.src = '';
                        thumb.style.display = 'none';
                    }
                    thumb.alt = page.pageNumber ? `Strana ${page.pageNumber}` : 'Náhled stránky';
                }
                if (frame) {
                    bindStitchPreviewHover(frame);
                    if (box) {
                        updateStitchFrameScale(frame, box);
                    }
                }
                if (thumb) {
                    thumb.addEventListener('load', () => {
                        if (frame && box) {
                            updateStitchFrameScale(frame, box);
                        }
                    });
                }
            }));
        }

        function bindStitchPreviewHover(frame) {
            if (!frame || frame.dataset.hoverBound === 'true') {
                return;
            }
            const box = frame.querySelector('.stitch-preview-box');
            if (!box) {
                return;
            }
            const activate = () => {
                frame.classList.add('is-active');
            };
            const deactivate = () => {
                frame.classList.remove('is-active');
            };
            ['pointerenter', 'mouseenter', 'focusin'].forEach((eventName) => {
                box.addEventListener(eventName, activate);
            });
            ['pointerleave', 'mouseleave', 'focusout', 'blur'].forEach((eventName) => {
                box.addEventListener(eventName, (event) => {
                    if (event.relatedTarget && frame.contains(event.relatedTarget)) {
                        return;
                    }
                    deactivate();
                });
            });
            frame.dataset.hoverBound = 'true';
        }

        function updateStitchFrameScale(frame, box) {
            if (!frame || !box) {
                return;
            }
            const host = document.querySelector('.main-content') || document.querySelector('.container') || document.body;
            const availableWidth = host ? host.clientWidth : window.innerWidth;
            const rect = box.getBoundingClientRect();
            const baseWidth = rect.width || 1;
            const targetWidth = Math.max(availableWidth * 0.5, baseWidth);
            const scale = Math.min(Math.max(targetWidth / baseWidth, 1.1), 4.0);
            frame.style.setProperty('--stitch-expand-scale', scale.toFixed(3));
        }

        function updateAllStitchScales() {
            document.querySelectorAll('.stitch-preview-frame').forEach((frame) => {
                const box = frame.querySelector('.stitch-preview-box');
                if (!box) {
                    return;
                }
                updateStitchFrameScale(frame, box);
            });
        }

        function updateStitchSnippets(context, options = {}) {
            const settings = Object.assign({ resetMerged: true }, options || {});
            const startSection = context && context.sections ? context.sections.start : null;
            const endSection = context && context.sections ? context.sections.end : null;
            renderStitchSnippet('stitch-start-previous', startSection ? startSection.previous : null, { allowHtml: true });
            renderStitchSnippet('stitch-start-current', startSection ? startSection.current : null, { allowHtml: true });
            renderStitchSnippet('stitch-end-current', endSection ? endSection.current : null, { allowHtml: true });
            renderStitchSnippet('stitch-end-next', endSection ? endSection.next : null, { allowHtml: true });
            const mergedPlaceholder = '<div class="muted">Klikněte na „Napoj“ pro vytvoření spojení.</div>';
            if (settings.resetMerged) {
                renderStitchSnippet('stitch-start-merged', null, { allowHtml: false, placeholder: mergedPlaceholder });
                renderStitchSnippet('stitch-end-merged', null, { allowHtml: false, placeholder: mergedPlaceholder });
            } else {
                renderStitchSnippet('stitch-start-merged', startSection ? startSection.merged : null, { allowHtml: Boolean(startSection && startSection.merged && startSection.merged.html), placeholder: mergedPlaceholder });
                renderStitchSnippet('stitch-end-merged', endSection ? endSection.merged : null, { allowHtml: Boolean(endSection && endSection.merged && endSection.merged.html), placeholder: mergedPlaceholder });
            }
        }

        async function refreshStitchUI(options = {}) {
            const previewRow = document.getElementById('stitch-preview-row');
            const mergeContainer = document.getElementById('stitch-merge-container');
            const runBtn = getContextElement(agentContexts.joiners, 'runButtonId');
            const reveal = Boolean(options && options.reveal);
            const keepVisible = Boolean(options && options.keepVisible);
            const keepMerged = Boolean(options && options.keepMerged);
            const token = ++stitchState.renderToken;
            let context = null;
            try {
                context = await buildStitchContext();
            } catch (error) {
                console.warn('Nepodařilo se připravit podklady pro napojení stran:', error);
            }
            if (token !== stitchState.renderToken) {
                return stitchState.context;
            }
            stitchState.context = context;
            agentContexts.joiners.stitchContext = context;

            if (!context) {
                if (previewRow) previewRow.style.display = 'none';
                if (!keepVisible && mergeContainer) mergeContainer.style.display = 'none';
                if (runBtn) runBtn.disabled = true;
                return null;
            }

            await updateStitchPreviewTiles(context);
            updateAllStitchScales();
            updateStitchSnippets(context, { resetMerged: !keepMerged });

            if (previewRow) {
                previewRow.style.display = 'flex';
            }
            if (mergeContainer && (reveal || (keepVisible && mergeContainer.style.display !== 'none'))) {
                mergeContainer.style.display = 'flex';
            } else if (!keepVisible && mergeContainer) {
                mergeContainer.style.display = 'none';
            }

            const hasStart = Boolean(context.sections.start.previous && context.sections.start.current);
            const hasEnd = Boolean(context.sections.end.current && context.sections.end.next);
            if (runBtn) {
                runBtn.disabled = !hasStart && !hasEnd;
            }

            scheduleThumbnailDrawerHeightSync();
            return context;
        }

        function applyJoinerAgentResult(ctx, resultText) {
            if (!ctx || !ctx.stitchContext) {
                return false;
            }
            if (!resultText || typeof resultText !== 'string') {
                return false;
            }
            const rawText = resultText.trim();
            if (!rawText) {
                return false;
            }

            const mapActionLabel = (label) => {
                if (label === undefined || label === null) {
                    return null;
                }
                const normalized = String(label).trim().toLowerCase();
                if (!normalized) {
                    return null;
                }
                if (normalized === 'join' || normalized === '1') {
                    return 'join';
                }
                if (normalized === 'merge' || normalized === '2') {
                    return 'merge';
                }
                if (normalized === 'split' || normalized === '0') {
                    return 'split';
                }
                return null;
            };

            const applyActionToAllSections = (action, tokenLabel) => {
                if (!action) {
                    return false;
                }
                const message = action === 'split'
                    ? `Agent doporučuje ponechat oddělené${tokenLabel ? ` (${tokenLabel})` : ''}.`
                    : '';
                const appliedSections = [];
                if (applyJoinerActionToSection(ctx, 'start', action, { message })) {
                    appliedSections.push('start');
                }
                if (applyJoinerActionToSection(ctx, 'end', action, { message })) {
                    appliedSections.push('end');
                }
                if (!appliedSections.length) {
                    return false;
                }
                const decision = {
                    start: appliedSections.includes('start') ? action : null,
                    end: appliedSections.includes('end') ? action : null,
                    token: tokenLabel,
                };
                finalizeJoinerUiAfterActions(ctx, rawText, {
                    source: 'tag',
                    decision,
                });
                return true;
            };

            const initialAction = mapActionLabel(rawText);
            if (applyActionToAllSections(initialAction, rawText)) {
                return true;
            }

            const stripJoinerCodeFences = (input) => {
                if (typeof input !== 'string') {
                    return input;
                }
                let text = input.trim();
                if (!text.startsWith('```')) {
                    return text;
                }
                const openingMatch = text.match(/^```[a-zA-Z0-9_-]*\s*/);
                if (openingMatch) {
                    text = text.slice(openingMatch[0].length);
                } else {
                    text = text.slice(3);
                }
                const closingIndex = text.lastIndexOf('```');
                if (closingIndex !== -1) {
                    text = text.slice(0, closingIndex);
                }
                return text.trim();
            };

            const safeParseJoinerJson = (input) => {
                if (typeof input !== 'string') {
                    return { parsed: null, error: null, usedRepair: false, repairedText: null };
                }
                const trimmed = stripJoinerCodeFences(input);
                if (!trimmed) {
                    return { parsed: null, error: null, usedRepair: false, repairedText: null };
                }
                try {
                    return { parsed: JSON.parse(trimmed), error: null, usedRepair: false, repairedText: null };
                } catch (error) {
                    const repaired = repairJsonStringLineBreaks(trimmed);
                    if (repaired && repaired !== trimmed) {
                        try {
                            console.warn('[JoinerDebug] Joiner JSON repaired before parse.');
                            return { parsed: JSON.parse(repaired), error: null, usedRepair: true, repairedText: repaired };
                        } catch (repairError) {
                            return { parsed: null, error: repairError, usedRepair: true, repairedText: repaired };
                        }
                    }
                    return { parsed: null, error, usedRepair: false, repairedText: null };
                }
            };

            const { parsed, error: parseError, usedRepair, repairedText } = safeParseJoinerJson(rawText);

            const applyPairsFormat = (pairsObject, sourceText) => {
                if (!pairsObject || typeof pairsObject !== 'object' || Array.isArray(pairsObject)) {
                    return false;
                }
                const decisions = { start: null, end: null };
                let applied = false;
                let hasValidAction = false;

                const updateSnippet = (pairKey, snippet, pairResult) => {
                    if (!snippet) {
                        return;
                    }
                    const merged = pairResult && typeof pairResult === 'object' ? pairResult.merged : null;
                    const providedText = merged && typeof merged.text === 'string'
                        ? merged.text
                        : (pairResult && typeof pairResult.text === 'string' ? pairResult.text : '');
                    const preferredTag = merged && typeof merged.tag === 'string' ? merged.tag : '';
                    const normalizedText = (providedText || '').trim();
                    if (normalizedText) {
                        snippet.text = normalizedText;
                        const tag = (preferredTag || snippet.tag || '').trim();
                        if (tag) {
                            snippet.tag = tag;
                            snippet.html = `<${tag}>${escapeHtml(normalizedText)}</${tag}>`;
                        } else {
                            snippet.html = escapeHtml(normalizedText);
                        }
                    }
                    const elementId = pairKey === 'start' ? 'stitch-start-merged' : 'stitch-end-merged';
                    renderStitchSnippet(
                        elementId,
                        snippet,
                        { allowHtml: Boolean(snippet.html), placeholder: '<div class="muted">Bez výsledku.</div>' }
                    );
                };

                const handlePair = (pairKey, pairResult) => {
                    if (!pairResult || typeof pairResult !== 'object') {
                        return;
                    }
                    const actionLabel = mapActionLabel(pairResult.action);
                    if (!actionLabel) {
                        return;
                    }
                    hasValidAction = true;
                    if (actionLabel === 'split') {
                        if (applyJoinerActionToSection(ctx, pairKey, 'split', {})) {
                            decisions[pairKey] = '0';
                            applied = true;
                        }
                        return;
                    }
                    const joinAction = actionLabel === 'merge' ? 'merge' : 'join';
                    if (applyJoinerActionToSection(ctx, pairKey, joinAction, {})) {
                        const section = ctx.stitchContext.sections ? ctx.stitchContext.sections[pairKey] : null;
                        const snippet = section ? section.merged : null;
                        if (snippet) {
                            updateSnippet(pairKey, snippet, pairResult);
                            decisions[pairKey] = '1';
                            applied = true;
                        }
                    }
                };

                handlePair('start', pairsObject.start);
                handlePair('end', pairsObject.end);

                if (applied) {
                    finalizeJoinerUiAfterActions(ctx, sourceText || rawText, {
                        source: 'pairs-json',
                        decision: decisions,
                    });
                }
                if (applied) {
                    return true;
                }
                return hasValidAction ? false : 'skip';
            };

            if (parsed === null || parsed === undefined) {
                if (initialAction && applyActionToAllSections(initialAction, rawText)) {
                    return true;
                }
                if (parseError) {
                    console.warn('Joiner agent nevrátil validní JSON:', parseError);
                    if (usedRepair && repairedText) {
                        console.warn('[JoinerDebug] Repaired joiner JSON candidate:', repairedText);
                    } else {
                        console.warn('[JoinerDebug] Raw joiner result text:', rawText);
                    }
                }
                const fallbackHtml = `<div class="muted">${escapeHtml(rawText)}</div>`;
                renderStitchSnippet('stitch-start-merged', null, { allowHtml: false, placeholder: fallbackHtml });
                renderStitchSnippet('stitch-end-merged', null, { allowHtml: false, placeholder: fallbackHtml });
                return false;
            }

            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) && parsed.pairs && typeof parsed.pairs === 'object') {
                const pairsOutcome = applyPairsFormat(parsed.pairs, rawText);
                if (pairsOutcome === true) {
                    return true;
                }
                if (pairsOutcome === 'skip') {
                    delete parsed.pairs;
                }
            }

            if (typeof parsed === 'string' || typeof parsed === 'number' || typeof parsed === 'boolean') {
                if (applyActionToAllSections(mapActionLabel(parsed), String(parsed))) {
                    return true;
                }
                return false;
            }

            if (Array.isArray(parsed)) {
                for (const item of parsed) {
                    if (typeof item === 'string' || typeof item === 'number') {
                        if (applyActionToAllSections(mapActionLabel(item), String(item))) {
                            return true;
                        }
                    } else if (item && typeof item === 'object') {
                        const decisionLabel = item.decision || item.action;
                        if (applyActionToAllSections(mapActionLabel(decisionLabel), decisionLabel)) {
                            return true;
                        }
                    }
                }
                return false;
            }

            const parsedSections = parsed.sections || parsed;
            if (!parsedSections || typeof parsedSections !== 'object') {
                if (applyActionToAllSections(mapActionLabel(rawText), rawText)) {
                    return true;
                }
                return false;
            }

            if (parsedSections.decision) {
                if (applyActionToAllSections(mapActionLabel(parsedSections.decision), parsedSections.decision)) {
                    return true;
                }
            }

            const normalizeMerged = (section) => {
                if (!section) {
                    return null;
                }
                if (typeof section === 'string') {
                    const trimmed = section.trim();
                    if (!trimmed) {
                        return null;
                    }
                    return { text: trimmed, html: '', tag: '' };
                }
                if (typeof section !== 'object') {
                    return null;
                }
                const candidateHtml = section.merged_html || section.mergedHtml || section.html;
                const textCandidate = section.merged_text || section.mergedText || section.merged || section.joined || section.result || section.text || '';
                let normalizedText = typeof textCandidate === 'string' ? textCandidate : '';
                let normalizedHtml = typeof candidateHtml === 'string' ? candidateHtml : '';
                const blockCandidates = Array.isArray(section.merged_blocks)
                    ? section.merged_blocks
                    : (Array.isArray(section.mergedBlocks)
                        ? section.mergedBlocks
                        : (Array.isArray(section.blocks) ? section.blocks : null));
                if (blockCandidates) {
                    if (!normalizedHtml.trim()) {
                        const html = documentBlocksToHtml({ blocks: blockCandidates });
                        if (html) {
                            normalizedHtml = html;
                        }
                    }
                    if (!normalizedText.trim()) {
                        const textFromBlocks = documentBlocksToText({ blocks: blockCandidates });
                        if (textFromBlocks) {
                            normalizedText = textFromBlocks;
                        }
                    }
                }
                if (!normalizedText.trim() && !normalizedHtml.trim()) {
                    return null;
                }
                return {
                    text: normalizedText,
                    html: normalizedHtml,
                    tag: section.tag || '',
                    page_uuid: section.page_uuid || '',
                    page_number: section.page_number || '',
                    page_index: section.page_index,
                    page_title: section.page_title || '',
                    split: Boolean(section.split),
                };
            };

            const decisions = { start: null, end: null };
            if (parsedSections.start) {
                const mergedStart = normalizeMerged(parsedSections.start.merged || parsedSections.start);
                ctx.stitchContext.sections.start.merged = mergedStart;
                decisions.start = mergedStart ? (mergedStart.split ? '0' : '1') : '0';
                renderStitchSnippet('stitch-start-merged', mergedStart, { allowHtml: Boolean(mergedStart && mergedStart.html), placeholder: '<div class="muted">Bez výsledku.</div>' });
            }
            if (parsedSections.end) {
                const mergedEnd = normalizeMerged(parsedSections.end.merged || parsedSections.end);
                ctx.stitchContext.sections.end.merged = mergedEnd;
                decisions.end = mergedEnd ? (mergedEnd.split ? '0' : '1') : '0';
                renderStitchSnippet('stitch-end-merged', mergedEnd, { allowHtml: Boolean(mergedEnd && mergedEnd.html), placeholder: '<div class="muted">Bez výsledku.</div>' });
            }
            finalizeJoinerUiAfterActions(ctx, rawText, {
                source: 'json',
                decision: decisions,
            });
            return true;
        }
        const MODEL_REGISTRY_DATA = (() => {
            const element = document.getElementById('model-registry-data');
            if (!element) {
                return { models: [], default_model: '' };
            }
            const raw = element.textContent || '';
            try {
                const parsed = JSON.parse(raw);
                return parsed && typeof parsed === 'object' ? parsed : { models: [], default_model: '' };
            } catch (err) {
                console.warn('Nelze načíst registr modelů:', err);
                return { models: [], default_model: '' };
            }
        })();
        const DEFAULT_REASONING_EFFORT = 'medium';
        const DEFAULT_TEMPERATURE = 0.2;
        const DEFAULT_TOP_P = 1.0;
        const ENABLE_RESPONSE_FORMAT = false; // sleeper feature – aktivuj, až OpenRouter začne schéma respektovat
        const FALLBACK_REASONING_PREFIXES = [
            'openai/gpt-5',
            'openai/o1',
            'openai/o3',
            'openai/o4',
            'gpt-5',
            'o1',
            'o3',
            'o4',
        ];
        const MODEL_CAPABILITIES_CACHE = new Map();

        function loadJsonIfString(value) {
            if (typeof value !== 'string') {
                return value;
            }
            const trimmed = value.trim();
            if (!trimmed.length) {
                return null;
            }
            try {
                return JSON.parse(trimmed);
            } catch (err) {
                return trimmed;
            }
        }

        function normalizeResponseFormat(value) {
            if (!ENABLE_RESPONSE_FORMAT) {
                return null;
            }
            const candidate = loadJsonIfString(value);
            if (!candidate) {
                return null;
            }
            if (typeof candidate === 'string') {
                const lowered = candidate.trim().toLowerCase();
                if (lowered === 'json_object') {
                    return { type: 'json_object' };
                }
                return null;
            }
            if (candidate && typeof candidate === 'object') {
                let typeValue = candidate.type;
                if (!typeValue && candidate.json_schema) {
                    typeValue = 'json_schema';
                } else if (!typeValue && (candidate.name || candidate.schema)) {
                    typeValue = 'json_schema';
                }
                if (typeof typeValue !== 'string') {
                    return null;
                }
                const normalizedType = typeValue.trim().toLowerCase();
                if (normalizedType === 'json_object') {
                    return { type: 'json_object' };
                }
                if (normalizedType === 'json_schema') {
                    let schemaPayload = candidate.json_schema;
                    if (!schemaPayload && (candidate.name || candidate.schema || typeof candidate.strict === 'boolean')) {
                        schemaPayload = {};
                        if (candidate.name !== undefined) schemaPayload.name = candidate.name;
                        if (candidate.schema !== undefined) schemaPayload.schema = candidate.schema;
                        if (typeof candidate.strict === 'boolean') schemaPayload.strict = candidate.strict;
                    }
                    schemaPayload = loadJsonIfString(schemaPayload);
                    if (!schemaPayload || typeof schemaPayload !== 'object') {
                        return null;
                    }
                    const nameValue = typeof schemaPayload.name === 'string' && schemaPayload.name.trim().length
                        ? schemaPayload.name.trim()
                        : '';
                    let schemaValue = schemaPayload.schema;
                    schemaValue = loadJsonIfString(schemaValue);
                    if (typeof schemaValue === 'string') {
                        try {
                            schemaValue = JSON.parse(schemaValue);
                        } catch (err) {
                            schemaValue = null;
                        }
                    }
                    if (!nameValue || !schemaValue || typeof schemaValue !== 'object') {
                        return null;
                    }
                    const sanitized = { name: nameValue, schema: schemaValue };
                    if (typeof schemaPayload.strict === 'boolean') {
                        sanitized.strict = schemaPayload.strict;
                    }
                    for (const [extraKey, extraValue] of Object.entries(schemaPayload)) {
                        if (extraKey === 'name' || extraKey === 'schema' || extraKey === 'strict') continue;
                        sanitized[extraKey] = extraValue;
                    }
                    return { type: 'json_schema', json_schema: sanitized };
                }
            }
            return null;
        }
        function normalizeModelId(model) {
            return String(model || '').trim().toLowerCase();
        }
        const MODEL_REGISTRY_MODELS = Array.isArray(MODEL_REGISTRY_DATA.models) ? MODEL_REGISTRY_DATA.models : [];
        const MODEL_REGISTRY_MAP = new Map();
        for (const entry of MODEL_REGISTRY_MODELS) {
            if (!entry || typeof entry !== 'object') {
                continue;
            }
            const normalized = normalizeModelId(entry.id);
            if (!normalized) {
                continue;
            }
            MODEL_REGISTRY_MAP.set(normalized, entry);
        }
        const MODEL_REGISTRY_DEFAULT_MODEL = typeof MODEL_REGISTRY_DATA.default_model === 'string'
            ? MODEL_REGISTRY_DATA.default_model
            : '';
        const AVAILABLE_AGENT_MODELS = MODEL_REGISTRY_MODELS
            .map((entry) => (entry && typeof entry.id === 'string') ? entry.id : null)
            .filter((id) => typeof id === 'string' && id.trim().length);
        const DEFAULT_AGENT_MODEL = (() => {
            const fallback = MODEL_REGISTRY_DEFAULT_MODEL && MODEL_REGISTRY_DEFAULT_MODEL.trim().length
                ? MODEL_REGISTRY_DEFAULT_MODEL
                : (AVAILABLE_AGENT_MODELS.length ? AVAILABLE_AGENT_MODELS[0] : 'openai/gpt-4o-mini');
            const element = document.getElementById('default-agent-model');
            if (!element) {
                return fallback;
            }
            const raw = element.textContent || '';
            try {
                const parsed = JSON.parse(raw);
                return typeof parsed === 'string' && parsed.trim().length ? parsed : fallback;
            } catch (err) {
                console.warn('Nelze načíst výchozí model agenta:', err);
                return raw.trim().length ? raw : fallback;
            }
        })();
        if (DEFAULT_AGENT_MODEL && !AVAILABLE_AGENT_MODELS.includes(DEFAULT_AGENT_MODEL)) {
            AVAILABLE_AGENT_MODELS.unshift(DEFAULT_AGENT_MODEL);
        }
        const DEFAULT_AGENT_PROMPT = (() => {
            const fallback = 'Jsi pečlivý korektor češtiny. Dostaneš JSON s klíči "language_hint" a "blocks". Blocks je pole objektů {id, type, text}. Oprav překlepy a zjevné OCR chyby pouze v hodnotách "text". Nesjednocuj styl, neměň typy bloků ani jejich pořadí. Zachovej diakritiku, pokud lze. Odpovídej pouze validním JSON se stejnou strukturou a klíči jako vstup.';
            const element = document.getElementById('default-agent-prompt-data');
            if (!element) {
                return fallback;
            }
            const raw = element.textContent || '';
            try {
                const parsed = JSON.parse(raw);
                return typeof parsed === 'string' && parsed.trim().length ? parsed : fallback;
            } catch (err) {
                console.warn('Nelze načíst výchozí prompt agenta:', err);
                return raw.trim().length ? raw : fallback;
            }
        })();
        const DEFAULT_LANGUAGE_HINT = 'cs';
        const MANUAL_JOINER_NAME = 'manual-joiner';
        const HYPHEN_LIKE_REGEX = /[\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]/;
        const TRAILING_HYPHEN_SEQUENCE_REGEX = /[\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]["'„“”«»‚‘’‹›)\]\}]*$/;
        const SENTENCE_END_REGEX = /[.!?…:]+["'„“”«»‚‘’‹›)\]\}]*$/u;
        const SENTENCE_START_REGEX = /^[\s"'„“”«»‚‘’‹›(\[{\-—–]*[A-Z\u00C0-\u0178]/u;
        function isReasoningModel(model) {
            const definition = getModelDefinition(model);
            if (definition && definition.capabilities) {
                return Boolean(definition.capabilities.reasoning);
            }
            const normalized = normalizeModelId(getUpstreamModelId(model));
            if (!normalized) return false;
            for (const prefix of FALLBACK_REASONING_PREFIXES) {
                const lowered = prefix.toLowerCase();
                if (normalized === lowered || normalized.startsWith(`${lowered}-`)) {
                    return true;
                }
            }
            return false;
        }

        function getModelDefinition(model) {
            const normalized = normalizeModelId(model);
            if (!normalized) {
                return null;
            }
            return MODEL_REGISTRY_MAP.get(normalized) || null;
        }

        function getUpstreamModelId(modelId) {
            const definition = getModelDefinition(modelId);
            if (definition && typeof definition.upstream_id === 'string' && definition.upstream_id.trim().length) {
                return definition.upstream_id;
            }
            return modelId;
        }

        function getModelUsageFlags(modelId) {
            const definition = getModelDefinition(modelId);
            const supportsText = definition && Object.prototype.hasOwnProperty.call(definition, 'supports_text')
                ? Boolean(definition.supports_text)
                : true;
            const supportsScan = definition && Object.prototype.hasOwnProperty.call(definition, 'supports_scan')
                ? Boolean(definition.supports_scan)
                : false;
            return { supportsText, supportsScan };
        }

        function getAllowedModelsForCollection(collection) {
            const normalized = String(collection || '').trim().toLowerCase();
            const filtered = AVAILABLE_AGENT_MODELS.filter((modelId) => {
                const usage = getModelUsageFlags(modelId);
                if (normalized === 'readers') {
                    return usage.supportsScan;
                }
                return usage.supportsText;
            });
            if (filtered.length) {
                return filtered;
            }
            return AVAILABLE_AGENT_MODELS.slice();
        }

        function getDefaultModelForCollection(collection) {
            const allowed = getAllowedModelsForCollection(collection);
            if (allowed.length) {
                return allowed[0];
            }
            return DEFAULT_AGENT_MODEL;
        }

        function getModelCapabilities(model) {
            const normalized = normalizeModelId(model);
            if (!normalized) {
                const defaults = { temperature: true, top_p: true, reasoning: false };
                if (ENABLE_RESPONSE_FORMAT) defaults.response_format = true;
                return defaults;
            }
            if (MODEL_CAPABILITIES_CACHE.has(normalized)) {
                return MODEL_CAPABILITIES_CACHE.get(normalized);
            }
            const definition = getModelDefinition(normalized);
            let capabilities;
            if (definition && definition.capabilities) {
                capabilities = {
                    temperature: Boolean(definition.capabilities.temperature),
                    top_p: Boolean(definition.capabilities.top_p),
                    reasoning: Boolean(definition.capabilities.reasoning),
                };
                if (ENABLE_RESPONSE_FORMAT) {
                    capabilities.response_format = Object.prototype.hasOwnProperty.call(definition.capabilities, 'response_format')
                        ? Boolean(definition.capabilities.response_format)
                        : true;
                }
            } else {
                capabilities = isReasoningModel(normalized)
                    ? { temperature: false, top_p: false, reasoning: true }
                    : { temperature: true, top_p: true, reasoning: false };
                if (ENABLE_RESPONSE_FORMAT) {
                    capabilities.response_format = true;
                }
            }
            MODEL_CAPABILITIES_CACHE.set(normalized, capabilities);
            return capabilities;
        }

        function clamp(value, min, max) {
            if (!Number.isFinite(value)) return min;
            return Math.min(max, Math.max(min, value));
        }

        function clampTemperature(value) {
            return clamp(value, 0, 2);
        }

        function clampTopP(value) {
            return clamp(value, 0, 1);
        }

        function createDefaultModelSettings() {
            const base = {
                temperature: DEFAULT_TEMPERATURE,
                top_p: DEFAULT_TOP_P,
                reasoning_effort: DEFAULT_REASONING_EFFORT,
            };
            if (ENABLE_RESPONSE_FORMAT) {
                base.response_format = null;
            }
            return base;
        }

        function buildManualJoinerAgent() {
            return {
                name: MANUAL_JOINER_NAME,
                display_name: 'manual-joiner',
                prompt: '',
                model: 'manual',
                manual: true,
                settings: {
                    defaults: createDefaultModelSettings(),
                    per_model: {},
                },
            };
        }

        function ensureManualJoinerAgent(ctx, targetMap) {
            if (!ctx || ctx.collection !== 'joiners') {
                return;
            }
            if (!ctx.agentsCache[MANUAL_JOINER_NAME]) {
                ctx.agentsCache[MANUAL_JOINER_NAME] = buildManualJoinerAgent();
                cacheAgentFingerprint(ctx, MANUAL_JOINER_NAME, ctx.agentsCache[MANUAL_JOINER_NAME]);
            }
            if (targetMap && !targetMap[MANUAL_JOINER_NAME]) {
                targetMap[MANUAL_JOINER_NAME] = {
                    name: MANUAL_JOINER_NAME,
                    display_name: 'manual-joiner',
                };
            }
        }

        function sanitizeSettingsObject(source) {
            const result = {};
            if (!source || typeof source !== 'object') {
                return result;
            }
            if (Object.prototype.hasOwnProperty.call(source, 'temperature')) {
                const value = Number(source.temperature);
                if (Number.isFinite(value)) {
                    result.temperature = clampTemperature(value);
                }
            }
            if (Object.prototype.hasOwnProperty.call(source, 'top_p')) {
                const value = Number(source.top_p);
                if (Number.isFinite(value)) {
                    result.top_p = clampTopP(value);
                }
            }
            if (Object.prototype.hasOwnProperty.call(source, 'reasoning_effort')) {
                const raw = String(source.reasoning_effort || '').trim().toLowerCase();
                if (['low', 'medium', 'high'].includes(raw)) {
                    result.reasoning_effort = raw;
                }
            }
            if (ENABLE_RESPONSE_FORMAT && Object.prototype.hasOwnProperty.call(source, 'response_format')) {
                const normalized = normalizeResponseFormat(source.response_format);
                if (normalized) {
                    result.response_format = normalized;
                }
            }
            return result;
        }

        function normalizeModelSettings(modelId, source, defaults) {
            const capabilities = getModelCapabilities(modelId);
            const sanitizedSource = sanitizeSettingsObject(source || {});
            const sanitizedDefaults = sanitizeSettingsObject(defaults || {});
            const registryDefaults = sanitizeSettingsObject((getModelDefinition(modelId) || {}).defaults);
            const baseDefaults = createDefaultModelSettings();
            const result = {};
            if (capabilities.temperature) {
                if (Object.prototype.hasOwnProperty.call(sanitizedSource, 'temperature')) {
                    result.temperature = clampTemperature(sanitizedSource.temperature);
                } else if (Object.prototype.hasOwnProperty.call(sanitizedDefaults, 'temperature')) {
                    result.temperature = clampTemperature(sanitizedDefaults.temperature);
                } else if (Object.prototype.hasOwnProperty.call(registryDefaults, 'temperature')) {
                    result.temperature = clampTemperature(registryDefaults.temperature);
                } else {
                    result.temperature = baseDefaults.temperature;
                }
            }
            if (capabilities.top_p) {
                if (Object.prototype.hasOwnProperty.call(sanitizedSource, 'top_p')) {
                    result.top_p = clampTopP(sanitizedSource.top_p);
                } else if (Object.prototype.hasOwnProperty.call(sanitizedDefaults, 'top_p')) {
                    result.top_p = clampTopP(sanitizedDefaults.top_p);
                } else if (Object.prototype.hasOwnProperty.call(registryDefaults, 'top_p')) {
                    result.top_p = clampTopP(registryDefaults.top_p);
                } else {
                    result.top_p = baseDefaults.top_p;
                }
           }
           if (capabilities.reasoning) {
                const sourceValue = sanitizedSource.reasoning_effort
                    || sanitizedDefaults.reasoning_effort
                    || registryDefaults.reasoning_effort
                    || baseDefaults.reasoning_effort;
                result.reasoning_effort = ['low', 'medium', 'high'].includes(sourceValue) ? sourceValue : baseDefaults.reasoning_effort;
            }
            if (ENABLE_RESPONSE_FORMAT && capabilities.response_format) {
                const candidates = [
                    sanitizedSource.response_format,
                    sanitizedDefaults.response_format,
                    registryDefaults.response_format,
                ];
                for (const candidate of candidates) {
                    const normalized = normalizeResponseFormat(candidate);
                    if (normalized) {
                        result.response_format = normalized;
                        break;
                    }
                }
            }
            return result;
        }

        function deepCloneAgent(agent) {
            try {
                return JSON.parse(JSON.stringify(agent || {}));
            } catch (err) {
                return agent ? { ...agent } : {};
            }
        }

        function normalizeAgentData(raw) {
            const base = deepCloneAgent(raw || {});
            const name = typeof base.name === 'string' ? base.name.trim() : '';
            const displayName = typeof base.display_name === 'string' && base.display_name.trim().length
                ? base.display_name
                : name;
            const manualFlag = Boolean(base.manual) || name === MANUAL_JOINER_NAME;
            const promptValue = typeof base.prompt === 'string' && base.prompt.trim().length
                ? base.prompt
                : DEFAULT_AGENT_PROMPT;
            const rawModel = typeof base.model === 'string' && base.model.trim().length
                ? base.model.trim()
                : '';
            const modelCandidate = (manualFlag && rawModel === '') ? 'manual'
                : (rawModel || DEFAULT_AGENT_MODEL);
            const defaults = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(base.settings && base.settings.defaults),
            };
            const perModel = {};
            if (base.settings && typeof base.settings === 'object' && base.settings.per_model && typeof base.settings.per_model === 'object') {
                for (const [modelIdRaw, settings] of Object.entries(base.settings.per_model)) {
                    const modelId = String(modelIdRaw || '').trim();
                    if (!modelId) continue;
                    perModel[modelId] = normalizeModelSettings(modelId, settings, defaults);
                }
            }
            const legacySettings = sanitizeSettingsObject(base);
            if (Object.keys(legacySettings).length) {
                const current = perModel[modelCandidate] || {};
                perModel[modelCandidate] = normalizeModelSettings(modelCandidate, { ...current, ...legacySettings }, defaults);
            }
            if (!perModel[modelCandidate]) {
                perModel[modelCandidate] = normalizeModelSettings(modelCandidate, {}, defaults);
            }
            return {
                name,
                display_name: displayName || name,
                prompt: promptValue,
                model: modelCandidate,
                settings: {
                    defaults,
                    per_model: perModel,
                },
                manual: manualFlag,
                updated_at: typeof base.updated_at === 'number' ? base.updated_at : undefined,
            };
        }

        function ensureAgentDraftStructure(draft) {
            if (!draft || typeof draft !== 'object') {
                return null;
            }
            if (!draft.settings || typeof draft.settings !== 'object') {
                draft.settings = { defaults: createDefaultModelSettings(), per_model: {} };
            }
            if (!draft.settings.defaults || typeof draft.settings.defaults !== 'object') {
                draft.settings.defaults = createDefaultModelSettings();
            }
            if (!draft.settings.per_model || typeof draft.settings.per_model !== 'object') {
                draft.settings.per_model = {};
            }
            return draft;
        }

        function ensureAgentDraftModelSettings(draft, modelId) {
            ensureAgentDraftStructure(draft);
            const modelKey = modelId || (draft && draft.model) || DEFAULT_AGENT_MODEL;
            if (!draft.settings.per_model[modelKey]) {
                draft.settings.per_model[modelKey] = normalizeModelSettings(modelKey, {}, draft.settings.defaults);
            }
            return draft.settings.per_model[modelKey];
        }

        function getEffectiveModelSettings(agent, modelId) {
            if (!agent) {
                return normalizeModelSettings(modelId, {}, createDefaultModelSettings());
            }
            const draft = ensureAgentDraftStructure(deepCloneAgent(agent));
            const defaults = draft.settings.defaults || createDefaultModelSettings();
            const perModel = draft.settings.per_model && draft.settings.per_model[modelId]
                ? draft.settings.per_model[modelId]
                : {};
            return normalizeModelSettings(modelId, perModel, defaults);
        }

        function markAgentDirty(ctx) {
            if (ctx && !(ctx.collection === 'joiners' && ctx.currentAgentDraft && ctx.currentAgentDraft.manual)) {
                ctx.agentDraftDirty = true;
            }
        }

        function resetAgentDirty(ctx) {
            if (ctx) {
                ctx.agentDraftDirty = false;
            }
        }

        function buildAgentSavePayload(ctx, draftInput) {
            const normalized = normalizeAgentData(draftInput || ctx.currentAgentDraft || {});
            ensureAgentDraftStructure(normalized);
            const modelId = normalized.model || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(normalized, modelId);
            const defaultsPayload = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(normalized.settings.defaults),
            };
            const perModelPayload = {};
            for (const [modelKey, settings] of Object.entries(normalized.settings.per_model || {})) {
                perModelPayload[modelKey] = normalizeModelSettings(modelKey, settings, defaultsPayload);
            }
            const effective = getEffectiveModelSettings(normalized, modelId);
            const capabilities = getModelCapabilities(modelId);
            const payload = {
                name: normalized.name,
                display_name: normalized.display_name || normalized.name,
                prompt: normalized.prompt || '',
                model: modelId,
                settings: {
                    defaults: defaultsPayload,
                    per_model: perModelPayload,
                },
            };
            if (capabilities.temperature && Object.prototype.hasOwnProperty.call(effective, 'temperature')) {
                payload.temperature = effective.temperature;
            }
            if (capabilities.top_p && Object.prototype.hasOwnProperty.call(effective, 'top_p')) {
                payload.top_p = effective.top_p;
            }
            if (capabilities.reasoning && Object.prototype.hasOwnProperty.call(effective, 'reasoning_effort')) {
                payload.reasoning_effort = effective.reasoning_effort;
            }
            if (ENABLE_RESPONSE_FORMAT && capabilities.response_format && Object.prototype.hasOwnProperty.call(effective, 'response_format')) {
                try {
                    payload.response_format = JSON.parse(JSON.stringify(effective.response_format));
                } catch (err) {
                    payload.response_format = effective.response_format;
                }
            }
            return payload;
        }

        function renderAgentModelOptions(ctx, selectedModel) {
            const selectEl = getContextElement(ctx, 'modelSelectId');
            if (!selectEl) {
                return;
            }
            if (ctx.collection === 'joiners' && ctx.currentAgentDraft && ctx.currentAgentDraft.manual) {
                selectEl.innerHTML = '<option value="">(bez modelu)</option>';
                selectEl.value = '';
                selectEl.disabled = true;
                return;
            }
            const modelOptions = getAllowedModelsForCollection(ctx.collection);
            if (ctx.collection !== 'readers' && ctx.currentAgentDraft && ctx.currentAgentDraft.settings && ctx.currentAgentDraft.settings.per_model) {
                Object.keys(ctx.currentAgentDraft.settings.per_model).forEach((modelId) => {
                    if (modelId && !modelOptions.includes(modelId)) {
                        modelOptions.push(modelId);
                    }
                });
            }
            if (ctx.collection !== 'readers' && selectedModel && !modelOptions.includes(selectedModel)) {
                modelOptions.push(selectedModel);
            }
            if (!modelOptions.length) {
                selectEl.innerHTML = '<option value="">Žádné dostupné modely</option>';
                selectEl.value = '';
                selectEl.disabled = true;
                return;
            }
            selectEl.disabled = false;
            selectEl.innerHTML = '';
            modelOptions.forEach((modelId) => {
                const option = document.createElement('option');
                option.value = modelId;
                const definition = getModelDefinition(modelId);
                option.textContent = definition && typeof definition.display_name === 'string'
                    ? definition.display_name
                    : modelId;
                selectEl.appendChild(option);
            });
            const normalized = (selectedModel && modelOptions.includes(selectedModel))
                ? selectedModel
                : modelOptions[0];
            selectEl.value = normalized;
            if (ctx.currentAgentDraft) {
                ctx.currentAgentDraft.model = normalized;
            }
        }

        function renderAgentParameterControls(ctx, modelId) {
            const container = getContextElement(ctx, 'parameterFieldsId');
            if (!container) {
                return;
            }
            if (!ctx.currentAgentDraft) {
                container.innerHTML = '<div class="muted">Nejprve vyberte agenta.</div>';
                return;
            }
            if (ctx.collection === 'joiners' && ctx.currentAgentDraft.manual) {
                container.innerHTML = '<div class="muted">Tento agent nepoužívá model.</div>';
                return;
            }
            const capabilities = getModelCapabilities(modelId);
            const modelSettings = ensureAgentDraftModelSettings(ctx.currentAgentDraft, modelId);
            const controls = [];
            let idPrefix = 'agent';
            if (ctx.collection === 'joiners') {
                idPrefix = 'stitch';
            } else if (ctx.collection === 'readers') {
                idPrefix = 'reader';
            }
            const supportsResponseFormat = ENABLE_RESPONSE_FORMAT && Boolean(capabilities.response_format);
            if (capabilities.temperature) {
                controls.push(`
                    <div class="agent-param" data-param="temperature" style="flex:1;min-width:220px;">
                        <label for="${idPrefix}-param-temperature" style="display:block;margin-bottom:4px;font-weight:600;">Temperature</label>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <input id="${idPrefix}-param-temperature" type="range" min="0" max="2" step="0.05" style="flex:1;">
                            <input id="${idPrefix}-param-temperature-number" type="number" min="0" max="2" step="0.05" style="width:80px;">
                        </div>
                    </div>
                `);
            }
            if (capabilities.top_p) {
                controls.push(`
                    <div class="agent-param" data-param="top_p" style="flex:1;min-width:220px;">
                        <label for="${idPrefix}-param-top-p" style="display:block;margin-bottom:4px;font-weight:600;">Top P</label>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <input id="${idPrefix}-param-top-p" type="range" min="0" max="1" step="0.01" style="flex:1;">
                            <input id="${idPrefix}-param-top-p-number" type="number" min="0" max="1" step="0.01" style="width:80px;">
                        </div>
                    </div>
                `);
            }
            if (capabilities.reasoning) {
                controls.push(`
                    <div class="agent-param" data-param="reasoning_effort" style="flex:1;min-width:220px;">
                        <label for="${idPrefix}-param-reasoning" style="display:block;margin-bottom:4px;font-weight:600;">Reasoning</label>
                        <select id="${idPrefix}-param-reasoning" style="min-width:140px;">
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                        </select>
                    </div>
                `);
            }
            if (supportsResponseFormat) {
                controls.push(`
                    <div class="agent-param" data-param="response_format" style="flex:1;min-width:260px;">
                        <label for="${idPrefix}-param-response-format" style="display:block;margin-bottom:4px;font-weight:600;">Response Format</label>
                        <select id="${idPrefix}-param-response-format" style="min-width:160px;">
                            <option value="none">(bez formátu)</option>
                            <option value="json_object">JSON object</option>
                            <option value="json_schema">JSON schema</option>
                        </select>
                        <textarea id="${idPrefix}-param-response-format-schema" rows="6" style="display:none;margin-top:8px;width:100%;font-family:var(--font-monospace, monospace);font-size:12px;"></textarea>
                        <div id="${idPrefix}-param-response-format-hint" class="muted" style="display:none;margin-top:4px;font-size:12px;">
                            Očekává objekt {"name": "...", "schema": {...}} dle Responses API (OpenAI/OpenRouter).
                        </div>
                    </div>
                `);
            }
            if (!controls.length) {
                container.innerHTML = '<div class="muted">Tento model nemá nastavitelné parametry.</div>';
                return;
            }
            container.innerHTML = `<div style="display:flex;gap:12px;flex-wrap:wrap;">${controls.join('')}</div>`;
            const tempRange = container.querySelector(`#${idPrefix}-param-temperature`);
            const tempNumber = container.querySelector(`#${idPrefix}-param-temperature-number`);
            const topRange = container.querySelector(`#${idPrefix}-param-top-p`);
            const topNumber = container.querySelector(`#${idPrefix}-param-top-p-number`);
            const reasoningSelect = container.querySelector(`#${idPrefix}-param-reasoning`);
            const responseFormatSelect = supportsResponseFormat ? container.querySelector(`#${idPrefix}-param-response-format`) : null;
            const responseFormatTextarea = supportsResponseFormat ? container.querySelector(`#${idPrefix}-param-response-format-schema`) : null;
            const responseFormatHint = supportsResponseFormat ? container.querySelector(`#${idPrefix}-param-response-format-hint`) : null;
            container.querySelectorAll('select').forEach(enableScrollPassthrough);
            if (responseFormatTextarea) {
                enableScrollPassthrough(responseFormatTextarea);
                responseFormatTextarea.addEventListener('input', () => autoResizeTextarea(responseFormatTextarea));
            }

            if (tempRange && tempNumber) {
                const value = Object.prototype.hasOwnProperty.call(modelSettings, 'temperature')
                    ? modelSettings.temperature
                    : DEFAULT_TEMPERATURE;
                tempRange.value = value;
                tempNumber.value = value;
                const updateTemperature = (nextValue) => {
                    const parsed = clampTemperature(Number(nextValue));
                    tempRange.value = parsed;
                    tempNumber.value = parsed;
                    modelSettings.temperature = parsed;
                    markAgentDirty(ctx);
                };
                tempRange.addEventListener('input', () => updateTemperature(tempRange.value));
                tempNumber.addEventListener('change', () => updateTemperature(tempNumber.value));
            }

            if (topRange && topNumber) {
                const value = Object.prototype.hasOwnProperty.call(modelSettings, 'top_p')
                    ? modelSettings.top_p
                    : DEFAULT_TOP_P;
                topRange.value = value;
                topNumber.value = value;
                const updateTopP = (nextValue) => {
                    const parsed = clampTopP(Number(nextValue));
                    topRange.value = parsed;
                    topNumber.value = parsed;
                    modelSettings.top_p = parsed;
                    markAgentDirty(ctx);
                };
                topRange.addEventListener('input', () => updateTopP(topRange.value));
                topNumber.addEventListener('change', () => updateTopP(topNumber.value));
            }

            if (reasoningSelect) {
                const raw = String(modelSettings.reasoning_effort || '').trim().toLowerCase();
                const normalized = ['low', 'medium', 'high'].includes(raw) ? raw : DEFAULT_REASONING_EFFORT;
                reasoningSelect.value = normalized;
                reasoningSelect.addEventListener('change', () => {
                    const candidate = String(reasoningSelect.value || '').trim().toLowerCase();
                    const next = ['low', 'medium', 'high'].includes(candidate) ? candidate : DEFAULT_REASONING_EFFORT;
                    reasoningSelect.value = next;
                    modelSettings.reasoning_effort = next;
                    markAgentDirty(ctx);
                });
            }
            if (responseFormatSelect) {
                let existingResponseFormat = normalizeResponseFormat(modelSettings.response_format);
                if (existingResponseFormat) {
                    modelSettings.response_format = existingResponseFormat;
                } else {
                    delete modelSettings.response_format;
                    existingResponseFormat = null;
                }
                let currentResponseFormatSignature = JSON.stringify(existingResponseFormat || null);
                const defaultSchemaTemplate = JSON.stringify({
                    name: 'response',
                    schema: {
                        type: 'object',
                        properties: {},
                        additionalProperties: true,
                    },
                }, null, 2);
                const initialType = existingResponseFormat
                    ? (existingResponseFormat.type === 'json_schema' ? 'json_schema' : 'json_object')
                    : 'none';
                const initialSchemaText = existingResponseFormat && existingResponseFormat.type === 'json_schema'
                    ? JSON.stringify(existingResponseFormat.json_schema, null, 2)
                    : '';
                responseFormatSelect.value = initialType;
                if (responseFormatTextarea) {
                    responseFormatTextarea.value = initialSchemaText;
                    if (initialType === 'json_schema') {
                        responseFormatTextarea.style.display = 'block';
                        if (responseFormatHint) responseFormatHint.style.display = 'block';
                        requestAnimationFrame(() => autoResizeTextarea(responseFormatTextarea));
                    } else {
                        responseFormatTextarea.style.display = 'none';
                        if (responseFormatHint) responseFormatHint.style.display = 'none';
                    }
                }

                const setResponseFormat = (format) => {
                    const signature = JSON.stringify(format || null);
                    let storedFormat = null;
                    if (!format) {
                        delete modelSettings.response_format;
                    } else {
                        try {
                            storedFormat = JSON.parse(signature);
                            modelSettings.response_format = storedFormat;
                        } catch (err) {
                            storedFormat = JSON.parse(JSON.stringify(format));
                            modelSettings.response_format = storedFormat;
                        }
                    }
                    if (!format) {
                        existingResponseFormat = null;
                    } else {
                        existingResponseFormat = storedFormat || JSON.parse(JSON.stringify(format));
                    }
                    if (signature !== currentResponseFormatSignature) {
                        currentResponseFormatSignature = signature;
                        markAgentDirty(ctx);
                    }
                };

                const setSchemaInvalid = (message) => {
                    if (!responseFormatTextarea) return;
                    responseFormatTextarea.dataset.invalid = 'true';
                    responseFormatTextarea.style.borderColor = '#c00';
                    responseFormatTextarea.title = message || 'Neplatné JSON schema';
                };

                const clearSchemaInvalid = () => {
                    if (!responseFormatTextarea) return;
                    delete responseFormatTextarea.dataset.invalid;
                    responseFormatTextarea.style.borderColor = '';
                    responseFormatTextarea.title = 'JSON schema {"name": "...", "schema": {...}}';
                };

                const handleSchemaUpdate = () => {
                    if (!responseFormatTextarea || (responseFormatSelect && responseFormatSelect.value !== 'json_schema')) {
                        return;
                    }
                    const rawValue = responseFormatTextarea.value.trim();
                    if (!rawValue.length) {
                        setSchemaInvalid('Zadejte JSON se strukturou {"name": "...", "schema": {...}}.');
                        setResponseFormat(null);
                        return;
                    }
                    let parsed;
                    try {
                        parsed = JSON.parse(rawValue);
                    } catch (err) {
                        setSchemaInvalid('Neplatný JSON – opravte schéma.');
                        setResponseFormat(null);
                        return;
                    }
                    const normalized = normalizeResponseFormat({ type: 'json_schema', json_schema: parsed });
                    if (!normalized) {
                        setSchemaInvalid('Schema musí obsahovat klíče "name" (string) a "schema" (object).');
                        setResponseFormat(null);
                        return;
                    }
                    clearSchemaInvalid();
                    setResponseFormat(normalized);
                };

                const applySelectValue = (initializing = false) => {
                    if (!responseFormatSelect) return;
                    const selected = responseFormatSelect.value;
                    if (selected === 'json_schema') {
                        if (responseFormatTextarea) {
                            responseFormatTextarea.style.display = 'block';
                            if (responseFormatHint) responseFormatHint.style.display = 'block';
                            if (!responseFormatTextarea.value.trim().length) {
                                if (existingResponseFormat && existingResponseFormat.type === 'json_schema') {
                                    responseFormatTextarea.value = JSON.stringify(existingResponseFormat.json_schema, null, 2);
                                } else {
                                    responseFormatTextarea.value = defaultSchemaTemplate;
                                }
                            }
                            requestAnimationFrame(() => autoResizeTextarea(responseFormatTextarea));
                        }
                        if (!initializing) {
                            handleSchemaUpdate();
                        } else {
                            clearSchemaInvalid();
                        }
                    } else {
                        if (responseFormatTextarea) {
                            responseFormatTextarea.style.display = 'none';
                            if (responseFormatHint) responseFormatHint.style.display = 'none';
                        }
                        clearSchemaInvalid();
                        if (!initializing) {
                            if (selected === 'json_object') {
                                setResponseFormat({ type: 'json_object' });
                            } else {
                                setResponseFormat(null);
                            }
                        }
                    }
                };

                applySelectValue(true);
                responseFormatSelect.addEventListener('change', () => applySelectValue(false));

                if (responseFormatTextarea) {
                    responseFormatTextarea.addEventListener('blur', handleSchemaUpdate);
                    responseFormatTextarea.addEventListener('change', handleSchemaUpdate);
                }
            } else if (responseFormatTextarea) {
                responseFormatTextarea.style.display = 'none';
                if (responseFormatHint) responseFormatHint.style.display = 'none';
            }
        }

        async function loadAgents(ctx, options = {}) {
            const force = options && typeof options === 'object' && options.force === true;
            const cache = ctx.agentsCache;
            ensureManualJoinerAgent(ctx);
            const keys = Object.keys(cache);
            const hasNonManual = keys.some((k) => k !== MANUAL_JOINER_NAME);
            if (keys.length && !force && (ctx.collection !== 'joiners' || hasNonManual)) {
                const out = {};
                keys.sort().forEach(k => { out[k] = { name: k, display_name: cache[k].display_name || k }; });
                ensureManualJoinerAgent(ctx, out);
                return out;
            }
            try {
                const url = buildAgentRequestUrl('/agents/list', ctx);
                const res = await fetch(url, { cache: 'no-store' });
                if (!res.ok) return {};
                const data = await res.json();
                const out = {};
                // server returns list items which may include parsed agent object
                const fetchedAgents = [];
                (data.agents || []).forEach(a => {
                    const baseAgent = a && a.agent && typeof a.agent === 'object'
                        ? { ...deepCloneAgent(a.agent), name: a.name, display_name: a.display_name || a.name }
                        : { name: a.name, display_name: a.display_name || a.name, prompt: DEFAULT_AGENT_PROMPT, model: DEFAULT_AGENT_MODEL };
                    const normalized = normalizeAgentData(baseAgent);
                    cache[a.name] = normalized;
                    cacheAgentFingerprint(ctx, a.name, normalized);
                    out[a.name] = { name: a.name, display_name: normalized.display_name || a.name };
                    fetchedAgents.push(a.name);
                });
                console.info('[AgentDebug] loadAgents ->', fetchedAgents);
                if (force) {
                    Object.keys(cache).forEach((name) => {
                        if (!fetchedAgents.includes(name)) {
                            delete cache[name];
                            ctx.agentFingerprintCache.delete(name);
                        }
                    });
                }
                ensureManualJoinerAgent(ctx, out);
                return out;
            } catch (err) {
                console.warn('Nelze načíst agenty ze serveru, fallback na empty:', err);
                const fallback = {};
                ensureManualJoinerAgent(ctx, fallback);
                return fallback;
            }
        }

        // saveAgents is no longer used client-side; server persists agents

        function cacheAgentFingerprint(ctx, name, agent) {
            if (!name) {
                return null;
            }
            if (agent && typeof agent === 'object') {
                const payload = extractAgentFingerprintPayload(agent);
                const serialized = JSON.stringify(payload);
                const hash = computeSimpleHash(serialized);
                const record = { hash, serialized };
                ctx.agentFingerprintCache.set(name, record);
                return record;
            }
            return ctx.agentFingerprintCache.get(name) || null;
        }

        function extractAgentFingerprintPayload(agent) {
            if (!agent || typeof agent !== 'object') {
                return null;
            }
            const normalized = normalizeAgentData(agent);
            ensureAgentDraftStructure(normalized);
            const modelId = normalized.model || DEFAULT_AGENT_MODEL;
            const defaults = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(normalized.settings.defaults),
            };
            const perModelEntries = Object.entries(normalized.settings.per_model || {})
                .map(([modelKey, settings]) => [modelKey, normalizeModelSettings(modelKey, settings, defaults)])
                .sort((a, b) => a[0].localeCompare(b[0]));
            const effective = getEffectiveModelSettings(normalized, modelId);
            return {
                prompt: String(normalized.prompt || ''),
                model: modelId,
                defaults,
                per_model: perModelEntries,
                effective,
            };
        }

        function computeAgentFingerprint(ctx, name) {
            if (!name) {
                return '';
            }
            const agent = (ctx.currentAgentName === name && ctx.currentAgentDraft)
                ? ctx.currentAgentDraft
                : ctx.agentsCache[name];
            const record = cacheAgentFingerprint(ctx, name, agent);
            if (record && typeof record.hash === 'number') {
                return `${name}:${record.hash}`;
            }
            const cached = ctx.agentFingerprintCache.get(name);
            if (cached && typeof cached.hash === 'number') {
                return `${name}:${cached.hash}`;
            }
            return name;
        }

        async function renderAgentSelector(ctx, force = false) {
            const select = getContextElement(ctx, 'selectId');
            const spinnerWrapper = getContextElement(ctx, 'selectSpinnerId');
            if (spinnerWrapper) spinnerWrapper.style.display = 'inline-block';
            if (!select) {
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return;
            }
            select.innerHTML = '';
            const agents = await loadAgents(ctx, { force });
            ensureManualJoinerAgent(ctx, agents);
            const selected = loadContextSelectedAgent(ctx);
            const names = Object.keys(agents).sort();
            // If no agents on server, create some defaults client-side and save
            if (ctx.collection === 'correctors' && !names.length) {
                const defaults = ['editor-default','cleanup-a','semantic-fix'];
                for (const n of defaults) {
                    const draftAgent = {
                        name: n,
                        display_name: n,
                        prompt: DEFAULT_AGENT_PROMPT,
                        model: DEFAULT_AGENT_MODEL,
                        settings: {
                            defaults: createDefaultModelSettings(),
                            per_model: {
                                [DEFAULT_AGENT_MODEL]: {
                                    temperature: DEFAULT_TEMPERATURE,
                                    top_p: DEFAULT_TOP_P,
                                },
                            },
                        },
                    };
                    const payload = buildAgentSavePayload(ctx, draftAgent);
                    await fetch('/agents/save', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify(buildAgentPayload(ctx, payload)),
                    });
                }
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return renderAgentSelector(ctx, force);
            }
            if (!names.length) {
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return;
            }

            // show nicer labels when display_name exists
            for (const name of names) {
                const option = document.createElement('option');
                option.value = name;
                // prefer display_name from cache
                option.textContent = (agents[name] && agents[name].display_name) ? agents[name].display_name : name;
                select.appendChild(option);
            }

            if (selected && names.includes(selected)) {
                select.value = selected;
            } else {
                select.selectedIndex = 0;
                persistContextSelectedAgent(ctx, select.value);
            }
            if (spinnerWrapper) spinnerWrapper.style.display = 'none';
            // update the UI fields for the selected agent from cache (avoid extra fetch)
            try {
                const sel = select.value;
                if (sel && ctx.agentsCache[sel]) {
                    setAgentFields(ctx, ctx.agentsCache[sel]);
                    persistContextSelectedAgent(ctx, sel);
                } else {
                    await updateAgentUIFromSelection(ctx);
                }
            } catch (e) {
                await updateAgentUIFromSelection(ctx);
            }
        }

        async function updateAgentUIFromSelection(ctx) {
            const select = getContextElement(ctx, 'selectId');
            if (!select) return;
            const name = select.value;
            const auto = getContextElement(ctx, 'autoCheckboxId');
            if (auto) auto.checked = loadContextAutoTrigger(ctx);
            const fallbackModel = getDefaultModelForCollection(ctx.collection);
            let fallbackAgent = normalizeAgentData({
                name,
                display_name: name,
                prompt: DEFAULT_AGENT_PROMPT,
                model: fallbackModel,
            });
            if (ctx.collection === 'joiners' && name === MANUAL_JOINER_NAME) {
                ensureManualJoinerAgent(ctx);
                fallbackAgent = buildManualJoinerAgent();
            }
            try {
                // prefer in-memory cache where possible
                if (ctx.agentsCache[name]) {
                    setAgentFields(ctx, deepCloneAgent(ctx.agentsCache[name]));
                    persistContextSelectedAgent(ctx, name);
                    return;
                }
                if (ctx.collection === 'joiners' && name === MANUAL_JOINER_NAME) {
                    setAgentFields(ctx, buildManualJoinerAgent());
                    persistContextSelectedAgent(ctx, name);
                    return;
                }
                const url = buildAgentRequestUrl('/agents/get', ctx, { name });
                const res = await fetch(url, { cache: 'no-store' });
                if (!res.ok) {
                    setAgentFields(ctx, fallbackAgent);
                    if (!(ctx.collection === 'joiners' && name === MANUAL_JOINER_NAME)) {
                        persistContextSelectedAgent(ctx, '');
                    }
                    return;
                }
                const agent = await res.json();
                const normalized = normalizeAgentData({ ...(agent || {}), name });
                ctx.agentsCache[name] = normalized;
                setAgentFields(ctx, normalized);
                persistContextSelectedAgent(ctx, name);
            } catch (err) {
                console.warn('Nelze načíst agenta:', err);
                setAgentFields(ctx, fallbackAgent);
                if (!(ctx.collection === 'joiners' && name === MANUAL_JOINER_NAME)) {
                    persistContextSelectedAgent(ctx, '');
                }
            }
        }

        function autoResizeTextarea(textarea) {
            if (!textarea) return;
            if (!textarea.dataset.minHeight) {
                const computed = window.getComputedStyle(textarea);
                const lineHeight = parseFloat(computed.lineHeight) || 0;
                const rows = parseInt(textarea.getAttribute('rows') || '0', 10);
                const paddingTop = parseFloat(computed.paddingTop) || 0;
                const paddingBottom = parseFloat(computed.paddingBottom) || 0;
                const borderTop = parseFloat(computed.borderTopWidth) || 0;
                const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
                const estimated = rows && lineHeight ? (rows * lineHeight) + paddingTop + paddingBottom + borderTop + borderBottom : textarea.clientHeight;
                textarea.dataset.minHeight = String(estimated || 0);
            }
            const minHeight = parseFloat(textarea.dataset.minHeight || '0') || 0;
            textarea.style.height = 'auto';
            const newHeight = textarea.scrollHeight;
            if (newHeight && newHeight > 0) {
                textarea.style.height = `${Math.max(newHeight, minHeight)}px`;
            } else if (minHeight) {
                textarea.style.height = `${minHeight}px`;
            }
            delete textarea.dataset.needsResize;
            scheduleThumbnailDrawerHeightSync();
        }

        function enableScrollPassthrough(element) {
            if (!element) {
                return;
            }
            if (element.dataset.scrollPassthrough === 'true' || element.dataset.preventWheelScroll === 'true') {
                return;
            }
            const wheelHandler = (event) => {
                if (!event) {
                    return;
                }
                if (event.ctrlKey || event.metaKey || event.altKey) {
                    return;
                }
                if (Math.abs(event.deltaY) < 0.5 && Math.abs(event.deltaX) < 0.5) {
                    return;
                }
                event.preventDefault();
                const target = document.scrollingElement;
                if (target && typeof target.scrollBy === 'function') {
                    target.scrollBy({
                        top: event.deltaY,
                        left: event.deltaX,
                        behavior: 'auto',
                    });
                } else {
                    window.scrollTo(
                        window.pageXOffset + (event.deltaX || 0),
                        window.pageYOffset + (event.deltaY || 0)
                    );
                }
            };
            element.addEventListener('wheel', wheelHandler, { passive: false });
            element.dataset.scrollPassthrough = 'true';
            element.dataset.preventWheelScroll = 'true';
        }

        function preparePromptTextarea(textarea) {
            if (!textarea) return;
            textarea.style.overflow = 'hidden';
            textarea.style.resize = 'none';
            enableScrollPassthrough(textarea);
        }

        function ensureAgentModelFitsContext(ctx) {
            if (!ctx || !ctx.currentAgentDraft) {
                return;
            }
            if (ctx.collection === 'joiners' && ctx.currentAgentDraft.manual) {
                return;
            }
            const allowed = getAllowedModelsForCollection(ctx.collection);
            if (!allowed.length) {
                ctx.currentAgentDraft.model = '';
                return;
            }
            if (!ctx.currentAgentDraft.model || !allowed.includes(ctx.currentAgentDraft.model)) {
                ctx.currentAgentDraft.model = allowed[0];
            }
        }

        function setAgentFields(ctx, agent) {
            const normalized = normalizeAgentData(agent || {});
            ctx.currentAgentDraft = deepCloneAgent(normalized);
            ctx.currentAgentName = normalized.name || '';
            ensureAgentDraftStructure(ctx.currentAgentDraft);
            ensureAgentModelFitsContext(ctx);
            const nameEl = getContextElement(ctx, 'nameInputId');
            const promptEl = getContextElement(ctx, 'promptTextareaId');
            const modelEl = getContextElement(ctx, 'modelSelectId');
            if (nameEl) {
                nameEl.value = ctx.currentAgentDraft.name || '';
            }
            if (promptEl) {
                promptEl.value = ctx.currentAgentDraft.prompt || DEFAULT_AGENT_PROMPT;
                preparePromptTextarea(promptEl);
                if (!promptEl.dataset.autoresizeBound) {
                    promptEl.dataset.autoresizeBound = 'true';
                    promptEl.addEventListener('input', () => autoResizeTextarea(promptEl));
                }
                if (promptEl.offsetParent !== null) {
                    requestAnimationFrame(() => autoResizeTextarea(promptEl));
                } else {
                    promptEl.dataset.needsResize = 'true';
                }
            }
            if (modelEl) {
                renderAgentModelOptions(ctx, ctx.currentAgentDraft.model);
            }
            renderAgentParameterControls(ctx, ctx.currentAgentDraft.model || DEFAULT_AGENT_MODEL);
            applyManualAgentUiState(ctx);
            resetAgentDirty(ctx);
        }

        function applyManualAgentUiState(ctx) {
            const isManual = Boolean(ctx && ctx.collection === 'joiners' && ctx.currentAgentDraft && ctx.currentAgentDraft.manual);
            const toggle = getContextElement(ctx, 'expandToggleId');
            const settings = getContextElement(ctx, 'settingsId');
            if (toggle) {
                if (isManual) {
                    if (settings) {
                        settings.style.display = 'none';
                    }
                    toggle.setAttribute('aria-expanded', 'false');
                    toggle.disabled = true;
                    if (!toggle.dataset.originalTitle) {
                        toggle.dataset.originalTitle = toggle.getAttribute('title') || '';
                    }
                    toggle.setAttribute('title', 'Přepni agenta');
                    toggle.style.opacity = '0.4';
                    toggle.style.cursor = 'not-allowed';
                    toggle.style.pointerEvents = 'none';
                } else {
                    toggle.disabled = false;
                    const originalTitle = toggle.dataset.originalTitle || 'Zobrazit nastavení agenta';
                    toggle.setAttribute('title', originalTitle);
                    toggle.style.opacity = '';
                    toggle.style.cursor = '';
                    toggle.style.pointerEvents = '';
                }
            }
            const promptEl = getContextElement(ctx, 'promptTextareaId');
            if (promptEl) {
                if (isManual) {
                    promptEl.value = 'Manual joiner nepoužívá prompt.';
                }
                promptEl.readOnly = isManual;
                promptEl.disabled = isManual;
                preparePromptTextarea(promptEl);
            }
            const modelEl = getContextElement(ctx, 'modelSelectId');
            if (modelEl) {
                modelEl.disabled = isManual;
            }
            const nameEl = getContextElement(ctx, 'nameInputId');
            if (nameEl) {
                nameEl.readOnly = isManual;
                nameEl.disabled = isManual;
            }
            const saveBtn = getContextElement(ctx, 'saveButtonId');
            if (saveBtn) {
                saveBtn.disabled = isManual;
                saveBtn.style.display = isManual ? 'none' : '';
            }
            const deleteBtn = getContextElement(ctx, 'deleteButtonId');
            if (deleteBtn) {
                deleteBtn.disabled = isManual;
                deleteBtn.style.display = isManual ? 'none' : '';
            }
        }

        async function saveCurrentAgent(ctx) {
            const nameEl = getContextElement(ctx, 'nameInputId');
            const promptEl = getContextElement(ctx, 'promptTextareaId');
            const modelEl = getContextElement(ctx, 'modelSelectId');
            if (!nameEl) return;
            if (ctx.collection === 'joiners' && ctx.currentAgentDraft && ctx.currentAgentDraft.manual) {
                return;
            }
            if (!ctx.currentAgentDraft) {
                ctx.currentAgentDraft = normalizeAgentData({
                    name: '',
                    prompt: promptEl ? promptEl.value : DEFAULT_AGENT_PROMPT,
                    model: modelEl ? modelEl.value : DEFAULT_AGENT_MODEL,
                });
            }
            const draft = ensureAgentDraftStructure(ctx.currentAgentDraft);
            const previousName = ctx.currentAgentName || draft.name || '';
            draft.name = String(nameEl.value || '').trim();
            if (!draft.name) {
                alert('Agent musí mít název');
                return;
            }
            draft.display_name = draft.display_name || draft.name;
            draft.prompt = promptEl ? promptEl.value : '';
            const selectedModel = modelEl ? String(modelEl.value || '').trim() : DEFAULT_AGENT_MODEL;
            draft.model = selectedModel || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(draft, draft.model);

            const payload = buildAgentSavePayload(ctx, draft);
            try {
                const res = await fetch('/agents/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(buildAgentPayload(ctx, payload)) });
                if (!res.ok) {
                    const txt = await res.text().catch(()=>'');
                    throw new Error('save failed ' + txt);
                }
                const data = await res.json().catch(() => null);
                const stored = (data && data.stored_name) ? data.stored_name : draft.name;
                const agentResponse = (data && data.agent) ? data.agent : payload;
                const normalized = normalizeAgentData({ ...(agentResponse || {}), name: stored });

                // update caches
                ctx.agentsCache[stored] = normalized;
                cacheAgentFingerprint(ctx, stored, normalized);
                if (previousName && stored !== previousName) {
                    delete ctx.agentsCache[previousName];
                    ctx.agentFingerprintCache.delete(previousName);
                }

                persistContextSelectedAgent(ctx, stored);
                ctx.currentAgentName = stored;
                ctx.currentAgentDraft = deepCloneAgent(normalized);
                await renderAgentSelector(ctx, true);
                setAgentFields(ctx, normalized);
                resetAgentDirty(ctx);
                showTooltip(ctx.saveTooltipId || 'agent-save-tooltip', 'Agent uložen');
            } catch (err) {
                alert('Nelze uložit agenta: ' + err);
                showTooltip(ctx.saveTooltipId || 'agent-save-tooltip', 'Uložení selhalo', 2600);
            }
        }

        async function deleteCurrentAgent(ctx) {
            const select = getContextElement(ctx, 'selectId');
            if (!select) return;
            const name = select.value;
            if (!name) return;
            if (ctx.collection === 'joiners' && name === MANUAL_JOINER_NAME) {
                return;
            }
            if (!confirm(`Opravdu chcete agenta "${name}" smazat?`)) return;
            try {
                const res = await fetch('/agents/delete', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(buildAgentPayload(ctx, { name })) });
                if (!res.ok) throw new Error('delete failed');
                delete ctx.agentsCache[name];
                ctx.agentFingerprintCache.delete(name);
                persistContextSelectedAgent(ctx, '');
                await renderAgentSelector(ctx, true);
            } catch (err) {
                alert('Nelze smazat agenta: ' + err);
            }
        }

        // newAgent removed — creation/editing handled via saveCurrentAgent and changing name

        function repairJsonStringLineBreaks(raw) {
            if (typeof raw !== 'string' || !raw) {
                return raw;
            }
            let needsRepair = false;
            for (let i = 0; i < raw.length; i += 1) {
                const code = raw.charCodeAt(i);
                if (code === 10 || code === 13 || code === 0x2028 || code === 0x2029) {
                    needsRepair = true;
                    break;
                }
            }
            if (!needsRepair) {
                return raw;
            }
            let fixed = '';
            let inString = false;
            let escaping = false;
            for (let i = 0; i < raw.length; i += 1) {
                const ch = raw[i];
                const code = raw.charCodeAt(i);
                if (escaping) {
                    fixed += ch;
                    escaping = false;
                    continue;
                }
                if (code === 92) {
                    fixed += ch;
                    escaping = true;
                    continue;
                }
                if (code === 34) {
                    inString = !inString;
                    fixed += ch;
                    continue;
                }
                if (inString && (code === 10 || code === 13 || code === 0x2028 || code === 0x2029)) {
                    fixed += '\\n';
                    continue;
                }
                fixed += ch;
            }
            return fixed;
        }


        function parseAgentResultDocument(text, options = {}) {
            if (!text || typeof text !== 'string') {
                return null;
            }
            const trimmed = text.trim();
            if (!trimmed) {
                return null;
            }
            try {
                return JSON.parse(trimmed);
            } catch (err) {
                const repaired = repairJsonStringLineBreaks(trimmed);
                if (repaired && repaired !== trimmed) {
                    try {
                        console.warn('[JoinerDebug] parseAgentResultDocument repaired newline characters.');
                        return JSON.parse(repaired);
                    } catch (repairError) {
                        if (!(options && options.silent)) {
                            console.warn('[JoinerDebug] parseAgentResultDocument repair failed:', repairError);
                            console.warn('[JoinerDebug] Original text:', trimmed);
                            console.warn('[JoinerDebug] Repaired text:', repaired);
                        }
                    }
                } else if (!(options && options.silent)) {
                    console.warn('[JoinerDebug] parseAgentResultDocument parse error:', err);
                    console.warn('[JoinerDebug] Document text:', trimmed);
                }
                return null;
            }
        }

        function documentBlocksToHtml(documentPayload) {
            if (!documentPayload || typeof documentPayload !== 'object' || !Array.isArray(documentPayload.blocks)) {
                return null;
            }
            const parts = [];
            for (const block of documentPayload.blocks) {
                if (!block || typeof block.text !== 'string') {
                    continue;
                }
                const text = block.text.trim();
                if (!text) {
                    continue;
                }
                const type = (block.type || '').toLowerCase();
                let tag = 'p';
                let attrs = '';
                let markup = '';
                switch (type) {
                    case 'h1':
                    case 'h2':
                    case 'h3':
                        tag = type;
                        break;
                    case 'small':
                        markup = `<p><small>${escapeHtml(text)}</small></p>`;
                        break;
                    case 'note':
                        tag = 'note';
                        attrs = NOTE_STYLE_ATTR;
                        break;
                    case 'centered':
                        tag = 'div';
                        attrs = ' class="centered"';
                        break;
                    case 'blockquote':
                        tag = 'blockquote';
                        break;
                    case 'li':
                        tag = 'p';
                        attrs = ' data-block-type="li"';
                        break;
                    default:
                        tag = 'p';
                        if (type && type !== 'p') {
                            attrs = ` data-block-type="${escapeHtml(type)}"`;
                        }
                        break;
                }
                if (!markup) {
                    if (tag === 'note') {
                        markup = `<br><${tag}${attrs}>${escapeHtml(text)}</${tag}>`;
                    } else {
                        markup = `<${tag}${attrs}>${escapeHtml(text)}</${tag}>`;
                    }
                }
                parts.push(markup);
            }
            return parts.length ? parts.join("") : null;
        }

        function documentBlocksToText(documentPayload) {
            if (!documentPayload || typeof documentPayload !== 'object' || !Array.isArray(documentPayload.blocks)) {
                return '';
            }
            const parts = [];
            for (const block of documentPayload.blocks) {
                if (!block || typeof block.text !== 'string') {
                    continue;
                }
                const normalized = block.text.replace(/\s+/g, ' ').trim();
                if (normalized) {
                    parts.push(normalized);
                }
            }
            return parts.join('\\n\\n');
        }



        function setContextAgentOutput(ctx, statusText, bodyText, state) {
            if (!ctx) {
                return;
            }
            const container = getContextElement(ctx, 'outputContainerId');
            const statusEl = getContextElement(ctx, 'outputStatusId');
            const textEl = getContextElement(ctx, 'outputTextId');
            if (!container || !statusEl || !textEl) {
                return;
            }

            const normalizedStatus = statusText ? String(statusText).trim() : '';
            const normalizedBody = bodyText ? String(bodyText).trim() : '';
            if (!normalizedStatus && !normalizedBody) {
                container.style.display = 'none';
                statusEl.textContent = '';
                statusEl.style.color = '';
                statusEl.style.fontWeight = '';
                textEl.textContent = '';
                scheduleThumbnailDrawerHeightSync();
                return;
            }

            container.style.display = 'block';
            statusEl.textContent = normalizedStatus;
            if (state === 'error') {
                statusEl.style.color = '#b91c1c';
                statusEl.style.fontWeight = '';
            } else if (state === 'pending') {
                statusEl.style.color = '';
                statusEl.style.fontWeight = '600';
            } else {
                statusEl.style.color = '';
                statusEl.style.fontWeight = '';
            }
            textEl.textContent = normalizedBody;
            scheduleThumbnailDrawerHeightSync();
        }

        function clearAgentOutput() {
            setContextAgentOutput(agentContexts.correctors, '', '', '');
            setContextAgentOutput(agentContexts.readers, '', '', '');
            setContextAgentOutput(agentContexts.joiners, '', '', '');
            setAgentResultPanels(null, null, false);
            setReaderResultPanels('', '', { forceVisible: false });
            lastReaderResultHtml = '';
            lastReaderResultIsHtml = false;
            rejectComparisonWaiters('corrector', new Error('Výsledek byl zrušen.'));
            rejectComparisonWaiters('reader', new Error('Výsledek byl zrušen.'));
            resetComparisonResults();
            resetComparison2Results();
        }

        function setAgentOutput(statusText, bodyText, state) {
            setContextAgentOutput(agentContexts.correctors, statusText, bodyText, state);
        }

        function formatHtmlForPre(html) {
            if (!html) {
                return '';
            }
            return escapeHtml(html || '');
        }

        function setAgentResultPanels(originalHtml, correctedContent, correctedIsHtml) {
            lastAgentOriginalHtml = originalHtml || '';
            lastAgentCorrectedHtml = correctedContent || '';
            lastAgentCorrectedIsHtml = Boolean(correctedIsHtml);
            if (!originalHtml || !correctedContent) {
                lastAgentCacheBaseKey = null;
            }
            if (!originalHtml) {
                lastAgentOriginalDocumentJson = '';
            }
            if (!correctedContent) {
                lastAgentCorrectedDocumentJson = '';
            }
            const container = document.getElementById('agent-results');
            const originalEl = document.getElementById('agent-result-original');
            const correctedEl = document.getElementById('agent-result-corrected');
            if (!container || !originalEl || !correctedEl) {
                return;
            }

            console.group('[AgentDebug] Agent result panels');
            console.debug('[AgentDebug] Original Python HTML length:', originalHtml ? originalHtml.length : 0);
            console.debug('[AgentDebug] Original Python HTML preview:', originalHtml);
            console.debug('[AgentDebug] Corrected content length:', correctedContent ? correctedContent.length : 0, 'isHtml:', correctedIsHtml);
            console.debug('[AgentDebug] Corrected content preview:', correctedContent);
            console.groupEnd();

            if (!originalHtml && !correctedContent) {
                container.style.display = 'none';
                originalEl.innerHTML = '';
                correctedEl.innerHTML = '';
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                scheduleThumbnailDrawerHeightSync();
                return;
            }

            container.style.display = 'grid';

            if (originalHtml) {
                originalEl.innerHTML = `<pre>${originalHtml}</pre>`;
            } else {
                originalEl.innerHTML = '<div class="muted">Žádná data.</div>';
            }

            if (correctedContent) {
                correctedEl.innerHTML = correctedIsHtml
                    ? `<pre>${correctedContent}</pre>`
                    : `<pre>${escapeHtml(correctedContent)}</pre>`;
            } else {
                correctedEl.innerHTML = '<div class="muted">Agent nevrátil HTML náhled.</div>';
            }

            if (correctedContent && correctedContent.trim()) {
                resolveComparisonWaiters('corrector', {
                    html: correctedContent,
                    isHtml: correctedIsHtml,
                });
            }

            const shouldHideDiff = !(originalHtml && correctedContent);
            renderAgentDiff(null, agentDiffMode, { hidden: shouldHideDiff });
            scheduleThumbnailDrawerHeightSync();
        }

        function isLikelyHtmlContent(value) {
            if (typeof value !== 'string') {
                return false;
            }
            const trimmed = value.trim();
            if (!trimmed.startsWith('<') || !trimmed.endsWith('>')) {
                return false;
            }
            return /<\/?[a-zA-Z][\s\S]*?>/.test(trimmed);
        }

        function setReaderResultPanels(imageUrl, textContent, options = {}) {
            const container = document.getElementById('reader-results');
            const frame = document.getElementById('reader-result-scan');
            const img = document.getElementById('reader-result-scan-img');
            const textEl = document.getElementById('reader-result-text');
            const placeholder = frame ? frame.querySelector('.scan-result-placeholder') : null;
            if (!container || !frame || !img || !textEl) {
                return;
            }
            const normalizedText = typeof textContent === 'string' ? textContent : '';
            const hasText = normalizedText.trim().length > 0;
            const hasImage = Boolean(imageUrl);
            if (hasImage) {
                if (img.src !== imageUrl) {
                    img.src = imageUrl;
                }
                img.style.display = 'block';
                frame.classList.add('has-image');
                if (placeholder) {
                    placeholder.style.display = 'none';
                }
            } else {
                img.removeAttribute('src');
                img.style.display = 'none';
                frame.classList.remove('has-image');
                if (placeholder) {
                    placeholder.style.display = 'block';
                }
            }
            const forceVisible = Boolean(options && options.forceVisible);
            if (!forceVisible && !hasImage && !hasText) {
                container.style.display = 'none';
                textEl.innerHTML = '<div class="muted">Výsledek se zobrazí po spuštění „Vyčti“.</div>';
                scheduleThumbnailDrawerHeightSync();
                return;
            }
            container.style.display = 'grid';
            if (hasText) {
                const treatAsHtml = Boolean(options && options.isHtml) || isLikelyHtmlContent(normalizedText);
                lastReaderResultHtml = normalizedText;
                lastReaderResultIsHtml = treatAsHtml;
                resolveComparisonWaiters('reader', {
                    html: normalizedText,
                    isHtml: treatAsHtml,
                });
                textEl.innerHTML = `<pre>${buildPreMarkup(normalizedText, treatAsHtml)}</pre>`;
            } else {
                lastReaderResultHtml = '';
                lastReaderResultIsHtml = false;
                textEl.innerHTML = '<div class="muted">Formátovaný text zatím není k dispozici.</div>';
            }
            scheduleThumbnailDrawerHeightSync();
        }

        async function getReaderScanPreviewUrl() {
            if (!currentPage || !currentPage.uuid) {
                return '';
            }
            if (previewImageUuid === currentPage.uuid && previewObjectUrl) {
                return previewObjectUrl;
            }
            try {
                const entry = await ensurePreviewEntry(currentPage.uuid);
                if (entry && entry.objectUrl) {
                    return entry.objectUrl;
                }
            } catch (error) {
                console.warn('Nelze načíst náhled pro čtení ze skenu:', error);
            }
            return `/preview?uuid=${encodeURIComponent(currentPage.uuid)}&stream=IMG_FULL`;
        }

        function buildPreMarkup(content, treatAsHtml) {
            if (!content) {
                return '';
            }
            if (treatAsHtml) {
                return content.replace(/>\s+</g, '><').trim();
            }
            return escapeHtml(content);
        }

        function normalizeContentForDiff(content, isHtml) {
            if (!content) {
                return '';
            }
            if (isHtml) {
                return content.replace(/>\s+</g, '><').trim();
            }
            return content;
        }

        async function applyReaderAgentResult(ctx, agentResult, rawText) {
            const normalizedRaw = typeof rawText === 'string' ? rawText.trim() : '';
            const htmlOutput = agentResult && typeof agentResult.html === 'string' ? agentResult.html.trim() : '';
            const formattedText = agentResult && typeof agentResult.formatted_text === 'string'
                ? agentResult.formatted_text.trim()
                : '';
            const finalText = htmlOutput || formattedText || normalizedRaw;
            const scanUrl = await getReaderScanPreviewUrl();
            setReaderResultPanels(scanUrl, finalText, {
                isHtml: Boolean(htmlOutput),
                forceVisible: Boolean(scanUrl || finalText),
            });
            return Boolean(scanUrl || finalText);
        }

        function setComparisonStatus(message, state) {
            const statusEl = document.getElementById('comparison-status');
            if (!statusEl) {
                return;
            }
            statusEl.classList.remove('is-error', 'is-pending');
            if (!message) {
                statusEl.textContent = '';
                statusEl.style.display = 'none';
                return;
            }
            statusEl.textContent = message;
            statusEl.style.display = 'block';
            if (state === 'error') {
                statusEl.classList.add('is-error');
            } else if (state === 'pending') {
                statusEl.classList.add('is-pending');
            }
        }

        function setComparisonRunButtonState(isRunning) {
            const runBtn = document.getElementById('comparison-run');
            if (!runBtn) {
                return;
            }
            if (!comparisonRunButtonLabel) {
                comparisonRunButtonLabel = runBtn.textContent || 'Porovnej';
            }
            if (isRunning) {
                runBtn.disabled = true;
                runBtn.textContent = 'Porovnávám...';
            } else {
                runBtn.disabled = false;
                runBtn.textContent = comparisonRunButtonLabel;
            }
        }

        function resetComparisonResults() {
            const container = document.getElementById('comparison-results');
            const leftEl = document.getElementById('comparison-result-left');
            const rightEl = document.getElementById('comparison-result-right');
            if (container) {
                container.style.display = 'none';
            }
            if (leftEl) {
                leftEl.innerHTML = '';
            }
            if (rightEl) {
                rightEl.innerHTML = '';
            }
            setComparisonDiffControlsVisible(false);
            comparisonOutputs.leftHtml = '';
            comparisonOutputs.leftIsHtml = false;
            comparisonOutputs.leftDisplay = '';
            comparisonOutputs.rightHtml = '';
            comparisonOutputs.rightIsHtml = false;
            comparisonOutputs.rightDisplay = '';
            comparisonOutputs.diffCache.word = null;
            comparisonOutputs.diffCache.char = null;
            comparisonDiffRequestToken += 1;
        }

        function setComparisonDiffControlsVisible(visible) {
            const controls = document.getElementById('comparison-diff-mode-controls');
            if (controls) {
                controls.style.display = visible ? 'inline-flex' : 'none';
            }
        }

        function updateComparisonDiffToggleState() {
            const container = document.getElementById('comparison-diff-mode-controls');
            if (!container) {
                return;
            }
            const buttons = container.querySelectorAll('.agent-diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) {
                    return;
                }
                const mode = button.getAttribute('data-mode');
                const isActive = mode === comparisonDiffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }

        function updateComparisonResultPanels(originalContent, originalIsHtml, readerContent, readerIsHtml) {
            const container = document.getElementById('comparison-results');
            const leftEl = document.getElementById('comparison-result-left');
            const rightEl = document.getElementById('comparison-result-right');
            if (!container || !leftEl || !rightEl) {
                return;
            }
            if (!originalContent && !readerContent) {
                container.style.display = 'none';
                leftEl.innerHTML = '';
                rightEl.innerHTML = '';
                return;
            }
            container.style.display = 'grid';
            const leftDisplay = originalContent ? buildPreMarkup(originalContent, originalIsHtml) : '';
            const rightDisplay = readerContent ? buildPreMarkup(readerContent, readerIsHtml) : '';
            leftEl.innerHTML = originalContent
                ? `<pre>${leftDisplay}</pre>`
                : '<div class="muted">Žádná data.</div>';
            rightEl.innerHTML = readerContent
                ? `<pre>${rightDisplay}</pre>`
                : '<div class="muted">Žádná data.</div>';
            comparisonOutputs.leftHtml = normalizeContentForDiff(originalContent, originalIsHtml);
            comparisonOutputs.leftIsHtml = Boolean(originalIsHtml);
            comparisonOutputs.leftDisplay = leftDisplay;
            comparisonOutputs.rightHtml = normalizeContentForDiff(readerContent, readerIsHtml);
            comparisonOutputs.rightIsHtml = Boolean(readerIsHtml);
            comparisonOutputs.rightDisplay = rightDisplay;
            comparisonOutputs.diffCache.word = null;
            comparisonOutputs.diffCache.char = null;
            comparisonDiffRequestToken += 1;
            setComparisonDiffControlsVisible(false);
            scheduleThumbnailDrawerHeightSync();
        }

        function applyComparisonDiffMarkup(diff) {
            const leftPre = document.querySelector('#comparison-result-left pre');
            const rightPre = document.querySelector('#comparison-result-right pre');
            if (!diff || !leftPre || !rightPre) {
                if (leftPre) {
                    leftPre.innerHTML = comparisonOutputs.leftDisplay || buildPreMarkup(comparisonOutputs.leftHtml, comparisonOutputs.leftIsHtml);
                }
                if (rightPre) {
                    rightPre.innerHTML = comparisonOutputs.rightDisplay || buildPreMarkup(comparisonOutputs.rightHtml, comparisonOutputs.rightIsHtml);
                }
                setComparisonDiffControlsVisible(false);
                return;
            }
            if (diff.original && leftPre) {
                leftPre.innerHTML = diff.original;
            }
            if (diff.corrected && rightPre) {
                rightPre.innerHTML = diff.corrected;
            }
            setComparisonDiffControlsVisible(true);
            scheduleThumbnailDrawerHeightSync();
        }

        function ensureCorrectorResult() {
            if (lastAgentCorrectedHtml && lastAgentCorrectedHtml.trim()) {
                return Promise.resolve({
                    html: lastAgentCorrectedHtml,
                    isHtml: lastAgentCorrectedIsHtml,
                });
            }
            const promise = new Promise((resolve, reject) => {
                comparisonResultWaiters.corrector.push({ resolve, reject });
            });
            triggerCorrectorRunIfIdle();
            return promise;
        }

        function ensureReaderResult() {
            if (lastReaderResultHtml && lastReaderResultHtml.trim()) {
                return Promise.resolve({
                    html: lastReaderResultHtml,
                    isHtml: lastReaderResultIsHtml,
                });
            }
            const promise = new Promise((resolve, reject) => {
                comparisonResultWaiters.reader.push({ resolve, reject });
            });
            triggerReaderRunIfIdle();
            return promise;
        }

        function triggerCorrectorRunIfIdle() {
            const ctx = agentContexts.correctors;
            if (!ctx) {
                rejectComparisonWaiters('corrector', new Error('Chybí konfigurace pro opravu textu.'));
                return;
            }
            const runBtn = getContextElement(ctx, 'runButtonId');
            if ((runBtn && runBtn.disabled) || comparisonState.correctorRunPending) {
                setComparisonStatus('Čekám na dokončení opravy textu...', 'pending');
                return;
            }
            const select = getContextElement(ctx, 'selectId');
            if (select && !select.value) {
                const error = new Error('Vyberte agenta pro opravu textu.');
                setComparisonStatus(error.message, 'error');
                rejectComparisonWaiters('corrector', error);
                return;
            }
            comparisonState.correctorRunPending = true;
            setComparisonStatus('Spouštím opravu textu...', 'pending');
            runSelectedAgent(ctx, { autoTriggered: false }).catch((error) => {
                rejectComparisonWaiters('corrector', error instanceof Error ? error : new Error('Oprava textu selhala.'));
            }).finally(() => {
                comparisonState.correctorRunPending = false;
                if (!(lastAgentCorrectedHtml && lastAgentCorrectedHtml.trim())) {
                    rejectComparisonWaiters('corrector', new Error('Oprava textu nevrátila výsledek.'));
                }
            });
        }

        function triggerReaderRunIfIdle() {
            const ctx = agentContexts.readers;
            if (!ctx) {
                rejectComparisonWaiters('reader', new Error('Chybí konfigurace pro čtení ze skenu.'));
                return;
            }
            const runBtn = getContextElement(ctx, 'runButtonId');
            if ((runBtn && runBtn.disabled) || comparisonState.readerRunPending) {
                setComparisonStatus('Čekám na dokončení čtení ze skenu...', 'pending');
                return;
            }
            const select = getContextElement(ctx, 'selectId');
            if (select && !select.value) {
                const error = new Error('Vyberte agenta pro čtení ze skenu.');
                setComparisonStatus(error.message, 'error');
                rejectComparisonWaiters('reader', error);
                return;
            }
            comparisonState.readerRunPending = true;
            setComparisonStatus('Spouštím čtení ze skenu...', 'pending');
            runSelectedAgent(ctx, { autoTriggered: false }).catch((error) => {
                rejectComparisonWaiters('reader', error instanceof Error ? error : new Error('Čtení ze skenu selhalo.'));
            }).finally(() => {
                comparisonState.readerRunPending = false;
                if (!(lastReaderResultHtml && lastReaderResultHtml.trim())) {
                    rejectComparisonWaiters('reader', new Error('Čtení ze skenu nevrátilo výsledek.'));
                }
            });
        }

        async function fetchComparisonDiffData(mode) {
            const payload = {
                python: comparisonOutputs.leftHtml || '',
                typescript: comparisonOutputs.rightHtml || '',
                mode: mode === COMPARISON_DIFF_MODES.CHAR ? 'char' : 'word',
            };
            const response = await fetch('/diff', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data || data.ok === false) {
                throw new Error((data && data.error) || 'Diff request selhal');
            }
            const diff = data.diff || {};
            return {
                original: diff.python || null,
                corrected: diff.typescript || null,
            };
        }

        async function refreshComparisonDiff() {
            if (!(comparisonOutputs.leftHtml && comparisonOutputs.rightHtml)) {
                applyComparisonDiffMarkup(null);
                return;
            }
            const cached = comparisonOutputs.diffCache[comparisonDiffMode];
            if (cached) {
                applyComparisonDiffMarkup(cached);
                updateComparisonDiffToggleState();
                return;
            }
            const token = ++comparisonDiffRequestToken;
            setComparisonStatus('Počítám rozdíly...', 'pending');
            try {
                const diff = await fetchComparisonDiffData(comparisonDiffMode);
                if (token !== comparisonDiffRequestToken) {
                    return;
                }
                comparisonOutputs.diffCache[comparisonDiffMode] = diff;
                applyComparisonDiffMarkup(diff);
                updateComparisonDiffToggleState();
                setComparisonStatus('');
            } catch (error) {
                if (token === comparisonDiffRequestToken) {
                    applyComparisonDiffMarkup(null);
                    setComparisonStatus('Porovnání se nepodařilo.', 'error');
                }
            }
        }

        function setComparisonDiffMode(mode) {
            if (mode !== COMPARISON_DIFF_MODES.WORD && mode !== COMPARISON_DIFF_MODES.CHAR) {
                return;
            }
            if (comparisonDiffMode === mode) {
                return;
            }
            comparisonDiffMode = mode;
            persistComparisonDiffMode(mode);
            comparisonOutputs.diffCache[mode] = comparisonOutputs.diffCache[mode] || null;
            comparisonDiffRequestToken += 1;
            updateComparisonDiffToggleState();
            refreshComparisonDiff().catch((error) => {
                console.warn('Porovnání diffu selhalo:', error);
            });
        }

        async function runComparison() {
            if (comparisonState.running) {
                setComparisonStatus('Porovnání již probíhá...', 'pending');
                setComparisonRunButtonState(true);
                return;
            }
            comparisonState.running = true;
             setComparisonRunButtonState(true);
            setComparisonStatus('Připravuji porovnání...', 'pending');
            comparisonDiffRequestToken += 1;
            try {
                const [correctorResult, readerResult] = await Promise.all([
                    ensureCorrectorResult(),
                    ensureReaderResult(),
                ]);
                updateComparisonResultPanels(
                    correctorResult && correctorResult.html ? correctorResult.html : '',
                    Boolean(correctorResult && correctorResult.isHtml),
                    readerResult && readerResult.html ? readerResult.html : '',
                    Boolean(readerResult && readerResult.isHtml)
                );
                setComparisonStatus('Počítám rozdíly...', 'pending');
                comparisonOutputs.diffCache.word = null;
                comparisonOutputs.diffCache.char = null;
                await refreshComparisonDiff();
                setComparisonStatus('');
            } catch (error) {
                const message = error && error.message ? error.message : 'Porovnání se nepodařilo.';
                setComparisonStatus(message, 'error');
            } finally {
                comparisonState.running = false;
                setComparisonRunButtonState(false);
            }
        }

        function handleComparisonDiffToggle(event) {
            const target = event.target;
            if (!target || !(target instanceof HTMLElement) || !target.matches('.agent-diff-toggle')) {
                return;
            }
            const requestedMode = target.getAttribute('data-mode');
            if (!requestedMode) {
                return;
            }
            setComparisonDiffMode(requestedMode);
        }

        const COMPARISON_AUTO_STORAGE_KEY = 'altoComparisonAutoRun_v1';
        function loadComparisonAutoFlag() {
            try {
                return localStorage.getItem(COMPARISON_AUTO_STORAGE_KEY) === '1';
            } catch (error) {
                return false;
            }
        }

        function persistComparisonAutoFlag(value) {
            try {
                localStorage.setItem(COMPARISON_AUTO_STORAGE_KEY, value ? '1' : '0');
            } catch (error) {
                // ignore
            }
        }

        function scheduleComparisonAutoRun(delay = 350) {
            const checkbox = document.getElementById('comparison-auto-run');
            if (!checkbox || !checkbox.checked) {
                comparisonState.autoRunScheduled = false;
                return;
            }
            if (comparisonState.autoRunScheduled || comparisonState.running) {
                return;
            }
            comparisonState.autoRunScheduled = true;
            window.setTimeout(() => {
                comparisonState.autoRunScheduled = false;
                if (checkbox.checked) {
                    runComparison().catch((error) => {
                        console.warn('Automatické porovnání selhalo:', error);
                    });
                }
            }, delay);
        }

        function initializeComparisonUI() {
            comparisonDiffMode = loadStoredComparisonDiffMode();
            updateComparisonDiffToggleState();
            const runBtn = document.getElementById('comparison-run');
            if (runBtn) {
                runBtn.addEventListener('click', () => {
                    runComparison().catch((error) => {
                        console.warn('Porovnání selhalo:', error);
                    });
                });
            }
            const autoCheckbox = document.getElementById('comparison-auto-run');
            if (autoCheckbox) {
                autoCheckbox.checked = loadComparisonAutoFlag();
                autoCheckbox.addEventListener('change', () => {
                    persistComparisonAutoFlag(autoCheckbox.checked);
                    if (autoCheckbox.checked) {
                        scheduleComparisonAutoRun(200);
                    }
                });
            }
            const diffControls = document.getElementById('comparison-diff-mode-controls');
            if (diffControls) {
                diffControls.addEventListener('click', handleComparisonDiffToggle);
            }
        }

        // --- Secondary comparison (processed ALTO Python vs OCR) ---
        function setComparison2Status(message, state) {
            const statusEl = document.getElementById('comparison2-status');
            if (!statusEl) {
                return;
            }
            statusEl.classList.remove('is-error', 'is-pending');
            if (!message) {
                statusEl.textContent = '';
                statusEl.style.display = 'none';
                return;
            }
            statusEl.textContent = message;
            statusEl.style.display = 'block';
            if (state === 'error') {
                statusEl.classList.add('is-error');
            } else if (state === 'pending') {
                statusEl.classList.add('is-pending');
            }
        }

        function setComparison2RunButtonState(isRunning) {
            const runBtn = document.getElementById('comparison2-run');
            if (!runBtn) {
                return;
            }
            if (isRunning) {
                runBtn.disabled = true;
                runBtn.textContent = 'Porovnávám...';
            } else {
                runBtn.disabled = false;
                runBtn.textContent = 'Porovnej';
            }
        }

        function resetComparison2Results() {
            const container = document.getElementById('comparison2-results');
            const leftEl = document.getElementById('comparison2-result-left');
            const rightEl = document.getElementById('comparison2-result-right');
            if (container) {
                container.style.display = 'none';
            }
            if (leftEl) leftEl.innerHTML = '';
            if (rightEl) rightEl.innerHTML = '';
            setComparison2DiffControlsVisible(false);
            comparison2Outputs.leftHtml = '';
            comparison2Outputs.leftIsHtml = false;
            comparison2Outputs.leftDisplay = '';
            comparison2Outputs.rightHtml = '';
            comparison2Outputs.rightIsHtml = false;
            comparison2Outputs.rightDisplay = '';
            comparison2Outputs.diffCache.word = null;
            comparison2Outputs.diffCache.char = null;
            comparison2DiffRequestToken += 1;
        }

        function setComparison2DiffControlsVisible(visible) {
            const controls = document.getElementById('comparison2-diff-mode-controls');
            if (controls) controls.style.display = visible ? 'inline-flex' : 'none';
        }

        function updateComparison2DiffToggleState() {
            const container = document.getElementById('comparison2-diff-mode-controls');
            if (!container) return;
            const buttons = container.querySelectorAll('.agent-diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) return;
                const mode = button.getAttribute('data-mode');
                const isActive = mode === comparison2DiffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }

        function updateComparison2ResultPanels(originalContent, originalIsHtml, readerContent, readerIsHtml) {
            const container = document.getElementById('comparison2-results');
            const leftEl = document.getElementById('comparison2-result-left');
            const rightEl = document.getElementById('comparison2-result-right');
            if (!container || !leftEl || !rightEl) return;
            if (!originalContent && !readerContent) {
                container.style.display = 'none';
                leftEl.innerHTML = '';
                rightEl.innerHTML = '';
                return;
            }
            container.style.display = 'grid';
            const leftDisplay = originalContent ? buildPreMarkup(originalContent, originalIsHtml) : '';
            const rightDisplay = readerContent ? buildPreMarkup(readerContent, readerIsHtml) : '';
            leftEl.innerHTML = originalContent ? `<pre>${leftDisplay}</pre>` : '<div class="muted">Žádná data.</div>';
            rightEl.innerHTML = readerContent ? `<pre>${rightDisplay}</pre>` : '<div class="muted">Žádná data.</div>';
            comparison2Outputs.leftHtml = normalizeContentForDiff(originalContent, originalIsHtml);
            comparison2Outputs.leftIsHtml = Boolean(originalIsHtml);
            comparison2Outputs.leftDisplay = leftDisplay;
            comparison2Outputs.rightHtml = normalizeContentForDiff(readerContent, readerIsHtml);
            comparison2Outputs.rightIsHtml = Boolean(readerIsHtml);
            comparison2Outputs.rightDisplay = rightDisplay;
            comparison2Outputs.diffCache.word = null;
            comparison2Outputs.diffCache.char = null;
            comparison2DiffRequestToken += 1;
            setComparison2DiffControlsVisible(false);
            scheduleThumbnailDrawerHeightSync();
        }

        async function fetchComparison2DiffData(mode) {
            const payload = {
                python: comparison2Outputs.leftHtml || '',
                typescript: comparison2Outputs.rightHtml || '',
                mode: mode === COMPARISON_DIFF_MODES.CHAR ? 'char' : 'word',
            };
            const response = await fetch('/diff', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data || data.ok === false) {
                throw new Error((data && data.error) || 'Diff request selhal');
            }
            const diff = data.diff || {};
            return {
                original: diff.python || null,
                corrected: diff.typescript || null,
            };
        }

        function applyComparison2DiffMarkup(diff) {
            const leftPre = document.querySelector('#comparison2-result-left pre');
            const rightPre = document.querySelector('#comparison2-result-right pre');
            if (!diff || !leftPre || !rightPre) {
                if (leftPre) {
                    leftPre.innerHTML = comparison2Outputs.leftDisplay || buildPreMarkup(comparison2Outputs.leftHtml, comparison2Outputs.leftIsHtml);
                }
                if (rightPre) {
                    rightPre.innerHTML = comparison2Outputs.rightDisplay || buildPreMarkup(comparison2Outputs.rightHtml, comparison2Outputs.rightIsHtml);
                }
                setComparison2DiffControlsVisible(false);
                return;
            }
            if (diff.original && leftPre) leftPre.innerHTML = diff.original;
            if (diff.corrected && rightPre) rightPre.innerHTML = diff.corrected;
            setComparison2DiffControlsVisible(true);
            scheduleThumbnailDrawerHeightSync();
        }

        async function refreshComparison2Diff() {
            if (!(comparison2Outputs.leftHtml && comparison2Outputs.rightHtml)) {
                applyComparison2DiffMarkup(null);
                return;
            }
            const cached = comparison2Outputs.diffCache[comparison2DiffMode];
            if (cached) {
                applyComparison2DiffMarkup(cached);
                updateComparison2DiffToggleState();
                return;
            }
            const token = ++comparison2DiffRequestToken;
            setComparison2Status('Počítám rozdíly...', 'pending');
            try {
                const diff = await fetchComparison2DiffData(comparison2DiffMode);
                if (token !== comparison2DiffRequestToken) return;
                comparison2Outputs.diffCache[comparison2DiffMode] = diff;
                applyComparison2DiffMarkup(diff);
                updateComparison2DiffToggleState();
                setComparison2Status('');
            } catch (error) {
                if (token === comparison2DiffRequestToken) {
                    applyComparison2DiffMarkup(null);
                    setComparison2Status('Porovnání se nepodařilo.', 'error');
                }
            }
        }

        function setComparison2DiffMode(mode) {
            if (mode !== COMPARISON_DIFF_MODES.WORD && mode !== COMPARISON_DIFF_MODES.CHAR) return;
            if (comparison2DiffMode === mode) return;
            comparison2DiffMode = mode;
            try { if (mode === COMPARISON_DIFF_MODES.WORD || mode === COMPARISON_DIFF_MODES.CHAR) localStorage.setItem(COMPARISON2_DIFF_MODE_STORAGE_KEY, mode); } catch (e) {}
            comparison2Outputs.diffCache[mode] = comparison2Outputs.diffCache[mode] || null;
            comparison2DiffRequestToken += 1;
            updateComparison2DiffToggleState();
            refreshComparison2Diff().catch((error) => {
                console.warn('Porovnání diffu (2) selhalo:', error);
            });
        }

        const COMPARISON2_AUTO_STORAGE_KEY = 'altoComparison2AutoRun_v1';
        function loadComparison2AutoFlag() { try { return localStorage.getItem(COMPARISON2_AUTO_STORAGE_KEY) === '1'; } catch (e) { return false; } }
        function persistComparison2AutoFlag(value) { try { localStorage.setItem(COMPARISON2_AUTO_STORAGE_KEY, value ? '1' : '0'); } catch (e) {} }

        function scheduleComparison2AutoRun(delay = 350) {
            const checkbox = document.getElementById('comparison2-auto-run');
            if (!checkbox || !checkbox.checked) {
                comparison2State.autoRunScheduled = false;
                return;
            }
            if (comparison2State.autoRunScheduled || comparison2State.running) return;
            comparison2State.autoRunScheduled = true;
            window.setTimeout(() => {
                comparison2State.autoRunScheduled = false;
                if (checkbox.checked) {
                    runComparison2().catch((error) => { console.warn('Automatické porovnání (2) selhalo:', error); });
                }
            }, delay);
        }

        async function runComparison2() {
            if (comparison2State.running) {
                setComparison2Status('Porovnání již probíhá...', 'pending');
                setComparison2RunButtonState(true);
                return;
            }
            comparison2State.running = true;
            setComparison2RunButtonState(true);
            setComparison2Status('Připravuji porovnání...', 'pending');
            comparison2DiffRequestToken += 1;
            try {
                // left: processed ALTO (python), right: reader (OCR)
                const pythonHtml = currentResults.python || '';
                const leftIsHtml = typeof pythonHtml === 'string' && /<[^>]+>/.test(pythonHtml);
                const readerResult = await ensureReaderResult();
                updateComparison2ResultPanels(
                    pythonHtml || '',
                    Boolean(leftIsHtml),
                    readerResult && readerResult.html ? readerResult.html : '',
                    Boolean(readerResult && readerResult.isHtml)
                );
                setComparison2Status('Počítám rozdíly...', 'pending');
                comparison2Outputs.diffCache.word = null;
                comparison2Outputs.diffCache.char = null;
                await refreshComparison2Diff();
                setComparison2Status('');
            } catch (error) {
                const message = error && error.message ? error.message : 'Porovnání se nepodařilo.';
                setComparison2Status(message, 'error');
            } finally {
                comparison2State.running = false;
                setComparison2RunButtonState(false);
            }
        }

        function handleComparison2DiffToggle(event) {
            const target = event.target;
            if (!target || !(target instanceof HTMLElement) || !target.matches('.agent-diff-toggle')) return;
            const requestedMode = target.getAttribute('data-mode');
            if (!requestedMode) return;
            setComparison2DiffMode(requestedMode);
        }

        function initializeComparison2UI() {
            try {
                const stored = localStorage.getItem(COMPARISON2_DIFF_MODE_STORAGE_KEY);
                if (stored === COMPARISON_DIFF_MODES.WORD || stored === COMPARISON_DIFF_MODES.CHAR) comparison2DiffMode = stored;
            } catch (e) {}
            updateComparison2DiffToggleState();
            const runBtn = document.getElementById('comparison2-run');
            if (runBtn) {
                runBtn.addEventListener('click', () => { runComparison2().catch((error) => { console.warn('Porovnání (2) selhalo:', error); }); });
            }
            const autoCheckbox = document.getElementById('comparison2-auto-run');
            if (autoCheckbox) {
                autoCheckbox.checked = loadComparison2AutoFlag();
                autoCheckbox.addEventListener('change', () => { persistComparison2AutoFlag(autoCheckbox.checked); if (autoCheckbox.checked) scheduleComparison2AutoRun(200); });
            }
            const diffControls = document.getElementById('comparison2-diff-mode-controls');
            if (diffControls) diffControls.addEventListener('click', handleComparison2DiffToggle);
        }

        function loadStoredAgentDiffMode() {
            try {
                const stored = localStorage.getItem(AGENT_DIFF_MODE_STORAGE_KEY);
                if (stored === AGENT_DIFF_MODES.WORD || stored === AGENT_DIFF_MODES.CHAR) {
                    return stored;
                }
            } catch (error) {
                console.warn('Nelze načíst uložený režim agent diffu:', error);
            }
            return AGENT_DIFF_MODES.WORD;
        }

        function persistAgentDiffMode(mode) {
            try {
                if (mode === AGENT_DIFF_MODES.WORD || mode === AGENT_DIFF_MODES.CHAR) {
                    localStorage.setItem(AGENT_DIFF_MODE_STORAGE_KEY, mode);
                }
            } catch (error) {
                console.warn('Nelze uložit režim agent diffu:', error);
            }
        }

        function loadStoredComparisonDiffMode() {
            try {
                const stored = localStorage.getItem(COMPARISON_DIFF_MODE_STORAGE_KEY);
                if (stored === COMPARISON_DIFF_MODES.WORD || stored === COMPARISON_DIFF_MODES.CHAR) {
                    return stored;
                }
            } catch (error) {
                console.warn('Nelze načíst režim porovnání:', error);
            }
            return COMPARISON_DIFF_MODES.WORD;
        }

        function persistComparisonDiffMode(mode) {
            try {
                if (mode === COMPARISON_DIFF_MODES.WORD || mode === COMPARISON_DIFF_MODES.CHAR) {
                    localStorage.setItem(COMPARISON_DIFF_MODE_STORAGE_KEY, mode);
                }
            } catch (error) {
                console.warn('Nelze uložit režim porovnání:', error);
            }
        }

        function updateAgentDiffToggleState() {
            const container = document.getElementById('agent-diff-mode-controls');
            if (!container) {
                return;
            }
            const buttons = container.querySelectorAll('.agent-diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) {
                    return;
                }
                const mode = button.getAttribute('data-diff-mode');
                const isActive = mode === agentDiffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }

        function renderAgentDiff(diff, mode, options) {
            const originalEl = document.getElementById('agent-result-original');
            const correctedEl = document.getElementById('agent-result-corrected');
            const controls = document.getElementById('agent-diff-mode-controls');
            updateAgentDiffToggleState();
            if (!originalEl || !correctedEl) {
                return;
            }
            const shouldHide = Boolean(options && options.hidden);
            if (!diff || shouldHide) {
                if (controls) {
                    controls.style.display = 'none';
                }
                scheduleThumbnailDrawerHeightSync();
                return;
            }
            const originalPre = originalEl.querySelector('pre');
            const correctedPre = correctedEl.querySelector('pre');
            if (diff.original && originalPre) {
                originalPre.innerHTML = diff.original;
            } else if (diff.original && !originalPre) {
                originalEl.innerHTML = `<pre>${diff.original}</pre>`;
            }
            if (diff.corrected && correctedPre) {
                correctedPre.innerHTML = diff.corrected;
            } else if (diff.corrected && !correctedPre) {
                correctedEl.innerHTML = `<pre>${diff.corrected}</pre>`;
            }
            if (controls) {
                controls.style.display = 'inline-flex';
            }
            scheduleThumbnailDrawerHeightSync();
        }

        function resolveComparisonWaiters(type, payload) {
            const waiters = comparisonResultWaiters[type];
            if (!waiters || !waiters.length) {
                comparisonResultWaiters[type] = [];
                return;
            }
            comparisonResultWaiters[type] = [];
            waiters.forEach(({ resolve }) => {
                try {
                    resolve(payload);
                } catch (error) {
                    console.warn('Comparison waiter resolve chyba:', error);
                }
            });
        }

        function rejectComparisonWaiters(type, error) {
            const waiters = comparisonResultWaiters[type];
            if (!waiters || !waiters.length) {
                comparisonResultWaiters[type] = [];
                return;
            }
            comparisonResultWaiters[type] = [];
            waiters.forEach(({ reject }) => {
                try {
                    reject(error);
                } catch (err) {
                    console.warn('Comparison waiter reject chyba:', err);
                }
            });
        }

        async function fetchAgentDiff(originalHtml, correctedHtml, mode) {
            if (!originalHtml || !correctedHtml) {
                return null;
            }
            try {
                const response = await fetch('/agents/diff', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        original: originalHtml,
                        corrected: correctedHtml,
                        mode: mode || agentDiffMode || AGENT_DIFF_MODES.WORD,
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                return data;
            } catch (error) {
                console.warn('Agent diff request selhal:', error);
                return null;
            }
        }

        function computeAgentCacheBaseKey() {
            if (lastAgentCacheBaseKey) {
                return lastAgentCacheBaseKey;
            }
            const select = document.getElementById('agent-select');
            const name = select && select.value ? String(select.value) : '';
            if (!name || !lastAgentOriginalHtml) {
                return null;
            }
            const fingerprint = computeAgentFingerprint(agentContexts.correctors, name);
            if (!fingerprint) {
                return null;
            }
            return `${fingerprint}:${computeSimpleHash(lastAgentOriginalHtml)}`;
        }

        async function requestAgentDiff(originalHtml, correctedHtml, correctedIsHtml, cacheBaseKey) {
            const originalPayload = lastAgentOriginalDocumentJson || originalHtml;
            const correctedPayload = lastAgentCorrectedDocumentJson || correctedHtml;
            if (!originalPayload || !correctedPayload) {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                return null;
            }
            const baseKey = cacheBaseKey || computeAgentCacheBaseKey();
            if (baseKey) {
                lastAgentCacheBaseKey = baseKey;
            }
            agentDiffRequestToken += 1;
            const requestId = agentDiffRequestToken;
            const payload = await fetchAgentDiff(originalPayload, correctedPayload, agentDiffMode);
            if (requestId !== agentDiffRequestToken) {
                return payload;
            }
            const diffData = payload && payload.diff ? payload.diff : null;
            if (diffData) {
                renderAgentDiff(diffData, agentDiffMode);
            } else {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
            }
            if (baseKey && diffData) {
                const cacheKey = `${baseKey}:${agentDiffMode}`;
                agentContexts.correctors.agentResultCache.set(cacheKey, {
                    correctedContent: correctedHtml,
                    correctedIsHtml,
                    diff: diffData,
                    mode: agentDiffMode,
                    originalDocumentJson: lastAgentOriginalDocumentJson,
                    correctedDocumentJson: lastAgentCorrectedDocumentJson,
                });
            }
            return payload;
        }

        async function refreshAgentDiff() {
            updateAgentDiffToggleState();
            if (!lastAgentOriginalHtml || !lastAgentCorrectedHtml) {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                return;
            }
            const baseKey = computeAgentCacheBaseKey();
            if (baseKey) {
                const cacheKey = `${baseKey}:${agentDiffMode}`;
                const cached = agentContexts.correctors.agentResultCache.get(cacheKey);
                if (cached && Object.prototype.hasOwnProperty.call(cached, 'diff')) {
                    if (cached.originalDocumentJson) {
                        lastAgentOriginalDocumentJson = cached.originalDocumentJson;
                    }
                    if (cached.correctedDocumentJson) {
                        lastAgentCorrectedDocumentJson = cached.correctedDocumentJson;
                    }
                    if (cached.diff) {
                        renderAgentDiff(cached.diff, cached.mode || agentDiffMode);
                        return;
                    }
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                    return;
                }
            }
            renderAgentDiff(null, agentDiffMode, { hidden: true });
            await requestAgentDiff(lastAgentOriginalHtml, lastAgentCorrectedHtml, lastAgentCorrectedIsHtml, baseKey);
        }

        function setAgentDiffMode(newMode) {
            const normalized = newMode === AGENT_DIFF_MODES.CHAR ? AGENT_DIFF_MODES.CHAR : AGENT_DIFF_MODES.WORD;
            if (agentDiffMode === normalized) {
                updateAgentDiffToggleState();
                return;
            }
            agentDiffMode = normalized;
            persistAgentDiffMode(agentDiffMode);
            updateAgentDiffToggleState();
            refreshAgentDiff().catch((error) => {
                console.warn('Agent diff přepočet selhal:', error);
            });
        }

        function initializeAgentDiffControls() {
            agentDiffMode = loadStoredAgentDiffMode();
            const container = document.getElementById('agent-diff-mode-controls');
            if (container) {
                container.addEventListener('click', (event) => {
                    const target = event.target;
                    if (!target || !(target instanceof HTMLElement)) {
                        return;
                    }
                    if (!target.matches('.agent-diff-toggle')) {
                        return;
                    }
                    const requestedMode = target.getAttribute('data-diff-mode');
                    if (!requestedMode) {
                        return;
                    }
                    setAgentDiffMode(requestedMode);
                });
            }
            updateAgentDiffToggleState();
        }

        function safeSetAgentOutput(ctx, primary, secondary, status) {
            if (!ctx || !ctx.hasOutputPanel) {
                return;
            }
            setContextAgentOutput(ctx, primary, secondary, status);
        }

        async function runSelectedAgent(ctx, options = {}) {
            ctx.autoRunScheduled = false;
            const select = getContextElement(ctx, 'selectId');
            if (!select) return;
            const name = select.value;
            if (!name) {
                alert('Žádný agent není vybrán');
                return;
            }
            const isAutoInvocation = Boolean(options && options.autoTriggered);
            const pythonHtml = currentResults && currentResults.python ? String(currentResults.python) : '';
            const hasInputData = contextHasInputData(ctx);
            if (!hasInputData) {
                alert(ctx.requiresPythonHtml !== false
                    ? 'Python výstup je prázdný – nejprve načtěte stránku.'
                    : 'Nejprve načtěte stránku se skenem.');
                return;
            }
            const autoCheckbox = getContextElement(ctx, 'autoCheckboxId');
            const willAuto = autoCheckbox && autoCheckbox.checked;
            persistContextAutoTrigger(ctx, Boolean(willAuto));
            const originalPythonHtml = pythonHtml;
            const runBtn = getContextElement(ctx, 'runButtonId');
            const extraRunButtons = (ctx.runButtonExtraIds || [])
                .map((extraId) => document.getElementById(extraId))
                .filter((btn) => btn);
            const extraRunButtonStates = extraRunButtons.map((btn) => ({
                element: btn,
                originalLabel: btn.textContent || '',
            }));

            let workingDraft = null;
            if (ctx.currentAgentDraft && ctx.currentAgentName === name) {
                workingDraft = ctx.currentAgentDraft;
            } else if (ctx.agentsCache[name]) {
                workingDraft = deepCloneAgent(ctx.agentsCache[name]);
                ctx.currentAgentDraft = deepCloneAgent(workingDraft);
                ctx.currentAgentName = name;
            } else {
                workingDraft = normalizeAgentData({
                    name,
                    prompt: DEFAULT_AGENT_PROMPT,
                    model: DEFAULT_AGENT_MODEL,
                });
                ctx.currentAgentDraft = deepCloneAgent(workingDraft);
                ctx.currentAgentName = name;
            }
            ensureAgentDraftStructure(workingDraft);
            const normalizedModel = (workingDraft.model && String(workingDraft.model).trim()) || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(workingDraft, normalizedModel);
            const capabilities = getModelCapabilities(normalizedModel);
            const effectiveSettings = getEffectiveModelSettings(workingDraft, normalizedModel);

            const payload = {
                name,
                auto_correct: Boolean(willAuto),
                language_hint: DEFAULT_LANGUAGE_HINT,
                page_uuid: currentPage && currentPage.uuid ? currentPage.uuid : '',
                book_uuid: currentBook && currentBook.uuid ? currentBook.uuid : '',
                page_number: currentPage && currentPage.pageNumber ? currentPage.pageNumber : '',
                page_index: currentPage && typeof currentPage.index === 'number' ? currentPage.index : null,
                model: normalizedModel,
                model_override: normalizedModel,
            };
            const resolvedLibrary = currentLibrary
                || (currentPage && currentPage.library)
                || (currentBook && currentBook.library)
                || null;
            if (resolvedLibrary && resolvedLibrary.api_base) {
                payload.api_base = resolvedLibrary.api_base;
            }
            if (capabilities.temperature && Object.prototype.hasOwnProperty.call(effectiveSettings, 'temperature')) {
                payload.temperature = effectiveSettings.temperature;
            }
            if (capabilities.top_p && Object.prototype.hasOwnProperty.call(effectiveSettings, 'top_p')) {
                payload.top_p = effectiveSettings.top_p;
            }
            if (capabilities.reasoning && Object.prototype.hasOwnProperty.call(effectiveSettings, 'reasoning_effort')) {
                payload.reasoning_effort = effectiveSettings.reasoning_effort;
            }
            if (ENABLE_RESPONSE_FORMAT && capabilities.response_format && Object.prototype.hasOwnProperty.call(effectiveSettings, 'response_format')) {
                try {
                    payload.response_format = JSON.parse(JSON.stringify(effectiveSettings.response_format));
                } catch (err) {
                    payload.response_format = effectiveSettings.response_format;
                }
            }
            payload.agent_snapshot = buildAgentSavePayload(ctx, workingDraft);

            const originalLabel = runBtn ? runBtn.textContent : '';
            const restoreRunButtons = () => {
                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.textContent = originalLabel || ctx.runButtonLabel;
                }
                extraRunButtonStates.forEach(({ element, originalLabel: label }) => {
                    element.disabled = false;
                    element.textContent = label;
                });
            };

            if (ctx.collection === 'joiners') {
                const stitchContext = await refreshStitchUI({ reveal: !isAutoInvocation, keepVisible: isAutoInvocation, keepMerged: isAutoInvocation });
                if (!stitchContext) {
                    safeSetAgentOutput(ctx, 'Nepodařilo se připravit data pro napojení stran.', '', 'error');
                    restoreRunButtons();
                    return;
                }
                const stitchPayload = buildJoinerAgentPayloadFromContext(stitchContext);
                if (!stitchPayload) {
                    safeSetAgentOutput(ctx, 'Nepodařilo se připravit podklady pro agenta.', '', 'error');
                    restoreRunButtons();
                    return;
                }
                if (workingDraft.manual) {
                    const manualStart = performance.now();
                    const decisions = runManualJoiner(ctx, stitchContext) || {};
                    const elapsedMs = performance.now() - manualStart;
                    const elapsedLabel = `${(elapsedMs / 1000).toFixed(2)} s`;
                    safeSetAgentOutput(ctx, `Hotovo za ${elapsedLabel}`, '', 'success');
                    restoreRunButtons();
                    return;
                }
                payload.stitch_context = stitchPayload;
            } else if (ctx.collection === 'readers') {
                payload.scan_uuid = currentPage && currentPage.uuid ? currentPage.uuid : '';
                payload.page_title = currentPage && currentPage.title ? currentPage.title : '';
                payload.book_title = currentBook && currentBook.title ? currentBook.title : '';
                const previewHost = document.getElementById('page-preview');
                const currentStream = previewHost && previewHost.dataset && previewHost.dataset.previewStream
                    ? previewHost.dataset.previewStream
                    : 'IMG_FULL';
                payload.scan_stream = currentStream;
            } else {
                payload.python_html = pythonHtml;
            }

            cacheAgentFingerprint(ctx, name, workingDraft);
            const agentFingerprint = computeAgentFingerprint(ctx, name);
            const pythonHash = computeSimpleHash(pythonHtml);
            const cacheBaseKey = agentFingerprint ? `${ctx.collection}:${agentFingerprint}:${pythonHash}` : null;
            const cacheKey = ctx.supportsDiff && cacheBaseKey ? `${cacheBaseKey}:${agentDiffMode}` : null;
            if (ctx.supportsDiff) {
                lastAgentCacheBaseKey = cacheBaseKey;
                lastAgentOriginalHtml = originalPythonHtml;
                lastAgentCorrectedHtml = '';
                lastAgentCorrectedIsHtml = false;
                lastAgentOriginalDocumentJson = '';
                lastAgentCorrectedDocumentJson = '';
            }
            const cachedResult = cacheKey ? ctx.agentResultCache.get(cacheKey) : null;

            if (ctx.supportsDiff && cachedResult && isAutoInvocation) {
                if (cachedResult.originalDocumentJson) {
                    lastAgentOriginalDocumentJson = cachedResult.originalDocumentJson;
                }
                if (cachedResult.correctedDocumentJson) {
                    lastAgentCorrectedDocumentJson = cachedResult.correctedDocumentJson;
                }
                setAgentResultPanels(originalPythonHtml, cachedResult.correctedContent, cachedResult.correctedIsHtml);
                lastAgentCacheBaseKey = cacheBaseKey;
                if (cachedResult.diff) {
                    renderAgentDiff(cachedResult.diff, cachedResult.mode || agentDiffMode);
                } else if (lastAgentOriginalHtml && lastAgentCorrectedHtml) {
                    await refreshAgentDiff();
                } else {
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                }
                console.info(`[AgentDebug] Použit cache výsledek pro agenta ${name}`);
                safeSetAgentOutput(ctx, '', '', 'success');
                return;
            }

            let startTime = null;
            try {
                if (runBtn) {
                    runBtn.disabled = true;
                    runBtn.textContent = 'Spouštím...';
                }
                extraRunButtonStates.forEach(({ element }) => {
                    element.disabled = true;
                    element.textContent = 'Spouštím...';
                });
                startTime = performance.now();
                safeSetAgentOutput(ctx, `Spouštím agenta ${name}...`, '', 'pending');
                const agentConfig = workingDraft || {};
                console.groupCollapsed(`[AgentDebug] Request → ${name}`);
                console.log('Agent config:', agentConfig);
                console.log('Payload:', payload);
                console.groupEnd();
                const response = await fetch('/agents/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(buildAgentPayload(ctx, payload)),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                const result = data.result || {};
                const text = typeof result.text === 'string' ? result.text.trim() : '';
                const statusParts = [];
                console.groupCollapsed(`[AgentDebug] Response → ${name}`);
                console.log('Výsledek:', result);
                console.groupEnd();

                if (ctx.supportsDiff) {
                    const parsedDoc = parseAgentResultDocument(text);
                    const originalDocument = result && typeof result.input_document === 'object' ? result.input_document : null;
                    if (originalDocument) {
                        try {
                            lastAgentOriginalDocumentJson = JSON.stringify(originalDocument);
                        } catch (jsonError) {
                            lastAgentOriginalDocumentJson = '';
                        }
                    } else {
                        lastAgentOriginalDocumentJson = '';
                    }
                    if (parsedDoc) {
                        try {
                            lastAgentCorrectedDocumentJson = JSON.stringify(parsedDoc);
                        } catch (jsonError) {
                            lastAgentCorrectedDocumentJson = '';
                        }
                    } else {
                        lastAgentCorrectedDocumentJson = '';
                    }
                    const htmlFromDoc = parsedDoc ? documentBlocksToHtml(parsedDoc) : null;
                    const autoRequested = Boolean(data.auto_correct);
                    if (autoRequested && !htmlFromDoc) {
                        statusParts.push('Výsledek nelze aplikovat (očekáván JSON se strukturou blocks)');
                    }

                    const correctedContent = htmlFromDoc || (text || '');
                    const correctedIsHtml = Boolean(htmlFromDoc);
                    setAgentResultPanels(originalPythonHtml, correctedContent, correctedIsHtml);

                    if (cacheKey) {
                        ctx.agentResultCache.set(cacheKey, {
                            correctedContent,
                            correctedIsHtml,
                            mode: agentDiffMode,
                            originalDocumentJson: lastAgentOriginalDocumentJson,
                            correctedDocumentJson: lastAgentCorrectedDocumentJson,
                        });
                    }

                    if (correctedContent && originalPythonHtml) {
                        await requestAgentDiff(originalPythonHtml, correctedContent, correctedIsHtml, cacheBaseKey);
                    } else {
                        renderAgentDiff(null, agentDiffMode, { hidden: true });
                    }
                } else if (ctx.collection === 'joiners') {
                    const applied = applyJoinerAgentResult(ctx, text);
                    if (!applied) {
                        ctx.lastJoinerResult = {
                            result,
                            rawText: text,
                            timestamp: Date.now(),
                        };
                    }
                } else if (ctx.collection === 'readers') {
                    await applyReaderAgentResult(ctx, result, text);
                }

                const elapsedMs = performance.now() - startTime;
                const elapsedLabel = `${(elapsedMs / 1000).toFixed(2)} s`;

                if (statusParts.length) {
                    const statusText = statusParts.join(' · ');
                    if (statusText.toLowerCase().includes('nelze')) {
                        safeSetAgentOutput(ctx, statusText, `Hotovo za ${elapsedLabel}`, 'error');
                    } else {
                        safeSetAgentOutput(ctx, `Hotovo za ${elapsedLabel}`, statusText, 'success');
                    }
                } else if (ctx.hasOutputPanel) {
                    safeSetAgentOutput(ctx, `Hotovo za ${elapsedLabel}`, '', 'success');
                }
            } catch (err) {
                const elapsedMs = startTime !== null && typeof performance !== 'undefined'
                    ? performance.now() - startTime
                    : 0;
                const elapsedLabel = elapsedMs ? ` · ${(elapsedMs / 1000).toFixed(2)} s` : '';
                const message = err && err.message ? err.message : String(err);
                if (ctx.supportsDiff) {
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                }
                safeSetAgentOutput(ctx, `Chyba agenta: ${message}${elapsedLabel}`, '', 'error');
                console.groupCollapsed(`[AgentDebug] Chyba → ${name}`);
                console.error(err);
                console.groupEnd();
            } finally {
                restoreRunButtons();
            }
        }

        function contextHasInputData(ctx) {
            if (!ctx) {
                return false;
            }
            if (ctx.requiresPythonHtml === false) {
                return Boolean(currentPage && currentPage.uuid);
            }
            return Boolean(currentResults && typeof currentResults.python === 'string' && currentResults.python.trim().length > 0);
        }

        function scheduleAutoAgentRun(ctx, attemptsRemaining = 5) {
            const autoCheckbox = getContextElement(ctx, 'autoCheckboxId');
            if (!autoCheckbox || !autoCheckbox.checked) {
                ctx.autoRunScheduled = false;
                return;
            }
            const runButton = getContextElement(ctx, 'runButtonId');
            const select = getContextElement(ctx, 'selectId');
            const hasAgent = Boolean(select && select.value);
            const hasInputs = contextHasInputData(ctx);
            if ((runButton && runButton.disabled) || !hasAgent || !hasInputs) {
                if (attemptsRemaining > 0 && autoCheckbox.checked) {
                    window.setTimeout(() => scheduleAutoAgentRun(ctx, attemptsRemaining - 1), 200);
                } else {
                    ctx.autoRunScheduled = false;
                }
                return;
            }
            runSelectedAgent(ctx, { autoTriggered: true }).catch((error) => {
                ctx.autoRunScheduled = false;
                console.warn('Automatické spuštění agenta se nezdařilo:', error);
            });
        }


        async function initializeAgentUI(ctx) {
            const select = getContextElement(ctx, 'selectId');
            const runBtn = getContextElement(ctx, 'runButtonId');
            const extraRunButtons = (ctx.runButtonExtraIds || [])
                .map((extraId) => document.getElementById(extraId))
                .filter((btn) => btn);
            const auto = getContextElement(ctx, 'autoCheckboxId');
            const toggle = getContextElement(ctx, 'expandToggleId');
            const settings = getContextElement(ctx, 'settingsId');
            const saveBtn = getContextElement(ctx, 'saveButtonId');
            const deleteBtn = getContextElement(ctx, 'deleteButtonId');
            const modelSelect = getContextElement(ctx, 'modelSelectId');
            const nameInput = getContextElement(ctx, 'nameInputId');
            const promptTextarea = getContextElement(ctx, 'promptTextareaId');
            preparePromptTextarea(promptTextarea);
            enableScrollPassthrough(select);
            enableScrollPassthrough(modelSelect);

            if (select) select.addEventListener('change', () => updateAgentUIFromSelection(ctx));
            const handleRunClick = async () => {
                try {
                    await runSelectedAgent(ctx, { autoTriggered: false });
                } catch (error) {
                    console.warn('Spuštění agenta se nezdařilo:', error);
                }
            };
            if (runBtn) {
                runBtn.addEventListener('click', handleRunClick);
            }
            extraRunButtons.forEach((btn) => {
                btn.addEventListener('click', handleRunClick);
            });
            if (auto) {
                auto.checked = loadContextAutoTrigger(ctx);
                auto.addEventListener('change', () => {
                    persistContextAutoTrigger(ctx, auto.checked);
                    if (auto.checked) {
                        ctx.autoRunScheduled = false;
                        const hasAgent = Boolean(select && select.value);
                        const hasInputs = contextHasInputData(ctx);
                        if (hasAgent && hasInputs && !(runBtn && runBtn.disabled)) {
                            runSelectedAgent(ctx, { autoTriggered: true }).catch((error) => {
                                console.warn('Automatické spuštění agenta po změně zaškrtávacího políčka selhalo:', error);
                            });
                        }
                    } else {
                        ctx.autoRunScheduled = false;
                    }
                });
            }
            if (toggle && settings) {
                toggle.addEventListener('click', async () => {
                    const visible = settings.style.display !== 'none';
                    const nowVisible = !visible;
                    settings.style.display = visible ? 'none' : 'block';
                    toggle.setAttribute('aria-expanded', (nowVisible).toString());
                    if (nowVisible) {
                        try { await updateAgentUIFromSelection(ctx); } catch (e) { /* ignore */ }
                        const promptField = getContextElement(ctx, 'promptTextareaId');
                        if (promptField) {
                            preparePromptTextarea(promptField);
                            requestAnimationFrame(() => autoResizeTextarea(promptField));
                        }
                    }
                    scheduleThumbnailDrawerHeightSync();
                    setTimeout(() => scheduleThumbnailDrawerHeightSync(), 80);
                });
            }
            if (saveBtn) saveBtn.addEventListener('click', async () => { await saveCurrentAgent(ctx); scheduleThumbnailDrawerHeightSync(); });
            if (deleteBtn) deleteBtn.addEventListener('click', async () => { await deleteCurrentAgent(ctx); scheduleThumbnailDrawerHeightSync(); });

            if (nameInput) {
                nameInput.addEventListener('input', () => {
                    if (!ctx.currentAgentDraft) {
                        return;
                    }
                    ctx.currentAgentDraft.name = String(nameInput.value || '').trim();
                    markAgentDirty(ctx);
                });
            }

            if (promptTextarea) {
                promptTextarea.addEventListener('input', () => {
                    if (!ctx.currentAgentDraft) {
                        return;
                    }
                    ctx.currentAgentDraft.prompt = promptTextarea.value;
                    markAgentDirty(ctx);
                    promptTextarea.dataset.needsResize = 'true';
                    autoResizeTextarea(promptTextarea);
                });
            }

            if (modelSelect) {
                modelSelect.addEventListener('change', () => {
                    if (!ctx.currentAgentDraft) {
                        return;
                    }
                    const selectedModel = String(modelSelect.value || '').trim() || DEFAULT_AGENT_MODEL;
                    ctx.currentAgentDraft.model = selectedModel;
                    ensureAgentDraftModelSettings(ctx.currentAgentDraft, selectedModel);
                    renderAgentParameterControls(ctx, selectedModel);
                    markAgentDirty(ctx);
                    scheduleThumbnailDrawerHeightSync();
                });
            }

            // Load selector asynchronously; don't block UI interactivity
            renderAgentSelector(ctx).catch(() => {});
        }

        function initializeStitchingUI() {
            if (!stitchScaleResizeBound) {
                window.addEventListener('resize', () => {
                    updateAllStitchScales();
                });
                stitchScaleResizeBound = true;
            }
            refreshStitchUI().then(() => {
                updateAllStitchScales();
            }).catch((error) => {
                console.warn('Nepodařilo se inicializovat sekci napojení stran:', error);
            });
        }
        function createThumbnailLoadManager(concurrency = 6) {
            const queue = [];
            let active = 0;

            function insertTask(task) {
                if (!queue.length) {
                    queue.push(task);
                    return;
                }
                const index = queue.findIndex(item => item.priority > task.priority || (item.priority === task.priority && item.created > task.created));
                if (index === -1) {
                    queue.push(task);
                } else {
                    queue.splice(index, 0, task);
                }
            }

            function startTask(task) {
                const img = task && task.img;
                if (!img || !img.dataset || !img.dataset.src) {
                    if (img && img.dataset) {
                        delete img.dataset.queueId;
                        delete img.dataset.loading;
                    }
                    if (active > 0) {
                        active -= 1;
                    }
                    schedule();
                    return;
                }

                delete img.dataset.queueId;
                if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                    if (active > 0) {
                        active -= 1;
                    }
                    schedule();
                    return;
                }

                active += 1;
                img.dataset.loading = 'true';
                img.src = img.dataset.src;
                if (img.complete) {
                    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                        img.dispatchEvent(new Event('load'));
                    } else if (img.naturalWidth === 0) {
                        img.dispatchEvent(new Event('error'));
                    }
                }
            }

            function schedule() {
                while (active < concurrency && queue.length) {
                    const task = queue.shift();
                    if (!task) {
                        continue;
                    }
                    startTask(task);
                }
            }

            function enqueue(img, priority = 1) {
                if (!img || !img.dataset || !img.dataset.src) {
                    return;
                }
                if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                    return;
                }

                const normalizedPriority = Number.isFinite(priority) ? priority : 1;

                if (normalizedPriority <= 0 && active < concurrency) {
                    startTask({ img, priority: normalizedPriority, created: performance.now() });
                    return;
                }

                if (img.dataset.queueId) {
                    return;
                }

                const task = {
                    img,
                    priority: normalizedPriority,
                    created: performance.now(),
                };
                img.dataset.queueId = String(task.created);
                insertTask(task);
                schedule();
            }

            function markComplete(img, success) {
                if (img && img.dataset) {
                    delete img.dataset.loading;
                    delete img.dataset.queueId;
                    if (success) {
                        img.dataset.loaded = 'true';
                    } else {
                        delete img.dataset.loaded;
                    }
                }
                if (active > 0) {
                    active -= 1;
                }
                schedule();
            }

            function reset() {
                while (queue.length) {
                    const task = queue.shift();
                    if (task && task.img && task.img.dataset) {
                        delete task.img.dataset.queueId;
                        delete task.img.dataset.loading;
                    }
                }
                active = 0;
            }

            return {
                enqueue,
                markComplete,
                reset,
            };
        }

        const thumbnailLoader = createThumbnailLoadManager(6);

        function finalizeThumbnail(img, success) {
            thumbnailLoader.markComplete(img, success);
        }

        function cacheProcessData(uuid, payload) {
            if (!uuid || !payload) {
                return;
            }
            pageCache.set(uuid, {
                payload,
                timestamp: Date.now(),
            });
        }

        async function ensureProcessData(uuid) {
            if (!uuid) {
                return null;
            }
            const cached = pageCache.get(uuid);
            if (cached) {
                cached.timestamp = Date.now();
                return cached.payload;
            }
            if (inflightProcessRequests.has(uuid)) {
                return inflightProcessRequests.get(uuid);
            }

            const promise = (async () => {
                // When available, forward the currently selected library api_base so the server
                // queries the same Kramerius instance the UI is showing. This prevents
                // thumbnail clicks from switching the source library unexpectedly.
                let processUrl = `/process?uuid=${encodeURIComponent(uuid)}`;
                try {
                    if (currentLibrary && currentLibrary.api_base) {
                        processUrl += `&api_base=${encodeURIComponent(currentLibrary.api_base)}`;
                    }
                } catch (err) {
                    // ignore and fall back to default
                }
                const response = await fetch(processUrl, { cache: "no-store" });
                const data = await response.json();
                if (!response.ok || data.error) {
                    const message = data && data.error ? data.error : response.statusText || `HTTP ${response.status}`;
                    throw new Error(message);
                }
                cacheProcessData(uuid, data);
                return data;
            })().finally(() => {
                inflightProcessRequests.delete(uuid);
            });

            inflightProcessRequests.set(uuid, promise);
            return promise;
        }

        function releasePreviewEntry(entry) {
            if (entry && entry.objectUrl && entry.objectUrl !== previewObjectUrl) {
                URL.revokeObjectURL(entry.objectUrl);
            }
        }

        function storePreviewEntry(uuid, result) {
            if (!uuid || !result) {
                return null;
            }
            const existing = previewCache.get(uuid);
            if (existing) {
                releasePreviewEntry(existing);
            }
            const objectUrl = result.objectUrl || URL.createObjectURL(result.blob);
            const payload = {
                blob: result.blob,
                stream: result.stream,
                contentType: result.contentType,
                objectUrl,
                timestamp: Date.now(),
            };
            previewCache.set(uuid, payload);
            return payload;
        }

        async function ensurePreviewEntry(uuid) {
            if (!uuid) {
                return null;
            }
            const cached = previewCache.get(uuid);
            if (cached) {
                cached.timestamp = Date.now();
                return cached;
            }
            if (inflightPreviewRequests.has(uuid)) {
                return inflightPreviewRequests.get(uuid);
            }

            const promise = (async () => {
                const result = await fetchPreviewImage(uuid);
                return storePreviewEntry(uuid, result);
            })().finally(() => {
                inflightPreviewRequests.delete(uuid);
            });

            inflightPreviewRequests.set(uuid, promise);
            return promise;
        }

        function computeCacheWindow(currentUuid, navigation) {
            const target = new Set();
            if (currentUuid) {
                target.add(currentUuid);
            }
            if (navigation) {
                if (navigation.prevUuid) {
                    target.add(navigation.prevUuid);
                }
                if (navigation.nextUuid) {
                    target.add(navigation.nextUuid);
                }
            }
            return target;
        }

        function updateCacheWindow(currentUuid, navigation) {
            cacheWindowUuids = computeCacheWindow(currentUuid, navigation);

            for (const key of Array.from(pageCache.keys())) {
                if (!cacheWindowUuids.has(key)) {
                    pageCache.delete(key);
                }
            }

            for (const [key, entry] of Array.from(previewCache.entries())) {
                if (!cacheWindowUuids.has(key)) {
                    releasePreviewEntry(entry);
                    previewCache.delete(key);
                }
            }
        }

        function schedulePrefetch(navigation) {
            if (!navigation) {
                return;
            }
            const candidates = [navigation.prevUuid, navigation.nextUuid].filter(Boolean);
            candidates.forEach(uuid => {
                prefetchProcess(uuid);
                prefetchPreview(uuid);
            });
        }

        async function prefetchProcess(uuid) {
            if (!uuid || pageCache.has(uuid) || inflightProcessRequests.has(uuid)) {
                return;
            }
            try {
                await ensureProcessData(uuid);
            } catch (error) {
                console.warn("Nepodařilo se přednačíst stránku", uuid, error);
            }
        }

        async function prefetchPreview(uuid) {
            if (!uuid || previewCache.has(uuid) || inflightPreviewRequests.has(uuid)) {
                return;
            }
            try {
                await ensurePreviewEntry(uuid);
            } catch (error) {
                console.warn("Nepodařilo se přednačíst náhled", uuid, error);
            }
        }

        function setThumbnailDrawerCollapsed(collapsed) {
            const desiredState = Boolean(collapsed);
            thumbnailDrawerCollapsed = desiredState;
            document.body.classList.toggle('thumbnail-drawer-collapsed', thumbnailDrawerCollapsed);
            const toggle = document.getElementById('thumbnail-toggle');
            if (toggle) {
                toggle.textContent = thumbnailDrawerCollapsed ? '<' : '>';
                toggle.setAttribute('aria-expanded', (!thumbnailDrawerCollapsed).toString());
                toggle.setAttribute('aria-label', thumbnailDrawerCollapsed ? 'Zobrazit náhledy' : 'Skrýt náhledy');
            }
            const drawerPanel = document.querySelector('#thumbnail-drawer .thumbnail-panel');
            if (drawerPanel) {
                drawerPanel.setAttribute('aria-hidden', thumbnailDrawerCollapsed ? 'true' : 'false');
            }
            const drawer = document.getElementById('thumbnail-drawer');
            if (drawer) {
                drawer.setAttribute('aria-hidden', thumbnailDrawerCollapsed ? 'true' : 'false');
            }
            if (thumbnailDrawerCollapsed) {
                resetThumbnailObserver();
            } else {
                ensureThumbnailObserver();
            }
            if (!thumbnailDrawerCollapsed) {
                scheduleThumbnailDrawerHeightSync(true);
            }
            schedulePreviewDrawerPositionUpdate();
        }

        function setPreviewDrawerCollapsed(collapsed) {
            const desiredState = Boolean(collapsed);
            previewDrawerCollapsed = desiredState;
            const body = document.body;
            if (body) {
                body.classList.toggle('preview-drawer-collapsed', previewDrawerCollapsed);
            }
            const toggle = document.getElementById('preview-drawer-toggle');
            if (toggle) {
                toggle.textContent = previewDrawerCollapsed ? '>' : '<';
                toggle.setAttribute('aria-expanded', (!previewDrawerCollapsed).toString());
                toggle.setAttribute('aria-label', previewDrawerCollapsed ? 'Zobrazit pevný náhled' : 'Skrýt pevný náhled');
            }
            const drawer = document.getElementById('preview-drawer');
            if (drawer) {
                drawer.setAttribute('aria-hidden', previewDrawerCollapsed ? 'true' : 'false');
            }
            schedulePreviewDrawerPositionUpdate();
        }

        function updatePreviewDrawerPosition() {
            previewDrawerPositionFrame = null;
            const drawer = document.getElementById('preview-drawer');
            const toggle = document.getElementById('preview-drawer-toggle');
            if (!drawer || !toggle) {
                return;
            }
            const isMobileLayout = window.matchMedia('(max-width: 1100px)').matches;
            if (isMobileLayout) {
                drawer.style.left = '';
                drawer.style.top = '';
                drawer.style.right = '';
                toggle.style.left = '';
                toggle.style.top = '';
                toggle.style.right = '';
                return;
            }
            const container = document.querySelector('.container');
            const anchor = container || document.querySelector('.page-shell');
            if (!anchor) {
                return;
            }
            const rect = anchor.getBoundingClientRect();
            const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
            const drawerWidth = drawer.getBoundingClientRect().width || parseFloat(getComputedStyle(drawer).width) || 0;
            const gapValue = getComputedStyle(document.documentElement).getPropertyValue('--preview-drawer-gap');
            const requestedGap = Number.parseFloat(gapValue);
            const baseGap = Number.isFinite(requestedGap) ? Math.max(0, requestedGap) : 0;
            const viewportPadding = Math.max(baseGap, 12);
            let drawerLeft = rect.right + baseGap;
            if (drawerWidth && viewportWidth) {
                const maxLeft = Math.max(0, viewportWidth - drawerWidth - viewportPadding);
                drawerLeft = Math.min(drawerLeft, maxLeft);
            }
            const drawerTop = Math.max(20, rect.top);
            drawer.style.left = `${drawerLeft}px`;
            drawer.style.top = `${drawerTop}px`;
            drawer.style.right = 'auto';

            const toggleWidth = toggle.getBoundingClientRect().width || parseFloat(getComputedStyle(toggle).width) || 0;
            const toggleGap = Math.max(8, baseGap / 2);
            let toggleLeft = rect.right + toggleGap;
            if (toggleWidth && viewportWidth) {
                const maxToggleLeft = viewportWidth - toggleWidth - viewportPadding;
                toggleLeft = Math.min(toggleLeft, maxToggleLeft);
            }
            const toggleTop = Math.max(38, rect.top + 18);
            toggle.style.left = `${toggleLeft}px`;
            toggle.style.top = `${toggleTop}px`;
            toggle.style.right = 'auto';
        }

        function schedulePreviewDrawerPositionUpdate() {
            if (previewDrawerPositionFrame) {
                return;
            }
            previewDrawerPositionFrame = window.requestAnimationFrame(() => {
                updatePreviewDrawerPosition();
            });
        }

        function initializePreviewDrawer() {
            const drawer = document.getElementById('preview-drawer');
            const toggle = document.getElementById('preview-drawer-toggle');
            if (!drawer || !toggle) {
                previewDrawerCollapsed = true;
                return;
            }
            toggle.addEventListener('click', () => {
                setPreviewDrawerCollapsed(!previewDrawerCollapsed);
            });
            window.addEventListener('resize', schedulePreviewDrawerPositionUpdate);
            window.addEventListener('scroll', schedulePreviewDrawerPositionUpdate, { passive: true });
            setPreviewDrawerCollapsed(false);
            schedulePreviewDrawerPositionUpdate();
        }

        function navigateToUuid(targetUuid) {
            if (!targetUuid) {
                return;
            }
            const uuidField = document.getElementById("uuid");
            if (uuidField) {
                uuidField.value = targetUuid;
                setUuidButtonsState();
            }
            processAlto();
        }

        function clearThumbnailGrid(message) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid) {
                return;
            }
            resetThumbnailObserver();
            grid.innerHTML = "";
            if (message) {
                const placeholder = document.createElement("div");
                placeholder.className = "thumbnail-empty";
                placeholder.textContent = message;
                grid.appendChild(placeholder);
            }
        }

        function normalizePages(pages) {
            if (!Array.isArray(pages)) {
                return [];
            }
            return pages.map((entry, idx) => {
                const normalized = Object.assign({}, entry || {});
                if (typeof normalized.index !== "number" || !Number.isFinite(normalized.index)) {
                    normalized.index = idx;
                }
                normalized.uuid = normalized.uuid || "";
                return normalized;
            });
        }

        function updateThumbnailLabels(pages) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid || !pages.length) {
                return;
            }
            const buttons = grid.querySelectorAll('.page-thumbnail');
            buttons.forEach(button => {
                const listIndex = Number.parseInt(button.dataset.listIndex || "-1", 10);
                const page = Number.isFinite(listIndex) && listIndex >= 0 && listIndex < pages.length ? pages[listIndex] : null;
                const pageNumber = page && page.pageNumber ? page.pageNumber : "";
                const displayIndex = page && typeof page.index === "number" && Number.isFinite(page.index) ? page.index : listIndex;
                const labelText = pageNumber ? `Strana ${pageNumber}` : `Strana ${displayIndex + 1}`;

                button.setAttribute('aria-label', labelText);

                const thumbImage = button.querySelector('img');
                if (thumbImage) {
                    thumbImage.alt = labelText;
                }

                let labelEl = button.querySelector('.page-thumbnail-label');
                if (pageNumber) {
                    if (!labelEl) {
                        labelEl = document.createElement('span');
                        labelEl.className = 'page-thumbnail-label';
                        button.appendChild(labelEl);
                    }
                    labelEl.textContent = pageNumber;
                } else if (labelEl) {
                    labelEl.remove();
                }
            });
        }

        function highlightActiveThumbnail(uuid) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid) {
                lastActiveThumbnailUuid = null;
                return;
            }

            let activeButton = null;
            grid.querySelectorAll('.page-thumbnail').forEach(button => {
                if (uuid && button.dataset.uuid === uuid) {
                    button.classList.add('is-active');
                    activeButton = button;
                } else {
                    button.classList.remove('is-active');
                }
            });

            const shouldScroll = uuid && uuid !== lastActiveThumbnailUuid && !thumbnailDrawerCollapsed;
            if (activeButton && shouldScroll) {
                const scrollContainer = document.getElementById('thumbnail-scroll');
                if (scrollContainer) {
                    const containerRect = scrollContainer.getBoundingClientRect();
                    const buttonRect = activeButton.getBoundingClientRect();
                    if (buttonRect.top < containerRect.top || buttonRect.bottom > containerRect.bottom) {
                        activeButton.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
                    }
                } else {
                    activeButton.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
                }
            }

            lastActiveThumbnailUuid = uuid || null;
        }

        function scheduleThumbnailDrawerHeightSync(immediate = false) {
            if (immediate) {
                thumbnailHeightSyncPending = false;
                try {
                    syncThumbnailDrawerHeight();
                } catch (error) {
                    console.warn('Thumbnail height sync failed:', error);
                }
                return;
            }
            if (thumbnailHeightSyncPending) {
                return;
            }
            thumbnailHeightSyncPending = true;
            requestAnimationFrame(() => {
                thumbnailHeightSyncPending = false;
                try {
                    syncThumbnailDrawerHeight();
                } catch (error) {
                    console.warn('Thumbnail height sync failed:', error);
                }
            });
        }

        function syncThumbnailDrawerHeight() {
            const drawer = document.getElementById('thumbnail-drawer');
            const panel = drawer ? drawer.querySelector('.thumbnail-panel') : null;
            const container = document.querySelector('.container');
            if (!drawer || !container) {
                return;
            }
            const rect = container.getBoundingClientRect();
            const containerHeight = rect && Number.isFinite(rect.height) ? rect.height : container.offsetHeight;
            if (!Number.isFinite(containerHeight) || containerHeight <= 0) {
                return;
            }

            let targetHeight = containerHeight;

            if (panel) {
                const styles = window.getComputedStyle(panel);
                const paddingTop = parseFloat(styles.paddingTop) || 0;
                const paddingBottom = parseFloat(styles.paddingBottom) || 0;
                const borderTop = parseFloat(styles.borderTopWidth) || 0;
                const borderBottom = parseFloat(styles.borderBottomWidth) || 0;
                const chromeHeight = paddingTop + paddingBottom + borderTop + borderBottom;

                const previousHeight = panel.style.height;
                const previousMaxHeight = panel.style.maxHeight;
                panel.style.height = 'auto';
                panel.style.maxHeight = 'none';
                const naturalContentHeight = panel.scrollHeight || 0;
                const naturalPanelHeight = naturalContentHeight + chromeHeight;

                targetHeight = Math.min(containerHeight, naturalPanelHeight || containerHeight);
                const innerHeight = Math.max(targetHeight - chromeHeight, 0);

                panel.style.height = `${innerHeight}px`;
                panel.style.maxHeight = `${innerHeight}px`;

                if (!innerHeight && (previousHeight || previousMaxHeight)) {
                    panel.style.height = previousHeight;
                    panel.style.maxHeight = previousMaxHeight;
                }
            }

            drawer.style.height = `${targetHeight}px`;
        }

        function resetThumbnailObserver() {
            if (thumbnailObserver) {
                thumbnailObserver.disconnect();
                thumbnailObserver = null;
            }
            thumbnailLoader.reset();
        }

        function ensureThumbnailObserver() {
            if (thumbnailObserver) {
                return thumbnailObserver;
            }
            if (!('IntersectionObserver' in window)) {
                return null;
            }
            const scrollContainer = document.getElementById('thumbnail-scroll');
            if (!scrollContainer) {
                return null;
            }
            thumbnailObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        loadThumbnailImage(img, 1);
                        if (thumbnailObserver) {
                            thumbnailObserver.unobserve(img);
                        }
                    }
                });
            }, {
                root: scrollContainer,
                rootMargin: '200px 0px',
                threshold: 0.05,
            });
            const grid = document.getElementById('thumbnail-grid');
            if (grid) {
                grid.querySelectorAll('img[data-src]').forEach(img => {
                    if (!img.dataset.loaded || img.dataset.loaded !== 'true') {
                        thumbnailObserver.observe(img);
                    }
                });
            }
            return thumbnailObserver;
        }

        function loadThumbnailImage(img, priority = 1) {
            if (!img || !img.dataset || !img.dataset.src) {
                return;
            }
            if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                return;
            }
            let normalizedPriority;
            if (typeof priority === 'boolean') {
                normalizedPriority = priority ? 0 : 1;
            } else if (Number.isFinite(priority)) {
                normalizedPriority = priority;
            } else {
                normalizedPriority = 1;
            }
            thumbnailLoader.enqueue(img, normalizedPriority);
        }

        function updatePageNumberInput() {
            const input = document.getElementById('page-number-input');
            const totalLabel = document.getElementById('page-number-total');
            const total = bookPages.length || (navigationState && typeof navigationState.total === 'number' ? navigationState.total : 0);

            if (totalLabel) {
                totalLabel.textContent = total ? `/ ${total}` : "";
            }

            if (!input) {
                return;
            }

            if (currentPage && typeof currentPage.index === 'number' && Number.isFinite(currentPage.index) && currentPage.index >= 0) {
                input.value = String(currentPage.index + 1);
            } else if (total) {
                input.value = '1';
            } else {
                input.value = "";
            }

            input.disabled = total === 0;
        }

        function ensureThumbnailGrid(pages, bookUuid) {
            const grid = document.getElementById('thumbnail-grid');
            const scrollContainer = document.getElementById('thumbnail-scroll');

            const normalizedPages = normalizePages(pages);
            bookPages = normalizedPages;

            if (!grid) {
                lastRenderedBookUuid = null;
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            if (!normalizedPages.length) {
                lastRenderedBookUuid = null;
                clearThumbnailGrid('Náhledy nejsou k dispozici.');
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            const normalizedBookUuid = bookUuid || null;
            const isSameBook = normalizedBookUuid === lastRenderedBookUuid;
            const shouldRender = !isSameBook || grid.querySelectorAll('.page-thumbnail').length !== normalizedPages.length;

            if (shouldRender) {
                const previousScrollTop = isSameBook && scrollContainer ? scrollContainer.scrollTop : 0;
                resetThumbnailObserver();
                grid.innerHTML = "";

                const priorityIndices = new Set();
                for (let i = 0; i < 6 && i < normalizedPages.length; i += 1) {
                    priorityIndices.add(i);
                }
                if (currentPage && typeof currentPage.index === 'number' && Number.isFinite(currentPage.index)) {
                    const base = Math.max(Math.min(currentPage.index - 1, normalizedPages.length - 1), 0);
                    for (let i = base; i <= Math.min(base + 2, normalizedPages.length - 1); i += 1) {
                        priorityIndices.add(i);
                    }
                }

                const observer = ensureThumbnailObserver();

                normalizedPages.forEach((page, listIndex) => {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'page-thumbnail';
                    button.dataset.uuid = page.uuid || "";
                    button.dataset.index = String(page.index);
                    button.dataset.listIndex = String(listIndex);

                    const labelText = page.pageNumber ? `Strana ${page.pageNumber}` : `Strana ${page.index + 1}`;
                    button.setAttribute('aria-label', labelText);

                    if (page.uuid) {
                        button.addEventListener('click', () => navigateToUuid(page.uuid));
                    } else {
                        button.disabled = true;
                    }

                    const placeholder = document.createElement('div');
                    placeholder.className = 'thumbnail-placeholder';
                    button.appendChild(placeholder);

                    const img = document.createElement('img');
                    img.loading = 'lazy';
                    img.decoding = 'async';
                    img.alt = labelText;
                    const normalizedUuid = typeof page.uuid === 'string' ? page.uuid : "";
                    const providedThumb = typeof page.thumbnail === 'string' ? page.thumbnail : "";
                    const fallbackThumb = normalizedUuid ? `/preview?uuid=${encodeURIComponent(normalizedUuid)}&stream=IMG_THUMB` : "";
                    const thumbSrc = providedThumb || fallbackThumb;
                    if (thumbSrc) {
                        img.dataset.src = thumbSrc;
                    }
                    const markLoaded = () => {
                        button.classList.add('is-loaded');
                        img.dataset.loaded = 'true';
                    };
                    img.addEventListener('load', () => {
                        const success = img.naturalWidth > 0 && img.naturalHeight > 0;
                        if (success) {
                            markLoaded();
                        } else {
                            button.classList.remove('is-loaded');
                        }
                        finalizeThumbnail(img, success);
                    });
                    img.addEventListener('error', () => {
                        button.classList.remove('is-loaded');
                        finalizeThumbnail(img, false);
                    });
                    button.appendChild(img);

                    if (page.pageNumber) {
                        const badge = document.createElement('span');
                        badge.className = 'page-thumbnail-label';
                        badge.textContent = page.pageNumber;
                        button.appendChild(badge);
                    }

                    grid.appendChild(button);

                    if (img.dataset.src) {
                        if (priorityIndices.has(listIndex)) {
                            loadThumbnailImage(img, 0);
                        } else if (observer) {
                            observer.observe(img);
                        } else {
                            loadThumbnailImage(img, 1);
                        }
                    }
                });

                if (scrollContainer) {
                    scrollContainer.scrollTop = previousScrollTop;
                }

                lastRenderedBookUuid = normalizedBookUuid;
            } else {
                updateThumbnailLabels(normalizedPages);
            }

            highlightActiveThumbnail(currentPage ? currentPage.uuid : null);
            updatePageNumberInput();
            scheduleThumbnailDrawerHeightSync();
        }

        function handlePageNumberJump(rawValue) {
            const input = document.getElementById('page-number-input');
            if (!input || !bookPages.length) {
                updatePageNumberInput();
                return;
            }

            const value = rawValue !== undefined ? rawValue : input.value;
            const parsed = Number.parseInt(String(value).trim(), 10);

            if (!Number.isFinite(parsed)) {
                updatePageNumberInput();
                return;
            }

            const boundedIndex = Math.min(Math.max(parsed - 1, 0), bookPages.length - 1);
            const target = bookPages[boundedIndex];

            if (!target || !target.uuid) {
                updatePageNumberInput();
                return;
            }

            if (currentPage && currentPage.uuid === target.uuid) {
                updatePageNumberInput();
                return;
            }

            navigateToUuid(target.uuid);
        }

        function setupPageNumberJump() {
            const input = document.getElementById('page-number-input');
            if (!input) {
                return;
            }

            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    handlePageNumberJump(input.value);
                }
            });

            input.addEventListener('change', () => {
                handlePageNumberJump(input.value);
            });

            input.addEventListener('blur', () => {
                updatePageNumberInput();
            });

            updatePageNumberInput();
        }

        function initializeThumbnailDrawer() {
            const toggle = document.getElementById('thumbnail-toggle');
            if (!toggle) {
                thumbnailDrawerCollapsed = false;
                return;
            }

            toggle.addEventListener('click', () => {
                setThumbnailDrawerCollapsed(!thumbnailDrawerCollapsed);
            });

            setThumbnailDrawerCollapsed(false);
        }

        function setToolsVisible(show) {
            const tools = document.getElementById("page-tools");
            if (tools) {
                tools.style.display = show ? "flex" : "none";
            }
        }

        function setLoadingState(active) {
            const container = document.querySelector('.container');
            const loading = document.getElementById('loading');
            const isActive = Boolean(active);

            if (container) {
                container.classList.toggle('is-loading', isActive);
            }
            if (loading) {
                loading.setAttribute('aria-hidden', isActive ? 'false' : 'true');
            }
            if (document && document.body) {
                document.body.classList.toggle('page-is-loading', isActive);
            }
            if (!isActive) {
                scheduleThumbnailDrawerHeightSync();
            }
        }

        function setLargePreviewActive() {
            const container = document.getElementById("page-preview");
            const largeBox = document.getElementById("preview-large");

            if (!container || !largeBox) {
                return;
            }

            const isHovered = container.matches(":hover") || largeBox.matches(":hover") || container.matches(":focus-within");
            const isActive = isHovered && container.classList.contains("preview-loaded");

            if (isActive) {
                largeBox.classList.add("preview-large-visible");
            } else {
                largeBox.classList.remove("preview-large-visible");
            }

            largeBox.setAttribute("aria-hidden", isActive ? "false" : "true");
        }

        function updatePinnedPreviewImage(src) {
            const pinnedImage = document.getElementById("preview-pinned-image");
            const placeholder = document.getElementById("preview-pinned-placeholder");
            if (!pinnedImage || !placeholder) {
                return;
            }
            const hasSource = Boolean(src);
            if (hasSource) {
                if (pinnedImage.src !== src) {
                    pinnedImage.src = src;
                }
                pinnedImage.style.display = "block";
                pinnedImage.setAttribute("aria-hidden", "false");
                placeholder.style.display = "none";
                placeholder.setAttribute("aria-hidden", "true");
            } else {
                pinnedImage.removeAttribute("src");
                pinnedImage.style.display = "none";
                pinnedImage.setAttribute("aria-hidden", "true");
                placeholder.style.display = "";
                placeholder.setAttribute("aria-hidden", "false");
            }
        }

        function resetPreview() {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            setLargePreviewActive();

            if (previewObjectUrl) {
                URL.revokeObjectURL(previewObjectUrl);
                previewObjectUrl = null;
            }

            previewImageUuid = null;
            previewFetchToken = null;

            if (thumb) {
                thumb.onload = null;
                thumb.onerror = null;
                thumb.src = "";
                thumb.style.display = "none";
                thumb.style.width = "";
                thumb.style.height = "";
                thumb.style.maxWidth = "";
                thumb.style.maxHeight = "";
                thumb.style.minHeight = "";
                thumb.style.opacity = "";
            }

            if (largeImg) {
                largeImg.onload = null;
                largeImg.src = "";
                largeImg.style.width = "";
                largeImg.style.height = "";
                largeImg.style.maxWidth = "";
                largeImg.style.maxHeight = "";
            }

            if (largeBox) {
                largeBox.style.width = "";
                largeBox.style.maxWidth = "";
                largeBox.style.height = "";
            }
            updatePinnedPreviewImage("");

            if (container) {
                container.style.display = "none";
                container.classList.remove("preview-visible", "preview-loaded", "preview-error", "preview-has-status");
                delete container.dataset.previewStream;
            }
            updatePreviewStatus("");
        }

        function updatePreviewStatus(message) {
            const status = document.getElementById('preview-status');
            const container = document.getElementById('page-preview');
            if (!status || !container) {
                return;
            }
            status.textContent = message || "";
            const hasMessage = Boolean(message);
            container.classList.toggle('preview-has-status', hasMessage);
        }

        function computeThumbMaxHeight() {
            const layout = document.querySelector('.page-info-layout');
            if (layout) {
                const rect = layout.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            const details = document.querySelector('.page-details');
            if (details) {
                const rect = details.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            const section = document.getElementById("page-info");
            if (section) {
                const rect = section.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            return 260;
        }

        function computeLargePreviewWidth() {
            const resultBox = document.querySelector('#results .result-box');
            if (resultBox && resultBox.offsetWidth) {
                return Math.round(resultBox.offsetWidth);
            }

            const container = document.querySelector('.container');
            if (container && container.offsetWidth) {
                const containerWidth = container.offsetWidth;
                const fallback = Math.min(containerWidth * 0.5, window.innerWidth * 0.9);
                return Math.round(Math.max(fallback, 360));
            }

            return Math.round(Math.min(window.innerWidth * 0.6, 900));
        }

        function applyLargePreviewSizing(img, box) {
            if (!img || !box) {
                return;
            }

            const maxWidth = computeLargePreviewWidth();
            const naturalWidth = img.naturalWidth || 0;
            const naturalHeight = img.naturalHeight || 0;
            const maxViewportHeight = Math.max(Math.round(window.innerHeight * 0.9), 320);

            box.style.maxWidth = `${Math.round(maxWidth)}px`;

            if (naturalWidth > 0 && naturalHeight > 0) {
                const ratio = naturalHeight / naturalWidth;
                if (ratio > 0) {
                    let targetWidth = maxWidth;
                    let targetHeight = Math.round(targetWidth * ratio);

                    if (targetHeight > maxViewportHeight) {
                        targetHeight = maxViewportHeight;
                        targetWidth = Math.round(targetHeight / ratio);
                    }

                    const safeWidth = Math.max(200, targetWidth);
                    const safeHeight = Math.max(1, targetHeight);

                    box.style.width = `${safeWidth}px`;
                    box.style.height = `${safeHeight}px`;
                    img.style.width = `${safeWidth}px`;
                    img.style.height = `${safeHeight}px`;
                    return;
                }
            }

            const fallbackWidth = Math.max(320, Math.round(Math.min(maxWidth, 720)));
            box.style.width = `${fallbackWidth}px`;
            box.style.height = "auto";
            img.style.width = `${fallbackWidth}px`;
            img.style.height = "auto";
        }

        function refreshLargePreviewSizing() {
            const container = document.getElementById("page-preview");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            if (!container || !largeImg || !largeBox) {
                return;
            }

            if (container.classList.contains("preview-loaded") && largeImg.complete && largeImg.naturalWidth > 0) {
                applyLargePreviewSizing(largeImg, largeBox);
            } else if (!container.classList.contains("preview-loaded")) {
                largeBox.style.width = "";
                largeBox.style.height = "";
                largeImg.style.width = "";
                largeImg.style.height = "";
            }
        }

        function sizeThumbnail(thumb, maxWidth) {
            const computedHeight = computeThumbMaxHeight();
            const safeHeight = Math.min(Math.max(Number.isFinite(computedHeight) ? computedHeight : 0, 120), 180);
            const safeWidth = Math.max(Number.isFinite(maxWidth) ? maxWidth : 0, 220);

            thumb.style.height = "auto";
            thumb.style.maxHeight = `${Math.round(safeHeight)}px`;
            thumb.style.width = "auto";
            thumb.style.maxWidth = `${Math.round(safeWidth)}px`;
            thumb.style.minHeight = "120px";
            thumb.style.opacity = "1";
            thumb.style.display = "block";
        }

        async function fetchPreviewImage(uuid) {
            const streamOrder = ["AUTO", "IMG_PREVIEW", "IMG_THUMB", "IMG_FULL"];
            let lastError = null;

            for (const requestedStream of streamOrder) {
                try {
                    const response = await fetch(`/preview?uuid=${encodeURIComponent(uuid)}&stream=${requestedStream}`, { cache: "no-store" });
                    if (!response.ok) {
                        lastError = new Error(`HTTP ${response.status} (${requestedStream})`);
                        continue;
                    }

                    const contentTypeHeader = response.headers.get("Content-Type") || "";
                    const contentType = contentTypeHeader.toLowerCase();

                    if (!contentType.startsWith("image/")) {
                        lastError = new Error(`Neočekávaný obsah (${requestedStream}): ${contentTypeHeader || 'bez Content-Type'}`);
                        continue;
                    }

                    if (contentType.includes("jp2")) {
                        lastError = new Error(`Formát ${contentTypeHeader} není prohlížečem podporovaný (${requestedStream}).`);
                        continue;
                    }

                    const blob = await response.blob();
                    const actualStream = response.headers.get("X-Preview-Stream") || requestedStream;

                    return {
                        blob,
                        stream: actualStream,
                        contentType: contentTypeHeader,
                    };
                } catch (error) {
                    lastError = error;
                }
            }

            throw lastError || new Error("Náhled se nepodařilo načíst.");
        }

        function showPreviewFromCache(entry) {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            if (!container || !thumb || !largeImg || !largeBox || !previewObjectUrl) {
                return;
            }

            const cached = entry || previewCache.get(previewImageUuid) || null;
            if (cached && cached.stream) {
                container.dataset.previewStream = cached.stream;
            }

            largeImg.onload = null;
            const handleLargeLoad = () => {
                applyLargePreviewSizing(largeImg, largeBox);
            };
            largeImg.addEventListener("load", handleLargeLoad, { once: true });
            largeImg.src = previewObjectUrl;

            applyLargePreviewSizing(largeImg, largeBox);
            updatePinnedPreviewImage(previewObjectUrl);

            if (largeImg.complete && largeImg.naturalWidth > 0) {
                applyLargePreviewSizing(largeImg, largeBox);
            }

            const finalize = () => {
                thumb.style.opacity = "1";
                thumb.onload = null;
            };

            if (thumb.src !== previewObjectUrl) {
                thumb.style.opacity = "0";
                thumb.onload = finalize;
                thumb.src = previewObjectUrl;
                thumb.onerror = () => {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
                if (thumb.complete && thumb.naturalWidth > 0) {
                    finalize();
                } else if (thumb.complete) {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive();
                }
            } else if (thumb.naturalWidth > 0) {
                finalize();
            } else {
                thumb.style.opacity = "0";
                thumb.onload = finalize;
                thumb.onerror = () => {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
            }

            container.style.display = "flex";
            container.classList.add("preview-visible", "preview-loaded");
            container.classList.remove("preview-error");
            updatePreviewStatus("");

            setLargePreviewActive();
            container.style.height = "";
        }

        async function loadPreview(uuid) {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");
            if (!container || !thumb || !largeImg || !largeBox || !uuid) {
                return;
            }

            setLargePreviewActive(false);

            const preservedHeight = container.offsetHeight || 0;
            if (preservedHeight > 0) {
                container.style.height = `${preservedHeight}px`;
            }

            if (previewImageUuid === uuid && previewObjectUrl) {
                showPreviewFromCache();
                container.style.height = "";
                return;
            }

            const cachedEntry = previewCache.get(uuid);
            if (cachedEntry && cachedEntry.objectUrl) {
                previewImageUuid = uuid;
                previewObjectUrl = cachedEntry.objectUrl;
                showPreviewFromCache(cachedEntry);
                container.style.height = "";
                return;
            }

            if (previewFetchToken === uuid) {
                return;
            }

            previewFetchToken = uuid;
            previewImageUuid = uuid;

            container.style.display = "flex";
            container.classList.add("preview-visible");
            container.classList.remove("preview-loaded", "preview-error");
            updatePreviewStatus("Načítám náhled...");
            thumb.style.display = "block";
            thumb.style.opacity = "0";

            let handleLoad;

            try {
                const entry = await ensurePreviewEntry(uuid);

                if (!entry || previewImageUuid !== uuid) {
                    container.style.height = "";
                    return;
                }

                const previousUrl = previewObjectUrl;
                previewObjectUrl = entry.objectUrl;

                if (previousUrl && previousUrl !== previewObjectUrl) {
                    const stillReferenced = Array.from(previewCache.values()).some(item => item && item.objectUrl === previousUrl);
                    if (!stillReferenced) {
                        URL.revokeObjectURL(previousUrl);
                    }
                }

                largeImg.onload = null;
                const handleLargeLoad = () => {
                    applyLargePreviewSizing(largeImg, largeBox);
                };
                largeImg.addEventListener("load", handleLargeLoad, { once: true });
                largeImg.src = previewObjectUrl;

                applyLargePreviewSizing(largeImg, largeBox);
                updatePinnedPreviewImage(previewObjectUrl);

                handleLoad = () => {
                    thumb.style.opacity = "1";
                };

                thumb.addEventListener("load", handleLoad, { once: true });
                thumb.onerror = () => {
                    if (previewImageUuid === uuid) {
                        console.error("Chyba při načítání obrázku náhledu");
                        updatePreviewStatus("Náhled se nepodařilo načíst.");
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                };
                thumb.src = previewObjectUrl;
                thumb.style.opacity = "1";

                if (thumb.complete && thumb.naturalWidth === 0) {
                    if (previewImageUuid === uuid) {
                        updatePreviewStatus("Náhled se nepodařilo načíst.");
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                }

                container.dataset.previewStream = entry.stream;

                container.classList.add("preview-loaded");
                updatePreviewStatus("");

                setLargePreviewActive();
            } catch (error) {
                if (previewImageUuid === uuid) {
                    console.error("Chyba při načítání náhledu:", error);
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    setLargePreviewActive(false);
                }
            } finally {
                if (previewFetchToken === uuid) {
                    previewFetchToken = null;
                }
                if (handleLoad) {
                    thumb.removeEventListener("load", handleLoad);
                }
                container.style.height = "";
            }
        }

        function refreshPagePosition() {
            const position = document.getElementById("page-position");
            if (!position) {
                return;
            }
            if (currentPage && typeof currentPage.index === "number" && navigationState && typeof navigationState.total === "number" && navigationState.total > 0) {
                position.textContent = `${currentPage.index + 1} / ${navigationState.total}`;
            } else if (currentPage && typeof currentPage.index === "number") {
                position.textContent = `${currentPage.index + 1}`;
            } else {
                position.textContent = "-";
            }
        }

        function updateNavigationControls(nav) {
            const prevBtn = document.getElementById("prev-page");
            const nextBtn = document.getElementById("next-page");

            navigationState = nav || null;

            if (!prevBtn || !nextBtn) {
                return;
            }

            if (!navigationState) {
                prevBtn.disabled = true;
                nextBtn.disabled = true;
                refreshPagePosition();
                return;
            }

            prevBtn.disabled = !navigationState.hasPrev;
            nextBtn.disabled = !navigationState.hasNext;
            refreshPagePosition();
        }

        function updateBookInfo() {
            const section = document.getElementById("book-info");
            const titleEl = document.getElementById("book-title");
            const handleEl = document.getElementById("book-handle");
            const libraryEl = document.getElementById("book-library");
            const metadataList = document.getElementById("book-metadata");
            const metadataEmpty = document.getElementById("book-metadata-empty");
            const constantsContainer = document.getElementById("book-constants");

            if (!section || !titleEl || !handleEl || !metadataList || !metadataEmpty) {
                return;
            }

            if (!currentBook) {
                section.style.display = "none";
                handleEl.textContent = "";
                if (libraryEl) {
                    libraryEl.textContent = "";
                    libraryEl.style.display = "none";
                }
                metadataList.innerHTML = "";
                metadataEmpty.style.display = "none";
                if (constantsContainer) {
                    constantsContainer.innerHTML = "";
                    constantsContainer.style.display = "none";
                }
                return;
            }

            section.style.display = "block";
            titleEl.textContent = currentBook.title || "(bez názvu)";

            if (currentBook.handle) {
                handleEl.innerHTML = `<a href="${currentBook.handle}" target="_blank" rel="noopener">Otevřít v Krameriovi</a>`;
            } else {
                handleEl.textContent = "";
            }

            if (libraryEl) {
                if (currentLibrary && currentLibrary.label) {
                    libraryEl.textContent = currentLibrary.label;
                    libraryEl.style.display = "block";
                } else {
                    libraryEl.textContent = "";
                    libraryEl.style.display = "none";
                }
            }

            if (constantsContainer) {
                constantsContainer.innerHTML = "";
                const constants = (currentBook.constants && typeof currentBook.constants === "object") ? currentBook.constants : {};
                const chips = [];

                if (constants && typeof constants === "object") {
                    const basic = (constants.basicTextStyle && typeof constants.basicTextStyle === "object") ? constants.basicTextStyle : null;
                    if (basic) {
                        const valueParts = [];
                        if (typeof basic.fontSize === "number" && Number.isFinite(basic.fontSize)) {
                            const sizeText = basic.fontSize % 1 === 0 ? basic.fontSize.toFixed(0) : basic.fontSize.toFixed(1);
                            valueParts.push(`${sizeText} pt`);
                        }
                        if (basic.fontFamily) {
                            valueParts.push(basic.fontFamily);
                        }
                        const styleFlags = [];
                        if (basic.isBold) {
                            styleFlags.push("tučné");
                        }
                        if (basic.isItalic) {
                            styleFlags.push("kurzíva");
                        }
                        if (!styleFlags.length) {
                            styleFlags.push("regular");
                        }
                        valueParts.push(styleFlags.join(", "));

                        const metaParts = [];
                        if (typeof constants.confidence === "number" && Number.isFinite(constants.confidence)) {
                            metaParts.push(`confidence ${constants.confidence}%`);
                        }
                        if (typeof constants.sampledPages === "number" && constants.sampledPages > 0) {
                            const label = constants.sampledPages === 1 ? "1 strana" : `${constants.sampledPages} stran`;
                            metaParts.push(`vzorek ${label}`);
                        }
                        if (typeof constants.linesSampled === "number" && constants.linesSampled > 0) {
                            const lineLabel = constants.linesSampled === 1 ? "1 řádek" : `${constants.linesSampled} řádků`;
                            metaParts.push(lineLabel);
                        }
                        if (typeof constants.distinctStyles === "number" && constants.distinctStyles > 0) {
                            const styleLabel = constants.distinctStyles === 1 ? "1 styl" : `${constants.distinctStyles} stylů`;
                            metaParts.push(styleLabel);
                        }
                        if (basic.styleId) {
                            metaParts.push(`styl ${basic.styleId}`);
                        }

                        chips.push({
                            label: "Typický text",
                            value: valueParts.filter(Boolean).join(" • ") || "neuvedeno",
                            meta: metaParts.filter(Boolean).join(" • "),
                        });
                    }
                }

                if (chips.length) {
                    chips.forEach(item => {
                        const chipEl = document.createElement("div");
                        chipEl.className = "book-chip";
                        chipEl.innerHTML = `<strong>${item.label}</strong><div class="book-chip-value">${item.value}</div>`;
                        if (item.meta) {
                            const metaEl = document.createElement("div");
                            metaEl.className = "book-chip-meta";
                            metaEl.textContent = item.meta;
                            chipEl.appendChild(metaEl);
                        }
                        constantsContainer.appendChild(chipEl);
                    });
                    constantsContainer.style.display = "grid";
                } else {
                    constantsContainer.style.display = "none";
                }
            }

            metadataList.innerHTML = "";
            const mods = Array.isArray(currentBook.mods) ? currentBook.mods : [];

            if (!mods.length) {
                metadataEmpty.style.display = "block";
                metadataEmpty.textContent = "Metadata se nepodařilo načíst.";
            } else {
                metadataEmpty.style.display = "none";
                mods.forEach(entry => {
                    const dt = document.createElement("dt");
                    dt.textContent = entry.label || "---";
                    const dd = document.createElement("dd");
                    dd.textContent = entry.value || "";
                    metadataList.appendChild(dt);
                    metadataList.appendChild(dd);
                });
            }
        }

        function updatePageInfo() {
            const section = document.getElementById("page-info");
            const summary = document.getElementById("page-summary");
            const side = document.getElementById("page-side");
            const uuidEl = document.getElementById("page-uuid");
            const handleEl = document.getElementById("page-handle");

            if (!section || !summary || !side || !uuidEl || !handleEl) {
                return;
            }

            if (!currentPage) {
                section.style.display = "none";
                summary.textContent = "";
                side.textContent = "";
                uuidEl.textContent = "";
                handleEl.textContent = "";
                resetPreview();
                setToolsVisible(false);
                refreshPagePosition();
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            section.style.display = "block";

            const parts = [];
            if (currentPage.pageNumber) {
                parts.push(`Strana: ${currentPage.pageNumber}`);
            }
            if (currentPage.pageType) {
                parts.push(`Typ: ${currentPage.pageType}`);
            }
            summary.textContent = parts.length ? parts.join(" • ") : "Informace o straně nejsou k dispozici.";

            if (currentPage.pageSide) {
                side.textContent = `Pozice: ${currentPage.pageSide}`;
            } else {
                side.textContent = "Pozice: neznámá (API neposkytuje údaj).";
            }

            uuidEl.textContent = currentPage.uuid ? `UUID: ${currentPage.uuid}` : "";

            if (currentPage.handle) {
                handleEl.innerHTML = `<a href="${currentPage.handle}" target="_blank" rel="noopener">Otevřít stránku v Krameriovi</a>`;
            } else {
                handleEl.textContent = "";
            }

            const previewContainer = document.getElementById("page-preview");
            if (previewContainer) {
                previewContainer.style.display = "flex";
                previewContainer.classList.add("preview-visible");
            }

            setToolsVisible(true);
            refreshPagePosition();
        }

        function goToAdjacent(direction) {
            if (!navigationState) {
                return;
            }
            const targetUuid = direction === "prev" ? navigationState.prevUuid : navigationState.nextUuid;
            if (!targetUuid) {
                return;
            }
            navigateToUuid(targetUuid);
        }

        function elementConsumesTextInput(element) {
            if (!element) {
                return false;
            }
            if (element === document.body) {
                return false;
            }
            if (element.isContentEditable) {
                return true;
            }
            const tag = element.tagName ? element.tagName.toUpperCase() : "";
            if (!tag) {
                return false;
            }
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
                return true;
            }
            return false;
        }

        function setUuidButtonsState() {
            const uuidField = document.getElementById("uuid");
            const copyBtn = document.getElementById("uuid-copy");
            const clearBtn = document.getElementById("uuid-clear");
            const hasValue = Boolean(uuidField && uuidField.value.trim().length);
            if (copyBtn) copyBtn.disabled = !hasValue;
            if (clearBtn) clearBtn.disabled = !hasValue;
        }

        function copyCurrentUuid() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }
            const value = uuidField.value.trim();
            if (!value) {
                showTooltip('uuid-copy-tooltip', 'UUID je prázdné');
                return;
            }
            const fallbackCopy = () => {
                try {
                    const currentSelectionStart = uuidField.selectionStart;
                    const currentSelectionEnd = uuidField.selectionEnd;
                    uuidField.focus();
                    uuidField.select();
                    document.execCommand("copy");
                    if (typeof currentSelectionStart === "number" && typeof currentSelectionEnd === "number") {
                        uuidField.setSelectionRange(currentSelectionStart, currentSelectionEnd);
                    } else {
                        uuidField.setSelectionRange(value.length, value.length);
                    }
                    showTooltip('uuid-copy-tooltip', 'UUID zkopírováno');
                    return true;
                } catch (error) {
                    console.warn("Nelze zkopírovat UUID do schránky:", error);
                    showTooltip('uuid-copy-tooltip', 'Kopírování selhalo', 2600);
                    return false;
                }
            };

            if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
                navigator.clipboard.writeText(value)
                    .then(() => {
                        showTooltip('uuid-copy-tooltip', 'UUID zkopírováno');
                    })
                    .catch(() => fallbackCopy());
            } else {
                fallbackCopy();
            }
        }

        function clearUuidInput() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }
            uuidField.value = "";
            uuidField.focus();
            setUuidButtonsState();
        }

        async function pasteUuidFromClipboard() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }

            const fallback = () => {
                uuidField.focus();
                console.warn('Schránku nelze přečíst - použijte klávesovou zkratku pro vložení.');
                showTooltip('uuid-paste-tooltip', 'Schránku nelze přečíst', 2600);
            };

            if (navigator.clipboard && typeof navigator.clipboard.readText === "function") {
                try {
                    const text = await navigator.clipboard.readText();
                    const trimmed = (text || "").trim();
                    uuidField.value = trimmed;
                    setUuidButtonsState();
                    if (trimmed) {
                        showTooltip('uuid-paste-tooltip', 'UUID vloženo');
                        handleLoadClick();
                    } else {
                        showTooltip('uuid-paste-tooltip', 'Schránka je prázdná', 2000);
                        uuidField.focus();
                    }
                    return;
                } catch (error) {
                    console.warn('Nelze načíst obsah schránky:', error);
                }
            }

            fallback();
        }

        function blurUuidField() {
            const uuidField = document.getElementById("uuid");
            if (uuidField && typeof uuidField.blur === "function") {
                uuidField.blur();
            }
        }

        function isModalActive() {
            const modal = document.getElementById("alto-modal");
            return Boolean(modal && modal.style.display === "block");
        }

        function handleLoadClick() {
            blurUuidField();
            processAlto();
        }

        function setupKeyboardShortcuts() {
            document.addEventListener("keydown", (event) => {
                if (event.defaultPrevented) {
                    return;
                }

                const key = event.key;
                const activeElement = document.activeElement;
                const modalVisible = isModalActive();
                const isWithinModal = modalVisible && activeElement && activeElement.closest("#alto-modal");

                if (event.altKey || event.ctrlKey || event.metaKey) {
                    return;
                }

                if ((key === "ArrowLeft" || key === "ArrowRight") && !isWithinModal) {
                    if (elementConsumesTextInput(activeElement)) {
                        return;
                    }
                    event.preventDefault();
                    goToAdjacent(key === "ArrowLeft" ? "prev" : "next");
                    return;
                }

                if (key === "Enter" && !isWithinModal) {
                    if (activeElement && (activeElement.tagName === "TEXTAREA" || activeElement.isContentEditable)) {
                        return;
                    }
                    const loadButton = document.getElementById("load-button");
                    if (loadButton && !loadButton.disabled) {
                        event.preventDefault();
                        handleLoadClick();
                    } else {
                        event.preventDefault();
                        blurUuidField();
                        processAlto();
                    }
                }
            });
        }

        function applyProcessResult(uuid, data, previousScrollY, toolsElement) {
            cacheProcessData(uuid, data);

            currentAltoXml = data.alto_xml || "";
            const altoBtn = document.getElementById("alto-preview-btn");
            if (altoBtn) {
                altoBtn.style.display = currentAltoXml ? "block" : "none";
            }

            currentBook = data.book || null;
            currentPage = data.currentPage || null;
            currentLibrary = data.library || (currentBook && currentBook.library) || null;

            if (currentBook && currentLibrary && !currentBook.library) {
                currentBook.library = currentLibrary;
            }
            if (currentPage && currentLibrary && !currentPage.library) {
                currentPage.library = currentLibrary;
            }

            updateBookInfo();
            updatePageInfo();
            updateNavigationControls(data.navigation || null);
            ensureThumbnailGrid(Array.isArray(data.pages) ? data.pages : [], currentBook && currentBook.uuid ? currentBook.uuid : null);

            if (currentPage && currentPage.uuid) {
                loadPreview(currentPage.uuid);
            } else {
                resetPreview();
            }

            currentResults = {
                python: data.python || "",
                typescript: data.typescript || "",
                baseKey: buildResultCacheKey(data.python || "", data.typescript || "", currentPage && currentPage.uuid ? currentPage.uuid : uuid),
            };
            diffRequestToken += 1;
            diffCache.clear();
            clearAgentOutput();
            renderComparisonResults();
            const stitchRefreshPromise = refreshStitchUI();

            const results = document.getElementById("results");
            if (results) {
                results.style.display = "grid";
            }

            updateDiffToggleState();

            const uuidField = document.getElementById("uuid");
            if (uuidField && currentPage && currentPage.uuid) {
                uuidField.value = currentPage.uuid;
                setUuidButtonsState();
            }

            const correctorCtx = agentContexts.correctors;
            const correctorAuto = getContextElement(correctorCtx, 'autoCheckboxId');
            if (correctorAuto && correctorAuto.checked) {
                if (!correctorCtx.autoRunScheduled) {
                    correctorCtx.autoRunScheduled = true;
                    scheduleAutoAgentRun(correctorCtx, 5);
                }
            } else {
                correctorCtx.autoRunScheduled = false;
            }

            const readerCtx = agentContexts.readers;
            const readerAuto = getContextElement(readerCtx, 'autoCheckboxId');
            if (readerAuto && readerAuto.checked) {
                if (!readerCtx.autoRunScheduled) {
                    readerCtx.autoRunScheduled = true;
                    scheduleAutoAgentRun(readerCtx, 5);
                }
            } else {
                readerCtx.autoRunScheduled = false;
            }

            const joinerCtx = agentContexts.joiners;
            stitchRefreshPromise.then(() => {
                const joinerAuto = getContextElement(joinerCtx, 'autoCheckboxId');
                if (joinerAuto && joinerAuto.checked) {
                    if (!joinerCtx.autoRunScheduled) {
                        joinerCtx.autoRunScheduled = true;
                        scheduleAutoAgentRun(joinerCtx, 5);
                    }
                } else {
                    joinerCtx.autoRunScheduled = false;
                }
            }).catch((error) => {
                console.warn('Nepodařilo se aktualizovat sekci napojení stran:', error);
                joinerCtx.autoRunScheduled = false;
            });

            const comparisonAuto = document.getElementById('comparison-auto-run');
            if (comparisonAuto && comparisonAuto.checked) {
                scheduleComparisonAutoRun(400);
            } else {
                comparisonState.autoRunScheduled = false;
            }

            updateCacheWindow(currentPage ? currentPage.uuid : uuid, data.navigation || null);
            schedulePrefetch(data.navigation || null);
        }

        function loadStoredDiffMode() {
            try {
                const stored = localStorage.getItem(DIFF_MODE_STORAGE_KEY);
                if (stored === DIFF_MODES.WORD || stored === DIFF_MODES.CHAR) {
                    return stored;
                }
            } catch (error) {
                console.warn('Nelze načíst uložený režim diffu:', error);
            }
            return DIFF_MODES.NONE;
        }

        function persistDiffMode(mode) {
            try {
                if (mode === DIFF_MODES.WORD || mode === DIFF_MODES.CHAR) {
                    localStorage.setItem(DIFF_MODE_STORAGE_KEY, mode);
                } else {
                    localStorage.removeItem(DIFF_MODE_STORAGE_KEY);
                }
            } catch (error) {
                console.warn('Nelze uložit režim diffu:', error);
            }
        }

        function updateDiffToggleState() {
            const container = document.getElementById("diff-mode-controls");
            if (!container) {
                return;
            }
            const buttons = container.querySelectorAll('.diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) {
                    return;
                }
                const mode = button.getAttribute('data-diff-mode');
                const isActive = mode === diffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                button.dataset.diffActive = isActive ? 'true' : 'false';
            });
        }

        function setDiffMode(newMode) {
            const normalized = newMode === DIFF_MODES.WORD || newMode === DIFF_MODES.CHAR ? newMode : DIFF_MODES.NONE;
            const hasChanged = diffMode !== normalized;
            diffMode = normalized;
            if (diffMode === DIFF_MODES.NONE) {
                persistDiffMode(null);
            } else {
                persistDiffMode(diffMode);
            }
            diffCache.clear();
            updateDiffToggleState();
            if (hasChanged) {
                renderComparisonResults();
            }
        }

        function initializeDiffControls() {
            diffMode = loadStoredDiffMode();
            const container = document.getElementById("diff-mode-controls");
            if (!container) {
                return;
            }
            container.addEventListener('click', (event) => {
                const target = event.target;
                if (!target || !(target instanceof HTMLElement)) {
                    return;
                }
                if (!target.matches('.diff-toggle')) {
                    return;
                }
                const requestedMode = target.getAttribute('data-diff-mode');
                if (!requestedMode) {
                    return;
                }
                const nextMode = diffMode === requestedMode ? DIFF_MODES.NONE : requestedMode;
                setDiffMode(nextMode);
            });
            updateDiffToggleState();
        }

        function computeSimpleHash(text) {
        if (!text) {
            return 0;
        }
        let hash = 0;
        for (let index = 0; index < text.length; index += 1) {
            hash = ((hash << 5) - hash) + text.charCodeAt(index);
            hash |= 0;
        }
        return hash >>> 0;
    }

        function showTooltip(elementId, message, duration = 2200) {
            if (!elementId) {
                return;
            }
            const el = document.getElementById(elementId);
            if (!el) {
                return;
            }
            if (tooltipTimers.has(elementId)) {
                clearTimeout(tooltipTimers.get(elementId));
                tooltipTimers.delete(elementId);
            }
            const text = message ? String(message) : '';
            if (!text) {
                el.classList.remove('is-visible');
                el.textContent = '';
                return;
            }
            el.textContent = text;
            el.classList.add('is-visible');
            const timerId = window.setTimeout(() => {
                el.classList.remove('is-visible');
                tooltipTimers.delete(elementId);
            }, Math.max(1000, duration || 0));
            tooltipTimers.set(elementId, timerId);
        }

        function buildResultCacheKey(pythonHtml, tsHtml, pageUuid) {
            const leftHash = computeSimpleHash(pythonHtml || "");
            const rightHash = computeSimpleHash(tsHtml || "");
            return `${pageUuid || "standalone"}:${leftHash}:${rightHash}`;
        }

        let diffRequestToken = 0;

        function renderComparisonResults() {
            const pythonRendered = document.getElementById("python-result");
            const tsRendered = document.getElementById("typescript-result");
            if (!pythonRendered || !tsRendered) {
                return;
            }

            const pythonHtml = currentResults.python || "";
            const tsHtml = currentResults.typescript || "";

            pythonRendered.innerHTML = pythonHtml
                ? `<pre>${pythonHtml}</pre>`
                : '<div class="muted">Žádná data.</div>';
            tsRendered.innerHTML = tsHtml
                ? `<pre>${tsHtml}</pre>`
                : '<div class="muted">Žádná data.</div>';

            const baseKey = currentResults.baseKey || buildResultCacheKey(
                pythonHtml,
                tsHtml,
                currentPage && currentPage.uuid ? currentPage.uuid : ""
            );
            updateDiffSection(pythonHtml, tsHtml, baseKey);
        }

        function hideDiffSection() {
            const diffSection = document.getElementById("diff-section");
            const pythonHtmlContainer = document.getElementById("python-html");
            const tsHtmlContainer = document.getElementById("typescript-html");
            if (pythonHtmlContainer) {
                pythonHtmlContainer.innerHTML = "";
            }
            if (tsHtmlContainer) {
                tsHtmlContainer.innerHTML = "";
            }
            if (diffSection) {
                diffSection.classList.remove("is-visible");
            }
        }

        async function updateDiffSection(pythonHtml, tsHtml, baseKey) {
            const diffSection = document.getElementById("diff-section");
            const pythonHtmlContainer = document.getElementById("python-html");
            const tsHtmlContainer = document.getElementById("typescript-html");

            if (!diffSection || !pythonHtmlContainer || !tsHtmlContainer) {
                return;
            }

            if (diffMode === DIFF_MODES.NONE) {
                hideDiffSection();
                return;
            }

            const requestedMode = diffMode;
            const cacheKey = `${requestedMode}:${baseKey}`;
            if (diffCache.has(cacheKey)) {
                const cached = diffCache.get(cacheKey) || {};
                pythonHtmlContainer.innerHTML = cached.python || "";
                tsHtmlContainer.innerHTML = cached.typescript || "";
                diffSection.classList.add("is-visible");
                return;
            }

            const requestId = ++diffRequestToken;
            pythonHtmlContainer.innerHTML = '<div class="diff-loading">Načítám diff…</div>';
            tsHtmlContainer.innerHTML = '<div class="diff-loading">Načítám diff…</div>';
            diffSection.classList.add("is-visible");

            try {
                const response = await fetch('/diff', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        python: pythonHtml,
                        typescript: tsHtml,
                        mode: requestedMode,
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false || !data.diff) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                diffCache.set(cacheKey, data.diff);
                if (requestId !== diffRequestToken || diffMode !== requestedMode) {
                    return;
                }
                pythonHtmlContainer.innerHTML = data.diff.python || "";
                tsHtmlContainer.innerHTML = data.diff.typescript || "";
                diffSection.classList.add("is-visible");
            } catch (error) {
                console.error('Chyba při načítání diffu:', error);
                if (requestId === diffRequestToken) {
                    const message = typeof error?.message === 'string' ? error.message : String(error);
                    pythonHtmlContainer.innerHTML = `<div class="diff-error">Diff se nepodařilo načíst: ${escapeHtml(message)}</div>`;
                    tsHtmlContainer.innerHTML = `<div class="diff-error">Diff se nepodařilo načíst.</div>`;
                    diffSection.classList.add("is-visible");
                }
            }
        }

        function escapeHtml(input) {
            if (!input) {
                return "";
            }
            return input.replace(/[&<>"']/g, (char) => {
                switch (char) {
                    case '&':
                        return '&amp;';
                    case '<':
                        return '&lt;';
                    case '>':
                        return '&gt;';
                    case '"':
                        return '&quot;';
                    case "'":
                        return '&#39;';
                    default:
                        return char;
                }
            });
        }

        async function processAlto() {
            const uuidField = document.getElementById("uuid");
            const uuid = uuidField ? uuidField.value.trim() : "";

            if (!uuid) {
                alert("Zadejte UUID");
                return;
            }

            const token = ++processRequestToken;
            const previousScrollY = window.pageYOffset || window.scrollY || 0;
            const toolsElement = document.getElementById("page-tools");
            const shouldShowLoading = !pageCache.has(uuid);

            if (shouldShowLoading) {
                setLoadingState(true);
            }

            try {
                const data = await ensureProcessData(uuid);

                if (token !== processRequestToken) {
                    return;
                }

                applyProcessResult(uuid, data, previousScrollY, toolsElement);
            } catch (error) {
                if (token !== processRequestToken) {
                    return;
                }
                console.error("Chyba při zpracování:", error);
                const message = error && error.message ? error.message : String(error);
                alert("Chyba při zpracování: " + message);
                window.requestAnimationFrame(() => {
                    window.scrollTo(0, previousScrollY);
                });
            } finally {
                if (token === processRequestToken) {
                    setLoadingState(false);
                }
            }
        }

        function makeDraggable(element, handle) {
            let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            const dragHandle = handle || element;
            dragHandle.onmousedown = dragMouseDown;
            function dragMouseDown(e) {
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
            }
            function elementDrag(e) {
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                element.style.top = (element.offsetTop - pos2) + "px";
                element.style.left = (element.offsetLeft - pos1) + "px";
            }
            function closeDragElement() {
                document.onmouseup = null;
                document.onmousemove = null;
            }
        }

        window.onload = function () {
            const prev = document.getElementById("prev-page");
            const next = document.getElementById("next-page");
            if (prev) {
                prev.addEventListener("click", () => goToAdjacent("prev"));
            }
            if (next) {
                next.addEventListener("click", () => goToAdjacent("next"));
            }

            const uuidField = document.getElementById("uuid");
            const uuidCopyBtn = document.getElementById("uuid-copy");
            const uuidPasteBtn = document.getElementById("uuid-paste");
            const uuidClearBtn = document.getElementById("uuid-clear");
            if (uuidField) {
                uuidField.addEventListener("input", () => setUuidButtonsState());
            }
            if (uuidCopyBtn) {
                uuidCopyBtn.addEventListener("click", copyCurrentUuid);
            }
            if (uuidPasteBtn) {
                uuidPasteBtn.addEventListener("click", () => {
                    pasteUuidFromClipboard().catch((error) => {
                        console.warn('Vložení UUID ze schránky se nezdařilo:', error);
                    });
                });
            }
            if (uuidClearBtn) {
                uuidClearBtn.addEventListener("click", clearUuidInput);
            }
            setUuidButtonsState();

            setupPageNumberJump();
            initializeThumbnailDrawer();
            initializePreviewDrawer();
            initializeDiffControls();
            initializeAgentUI(agentContexts.correctors);
            initializeAgentUI(agentContexts.readers);
            initializeAgentUI(agentContexts.joiners);
            initializeStitchingUI();
            initializeAgentDiffControls();
            initializeComparisonUI();
            initializeComparison2UI();

            const previewContainer = document.getElementById("page-preview");
            const largeBox = document.getElementById("preview-large");
            if (previewContainer) {
                const handleEnter = () => setLargePreviewActive(true);
                const handleLeave = () => setLargePreviewActive(false);

                previewContainer.addEventListener("pointerenter", handleEnter);
                previewContainer.addEventListener("pointerleave", handleLeave);
                previewContainer.addEventListener("mouseenter", handleEnter);
                previewContainer.addEventListener("mouseleave", handleLeave);
                previewContainer.addEventListener("focusin", handleEnter);
                previewContainer.addEventListener("focusout", handleLeave);
            }
            if (largeBox) {
                const handleEnter = () => setLargePreviewActive(true);
                const handleLeave = () => setLargePreviewActive(false);

                largeBox.addEventListener("pointerenter", handleEnter);
                largeBox.addEventListener("pointerleave", handleLeave);
                largeBox.addEventListener("mouseenter", handleEnter);
                largeBox.addEventListener("mouseleave", handleLeave);
                largeBox.addEventListener("focusin", handleEnter);
                largeBox.addEventListener("focusout", handleLeave);
            }

            const altoBtn = document.getElementById("alto-preview-btn");
            if (altoBtn) {
                altoBtn.style.color = "#007bff";
                altoBtn.style.cursor = "pointer";
                altoBtn.style.textDecoration = "underline";
            altoBtn.addEventListener("click", () => {
                const modal = document.getElementById("alto-modal");
                const content = document.getElementById("alto-content");
                if (modal && content) {
                    content.textContent = currentAltoXml;
                    modal.style.display = "block";
                    modal.focus();
                    const modalContent = modal.querySelector('.modal-content');
                    const modalHeader = modal.querySelector('.modal-header');
                    if (modalContent) {
                        // Set initial position to center
                        modalContent.style.top = "";
                        modalContent.style.left = "";
                        modalContent.style.transform = 'translate(-50%, -50%)';
                        // Make draggable only on header
                        if (modalHeader) {
                            makeDraggable(modalContent, modalHeader);
                        }
                    }
                }
            });
            }

            const closeBtn = document.querySelector(".close");
            if (closeBtn) {
                closeBtn.addEventListener("click", () => {
                    const modal = document.getElementById("alto-modal");
                    if (modal) modal.style.display = "none";
                });
            }

            const modal = document.getElementById("alto-modal");
            if (modal) {
                modal.addEventListener("keydown", (e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === "a") {
                        e.preventDefault();
                        const content = document.getElementById("alto-content");
                        if (content) {
                            const range = document.createRange();
                            range.selectNodeContents(content);
                            const selection = window.getSelection();
                            selection.removeAllRanges();
                            selection.addRange(range);
                        }
                    }
                });
            }

            setupKeyboardShortcuts();

            window.addEventListener("click", (event) => {
                const modal = document.getElementById("alto-modal");
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            });

            processAlto();
        };

        window.addEventListener("resize", () => {
            refreshLargePreviewSizing();
            setLargePreviewActive();
            scheduleThumbnailDrawerHeightSync(true);
        });
    </script>
    <div id="alto-modal" class="modal" tabindex="-1">
        <div class="modal-content">
            <div class="modal-header">
                <h2>ALTO XML Obsah</h2>
                <span class="close">&times;</span>
            </div>
            <pre id="alto-content"></pre>
        </div>
    </div>
</body>
</html>'''
            html = html.replace('__MODEL_REGISTRY_DATA__', MODEL_REGISTRY_JSON)
            html = html.replace('__DEFAULT_AGENT_MODEL__', json.dumps(DEFAULT_MODEL))
            html = html.replace('__DEFAULT_AGENT_PROMPT__', json.dumps(DEFAULT_AGENT_PROMPT_TEXT))
            self.wfile.write(html.encode('utf-8'))

        elif self.path.startswith('/process'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()

            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]
            api_base_override = query_params.get('api_base', [''])[0] or None

            if not uuid:
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            try:
                # Create processor with optional api_base_override so the server-side
                # calls are made against the same Kramerius instance the UI selected.
                processor = AltoProcessor(api_base_url=api_base_override)
                context = processor.get_book_context(uuid)

                if not context:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se načíst metadata pro zadané UUID'}).encode('utf-8'))
                    return

                print(f"Book constants for {uuid}: {context.get('book_constants')}")

                page_uuid = context.get('page_uuid')
                if not page_uuid:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se určit konkrétní stránku pro zadané UUID'}).encode('utf-8'))
                    return

                book_uuid = context.get('book_uuid')
                # Prefer explicit api_base from context, otherwise use processor's base (which
                # may have been initialized from api_base_override)
                active_api_base = context.get('api_base') or processor.api_base_url
                library_info = describe_library(active_api_base)
                handle_base = library_info.get('handle_base') or ''

                alto_xml = processor.get_alto_data(page_uuid)
                if not alto_xml:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se stáhnout ALTO data'}).encode('utf-8'))
                    return

                pretty_alto = minidom.parseString(alto_xml).toprettyxml(indent="  ")

                python_result = processor.get_formatted_text(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
                typescript_result = simulate_typescript_processing(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)

                pages = context.get('pages', [])
                current_index = context.get('current_index', -1)
                total_pages = len(pages)
                has_prev = current_index > 0
                has_next = current_index >= 0 and current_index < total_pages - 1

                prev_uuid = pages[current_index - 1]['uuid'] if has_prev else None
                next_uuid = pages[current_index + 1]['uuid'] if has_next else None

                book_data = context.get('book') or {}
                mods_metadata = context.get('mods') or []

                def clean(value: str) -> str:
                    if not value:
                        return ''
                    return ' '.join(value.replace('\xa0', ' ').split())

                page_summary = context.get('page') or {}
                page_item = context.get('page_item') or {}
                book_handle = f"{handle_base}/handle/uuid:{book_uuid}" if handle_base and book_uuid else ''
                page_handle = f"{handle_base}/handle/uuid:{page_uuid}" if handle_base and page_uuid else ''

                page_info = {
                    'uuid': page_uuid,
                    'title': clean(page_summary.get('title') or page_item.get('title') or ''),
                    'pageNumber': clean(page_summary.get('pageNumber') or (page_item.get('details') or {}).get('pagenumber') or ''),
                    'pageType': clean(page_summary.get('pageType') or (page_item.get('details') or {}).get('type') or ''),
                    'pageSide': clean(page_summary.get('pageSide') or (page_item.get('details') or {}).get('pageposition') or (page_item.get('details') or {}).get('pagePosition') or (page_item.get('details') or {}).get('pagerole') or ''),
                    'index': current_index,
                    'iiif': page_item.get('iiif'),
                    'handle': page_handle,
                    'library': library_info,
                }

                book_info = {
                    'uuid': context.get('book_uuid'),
                    'title': clean(book_data.get('title') or ''),
                    'model': book_data.get('model'),
                    'handle': book_handle,
                    'mods': mods_metadata,
                    'constants': context.get('book_constants') or {},
                    'library': library_info,
                }

                navigation = {
                    'hasPrev': has_prev,
                    'hasNext': has_next,
                    'prevUuid': prev_uuid,
                    'nextUuid': next_uuid,
                    'total': total_pages,
                }

                response_data = {
                    'python': python_result,
                    'typescript': typescript_result,
                    'book': book_info,
                    'pages': pages,
                    'currentPage': page_info,
                    'navigation': navigation,
                    'alto_xml': pretty_alto,
                    'library': library_info,
                }

                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))

            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        elif self.path.startswith('/preview'):
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]
            stream = query_params.get('stream', ['IMG_PREVIEW'])[0]
            allowed_streams = {'IMG_THUMB', 'IMG_PREVIEW', 'IMG_FULL', 'AUTO'}

            if not uuid:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            if stream not in allowed_streams:
                stream = 'IMG_PREVIEW'

            candidate_streams = [stream]
            if stream == 'AUTO':
                candidate_streams = ['IMG_FULL', 'IMG_PREVIEW', 'IMG_THUMB']

            candidate_bases = list(dict.fromkeys(DEFAULT_API_BASES))

            last_error = None

            try:
                for candidate in candidate_streams:
                    for base in candidate_bases:
                        upstream_url = f"{base}/item/uuid:{uuid}/streams/{candidate}"
                        response = requests.get(upstream_url, timeout=20)

                        if response.status_code != 200 or not response.content:
                            last_error = f'Nepodařilo se načíst náhled (status {response.status_code} pro {candidate} z {base})'
                            response.close()
                            continue

                        content_type = response.headers.get('Content-Type', 'image/jpeg')
                        if 'jp2' in content_type.lower():
                            last_error = f'Stream {candidate} vrací nepodporovaný formát {content_type}'
                            response.close()
                            continue

                        self.send_response(200)
                        self.send_header('Content-type', content_type)
                        self.send_header('Content-Length', str(len(response.content)))
                        self.send_header('Cache-Control', 'no-store')
                        self.send_header('X-Preview-Stream', candidate)
                        self.end_headers()
                        self.wfile.write(response.content)
                        response.close()
                        return

                self.send_response(502)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                message = last_error or 'Nepodařilo se načíst náhled.'
                self.wfile.write(json.dumps({'error': message}).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(content_length) if content_length else b''
        try:
            payload = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            payload = {}
        collection = ''
        if isinstance(payload, dict):
            collection = payload.get('collection') or ''

        if path == '/diff':
            python_html = ''
            ts_html = ''
            mode = DIFF_MODE_WORD
            if isinstance(payload, dict):
                python_html = str(payload.get('python') or payload.get('python_html') or '')
                ts_html = str(payload.get('typescript') or payload.get('typescript_html') or '')
                requested_mode = payload.get('mode')
                if isinstance(requested_mode, str):
                    mode = requested_mode
            try:
                diff_result = build_html_diff(python_html, ts_html, mode)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {
                    'ok': True,
                    'diff': diff_result,
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
            return

        if path == '/agents/diff':
            original_html = ''
            corrected_html = ''
            mode = DIFF_MODE_WORD
            if isinstance(payload, dict):
                original_html = str(payload.get('original') or payload.get('original_html') or '')
                corrected_html = str(payload.get('corrected') or payload.get('corrected_html') or '')
                requested_mode = payload.get('mode')
                if isinstance(requested_mode, str):
                    mode = requested_mode
            try:
                diff_result = build_agent_diff(original_html, corrected_html, mode)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {
                    'ok': True,
                    'diff': diff_result,
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
            return

        if path == '/agents/save':
            stored = write_agent_file(payload if isinstance(payload, dict) else {}, collection)
            # write_agent_file now returns canonical name on success, or None on failure
            if stored:
                # return the saved agent data back to client for immediate UI sync
                data = read_agent_file(stored, collection) or {}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'stored_name': stored, 'collection': normalize_agent_collection(collection), 'agent': data}, ensure_ascii=False).encode('utf-8'))
                return
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'invalid'}).encode('utf-8'))
                return

        if path == '/agents/delete':
            name = payload.get('name') if isinstance(payload, dict) else None
            ok = delete_agent_file(name or '', collection)
            if ok:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
                return
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'not_found'}).encode('utf-8'))
                return
            return

        if path == '/agents/run':
            request_payload = payload if isinstance(payload, dict) else {}
            agent_name = request_payload.get('name')
            agent = read_agent_file(agent_name or '', collection)
            if not agent:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'agent_not_found'}).encode('utf-8'))
                return
            model_override = str(request_payload.get('model_override') or '').strip()
            if not model_override:
                model_override = str(request_payload.get('model') or '').strip()
            reasoning_override = str(request_payload.get('reasoning_effort') or '').strip().lower()
            snapshot = request_payload.get('agent_snapshot')
            if isinstance(snapshot, dict):
                agent_for_run = dict(agent)
                agent_for_run.update(snapshot)
            else:
                agent_for_run = dict(agent)
            agent_for_run.setdefault('name', agent_name or '')
            if model_override:
                agent_for_run['model'] = model_override
            if reasoning_override in {'low', 'medium', 'high'}:
                agent_for_run['reasoning_effort'] = reasoning_override
            try:
                result = run_agent_via_responses(agent_for_run, request_payload)
            except AgentRunnerError as err:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
                return
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
                return

            response_body = {
                'ok': True,
                'result': result,
                'auto_correct': bool(request_payload.get('auto_correct')),
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response_body, ensure_ascii=False).encode('utf-8'))
            return

def ensure_typescript_build() -> bool:
    if TS_DIST_PATH.exists():
        return True

    npx_path = shutil.which('npx')
    if not npx_path:
        return False

    result = subprocess.run(
        [npx_path, 'tsc'],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_output = result.stderr.strip() or result.stdout.strip()
        print(f"TypeScript build failed: {error_output}")
        return False

    return TS_DIST_PATH.exists()


def simulate_typescript_processing(alto_xml: str, uuid: str, width: int, height: int) -> str:
    """Spuštění původní TypeScript logiky přes Node.js"""
    if not ensure_typescript_build():
        return "TypeScript build není k dispozici (zkontrolujte instalaci Node.js a spusťte 'npx tsc')."

    node_path = shutil.which('node')
    if not node_path:
        return "Node.js není dostupný v PATH."

    try:
        completed = subprocess.run(
            [
                node_path,
                str(TS_DIST_PATH),
                'formatted',
                '--stdin',
                '--uuid',
                uuid,
                '--width',
                str(width),
                '--height',
                str(height)
            ],
            input=alto_xml,
            text=True,
            capture_output=True,
            timeout=45,
            cwd=str(ROOT_DIR)
        )

        if completed.returncode != 0:
            error_output = completed.stderr.strip() or completed.stdout.strip()
            return f"TypeScript chyba: {error_output}"

        return completed.stdout.strip()

    except subprocess.TimeoutExpired:
        return "TypeScript zpracování vypršelo (timeout)."
    except Exception as err:
        return f"TypeScript výjimka: {err}"

def run_server(port=8000):
    """Spuštění webového serveru"""
    with ThreadingHTTPServer(("", port), ComparisonHandler) as httpd:
        print(f"Server běží na http://localhost:{port}")
        print("Otevírám prohlížeč...")
        webbrowser.open(f'http://localhost:{port}')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer zastaven")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()
