#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stáhne knihu/stránku přes /download API.")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL služby (bez koncového /api).")
    parser.add_argument("--uuid", required=True, help="UUID knihy nebo stránky.")
    parser.add_argument("--format", default="txt", help="Výstupní formát: txt|html|md|epub.")
    parser.add_argument("--range", dest="range_value", help="Rozsah stránek, např. 1-3,5.")
    parser.add_argument("--llm-agent", dest="llm_agent", help="LLM agent jako JSON (např. '{\"name\":\"gpt4o\"}').")
    parser.add_argument("--drop-small", action="store_true", help="Odfiltrovat malé bloky textu.")
    parser.add_argument("--ignore-images", action="store_true", help="Vynechat obrázky (má smysl hlavně pro EPUB).")
    parser.add_argument("--language-hint", default="cs", help="Jazykový hint pro agenty (default: cs).")
    parser.add_argument("--api-base", dest="api_base", help="Kramerius API base URL (přepíše default).")
    parser.add_argument("--output", dest="output_path", help="Cesta pro uložení výsledku.")
    parser.add_argument("--output-name", dest="output_name", help="Název souboru předaný serveru (výchozí z --output).")
    parser.add_argument("--token", help="Auth token; když není, vezme se ALTO_TOKEN z prostředí.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Jak často zjišťovat stav (v sekundách).")
    return parser.parse_args()


def load_llm_agent(raw: Optional[str]) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("LLM agent musí být validní JSON.", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("ALTO_TOKEN") or ""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    output_name = args.output_name
    if not output_name and args.output_path:
        output_name = Path(args.output_path).name

    payload = {
        "uuid": args.uuid,
        "format": args.format,
        "range": args.range_value,
        "llmAgent": load_llm_agent(args.llm_agent),
        "dropSmall": bool(args.drop_small),
        "ignoreImages": bool(args.ignore_images),
        "languageHint": args.language_hint,
        "apiBase": args.api_base,
        "outputName": output_name,
    }
    api_base = args.url.rstrip("/")
    download_endpoint = f"{api_base}/download"

    print(f"Spouštím download přes {download_endpoint} ...")
    resp = requests.post(download_endpoint, json=payload, headers=headers, timeout=30)
    if resp.status_code >= 300:
        print(f"Chyba při startu: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    job = resp.json()
    job_id = job.get("job_id")
    if not job_id:
        print("Chybí job_id ve odpovědi.", file=sys.stderr)
        sys.exit(1)

    status_endpoint = f"{api_base}/exports/{job_id}"
    download_url = f"{api_base}/exports/{job_id}/download"
    print(f"Job {job_id} spuštěn. Kontroluji stav na {status_endpoint}")

    last_line_len = 0
    while True:
        time.sleep(max(0.2, args.poll_interval))
        status_resp = requests.get(status_endpoint, headers=headers, timeout=30)
        if status_resp.status_code >= 300:
            print(f"Chyba při zjišťování stavu: {status_resp.status_code} {status_resp.text}", file=sys.stderr)
            sys.exit(1)
        data = status_resp.json()
        state = data.get("state")
        progress = data.get("progress") or {}
        percent = progress.get("percent") or 0
        message = progress.get("message") or ""
        line = f"[{state}] {percent:.1f}% {message}"
        padding = " " * max(0, last_line_len - len(line))
        print(line + padding, end="\r", flush=True)
        last_line_len = len(line)
        if state in {"completed", "failed", "aborted"}:
            print()  # newline after carriage return
            if state != "completed":
                print(f"Job skončil se stavem {state}: {data.get('error')}", file=sys.stderr)
                sys.exit(1)
            break

    outfile = args.output_path or (data.get("filename") or f"{job_id}.{args.format}")
    print(f"Stahuji výsledek do {outfile} ...")
    dl_resp = requests.get(download_url, headers=headers, timeout=120)
    if dl_resp.status_code >= 300:
        print(f"Chyba při stahování: {dl_resp.status_code} {dl_resp.text}", file=sys.stderr)
        sys.exit(1)
    Path(outfile).write_bytes(dl_resp.content)
    print(f"Hotovo: {outfile}")


if __name__ == "__main__":
    main()
