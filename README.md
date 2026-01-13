# Alto Processing Web Service

FastAPI aplikace s UI (šablona `app/templates/compare.html`) a REST endpointy `/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`. Umí algoritmický převod ALTO, LLM korekce, OCR, export do TXT/HTML/MD/EPUB včetně ilustrací (K5 manuální ořez, K7 IIIF). Běží jako jeden proces; UI komunikuje přes REST.

## Co potřebujete
- Python 3.10+
- Node.js + npm (jen kvůli legacy TypeScript bundlu `dist/run_original.js`)
- Docker + Docker Compose plugin (pro kontejnerové spuštění)

## Rychlý start lokálně (bez Dockeru)
1) `cp .env.example .env` a doplňte API klíče (OpenRouter/OpenAI) + další proměnné.  
2) Virtuální env a závislosti:
```bash
python3 -m venv .webenv
source .webenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3) Node závislosti a případný rebuild TS (jen pro legacy `dist/run_original.js`):
```bash
npm install
npx tsc   # jen pokud potřebuješ přebuildit dist/run_original.js
```
4) Spuštění:
```bash
./start.sh  # respektuje HOST, PORT, WEB_CONCURRENCY, LOG_LEVEL
# nebo:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080 --app-dir .
```
Otevři http://localhost:8080. `/healthz` vrací `{"status": "ok", "environment": "<ALTO_WEB_ENVIRONMENT>"}`.

## Docker / Compose
- Build a run:
```bash
docker build -t alto-web .
docker run --rm -p 8080:8080 --env-file .env alto-web
```
- Compose ( pohodlnější, bez port mappingu v defaultu ):
```bash
cp .env.example .env
docker compose up --build -d
```
Kontejner mapuje `./agents` a `./config` jako bind mounty. Port 8080 je dostupný jen v síti Compose; do světa ho publikuje Nginx v override (viz níže).

## Produkce: Nginx reverse proxy + Let’s Encrypt
Repo má připravené `deploy/docker-compose.override.yml` (služba `nginx`) a `deploy/nginx.conf`.

1) Certy Let’s Encrypt na hostu (certbot webroot):
```bash
sudo mkdir -p /var/www/certbot/.well-known/acme-challenge
sudo certbot certonly --webroot -w /var/www/certbot \
  -d alto-processing.trinera.cloud --agree-tos -m tvuj@email.cz --non-interactive
```
2) Spuštění s proxy:
```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.override.yml up --build -d
```
Nginx vystaví 80/443, předává na `alto-web:8080`, posílá hlavičky Host/X-Forwarded-For/Proto, přesměrovává HTTP→HTTPS, HSTS je zapnuté.
3) Ověření: `https://alto-processing.trinera.cloud/healthz` (200), `curl -I http://.../healthz` (301 na https).
4) Obnova certů (cron pro root, 3:00 denně):
```
0 3 * * * certbot renew --webroot -w /var/www/certbot --post-hook "docker compose -f /root/alto_processing/docker-compose.yml -f /root/alto_processing/deploy/docker-compose.override.yml exec nginx nginx -s reload"
```

Volitelně můžete přidat pravidelný restart aplikačního kontejneru (např. denně kvůli úklidu dočasných exportů):
```
0 4 * * * docker compose -f /root/alto_processing/docker-compose.yml -f /root/alto_processing/deploy/docker-compose.override.yml restart alto-web
```

## Struktura
- `app/` FastAPI (`app/main.py`), šablony `app/templates/compare.html`, logika v `app/core/`
- `static/` statická aktiva UI
- `agents/`, `config/` runtime data (bind mount v Compose)
- `dist/` bundlovaný legacy JS `run_original.js` (build přes `npx tsc`)
- `start.sh` wrapper nad uvicorn (čte HOST, PORT, WEB_CONCURRENCY, LOG_LEVEL)

## API / UI
- UI na `/`, REST `/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`.
- `/healthz` pro monitoring.
- Token v hlavičce `Authorization: Bearer <ALTO_WEB_AUTH_TOKEN>` (healthz je veřejné, chráněné endpointy vrací 303/401 bez tokenu).
- Lokální testování: BASE typicky `http://127.0.0.1:8080` (bez `/auth`). `/auth` je jen HTML login pro cookie.

### Download přes API
`POST /download` → vrátí `job_id`. Stav zjistíte přes `GET /exports/{job_id}` (obsahuje `state`, `progress.percent`, `progress.eta_seconds`) a výsledek stáhnete z `GET /exports/{job_id}/download`.

Podporované pole requestu:
- `uuid` (povinné) – UUID knihy nebo stránky.
- `format` – `txt` (default) | `html` | `md` | `epub`.
- `range` – `all` / `book` / `*` nebo např. `"7-11,23"`; při UUID stránky a bez `range` se použije jen daná stránka.
- `llmAgent` – např. `{ "name": "cleanup-diff-generated-mid" }` (přepne source na LLM).
- `dropSmall` – odfiltruje malé bloky (mapuje se na `omit_small_text`).
- `ignoreImages` – u EPUB potlačí obrázky (pro OCR se ignorují vždy).
- `languageHint` – jazykový hint pro agenty, default `cs`.
- `apiBase` – volitelná Kramerius API URL; přebije autodetekci.
- `authors`, `coverUuid`, `outputName` – volitelná metadata (cover se jinak vybere automaticky z prioritních stránek).

Příklad (produkce, BASE míří na `/auth`):
```bash
TOKEN="..."; BASE="https://alto-processing.trinera.cloud/auth"
curl -X POST "$BASE/download" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "uuid": "49c6424a-c820-4224-9475-4aa0d8a9d844",
    "format": "epub",
    "range": "all",
    "dropSmall": true,
    "ignoreImages": false,
    "languageHint": "cs",
    "outputName": "output.epub"
  }'
```
Polling (stav + procenta + ETA) a stažení:
```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/exports/<job_id>"
curl -H "Authorization: Bearer $TOKEN" -o output.txt "$BASE/exports/<job_id>/download"
```
Ukázka odpovědi stavu:
```json
{
  "job_id": "abc123",
  "state": "running",
  "progress": {
    "processed": 3,
    "total": 10,
    "message": "Zpracovávám stránku 3/10",
    "percent": 30.0,
    "eta_seconds": 42.5
  },
  "error": null,
  "download_url": null,
  "filename": null
}
```

### CLI helper
`cli/download.py` dělá totéž; token z `--token` nebo `ALTO_TOKEN`.
```bash
ALTO_TOKEN=TVŮJ_TOKEN python cli/download.py \
  --url http://localhost:8080 \
  --uuid 49c6424a-c820-4224-9475-4aa0d8a9d844 \
  --format epub \
  --range "7-11,23" \
  --llm-agent '{"name":"cleanup-diff-generated-mid"}' \
  --api-base https://kramerius5.nkp.cz/search/api/v5.0 \
  --drop-small \
  --ignore-images \
  --language-hint cs \
  --output output.epub
```
CLI přepínače: `--ignore-images`, `--api-base`, `--language-hint`, `--output-name` (jinak se bere z `--output`), `--drop-small`, `--llm-agent`.

## Poznámky
- Node.js je jen kvůli legacy TypeScriptu; pokud bundl nebude potřeba, lze NPM kroky z Dockerfile odstranit.
- `WEB_CONCURRENCY` ve `start.sh` nastav na počet CPU jader.
- Exportní joby ukládají do dočasných souborů; při restartu zmizí. Bind mounty `agents/` a `config/` si drž ve storage.
- Server musí mít odchozí HTTPS k API Krameria (včetně K7/IIIF) a OpenRouter/OpenAI; jinak zpracování padne.
- Ilustrace: ALTO se převádí na `<note class="illustration" data-bbox=... data-page-width=...>`; EPUB může stáhnout obrázky (K7 přes IIIF, K5 ořez).
- Podporovaní provider filtrování a modely v UI (OpenRouter/CERIT), vlastní LLM úpravy (custom agent), manual-joiner pro napojování stran.
