# Internet Explorer

`internet-explorer` is an orchestration service for discovering new datasources for a given intent.

Runtime dependencies are local to this repo + installed libraries:
- native Google Custom Search API calls from this repo
- `eu-swarm` + `browser-use` imports as libraries
- local known-tools inventory file for duplicate checks
- local VPN helper scripts/config

## What it does

1. Generate a fixed set of strategies for an intent.
2. Generate Google-only queries for each strategy.
3. Fetch two SERP pages per query.
4. Collapse SERP hits to one unique registrable domain per candidate source.
5. Choose a start surface per candidate (`domain_homepage` or `first_result_url`).
6. Check novelty against a baseline domain file.
7. Build an initial link seed once from `sitemap.xml`, `robots.txt`, and `llms.txt`.
8. Pass that initial link list to browser delegation at start; do not maintain a mutable site map in-memory.
9. Use `eu-swarm` only for structured browser plan output.
10. Execute browser exploration with native `browser-use`, which can navigate to new URLs mid-run and call extra tools such as PDF verification.
11. Decide whether the source is useful, store evidence, and save one combined reasoning field.
12. Log every phase into append-only events.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m internet_explorer.cli --intent "your intent here"
```

Or use the console entrypoint:

```bash
internet-explorer --intent "your intent here"
```

Generic VPN commands:

```bash
internet-explorer --vpn-status
internet-explorer --vpn-start
internet-explorer --vpn-stop
```

## Config

Start from `.env.example`.

Important behavior:
- Config is loaded from this repo's environment (`.env` + process env overrides).
- `MONGODB_URI` must be set in this repo config.
- `MAX_BROWSER_CONCURRENCY=0` means unlimited.
- `MAX_URL_CONCURRENCY=0` means unlimited.
- `CANDIDATE_START_MODE=domain_homepage` means one candidate per unique domain and the normal agent starts from the domain homepage.
- `CANDIDATE_START_MODE=first_result_url` means one candidate per unique domain but the normal agent starts from the first SERP URL seen for that domain.
- `MAX_SITE_GRAPH_FRONTIER` controls how many initial seeded links are passed into browser delegation.
- `PDF_INLINE_MAX_BYTES` limits direct inline Gemini PDF verification size.
- `KNOWN_TOOLS_FILE` provides the static duplicate-tool baseline.
- `VPN_OVPN_CONFIG` points to your OpenVPN config path.
- `VPN_DEFAULTS_FILE` provides local fallback defaults for DocDB host/port and split tunnel preference.

## Notes

- The VPN starter can run through local `scripts/vpn_start.sh` and does not depend on tool-flow/query-optimizer scripts.
- If `AUTO_START_VPN=true` (default), CLI startup first ensures VPN is up, then service startup connects to Mongo immediately (ping on init).
- Service teardown still stops VPN only when that service call started the tunnel itself.
- Browser peak concurrency is persisted on the run document so crashy experiments still leave a useful number behind in Mongo.
