# Internet Explorer

`internet-explorer` is a fresh orchestration service for discovering new datasources for a given intent.

It reuses adjacent repos instead of copying code:
- `../query_optimizer_repo`: Google Custom Search client
- `../eu-swarm`: browser-agent delegation
- `../tool-flow`: tool baseline inventory + optional VPN script hints

## What it does

1. Generate a fixed set of strategies for an intent.
2. Generate Google-only queries for each strategy.
3. Fetch two SERP pages per query.
4. Canonicalize and deduplicate URLs.
5. Check novelty against a baseline domain file.
6. Build an initial link seed once from `sitemap.xml`, `robots.txt`, and `llms.txt`.
7. Pass that initial link list to browser delegation at start; do not maintain a mutable site map in-memory.
8. Use `eu-swarm` only for structured browser plan output.
9. Execute browser exploration with native `browser-use`, which can navigate to new URLs mid-run.
10. Decide whether the source is useful and store the result in Mongo.
11. Log every phase into append-only events.

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
- `MAX_SITE_GRAPH_FRONTIER` controls how many initial seeded links are passed into browser delegation.
- The generic VPN starter reuses `../query_optimizer_repo/client-config-staging.ovpn`.
- If `VPN_DOCDB_HOST` is empty, the loader infers the DocDB host default from the discovered `../tool-flow/scripts/vpn_and_run_*.sh` scripts.

## Notes

- The discovered `tool-flow` VPN scripts are used as reference material for defaults. The generic starter in this repo runs `openvpn` directly and does not execute those task-specific wrappers.
- If `AUTO_START_VPN=true` (default), CLI startup first ensures VPN is up, then service startup connects to Mongo immediately (ping on init).
- Service teardown still stops VPN only when that service call started the tunnel itself.
- Browser peak concurrency is persisted on the run document so crashy experiments still leave a useful number behind in Mongo.
