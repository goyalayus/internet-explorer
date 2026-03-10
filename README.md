# Internet Explorer

`internet-explorer` is a fresh orchestration service for discovering new datasources for a given intent.

It reuses adjacent repos instead of copying code:
- `../query_optimizer_repo`: Google Custom Search client
- `../eu-swarm`: browser-agent delegation
- `../tool-flow`: Mongo/VPN/env source of truth

## What it does

1. Generate a fixed set of strategies for an intent.
2. Generate Google-only queries for each strategy.
3. Fetch two SERP pages per query.
4. Canonicalize and deduplicate URLs.
5. Check novelty against a baseline domain file.
6. Build a shared site graph from `sitemap.xml`, `robots.txt`, `llms.txt`, and discovered internal links.
7. Traverse the highest-priority unvisited pages instead of blindly following homepage links.
8. Store compact page summaries and signals on site-graph nodes so both agents share state.
9. Delegate dynamic sites to the `eu-swarm` browser agent, which can also read and update the shared site graph.
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
- If `MONGODB_URI` is empty, config falls back to `../tool-flow/.env -> MONGODB_CREDENTIALS_DEV`.
- If Azure OpenAI env vars are empty, config also falls back to `../tool-flow/.env`.
- `MAX_BROWSER_CONCURRENCY=0` means unlimited.
- `MAX_URL_CONCURRENCY=0` means unlimited.
- `MAX_SITE_GRAPH_VISITS` controls how many pages the normal evaluator can analyze per site.
- `MAX_SITE_GRAPH_NODES` caps stored site-graph size per candidate.
- `MAX_SITE_GRAPH_FRONTIER` controls how many high-priority unvisited nodes are exposed at once.
- The generic VPN starter reuses `../query_optimizer_repo/client-config-staging.ovpn`.
- If `VPN_DOCDB_HOST` is empty, the loader infers the DocDB host default from the discovered `../tool-flow/scripts/vpn_and_run_*.sh` scripts.

## Notes

- The discovered `tool-flow` VPN scripts are used as reference material for defaults. The generic starter in this repo runs `openvpn` directly and does not execute those task-specific wrappers.
- If `AUTO_START_VPN=true`, the service will bring up the tunnel at run start and tear it down at the end only if it started that tunnel itself.
- Browser peak concurrency is persisted on the run document so crashy experiments still leave a useful number behind in Mongo.
