Experiment ID: `exp-003-quality-path-optimization`
Experiment name: `Strong-source quality optimization`
Branch name: `exp/<timestamp>-quality-path-optimization`
Owner agent: `Codex`
Start timestamp: `TBD at run start`

This experiment is focused on one thing: raise the count of genuinely strong sources that also have clear recurring scraping paths, while pushing down noisy useful classifications.

Done here is not “high useful count.” Done means the ranking quality improves in a way we can trust: stronger sources with explicit path guidance go up, weak generic positives go down, and we can explain why from Mongo evidence.

Hard constraints:
- Query volume must not increase:
`STRATEGY_COUNT <= 10`, `QUERIES_PER_STRATEGY <= 5`, `SERP_PAGES_PER_QUERY <= 2`.
- No architecture shortcuts that bypass the existing pipeline stages.
- No silent failures. Any misleading “success” state is treated as a stop/fix/restart event.
- Quality metric is primary:
maximize `strong_clear_path_count` and reduce `noise_useful_count`.
- If a change improves headline counts but degrades path quality clarity, we reject that change.

Optimization metric definition:
- `strong_count`: useful=true, relevance>=0.8, source_evidence present, no decision/evaluation fallback error notes.
- `strong_clear_path_count`: strong_count subset where reasoning has a good recurring scraping path.
- `noise_useful_count`: useful_count - strong_clear_path_count.

Loop policy:
- Run repeated full experiments.
- Track run-over-run improvement.
- Stop when `strong_clear_path_count` no longer improves for consecutive runs (`plateau_patiance` in loop config).

Delegation plan:
- Main agent: owns metric definition, guardrails, stop criteria, and final merges.
- Delegated workers: independent bug triage, prompt edits, and test updates.

Runtime mode:
- `communication_mode=no-email`
- `closure_mode=auto-close`
- `send_final_email=false`

