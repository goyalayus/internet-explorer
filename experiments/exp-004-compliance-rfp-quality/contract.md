Experiment ID: `exp-004-compliance-rfp-quality`
Experiment name: `Compliance RFP quality optimization`
Branch name: `exp/20260325-151247-compliance-rfp-quality`
Owner agent: `Codex`
Start timestamp: `2026-03-25T15:12:47+05:30`

Intent:
`find compliance RFP websites for SOC 2, ISO 27001, audit, security compliance, and GRC services`

This experiment uses the same domain-homepage pipeline that worked for the data-annotation runs.
The goal is not to chase raw useful counts. The goal is to find genuinely strong compliance-RFP sources
with clear recurring extraction paths, while keeping noise low and catching any subsystem breakage or
agent stupidity we can see in traces.

Definition of done:
- We complete at least one clean end-to-end run for the compliance intent.
- The run reaches real final decisions without silent infra or persistence bugs.
- We inspect traces and logs for obvious model mistakes, tool misuse, and weak decision behavior.
- If we find a real bug or a clear agent-quality mistake, we stop, fix, and restart cleanly.
- We keep improving until the strong-source metric stops moving in a meaningful way.

Hard constraints:
- Query volume must not increase:
  `STRATEGY_COUNT <= 10`, `QUERIES_PER_STRATEGY <= 5`, `SERP_PAGES_PER_QUERY <= 2`.
- Use the existing architecture. Do not bypass strategy -> query -> result -> normal-agent evaluation -> optional browser delegation.
- No silent failures. If counters, summaries, or final decisions become misleading, treat that as a stop/fix/restart event.
- Primary metric stays quality-first:
  maximize `strong_clear_path_count` and reduce `noise_useful_count`.
- If a change raises positive counts but weakens path clarity or evidence quality, reject the change.

Optimization metric definition:
- `strong_count`: useful=true, relevance>=0.8, source_evidence present, no decision/evaluation fallback error notes.
- `strong_clear_path_count`: strong_count subset where reasoning has a good recurring scraping path.
- `noise_useful_count`: useful_count - strong_clear_path_count.

Loop policy:
- Run repeated full experiments for the compliance intent.
- Monitor run logs, Mongo summaries, and failure reasons while the run is active.
- Fix obvious bugs or obvious agent-quality mistakes when confidence is high.
- Restart after every real fix.
- Stop only when the strong-source metric plateaus and no new meaningful bugs are showing up.

Delegation plan:
- Main agent: owns the contract, guardrails, metric, and restart decisions.
- Delegated workers: independent log triage, trace review, and targeted bug-fix support when needed.

Runtime mode:
- `communication_mode=no-email`
- `closure_mode=auto-close`
- `send_final_email=false`
