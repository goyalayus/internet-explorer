from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_run_observation_report(
    *,
    run_doc: dict[str, Any],
    url_summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    phase_counts = Counter(str(event.get("phase", "")) for event in events)
    decision_counts = Counter(str(event.get("decision", "")) for event in events)
    error_counts = Counter(str(event.get("error_code", "")) for event in events if event.get("error_code"))

    by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in sorted(events, key=lambda item: int(item.get("step_no", 0))):
        url_id = str(event.get("url_id") or "")
        if url_id:
            by_url[url_id].append(event)

    lines: list[str] = []

    lines.append(f"run_id: {run_doc.get('run_id', '')}")
    lines.append(f"intent: {run_doc.get('intent', '')}")
    lines.append(f"status: {run_doc.get('status', '')}")
    lines.append(
        "counts: "
        f"raw_results={run_doc.get('raw_result_count', 0)} "
        f"unique_sources={run_doc.get('unique_source_count', run_doc.get('unique_url_count', 0))} "
        f"evaluated={run_doc.get('evaluated_source_count', run_doc.get('evaluated_url_count', 0))} "
        f"useful={run_doc.get('useful_source_count', run_doc.get('useful_url_count', 0))}"
    )
    lines.append(f"browser_peak_active: {run_doc.get('browser_peak_active', 0)}")
    lines.append("")

    lines.append("phase_counts:")
    for phase, count in phase_counts.most_common():
        lines.append(f"- {phase}: {count}")
    lines.append("")

    if error_counts:
        lines.append("error_counts:")
        for error_code, count in error_counts.most_common():
            lines.append(f"- {error_code}: {count}")
        lines.append("")

    lines.append("obvious_faults:")
    faults = _derive_faults(run_doc=run_doc, phase_counts=phase_counts, decision_counts=decision_counts, error_counts=error_counts, by_url=by_url)
    if faults:
        for fault in faults:
            lines.append(f"- {fault}")
    else:
        lines.append("- none detected by heuristics")
    lines.append("")

    lines.append("candidate_paths:")
    for summary in _ordered_summaries(url_summaries):
        url_id = str(summary.get("url_id"))
        lines.extend(_format_candidate_block(summary=summary, events=by_url.get(url_id, [])))
        lines.append("")

    return "\n".join(line.rstrip() for line in lines).strip() + "\n"


def _derive_faults(
    *,
    run_doc: dict[str, Any],
    phase_counts: Counter[str],
    decision_counts: Counter[str],
    error_counts: Counter[str],
    by_url: dict[str, list[dict[str, Any]]],
) -> list[str]:
    faults: list[str] = []
    fetched = phase_counts.get("page_fetch", 0)
    fetch_failed = decision_counts.get("fetch_failed", 0)
    if fetched and fetch_failed / max(fetched, 1) >= 0.25:
        dominant = ", ".join(f"{name}={count}" for name, count in error_counts.most_common(3))
        faults.append(f"High fetch-failure rate relative to page_fetch events. Dominant errors: {dominant or 'unknown'}.")

    if decision_counts.get("delegate_failed_fallback", 0) > 0:
        faults.append("Browser delegation fell back at least once; inspect browser_delegate raw output and browser_step traces.")

    if decision_counts.get("planner_fallback", 0) > 0:
        faults.append("Browser planner fallback happened; planner prompt or provider stability needs review.")

    unknown_count = sum(1 for events in by_url.values() if _final_decision(events) == "unknown")
    evaluated = int(run_doc.get("evaluated_source_count", run_doc.get("evaluated_url_count", 0)) or 0)
    if evaluated and unknown_count / max(evaluated, 1) >= 0.2:
        faults.append("Too many final `unknown` outcomes; final decision schema or evidence sufficiency is weak.")

    loop_candidates = _find_loop_candidates(by_url)
    if loop_candidates:
        faults.append(f"Repeated navigation targets suggest possible loops for: {', '.join(loop_candidates[:8])}.")

    return faults


def _ordered_summaries(url_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        url_summaries,
        key=lambda item: (
            not bool(item.get("useful")),
            str(item.get("outcome", "")) == "unknown",
            str(item.get("domain", "")),
        ),
    )


def _format_candidate_block(*, summary: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    header = (
        f"- {summary.get('url_id', '')} "
        f"domain={summary.get('domain', '')} "
        f"start={summary.get('start_url', summary.get('canonical_url', ''))} "
        f"outcome={summary.get('outcome', '')} "
        f"useful={summary.get('useful', False)}"
    )
    lines = [header]
    reasoning = str(summary.get("reasoning", "") or "").strip()
    if reasoning:
        lines.append(f"  reasoning: {reasoning[:500]}")
    evidence = summary.get("source_evidence") or []
    if evidence:
        lines.append("  evidence:")
        for item in evidence[:4]:
            kind = str(item.get("kind", ""))
            url = str(item.get("url", ""))
            desc = str(item.get("summary", "") or item.get("title", "")).strip()
            lines.append(f"  - {kind}: {url} {desc[:220]}".rstrip())
    trace_lines = _format_trace(events)
    if trace_lines:
        lines.append("  path:")
        lines.extend([f"  - {line}" for line in trace_lines[:18]])
    notes = summary.get("notes") or []
    if notes:
        lines.append(f"  notes: {', '.join(str(note) for note in notes[:8])}")
    return lines


def _format_trace(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        phase = str(event.get("phase", ""))
        decision = str(event.get("decision", ""))
        output = event.get("output_summary")
        if phase == "triage" and decision.startswith("navigation_"):
            target = ""
            if isinstance(output, dict):
                target = str(output.get("target_url", "")).strip()
                reason = str(output.get("reasoning", "")).strip()
                lines.append(f"{phase}:{decision} target={target or '-'} reason={reason[:160]}")
            else:
                lines.append(f"{phase}:{decision}")
            continue
        if phase == "page_fetch":
            if isinstance(output, dict):
                status = output.get("status_code")
                final_url = output.get("final_url", "")
                error = output.get("error", "")
                if final_url or status is not None:
                    lines.append(f"{phase}:{decision} status={status} url={final_url}")
                elif error:
                    lines.append(f"{phase}:{decision} error={str(error)[:160]}")
                else:
                    lines.append(f"{phase}:{decision}")
            else:
                lines.append(f"{phase}:{decision}")
            continue
        if phase in {"pdf_verify", "api_verify", "browser_delegate", "render_detect", "final_decision"}:
            if isinstance(output, dict):
                short = _compact_output(output)
                lines.append(f"{phase}:{decision} {short}".strip())
            else:
                lines.append(f"{phase}:{decision}")
            continue
        if phase == "browser_step":
            lines.append(f"{phase}:{decision}")
    return lines


def _compact_output(output: dict[str, Any]) -> str:
    interesting_keys = ("render_profile", "status_code", "url", "final_url", "content_type", "error", "active_browser_sessions")
    bits: list[str] = []
    for key in interesting_keys:
        value = output.get(key)
        if value not in (None, "", [], {}):
            bits.append(f"{key}={value}")
    if not bits and "reasoning" in output:
        bits.append(str(output.get("reasoning", ""))[:180])
    return " ".join(bits)[:240]


def _find_loop_candidates(by_url: dict[str, list[dict[str, Any]]]) -> list[str]:
    suspects: list[str] = []
    for url_id, events in by_url.items():
        targets: list[str] = []
        for event in events:
            if event.get("phase") != "triage":
                continue
            output = event.get("output_summary")
            if isinstance(output, dict):
                target = str(output.get("target_url", "")).strip()
                if target:
                    targets.append(target)
        repeated = [target for target, count in Counter(targets).items() if count >= 3]
        if repeated:
            suspects.append(url_id)
    return suspects


def _final_decision(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("phase") == "final_decision":
            return str(event.get("decision", ""))
    return ""
