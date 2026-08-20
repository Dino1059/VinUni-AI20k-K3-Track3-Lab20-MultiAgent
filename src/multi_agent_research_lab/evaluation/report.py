"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    examples: list[dict[str, str]] | None = None,
    trace_links: list[str] | None = None,
) -> str:
    """Render benchmark metrics (and optional examples/trace links) to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines += _render_analysis(metrics)

    if examples:
        lines += ["", "## Example outputs", ""]
        for example in examples:
            lines.append(f"### {example.get('run_name', 'run')} — {example.get('query', '')}")
            lines.append("")
            lines.append("```text")
            lines.append(example.get("answer", "")[:2000])
            lines.append("```")
            lines.append("")

    if trace_links:
        lines += ["## Trace links / screenshots", ""]
        lines += [f"- {link}" for link in trace_links]
        lines.append("")

    lines += [
        "## Failure modes observed",
        "",
        "TODO(student): describe at least one failure mode you hit (e.g. supervisor loop, "
        "missing citations, search timeout) and how you fixed it.",
        "",
    ]

    return "\n".join(lines) + "\n"


def _render_analysis(metrics: list[BenchmarkMetrics]) -> list[str]:
    """Compare the two most recent runs (assumed baseline vs multi-agent) if present."""

    if len(metrics) < 2:
        return []
    baseline, multi = metrics[0], metrics[-1]
    lines = ["", "## Analysis", ""]
    latency_delta = multi.latency_seconds - baseline.latency_seconds
    lines.append(
        f"- Latency: multi-agent ({multi.run_name}) was "
        f"{'slower' if latency_delta > 0 else 'faster'} than baseline ({baseline.run_name}) "
        f"by {abs(latency_delta):.2f}s."
    )
    if baseline.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
        cost_delta = multi.estimated_cost_usd - baseline.estimated_cost_usd
        lines.append(
            f"- Cost: multi-agent used ${abs(cost_delta):.4f} "
            f"{'more' if cost_delta > 0 else 'less'} than baseline "
            f"(multiple LLM calls vs one)."
        )
    if multi.citation_coverage is not None:
        lines.append(f"- Citation coverage (multi-agent): {multi.citation_coverage:.0%}.")
    lines.append("")
    return lines
