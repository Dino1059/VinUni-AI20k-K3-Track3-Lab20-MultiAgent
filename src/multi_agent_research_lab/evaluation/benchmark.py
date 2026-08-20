"""Benchmark utilities for comparing single-agent vs multi-agent runs.

`quality_score` is intentionally left to peer review (see docs/peer_review_rubric.md) since
judging answer quality automatically is out of scope for this skeleton — it can be filled in
after a human (or an LLM-as-judge, as a stretch goal) scores the transcripts.
"""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query through `runner`, measuring latency/cost/citation coverage/failures."""

    started = perf_counter()
    try:
        state = runner(query)
        failed = state.final_answer is None
    except Exception as exc:  # noqa: BLE001 - a failed run is a valid benchmark outcome
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Runner raised: {exc}",
        )
        empty_state = ResearchState(request=ResearchQuery(query=query))
        empty_state.errors.append(str(exc))
        return empty_state, metrics
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes=f"{len(state.errors)} agent error(s) recovered" if state.errors else "",
    )
    return state, metrics


def _estimate_cost(state: ResearchState) -> float | None:
    costs: list[float] = [
        result.metadata["cost_usd"]
        for result in state.agent_results
        if result.metadata.get("cost_usd") is not None
    ]
    if not costs:
        return None
    return sum(costs)


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of available sources that are actually cited ([n]) in the final answer."""

    if not state.sources or not state.final_answer:
        return None
    cited = {int(n) for n in _CITATION_PATTERN.findall(state.final_answer)}
    valid_cited = {n for n in cited if 1 <= n <= len(state.sources)}
    return min(len(valid_cited) / len(state.sources), 1.0)
