from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def _state_with_answer(answer: str, cost: float) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="test query"))
    state.sources = [
        SourceDocument(title="a", snippet="s"),
        SourceDocument(title="b", snippet="s"),
    ]
    state.final_answer = answer
    state.agent_results.append(
        AgentResult(agent=AgentName.WRITER, content=answer, metadata={"cost_usd": cost})
    )
    return state


def test_run_benchmark_computes_latency_cost_and_citation_coverage() -> None:
    state = _state_with_answer("answer with [1] citation", cost=0.01)
    _, metrics = run_benchmark("run", "test query", lambda _query: state)

    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd == 0.01
    assert metrics.citation_coverage == 0.5  # 1 of 2 sources cited
    assert metrics.failure_rate == 0.0


def test_run_benchmark_marks_failure_when_no_final_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="test query"))
    _, metrics = run_benchmark("run", "test query", lambda _query: state)

    assert metrics.failure_rate == 1.0
    assert metrics.citation_coverage is None


def test_run_benchmark_catches_runner_exceptions() -> None:
    def _boom(_query: str) -> ResearchState:
        raise RuntimeError("network down")

    _, metrics = run_benchmark("run", "test query", _boom)

    assert metrics.failure_rate == 1.0
    assert "network down" in metrics.notes
