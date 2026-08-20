"""Unit tests for SupervisorAgent's routing policy.

Replaces the original skeleton-guard test (which just asserted `StudentTodoError` was
raised) now that `SupervisorAgent.run` is implemented.
"""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_when_no_sources() -> None:
    state = SupervisorAgent().run(_state())
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_supervisor_routes_to_analyst_once_research_notes_exist() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "some notes"

    state = SupervisorAgent().run(state)

    assert state.route_history == ["analyst"]


def test_supervisor_routes_to_writer_once_analysis_exists() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"

    state = SupervisorAgent().run(state)

    assert state.route_history == ["writer"]


def test_supervisor_routes_to_done_once_final_answer_exists() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"

    state = SupervisorAgent().run(state)

    assert state.route_history == ["done"]


def test_supervisor_stops_at_max_iterations_guardrail() -> None:
    state = _state()
    for _ in range(6):
        state.record_route("researcher")

    state = SupervisorAgent().run(state)

    assert state.route_history[-1] == "done"


def test_supervisor_skips_stuck_researcher_after_repeated_failures() -> None:
    state = _state()
    state.errors = ["researcher failed: boom", "researcher failed: boom again"]

    state = SupervisorAgent().run(state)

    assert state.route_history == ["analyst"]
