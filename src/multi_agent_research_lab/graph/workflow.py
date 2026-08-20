"""LangGraph workflow wiring Supervisor, Researcher, Analyst, and Writer."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()
        self._settings = get_settings()
        self._compiled: Any = None

    def build(self) -> object:
        """Create and compile the LangGraph graph.

        Nodes: supervisor (router), researcher, analyst, writer.
        Supervisor decides the next route each time it runs; workers always hand control
        back to the supervisor so it can re-evaluate state (handles retries and stop
        conditions). `max_iterations` is enforced inside SupervisorAgent itself.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._supervisor.run)
        graph.add_node("researcher", self._wrap_worker(self._researcher))  # type: ignore[arg-type]
        graph.add_node("analyst", self._wrap_worker(self._analyst))  # type: ignore[arg-type]
        graph.add_node("writer", self._wrap_worker(self._writer))  # type: ignore[arg-type]

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        self._compiled = graph.compile()
        return self._compiled

    def _wrap_worker(self, agent: BaseAgent) -> Callable[[ResearchState], ResearchState]:
        """Run a worker agent; on failure, record the error instead of crashing the graph.

        This lets the Supervisor's retry/fallback policy (see `SupervisorAgent`) react to
        repeated failures instead of the whole workflow dying on one bad LLM/search call.
        """

        def _run(state: ResearchState) -> ResearchState:
            with trace_span(f"agent.{agent.name}", {"query": state.request.query}) as span:
                try:
                    result = agent.run(state)
                except Exception as exc:  # noqa: BLE001 - guardrail boundary, not a bug swallow
                    state.errors.append(f"{agent.name} failed: {exc}")
                    state.add_trace_event("agent_error", {"agent": agent.name, "error": str(exc)})
                    return state
                span["attributes"]["ok"] = True
                return result

        return _run

    def _next_route(self, state: ResearchState) -> str:
        if not state.route_history:
            return "researcher"
        return state.route_history[-1]

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        if self._compiled is None:
            self.build()

        with trace_span("multi_agent_workflow.run", {"query": state.request.query}):
            result = self._compiled.invoke(
                state,
                config={"recursion_limit": self._settings.max_iterations * 4 + 10},
            )

        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
