"""Supervisor / router: decides which worker runs next, and when to stop."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

# Retry budget per worker before the supervisor gives up on it and forces the next stage.
_MAX_RETRIES_PER_STAGE = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history`.

        Routing policy (in priority order):
        1. If max_iterations reached -> "done" (guardrail against infinite loops).
        2. If a worker has failed repeatedly (see `state.errors`) -> skip it, move on
           with whatever partial data exists, so one flaky agent can't block the run.
        3. Missing sources/research notes -> "researcher".
        4. Missing analysis notes -> "analyst".
        5. Missing final answer -> "writer".
        6. Otherwise -> "done".
        """

        settings = get_settings()
        next_route = self._decide(state, settings.max_iterations)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {"route": next_route, "iteration": state.iteration},
        )
        return state

    def _decide(self, state: ResearchState, max_iterations: int) -> str:
        if state.iteration >= max_iterations:
            return "done"

        if state.final_answer:
            return "done"

        if not state.research_notes or not state.sources:
            if self._stage_failed_too_often(state, "researcher"):
                return "analyst" if not state.analysis_notes else "writer"
            return "researcher"

        if not state.analysis_notes:
            if self._stage_failed_too_often(state, "analyst"):
                return "writer"
            return "analyst"

        if not state.final_answer:
            return "writer"

        return "done"

    def _stage_failed_too_often(self, state: ResearchState, stage: str) -> bool:
        failures = sum(1 for error in state.errors if error.startswith(f"{stage} "))
        return failures >= _MAX_RETRIES_PER_STAGE
