"""Analyst agent: turns research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a critical analyst. Given research notes and their sources, extract the key "
    "claims, compare viewpoints where sources disagree, and flag any claim with weak or "
    "single-source evidence. Keep citations ([n]) intact."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        sources_block = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url or 'no url'})" for i, doc in enumerate(state.sources)
        )
        user_prompt = (
            f"Research query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Produce: (1) key claims list, (2) points of agreement/disagreement across "
            "sources, (3) weak-evidence flags."
        )
        response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst_run", {"cost_usd": response.cost_usd})
        return state
