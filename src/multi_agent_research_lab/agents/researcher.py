"""Researcher agent: gathers sources and writes research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Summarize the provided source snippets into "
    "concise research notes for an analyst to review next. Reference sources by their number "
    "in brackets, e.g. [1]. Do not invent facts beyond the given sources."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self._search_client.search(
            state.request.query, max_results=state.request.max_sources
        )
        state.sources = sources

        sources_block = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url or 'no url'}): {doc.snippet}"
            for i, doc in enumerate(sources)
        )
        user_prompt = (
            f"Research query: {state.request.query}\n\nSources:\n{sources_block}\n\n"
            "Write 3-6 bullet point research notes summarizing what these sources say, "
            "citing sources by [n]."
        )
        response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_run", {"source_count": len(sources), "cost_usd": response.cost_usd}
        )
        return state
