"""Writer agent: synthesizes the final answer from research and analysis notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Synthesize the research notes and analysis into a clear, "
    "well-structured answer for the given audience. Preserve citation markers like [n] and "
    "end with a 'Sources' section listing each numbered source and its URL."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        sources_block = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url or 'no url'})" for i, doc in enumerate(state.sources)
        )
        user_prompt = (
            f"Research query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Write the final answer now, with inline [n] citations and a trailing Sources "
            "section."
        )
        response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("writer_run", {"cost_usd": response.cost_usd})
        return state
