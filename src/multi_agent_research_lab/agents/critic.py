"""Optional critic agent for bonus work: flags unsupported claims before the answer ships."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a fact-checking critic. Given a final answer and its numbered sources, list any "
    "sentences that make a factual claim but have no [n] citation, and flag citations that "
    "point to a source number that does not exist. Be terse."
)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent (not wired into the default graph)."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate `state.final_answer` and append findings to `state.trace`."""

        if not state.final_answer:
            state.errors.append("critic skipped: no final_answer to review")
            return state

        cited = {int(n) for n in _CITATION_PATTERN.findall(state.final_answer)}
        invalid_citations = sorted(n for n in cited if n < 1 or n > len(state.sources))

        sources_block = "\n".join(f"[{i + 1}] {doc.title}" for i, doc in enumerate(state.sources))
        user_prompt = (
            f"Final answer:\n{state.final_answer}\n\nSources:\n{sources_block}\n\n"
            "List unsupported claims and bad citations."
        )
        response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "invalid_citations": invalid_citations,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic_run", {"invalid_citations": invalid_citations, "cost_usd": response.cost_usd}
        )
        return state
