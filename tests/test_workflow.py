"""End-to-end test for the LangGraph multi-agent workflow, using fake clients so it never
hits a real network/API regardless of what's in `.env`.
"""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self._content, input_tokens=10, output_tokens=5, cost_usd=0.0)


class FakeSearchClient:
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [SourceDocument(title="Fake Source", url="https://example.com/1", snippet="snippet")]


def test_workflow_runs_supervisor_researcher_analyst_writer_in_order() -> None:
    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(),
        researcher=ResearcherAgent(
            search_client=FakeSearchClient(), llm_client=FakeLLMClient("research notes [1]")
        ),
        analyst=AnalystAgent(llm_client=FakeLLMClient("analysis notes")),
        writer=WriterAgent(llm_client=FakeLLMClient("final answer [1]\n\nSources:\n[1] ...")),
    )

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    assert result.final_answer is not None
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert len(result.sources) == 1
    assert result.errors == []


def test_workflow_recovers_from_a_failing_worker() -> None:
    class BoomLLMClient:
        def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
            raise RuntimeError("provider outage")

    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(),
        researcher=ResearcherAgent(search_client=FakeSearchClient(), llm_client=BoomLLMClient()),
        analyst=AnalystAgent(llm_client=FakeLLMClient("analysis notes")),
        writer=WriterAgent(llm_client=FakeLLMClient("final answer")),
    )

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    # Researcher keeps failing; after 2 retries the supervisor's failure-guard routes
    # around it (analyst -> writer) instead of looping forever, and still produces an
    # answer from whatever partial state exists.
    assert result.route_history == ["researcher", "researcher", "analyst", "writer", "done"]
    assert result.final_answer == "final answer"
    assert sum(1 for error in result.errors if "researcher failed" in error) == 2
    assert result.iteration <= 6
