#!/usr/bin/env python3
"""Run the benchmark queries from `configs/lab_default.yaml` through both the single-agent
baseline and the multi-agent workflow, then write `reports/benchmark_report.md`.

Usage:
    python scripts/run_benchmark.py
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent_research_lab.core.schemas import ResearchQuery  # noqa: E402
from multi_agent_research_lab.core.state import ResearchState  # noqa: E402
from multi_agent_research_lab.evaluation.benchmark import run_benchmark  # noqa: E402
from multi_agent_research_lab.evaluation.report import render_markdown_report  # noqa: E402
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow  # noqa: E402
from multi_agent_research_lab.observability.logging import configure_logging  # noqa: E402
from multi_agent_research_lab.services.llm_client import LLMClient  # noqa: E402
from multi_agent_research_lab.services.storage import LocalArtifactStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_queries() -> list[str]:
    config = yaml.safe_load((REPO_ROOT / "configs" / "lab_default.yaml").read_text())
    return config["benchmark"]["queries"]


def _run_baseline(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a single research assistant responsible for the entire task: research the "
        "query, reason about the evidence, and write a clear final answer for the given "
        "audience, all in one pass."
    )
    user_prompt = f"Research query: {query}\nAudience: {request.audience}\n\nAnswer fully."
    response = llm.complete(system_prompt, user_prompt)
    state.final_answer = response.content
    state.route_history = ["baseline"]
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult

    state.agent_results.append(
        AgentResult(
            agent=AgentName.BASELINE,
            content=response.content,
            metadata={"cost_usd": response.cost_usd},
        )
    )
    return state


def _run_multi_agent(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def main() -> None:
    configure_logging("WARNING")
    queries = _load_queries()
    store = LocalArtifactStore(root=REPO_ROOT / "reports")

    all_metrics = []
    examples = []
    trace_events = {}

    for i, query in enumerate(queries):
        baseline_state, baseline_metrics = run_benchmark(f"baseline-{i}", query, _run_baseline)
        multi_state, multi_metrics = run_benchmark(f"multi-agent-{i}", query, _run_multi_agent)

        all_metrics += [baseline_metrics, multi_metrics]
        examples += [
            {
                "run_name": f"baseline-{i}",
                "query": query,
                "answer": baseline_state.final_answer or "(no answer)",
            },
            {
                "run_name": f"multi-agent-{i}",
                "query": query,
                "answer": multi_state.final_answer or "(no answer)",
            },
        ]
        trace_events[f"multi-agent-{i}"] = {
            "query": query,
            "route_history": multi_state.route_history,
            "trace": multi_state.trace,
            "errors": multi_state.errors,
        }

    trace_path = store.write_text(
        "trace_export.json", json.dumps(trace_events, indent=2, default=str)
    )
    report_md = render_markdown_report(
        all_metrics,
        examples=examples,
        trace_links=[f"Local trace export: {trace_path.relative_to(REPO_ROOT)}"],
    )
    report_path = store.write_text("benchmark_report.md", report_md)
    print(f"Wrote {report_path}")
    print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
