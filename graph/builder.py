"""
The main publish pipeline: supervisor graph wiring every worker + critic node from
graph/nodes/*.py. Two gated loops (Script QA, Compliance) each retry up to `max_retries`
before escalating to a human. The human-approval gate is either a real interrupt() or an
auto-approve passthrough, chosen at build time from REQUIRE_HUMAN_APPROVAL.

    START
      -> trend_research -> strategy -> script_writer -> critic_script_qa
           (reject, retries left)  -> back to script_writer
           (reject, retries used)  -> mark_escalation_script_qa -> escalate -> END
           (pass)                  -> fact_check -> visual_planning -> asset_visual
      -> voiceover -> video_assembly -> metadata_seo -> thumbnail -> critic_compliance
           (reject, retries left)  -> back to metadata_seo
           (reject, retries used)  -> mark_escalation_compliance -> escalate -> END
           (pass)                  -> human_approval (interrupt) | auto_approve
      -> [approved]   -> upload -> END
      -> [rejected]   -> rejected -> END
"""
from langgraph.graph import END, START, StateGraph

from core.settings import get_settings
from graph.nodes._helpers import current_retry_count, log_and_trace
from graph.nodes.audio_render import video_assembly_node, voiceover_node
from graph.nodes.escalation import escalate_node
from graph.nodes.publish import (
    auto_approve_node,
    critic_compliance_node,
    human_approval_node,
    metadata_seo_node,
    thumbnail_node,
    upload_node,
)
from graph.nodes.research import strategy_node, trend_research_node
from graph.nodes.script import critic_script_qa_node, fact_check_node, script_writer_node
from graph.nodes.visual import asset_visual_node, visual_planning_node
from graph.state import PipelineState

STAGE_SCRIPT_QA = "critic_script_qa"
STAGE_COMPLIANCE = "critic_compliance"


def _mark_escalation(stage: str, max_retries_key: str = "max_retries"):
    def node(state: PipelineState) -> dict:
        reason = f"{stage} rejected {current_retry_count(state, stage)} times, exceeding max_retries"
        return {"escalation_reason": reason, "trace": [log_and_trace(stage, "max_retries_exceeded", reason=reason)]}

    return node


def _rejected_node(state: PipelineState) -> dict:
    return {"trace": [log_and_trace("human_approval", "rejected_by_reviewer", run_id=state.get("run_id"))]}


def route_script_qa(state: PipelineState) -> str:
    result = state["script_qa_result"]
    if result["passed"]:
        return "fact_check"
    if current_retry_count(state, STAGE_SCRIPT_QA) >= state.get("max_retries", 3):
        return "mark_escalation_script_qa"
    return "script_writer"


def route_compliance(state: PipelineState) -> str:
    result = state["compliance_result"]
    if result["passed"]:
        return "approval_gate"
    if current_retry_count(state, STAGE_COMPLIANCE) >= state.get("max_retries", 3):
        return "mark_escalation_compliance"
    return "metadata_seo"


def route_approval(state: PipelineState) -> str:
    decision = state["approval_decision"]
    return "upload" if decision["approved"] else "rejected"


def build_publish_graph(require_human_approval: bool | None = None) -> StateGraph:
    """Returns an unconmpiled StateGraph. Caller compiles with a checkpointer:
    `build_publish_graph().compile(checkpointer=...)`.
    """
    if require_human_approval is None:
        require_human_approval = get_settings().require_human_approval

    graph = StateGraph(PipelineState)

    graph.add_node("trend_research", trend_research_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("script_writer", script_writer_node)
    graph.add_node("critic_script_qa", critic_script_qa_node)
    graph.add_node("mark_escalation_script_qa", _mark_escalation(STAGE_SCRIPT_QA))
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("visual_planning", visual_planning_node)
    graph.add_node("asset_visual", asset_visual_node)
    graph.add_node("voiceover", voiceover_node)
    graph.add_node("video_assembly", video_assembly_node)
    graph.add_node("metadata_seo", metadata_seo_node)
    graph.add_node("thumbnail", thumbnail_node)
    graph.add_node("critic_compliance", critic_compliance_node)
    graph.add_node("mark_escalation_compliance", _mark_escalation(STAGE_COMPLIANCE))
    graph.add_node("approval_gate", human_approval_node if require_human_approval else auto_approve_node)
    graph.add_node("rejected", _rejected_node)
    graph.add_node("upload", upload_node)
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "trend_research")
    graph.add_edge("trend_research", "strategy")
    graph.add_edge("strategy", "script_writer")
    graph.add_edge("script_writer", "critic_script_qa")
    graph.add_conditional_edges(
        "critic_script_qa", route_script_qa,
        {"fact_check": "fact_check", "script_writer": "script_writer", "mark_escalation_script_qa": "mark_escalation_script_qa"},
    )
    graph.add_edge("mark_escalation_script_qa", "escalate")
    graph.add_edge("fact_check", "visual_planning")
    graph.add_edge("visual_planning", "asset_visual")
    graph.add_edge("asset_visual", "voiceover")
    graph.add_edge("voiceover", "video_assembly")
    graph.add_edge("video_assembly", "metadata_seo")
    graph.add_edge("metadata_seo", "thumbnail")
    graph.add_edge("thumbnail", "critic_compliance")
    graph.add_conditional_edges(
        "critic_compliance", route_compliance,
        {"approval_gate": "approval_gate", "metadata_seo": "metadata_seo", "mark_escalation_compliance": "mark_escalation_compliance"},
    )
    graph.add_edge("mark_escalation_compliance", "escalate")

    if require_human_approval:
        graph.add_conditional_edges("approval_gate", route_approval, {"upload": "upload", "rejected": "rejected"})
    else:
        graph.add_edge("approval_gate", "upload")  # auto_approve_node always approves

    graph.add_edge("upload", END)
    graph.add_edge("rejected", END)
    graph.add_edge("escalate", END)

    return graph
