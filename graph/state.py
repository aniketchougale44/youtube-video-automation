"""
Shared LangGraph state for the AutoTube AI publish pipeline.

Every worker/critic node reads from and writes partial updates to this TypedDict. Agent
outputs are stored as plain dicts (via `Model.model_dump(mode="json")`) rather than Pydantic
instances so the state stays JSON-serializable for the checkpointer.
"""
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    run_id: str
    thread_id: str

    # --- injected by the caller (worker/tasks.py) before start_run(), from Postgres ---
    past_performance_summary: str | None
    # numeric signal from the last feedback run (LearningUpdate.strategy_weight_adjustments),
    # e.g. {"evergreen_bias": 0.05}; strategy_node folds it into its prompt
    strategy_weight_adjustments: dict[str, float] | None

    # --- stage outputs (dicts matching agents.schemas.*) ---
    trend_output: dict | None
    strategy_decision: dict | None
    script_output: dict | None
    script_qa_result: dict | None
    fact_check_output: dict | None
    visual_plan: dict | None
    asset_output: dict | None
    voiceover_output: dict | None
    assembly_output: dict | None
    metadata_output: dict | None
    thumbnail_output: dict | None
    compliance_result: dict | None
    approval_decision: dict | None
    upload_result: dict | None

    # --- control flow ---
    retry_counts: dict[str, int]                       # stage name -> attempts so far
    max_retries: int
    escalated: bool
    escalation_reason: str | None

    # --- observability / accumulators (reducer = append across the whole run) ---
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]

    # --- test/demo hook: {"critic_script_qa": 2} forces that critic to reject the first N attempts ---
    debug_force_reject: dict[str, int]


def initial_state(
    run_id: str,
    thread_id: str,
    max_retries: int = 3,
    debug_force_reject: dict | None = None,
    past_performance_summary: str | None = None,
    strategy_weight_adjustments: dict[str, float] | None = None,
) -> PipelineState:
    return PipelineState(
        run_id=run_id,
        thread_id=thread_id,
        past_performance_summary=past_performance_summary,
        strategy_weight_adjustments=strategy_weight_adjustments or {},
        retry_counts={},
        max_retries=max_retries,
        escalated=False,
        escalation_reason=None,
        trace=[],
        errors=[],
        debug_force_reject=debug_force_reject or {},
    )
