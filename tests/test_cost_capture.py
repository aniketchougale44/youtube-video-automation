"""Per-run spend must be recorded, attributed to a stage, and never able to break a run.

db.models.Cost and agents.schemas.common.StageCost existed from the initial schema with nothing
writing to them, which is why the README's cost-per-video table was a price-list estimate rather
than a measurement.
"""
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from agents.schemas.common import RunStatus
from db.crud import persist_costs
from db.models import Base, Cost, Run
from graph.nodes._helpers import log_and_trace
from tools import cost as cost_tool


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# --- collection scope ---


def test_recording_outside_a_scope_is_a_no_op():
    """Importing tools.cost must not change behaviour for scripts or tests that never opt in."""
    cost_tool.record("openai", units=1, unit_cost_usd=1.0)
    assert cost_tool.collected() == []


def test_entries_must_be_drained_inside_the_scope():
    """The mistake this guards: collecting() resets the ContextVar on exit, so draining after the
    `with` block returns nothing and every cost row is silently lost."""
    with cost_tool.collecting():
        cost_tool.record("openai", units=2, unit_cost_usd=0.5, stage="voiceover")
        inside = cost_tool.collected()
    outside = cost_tool.collected()

    assert len(inside) == 1
    assert outside == [], "draining after the scope must not appear to succeed"


def test_total_is_units_times_rate():
    with cost_tool.collecting():
        cost_tool.record("openai", units=1500 / 1_000_000, unit_cost_usd=15.0, stage="voiceover")
        entry = cost_tool.collected()[0]
    assert entry["total_usd"] == pytest.approx(0.0225)


# --- stage attribution ---


def test_log_and_trace_attributes_spend_to_its_stage():
    """Every node calls log_and_trace on entry, which is the single hook that makes attribution
    possible without threading a collector through every node signature."""
    with cost_tool.collecting():
        log_and_trace("voiceover", "start")
        cost_tool.record("edge", units=100, unit_cost_usd=0.0)
        log_and_trace("thumbnail", "start")
        cost_tool.record("openai", units=1, unit_cost_usd=0.04)
        stages = [e["stage"] for e in cost_tool.collected()]
    assert stages == ["voiceover", "thumbnail"]


# --- LLM pricing ---


def test_llm_input_and_output_are_priced_separately():
    """Rates differ by up to 5x, so averaging them would make stored totals untraceable."""
    with cost_tool.collecting():
        cost_tool.record_llm("anthropic", "claude-sonnet-4-5-20250929", 1_000_000, 1_000_000)
        entries = cost_tool.collected()
    kinds = {e["metadata"]["kind"]: e["total_usd"] for e in entries}
    assert kinds["input_tokens"] == pytest.approx(3.0)
    assert kinds["output_tokens"] == pytest.approx(15.0)


def test_an_unpriced_model_still_records_its_tokens():
    """A zero rate with real units is visible and fixable; a dropped row is neither."""
    with cost_tool.collecting():
        cost_tool.record_llm("mystery", "some-model-nobody-priced", 1000, 500)
        entries = cost_tool.collected()
    assert len(entries) == 2
    assert all(e["unit_cost_usd"] == 0.0 for e in entries)
    assert sum(e["units"] for e in entries) == pytest.approx(1500 / 1_000_000)


def test_zero_token_calls_record_nothing():
    with cost_tool.collecting():
        cost_tool.record_llm("groq", "llama-3.3-70b", 0, 0)
        assert cost_tool.collected() == []


# --- persistence ---


def test_costs_are_written_to_postgres_shaped_rows(db):
    run = Run(id=uuid.uuid4(), status=RunStatus.PENDING, langgraph_thread_id=str(uuid.uuid4()))
    db.add(run)
    db.commit()

    with cost_tool.collecting():
        log_and_trace("script_writer", "start")
        cost_tool.record_llm("anthropic", "claude-sonnet-4-5", 10_000, 2_000)
        entries = cost_tool.collected()

    written = persist_costs(db, run.id, entries)
    rows = db.execute(select(Cost)).scalars().all()

    assert written == 2
    assert {r.stage.value for r in rows} == {"script_writer"}
    assert sum(r.total_usd for r in rows) == pytest.approx(10_000 / 1e6 * 3.0 + 2_000 / 1e6 * 15.0)


def test_an_unknown_stage_is_dropped_not_raised(db):
    """`stage` is an enum column. A cost recorded outside any node has no valid stage, and losing
    that row must not take down the task that just produced a video."""
    run = Run(id=uuid.uuid4(), status=RunStatus.PENDING, langgraph_thread_id=str(uuid.uuid4()))
    db.add(run)
    db.commit()

    written = persist_costs(db, run.id, [
        {"stage": "", "provider": "openai", "units": 1, "unit_cost_usd": 1.0, "total_usd": 1.0},
        {"stage": "not_a_real_stage", "provider": "openai", "units": 1, "unit_cost_usd": 1.0, "total_usd": 1.0},
        {"stage": "upload", "provider": "youtube", "units": 1, "unit_cost_usd": 0.0, "total_usd": 0.0},
    ])
    assert written == 1
    assert db.execute(select(Cost)).scalars().one().stage.value == "upload"
