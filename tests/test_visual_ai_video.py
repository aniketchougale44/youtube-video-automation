"""Focused tests for AI_VIDEO beat routing in asset_visual_node: the per-run cap is enforced in
beat order regardless of scene order, a failed generation falls back through the chain and then to
character animation rather than crashing the run, and the feature stays fully inert while
settings.enable_ai_video_beats is off even if a scene somehow carries AI_VIDEO.
"""
from agents.schemas.visual import (
    AssetOutput,
    AssetType,
    SceneType,
    SceneVisualPlan,
    VisualPlanOutput,
)
from core.settings import get_settings
from graph.nodes.visual import asset_visual_node
from graph.state import initial_state

# _try_ai_video walks these in order (provider="auto"); every one hits a real module by default,
# so tests must stub the whole set or a dead Colab/fal call leaks out over the network.
_GENERATOR_ATTRS = {
    "colab": "colab_video_tool",
    "fal": "fal_video_tool",
    "hf": "hf_video_tool",
    "nvidia": "nvidia_video_tool",
    "veo": "veo_video_tool",
}


def _stub_generators(monkeypatch, **overrides):
    """Point every AI-video generator at a raising stub, then apply per-name overrides
    (colab/fal/hf/nvidia/veo -> callable(prompt, output_path, **kwargs))."""
    def _raise(*_a, **_k):
        raise RuntimeError("generator not configured (test stub)")

    for name, attr in _GENERATOR_ATTRS.items():
        monkeypatch.setattr(f"graph.nodes.visual.{attr}.generate_video", overrides.get(name, _raise))


def _state_with_scenes(scenes: list[SceneVisualPlan], run_id: str = "run-ai-video-test") -> dict:
    state = initial_state(run_id=run_id, thread_id="t1")
    state["visual_plan"] = VisualPlanOutput(scenes=scenes).model_dump(mode="json")
    return state


def _writes_file(calls: list | None = None):
    def _fake(prompt, output_path, **kwargs):
        if calls is not None:
            calls.append(output_path)
        with open(output_path, "wb") as f:
            f.write(b"\x00")
        return output_path

    return _fake


def test_ai_video_respects_per_run_cap_in_beat_order(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "ai_video_provider", "auto")
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 1)

    calls: list = []
    _stub_generators(monkeypatch, colab=_writes_file(calls))

    # scenes deliberately out of beat-index order -- the cap must key off beat_index, not list order
    scenes = [
        SceneVisualPlan(beat_index=2, scene_type=SceneType.AI_VIDEO, description="c", ai_image_prompt="p2"),
        SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="a", ai_image_prompt="p0"),
        SceneVisualPlan(beat_index=1, scene_type=SceneType.AI_VIDEO, description="b", ai_image_prompt="p1"),
    ]
    result = asset_visual_node(_state_with_scenes(scenes))

    output = AssetOutput.model_validate(result["asset_output"])
    by_beat = {a.beat_index: a for a in output.assets}

    assert by_beat[0].asset_type == AssetType.AI_VIDEO
    assert by_beat[1].asset_type == AssetType.CHARACTER_ANIMATION
    assert by_beat[2].asset_type == AssetType.CHARACTER_ANIMATION
    assert calls == [by_beat[0].local_path]  # generator only ever invoked for the one budgeted beat


def test_ai_video_falls_through_chain_to_veo_when_earlier_generators_fail(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "ai_video_provider", "auto")
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 5)

    # colab/fal/hf/nvidia all raise (default stub), veo succeeds
    _stub_generators(monkeypatch, veo=_writes_file())

    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="a", ai_image_prompt="p0")]
    result = asset_visual_node(_state_with_scenes(scenes))

    output = AssetOutput.model_validate(result["asset_output"])
    assert output.assets[0].asset_type == AssetType.AI_VIDEO
    assert output.assets[0].source == "veo"


def test_ai_video_failure_falls_back_to_character_animation(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "ai_video_provider", "auto")
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 5)

    _stub_generators(monkeypatch)  # every generator raises

    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="a", ai_image_prompt="p0")]
    result = asset_visual_node(_state_with_scenes(scenes))

    output = AssetOutput.model_validate(result["asset_output"])
    assert output.assets[0].asset_type == AssetType.CHARACTER_ANIMATION


def test_ai_video_provider_colab_never_touches_a_paid_generator(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 5)
    monkeypatch.setattr(settings, "ai_video_provider", "colab")

    def _forbidden(*_a, **_k):
        raise AssertionError("provider=colab must not call a hosted/paid generator")

    _stub_generators(monkeypatch, fal=_forbidden, hf=_forbidden, nvidia=_forbidden, veo=_forbidden)

    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="a", ai_image_prompt="p0")]
    result = asset_visual_node(_state_with_scenes(scenes))  # colab stub raises -> mascot fallback

    output = AssetOutput.model_validate(result["asset_output"])
    assert output.assets[0].asset_type == AssetType.CHARACTER_ANIMATION


def test_ai_video_never_calls_generators_when_disabled(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", False)
    # 0 is the "no cap" sentinel, so this pins the one combination where a sloppy budget
    # calculation would invert the kill switch into "generate every beat".
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 0)

    def _unexpected(*_a, **_k):
        raise AssertionError("no video generator must be called while the feature is disabled")

    _stub_generators(
        monkeypatch, colab=_unexpected, fal=_unexpected, hf=_unexpected, nvidia=_unexpected, veo=_unexpected
    )

    # a scene carrying AI_VIDEO despite the flag being off (e.g. a stale/replayed plan) must still
    # be fully inert -- asset_visual_node's own budget=0 guard is what's under test here
    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="a", ai_image_prompt="p0")]
    result = asset_visual_node(_state_with_scenes(scenes))

    output = AssetOutput.model_validate(result["asset_output"])
    assert output.assets[0].asset_type == AssetType.CHARACTER_ANIMATION


def test_ai_video_cap_of_zero_means_every_beat(monkeypatch):
    """max_ai_video_beats_per_run=0 is "no cap", not "no beats" -- the free Colab generator costs
    time rather than money, so an animated story wants a real clip on every beat."""
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "ai_video_provider", "colab")
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 0)

    calls: list = []
    _stub_generators(monkeypatch, colab=_writes_file(calls))

    scenes = [
        SceneVisualPlan(beat_index=i, scene_type=SceneType.AI_VIDEO, description=f"b{i}", ai_image_prompt=f"p{i}")
        for i in range(5)
    ]
    result = asset_visual_node(_state_with_scenes(scenes))

    output = AssetOutput.model_validate(result["asset_output"])
    assert [a.asset_type for a in output.assets] == [AssetType.AI_VIDEO] * 5
    assert len(calls) == 5


def test_visual_planning_prompt_switches_to_ai_video_first_when_uncapped(monkeypatch):
    """The capped prompt tells the LLM to reserve AI_VIDEO for one standout beat, which is the
    opposite of what an uncapped run wants -- the two must not share wording."""
    from graph.nodes.visual import _visual_planning_system_prompt

    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)

    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 0)
    uncapped = _visual_planning_system_prompt()
    assert "DEFAULT choice" in uncapped
    assert "standout" not in uncapped

    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 2)
    capped = _visual_planning_system_prompt()
    assert "at most 2 beat(s)" in capped
    assert "standout" in capped

    monkeypatch.setattr(settings, "enable_ai_video_beats", False)
    assert "AI_VIDEO" not in _visual_planning_system_prompt()
