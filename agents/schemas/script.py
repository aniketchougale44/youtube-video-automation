"""Script Writer Agent, Script QA Critic, and Fact-Check Agent I/O."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agents.schemas.common import ContentFormat, PipelineStage, utcnow
from agents.schemas.research import StrategyDecision


class BeatType(StrEnum):
    HOOK = "hook"
    STORY = "story"
    DATA_POINT = "data_point"
    TRANSITION = "transition"
    CTA = "cta"


class ScriptBeat(BaseModel):
    index: int
    timestamp_seconds: float
    beat_type: BeatType
    voiceover_text: str
    visual_cue: str


class ScriptWriterInput(BaseModel):
    strategy: StrategyDecision
    feedback_from_critic: str | None = None
    revision_number: int = 0


class ScriptOutput(BaseModel):
    working_title: str
    hook: str
    beats: list[ScriptBeat]
    cta: str
    full_voiceover_text: str
    target_length_seconds: int
    content_format: ContentFormat
    revision_number: int = 0
    written_at: datetime = Field(default_factory=utcnow)


class SimilarityMatch(BaseModel):
    source_video_id: str
    similarity_score: float = Field(ge=0, le=1)
    matched_snippet: str = ""


class ClaimFlag(BaseModel):
    beat_index: int
    claim_text: str
    reason: str


class ScriptQAResult(BaseModel):
    stage: PipelineStage = PipelineStage.CRITIC_SCRIPT_QA
    originality_score: float = Field(ge=0, le=10)
    max_similarity_score: float = Field(ge=0, le=1)
    similarity_matches: list[SimilarityMatch] = Field(default_factory=list)
    flagged_claims: list[ClaimFlag] = Field(default_factory=list)
    policy_flags: list[str] = Field(default_factory=list)
    passed: bool
    feedback_for_retry: str | None = None
    retry_count: int = 0
    evaluated_at: datetime = Field(default_factory=utcnow)


class ClaimVerdict(StrEnum):
    VERIFIED = "verified"
    FALSE = "false"
    UNVERIFIED = "unverified"
    UNCERTAIN = "uncertain"


class ClaimVerification(BaseModel):
    claim_text: str
    beat_index: int
    verdict: ClaimVerdict
    confidence_score: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)


class FactCheckOutput(BaseModel):
    verifications: list[ClaimVerification]
    checked_at: datetime = Field(default_factory=utcnow)
