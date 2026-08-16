"""Visual Planning Agent and Asset/Visual Agent I/O."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agents.schemas.common import utcnow
from agents.schemas.script import ScriptOutput


class SceneType(StrEnum):
    BROLL = "broll"
    AI_IMAGE = "ai_image"
    CHART = "chart"
    MOTION_GRAPHIC = "motion_graphic"


class SceneVisualPlan(BaseModel):
    beat_index: int
    scene_type: SceneType
    description: str
    search_keywords: list[str] = Field(default_factory=list)
    ai_image_prompt: str | None = None
    transition: str = "cut"


class VisualPlanningInput(BaseModel):
    script: ScriptOutput


class VisualPlanOutput(BaseModel):
    scenes: list[SceneVisualPlan]
    planned_at: datetime = Field(default_factory=utcnow)


class AssetType(StrEnum):
    STOCK_VIDEO = "stock_video"
    STOCK_IMAGE = "stock_image"
    AI_IMAGE = "ai_image"
    CHARACTER_ANIMATION = "character_animation"


class SourcedAsset(BaseModel):
    beat_index: int
    scene_type: SceneType
    asset_type: AssetType
    source: str = Field(description="e.g. 'pexels', 'pixabay', 'openai-image'")
    local_path: str | None = None
    source_url: str | None = None
    license: str = ""
    attribution_required: bool = False


class AssetSourcingInput(BaseModel):
    visual_plan: VisualPlanOutput


class AssetOutput(BaseModel):
    assets: list[SourcedAsset]
    sourced_at: datetime = Field(default_factory=utcnow)
