"""Centralized configuration. Every module reads config from here, never from os.environ directly."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me"

    # Human-in-the-loop / quality gate
    require_human_approval: bool = True
    quality_score_threshold: float = 8.5
    max_critic_retries: int = 3
    originality_similarity_threshold: float = 0.82  # cosine similarity (0-1); separate scale from
    # quality_score_threshold (0-10) — don't conflate the two

    # Channel strategy
    channel_goals: str = (
        "Grow a general knowledge/educational YouTube channel with evergreen, "
        "highly re-watchable long-form content."
    )
    trend_region_code: str = "US"
    trend_category_ids: str = ""  # comma-separated YouTube category IDs; empty = no filter
    # comma-separated search queries, e.g. "nursery rhymes for kids,abc song for toddlers" --
    # supplements the generic regional mostPopular chart for niches it won't reliably surface
    trend_search_queries: str = ""
    max_trend_candidates: int = 5

    # Postgres
    database_url: str = "postgresql+psycopg://autotube:autotube@localhost:5432/autotube"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Scheduling
    pipeline_cron_schedule: str = "0 14 * * mon,wed,fri"

    # LLM (fallback order: anthropic -> openai -> groq -> gemini; each is skipped if its key is
    # unset, so a deployment can run on just the free tiers with no paid provider configured)
    anthropic_api_key: str = ""
    llm_primary_model: str = "claude-sonnet-4-5"
    openai_api_key: str = ""
    llm_fallback_model: str = "gpt-4o-mini"
    groq_api_key: str = ""  # free, no card required: console.groq.com
    llm_groq_model: str = "llama-3.3-70b-versatile"
    google_api_key: str = ""  # free, no card required: aistudio.google.com/apikey
    llm_gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    langsmith_api_key: str = ""
    langsmith_project: str = "autotube-ai"
    langsmith_tracing: bool = True

    # YouTube
    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    youtube_refresh_token: str = ""
    youtube_channel_id: str = ""
    youtube_api_key: str = ""
    youtube_daily_quota_units: int = 10_000

    # Research
    tavily_api_key: str = ""

    # TTS: "openai" (paid, needs openai_api_key) | anything else -> edge-tts (free, keyless,
    # Microsoft Edge neural voices, always available as a fallback even when tts_provider="openai"
    # but the key is unset/unfunded). elevenlabs/azure are reserved for a future provider.
    tts_provider: str = "edge"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    azure_speech_key: str = ""
    azure_speech_region: str = ""
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "alloy"
    edge_tts_voice: str = "en-US-AndrewNeural"

    # Stock media
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    # Image gen: "openai" (paid, needs openai_api_key) | anything else -> Pollinations.ai (free,
    # keyless, always available as a fallback even when image_gen_provider="openai" but the key
    # is unset/unfunded)
    image_gen_provider: str = "pollinations"
    openai_image_model: str = "gpt-image-1"
    stability_api_key: str = ""
    replicate_api_token: str = ""

    # Notifications
    slack_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alert_email_from: str = ""
    alert_email_to: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # Media (rendered assets/audio/video land under here, per-run subfolders)
    media_dir: str = "./media"

    # Dashboard
    dashboard_base_url: str = "http://localhost:8000"
    dashboard_basic_auth_user: str = "admin"
    dashboard_basic_auth_password: str = "change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
