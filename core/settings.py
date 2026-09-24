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
    # quality_score_threshold (0-10) — don't conflate the two. Applied to matches against
    # transcript_embeddings (real external/other-channel content) -- genuine plagiarism protection.
    originality_own_catalog_similarity_threshold: float = 0.95  # applied to matches against this
    # channel's OWN script_embeddings back-catalog instead. Deliberately much looser than the
    # external threshold: a narrow, format-consistent niche (e.g. nursery rhymes for toddlers)
    # legitimately produces similar-sounding scripts episode to episode -- that's genre convention,
    # not plagiarism -- so gating it at the strict external bar rejected every draft outright.

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

    # Scheduling -- 19:00 UTC = ~3pm US Eastern, timed to land a couple hours ahead of the
    # after-school/evening viewing peak for this channel's US kids/nursery audience (see
    # channel_goals, trend_region_code); Sat added alongside Mon/Wed/Fri since kids' content
    # over-indexes on weekend viewership.
    pipeline_cron_schedule: str = "0 19 * * mon,wed,fri,sat"

    # LLM (fallback order: anthropic -> openai -> groq -> gemini; each is skipped if its key is
    # unset, so a deployment can run on just the free tiers with no paid provider configured)
    anthropic_api_key: str = ""
    llm_primary_model: str = "claude-sonnet-4-5"
    openai_api_key: str = ""
    llm_fallback_model: str = "gpt-4o-mini"
    groq_api_key: str = ""  # free, no card required: console.groq.com
    llm_groq_model: str = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned
    google_api_key: str = ""  # free, no card required: aistudio.google.com/apikey
    llm_gemini_model: str = "gemini-3.6-flash"  # gemini-2.0-flash was retired; this is Google's
    # suggested replacement as of the retirement notice
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
    # Every upload is added to this playlist (created on the channel the first time it's needed,
    # then reused by exact title match) -- keeps nursery/preschool content grouped separately from
    # the rest of the channel.
    youtube_nursery_playlist_title: str = "Nursery Rhymes & Learning for Toddlers"
    # This channel's content is always made-for-kids (see channel_goals) -- explicitly self-declare
    # it on every upload rather than relying on YouTube's automatic classifier, which does not
    # consistently mark every upload as made-for-kids on its own.
    youtube_made_for_kids: bool = True
    # Privacy status every upload_node publish uses: "unlisted" (default -- link-only, safe for an
    # unattended first run), "private", or "public". Flip to "public" only once you trust the
    # unattended output. Change visibility later without re-uploading via youtube.set_visibility.
    youtube_upload_visibility: str = "unlisted"
    # PubSubHubbub / WebSub push notifications (api/routes/webhooks.py). YouTube's hub
    # (https://pubsubhubbub.appspot.com/) posts an Atom entry whenever a video on
    # youtube_channel_id is added, updated, or deleted -- near-real-time vs. the hourly
    # scheduler.check_performance_windows poll. Leave the token/secret blank to accept
    # unauthenticated callbacks (fine for a hidden path; set them once the endpoint is public).
    # Subscribe with: python -m tools.websub subscribe  (see tools/websub.py).
    youtube_websub_callback_url: str = ""  # public https URL of POST /api/webhooks/youtube
    youtube_websub_verify_token: str = ""  # echoed back by the hub on the GET verification hit
    youtube_websub_secret: str = ""  # HMAC key; when set, POSTs without a valid X-Hub-Signature are rejected

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

    # AI video beats: an optional motion-video source for standout beats. Off by default -- the
    # hosted generators (fal/HF/Veo) are all metered with no usable free tier, so leaving this on
    # with only those configured just wastes a few seconds hitting 402/429s before falling back to
    # the mascot. Turn it on when either (a) a hosted key has prepaid credits, or (b) you run the
    # free offline model on Google Colab and set ai_video_provider="colab" below.
    enable_ai_video_beats: bool = False
    # Which generator(s) _try_ai_video uses (graph/nodes/visual.py):
    #   "auto"  -> Colab (if COLAB_VIDEO_URL set) then fal -> HF -> NVIDIA -> Veo, first hit wins
    #   "colab" -> the offline Wan2.1-T2V-1.3B endpoint ONLY (colab/wan_video_colab.ipynb); on any
    #              failure the beat falls straight back to the mascot -- never touches a paid API
    #   "none"  -> skip AI video entirely (equivalent to enable_ai_video_beats=False for sourcing)
    ai_video_provider: str = "auto"
    # Applies only while enable_ai_video_beats is True -- caps how many beats per script get a real
    # AI video generation (asset_visual_node enforces it in beat order; the visual-planning LLM is
    # told the same number so it reserves AI_VIDEO for the standout, motion-worthy beats rather
    # than spreading a small budget thin).
    #   0 -> NO CAP: every beat the planner routes to AI_VIDEO generates a real clip. This is what
    #        makes the output an actually-animated story rather than a mostly-mascot slideshow, and
    #        it's the right setting for the free Colab generator where clips cost time, not money.
    #        Budget accordingly: ~6-12 min per beat on a free T4, so a 12-beat script is ~2 hours.
    #   >0 -> cap, for the metered hosted generators where each beat is a real billed generation.
    # enable_ai_video_beats=False still wins over either -- it zeroes the budget outright.
    max_ai_video_beats_per_run: int = 0
    # Primary generator (see tools/hf_video.py): Hugging Face Inference Providers, routed to a
    # fal.ai-hosted Wan2.2-TI2V-5B (Apache-2.0, no revenue-cap restrictions, confirmed pure
    # text-to-video). Free accounts get a small monthly credit (~$0.10, well under one clip's
    # ~$0.15) -- unlike Veo this has no billing-account gate, but sustained use still needs
    # occasional prepaid top-ups at huggingface.co/settings/billing.
    huggingface_api_key: str = ""  # free to create, no card required: huggingface.co/settings/tokens
    hf_video_model: str = "Wan-AI/Wan2.2-TI2V-5B"
    hf_video_provider: str = "fal-ai"
    # Secondary/legacy generator (see tools/veo_video.py): real Veo 3.1, paid-tier only -- Google's
    # Gemini API has no free tier for video at all (unlike every other provider in this file), so
    # this stays dormant until GOOGLE_API_KEY's project has Cloud Billing enabled. asset_visual_node
    # tries Hugging Face first and only falls back to Veo if it's configured and HF fails.
    veo_model: str = "veo-3.1-generate-preview"
    veo_video_duration_seconds: int = 8  # length of each Veo-generated clip -- 8 is the API's max
    # (durationSeconds must be 4-8 inclusive for veo-3.1-generate-preview; 10 is rejected outright)
    # Third generator (see tools/nvidia_video.py): nvidia/cosmos3-nano, the one Cosmos model with
    # an actual hosted Preview API on build.nvidia.com (the rest of the Cosmos family is
    # self-hosted-only). asset_visual_node tries this after HF and before falling back to Veo/
    # mascot.
    nvidia_api_key: str = ""  # https://build.nvidia.com/settings/api-keys ("nvapi-..." key)
    nvidia_video_invoke_url: str = ""  # override only if NVIDIA changes the hosted endpoint or
    # you switch to a different Cosmos3-family model
    nvidia_video_resolution: str = "720_16_9"
    nvidia_video_num_frames: int = 121
    nvidia_video_fps: int = 24
    # Fourth generator (see tools/fal_video.py): fal.ai's hosted API called directly with a native
    # fal key, rather than indirectly through Hugging Face Inference Providers like hf_video_model
    # above. asset_visual_node tries this first -- pay-per-use with no monthly credit ceiling to
    # run into, unlike HF's free tier.
    fal_api_key: str = ""  # "id:secret" format key from fal.ai/dashboard/keys
    fal_video_model: str = "fal-ai/wan-t2v"
    fal_video_resolution: str = "720p"
    fal_video_aspect_ratio: str = "16:9"

    # Background sound for scripts/produce_reel_clip.py's short vertical clips AND the long-form
    # pipeline's music bed (see tools/freesound_audio.py, graph/nodes/audio_render.py). Searched by
    # mood/vibe keyword against Freesound's CC-licensed library; falls back to a locally
    # synthesized ambient pad when unset or the search/download fails, so nothing renders silent.
    freesound_api_key: str = ""  # free: https://freesound.org/apiv2/apply
    # Mixes a music bed under the voiceover in video_assembly_node. Safe to leave on with no
    # Freesound key -- the synthesized pad is the fallback. Sourcing or mixing failures downgrade
    # to a voiceover-only render (has_music=False) rather than failing the run.
    enable_background_music: bool = True
    # Gain applied to the bed relative to the voiceover. Deliberately low: the voiceover is the
    # content, and anything much above ~0.15 starts competing with it for intelligibility on
    # phone speakers, which is where most of this content is watched.
    background_music_volume: float = 0.10
    # Freesound search keyword. Blank -> derived per run from the strategy's topic category, so a
    # nursery-rhyme video and a documentary don't get the same bed.
    background_music_mood: str = ""

    # Offline text-to-video model, run by the operator on Google Colab (see colab/
    # wan_video_colab.ipynb + tools/colab_video.py) and reached over a Gradio public URL that
    # changes every notebook restart. Used by scripts/produce_reel_clip.py and -- when
    # ai_video_provider is "auto" or "colab" -- by the main pipeline's AI_VIDEO beats.
    #
    # Model: Wan-AI/Wan2.1-T2V-1.3B (diffusers WanPipeline), Apache-2.0. The 1.3B DiT is built for
    # ~8GB consumer cards; the heavy part is the UMT5-XXL text encoder, which the notebook keeps on
    # CPU via enable_model_cpu_offload() (+ VAE tiling) so peak VRAM stays ~8-9GB and it fits a
    # free-tier T4. Native 832x480, 16fps; num_frames must be 4k+1 (VAE temporal stride 4) -- the
    # notebook rounds to the nearest valid value. ~10-20 min/clip on a T4 at 25 steps.
    # History (git log has the code): ModelScope-1.7b (256x256, too soft), CogVideoX-5B/-2B (~9GB
    # T5-XXL encoder -> 30-45min/clip or OOM), SD1.5/DreamShaper+AnimateDiff (flat retrofitted
    # motion), DreamShaper-XL+SVD-XT (wobbly, video stage never saw the prompt). Wan2.1-1.3B with
    # the text encoder actually offloaded is the first that's both T4-viable and real video motion.
    # On Colab Pro (L4/A100) switch the notebook to Wan-AI/Wan2.2-TI2V-5B for 720p.
    colab_video_url: str = ""
    colab_video_num_frames: int = 81  # 4k+1 -> ~5s at 16fps. produce_reel_clip.py loops it up to
    # its 10s target; the pipeline's audio_render trims/loops per beat.
    colab_video_height: int = 480
    colab_video_width: int = 832  # Wan2.1-1.3B is 832x480-native (16:9). The notebook rounds
    # height/width to multiples of 16 and swaps to 480x832 if you pass a portrait aspect.
    colab_video_guidance_scale: float = 5.0  # Wan's recommended range is 4-6 for the 1.3B model.

    # Mascot character (see tools/character_assets.py): a small pose pack generated once and
    # cached under media_dir/characters/, then reused by every render -- this replaces the old
    # single-AI-still-plus-Ken-Burns-pan for AI_IMAGE beats with an actually-animated character.
    mascot_character_name: str = "Mimi the Fox"
    mascot_character_prompt: str = (
        "Mimi the Fox: a cheerful, round-bodied orange fox cub with big sparkling brown eyes, a "
        "white belly and cheek patches, small rounded ears, and a red-and-white striped overall"
    )

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
