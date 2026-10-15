# Deploying AutoTube Studio

Running this on a domain, for real users. It assumes you have a server you control and a domain
you can point at it.

## What this needs from a host

Not every platform can run it, so check this before choosing one:

| Needs | Why |
|---|---|
| **Long-running processes** | A render is ~20 minutes of ffmpeg. Platforms that kill a request after 30–60s cannot run the worker. |
| **Persistent disk** | Renders are written to disk and served from it until upload. Ephemeral filesystems lose them mid-run. |
| **~2 CPU, 4 GB RAM minimum** | ffmpeg encoding is the floor. 4 CPU is noticeably better if more than one account is active. |
| **Docker** | Postgres, Redis, the API, the worker and the scheduler come up together. |

A small VPS (Hetzner, DigitalOcean, Linode) is the straightforward fit. **GitHub Pages cannot host
this** — it serves static files only, with no Python, database or background workers. If you want
to use Pages, it can serve a marketing page that links to this app; the app itself needs a server.

## Before you start

1. A domain, with an **A record pointing at the server's IP**. Caddy cannot get a certificate
   until DNS resolves, so do this first and let it propagate.
2. Ports **80 and 443** open to the internet. Port 80 is not optional — Let's Encrypt validates
   over it.
3. Docker and the compose plugin installed.

## Deploy

```bash
git clone <your-repo> autotube && cd autotube
cp .env.example .env     # then edit, see below
```

Fill in `.env`. The ones that matter in production:

```bash
APP_DOMAIN=studio.example.com        # used by Caddy for TLS
ACME_EMAIL=you@example.com           # Let's Encrypt expiry warnings
APP_BASE_URL=https://studio.example.com

APP_SECRET_KEY=<generate, see below> # BACK THIS UP
POSTGRES_PASSWORD=<a real password>  # the default is "change-me"

REQUIRE_HUMAN_APPROVAL=true          # keep this on
YOUTUBE_UPLOAD_VISIBILITY=unlisted   # "public" when you trust the output
```

Generate the secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**Back up `APP_SECRET_KEY` somewhere outside the server.** It signs every session and encrypts
every stored Google credential. Losing it logs everyone out *and* makes every connected channel
unreadable, so each user has to reconnect. Rotating it deliberately does the same thing, which is
the correct response to a suspected compromise and a disaster otherwise.

Then bring it up:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The API runs `alembic upgrade head` on boot, so the schema is created on first start. Watch the
certificate being issued:

```bash
docker compose logs -f caddy
```

## Google Cloud, once per deployment

Each **user** brings their own Google OAuth client — that is what gives every account its own
10,000 units/day and avoids app verification. But your deployment's own `.env` credentials (used
by the scheduler and the `scripts/` one-offs) still need the redirect URI updated:

- Add `https://studio.example.com/onboarding/callback` as an **Authorised redirect URI**.
- Keep `http://localhost:8001/onboarding/callback` alongside it so local development still works.
- **Publish the OAuth consent screen.** While it is in *Testing*, Google expires refresh tokens
  after **7 days** and only listed test users can authorise at all. Publishing stops both. The
  "unverified app" warning that remains is expected for a personal project.

Tell your users the same about *their* projects — the onboarding page already does.

## What to check once it is up

```bash
curl -I https://studio.example.com/            # 303 to /auth/login
curl -I https://studio.example.com/api/runs    # 401, not 202
```

The second one matters. Every endpoint that can start a run, read another account's data, or
publish a video requires a session; if any of those answer without one, stop and fix it before
letting anyone in.

Then sign up, connect a channel, and make one video end to end before inviting anyone.

## Backups

Two things are not reproducible:

```bash
# The database: accounts, encrypted credentials, run history, performance data
docker compose exec -T postgres pg_dump -U autotube autotube | gzip > backup-$(date +%F).sql.gz

# APP_SECRET_KEY, which is what makes the credentials in that dump readable.
# A database backup without it is a backup of unreadable secrets.
```

Media (`media_data`) is renders and thumbnails. Published videos are on YouTube already, so this
is mostly scratch — back it up if you want the source files, skip it if disk is tight.

## Limits that will bite before you expect them

- **Rendering is serial-ish.** `--concurrency=2` on the worker and ~20 minutes a video means
  roughly 6 videos an hour per server, whatever the per-account limits say.
- **Per-account limits** default to 2 concurrent and 10/day (`MAX_CONCURRENT_RUNS_PER_USER`,
  `MAX_RUNS_PER_USER_PER_DAY`). Raise them only as far as the CPU above allows.
- **LLM keys are shared.** Unlike YouTube credentials, the groq/gemini keys in `.env` belong to
  the deployment, so every account spends the operator's free tier.
- **Disk fills quietly.** Each run leaves a render, its thumbnails, voiceover segments and source
  clips. Prune `media_data` periodically, or the first symptom is runs failing to write.

## Updating

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Rebuild rather than restart. A dependency change in `pyproject.toml` only takes effect when the
image is rebuilt — restarting a container reverts it to whatever the image holds, which once cost
this project an afternoon of a "fixed" text-to-speech bug reappearing on every restart.
