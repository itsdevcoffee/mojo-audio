# Shade Redeployment with mojo-audio

**Date:** 2026-04-09
**Goal:** Get the Shade voice conversion app running on DGX Spark using the current mojo-audio engine (post-RMVPE fixes), replacing the old Applio-only setup.
**Spark:** `visage@visage-spark`

---

## Current State

| Component | Status | Details |
|-----------|--------|---------|
| Web frontend | Running | SvelteKit on port 3100 (`node build/index.js`) |
| API backend | **Down** | No process on port 8000 |
| Redis | Running | localhost:6379, responds to PING |
| Celery worker | Down | No process |
| Docker containers | None exist | Compose stack never brought up or was cleaned up |
| mojo-audio repo | Up to date | `3d9b1ef` — includes all RMVPE fixes + mel alignment |
| Voice models | Present | 23 models at `/home/visage/repos/shade/models/` |

### Why not Docker?

The Docker image (`wunjo-dgx-spark`) has PyTorch CUDA + Python 3.10 but no MAX Engine. Adding MAX Engine to the container is a separate effort. The host-based approach works today because mojo-audio's pixi env already has everything Shade needs: fastapi, uvicorn, MAX Engine, torch, audio-separator, noisereduce.

---

## Phase 1: Inference API (minimum viable)

**Outcome:** `/convert`, `/separate`, `/clean`, `/health`, `/models` all working on port 8000.

### Step 1: Start the API server

The sprint-5 handoff established this pattern — run the Shade API using mojo-audio's pixi environment:

```bash
cd /home/visage/repos/mojo-audio
MOJO_AUDIO_PATH=/home/visage/repos/mojo-audio/src \
MODELS_DIR=/home/visage/repos/shade/models \
AUDIO_UPLOAD_DIR=/tmp/shade-uploads \
CORS_ORIGINS="https://app.tryshade.io,http://localhost:5173,http://localhost:3100" \
~/.pixi/bin/pixi run python -c '
import sys, os
sys.path.insert(0, "/home/visage/repos/shade/api")
sys.path.insert(0, "/home/visage/repos/mojo-audio/src")
os.chdir("/home/visage/repos/shade/api")
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
'
```

### Step 2: Wait for cold start

First request triggers MAX graph compilation for all models (~380s / ~6 min). Subsequent requests use cached graphs (~1.3s inference). The MAX cache lives at `~/.cache/modular/.max_cache`.

### Step 3: Verify endpoints

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Test conversion (pick any model from /models response)
curl -X POST http://localhost:8000/convert \
  -F "file=@/home/visage/drop/sample-voice-1.wav" \
  -F "model=<model_name>" \
  --output /tmp/test-converted.wav

# Test separation
curl -X POST http://localhost:8000/separate \
  -F "file=@/home/visage/drop/sample-voice-1.wav" \
  --output /tmp/test-separated.zip

# Test clean
curl -X POST http://localhost:8000/clean \
  -F "file=@/home/visage/drop/sample-voice-1.wav" \
  --output /tmp/test-cleaned.wav
```

### Step 4: Clear MAX cache (if needed)

If the API was previously run with old mojo-audio code, stale compiled graphs may cause issues:

```bash
rm -rf /home/visage/.cache/modular/.max_cache
rm -rf /home/visage/repos/mojo-audio/.pixi/envs/default/share/max/.max_cache
find /home/visage/repos/mojo-audio/src -name '__pycache__' -exec rm -rf {} +
```

---

## Phase 2: Persistence (systemd user services)

**Outcome:** API and frontend survive reboots without manual intervention.

### Step 1: Create API service

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/shade-api.service << 'EOF'
[Unit]
Description=Shade API (mojo-audio)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/visage/repos/mojo-audio
Environment=MOJO_AUDIO_PATH=/home/visage/repos/mojo-audio/src
Environment=MODELS_DIR=/home/visage/repos/shade/models
Environment=AUDIO_UPLOAD_DIR=/tmp/shade-uploads
Environment=CORS_ORIGINS=https://app.tryshade.io,http://localhost:5173,http://localhost:3100
ExecStart=/home/visage/.pixi/bin/pixi run python -c "import sys, os; sys.path.insert(0, '/home/visage/repos/shade/api'); sys.path.insert(0, '/home/visage/repos/mojo-audio/src'); os.chdir('/home/visage/repos/shade/api'); import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
```

### Step 2: Create frontend service

```bash
cat > ~/.config/systemd/user/shade-web.service << 'EOF'
[Unit]
Description=Shade Web Frontend
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/visage/repos/shade/web
ExecStart=/usr/bin/node build/index.js
Environment=PORT=3100
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
```

### Step 3: Enable and start

```bash
systemctl --user daemon-reload
systemctl --user enable shade-api shade-web
systemctl --user start shade-api shade-web

# Enable lingering so user services start at boot (not just at login)
sudo loginctl enable-linger visage
```

### Step 4: Verify persistence

```bash
systemctl --user status shade-api shade-web
# Optionally reboot and confirm both come back
```

---

## Phase 3: Training Pipeline (optional, lower priority)

**Outcome:** `/train` endpoint works for queuing voice model training jobs.

Training runs through Applio (not mojo-audio) and needs Celery + Redis.

### Step 1: Install missing Python packages

```bash
cd /home/visage/repos/mojo-audio
~/.pixi/bin/pixi run pip install celery redis
```

### Step 2: Start Celery worker

```bash
cd /home/visage/repos/Applio
PYTHONPATH="/home/visage/repos/shade/api:/home/visage/repos/Applio" \
/home/visage/repos/mojo-audio/.pixi/envs/default/bin/celery \
  -A celery_app worker --loglevel=info --concurrency=1 -n shade@%h
```

### Step 3: Create systemd service for Celery (if persisting)

Similar to the API service, but with the Celery entrypoint and Applio working directory.

### Notes

- Training still depends on Applio + PyTorch CUDA — this is expected and won't change until mojo-audio has its own training pipeline (not planned).
- The Celery worker needs GPU access for training. Verify CUDA is visible from the pixi env's torch installation.
- Redis is already running on Spark; no additional setup needed.

---

## Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| Cold start is ~6 min (MAX graph compilation) | Cached after first run. Pre-warm with a test request after starting. |
| pixi env has pip-installed packages (noisereduce, audio-separator) that could drift | Document the manual pip installs. Consider adding them to pixi.toml. |
| `/separate` needs torch CUDA for GPU-accelerated source separation | Verify `torch.cuda.is_available()` in pixi env. CPU fallback exists but is slow. |
| MAX cache invalidation after mojo-audio updates | Always clear MAX cache after `git pull` on mojo-audio. Add to deploy script. |
| `USE_APPLIO` fallback not tested recently | Verify Applio path + imports still work if fallback is needed. |

---

## Deploy Script (future convenience)

After Phase 2 is in place, updating Shade becomes:

```bash
ssh visage@visage-spark << 'DEPLOY'
cd /home/visage/repos/mojo-audio && git pull
rm -rf ~/.cache/modular/.max_cache
find src -name '__pycache__' -exec rm -rf {} +
systemctl --user restart shade-api
# Wait for cold start, then verify
sleep 10 && curl -s http://localhost:8000/health
DEPLOY
```
