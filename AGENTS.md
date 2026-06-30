
# AGENTS.md — parking-slot-counter-computer-vision

## Project structure

- `application/app.py` — **production FastAPI server** (the real entrypoint). Serves MJPEG at `/video` and JSON slot counts at `/slots`.
- `app.py` (root) — **legacy standalone** detection script. Hardcoded to read `please.jpg` and `model.pkl`. Not used by the web app.
- `train_model_final.py` — trains HOG + LinearSVC pipeline, saves `model-v2.pkl`. Also generates `confusion_matrix.png` and `sample_predictions.png`.
- `application/make-mask.py` — **interactive** parking spot mask drawer. Click-drag rectangles, `s` to save `mask.jpg`.
- `web-ui/index.html` — **vanilla HTML/CSS/JS** dashboard (no build step). Hardcoded to `https://comvis-backend.onrender.com/slots` — must edit for local dev.

## Model files

| File | Used by | Notes |
|------|---------|-------|
| `application/model-final.pkl` | `application/app.py` | Production model |
| `application/model-bird-eye.pkl` | (unused?) | Alternate, source unclear |
| `model.pkl` (root) | Root `app.py` | Legacy |
| `model-v2.pkl` | Created by `train_model_final.py` | Training output, **not** auto-copied to `application/` |

## Commands

```sh
# Start backend (direct)
cd application
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# Start backend (Docker)
cd application
docker compose up --build

# Train model
python train_model_final.py          # saves model-v2.pkl (not model-final.pkl)

# Draw parking mask
cd application
python make-mask.py                  # CAMERA_SRC must be set in .env
```

## Quirks & gotchas

- **No tests, no linting, no CI/CD** in this repo. Don't look for `pytest`, `ruff`, or GitHub Actions.
- **requirements.txt** pins 67 exact versions, many incidental (e.g. `rignore`, `detect-installer`). Prefer adding new deps loosely to avoid conflicts.
- **Dataset is gitignored** (`dataset/*`). Only exists locally. Training cannot run from a fresh clone.
- **Two models coexist** — `train_model_final.py` writes `model-v2.pkl` but the production server reads `application/model-final.pkl`. If retraining, copy the new model manually.
- **Web UI URL is hardcoded** at `web-ui/index.html:314` — the API endpoint and the camera feed link both point to `comvis-backend.onrender.com`. Change these when running locally.
- **`.env` config**: `CAMERA_SRC` can be a video file path, RTSP URL, or device index. Default is `dummy-video.mp4`. The camera thread **loops the video forever** (re-opens on EOF), so it works with a file as if it were a live feed.
- **Docker runtime deps**: OpenCV needs `libgl1`, `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6` (already in `Dockerfile`).
- **Mask format**: Binary image (white rectangles on black). Processed via `cv2.connectedComponentsWithStats`. Generated with `make-mask.py` or any image editor.
- **Frame diff optimization**: The server only runs the ML model every 30 frames on spots whose mean pixel difference exceeds 40% of the max diff. Unchanged spots keep their previous classification.
- **Thread safety**: `shared_state` and `latest_frame` are protected by `threading.Lock`. Potential frame tearing on concurrent reads.
- **Training uses `plt.show()`** which requires a display. On headless systems, comment out `plt.show()` calls or set `MPLBACKEND=Agg`.
