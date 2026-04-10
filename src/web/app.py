"""
Week 8: FastAPI app + static UI.

Run from project root:
  uvicorn src.web.app:app --host 0.0.0.0 --port 8000

Env:
  FAKE_NEWS_CHECKPOINT  path to best_model.pt (default: week5 seed_1337).
  Week 6 weights: outputs/week6/finetune_attention/seed_*/best_model.pt
  Week 7 does not produce a checkpoint (evaluation only).
"""

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.web.predictor import MultimodalPredictor

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Multimodal Fake News Detector", version="1.0")
_predictor: Optional[MultimodalPredictor] = None


def get_predictor() -> MultimodalPredictor:
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get(
            "FAKE_NEWS_CHECKPOINT",
            "outputs/week5/attention/seed_1337/best_model.pt",
        )
        _predictor = MultimodalPredictor(checkpoint_path=ckpt)
    return _predictor


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return HTMLResponse(
        "<p>Missing static/index.html. Open /docs for API.</p>",
        status_code=404,
    )


class JsonPredictBody(BaseModel):
    text: str = ""
    image_base64: Optional[str] = None


@app.post("/predict")
async def predict(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    raw = None
    if image is not None:
        raw = await image.read()
    try:
        out = get_predictor().predict(text, raw)
        return JSONResponse(out)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict_json")
async def predict_json(payload: JsonPredictBody):
    """
    JSON body: {"text": "...", "image_base64": "..."}  (image optional, raw base64)
    """
    import base64

    text = payload.text or ""
    raw = None
    if payload.image_base64:
        try:
            raw = base64.b64decode(payload.image_base64)
        except Exception:
            pass
    try:
        out = get_predictor().predict(text, raw)
        return JSONResponse(out)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
