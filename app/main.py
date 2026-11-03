"""CauseLens Robot Preview server: before a home robot acts, it shows you what it expects to happen, and why.

Run: .venv/bin/python -m uvicorn app.main:app --port 8000   then open http://localhost:8000/robot
The page and its streaming API live in app/robot.py; this file loads .env and provides the image edit and file serving.
"""
import io, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():
    k, _, v = line.partition("=")
    if k.strip():
        os.environ.setdefault(k.strip(), v.strip())
# Contributor tier: much cheaper, but inputs (including uploaded photos) are used to train Meta's models.
# Set SPARK_MODEL=muse-spark-1.3 to switch back.
os.environ.setdefault("SPARK_MODEL", "muse-spark-1.3-contributor")

from causal import api, spark  # noqa: E402  (needs the env above)
from app import robot  # noqa: E402

WORK = Path(__file__).resolve().parent / "uploads"
WORK.mkdir(exist_ok=True)
pool = ThreadPoolExecutor(16)
app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


def edit(src, prompt, dest):
    """Muse Image edit; content-filter false positives are rephrased and retried."""
    for attempt in range(3):
        try:
            api.edit(src, prompt, dest)
            return prompt
        except RuntimeError as e:
            if "content_policy_violation" not in str(e) or attempt == 2:
                raise
            prompt = spark.rephrase_safe(prompt)


def save_upload(data, rid):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((1024, 1024))
    dest = WORK / f"{rid}_src.png"
    im.save(dest)
    return dest


@app.get("/")
def index():
    return RedirectResponse("/robot")


@app.get("/api/img/{name}")
def image(name: str):
    p = WORK / name
    if "/" in name or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


robot.setup(pool, WORK, edit, save_upload)
app.include_router(robot.router)
