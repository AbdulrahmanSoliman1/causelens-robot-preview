"""Thin client for the Meta Model API (Muse Spark, Muse Image)."""
import base64, json, os, time
from pathlib import Path
import requests

BASE = "https://api.meta.ai/v1"
SPARK = os.environ.get("SPARK_MODEL", "muse-spark-1.3-contributor")
IMAGE = "muse-image-1.0"


def _headers():
    return {"Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"}


def post(path, body, retries=4):
    for attempt in range(retries):
        r = requests.post(f"{BASE}{path}", headers=_headers(), json=body, timeout=300)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt * 3)
            continue
        if not r.ok:
            raise RuntimeError(f"{path} {r.status_code}: {r.text[:500]}")
        return r.json()


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def spark(text, image=None, schema=None, effort=None):
    """One Muse Spark call. Returns (parsed JSON or text, usage dict)."""
    content = [{"type": "input_text", "text": text}]
    if image:
        content.append({"type": "input_image", "image_url": data_uri(image)})
    body = {"model": SPARK, "input": [{"type": "message", "role": "user", "content": content}]}
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}
    if effort:
        body["reasoning"] = {"effort": effort}
    resp = post("/responses", body)
    out = "".join(c.get("text", "") for o in resp["output"] if o["type"] == "message" for c in o["content"])
    return (json.loads(out) if schema else out), resp.get("usage", {})


def _save(resp, dest):
    item = resp["data"][0]
    data = base64.b64decode(item["b64_json"]) if "b64_json" in item else requests.get(item["url"], timeout=120).content
    Path(dest).write_bytes(data)
    return Path(dest)


def generate(prompt, dest):
    return _save(post("/images/generations", {"model": IMAGE, "prompt": prompt, "n": 1}), dest)


def edit(image, prompt, dest):
    body = {"model": IMAGE, "prompt": prompt, "images": [{"image_url": data_uri(image)}], "n": 1}
    return _save(post("/images/edits", body), dest)
