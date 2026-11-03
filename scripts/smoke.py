"""Smoke test: generate an unplugged lamp, ask Muse Spark about its state, then edit with Muse Image."""
import base64, json, os, sys, time
from pathlib import Path
import requests

BASE = "https://api.meta.ai/v1"
KEY = os.environ["META_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
OUT = Path(__file__).resolve().parent.parent / "out"


def post(path, body):
    t = time.time()
    r = requests.post(f"{BASE}{path}", headers=H, json=body, timeout=300)
    dt = time.time() - t
    if not r.ok:
        sys.exit(f"{path} {r.status_code}: {r.text[:500]}")
    return r.json(), dt


def save_image(resp, name):
    item = resp["data"][0]
    if "b64_json" in item:
        data = base64.b64decode(item["b64_json"])
    else:
        data = requests.get(item["url"], timeout=120).content
    p = OUT / name
    p.write_bytes(data)
    return p


def data_uri(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def ask_spark(img, question):
    resp, dt = post("/responses", {
        "model": "muse-spark-1.3",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": data_uri(img)},
        ]}],
    })
    text = "".join(c.get("text", "") for o in resp["output"] if o["type"] == "message" for c in o["content"])
    return text, dt, resp.get("usage")


gen, dt = post("/images/generations", {
    "model": "muse-image-1.0",
    "prompt": "Photo of a table lamp on a nightstand, lamp switched off and dark, its power cord clearly "
              "unplugged with the plug lying on the nightstand, no wall outlet connection, dim room",
    "n": 1,
})
src = save_image(gen, "lamp_unplugged.png")
print(f"generate: {dt:.1f}s -> {src}")

q = ('Answer as JSON {"plug_in_outlet": bool, "switch_on": bool, "lamp_emitting_light": bool} '
     "for this image. Only JSON.")
text, dt, usage = ask_spark(src, q)
print(f"spark state: {dt:.1f}s {usage}\n{text}")

edit, dt = post("/images/edits", {
    "model": "muse-image-1.0",
    "prompt": "generate image if the switch of this lamp is on",
    "images": [{"image_url": data_uri(src)}],
    "n": 1,
})
dst = save_image(edit, "lamp_switch_on_direct.png")
print(f"edit: {dt:.1f}s -> {dst}")

text, dt, usage = ask_spark(dst, q)
print(f"spark state after edit: {dt:.1f}s {usage}\n{text}")
