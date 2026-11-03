"""Robot preview: before a home robot acts, it shows a supervisor what it expects to happen and why.

From the robot's camera picture and the planned action: causal/open.py decides whether the action will work (open
vocabulary: Muse Spark names this scene's parts and their roles; every condition is read again, unverified ones listed), Muse Image draws the expected result from the
computed outcome, and Muse Spark checks that picture (one redraw if a check fails). A plain Muse Image edit of the same
action runs alongside for comparison. "Fix first" renders the what-if with every blocking cause fixed.
On 18 fresh scenes the decision was right 30/36 (fresh/README.md; the fixed-vocabulary version 22/36, asking Muse Spark
26/36, small sample); what this adds is a picture drawn from the decision and reasons a person can check.
"""
import json, time, uuid
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from causal import open as engine, spark

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = STATIC / "robot"
router = APIRouter()
RSESS = {}  # rid -> {src, instruction, decision}; in memory, fine for a demo server
EFFORT = "low"


def setup(pool, work, edit, save_upload):
    router.pool, router.work, router.edit, router.save_upload = pool, work, edit, save_upload


def line(**ev):
    return json.dumps(ev) + "\n"


@router.get("/robot")
def page():
    return FileResponse(STATIC / "robot.html")


@router.get("/api/robot/examples")
def examples():
    return json.load(open(EXAMPLES / "examples.json"))


def check(img, checks):
    if not checks:
        return []
    got = spark.judge(img, [f"Is this true of the image: {t}?" for _, _, t in checks], EFFORT)
    return [dict(text=t, ok=bool(g)) for (_, _, t), g in zip(checks, got)]


def imagine(src, prompt, checks, dest, repair_dest, kind):
    """Edit, check, one redraw if a check fails. -> list of events."""
    t0 = time.time()
    router.edit(src, prompt, dest)
    res = check(dest, checks)
    out = [dict(type=kind, url=f"/api/img/{dest.name}", checks=res, repaired=False, seconds=round(time.time() - t0, 1))]
    failed = [c["text"] for c in res if not c["ok"]]
    if failed:
        router.edit(src, prompt + " Important: the result must clearly show that " + "; ".join(failed) + ".", repair_dest)
        out.append(dict(type=kind, url=f"/api/img/{repair_dest.name}", checks=check(repair_dest, checks), repaired=True,
                        seconds=round(time.time() - t0, 1)))
    return out


@router.post("/api/robot/preview")
def preview(instruction: str = Form(...), example: str = Form(""), image: UploadFile | None = File(None)):
    rid = uuid.uuid4().hex[:10]
    if image is not None and image.filename:
        src = router.save_upload(image.file.read(), rid)
    elif example and "/" not in example and (EXAMPLES / f"{example}.jpg").exists():
        src = EXAMPLES / f"{example}.jpg"
    else:
        raise HTTPException(400, "pick an example or upload a picture")

    def events():
        t0 = time.time()
        # plain Muse Image gets the same question a person asks
        plain = router.pool.submit(router.edit, src, engine.question(instruction), router.work / f"robot_{rid}_plain.png")
        try:
            d = engine.decide(src, instruction)
        except Exception as e:
            yield line(type="error", message=f"couldn't read the scene: {str(e)[:150]}")
            return
        p = engine.plan(d, instruction)
        RSESS[rid] = dict(src=src, instruction=instruction, decision=d)
        yield line(type="decision", rid=rid, verdict=p["verdict"], causes=p["causes"], blocking=p["blocking"],
                   expect=p["expect"], unverified=p["unverified"], prompt=p["prompt"], seconds=round(time.time() - t0, 1))
        causal = router.pool.submit(imagine, src, p["prompt"], p["checks"], router.work / f"robot_{rid}_causal.png",
                                    router.work / f"robot_{rid}_repair.png", "causal")
        for name, fut in (("plain", plain), ("causal", causal)):
            try:
                res = fut.result()
                if name == "plain":
                    yield line(type="plain", url=f"/api/img/robot_{rid}_plain.png")
                else:
                    for ev in res:
                        yield line(**ev)
            except Exception as e:
                yield line(type="error", message=f"{name} picture failed: {str(e)[:150]}")
        yield line(type="done", seconds=round(time.time() - t0, 1))

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/api/robot/fix")
def fix(rid: str = Body(..., embed=True)):
    s = RSESS.get(rid)
    if not s:
        raise HTTPException(404)

    def events():
        prompt, checks, fixes, works = engine.fix_first(s["decision"], s["instruction"])
        yield line(type="fixplan", fixes=fixes, works=works, prompt=prompt)
        try:
            for ev in imagine(s["src"], prompt, checks, router.work / f"robot_{rid}_fix.png",
                              router.work / f"robot_{rid}_fix2.png", "fixed"):
                yield line(**ev)
        except Exception as e:
            yield line(type="error", message=f"picture failed: {str(e)[:150]}")
        yield line(type="done")

    return StreamingResponse(events(), media_type="application/x-ndjson")
