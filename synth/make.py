"""Generate (before, action, after) triplets in three arms: plain Muse Image, Spark rewrite, and the causal layer.

Resumable: writes synth/out/<scenario>_<state>__<arm>.png and one line per triplet to synth/runs.jsonl.
Usage: python3 synth/make.py [n_scenarios]   (default 48; preregistration in synth/README.md)
"""
import json, sys, threading, time, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causal import api, pipeline, scm, spark, vocab  # noqa: E402

HERE = ROOT / "synth"
OUT, RUNS, SOURCES, ERRORS = HERE / "out", HERE / "runs.jsonl", HERE / "sources.json", HERE / "errors.jsonl"
DATA = ROOT / "data"
VISIBLE_TARGETS = {"object_moves", "liquid_flowing", "motor_running", "element_hot", "ignition_applied", "device_on",
                   "liquid_spilled", "indicator_on", "screen_on", "light_emitted", "object_falls", "object_breaks",
                   "wheel_turning", "basin_overflowing"}
# Blockers the action itself may operate ("blend the fruit" includes flipping the switch) make the ground truth ambiguous.
AMBIGUOUS = {"power_switch_on", "force_applied", "door_open", "device_powered"}
ARMS = ("plain", "rewrite", "causal")
EFFORT = "minimal"  # the app's setting for the causal arm's graph and verify step
WORKERS = 8
lock = threading.Lock()


def append(path, row):
    with lock, open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def blocker_value():
    """scenario -> the blocker node's value when the outcome is enabled, from the teacher graphs' edges."""
    out = {}
    for r in map(json.loads, open(DATA / "labels.jsonl")):
        for e in r["graph"]["edges"]:
            if e["source"] == r["blocker_node"]:
                out.setdefault(r["scenario"], e["required"])
    return out


def select(n):
    req = blocker_value()
    per = defaultdict(list)
    for s in json.load(open(DATA / "catalog.json")):
        ins = s["instructions"][0]
        if (ins["target_node"] in VISIBLE_TARGETS and s["blocker_node"] in vocab.NODES
                and s["blocker_node"] not in AMBIGUOUS):
            per[s["family"]].append(dict(id=s["id"], family=s["family"], device=s["device"], instruction=ins["text"],
                                         target=ins["target_node"], blocker=s["blocker_node"],
                                         blocker_ok=req.get(s["id"], True)))
    k = n // len(per)
    return [s for fam in sorted(per) for s in per[fam][:k]]


def q(device, node, value):
    return f"About the {device}: is it true that {vocab.text(node, value)}?"


def source_gate(scen):
    """Keep a scenario only if the judge confirms the blocker state in both of its sources."""
    cache = json.load(open(SOURCES)) if SOURCES.exists() else {}
    todo = [s for s in scen if s["id"] not in cache]

    def check(s):
        ok = True
        for state in ("blocked", "enabled"):
            want = s["blocker_ok"] if state == "enabled" else not s["blocker_ok"]
            ok &= spark.judge(DATA / "images" / f"{s['id']}_{state}_0.jpg", [q(s["device"], s["blocker"], want)])[0]
        return s["id"], bool(ok)

    with ThreadPoolExecutor(WORKERS) as ex:
        for sid, ok in ex.map(check, todo):
            cache[sid] = ok
    json.dump(cache, open(SOURCES, "w"), indent=1)
    return [s for s in scen if cache[s["id"]]]


def causal_arm(src, instruction, dest):
    t0 = time.time()
    graph, _ = spark.build_canon_graph(src, instruction, EFFORT)
    graph = scm.break_cycles(graph)
    iv = graph["intervention"]
    prompt, info = scm.to_prompt(graph, instruction, iv["node"], iv["value"])
    desc = scm.descendants(graph, iv["node"])
    blockers = sorted({b for v in info["blockers"].values() for b in v})
    checks = [(i, info["after"][i], vocab.text(i, info["after"][i])) for i in scm.topo_order(graph)
              if (i in desc or i in blockers) and i in vocab.VISUAL]
    api.edit(src, prompt, dest)
    rec = dict(prompt=prompt, graph=graph, blockers=blockers, checks=[c[2] for c in checks], repaired=False)
    if checks:
        got = spark.judge(dest, [f"Is this true of the image: {t}?" for _, _, t in checks], EFFORT)
        failed = [t for (_, _, t), g in zip(checks, got) if not g]
        rec["verify"] = got
        if failed:
            prompt = prompt + " Important: the result must clearly show that " + "; ".join(failed) + "."
            api.edit(src, prompt, dest)
            rec.update(repaired=True, repair_prompt=prompt)
    rec["s"] = round(time.time() - t0, 1)
    return rec


def edit_safe(src, prompt, dest):
    """Muse Image edit with the pipeline's content-filter policy: rephrase and retry, up to twice."""
    for attempt in range(3):
        try:
            return api.edit(src, prompt, dest)
        except RuntimeError as e:
            if "content_policy_violation" not in str(e) or attempt == 2:
                raise
            prompt = spark.rephrase_safe(prompt)


def causal2_arm(src, instruction, dest):
    """v2, tuned on the dev scenes. Graph at low effort; every exogenous precondition re-read with a targeted yes/no
    question; a separate prompt when the action is blocked; verify every changed or unchanged descendant at low effort,
    up to two repairs; if a blocked render still shows the outcome, fall back to the unchanged source (nothing happens)."""
    t0 = time.time()
    graph, _ = spark.build_canon_graph(src, instruction, "low")
    graph = scm.break_cycles(graph)
    iv = graph["intervention"]
    pre = [n["id"] for n in graph["nodes"] if n["rule"] == "exogenous" and n["id"] != iv["node"]]
    reread = {}
    if pre:
        # Ask for visible evidence of the blocking state; only a "yes" overrides the graph (a "no" can't be told
        # apart from "not visible", e.g. whether the outlet has power).
        got = spark.judge(src, [f"Is it clearly visible in this image that {vocab.text(i, False)}?" for i in pre], "low")
        reread = {i: False for i, g in zip(pre, got) if g}
        graph = dict(graph, nodes=[dict(n, observed=reread.get(n["id"], n["observed"])) for n in graph["nodes"]])
    prompt, info = scm.to_prompt(graph, instruction, iv["node"], iv["value"])
    desc = scm.descendants(graph, iv["node"])
    blockers = sorted({b for v in info["blockers"].values() for b in v})
    unchanged = [i for i in scm.topo_order(graph) if i in desc and info["after"][i] == info["before"][i]]
    blocked = bool(blockers) and bool(unchanged)
    if blocked:
        why = "; ".join(vocab.text(b, info["after"][b]) for b in blockers)
        keep = "; ".join(vocab.text(i, info["after"][i]) for i in unchanged)
        prompt = (f"Show the moment right after someone tried this: {instruction}. It has no effect, because {why}. "
                  f"The result must clearly show: {keep}. {why[0].upper() + why[1:]}, exactly as in the original image. "
                  "Keep every other object, its position and state exactly as in the original image.")
    checks = [(i, info["after"][i], vocab.text(i, info["after"][i])) for i in scm.topo_order(graph)
              if i in desc or (i in blockers and i in vocab.VISUAL)]
    edit_safe(src, prompt, dest)
    rec = dict(prompt=prompt, graph=graph, reread=reread, blockers=blockers, blocked=blocked,
               checks=[c[2] for c in checks], repairs=0, fallback=False)
    for _ in range(2):
        if not checks:
            break
        got = spark.judge(dest, [f"Is this true of the image: {t}?" for _, _, t in checks], "low")
        failed = [t for (_, _, t), g in zip(checks, got) if not g]
        rec["verify"] = got
        if not failed:
            break
        edit_safe(src, prompt + " Important: the result must clearly show that " + "; ".join(failed) + ".", dest)
        rec["repairs"] += 1
    else:
        if checks:
            got = spark.judge(dest, [f"Is this true of the image: {t}?" for _, _, t in checks], "low")
            rec["verify"] = got
            if blocked and not all(got):
                Image.open(src).save(dest)
                rec["fallback"] = True
    rec["s"] = round(time.time() - t0, 1)
    return rec


def run(job):
    s, state, arm = job
    src = DATA / "images" / f"{s['id']}_{state}_0.jpg"
    dest = OUT / f"{s['id']}_{state}__{arm}.png"
    try:
        t0 = time.time()
        if arm == "causal":
            rec = causal_arm(src, s["instruction"], dest)
        elif arm == "causal2":
            rec = causal2_arm(src, s["instruction"], dest)
        else:
            rec = pipeline.run(src, s["instruction"], "direct" if arm == "plain" else "rewrite", dest, EFFORT)
            rec = dict(prompt=rec["prompt"], s=round(time.time() - t0, 1))
        append(RUNS, dict(scenario=s["id"], state=state, arm=arm, family=s["family"], device=s["device"],
                          instruction=s["instruction"], target=s["target"], blocker=s["blocker"],
                          blocker_ok=s["blocker_ok"], out=dest.name, **rec))
        print(f"ok {dest.name} {rec['s']}s", flush=True)
    except Exception as e:
        append(ERRORS, dict(job=f"{s['id']}_{state}__{arm}", error=str(e)[:500], tb=traceback.format_exc()[-800:]))
        print(f"ERR {dest.name}: {str(e)[:120]}", flush=True)


HELDOUT_AMBIGUOUS = AMBIGUOUS | {"device_on", "heat_applied", "container_tipped"}


def select_heldout():
    """Every catalog scenario not in the dev selection, with its first visible-outcome instruction (v2 preregistration)."""
    dev = {s["id"] for s in select(48)}
    req = blocker_value()
    out = []
    for s in json.load(open(DATA / "catalog.json")):
        if s["id"] in dev or s["blocker_node"] not in vocab.NODES or s["blocker_node"] in HELDOUT_AMBIGUOUS:
            continue
        ins = next((i for i in s["instructions"] if i["target_node"] in VISIBLE_TARGETS), None)
        if ins:
            out.append(dict(id=s["id"], family=s["family"], device=s["device"], instruction=ins["text"],
                            target=ins["target_node"], blocker=s["blocker_node"], blocker_ok=req.get(s["id"], True)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    heldout = len(sys.argv) > 1 and sys.argv[1] == "heldout"
    scen = select_heldout() if heldout else select(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
    arms = ("plain", "rewrite", "causal2") if heldout else ARMS
    kept = source_gate(scen)
    print(f"{len(scen)} selected, {len(kept)} pass the source gate", flush=True)
    done = {(r["scenario"], r["state"], r["arm"]) for r in map(json.loads, open(RUNS))} if RUNS.exists() else set()
    jobs = [(s, st, a) for s in kept for st in ("blocked", "enabled") for a in arms if (s["id"], st, a) not in done]
    print(f"{len(jobs)} triplets to make", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
