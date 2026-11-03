"""Ablation: where do the causal arm's errors come from?

decisions: for every dev and held-out source (both states), predict "blocked or not" without rendering, in several ways:
  graph_minimal, graph_low, graph_default  canonical graph + do() at that effort (blocked = a blocker leaves a descendant unchanged)
  graph_low_reread                         graph_low plus the v2 targeted re-read of the blocking state
  ask_low, ask_default                     one direct Muse Spark yes/no question, no graph
renders: give the true decision (oracle) and only test rendering on the held-out scenes:
  oracle        compiled prompt from the scenario spec, one edit, no verify
  oracle_vr     same, plus verify at low effort and up to two repairs, no fallback
  naive         control: the action plus the spec's blocker fact, no template, one edit
Usage: python3 synth/ablate.py decisions | renders [arms]
"""
import json, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from causal import api, scm, spark, vocab  # noqa: E402
import make  # noqa: E402

HERE = ROOT / "synth"
DEC, REN = HERE / "ablate_decisions.jsonl", HERE / "ablate_renders.jsonl"
lock = threading.Lock()


def append(path, row):
    with lock, open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def scenes():
    """The 32 dev and 25 held-out scenarios that have complete, eye-labelled triplets."""
    runs = [json.loads(l) for l in open(HERE / "runs.jsonl")]
    dev_ids = {r["scenario"] for r in runs if r["arm"] == "causal"}
    held_ids = {r["scenario"] for r in runs if r["arm"] == "causal2"} - dev_ids
    labelled = set(json.load(open(HERE / "heldout_key.json")).values())
    held_ids = {s for s in held_ids if f"{s}_blocked__causal2.png" in labelled}
    dev = [dict(s, split="dev") for s in make.select(48) if s["id"] in dev_ids]
    held = [dict(s, split="held") for s in make.select_heldout() if s["id"] in held_ids]
    return dev + held


def src(s, state):
    return make.DATA / "images" / f"{s['id']}_{state}_0.jpg"


def graph_blocked(image, instruction, effort, reread=False):
    graph, _ = spark.build_canon_graph(image, instruction, effort)
    graph = scm.break_cycles(graph)
    iv = graph["intervention"]
    if reread:
        pre = [n["id"] for n in graph["nodes"] if n["rule"] == "exogenous" and n["id"] != iv["node"]]
        if pre:
            got = spark.judge(image, [f"Is it clearly visible in this image that {vocab.text(i, False)}?" for i in pre], "low")
            off = {i for i, g in zip(pre, got) if g}
            graph = dict(graph, nodes=[dict(n, observed=False) if n["id"] in off else n for n in graph["nodes"]])
    _, info = scm.to_prompt(graph, instruction, iv["node"], iv["value"])
    desc = scm.descendants(graph, iv["node"])
    blockers = sorted({b for v in info["blockers"].values() for b in v})
    unchanged = [i for i in desc if info["after"][i] == info["before"][i]]
    return bool(blockers) and bool(unchanged), blockers


def ask_blocked(image, s, effort):
    q = (f"Someone does this: {s['instruction']}. Look carefully at everything visible in the image. "
         f"Would the result be that {vocab.text(s['target'], True)}?")
    return not spark.judge(image, [q], effort)[0], []


METHODS = {
    "graph_minimal": lambda im, s: graph_blocked(im, s["instruction"], "minimal"),
    "graph_low": lambda im, s: graph_blocked(im, s["instruction"], "low"),
    "graph_default": lambda im, s: graph_blocked(im, s["instruction"], None),
    "graph_low_reread": lambda im, s: graph_blocked(im, s["instruction"], "low", reread=True),
    "ask_low": lambda im, s: ask_blocked(im, s, "low"),
    "ask_default": lambda im, s: ask_blocked(im, s, None),
}


def decisions():
    done = {(r["scenario"], r["state"], r["method"]) for r in map(json.loads, open(DEC))} if DEC.exists() else set()
    jobs = [(s, st, m) for s in scenes() for st in ("blocked", "enabled") for m in METHODS if (s["id"], st, m) not in done]
    print(len(jobs), "decision jobs", flush=True)

    def run(job):
        s, st, m = job
        try:
            t0 = time.time()
            pred, blockers = METHODS[m](src(s, st), s)
            append(DEC, dict(scenario=s["id"], split=s["split"], family=s["family"], state=st, method=m,
                             pred_blocked=pred, blockers=blockers, s=round(time.time() - t0, 1)))
        except Exception as e:
            print("ERR", s["id"], st, m, str(e)[:120], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(run, jobs))


def oracle_prompt(s, state):
    if state == "blocked":
        why = vocab.text(s["blocker"], not s["blocker_ok"])
        keep = vocab.text(s["target"], False)
        return (f"Show the moment right after someone tried this: {s['instruction']}. It has no effect, because {why}. "
                f"The result must clearly show: {keep}. {why[0].upper() + why[1:]}, exactly as in the original image. "
                "Keep every other object, its position and state exactly as in the original image.")
    return (f"Edit: {s['instruction']}. Result: {vocab.text(s['target'], True)}. "
            "Keep every other object, its position and state exactly as in the original image.")


def naive_prompt(s, state):
    """Control: the action plus the spec's fact, the way a user would type it; no template."""
    fact = vocab.text(s["blocker"], s["blocker_ok"] if state == "enabled" else not s["blocker_ok"])
    return f"{s['instruction']}. {fact[0].upper() + fact[1:]}."


def render(job):
    s, st, arm = job
    dest = make.OUT / f"{s['id']}_{st}__{arm}.png"
    try:
        t0 = time.time()
        prompt = naive_prompt(s, st) if arm == "naive" else oracle_prompt(s, st)
        make.edit_safe(src(s, st), prompt, dest)
        rec = dict(repairs=0)
        if arm == "oracle_vr":
            want_blocked = st == "blocked"
            checks = [vocab.text(s["target"], not want_blocked),
                      vocab.text(s["blocker"], s["blocker_ok"] if st == "enabled" else not s["blocker_ok"])]
            for _ in range(2):
                got = spark.judge(dest, [f"Is this true of the image: {t}?" for t in checks], "low")
                failed = [t for t, g in zip(checks, got) if not g]
                if not failed:
                    break
                make.edit_safe(src(s, st), prompt + " Important: the result must clearly show that " + "; ".join(failed) + ".", dest)
                rec["repairs"] += 1
            rec["final_verify"] = spark.judge(dest, [f"Is this true of the image: {t}?" for t in checks], "low")
        append(REN, dict(scenario=s["id"], state=st, arm=arm, family=s["family"], out=dest.name, prompt=prompt,
                         s=round(time.time() - t0, 1), **rec))
        print("ok", dest.name, flush=True)
    except Exception as e:
        print("ERR", dest.name, str(e)[:120], flush=True)


ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ("oracle", "oracle_vr")


def renders():
    done = {(r["scenario"], r["state"], r["arm"]) for r in map(json.loads, open(REN))} if REN.exists() else set()
    jobs = [(s, st, a) for s in scenes() if s["split"] == "held" for st in ("blocked", "enabled")
            for a in ARMS if (s["id"], st, a) not in done]
    print(len(jobs), "render jobs", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(render, jobs))


if __name__ == "__main__":
    {"decisions": decisions, "renders": renders}[sys.argv[1]]()
