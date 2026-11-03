"""Run three "will it work?" methods once on the eye-gated fresh scenes (fresh/gate.json):
  open     causal/open.py: open vocabulary, roles, no fixed lists
  canon    causal/robust.py: the 49-concept vocabulary with the hand-made world-state list
  ask      one Muse Spark yes/no question at low effort, given the goal sentence
  python3 fresh/run.py run | report      -> fresh/decisions.jsonl
"""
import json, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causal import open as openv, robust, spark  # noqa: E402

HERE = ROOT / "fresh"
OUT = HERE / "decisions.jsonl"
lock = threading.Lock()


def ask(img, s):
    q = (f"Someone does this: {s['instruction']} Look carefully at everything visible in the image. "
         f"Would the result be that {s['goal_true'].rstrip('.')}?")
    return "works" if spark.judge(img, [q], "low")[0] else "blocked", {}


OPEN = lambda img, s: (lambda d: (d["verdict"], dict(goal=d["goal"], blocking=d["blocking"],
                                                     unverified=d["unverified"], graph=d["graph"])))(openv.decide(img, s["instruction"]))
METHODS = {"open3": OPEN, "open2": OPEN, "open": lambda img, s: (lambda d: (d["verdict"], dict(goal=d["goal"], blocking=d["blocking"],
                                                                  unverified=d["unverified"], graph=d["graph"])))(
                                       openv.decide(img, s["instruction"])),
           "canon": lambda img, s: (lambda d: (d["verdict"], dict(blockers=d["blockers"], unverified=d["unverified"],
                                                                  reads=d["reads"])))(robust.decide(img, s["instruction"])),
           "ask": ask}


def run():
    keep = set(json.load(open(HERE / "gate.json"))["keep"])
    sc = [s for s in json.load(open(HERE / "scenarios.json")) if s["id"] in keep]
    done = {(r["id"], r["state"], r["method"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    jobs = [(s, st, m) for s in sc for st in ("blocked", "enabled") for m in (sys.argv[2].split(",") if len(sys.argv) > 2 else METHODS) if (s["id"], st, m) not in done]
    print(len(jobs), "jobs", flush=True)

    def one(job):
        s, st, m = job
        try:
            t0 = time.time()
            v, extra = METHODS[m](HERE / "images" / f"{s['id']}_{st}.jpg", s)
            with lock, open(OUT, "a") as f:
                f.write(json.dumps(dict(id=s["id"], kind=s["kind"], state=st, method=m, verdict=v,
                                        right=v == ("blocked" if st == "blocked" else "works"),
                                        s=round(time.time() - t0, 1), **extra)) + "\n")
        except Exception as e:
            print("ERR", s["id"], st, m, str(e)[:150], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, jobs))


def report():
    rows = [json.loads(l) for l in open(OUT)]
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        for k in ((r["method"], r["kind"], r["state"]), (r["method"], r["kind"], "all"), (r["method"], "all", "all")):
            agg[k][0] += r["right"]; agg[k][1] += 1
    print(f"{'method':7s} {'familiar blk':>12s} {'familiar ok':>11s} {'novel blk':>9s} {'novel ok':>8s} {'familiar':>8s} {'novel':>6s} {'all':>6s} {'med s':>6s}")
    for m in ("open3", "open2", "open", "canon", "ask"):
        f = lambda *k: "%d/%d" % tuple(agg[(m,) + k])
        med = sorted(r["s"] for r in rows if r["method"] == m)
        print(f"{m:7s} {f('familiar','blocked'):>12s} {f('familiar','enabled'):>11s} {f('novel','blocked'):>9s} "
              f"{f('novel','enabled'):>8s} {f('familiar','all'):>8s} {f('novel','all'):>6s} {f('all','all'):>6s} {med[len(med)//2]:>6}")
    print("\nwrong:")
    for r in sorted(rows, key=lambda r: (r["method"], r["id"])):
        if not r["right"]:
            print(f"  {r['method']:6s} {r['id']:32s} {r['state']:8s} -> {r['verdict']:8s} "
                  f"{r.get('blocking', r.get('blockers', ''))} unverified={r.get('unverified', '')}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
