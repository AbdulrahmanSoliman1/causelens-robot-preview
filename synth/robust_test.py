"""Re-test the decision with causal/robust.py: structural fix + three-way targeted read + "check first".

  python3 synth/robust_test.py decide dev|held    -> synth/robust_decisions.jsonl (resumable)
  python3 synth/robust_test.py report [dev|held]
Tuned on dev only; held-out is run once.
"""
import json, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from causal import robust  # noqa: E402
import ablate  # noqa: E402

OUT = ROOT / "synth" / "robust_decisions.jsonl"
TAG = "v3"
lock = threading.Lock()


def decide(split):
    done = {(r["scenario"], r["state"], r["tag"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    jobs = [(s, st) for s in ablate.scenes() if s["split"] == split for st in ("blocked", "enabled")
            if (s["id"], st, TAG) not in done]
    print(len(jobs), "jobs", flush=True)

    def one(job):
        s, st = job
        try:
            t0 = time.time()
            d = robust.decide(ablate.src(s, st), s["instruction"])
            with lock, open(OUT, "a") as f:
                f.write(json.dumps(dict(tag=TAG, scenario=s["id"], split=s["split"], family=s["family"], state=st,
                                        instruction=s["instruction"], blocker=s["blocker"], verdict=d["verdict"],
                                        unknown=d["unverified"], blockers=d["blockers"], reads=d["reads"],
                                        graph=d["graph"], s=round(time.time() - t0, 1))) + "\n")
        except Exception as e:
            print("ERR", s["id"], st, str(e)[:150], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, jobs))


def report(split=None):
    rows = [r for r in map(json.loads, open(OUT)) if r["tag"] == TAG and (split is None or r["split"] == split)]
    for st, want in (("blocked", "blocked"), ("enabled", "works")):
        sub = [r for r in rows if r["state"] == st]
        c = Counter("right" if r["verdict"] == want else "check_first" if r["verdict"] == "check_first" else "wrong"
                    for r in sub)
        print(f"{st:8s} n={len(sub)} right={c['right']} wrong={c['wrong']} check_first={c['check_first']}")
    print("median s", sorted(r["s"] for r in rows)[len(rows) // 2] if rows else "-")
    for r in rows:
        want = "blocked" if r["state"] == "blocked" else "works"
        if r["verdict"] != want:
            print(f"  {r['scenario']:15s} {r['state']:8s} -> {r['verdict']:11s} true_blocker={r['blocker']:22s} "
                  f"unknown={r['unknown']} blockers={r['blockers']} reads={r['reads']} | {r['instruction']}")


if __name__ == "__main__":
    {"decide": lambda: decide(sys.argv[2]), "report": lambda: report(sys.argv[2] if len(sys.argv) > 2 else None)}[sys.argv[1]]()
