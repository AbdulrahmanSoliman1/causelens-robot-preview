"""Diagnose the causal arm's decision errors: rebuild the canonical graph (minimal effort, as in the app) for every dev and
held-out source, keep the full graph, and sort each wrong decision into a type:

  missing_node   blocked scene, the true blocker node isn't in the graph at all
  misread        the true blocker node is there, but its observed value is wrong
  miswired       the blocker node is read right, but no edge carries it to the target
  invented       enabled scene, the graph blocks on a different node than the true blocker
  target_absent  the target node isn't a descendant of the action
Writes synth/diagnose.jsonl. Usage: python3 synth/diagnose.py run | report
"""
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from causal import scm, spark  # noqa: E402
import ablate  # noqa: E402

OUT = ROOT / "synth" / "diagnose.jsonl"
lock = threading.Lock()


def run():
    done = {(r["scenario"], r["state"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    jobs = [(s, st) for s in ablate.scenes() for st in ("blocked", "enabled") if (s["id"], st) not in done]
    print(len(jobs), "jobs", flush=True)

    def one(job):
        s, st = job
        try:
            g, _ = spark.build_canon_graph(ablate.src(s, st), s["instruction"], "minimal")
            g = scm.break_cycles(g)
            with lock, open(OUT, "a") as f:
                f.write(json.dumps(dict(scenario=s["id"], split=s["split"], family=s["family"], state=st,
                                        instruction=s["instruction"], target=s["target"], blocker=s["blocker"],
                                        blocker_ok=s["blocker_ok"], graph=g)) + "\n")
        except Exception as e:
            print("ERR", s["id"], st, str(e)[:120], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, jobs))


def classify(r):
    g = r["graph"]
    iv = g["intervention"]
    nodes = {n["id"]: n for n in g["nodes"]}
    _, info = scm.to_prompt(g, r["instruction"], iv["node"], iv["value"])
    desc = scm.descendants(g, iv["node"])
    blockers = sorted({b for v in info["blockers"].values() for b in v})
    unchanged = [i for i in desc if info["after"][i] == info["before"][i]]
    pred = bool(blockers) and bool(unchanged)
    truth = r["state"] == "blocked"
    out = dict(pred=pred, ok=pred == truth, blockers=blockers, action=iv["node"],
               target_in=r["target"] in desc, blocker_in=r["blocker"] in nodes,
               blocker_read=nodes[r["blocker"]]["observed"] if r["blocker"] in nodes else None,
               blocker_want=(r["blocker_ok"] if r["state"] == "enabled" else not r["blocker_ok"]))
    if out["ok"]:
        out["kind"] = "ok"
    elif truth:  # missed a real blocker
        if not out["target_in"]:
            out["kind"] = "target_absent"
        elif not out["blocker_in"]:
            out["kind"] = "missing_node"
        elif out["blocker_read"] != out["blocker_want"]:
            out["kind"] = "misread"
        else:
            out["kind"] = "miswired"
    else:  # blocked an enabled scene
        out["kind"] = "misread" if r["blocker"] in blockers else "invented"
    return out


def report():
    from collections import Counter
    rows = [dict(r, **classify(r)) for r in map(json.loads, open(OUT))]
    print("correct", sum(r["ok"] for r in rows), "/", len(rows))
    for st in ("blocked", "enabled"):
        sub = [r for r in rows if r["state"] == st]
        print(st, sum(r["ok"] for r in sub), "/", len(sub), dict(Counter(r["kind"] for r in sub if not r["ok"])))
    print("\nwrong cases:")
    for r in rows:
        if not r["ok"]:
            n = {x["id"]: x for x in r["graph"]["nodes"]}
            print(f"{r['scenario']:15s} {r['state']:8s} {r['kind']:13s} action={r['action']:18s} true_blocker={r['blocker']}"
                  f"(want {r['blocker_want']}, read {r['blocker_read']}) graph_blockers={r['blockers']} | {r['instruction']}")
    json.dump([{k: r[k] for k in ("scenario", "state", "kind", "action", "blocker", "blockers", "blocker_read",
                                  "blocker_want", "instruction")} for r in rows if not r["ok"]],
              open(ROOT / "synth" / "diagnose_errors.json", "w"), indent=1)



# --- Structural fix, applied in code to the same saved graphs (no new model calls) ---
# World-state nodes: conditions of the scene that a robot's action can't set, and that never follow from other nodes.
WORLD = {"mains_power", "batteries_charged", "batteries_inside", "fuel_available", "dry_fuel", "container_has_liquid",
         "container_intact", "rope_intact", "chain_attached", "wind_present", "water_supply_on", "hose_connected",
         "bulb_intact", "wick_present", "object_fragile"}


def repair(g):
    """1. World-state nodes become roots (drop incoming edges): water in the reservoir is a cause of flow, not an effect.
    2. The robot's action is an attempt: if do() targets a node that has causes (the screen, the motor) or a world-state
       node (the wind), an exogenous 'attempt' node is added as one more required cause instead, so the other causes
       still decide whether it happens."""
    g = json.loads(json.dumps(g))
    for n in g["nodes"]:
        if n["id"] in WORLD:
            n["rule"] = "exogenous"
    g["edges"] = [e for e in g["edges"] if e["target"] not in WORLD]
    iv = g["intervention"]
    has_parents = any(e["target"] == iv["node"] for e in g["edges"])
    if iv["value"] and (has_parents or iv["node"] in WORLD):
        targets = [e["target"] for e in g["edges"] if e["source"] == iv["node"]] if iv["node"] in WORLD else [iv["node"]]
        g["nodes"].append(dict(id="attempt", observed=False, rule="exogenous", true_text="the robot tries the action",
                                false_text="the robot does nothing"))
        g["edges"] += [dict(source="attempt", target=t, required=True) for t in targets]
        for n in g["nodes"]:
            if n["id"] in targets and n["rule"] == "exogenous":
                n["rule"] = "all"
        g["intervention"] = dict(node="attempt", value=True)
    return g


def report_fixed():
    from collections import Counter
    rows = [json.loads(l) for l in open(OUT)]
    before = [classify(r) for r in rows]
    after = []
    for r in rows:
        try:
            after.append(classify(dict(r, graph=repair(r["graph"]))))
        except Exception as e:
            after.append(dict(ok=False, kind=f"error:{e}"))
    for name, res in (("as built", before), ("structural fix", after)):
        b = [x for x, r in zip(res, rows) if r["state"] == "blocked"]
        en = [x for x, r in zip(res, rows) if r["state"] == "enabled"]
        print(f"{name:15s} blocked {sum(x['ok'] for x in b)}/{len(b)}  enabled {sum(x['ok'] for x in en)}/{len(en)}  "
              f"all {sum(x['ok'] for x in res)}/{len(res)}")
    print("newly right:", [(r["scenario"], r["state"]) for x, y, r in zip(before, after, rows) if y["ok"] and not x["ok"]])
    print("newly wrong:", [(r["scenario"], r["state"], y["kind"]) for x, y, r in zip(before, after, rows) if x["ok"] and not y["ok"]])
    print("still wrong:", Counter(y["kind"] for y in after if not y["ok"]))


if __name__ == "__main__":
    {"run": run, "report": report, "fixed": report_fixed}[sys.argv[1]]()
