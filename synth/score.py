"""Score synth/runs.jsonl with the Muse Spark judge at default effort and build a blind by-eye audit page.

A triplet is correct when the outcome matches the ground truth (blocked: absent, enabled: present) and the blocker's
state is unchanged. Writes synth/scores.json and synth/audit.html (+ synth/audit_key.json).
"""
import json, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causal import spark, vocab  # noqa: E402

HERE = ROOT / "synth"
ARMS = ("plain", "rewrite", "causal")


def q(device, node, value):
    return f"About the {device}: is it true that {vocab.text(node, value)}?"


def score(r):
    want_outcome = r["state"] == "enabled"
    blocker_state = r["blocker_ok"] if r["state"] == "enabled" else not r["blocker_ok"]
    got = spark.judge(HERE / "out" / r["out"], [q(r["device"], r["target"], True), q(r["device"], r["blocker"], blocker_state)])
    return dict(key=r["out"], scenario=r["scenario"], state=r["state"], arm=r["arm"], family=r["family"],
                outcome_ok=got[0] == want_outcome, blocker_kept=bool(got[1]),
                correct=got[0] == want_outcome and bool(got[1]))


def main():
    rows = {r["out"]: r for r in map(json.loads, open(HERE / "runs.jsonl"))}
    cache_p = HERE / "scores.json"
    cache = {s["key"]: s for s in json.load(open(cache_p))} if cache_p.exists() else {}
    todo = [r for k, r in rows.items() if k not in cache]
    with ThreadPoolExecutor(8) as ex:
        for s in ex.map(score, todo):
            cache[s["key"]] = s
    json.dump(sorted(cache.values(), key=lambda s: s["key"]), open(cache_p, "w"), indent=1)

    # Only scenarios with all six triplets count, so every arm sees the same sources.
    by_s = defaultdict(list)
    for s in cache.values():
        by_s[s["scenario"]].append(s)
    full = {k for k, v in by_s.items() if len(v) == 6}
    tab = defaultdict(lambda: [0, 0])
    for s in cache.values():
        if s["scenario"] in full:
            for key in ((s["arm"], s["state"]), (s["arm"], s["state"], s["family"])):
                tab[key][0] += s["correct"]
                tab[key][1] += 1
    print(f"{len(full)} complete scenarios")
    for arm in ARMS:
        line = "  ".join(f"{st} {tab[(arm, st)][0]}/{tab[(arm, st)][1]}" for st in ("blocked", "enabled"))
        print(f"{arm:8s} {line}")
    for arm in ARMS:
        fams = sorted({k[2] for k in tab if len(k) == 3})
        print(arm, " ".join(f"{f}:{tab[(arm, 'blocked', f)][0]}/{tab[(arm, 'blocked', f)][1]}" for f in fams))

    # Blind audit: 20 random after-images per arm, shuffled, arm hidden.
    rng = random.Random(7)
    pick = []
    for arm in ARMS:
        pool = sorted(k for k, s in cache.items() if s["arm"] == arm and s["scenario"] in full)
        pick += rng.sample(pool, min(20, len(pool)))
    rng.shuffle(pick)
    key = {f"a{i:02d}": k for i, k in enumerate(pick)}
    json.dump(key, open(HERE / "audit_key.json", "w"), indent=1)
    cards = []
    for aid, k in key.items():
        r = rows[k]
        src = f"../data/images/{r['scenario']}_{r['state']}_0.jpg"
        cards.append(f"""<div class=c><b>{aid}</b> — {r['device']}: <i>{r['instruction']}</i><br>
<small>source: {vocab.text(r['blocker'], r['blocker_ok'] if r['state'] == 'enabled' else not r['blocker_ok'])}</small><br>
<img src="{src}"><img src="out/{k}"><br>
outcome happened? <label><input type=radio name={aid}o value=1>yes</label><label><input type=radio name={aid}o value=0>no</label>
&nbsp; blocker kept? <label><input type=radio name={aid}b value=1>yes</label><label><input type=radio name={aid}b value=0>no</label></div>""")
    (HERE / "audit.html").write_text("""<!doctype html><meta charset=utf-8><title>Synth audit</title>
<style>body{font:14px sans-serif;margin:16px;background:#fff;color:#111}.c{margin:0 0 24px}img{width:360px;margin:4px}</style>
<p>Left: source. Right: after-image. Arm is hidden. Answer both questions, then press Export.</p>""" + "\n".join(cards) + """
<button onclick="const o={};document.querySelectorAll('input:checked').forEach(i=>o[i.name]=+i.value);
const b=new Blob([JSON.stringify(o)],{type:'application/json'});const a=document.createElement('a');
a.href=URL.createObjectURL(b);a.download='audit_labels.json';a.click()">Export</button>""")


if __name__ == "__main__":
    main()
