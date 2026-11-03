"""Look before you act: decide whether an action will work, from one picture, with three fixes found in synth/diagnose.py.

1. Structure: world-state nodes (fuel, water in a container, battery charge, wind, ...) are always causes, never effects,
   and the robot's action is an `attempt` added as one more required cause, so it can't set an outcome (the screen) or a
   world state (the wind) directly.
2. Targeted read: every precondition that matters is read again from the picture with a three-way answer
   (yes / no / can't tell), which overrides the graph builder's guess.
3. Can't tell: a precondition that can't be seen is assumed to be in its working state, and the verdict says which ones
   weren't verified ("works, unverified: fuel"). A first version answered "check first" instead; on the dev scenes it
   refused 14 of 34 working scenes (wall-outlet power, batteries, the inside of a carton are never visible).
   A precondition that isn't part of the scene at all (a "chain" on a flagpole) is ignored.
"""
import json

from causal import api, scm, spark, vocab

WORLD = {"mains_power", "batteries_charged", "batteries_inside", "fuel_available", "dry_fuel", "container_has_liquid",
         "container_intact", "rope_intact", "chain_attached", "wind_present", "water_supply_on", "hose_connected",
         "bulb_intact", "wick_present", "object_fragile"}

READ = """Look carefully at this image. For each statement, answer:
- "yes" if the image clearly shows it is true,
- "no" if the image clearly shows it is false,
- "cant_tell" if the thing is in the scene but this can't be seen (hidden inside something, out of frame, too small),
- "not_in_scene" if this kind of device simply doesn't have such a part (e.g. a lock on window blinds).
Something that belongs there but is missing counts as "no", not "cant_tell" or "not_in_scene": a water cooler with no
bottle does not hold liquid, a grill or oven with no charcoal or wood has no fuel, a lamp with an empty socket has no bulb.
Judge only what is visible; do not assume the usual state.
The scene: someone is about to do this: {instruction}
{statements}"""

READ_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"],
               "properties": {"answers": {"type": "array",
                                          "items": {"type": "string", "enum": ["yes", "no", "cant_tell", "not_in_scene"]}}}}


def repair(g):
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


def preconditions(g):
    """Root nodes (other than the action) that have a path into something the action affects."""
    iv = g["intervention"]["node"]
    desc = scm.descendants(g, iv)
    roots = [n["id"] for n in g["nodes"] if n["rule"] == "exogenous" and n["id"] != iv]
    return [r for r in roots if scm.descendants(g, r) & desc]


def read_states(image, instruction, ids, effort="low"):
    if not ids:
        return {}
    lines = "\n".join(f"{k + 1}. {vocab.text(i, True)}" for k, i in enumerate(ids))
    out, _ = api.spark(READ.format(instruction=instruction, statements=lines) + "\nReturn answers in order.", image,
                       READ_SCHEMA, effort)
    return {i: {"yes": True, "no": False, "cant_tell": None, "not_in_scene": "na"}[a] for i, a in zip(ids, out["answers"])}


def working_value(g, node):
    """The value of a root that lets its effects happen (the `required` value on its outgoing edges)."""
    reqs = [e["required"] for e in g["edges"] if e["source"] == node]
    return reqs[0] if reqs else True


def outcome(g, overrides):
    """-> (blocked, blockers, info) for the graph with some root values overridden."""
    g = dict(g, nodes=[dict(n, observed=overrides.get(n["id"], n["observed"])) for n in g["nodes"]])
    iv = g["intervention"]
    _, info = scm.to_prompt(g, "", iv["node"], iv["value"])
    desc = scm.descendants(g, iv["node"])
    blockers = sorted({b for v in info["blockers"].values() for b in v})
    unchanged = [i for i in desc if info["after"][i] == info["before"][i]]
    return bool(blockers) and bool(unchanged), blockers, info, g


def decide(image, instruction, effort="minimal", read_effort="low"):
    """-> dict(verdict: works | blocked, unverified: preconditions assumed working, graph, reads, blockers, info)."""
    g, _ = spark.build_canon_graph(image, instruction, effort)
    g = repair(scm.break_cycles(g))
    pre = preconditions(g)
    reads = read_states(image, instruction, pre, read_effort)
    values = {i: (working_value(g, i) if v in (None, "na") else v) for i, v in reads.items()}
    unverified = [i for i, v in reads.items() if v is None]
    blocked, blockers, info, g2 = outcome(g, values)
    return dict(verdict="blocked" if blocked else "works", unverified=unverified, graph=g2, reads=reads,
                blockers=blockers, info=info)


def node_text(g, i, value):
    if i in vocab.NODES:
        return vocab.text(i, value)
    n = next(n for n in g["nodes"] if n["id"] == i)
    return n.get("true_text" if value else "false_text", i)


def plan(d, instruction):
    """What the robot expects, for a person to check: root causes that block, what changes, what stays, the edit
    prompt for the imagined picture and the yes/no checks for it."""
    g, info = d["graph"], d["info"]
    iv = g["intervention"]["node"]
    order = scm.topo_order(g)
    desc = scm.descendants(g, iv)
    pre = [i for i in preconditions(g)]
    causes = [dict(id=i, text=node_text(g, i, info["before"][i]), ok=info["before"][i] == working_value(g, i),
                   verified=d["reads"].get(i) is not None and d["reads"].get(i) != "na") for i in pre
              if d["reads"].get(i) != "na"]
    blocking = [c for c in causes if not c["ok"]]
    changed = [i for i in order if i in desc and info["after"][i] != info["before"][i]]
    kept = [i for i in order if i in desc and info["after"][i] == info["before"][i]]
    expect = [node_text(g, i, info["after"][i]) for i in changed + kept]
    if d["verdict"] == "blocked" and blocking:
        why = "; ".join(c["text"] for c in blocking)
        keep = "; ".join(node_text(g, i, info["after"][i]) for i in kept)
        prompt = (f"Show the moment right after someone tried this: {instruction}. It has no effect, because {why}. "
                  f"The result must clearly show: {keep}. {why[0].upper() + why[1:]}, exactly as in the original image. "
                  "Keep every other object, its position and state exactly as in the original image. No people, no hands, no text, no logos.")
    else:
        prompt = (f"Show the moment right after this action: {instruction}. The result must clearly show: "
                  + "; ".join(node_text(g, i, info["after"][i]) for i in changed) + "."
                  + (" These stay as they are: " + "; ".join(node_text(g, i, info["after"][i]) for i in kept) + "."
                     if kept else "")
                  + " Keep every other object, its position and state exactly as in the original image. No people, no hands, no text, no logos.")
    checks = [(i, info["after"][i], node_text(g, i, info["after"][i])) for i in changed + kept if i in vocab.VISUAL] + \
             [(c["id"], info["before"][c["id"]], c["text"]) for c in blocking if c["id"] in vocab.VISUAL]
    return dict(verdict=d["verdict"], causes=causes, blocking=blocking, expect=expect, prompt=prompt, checks=checks,
                unverified=[node_text(g, i, True) for i in d["unverified"]])


def fix_first(d, instruction):
    """The what-if a supervisor can ask for: fix every blocking cause, then act. -> (prompt, checks, fixes)."""
    g = d["graph"]
    p = plan(d, instruction)
    fixes = {c["id"]: working_value(g, c["id"]) for c in p["blocking"]}
    blocked, blockers, info, g2 = outcome(g, fixes)
    iv = g2["intervention"]["node"]
    order = scm.topo_order(g2)
    changed = [i for i in order if i in scm.descendants(g2, iv) and info["after"][i] != info["before"][i]]
    fix_texts = [node_text(g2, i, v) for i, v in fixes.items()]
    prompt = (f"Show the result after first making it so that {'; '.join(fix_texts)}, and then: {instruction}. "
              "The result must clearly show: " + "; ".join(fix_texts + [node_text(g2, i, info["after"][i]) for i in changed])
              + ". Keep every other object, its position and state exactly as in the original image. No people, no hands, no text, no logos.")
    checks = [(i, v, node_text(g2, i, v)) for i, v in fixes.items() if i in vocab.VISUAL] + \
             [(i, info["after"][i], node_text(g2, i, info["after"][i])) for i in changed if i in vocab.VISUAL]
    return prompt, checks, fix_texts, not blocked
