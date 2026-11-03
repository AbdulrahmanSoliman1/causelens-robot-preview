"""Open-vocabulary "will this action work?" from one camera picture: no fixed concept list, no hand-picked node sets.

Muse Spark names the parts and states of THIS scene, and tags each node with a role:
  action     what the robot does (exactly one)
  condition  a state of the scene the task depends on that the action doesn't change (power, fuel, air in a tyre, ...)
  effect     follows from other nodes; one effect is the goal (the task succeeded)
The general rules from causal/robust.py are applied by role instead of by id: conditions are always causes, the action is
the intervention, every condition is read again (yes / no / can't tell / not in the scene; missing counts as no), and
what can't be seen is assumed working and listed as unverified. The outcome is do(action) in plain code.
"""
import json

from causal import api, scm
from causal.robust import READ, READ_SCHEMA

DRAFT = """A home robot is about to do this task: "{instruction}".
Look at the camera picture and build a small cause-and-effect model of what the task needs in THIS scene.
- 3 to 8 nodes. ids are snake_case. true_text / false_text are short visible states about this scene's objects
  ("the front tyre is inflated" / "the front tyre is completely flat").
- role "action": exactly one node, what the robot does ("the robot pushes the wheelbarrow"). The robot does exactly the
  task as stated and nothing else: it does not plug things in, refill, repair, untangle or clear anything unless the task
  says so. Whatever such a step would change is a condition, read as it is in the picture now.
- role "condition": a state of the scene the task depends on that the robot's action does not change (power at the
  outlet, fuel, water in a container, air in a tyre, a string untangled, a lid open). Conditions have no incoming edges.
  Include every condition the goal physically depends on, including hidden ones, and always the everyday ones for its
  kind of task, even if they look fine: anything electric must be plugged in and powered; anything that holds or
  dispenses something must not be empty; anything that moves must have its parts attached and intact and nothing jamming
  or blocking it; anything that burns must have fuel.
- role "effect": follows from other nodes; it has incoming edges. rule "all": every requirement must hold; "any": one is enough.
- edges: source -> target with "required" = the source value the target needs.
- observed: each node's state as you read it in the picture now (the action is false: it hasn't happened yet).
- goal: the id of the effect that means the task succeeded.
{note}"""

NODE = {"type": "object", "additionalProperties": False,
        "required": ["id", "true_text", "false_text", "role", "rule", "observed"],
        "properties": {"id": {"type": "string"}, "true_text": {"type": "string"}, "false_text": {"type": "string"},
                       "role": {"type": "string", "enum": ["action", "condition", "effect"]},
                       "rule": {"type": "string", "enum": ["all", "any"]}, "observed": {"type": "boolean"}}}
EDGE = {"type": "object", "additionalProperties": False, "required": ["source", "target", "required"],
        "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "required": {"type": "boolean"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["nodes", "edges", "goal"],
          "properties": {"nodes": {"type": "array", "items": NODE}, "edges": {"type": "array", "items": EDGE},
                         "goal": {"type": "string"}}}


def to_scm(d):
    """Apply the role rules: conditions and the action are roots, effects are computed."""
    roots = {n["id"] for n in d["nodes"] if n["role"] != "effect"}
    edges = [e for e in d["edges"] if e["target"] not in roots]
    nodes = [dict(id=n["id"], true_text=n["true_text"], false_text=n["false_text"],
                  observed=False if n["role"] == "action" else n["observed"],
                  rule="exogenous" if n["id"] in roots else n["rule"]) for n in d["nodes"]]
    action = next(n["id"] for n in d["nodes"] if n["role"] == "action")
    return dict(nodes=nodes, edges=edges, intervention=dict(node=action, value=True)), action


def validate(d):
    ids = [n["id"] for n in d["nodes"]]
    assert len(ids) == len(set(ids)), "node ids must be unique"
    assert sum(n["role"] == "action" for n in d["nodes"]) == 1, "exactly one action node"
    role = {n["id"]: n["role"] for n in d["nodes"]}
    assert all(e["source"] in role and e["target"] in role for e in d["edges"]), "edge names an unknown node"
    assert role.get(d["goal"]) == "effect", "goal must be an effect node"
    g, action = to_scm(d)
    for i, r in role.items():
        if r == "effect":
            assert any(e["target"] == i for e in g["edges"]), f"effect {i} needs an incoming edge"
    scm.topo_order(g)
    assert d["goal"] in scm.descendants(g, action), "the goal must follow from the action"


def draft(image, instruction, effort="low"):
    out, _ = api.spark(DRAFT.format(instruction=instruction, note=""), image, SCHEMA, effort)
    try:
        validate(out)
        return out
    except (AssertionError, ValueError, KeyError, StopIteration) as e:
        out, _ = api.spark(DRAFT.format(instruction=instruction, note=f"A previous draft was rejected: {e}. Fix that."),
                           image, SCHEMA, effort)
        validate(out)
        return out


def working_value(g, node):
    reqs = [e["required"] for e in g["edges"] if e["source"] == node]
    return reqs[0] if reqs else True


def decide(image, instruction, effort="low", read_effort="low"):
    """-> dict(verdict: works | blocked, goal, blocking, unverified, graph, reads)."""
    d = draft(image, instruction, effort)
    g, action = to_scm(d)
    desc = scm.descendants(g, action)
    conds = [n["id"] for n in d["nodes"] if n["role"] == "condition" and scm.descendants(g, n["id"]) & desc]
    text = {n["id"]: n for n in d["nodes"]}
    reads = {}
    if conds:
        lines = "\n".join(f"{k + 1}. {text[i]['true_text']}" for k, i in enumerate(conds))
        out, _ = api.spark(READ.format(instruction=instruction, statements=lines) + "\nReturn answers in order.", image,
                           READ_SCHEMA, read_effort)
        reads = {i: {"yes": True, "no": False, "cant_tell": None, "not_in_scene": "na"}[a]
                 for i, a in zip(conds, out["answers"])}
    values = {i: (working_value(g, i) if v in (None, "na") else v) for i, v in reads.items()}
    g = dict(g, nodes=[dict(n, observed=values.get(n["id"], n["observed"])) for n in g["nodes"]])
    before, after, _ = scm.intervene(g, action, True)
    blocking = [i for i in conds if values.get(i, True) != working_value(g, i)]
    return dict(verdict="works" if after[d["goal"]] else "blocked", goal=text[d["goal"]]["true_text"],
                blocking=[text[i]["false_text"] if not values[i] else text[i]["true_text"] for i in blocking],
                unverified=[text[i]["true_text"] for i, v in reads.items() if v is None], graph=d, reads=reads,
                scm=g, action=action, conds=conds, blocking_ids=blocking, before=before, after=after)


def question(instruction):
    """The question a person asks, and what both image models are given: "Draw what will happen if the robot does this: ..."."""
    return f"Draw what will happen if the robot does this: {instruction.rstrip('.')}."


def node_text(g, i, value):
    n = next(n for n in g["nodes"] if n["id"] == i)
    return n["true_text"] if value else n["false_text"]


KEEP = " Keep every other object, its position and state exactly as in the original image. No people, no hands, no text, no logos."


def plan(d, instruction):
    """Same shape as causal.robust.plan, for the open graph: causes, blocking, expected result, edit prompt, checks."""
    g, before, after = d["scm"], d["before"], d["after"]
    order = scm.topo_order(g)
    desc = scm.descendants(g, d["action"])
    causes = [dict(id=i, text=node_text(g, i, before[i]), ok=i not in d["blocking_ids"],
                   verified=d["reads"].get(i) not in (None, "na")) for i in d["conds"] if d["reads"].get(i) != "na"]
    blocking = [c for c in causes if not c["ok"]]
    effects = [i for i in order if i in desc]
    expect = [node_text(g, i, after[i]) for i in effects]
    if d["verdict"] == "blocked" and blocking:
        why = "; ".join(c["text"] for c in blocking)
        keep = "; ".join(node_text(g, i, after[i]) for i in effects if not after[i])
        prompt = (f"{question(instruction)} It has no effect, because {why}. "
                  f"The result must clearly show: {keep}. {why[0].upper() + why[1:]}, exactly as in the original image." + KEEP)
    else:
        changed = [i for i in effects if after[i] != before[i]]
        kept = [i for i in effects if after[i] == before[i]]
        prompt = (f"{question(instruction)} The result must clearly show: "
                  + "; ".join(node_text(g, i, after[i]) for i in changed) + "."
                  + (" These stay as they are: " + "; ".join(node_text(g, i, after[i]) for i in kept) + "." if kept else "")
                  + KEEP)
    checks = [(i, after[i], node_text(g, i, after[i])) for i in effects] + \
             [(c["id"], before[c["id"]], c["text"]) for c in blocking]
    return dict(verdict=d["verdict"], causes=causes, blocking=blocking, expect=expect, prompt=prompt, checks=checks,
                unverified=d["unverified"])


def fix_first(d, instruction):
    """Fix every blocking condition, then act. -> (prompt, checks, fix texts, works)."""
    g = d["scm"]
    fixes = {i: working_value(g, i) for i in d["blocking_ids"]}
    g2 = dict(g, nodes=[dict(n, observed=fixes.get(n["id"], n["observed"])) for n in g["nodes"]])
    before, after, _ = scm.intervene(g2, d["action"], True)
    desc = scm.descendants(g2, d["action"])
    changed = [i for i in scm.topo_order(g2) if i in desc and after[i] != before[i]]
    fix_texts = [node_text(g2, i, v) for i, v in fixes.items()]
    prompt = (f"Draw what will happen if, first, {'; '.join(fix_texts)}, and then the robot does this: {instruction.rstrip('.')}. "
              "The result must clearly show: " + "; ".join(fix_texts + [node_text(g2, i, after[i]) for i in changed]) + "."
              + KEEP)
    checks = [(i, v, node_text(g2, i, v)) for i, v in fixes.items()] + [(i, after[i], node_text(g2, i, after[i])) for i in changed]
    return prompt, checks, fix_texts, bool(after[d["graph"]["goal"]])
