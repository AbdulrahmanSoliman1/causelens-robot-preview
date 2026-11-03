"""Muse Spark steps: build the causal graph (teacher), rewrite baseline, and judge."""
from . import api

NODE = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "true_text", "false_text", "observed", "rule", "visible"],
    "properties": {
        "id": {"type": "string"},
        "true_text": {"type": "string"},
        "false_text": {"type": "string"},
        "observed": {"type": "boolean"},
        "rule": {"type": "string", "enum": ["exogenous", "all", "any"]},
        "visible": {"type": "boolean"},
    },
}
EDGE = {
    "type": "object", "additionalProperties": False,
    "required": ["source", "target", "required"],
    "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "required": {"type": "boolean"}},
}
GRAPH = {
    "type": "object", "additionalProperties": False,
    "required": ["mechanism", "nodes", "edges", "intervention"],
    "properties": {
        "mechanism": {"type": "string", "enum": ["electrical", "water", "heat", "support", "other"]},
        "nodes": {"type": "array", "items": NODE},
        "edges": {"type": "array", "items": EDGE},
        "intervention": {
            "type": "object", "additionalProperties": False, "required": ["node", "value"],
            "properties": {"node": {"type": "string"}, "value": {"type": "boolean"}},
        },
    },
}

GRAPH_PROMPT = """You build a small deterministic boolean causal model of the scene in this image, for this edit request:
"{instruction}"

Rules:
- Nodes are physical conditions with a true/false state (e.g. plug_inserted, switch_on, bulb_works, lamp_lit, wall_glow).
- Include every precondition the outcome physically depends on, visible or hidden (power supply, plug, batteries, valve, water/fuel supply), and the visible effects of the outcome.
- "observed": the node's state in THIS image. Read visible states carefully from the pixels (is the plug actually inside an outlet? are batteries inside?). For hidden nodes assume the typical state (e.g. wall outlet has power).
- "rule": "exogenous" for nodes with no parents; "all" if every parent condition is needed; "any" if one is enough.
- Edge "required": the parent value needed for the child to be true (false for blockers such as valve_closed).
- true_text/false_text: short visual descriptions, e.g. "the lamp is lit and glowing warmly" / "the lamp is dark".
- "intervention": the single node the request directly sets, and its value. Only that node is set; do not also fix other preconditions.
- Keep it small: 4-10 nodes."""

REWRITE_PROMPT = """Rewrite this image-edit request for an image editor so that the edited image is physically and causally correct for the exact scene in this image (take into account every precondition visible in the scene). Do not change the scene beyond what the request implies.
Request: "{instruction}"
Return only the rewritten request."""

SAFE_PROMPT = """This image-edit prompt was rejected by the image editor's content filter as a false positive. Rephrase it with neutral, plain wording (e.g. avoid body-part words like "mouth" and graphic fluid descriptions) while keeping exactly the same visual meaning and every physical detail.
Prompt: "{prompt}"
Return only the rephrased prompt."""

JUDGE_PROMPT = """Look at this image and answer each yes/no question strictly from what is visible.
{questions}"""


def build_graph(image, instruction, effort=None, context=""):
    """context: facts known from outside the image (e.g. a troubleshooting conversation), given as observed states."""
    text = GRAPH_PROMPT.format(instruction=instruction)
    if context:
        text += ("\n- Facts already established (use them as the observed state of the matching nodes, even when they "
                 "are not visible, and include those nodes):\n" + context)
    return api.spark(text, image, GRAPH, effort)


def rewrite(image, instruction, effort=None):
    schema = {"type": "object", "additionalProperties": False, "required": ["prompt"],
              "properties": {"prompt": {"type": "string"}}}
    out, usage = api.spark(REWRITE_PROMPT.format(instruction=instruction), image, schema, effort)
    return out["prompt"], usage


def rephrase_safe(prompt):
    schema = {"type": "object", "additionalProperties": False, "required": ["prompt"],
              "properties": {"prompt": {"type": "string"}}}
    out, _ = api.spark(SAFE_PROMPT.format(prompt=prompt), schema=schema, effort="low")
    return out["prompt"]


def judge(image, questions, effort=None):
    """questions: list of strings -> list of bools."""
    schema = {"type": "object", "additionalProperties": False, "required": ["answers"],
              "properties": {"answers": {"type": "array", "items": {"type": "boolean"}}}}
    q = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(questions))
    out, _ = api.spark(JUDGE_PROMPT.format(questions=q) + "\nReturn answers in order.", image, schema, effort)
    return out["answers"]


# --- Canonical-vocabulary teacher: same graph, but node ids restricted to causal/vocab.py ---
from . import vocab

_ID = {"type": "string", "enum": vocab.IDS}
CANON_GRAPH = {
    "type": "object", "additionalProperties": False,
    "required": ["family", "nodes", "edges", "intervention"],
    "properties": {
        "family": {"type": "string", "enum": vocab.FAMILIES},
        "nodes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "observed", "rule"],
            "properties": {"id": _ID, "observed": {"type": "boolean"},
                           "rule": {"type": "string", "enum": ["exogenous", "all", "any"]}}}},
        "edges": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["source", "target", "required"],
            "properties": {"source": _ID, "target": _ID, "required": {"type": "boolean"}}}},
        "intervention": {"type": "object", "additionalProperties": False, "required": ["node", "value"],
                         "properties": {"node": _ID, "value": {"type": "boolean"}}},
    },
}

CANON_PROMPT = GRAPH_PROMPT.split("Rules:")[0] + """Rules:
- Use ONLY node ids from this vocabulary (id: meaning when true):
{vocab}
- Include every precondition the outcome physically depends on, visible or hidden, and the visible effects of the outcome.
- "observed": the node's state in THIS image. Read visible states carefully from the pixels (is the plug actually inside an outlet? is the cap on?). For hidden nodes assume the typical state (e.g. mains_power true).
- "rule": "exogenous" for nodes with no parents; "all" if every parent condition is needed; "any" if one is enough.
- Edge "required": the parent value needed for the child to be true.
- "intervention": the single node the request directly sets, and its value. Only that node is set; do not also fix other preconditions.
- Keep it small: 3-9 nodes."""


def build_canon_graph(image, instruction, effort=None):
    lines = "\n".join(f"  {i}: {t}" for i, (t, _) in vocab.NODES.items())
    g, usage = api.spark(CANON_PROMPT.format(instruction=instruction, vocab=lines), image, CANON_GRAPH, effort)
    return attach_texts(g), usage


def attach_texts(g):
    """Dedupe nodes, drop dangling edges, and add vocabulary texts so scm.to_prompt can use the graph."""
    seen, nodes = set(), []
    for n in g["nodes"]:
        if n["id"] not in seen:
            seen.add(n["id"])
            nodes.append(dict(n, true_text=vocab.text(n["id"], True), false_text=vocab.text(n["id"], False)))
    iv = g["intervention"]["node"]
    if iv not in seen:
        nodes.append(dict(id=iv, observed=not g["intervention"]["value"], rule="exogenous",
                          true_text=vocab.text(iv, True), false_text=vocab.text(iv, False)))
        seen.add(iv)
    edges = [e for e in g["edges"] if e["source"] in seen and e["target"] in seen and e["source"] != e["target"]]
    from .scm import break_cycles
    return break_cycles(dict(g, nodes=nodes, edges=edges))
