"""Deterministic boolean SCM: abduct node values from the image, apply do(), propagate to descendants."""


def descendants(graph, node):
    children = {}
    for e in graph["edges"]:
        children.setdefault(e["source"], []).append(e["target"])
    seen, stack = set(), [node]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def topo_order(graph):
    ids = [n["id"] for n in graph["nodes"]]
    parents = {i: [e["source"] for e in graph["edges"] if e["target"] == i] for i in ids}
    order, done = [], set()

    def visit(i, path=()):
        if i in done:
            return
        if i in path:
            raise ValueError(f"cycle at {i}")  # callers drop cycles first via break_cycles
        for p in parents[i]:
            visit(p, path + (i,))
        done.add(i)
        order.append(i)

    for i in ids:
        visit(i)
    return order


def _forward(graph, nodes, in_edges, do=None):
    """Evaluate the deterministic SCM: root nodes take their observed value (abduction), every node
    with parents is computed from them, and the do() node is clamped."""
    vals = {}
    for i in topo_order(graph):
        if do and i == do[0]:
            vals[i] = do[1]
        elif in_edges.get(i):
            sat = [vals[e["source"]] == e["required"] for e in in_edges[i]]
            vals[i] = any(sat) if nodes[i]["rule"] == "any" else all(sat)
        else:
            vals[i] = nodes[i]["observed"]
    return vals


def intervene(graph, node, value):
    """Return (before, after, blockers). `before` is the scene as observed in the image. `after` is
    recomputed from the root causes with do() applied, so intermediate nodes come from their causes
    (a device that is merely switched off is not mistaken for unpowered). What changed is `after`
    against the observed image, which still works when the teacher left out an action node."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    if node not in nodes:
        raise ValueError(f"unknown intervention node {node}")
    in_edges = {}
    for e in graph["edges"]:
        in_edges.setdefault(e["target"], []).append(e)
    before = {i: n["observed"] for i, n in nodes.items()}
    after = _forward(graph, nodes, in_edges, (node, value))
    affected = descendants(graph, node)
    blockers = {}
    for i in affected:
        if not after[i] and in_edges.get(i):
            blockers[i] = [e["source"] for e in in_edges[i]
                           if after[e["source"]] != e["required"] and e["source"] not in affected and e["source"] != node]
    return before, after, blockers


def text_of(graph, i, value):
    n = next(n for n in graph["nodes"] if n["id"] == i)
    return n["true_text"] if value else n["false_text"]


def to_prompt(graph, instruction, node, value):
    """Compile the intervention result into an explicit edit prompt for Muse Image."""
    before, after, blockers = intervene(graph, node, value)
    affected = descendants(graph, node)
    lines = [f"Edit: {instruction}.", f"Result: {text_of(graph, node, value)}."]
    for i in topo_order(graph):
        if i not in affected:
            continue
        t = text_of(graph, i, after[i])
        if after[i] == before[i]:
            why = [text_of(graph, b, after[b]) for b in blockers.get(i, [])]
            lines.append(f"{t} (unchanged" + (f", because {'; '.join(why)}" if why else "") + ").")
        else:
            lines.append(f"{t}.")
    lines.append("Keep every other object, its position and state exactly as in the original image.")
    return " ".join(lines), {"before": before, "after": after, "blockers": blockers}


def break_cycles(graph):
    """Drop edges that close a cycle (teacher graphs are occasionally cyclic)."""
    kept = []
    for e in graph["edges"]:
        trial = dict(graph, edges=kept + [e])
        if e["source"] not in descendants(trial, e["source"]):
            kept.append(e)
    return dict(graph, edges=kept)
