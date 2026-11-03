"""Edit an image in one of three modes: direct, rewrite (Muse Spark prompt rewrite), scm (causal graph + do()),
canon (scm with node ids restricted to causal/vocab.py, the encoder's label space)."""
import time
from . import api, scm, spark


def run(image, instruction, mode, dest, effort=None):
    t0 = time.time()
    info = {"mode": mode}
    if mode == "direct":
        prompt = instruction
    elif mode == "rewrite":
        prompt, info["usage"] = spark.rewrite(image, instruction, effort)
    elif mode in ("scm", "canon"):
        build = spark.build_graph if mode == "scm" else spark.build_canon_graph
        graph, info["usage"] = build(image, instruction, effort)
        iv = graph["intervention"]
        prompt, info["scm"] = scm.to_prompt(graph, instruction, iv["node"], iv["value"])
        info["graph"] = graph
    else:
        raise ValueError(mode)
    info["prompt"] = prompt
    info["layer_s"] = round(time.time() - t0, 2)
    t1 = time.time()
    # Content-filter false positives: rephrase and retry, up to twice (same policy for every mode).
    for attempt in range(3):
        try:
            api.edit(image, prompt, dest)
            break
        except RuntimeError as e:
            if "content_policy_violation" not in str(e) or attempt == 2:
                raise
            prompt = info["prompt"] = spark.rephrase_safe(prompt)
            info["rephrased"] = attempt + 1
    info["edit_s"] = round(time.time() - t1, 2)
    return info
