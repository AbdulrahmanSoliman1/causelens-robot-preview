"""Fresh test scenes for the open-vocabulary test: none of these were seen while building or tuning anything.

Half "familiar" (power, fuel, liquid in a container, ropes: covered by causal/vocab.py), half "novel" (a blocker that none
of the 49 canonical concepts can express: a flat tyre, a snow-covered solar panel, a tangled kite string, ...).
Each scenario: a robot task, one visible blocker, a blocked and an enabled photo.

  python3 fresh/make.py scenarios   -> fresh/scenarios.json
  python3 fresh/make.py images      -> fresh/images/<id>_<state>.jpg (resumable)
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from causal import api, vocab  # noqa: E402

HERE = ROOT / "fresh"
STYLE = "Photorealistic photo, natural light, eye-level, everything relevant clearly visible in frame. No people, no text."

PROMPT = """Write {n} test scenes for a home or garden robot. In each, the robot is about to do ONE task, and ONE visible
condition decides whether the task can succeed. {kind}
Rules:
- Everyday homes, gardens, garages, offices. No people, no fire hazards to people, nothing medical, legal or political.
- The blocker must be LARGE and OBVIOUS in a photo (a completely flat tyre, a solar panel buried under thick snow), not
  hidden inside a device.
- blocked_desc and enabled_desc describe the same scene, differing only in the blocker, before the task is done (the goal
  must NOT already be visible: the light is still off, the wheel is still).
- goal_true / goal_false: the visible outcome if the task works / if it doesn't ("the fountain sprays water" / "the fountain
  stays dry").
- blocker_ok / blocker_bad: the condition's working and blocking state, as short visible sentences.
- All scenes different from each other. ids snake_case."""

FAMILIAR = ("Use blockers from these familiar kinds only: unplugged power, no fuel or wood, an empty container, a closed "
            "lid or valve, a cut rope or a chain off, dead batteries.")
NOVEL = ("The blocker must be something none of these concepts can express: {vocab}. Examples of the kind wanted: a flat "
         "tyre, a solar panel covered in snow, a tangled or snagged kite string, a padlock with no key in it, a garden hose "
         "kinked shut, a frozen water bowl, a seed tray with no soil.")

SCENE = {"type": "object", "additionalProperties": False,
         "required": ["id", "device", "instruction", "goal_true", "goal_false", "blocker_ok", "blocker_bad",
                      "blocked_desc", "enabled_desc"],
         "properties": {k: {"type": "string"} for k in ("id", "device", "instruction", "goal_true", "goal_false",
                                                        "blocker_ok", "blocker_bad", "blocked_desc", "enabled_desc")}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scenes"],
          "properties": {"scenes": {"type": "array", "items": SCENE}}}


def scenarios(n=12):
    out = []
    for kind, text in (("familiar", FAMILIAR), ("novel", NOVEL.format(vocab=", ".join(sorted(vocab.NODES))))):
        res, _ = api.spark(PROMPT.format(n=n, kind=text), schema=SCHEMA, effort="low")
        out += [dict(s, kind=kind, id=f"{kind[:3]}_{k:02d}_{s['id']}") for k, s in enumerate(res["scenes"])]
    json.dump(out, open(HERE / "scenarios.json", "w"), indent=1)
    for s in out:
        print(s["kind"], s["id"], "|", s["instruction"], "| blocker:", s["blocker_bad"])


def images():
    (HERE / "images").mkdir(exist_ok=True)
    sc = json.load(open(HERE / "scenarios.json"))
    jobs = [(s, st) for s in sc for st in ("blocked", "enabled") if not (HERE / "images" / f"{s['id']}_{st}.jpg").exists()]

    def run(job):
        s, st = job
        tmp = HERE / "images" / f"{s['id']}_{st}.png"
        try:
            api.generate(f"{s[st + '_desc']} {STYLE}", tmp)
            im = Image.open(tmp).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(HERE / "images" / f"{s['id']}_{st}.jpg", quality=90)
            tmp.unlink()
        except Exception as e:
            print("ERR", s["id"], st, str(e)[:100], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(run, jobs))
    print(len(list((HERE / "images").glob("*.jpg"))), "images")


if __name__ == "__main__":
    {"scenarios": scenarios, "images": images}[sys.argv[1]]()
