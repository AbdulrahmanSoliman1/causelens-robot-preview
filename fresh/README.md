# Open vocabulary vs the fixed 49 concepts (`fresh/`, 2026-11-03)

Question: the robot pipeline only knew 49 hand-written concepts (`causal/vocab.py`) and a hand-picked list of "world-state"
nodes. Does an open vocabulary, where Muse Spark names this scene's parts and tags each node with a role, do as well, and
does it handle blockers outside the 49?

- `causal/open.py`: Muse Spark drafts the graph with free node names and roles (action / condition / effect) plus a goal.
  The same general rules as `causal/robust.py`, keyed on roles instead of ids: conditions are roots, the action is the
  intervention, every condition is read again (yes / no / can't tell / not in the scene; missing counts as no), and what
  can't be seen is assumed working and listed.
- Fresh scenes (`make.py`), never seen while building or tuning: Muse Spark wrote 12 "familiar" scenarios (power, fuel,
  containers, ropes, chains) and 12 "novel" ones whose blocker none of the 49 concepts can express (flat tyre, snow on a solar
  panel, tangled kite string, kinked hose, missing key, empty paper tray, a branch across the path ...). Muse Image drew a
  blocked and an enabled photo of each.
- Eye gate (`gate.json`): a scene stays only if both photos clearly show their condition and neither already shows the goal.
  18 of 24 kept (7 familiar, 11 novel). Dropped: fuel, water or batteries that can't be seen, an ambiguous tap, a "blocked"
  fireplace that already held logs, and a mud-caked mower blade that might cut anyway.
- Methods, run once (`run.py`): `open`; `canon` (the robust 49-concept version); `ask` (one Muse Spark yes/no question at
  low effort, which also gets the goal sentence from the spec). Ground truth comes from the scenario.

| Method | Familiar (14) | Novel (22) | All (36) | Both states right (18 scenes) | Median time |
|---|---|---|---|---|---|
| **open** | 11 | **18** | **29 (81%)** | **11** | 25 s |
| canon (49 concepts) | 9 | 13 | 22 (61%) | 5 | 20 s |
| ask Muse Spark | 11 | 15 | 26 (72%) | 8 | 7 s |

- **The fixed vocabulary fails outside its list, as expected.** It called the flat tyre, the snow-covered solar mower, the
  kinked hose and the branch across the path "works", because it has no concept for them; it reached for "batteries" or
  "water supply" instead.
- **The open vocabulary handles them:** 18/22 novel decisions right.
- **Asking Muse Spark directly is pessimistic:** 17/18 blocked right but only 9/18 working scenes, so it mostly says "no".
  Counting scenes right in both states, which a method that always says "no" can't game: open 11, ask 8, canon 5.
- Paired: open vs canon 12–5 (p = 0.14); open vs ask 7–4 (p = 0.55). **Small sample, not significant.** The direction
  matches the prediction, but this is evidence, not proof.
- Open's errors: an invented extra condition read as failing (a dry fountain basin, a rolled-up rug edge, a step "too high",
  an unplugged printer cable), and hidden conditions assumed working (seeds in a drone hopper, the washer's power).
- Caveats: one labeller wrote the gate; the scenes are generated (many novel ones include a robot); open is the slowest.

## Switching `/robot` to the open version, and a category checklist (2026-11-03)

- On the robot page's demo scenes, the open version was unstable: the unplugged heater was called "works" in 3 of 4 runs.
  It never asked whether the heater was plugged in, only whether the outlet had power. The fixed vocabulary had that check built in.
- Fix: the drafting prompt now asks for the everyday preconditions of the task's *kind*, not of named concepts: anything
  electric must be plugged in and powered; anything that holds or dispenses something must not be empty; anything that
  moves must have its parts attached and nothing blocking it; anything that burns must have fuel. Heater blocked 4/4 (once
  blamed on outlet power), bike works 4/4.
- Re-run on the fresh scenes (a **second look** at them, labelled `open2`):

| Method | Familiar (14) | Novel (22) | All (36) | Both states right |
|---|---|---|---|---|
| **open + category checklist** | 11 | **19** | **30** | **12/18** |
| open | 11 | 18 | 29 | 11/18 |
| canon (49 concepts) | 9 | 13 | 22 | 5/18 |
| ask Muse Spark | 11 | 15 | 26 | 8/18 |

- Paired: open2 vs canon 11–3 (p = 0.057); vs ask 7–3 (p = 0.34); vs open 2–1. The checklist didn't cost accuracy.
- Remaining errors are invented extra conditions (no clothespins, a patio step "too high", a clogged pump intake) and
  hidden ones assumed fine (string lights left unplugged off-frame, seed tray soil).
- Honest note: the checklist is a short, general hint we wrote, about kinds of task rather than specific objects.

## Literal task rule (2026-11-03, `open3`, third look at the fresh scenes)

- In a page run, the heater was called "works" again: the model treated "the cord is plugged in" as something the robot's
  own action would do ("turn on the heater" → plug it in and switch it on).
- Rule added to the draft: the robot does exactly the task and nothing else (no plugging in, refilling, repairing or clearing
  unless the task says so); whatever such a step would change is a condition, read as it is now.
- Heater blocked 6/6, all for the right reason; bike works 3/3; empty cooler blocked 3/3.
- Fresh scenes: 30/36 again, but **blocked 18/18**: on these scenes it never called a blocked action "works". All 6 errors are
  false alarms (a kickstand down, a washer door left open that the literal robot won't close, a dry fountain basin, an empty
  feeder, an "unplugged" printer cable, an "obstacle" on the floor). For a safety preview that is the safer kind of mistake.
  Both states right 12/18. vs canon 11–3 (p = 0.057), vs ask 6–2.
