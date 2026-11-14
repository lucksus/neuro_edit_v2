# Getting started

The first time you open the editor it walks you through a seven-step
tutorial. Each step ends when the app sees you really do the thing —
there is no "Next" button, only the actual gesture. Skip any time from
the bar at the bottom, and reopen it later via **Help > Tutorial**
(it asks before clearing an open network — Undo restores it).

1. **Add a neuron** — pick `insert` in the left tools, then click
   anywhere in the scene.
2. **Add an input** — **Insert > Input… > Create**. With the neuron
   selected the input attaches to it right away.
3. **Connect input → neuron** — pick `connect`, then drag from the
   input's marker onto the neuron.
4. **Run it** — press **RUN** on the left. The status bar's `rate`
   rises above 0 Hz as the cell fires.
5. **Grow the net** — the run pauses for edits. **Insert > 2D grid…**
   (a 2×2 grid adds 4 neurons at once) or a few more `insert` clicks —
   you're done at five neurons.
6. **Inspect a neuron** — click one in the scene; the inspector on the
   right shows its live v(t) trace.
7. **Script it** — the docked console opens prefilled with a snippet
   that builds a small network. Press **Ctrl+Enter** (or its **Run**).
   For real experiments look at **Help > Examples** — the LSM
   walkthrough lives in `docs/tutorial-lsm-hidden.md`.

After the last step (or a skip) the full UI comes back. The editor
remembers done-vs-skipped, so it won't auto-start again; it asks only
on a fresh install, or from Help > Tutorial whenever you want.

## Scripting

Everything the mouse can do, the console can do with the Rhai API —
`net.create_2d_grid`, `net.create_connections`, `sim.run`, `sim.set_*`,
readouts, watches, and more. `docs/scripting.md` has the complete
reference and `examples/` holds ready-made scripts; the in-app Help >
Examples menu runs them in place.
