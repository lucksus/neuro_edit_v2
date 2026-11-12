# Console

The console (Ctrl+3) evaluates one line of Rhai against the live
network. Up/down walk history.

## help

- `help` or `help()` — the whole API
- `help("topic")` — one section: `net`, `grid`, `syn`, `input`,
  `sim`, `readout`, `io`, `streams`, `misc`, `examples`
- `examples` — lists the bundled example scripts

## The `net` and `sim` handles

`net` mutates the network (`net.add_neuron(0,0,0)`,
`net.connect(0,1,2.5)`, `net.create_2d_grid("g",20,20,10)`,
`net.add_input("stim", [[0, 30.0]])`).

`sim` drives the simulation (`sim.run(100)`, `sim.reset()`,
`sim.v(0)`, `sim.trace()`, `sim.play_input("stim", arr)`).

Multi-line programs go in the Script window (Ctrl+4) — the console is
one-liners only. `import` is confined to the script's directory.
