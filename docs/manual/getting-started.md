# Getting started

NeuroEdit is a network editor + simulator. The 3D view shows neurons
as cubes (blue = excitatory, purple = inhibitory; both flash orange
when spiking) and synapses as thin lines (salmon = from an excitatory
source, blue = inhibitory; they turn golden when touching a selected
neuron). Small dots travel along axons when "spike dots" is on —
yellow from excitatory cells, light blue from inhibitory.

## The toolbar

- **Run / Stop / step / 100ms / Reset** drive the sim. Reset rewinds
  the sim clock to 0 and restarts all plots.
- **Speed** slider scales sim time vs wall time.
- Modes: **navigate** (V), **insert** (I), **connect** (C), **move** (M).
- Plane picker: x–z / x–y / y–z work plane for inserts and moves.
- Toggles: synapses, floor grid, spike dots, grid-view mode.

## Windows

Ctrl+1..6 toggles Inspector · Plots · Console · Script · Inputs ·
Files; the View menu does the same, and "Reset layout" restores
defaults. Every window can be dragged by its title bar and closed
with its x.

## Loading a network

File -> Open loads `.ne2` files; File -> Examples loads and runs the
bundled demo scripts. Networks can also be built in the console or
the script window (`net.add_neuron`, `net.create_2d_grid`, …) — see
the Console page.
