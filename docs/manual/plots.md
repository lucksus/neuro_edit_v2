# Plots

The Plots window (Ctrl+2) has tabs:

- **v(t)** — membrane voltage of watched neurons.
- **v–u** — phase portrait (v on x, u on y) of the same watched
  neurons; the newest point is marked.
- **raster** — spike dots; watched neurons get labelled rows,
  otherwise the whole id space is sampled.
- **readouts** — trained weights of the selected readout.
- **samples** — traces recorded by scripts (`net.samples_*`).
- **jobs** — server-side background jobs.

## Watching

v(t) and v–u plot *watched* neurons. Watch by selecting — click a
neuron, box-select a few, right-click → "Watch v(t)", or use the
inspector's "Plot v(t)" button. Up to 8 neurons are watched at once.

## Window + follow

v(t), v–u and raster show a sliding window (default 200 ms,
adjustable) ending at "now" while **follow** is on. Dragging or
zooming the plot turns follow off and reveals a **Follow** button;
double-clicking the plot also re-engages it. Resetting the sim (any
cause) restarts the traces and snaps back to follow.

The v axis is pinned to −90…+40 mV unless **Auto Y** is on, so spikes
don't bounce the scale.
