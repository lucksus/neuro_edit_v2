# Readouts

A readout is a named list of liquid neurons whose activity a trained
model reads. Scripts make them (`net.add_readout("ro", units)`) and
train them; the editor shows and edits what they hold.

## The MLP readout

The examples train a **non-spiking multi-layer perceptron** on the
liquid's per-neuron spike rates, by backpropagation (`fit_mlp`,
`net.train_readout_mlp` — see the console's `help("readout")`).
A readout with a trained MLP draws in the scene as a small layered
stack above its liquid: one bar per layer, taller for wider layers,
labelled **128 tanh**, **64 tanh**, **4 softmax**, with thin lines
between consecutive layers. A readout without an MLP stays a small
marker cube.

## Set the layers, then Train

Click the readout's marker (the cube under the stack) to inspect it:

- **input** — how many features the MLP reads (one per unit).
- **hidden 1, 2, …** — drag or type a size; ✖ removes a layer;
  **+ layer** appends one (at most 8).
- **output** — 1 score (binary) or K softmax classes.
- **Apply** stores the layers on the readout (undoable, saved in the
  file); **Script default** clears them.
- The trained line says whether the attached MLP **matches config** or
  **needs retraining**.
- **Train: re-run …** re-runs the script that built the net. Every
  bundled MLP example reads the readout's layers before it rebuilds
  the net, trains exactly those layers, and prints the held-out numbers
  in the console.

To show the liquid first and train later: Examples -> *Deep LSM —
build + live inputs* builds the net with six looping input trains and no
training; press Run and the spikes keep flowing. Then load *Deep
readout* to train.

Try it: Examples -> *Deep readout*, wait for the numbers, click the
readout, add a layer (e.g. 128 · 64 · 32), Apply, Train.
