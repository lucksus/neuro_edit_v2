// First spikes: five neurons driven by one steady input. Press Run and
// open Plots → v(t): the editor watches ids 0..4 automatically when this
// script was loaded from the Examples picker, so a trace appears within
// a second. Look for: the input pushing neurons over threshold, spikes
// in the raster, v snapping back after each spike.

net.clear();
seed(7);

// A loose ring of five neurons ~60 apart.
for i in 0..5 {
    net.add_neuron(60.0 * i - 120.0, 0.0, 0.0);
}
// Chain them so spikes propagate left to right. A single incoming spike
// has to push a resting neuron over threshold on its own, which takes a
// weight of ~25+; the classic demo weight is 30.
for i in 0..4 {
    net.connect(i, i + 1, 30.0);
}
// One input drives neuron 0 hard enough to fire.
net.add_input("stim", [[0, 12.0]]);
sim.set_input("stim", 1.0);

sim.run(2000.0); // 2 s of sim time
print("spikes so far: " + sim.spikes());
print("mean rate: " + sim.rate() + " Hz");
print("per-neuron spike counts: " + sim.spike_counts(20000));
print("Plots → v(t) / raster show neurons 0-4 (auto-watched)");
