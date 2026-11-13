// Grid wave: a 20x20 sheet with distance-decaying random connections,
// kicked on the left edge — activity rolls across the plane. Press Run,
// then watch the scene (GridView: auto) or Plots → raster.

net.clear();
seed(3);

net.create_2d_grid("sheet", 20, 20, 12.0);
net.set_exc_inh("sheet", 0.2);               // 20% inhibitory
// p(edge) ≈ exp(-(d/lambda)^2): lambda must be at least the 12-unit
// spacing or the sheet stays empty; weight 14 lets a spiking neighbor
// push the next cell over threshold.
net.create_connections("sheet", 1.0, 15.0, 14.0); // denser near, sparse far

// Drive the left column (every other row). Group members run column-
// major: sheet[0..20] is the left edge, so every other row is r * 2.
let sheet = net.neurons_of("sheet");
let drive = [];
for r in 0..10 {
    drive.push([sheet[r * 2], 9.0]);
}
net.add_input("kick", drive);
sim.set_input("kick", 1.0);
sim.set_noise(2.0, 0.0); // a whisper of background noise

sim.run(3000.0); // 3 s
print("mean rate over the sheet: " + sim.rate() + " Hz");
print("spikes: " + sim.spikes() + " — tilt the camera low to see the wave");
