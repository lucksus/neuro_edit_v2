// Grid wave: a 20x20 sheet with distance-decaying random connections,
// kicked on the left edge — activity rolls across the plane. Press Run,
// then watch the scene (GridView: auto) or Plots → raster.

net.clear();
seed(3);

net.create_2d_grid("sheet", 20, 20, 12.0);
net.set_exc_inh("sheet", 0.2);              // 20% inhibitory
net.create_connections("sheet", 1.0, 3.0, 6.0); // denser near, sparse far

// Drive the left column (every other row).
let sheet = net.neurons_of("sheet");
let drive = [];
for r in 0..10 {
    drive.push([sheet[r * 40], 9.0]);
}
net.add_input("kick", drive);
sim.set_input("kick", 1.0);
sim.set_noise(2.0, 0.0); // a whisper of background noise

sim.run(3000.0); // 3 s
print("mean rate over the sheet: " + sim.rate() + " Hz");
print("spikes: " + sim.spikes() + " — tilt the camera low to see the wave");
