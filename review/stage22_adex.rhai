// Stage 22 screenshot setup — grid_wave, but neuron 0 (a driven
// left-edge cell) is converted to AdEx so the inspector shows the
// descriptor-driven v + w state plots instead of the izh map.

net.clear();
seed(3);

net.create_2d_grid("sheet", 20, 20, 12.0);
net.set_exc_inh("sheet", 0.2);
net.create_connections("sheet", 1.0, 15.0, 14.0);

let sheet = net.neurons_of("sheet");
let drive = [];
for r in 0..10 {
    drive.push([sheet[r * 2], 9.0]);
}
net.add_input("kick", drive);
sim.set_input("kick", 1.6);
sim.set_noise(2.0, 0.0);

// The AdEx conversion — defaults for kind "exc" (the tonic row).
net.def_type("adex_tonic", "adex", #{});
net.set_type(sheet[0], "adex_tonic");
print("adex neuron id: " + sheet[0] + "  (" + net.type_name(sheet[0]) + ")");

sim.run(3000.0);
print("mean rate over the sheet: " + sim.rate() + " Hz");
