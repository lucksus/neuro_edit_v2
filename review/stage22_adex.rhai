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
// Stage 22B: the shared kick's ~14 pA sat ~40x under the ~600 pA a
// tonic AdEx needs to cross VT — the cell rested at -70 mV and Auto Y
// stretched its noise floor into a fake "spike" in the Plots window
// while the inspector honestly drew a flat line. Give the converted
// cell its own ~900 pA drive so it genuinely spikes.
net.add_input("adex_drive", [[sheet[0], 900.0]]);
sim.set_input("adex_drive", 1.0);
print("adex neuron id: " + sheet[0] + "  (" + net.type_name(sheet[0]) + ")");

sim.run(3000.0);
print("mean rate over the sheet: " + sim.rate() + " Hz");
