// Stage 22 screenshot setup — the grid_wave net WITHOUT a run, so the
// editor sits in Build state (net visible, no sim yet).

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
print("build-only net ready — " + sheet.len() + " neurons, no sim yet");
