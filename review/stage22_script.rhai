// Stage 22 screenshot setup — a deliberately long eval so the editor
// sits in Script state (worker busy, controls visibly disabled) while
// the screenshot is taken. The busy loop is plain Rhai arithmetic —
// it outlasts the app's GPU init on a slow headless render path.

net.clear();
seed(3);
net.create_2d_grid("sheet", 20, 20, 12.0);
net.create_connections("sheet", 1.0, 15.0, 14.0);

print("starting a long eval — the editor should say SCRIPT");
let x = 0;
for i in 0..1000000000 {
    x += i % 7;
}
print("done: " + x);
