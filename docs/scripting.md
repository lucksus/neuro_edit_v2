# Scripting reference

Both binaries embed the same Rhai engine (`sim/src/script.rs`). Scripts run:

- headless: `sim --script file.rhai` or repeated `sim --eval 'code'`
- in the editor's bottom **console** (single line, `Ctrl+Enter` or the Run
  button, Up/Down history)
- in the editor's **script editor** panel (multiline, `script` checkbox in the
  toolbar, run/save-to-file/open-from-file, and "Save to net" which stores the
  script inside the `.ne2`)
- as stored scripts: `net.script_add(name, code)` attaches code to the
  network; saved `.ne2` files keep them.

## Globals and semantics

Three objects are in scope in every evaluation:

| global | type | role |
|---|---|---|
| `net` | `Net` | the shared `Network` — structure edits go here |
| `sim` | `Sim` | the simulator, built lazily on first `sim.run*` |
| `rng` | `Rng` | seeded RNG used by `rand`/`gauss`/builders |

Semantics to know:

- **Mutation invalidates the sim.** Every `net.*` call that changes
  structure drops the built sim; the next `sim.run`/`run_steps` rebuilds it
  (`t` resets, weights/state re-upload). Readout weights, samples and the
  script list only bump the editor's geometry version — they don't rebuild.
- **Run granularity is coarse.** `sim.run(ms)` runs whole-millisecond chunks;
  never call the engine once per step.
- **Int/float coercion.** All float parameters accept both `120` and `120.0`
  (args are taken as `Dynamic` and coerced); all int parameters likewise.
- **Console echo.** In the editor a non-unit result is logged as `= value`,
  errors as `! error`. Headless, `print` goes to stdout; in the editor it
  goes to the console pane.
- Editor runs are chunked (default 200 steps) so pause/abort stays
  responsive — `sim.run` returns between chunks.

`help()` prints a one-line-per-function cheat sheet.

## Building networks — `net.*`

```rhai
let id  = net.add_neuron(10, 20, 30);        // x,y,z -> new neuron index
net.clear();                                  // empty network
net.neuron_count()                            // -> INT
net.synapse_count()                           // -> INT
net.input_count()                             // -> INT
net.connect(a, b, w)                          // a->b synapse, weight w -> index
```

### Grids and random connectivity

```rhai
net.create_2d_grid("group", w, h, spacing);      // w×h plane, adds a group
net.create_3d_grid("group", w, h, d, spacing);   // w×h×d volume
net.create_connections("group", scaling, lambda, weight);
//     probability p(d) = scaling * exp(-(d/lambda)^2) per ordered pair
//     (a Gaussian in distance, as in Maass 2002 — not an exponential)
net.set_synapse_weights("group", mean, sd);      // gaussian on existing synapses
net.set_exc_inh("group", frac);                  // frac -> inhibitory kind

// per-synapse current-trace time constants (ms). 0 = default merged
// path (each synapse's current folded into the target's single decay,
// global synapse_tau). >0 gives that synapse its own exponential trace.
net.set_syn_tau("src_g", "tgt_g", tau);          // every src->tgt edge
net.set_syn_tau_dist("src_g", "tgt_g", mean, sd);// gaussian per edge
net.set_syn_tau_inh("g", tau);                   // inhibitory-source edges
net.set_syn_tau_one(src, tgt, tau);              // one edge
net.set_syn_tau_idx(i, tau);                     // by synapse index
net.set_delay(src, tgt, ms);                     // per-edge delay (ms)
net.set_conduction_speed(v);                     // units/ms; recomputes
                                                 // all delays from distance
net.syn_tau(src, tgt)                            // -> ms (0 = merged)
```

Grid builders record persistent lattice metadata on the group (member
order = lattice order: `(i2·dims0 + i0)·dims1 + i1`). The editor uses it
for the activity-plane view — the sheet only exists while the member set
still matches the lattice, so `set_pos`, `delete_neuron(s)`,
`add_to_group` or a partial subnet paste on a grid member transparently
drops the metadata and the group draws as cubes again.

### Neuron properties

```rhai
net.set_izh(id, a, b, c, d);            // Izhikevich parameters
net.kind(id)                            // -> "exc" | "inh" | "regular"
net.set_kind(id, "exc"|"inh");          // Dale's law: re-signs its synapses;
                                        // swaps in fast-spiking defaults while
                                        // the model is stock
net.set_pos(id, x, y, z);               // move + recompute axonal delays
net.pos(id)                             // -> [x,y,z]
net.neuron_name(id)                     // -> name
net.set_neuron_name(id, "name");
net.neuron("name")                      // -> index or -1
net.delete_neuron(id);
net.delete_neurons([id1, id2, …]);
```

### Groups

```rhai
net.neurons_of("group")                 // -> [id, …]
net.group_names()                       // -> ["liquid", …]
net.add_to_group("group", [ids]);       // creates the group if needed
```

### Inputs, readouts, samples

```rhai
net.add_input("src0", [[neuron_id, weight], [id2, w2], …]);
net.add_readout("ro", [unit_ids]);      // named unit list (no dedup)
net.set_readout_weights("ro", [w,…]);   // persisted in .ne2, drawn in editor
net.readout_weights("ro")               // -> [w,…]

net.samples_write("sm", [t,…], [v,…]);  // replaces the samples channel
net.samples_save("sm", "sm.bin");
net.samples_load("sm", "sm.bin");
net.samples("sm")                       // -> [[t,v],…]
```

### Persistence and stored scripts

```rhai
net.save("x.ne2");                      // binary .ne2
net.load("x.ne2");                      // replaces the whole network
let other = net_load("y.ne2");          // detached Net handle
net.script_add("train", "sim.run(100)");
net.script_remove("train");
```

## Driving the sim — `sim.*`

```rhai
sim.run(500);            // ms (converted to steps via dt)
sim.run_steps(5000);     // raw steps
sim.build();             // force-build now (also lazily on first run)
sim.rebuild();           // drop the sim; rebuild on next use
sim.time()               // -> ms
sim.reset();             // reset state, keep weights/topology
sim.rate()               // mean firing rate, Hz
sim.spikes()             // total spikes since reset
```

### State access

```rhai
sim.trace()                       // per-neuron readout trace: 1 on spike,
                                  // decays by (1 - dt/readout_tau) per step
sim.trace_of([ids])               // trace values for specific neurons
sim.v(id)                         // current membrane potential
sim.vs()                          // all membrane potentials
sim.spike_counts(rows)            // per-neuron counts over last `rows` steps
sim.spike_counts_of(rows, [ids])  // subset
```

### Inputs and configuration

```rhai
sim.set_input("src0", 0.7);            // constant value
sim.play_input("src0", [0,1,1,0,…]);   // per-step series
sim.set_external(id, amps);            // direct current injection
sim.set_noise(rate_hz, kick_mv);       // Poisson voltage kicks
sim.set_dt(0.1);                       // step size (rebuilds the ring)
sim.set_synapse_tau(5.0);              // synaptic current decay, ms
sim.set_readout_tau(20.0);             // readout feature decay, ms
sim.set_int_weights(1);                // fixed-point CUDA accumulation
                                       // -> bitwise-reproducible GPU runs
```

## Readouts and metrics

Online delta-rule readout (the historical NeuroEdit readout):

```rhai
let ro = delta_new(n_features);
ro.init_random(-0.02, 0.02);
ro.set_proper(1);                       // corrected delta rule
ro.learn(trace_features, target, rate); // one sample
ro.output(trace_features)               // -> f64 score
ro.weights()                            // -> [w,…]
```

Batch ridge regression + metrics:

```rhai
let w = fit_ridge(X_flat, n_rows, n_feat, y, lambda);
ridge_output(w, x)                      // -> f64
accuracy(scores, labels)                // 0/1 labels, threshold 0.5
auc(scores, labels)
```

## Random numbers

```rhai
rand(lo, hi)        // uniform
gauss(mean, sd)
seed(12345)         // reseeds the shared rng AND the builders' draws
```

Bit-stream generator for correlation experiments
(`rng_stream(seed, n_src, p)`): each source is i.i.d. Bernoulli(`p`).
`window(steps, event, pairs, strength)` returns a `steps × n_src` matrix;
when `event` is true, each `[i,j]` pair shares a common Bernoulli bit with
probability `strength` per step — pairwise correlation with unchanged
marginals. See `docs/tutorial-lsm-hidden.md` for the full experiment.

```rhai
let gen = rng_stream(11, 8, 0.5);
let m = gen.window(50, true, [[0,1],[2,3]], 0.8);   // 50×8 matrix
m[0][3]                                             // bit of source 3, step 0
```

## Files

```rhai
write_to_file("out.txt", [1.0, 2.5, 3]);   // one value per line
read_to_vec("out.txt")                     // -> [f64,…]
```

## Datasets

Reproducible multi-channel series live in `data_dir/datasets/<name>/`.
`dataset_gen(#{...})` generates (or loads the cached) series from a
spec; `dataset(name)` loads one (including uploads); `datasets()` lists
names. Everything is deterministic — same spec + seed => same bytes.

```rhai
let d = dataset_gen(#{
    name: "det_corr", channels: 8, len_ms: 45000, dt_ms: 1.0,
    dist: "poisson", rate_hz: 25.0, seed: 5,
    events: [#{ kind: "corr", every_ms: 500.0, jitter_ms: 150.0,
               window_ms: 12.0, strength: 35.0, seed: 17 }],
});
d.n_channels;  d.steps;  d.len_ms;          // dimensions
d.channel(i);                              // -> [f32; steps]
d.events();  d.events(lo_ms, hi_ms);       // event label times
```

Event kinds: `rate` (rate bump), `corr` (coincident burst across all
channels), `motif` (`channels:`, `pattern:`, `gap_ms:` — injected
temporal pattern). Uploads take the same path via
`ClientMsg::UploadDataset` (`.f32` or `.npy`, name confined, size cap
`server --max-upload-mb`).

## Jobs

On the WebSocket `server` a script can run as a background job (its own
`Env` + thread, survives client disconnect, `CancelJob` stops it via
the progress callback). `job_progress(#{ k: v, … })` emits numeric
fields to the editor's Jobs plot; `job_dir()` returns `runs/<id>` for
artifacts ("" when run interactively).

```rhai
job_progress(#{ seed: s, auc: test_auc, done: 1.0 });
write_to_file(job_dir() + "/result.json", [1.0]);
```

## A minimal script

```rhai
net.create_2d_grid("g", 10, 10, 50.0);
net.create_connections("g", 0.4, 80.0, 1.0);
net.set_exc_inh("g", 0.2);
sim.set_noise(20, 0.8);
sim.run(1000);
print("rate = " + sim.rate() + " Hz, spikes = " + sim.spikes());
```
