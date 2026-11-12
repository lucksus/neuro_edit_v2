//! P2-11: the editor v(t) plot used to get ONE sample per published chunk
//! (~200 steps), so spikes were invisible. `set_v_watch`/`drain_v` record
//! every step for watched neurons on both backends.

use net_format::{Network, NeuronKind};
use sim::cpu::CpuSim;
use sim::SimConfig;

fn net() -> Network {
    let mut net = Network::new();
    for _ in 0..4 {
        net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    }
    net
}

#[test]
fn watched_neurons_record_every_step() {
    let mut sim = CpuSim::new(&net(), SimConfig::default()).unwrap();
    sim.set_v_watch(&[0, 2]);
    sim.set_external(0, 40.0); // tonic drive -> spikes
    sim.run_steps(200).unwrap();
    let h = sim.drain_v();
    assert_eq!(h.len(), 2);
    assert_eq!(h[0].len(), 200, "one sample per step, not one per run");
    assert_eq!(h[1].len(), 200);
    // driven neuron visibly depolarizes past rest (-65 mV)
    assert!(
        h[0].iter().any(|&v| v > 0.0),
        "spike upstroke should be captured: max={:?}",
        h[0].iter().cloned().reduce(f32::max)
    );
    // undriven neuron stays near rest
    assert!(h[1].iter().all(|&v| v < -50.0));
    // drained: second call is empty until more steps run
    assert!(sim.drain_v()[0].is_empty());
    sim.run_steps(30).unwrap();
    assert_eq!(sim.drain_v()[0].len(), 30);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_drain_matches_cpu() {
    if !sim::cuda_available() {
        return;
    }
    use sim::cuda::CudaSim;
    let mut c = CpuSim::new(&net(), SimConfig::default()).unwrap();
    let mut g = CudaSim::new(&net(), SimConfig::default()).unwrap();
    c.set_v_watch(&[0, 2]);
    g.set_v_watch(&[0, 2]);
    c.set_external(0, 40.0);
    g.set_external(0, 40.0);
    c.run_steps(300).unwrap();
    g.run_steps(300).unwrap();
    let hc = c.drain_v();
    let hg = g.drain_v().unwrap();
    assert_eq!(hc.len(), hg.len());
    for k in 0..hc.len() {
        assert_eq!(hc[k].len(), hg[k].len(), "watched {k} sample count");
        // CPU/GPU arithmetic differs at fmad level (documented in NOTES);
        // samples must be step-aligned and equal within float noise.
        for (i, (a, b)) in hc[k].iter().zip(&hg[k]).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "watched {k} step {i}: cpu={a} gpu={b}"
            );
        }
    }
}
