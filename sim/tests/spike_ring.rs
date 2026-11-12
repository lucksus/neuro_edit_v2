//! Spike record ring: rows are reused every `spike_record_steps` steps, so
//! stale bits must be cleared on wrap and over-long windows must not be
//! silently capped.

use net_format::{Network, NeuronKind};
use sim::cpu::CpuSim;
use sim::SimConfig;

fn rec_cfg(depth: usize) -> SimConfig {
    SimConfig {
        record_spikes: true,
        spike_record_steps: depth,
        ..Default::default()
    }
}

fn one_neuron_net() -> Network {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    net
}

/// Tonic firing wraps the ring 20x; after going quiet, the last window's
/// counts must equal the true spikes, not OR-ed stale bits.
#[test]
fn cpu_spike_bits_cleared_on_ring_wrap() {
    let mut sim = CpuSim::new(&one_neuron_net(), rec_cfg(64));
    sim.set_external(0, 40.0);
    sim.run_steps(64 * 20);
    sim.set_external(0, 0.0);
    sim.run_steps(200);
    let before = sim.spikes_total;
    sim.run_steps(64);
    let true_spikes = (sim.spikes_total - before) as u32;
    let reported = sim.spike_counts(64).unwrap()[0];
    assert_eq!(true_spikes, reported, "stale bits inflated the count");
}

/// `rows >= t` answers with exact lifetime totals; `ring < rows < t` errors
/// instead of silently capping at the ring depth.
#[test]
fn cpu_spike_counts_window_semantics() {
    let mut sim = CpuSim::new(&one_neuron_net(), rec_cfg(64));
    sim.set_external(0, 40.0);
    sim.run_steps(300); // t = 300, ring = 64

    // 5000-step window covers all of t → direct count
    let direct = sim.pn_total[0];
    assert!(direct > 0);
    assert_eq!(sim.spike_counts(5000).unwrap()[0] as u64, direct);

    // inside the ring → sliding window, bounded by the total
    let w64 = sim.spike_counts(64).unwrap()[0];
    assert!((w64 as u64) <= direct);

    // between ring and t → error, not a silent cap
    let e = sim.spike_counts(200).unwrap_err();
    assert!(e.contains("exceeds"), "got: {e}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_spike_bits_cleared_on_ring_wrap() {
    use sim::{BackendKind, Sim};
    let net = one_neuron_net();
    let mut s = Sim::new(&net, rec_cfg(64), BackendKind::Cuda).unwrap();
    s.set_external(0, 40.0);
    s.run_steps(64 * 20);
    s.set_external(0, 0.0);
    s.run_steps(200);
    let before = s.spikes_total();
    s.run_steps(64);
    let true_spikes = (s.spikes_total() - before) as u32;
    let reported = s.spike_counts(64).unwrap()[0];
    assert_eq!(true_spikes, reported, "stale bits inflated the count");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_spike_counts_window_semantics() {
    use sim::{BackendKind, Sim};
    let net = one_neuron_net();
    let mut s = Sim::new(&net, rec_cfg(64), BackendKind::Cuda).unwrap();
    s.set_external(0, 40.0);
    s.run_steps(300);
    let direct = s.spike_counts(5000).unwrap()[0];
    assert!(direct > 0);
    // equals lifetime total: spikes_total counts the single neuron
    assert_eq!(direct as u64, s.spikes_total());
    assert!(s.spike_counts(200).is_err());
}
