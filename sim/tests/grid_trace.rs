//! Grid-plane spike traces: the sim marks, per watched member list, the
//! last step each member spiked and materializes `exp(-age*dt/tau)` on
//! read — the editor's activity-plane texels.

use net_format::{Network, NeuronKind};
use sim::{BackendKind, Sim, SimConfig};

fn net9() -> Network {
    let mut net = Network::new();
    for i in 0..9 {
        net.add_neuron([i as f32 * 10.0, 0.0, 0.0], NeuronKind::Excitatory);
    }
    net
}

fn cfg() -> SimConfig {
    SimConfig {
        dt_ms: 0.1,
        ..Default::default()
    }
}

#[test]
fn trace_marks_spiked_member_and_decays() {
    let net = net9();
    let members: Vec<u32> = (0..9).collect();
    let mut sim = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    sim.set_grid_watches(&[members]);

    // force neuron 4 to spike on step 0 only
    sim.set_external(4, 1000.0);
    sim.run_steps(1).unwrap();
    sim.set_external(4, 0.0);
    sim.run_steps(99).unwrap(); // t = 100

    let tr = sim.grid_traces(20.0).unwrap().remove(0);
    assert_eq!(tr.len(), 9);
    for (k, &v) in tr.iter().enumerate() {
        if k == 4 {
            let expect = (-(sim_steps(&sim) as f32) * 0.1 / 20.0).exp();
            assert!(
                (v - expect).abs() < 1e-5,
                "t={} v={v} exp={expect}",
                sim_steps(&sim)
            );
        } else {
            assert_eq!(v, 0.0, "member {k} never spiked");
        }
    }
}

/// Member->texel mapping is by list position, not neuron id.
#[test]
fn texel_index_is_member_list_position() {
    let net = net9();
    let mut sim = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    sim.set_grid_watches(&[vec![7, 3, 5]]);
    sim.set_external(5, 1000.0);
    sim.run_steps(1).unwrap();
    sim.set_external(5, 0.0);
    let tr = sim.grid_traces(20.0).unwrap().remove(0);
    assert_eq!(tr.len(), 3);
    // neuron 5 sits at member-list position 2 -> texel 2
    assert!(tr[2] > 0.0 && tr[0] == 0.0 && tr[1] == 0.0);
}

#[test]
fn reset_clears_traces() {
    let net = net9();
    let mut sim = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    sim.set_grid_watches(&[(0..9).collect::<Vec<u32>>()]);
    sim.set_external(4, 1000.0);
    sim.run_steps(1).unwrap();
    assert!(sim.grid_traces(20.0).unwrap()[0][4] > 0.0);
    sim.reset_state();
    assert!(sim.grid_traces(20.0).unwrap()[0].iter().all(|&v| v == 0.0));
}

/// Two groups watch overlapping subsets; a spike marks each independently.
#[test]
fn multiple_groups_independent() {
    let net = net9();
    let mut sim = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    sim.set_grid_watches(&[vec![0, 1, 2], vec![2, 4]]);
    sim.set_external(2, 1000.0);
    sim.run_steps(1).unwrap();
    sim.set_external(2, 0.0);
    let trs = sim.grid_traces(20.0).unwrap();
    assert!(trs[0][2] > 0.0 && trs[0][0] == 0.0);
    assert!(trs[1][0] > 0.0 && trs[1][1] == 0.0);
}

#[test]
fn no_watches_means_no_traces() {
    let net = net9();
    let mut sim = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    sim.run_steps(5).unwrap();
    assert!(sim.grid_traces(20.0).unwrap().is_empty());
}

fn sim_steps(sim: &Sim) -> u64 {
    sim.step_count()
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_trace_matches_cpu() {
    let net = net9();
    if !sim::cuda_available() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let members: Vec<u32> = (0..9).collect();
    let mut cpu = Sim::new(&net, cfg(), BackendKind::Cpu).unwrap();
    let mut gpu = Sim::new(&net, cfg(), BackendKind::Cuda).unwrap();
    for s in [&mut cpu, &mut gpu] {
        s.set_grid_watches(&[members.clone()]);
        s.set_external(4, 1000.0);
        s.run_steps(1).unwrap();
        s.set_external(4, 0.0);
        s.run_steps(49).unwrap();
    }
    let a = cpu.grid_traces(20.0).unwrap().remove(0);
    let b = gpu.grid_traces(20.0).unwrap().remove(0);
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() < 1e-4, "texel {k}: cpu={x} cuda={y}");
    }
}
