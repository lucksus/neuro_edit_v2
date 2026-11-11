//! Network builders mirroring the original's `Group` methods:
//! `create_2d_grid`, `create_connections(scaling, lambda)`, weight randomization,
//! plus a 3D grid (new) and input wiring helpers.

use crate::rng::Rng;
use crate::{Network, NeuronKind};

/// `Group::create_2d_grid(width, height, distance)` — a w×h grid in the y–z
/// plane centred on `center`, like the original
/// (`neuron_pos = center + (0, (y-h/2)*dist, (x-w/2)*dist)`).
pub fn create_2d_grid(
    net: &mut Network,
    group: &str,
    width: usize,
    height: usize,
    distance: f32,
    center: [f32; 3],
) {
    let gi = net.add_group(group);
    let mut members = Vec::with_capacity(width * height);
    for x in 0..width {
        for y in 0..height {
            let pos = [
                center[0],
                center[1] + (y as f32 - height as f32 / 2.0) * distance,
                center[2] + (x as f32 - width as f32 / 2.0) * distance,
            ];
            members.push(net.add_neuron(pos, NeuronKind::Excitatory));
        }
    }
    net.groups[gi].members.extend(members);
}

/// 3D extension: w×h×d grid centred on `center`.
pub fn create_3d_grid(
    net: &mut Network,
    group: &str,
    width: usize,
    height: usize,
    depth: usize,
    distance: f32,
    center: [f32; 3],
) {
    let gi = net.add_group(group);
    let mut members = Vec::with_capacity(width * height * depth);
    for z in 0..depth {
        for x in 0..width {
            for y in 0..height {
                let pos = [
                    center[0] + (x as f32 - width as f32 / 2.0) * distance,
                    center[1] + (y as f32 - height as f32 / 2.0) * distance,
                    center[2] + (z as f32 - depth as f32 / 2.0) * distance,
                ];
                members.push(net.add_neuron(pos, NeuronKind::Excitatory));
            }
        }
    }
    net.groups[gi].members.extend(members);
}

/// `Group::create_connections(connection_scaling, lambda)` —
/// for every ordered pair (n1 != n2) in the group, connect with probability
/// `scaling * exp(-(distance/lambda)^2)`, as in the original (and Maass 2002).
pub fn create_connections(
    net: &mut Network,
    group: &str,
    scaling: f64,
    lambda: f64,
    weight: f32,
    rng: &mut Rng,
) {
    let members = match net.group(group) {
        Some(g) => g.members.clone(),
        None => return,
    };
    for &n1 in &members {
        for &n2 in &members {
            if n1 == n2 {
                continue;
            }
            let a = net.neuron(n1).pos;
            let b = net.neuron(n2).pos;
            let dist = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                .sqrt() as f64;
            if rng.f64() < scaling * (-(dist / lambda).powi(2)).exp() {
                net.connect(n1, n2, weight);
            }
        }
    }
}

/// `Group::set_synapse_weights(mean, var)` — gaussian weights for all synapses
/// whose target is a group member (the original iterated incoming synapses of
/// the group's neurons).
pub fn set_synapse_weights(net: &mut Network, group: &str, mean: f64, stddev: f64, rng: &mut Rng) {
    let members = match net.group(group) {
        Some(g) => g.members.clone(),
        None => return,
    };
    let in_group: std::collections::HashSet<u32> = members.into_iter().collect();
    for s in &mut net.synapses {
        if in_group.contains(&s.tgt) {
            s.w = rng.gauss(mean, stddev) as f32;
        }
    }
}

/// Assign `NeuronKind` by 80/20 excitatory/inhibitory split of the group's
/// members (seeded), a standard LSM mix the original made by hand.
pub fn set_exc_inh(net: &mut Network, group: &str, inh_fraction: f64, rng: &mut Rng) {
    let members = match net.group(group) {
        Some(g) => g.members.clone(),
        None => return,
    };
    for m in members {
        let kind = if rng.f64() < inh_fraction {
            NeuronKind::Inhibitory
        } else {
            NeuronKind::Excitatory
        };
        net.neuron_mut(m).kind = kind;
    }
}

/// Mark `count` randomly chosen members as inhibitory, the rest excitatory.
pub fn set_exc_inh_exact(net: &mut Network, group: &str, n_inhibitory: usize, rng: &mut Rng) {
    let mut members = match net.group(group) {
        Some(g) => g.members.clone(),
        None => return,
    };
    rng.shuffle(&mut members);
    for (i, &m) in members.iter().enumerate() {
        net.neuron_mut(m).kind = if i < n_inhibitory {
            NeuronKind::Inhibitory
        } else {
            NeuronKind::Excitatory
        };
    }
}
