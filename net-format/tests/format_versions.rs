//! Format-version fixtures: one small `.ne2` written in each of the
//! v1/v2/v3 layouts (see `examples/gen_format_fixtures.rs`) must still
//! load — through the migration loaders — into the same typed net the
//! v4 format stores natively.

use net_format::{io, Network, NeuronKind, NeuronType};

const V1: &[u8] = include_bytes!("fixtures/v1.ne2");
const V2: &[u8] = include_bytes!("fixtures/v2.ne2");
const V3: &[u8] = include_bytes!("fixtures/v3.ne2");

fn load_bytes(name: &str, b: &[u8]) -> Network {
    let dir = std::env::temp_dir().join(format!("ne2fx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, b).unwrap();
    io::load(&p).unwrap()
}

/// Every fixture holds the same logical net: 4 neurons (2 stock RS
/// excitatory, 1 stock FS inhibitory, 1 custom-param excitatory),
/// 2 synapses (0→1 w 1.5, 2→0 w 2.0), a 3-member group, an input, a
/// script. The lift must produce the shared "exc"/"inh" types plus one
/// hashed custom type, in first-use order.
fn check_lifted(net: &Network) {
    assert_eq!(net.neurons.len(), 4);
    assert_eq!(net.synapses.len(), 2);
    assert_eq!(net.types.len(), 3);
    assert_eq!(net.types[0].name, "exc");
    assert_eq!(net.types[0].model, "izh");
    assert_eq!(net.types[1].name, "inh");
    assert!(net.types[2].name.starts_with("izh#"));

    // per-neuron ty in first-use order; both RS cells share a type
    assert_eq!(net.neurons[0].ty, 0);
    assert_eq!(net.neurons[1].ty, 0);
    assert_eq!(net.neurons[2].ty, 1);
    assert_eq!(net.neurons[3].ty, 2);
    assert!(net.neurons.iter().all(|n| n.params.is_none()));

    // effective params and kinds match the original rows exactly
    assert_eq!(net.neuron_params(0), &[0.02, 0.2, -65.0, 8.0]);
    assert_eq!(net.neuron_params(2), &[0.1, 0.2, -65.0, 2.0]);
    assert_eq!(net.neuron_params(3), &[0.03, 0.25, -60.0, 4.0]);
    assert_eq!(net.neuron_kind(0), NeuronKind::Excitatory);
    assert_eq!(net.neuron_kind(2), NeuronKind::Inhibitory);

    // Dale's law still signs the inhibitory source at CSR build
    let csr = net.synapse_csr(0.1).unwrap();
    assert_eq!(csr.w, vec![1.5, -2.0]);
    assert_eq!(net.groups[0].members, vec![0, 1, 2]);
    assert_eq!(net.inputs[0].name, "i");
    assert_eq!(net.scripts[0].1, "print(1)");
}

#[test]
fn v1_fixture_loads_into_typed_net() {
    let net = load_bytes("v1.ne2", V1);
    check_lifted(&net);
    // v1 predates Synapse::tau_ms — lifted to 0 = merged global path
    assert_eq!(net.synapses[0].tau_ms, 0.0);
    assert!(net.groups[0].grid.is_none());
}

#[test]
fn v2_fixture_loads_into_typed_net() {
    let net = load_bytes("v2.ne2", V2);
    check_lifted(&net);
    assert_eq!(net.synapses[0].tau_ms, 0.0);
}

#[test]
fn v3_fixture_loads_into_typed_net() {
    let net = load_bytes("v3.ne2", V3);
    check_lifted(&net);
    // v3 already had per-synapse tau — must survive the lift
    assert_eq!(net.synapses[0].tau_ms, 7.5);
}

/// v4 save → load keeps the type table, neuron ty indices, and a
/// per-neuron param override — plus everything else already covered by
/// io::tests::roundtrip.
#[test]
fn v4_round_trip_keeps_types_and_overrides() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let b = net.add_neuron([60.0; 3], NeuronKind::Excitatory);
    net.set_izh(b, 0.03, 0.25, -60.0, 4.0); // per-neuron override
    net.types.push(NeuronType {
        name: "burst".into(),
        model: "izh".into(),
        params: vec![0.02, 0.2, -55.0, 4.0],
        kind: NeuronKind::Excitatory,
    });
    let c = net.add_neuron([120.0; 3], NeuronKind::Excitatory);
    net.neuron_mut(c).ty = net.type_index("burst").unwrap();
    net.connect(a, b, 1.0);

    let dir = std::env::temp_dir().join(format!("ne2fx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rt4.ne2");
    io::save(&net, &path).unwrap();
    let back = io::load(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(back.types.len(), 2);
    assert_eq!(back.types[1].name, "burst");
    assert_eq!(back.neuron_params(b), &[0.03, 0.25, -60.0, 4.0]);
    assert_eq!(
        back.neurons[c as usize].ty,
        back.type_index("burst").unwrap()
    );
    assert_eq!(back.neuron_params(c), &[0.02, 0.2, -55.0, 4.0]);
}

/// Subnet copy/paste keys types on name: pasting a subnet that uses a
/// custom-named type brings the type along, and pasting it twice reuses
/// the same row instead of duplicating it.
#[test]
fn subnet_copy_paste_preserves_type_names() {
    let mut src = Network::new();
    src.types.push(NeuronType {
        name: "burst".into(),
        model: "izh".into(),
        params: vec![0.02, 0.2, -55.0, 4.0],
        kind: NeuronKind::Excitatory,
    });
    let a = src.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let b = src.add_neuron([60.0; 3], NeuronKind::Excitatory);
    src.neuron_mut(b).ty = src.type_index("burst").unwrap();
    src.connect(a, b, 1.0);

    let sub = src.subnet(&[a, b].into_iter().collect());
    assert!(sub.types.iter().any(|t| t.name == "burst"));

    let mut dst = Network::new();
    dst.add_neuron([500.0; 3], NeuronKind::Excitatory);
    let map = dst.insert_subnet(&sub, [100.0; 3]);
    let t = dst.type_index("burst").expect("type name carried across");
    assert_eq!(dst.neurons[map[1] as usize].ty, t);
    assert_eq!(dst.neuron_params(map[1]), &[0.02, 0.2, -55.0, 4.0]);

    // second paste reuses the same-named type — no "burst2"
    let types_before = dst.types.len();
    dst.insert_subnet(&sub, [200.0; 3]);
    assert_eq!(dst.types.len(), types_before);
    assert_eq!(dst.types.iter().filter(|t| t.name == "burst").count(), 1);
}
