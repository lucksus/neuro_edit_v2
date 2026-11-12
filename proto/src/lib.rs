//! Wire protocol between a neuro_edit server and its clients (native
//! editor now, browser later).
//!
//! Everything that crosses the line is a `ClientMsg` or a `ServerMsg`;
//! on a WebSocket these are bincode-encoded binary frames. The local
//! (same-process) transport passes the same structs over channels, so
//! the UI never knows which side of a socket the sim lives on.
//!
//! Versioning: the first frame a client sends is `ClientMsg::Hello`
//! carrying `PROTO_VERSION`; the server answers with `ServerMsg::Hello`
//! carrying its own. A mismatch is a hard error — the client disconnects.

use serde::{Deserialize, Serialize};

/// Bump when any wire type changes shape or meaning.
pub const PROTO_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// shared wire copies of host types (proto must not depend on `sim`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
}

impl From<net_format::NeuronKind> for NeuronKind {
    fn from(k: net_format::NeuronKind) -> Self {
        match k {
            net_format::NeuronKind::Inhibitory => Self::Inhibitory,
            _ => Self::Excitatory,
        }
    }
}

impl From<NeuronKind> for net_format::NeuronKind {
    fn from(k: NeuronKind) -> Self {
        match k {
            NeuronKind::Inhibitory => Self::Inhibitory,
            NeuronKind::Excitatory => Self::Excitatory,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    Cpu,
    Cuda,
}

/// The sim-config fields a client needs to mirror (toolbar, dt-aware plots).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub dt_ms: f32,
    pub synapse_tau_ms: f32,
    pub readout_tau_ms: f32,
    pub noise_rate_hz: f32,
    pub noise_kick_mv: f32,
    pub seed: u64,
    pub int_weights: bool,
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

/// One group's lattice, ready to render as an activity plane: metadata plus
/// the member list re-ordered into texel order (position k = lattice k).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    /// Index into `Geo.groups`.
    pub group: usize,
    /// Texel counts per lattice dim `[w, h, d]` — `d == 1` is a 2D grid.
    pub dims: [u32; 3],
    /// World axis (0=x, 1=y, 2=z) each lattice dim runs along.
    pub axes: [u8; 3],
    /// World position of lattice texel (0,0,0).
    pub origin: [f32; 3],
    /// World distance between neighbouring texels.
    pub spacing: f32,
    /// Texel-ordered member list (neuron ids), `len == w*h*d`.
    pub members: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub name: String,
    pub pos: [f32; 3],
    pub targets: Vec<u32>,
}

/// Immutable geometry snapshot of the whole network — the client's mirror
/// of the server's `Network` restricted to what the UI can render.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Geo {
    /// Monotonic counter bumped on every accepted edit/script change.
    pub version: u64,
    /// Replacement generation — bumped by New/load/`net.clear()`. A change
    /// means every neuron-id-keyed client state is stale.
    pub gen: u64,
    pub pos: Vec<[f32; 3]>,
    /// 0 = excitatory, 1 = inhibitory.
    pub kind: Vec<u8>,
    pub names: Vec<String>,
    /// (src, tgt) pairs, source order.
    pub syn: Vec<(u32, u32)>,
    /// Per-synapse delay in sim steps (spike-dot travel time).
    pub syn_delay_steps: Vec<u16>,
    /// First K display synapses per neuron: (tgt, delay_steps) flattened.
    pub disp_edges: Vec<(u32, u16)>,
    /// Start index of neuron i's display edges in `disp_edges`.
    pub disp_ptr: Vec<u32>,
    pub groups: Vec<(String, Vec<u32>)>,
    pub inputs: Vec<Input>,
    /// (name, units, weights).
    pub readouts: Vec<(String, Vec<u32>, Vec<f32>)>,
    /// (name, times, values).
    pub samples: Vec<(String, Vec<f32>, Vec<f32>)>,
    /// (name, code) — scripts saved in the file.
    pub scripts: Vec<(String, String)>,
    /// Grid groups with members in texel order.
    pub grids: Vec<Grid>,
    /// Neuron ids not covered by any grid group — the index space
    /// `Activity::spike_bits`/`non_grid_glow` are coded against.
    pub non_grid: Vec<u32>,
    pub dt_ms: f32,
}

/// A changed neuron row (id stays stable; pos/kind/name may differ).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuronRow {
    pub id: u32,
    pub pos: [f32; 3],
    pub kind: u8,
    pub name: String,
}

/// A changed synapse row for deltas.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynRow {
    pub src: u32,
    pub tgt: u32,
    pub delay_steps: u16,
}

/// Incremental geometry update for one accepted `NetEdit`. Edits that
/// renumber neuron ids (deletion) are NOT delta-able — the server sends a
/// full `Geo` for those instead.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GeoDelta {
    pub version: u64,
    pub gen: u64,
    /// Appended rows — their ids are `old_len .. old_len + added.len()`.
    pub neurons_added: Vec<NeuronRow>,
    /// In-place changed rows.
    pub neurons_changed: Vec<NeuronRow>,
    pub syn_added: Vec<SynRow>,
    /// (src, tgt) pairs removed.
    pub syn_removed: Vec<(u32, u32)>,
    /// Per-source display-edge lists that changed: (src, edges).
    pub disp_changed: Vec<(u32, Vec<(u32, u16)>)>,
    /// Groups whose content changed: (group idx, name, members).
    pub groups_changed: Vec<(u32, String, Vec<u32>)>,
    /// Groups whose count changed — appended groups (idx, name, members).
    pub groups_added: Vec<(u32, String, Vec<u32>)>,
    /// New/replaced grids (keyed by `Grid::group`).
    pub grids_changed: Vec<Grid>,
    /// Group indices whose grid metadata was invalidated.
    pub grids_removed: Vec<u32>,
    /// Full replacement when changed (small collections).
    pub inputs: Option<Vec<Input>>,
    pub readouts: Option<Vec<(String, Vec<u32>, Vec<f32>)>>,
    pub samples: Option<Vec<(String, Vec<f32>, Vec<f32>)>>,
    pub scripts: Option<Vec<(String, String)>>,
    /// Replacement non-grid index set (recomputed when geometry shifts).
    pub non_grid: Vec<u32>,
    pub dt_ms: f32,
}

// ---------------------------------------------------------------------------
// edits (the data version of today's EditNet closures)
// ---------------------------------------------------------------------------

/// A network mutation. Sent as `ClientMsg::Edit { base, edit }`; the server
/// applies it only if `base` equals its current `net_version`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NetEdit {
    AddNeuron {
        pos: [f32; 3],
        kind: NeuronKind,
    },
    /// Always answered with a full Geo (deletion renumbers ids).
    DeleteNeurons {
        ids: Vec<u32>,
    },
    Connect {
        from: u32,
        to: u32,
        w: f32,
    },
    Disconnect {
        from: u32,
        to: u32,
    },
    /// Set absolute positions (post-drag). Server refreshes delays and
    /// revalidates grid metadata.
    MoveNeurons {
        items: Vec<(u32, [f32; 3])>,
    },
    SetNeuronName {
        id: u32,
        name: String,
    },
    SetNeuronPos {
        id: u32,
        pos: [f32; 3],
    },
    /// Paste a serialized subnet at an offset.
    InsertSubnetJson {
        json: String,
        offset: [f32; 3],
    },
    CreateGrid2d {
        group: String,
        w: u32,
        h: u32,
        spacing: f32,
        origin: [f32; 3],
    },
    CreateGrid3d {
        group: String,
        w: u32,
        h: u32,
        d: u32,
        spacing: f32,
        origin: [f32; 3],
    },
    AddInput {
        name: String,
        targets: Vec<(u32, f32)>,
        pos: [f32; 3],
    },
    /// Create (or clear) a named samples series.
    AddSamples {
        name: String,
    },
    AddReadout {
        name: String,
        units: Vec<u32>,
    },
    /// Add ids to a group, creating the group when missing.
    AddToGroup {
        name: String,
        ids: Vec<u32>,
    },
}

// ---------------------------------------------------------------------------
// messages
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    /// First frame on a connection.
    Hello { proto: u32 },
    /// Re-request full state (Geo + Status + GridTraces) — used on
    /// (re)connect and by later subscribers on a shared run.
    Subscribe,
    /// Run a Rhai script on the worker.
    Eval { code: String },
    /// Replace the whole network from a .ne2 path resolved server-side.
    LoadNet { path: String },
    /// Replace with an empty network.
    NewNet,
    /// Continuous run at `speed × realtime`.
    AutoRun(bool),
    SetSpeed(f32),
    /// Single synchronous run of `ms` (toolbar Step).
    RunMs(f32),
    /// Abort a chunked run / script in flight.
    Abort,
    Reset,
    /// Neuron ids to record v(t) for (plots).
    Watch(Vec<u32>),
    /// Apply an edit against `base` — refused when `base` is stale.
    Edit { base: u64, edit: NetEdit },
    /// Serialize the subnet on these ids to JSON; the server answers with
    /// `ServerMsg::Clipboard`.
    CopySubnet { ids: Vec<u32> },
    SetBackend(Backend),
    SetDtMs(f32),
    SetNoise { hz: f32, kick: f32 },
    SetInputConst { name: String, value: f32 },
    PlayInput { name: String, series: Vec<f32> },
    SetExternal { neuron: u32, amps: f32 },
    /// Polite disconnect.
    Bye,
}

/// Sparse u8 texel update for one grid plane: (texel index, new value).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridDelta {
    pub grid: u32,
    pub texels: Vec<(u32, u8)>,
}

/// The per-publish dynamic frame — rate-limited by the server
/// (≤ ~30 Hz). All large fields are sparse or quantized so a
/// 1M-neuron grid stays inside a few MB/s.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Activity {
    pub seq: u64,
    pub t_steps: u64,
    pub time_ms: f64,
    pub spikes_total: u64,
    pub rate_hz: f64,
    /// Measured wall-time speed (steps/s).
    pub steps_per_sec: f32,
    /// Bit i set ⇔ `Geo.non_grid[i]` spiked since the previous Activity.
    /// Grid members don't appear here — their texel deltas carry it.
    pub spike_bits: Vec<u8>,
    /// u8-quantized activity glow for non-grid neurons (same indexing as
    /// `spike_bits`).
    pub non_grid_glow: Vec<u8>,
    /// New v samples for watched neurons: (neuron id, samples).
    pub watched_v: Vec<(u32, Vec<f32>)>,
    /// Per-grid u8 texel changes since the previous Activity.
    pub grid_delta: Vec<GridDelta>,
    /// Live constant drive per input (same order as `Geo.inputs`).
    pub input_values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerMsg {
    /// Answer to `ClientMsg::Hello`.
    Hello {
        proto: u32,
        net_version: u64,
        net_gen: u64,
    },
    /// Full geometry — on connect, on net replacement, and for edits that
    /// can't be expressed as a delta (neuron deletion renumbers).
    Geo(Box<Geo>),
    /// Incremental geometry after an accepted edit.
    GeoDelta(Box<GeoDelta>),
    /// Non-activity worker state — sent on connect and whenever it changes.
    Status {
        running: bool,
        busy: bool,
        sim_built: bool,
        backend: Backend,
        cfg: Config,
        notice: Option<String>,
    },
    /// One console line (print results, eval errors, log).
    Console(String),
    /// Rate-limited dynamic frame.
    Activity(Box<Activity>),
    /// Full u8 trace planes — on connect and after geometry changes so the
    /// client's delta accumulators start from truth.
    GridTraces {
        /// `Activity.seq` this snapshot lines up with.
        base_seq: u64,
        grids: Vec<Vec<u8>>,
    },
    /// Answer to `ClientMsg::CopySubnet` — JSON for the system clipboard.
    Clipboard(String),
    /// Refused edit / protocol error / worker failure worth a dialog.
    Error(String),
}

/// A `ServerMsg` plus routing: `to: None` broadcasts to every connected
/// client; `Some(client)` targets one. The local transport uses `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerEvent {
    pub to: Option<u64>,
    pub msg: ServerMsg,
}

impl ServerEvent {
    pub fn broadcast(msg: ServerMsg) -> Self {
        Self { to: None, msg }
    }
    pub fn to(client: u64, msg: ServerMsg) -> Self {
        Self {
            to: Some(client),
            msg,
        }
    }
}

// ---------------------------------------------------------------------------
// codec
// ---------------------------------------------------------------------------

/// Encode a message as one WebSocket binary frame's payload.
pub fn encode<T: Serialize>(m: &T) -> Vec<u8> {
    bincode::serialize(m).expect("proto encode")
}

/// Decode one payload. `Err` on malformed bytes or trailing data.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, bincode::Error> {
    bincode::deserialize(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T>(m: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let bytes = encode(m);
        let back: T = decode(&bytes).unwrap();
        assert_eq!(*m, back);
    }

    #[test]
    fn client_msgs_round_trip() {
        for m in [
            ClientMsg::Hello {
                proto: PROTO_VERSION,
            },
            ClientMsg::Subscribe,
            ClientMsg::Eval {
                code: "sim.run(10)".into(),
            },
            ClientMsg::LoadNet {
                path: "nets/a.ne2".into(),
            },
            ClientMsg::NewNet,
            ClientMsg::AutoRun(true),
            ClientMsg::SetSpeed(2.5),
            ClientMsg::RunMs(100.0),
            ClientMsg::Abort,
            ClientMsg::Reset,
            ClientMsg::Watch(vec![1, 2, 3]),
            ClientMsg::Edit {
                base: 7,
                edit: NetEdit::Connect {
                    from: 1,
                    to: 2,
                    w: 6.0,
                },
            },
            ClientMsg::CopySubnet { ids: vec![4, 5] },
            ClientMsg::SetBackend(Backend::Cuda),
            ClientMsg::SetDtMs(0.05),
            ClientMsg::SetNoise {
                hz: 60.0,
                kick: 20.0,
            },
            ClientMsg::SetInputConst {
                name: "i".into(),
                value: 0.7,
            },
            ClientMsg::PlayInput {
                name: "i".into(),
                series: vec![0.0, 1.0],
            },
            ClientMsg::SetExternal {
                neuron: 9,
                amps: 2.0,
            },
            ClientMsg::Bye,
        ] {
            round_trip(&m);
        }
    }

    #[test]
    fn net_edits_round_trip() {
        for e in [
            NetEdit::AddNeuron {
                pos: [1.0, 2.0, 3.0],
                kind: NeuronKind::Inhibitory,
            },
            NetEdit::DeleteNeurons { ids: vec![0, 9] },
            NetEdit::Connect {
                from: 0,
                to: 1,
                w: 6.0,
            },
            NetEdit::Disconnect { from: 0, to: 1 },
            NetEdit::MoveNeurons {
                items: vec![(0, [0.0; 3]), (4, [1.0, 2.0, 3.0])],
            },
            NetEdit::SetNeuronName {
                id: 3,
                name: "fred".into(),
            },
            NetEdit::SetNeuronPos {
                id: 3,
                pos: [0.5; 3],
            },
            NetEdit::InsertSubnetJson {
                json: "{}".into(),
                offset: [10.0, 0.0, 0.0],
            },
            NetEdit::CreateGrid2d {
                group: "g".into(),
                w: 10,
                h: 20,
                spacing: 5.0,
                origin: [0.0; 3],
            },
            NetEdit::CreateGrid3d {
                group: "g".into(),
                w: 8,
                h: 8,
                d: 8,
                spacing: 30.0,
                origin: [1.0; 3],
            },
            NetEdit::AddInput {
                name: "i".into(),
                targets: vec![(0, 1.0), (5, 0.5)],
                pos: [0.0; 3],
            },
            NetEdit::AddSamples { name: "s".into() },
            NetEdit::AddReadout {
                name: "r".into(),
                units: vec![1, 2, 3],
            },
            NetEdit::AddToGroup {
                name: "g".into(),
                ids: vec![7, 8],
            },
        ] {
            round_trip(&e);
        }
    }

    #[test]
    fn server_msgs_round_trip() {
        let geo = Geo {
            version: 12,
            gen: 2,
            pos: vec![[0.0; 3], [5.0, 0.0, 0.0]],
            kind: vec![0, 1],
            names: vec!["a".into(), "b".into()],
            syn: vec![(0, 1)],
            syn_delay_steps: vec![3],
            disp_edges: vec![(1, 3)],
            disp_ptr: vec![0, 1, 1],
            groups: vec![("g".into(), vec![0, 1])],
            inputs: vec![Input {
                name: "i".into(),
                pos: [0.0; 3],
                targets: vec![0],
            }],
            readouts: vec![("r".into(), vec![0], vec![1.0])],
            samples: vec![("s".into(), vec![0.0], vec![0.5])],
            scripts: vec![("main".into(), "sim.run(1)".into())],
            grids: vec![Grid {
                group: 0,
                dims: [2, 1, 1],
                axes: [0, 1, 2],
                origin: [0.0; 3],
                spacing: 5.0,
                members: vec![0, 1],
            }],
            non_grid: vec![],
            dt_ms: 0.1,
        };
        for m in [
            ServerMsg::Hello {
                proto: PROTO_VERSION,
                net_version: 12,
                net_gen: 2,
            },
            ServerMsg::Geo(Box::new(geo)),
            ServerMsg::GeoDelta(Box::new(GeoDelta {
                version: 13,
                gen: 2,
                neurons_added: vec![NeuronRow {
                    id: 2,
                    pos: [9.0; 3],
                    kind: 0,
                    name: "c".into(),
                }],
                syn_added: vec![SynRow {
                    src: 0,
                    tgt: 2,
                    delay_steps: 7,
                }],
                syn_removed: vec![(0, 1)],
                disp_changed: vec![(0, vec![(2, 7)])],
                groups_changed: vec![(0, "g".into(), vec![0, 1, 2])],
                dt_ms: 0.1,
                ..Default::default()
            })),
            ServerMsg::Status {
                running: true,
                busy: false,
                sim_built: true,
                backend: Backend::Cpu,
                cfg: Config {
                    dt_ms: 0.1,
                    synapse_tau_ms: 2.0,
                    readout_tau_ms: 2.0,
                    noise_rate_hz: 60.0,
                    noise_kick_mv: 20.0,
                    seed: 1,
                    int_weights: false,
                },
                notice: Some("hi".into()),
            },
            ServerMsg::Console("= 42".into()),
            ServerMsg::Activity(Box::new(Activity {
                seq: 9,
                t_steps: 900,
                time_ms: 90.0,
                spikes_total: 12,
                rate_hz: 55.5,
                steps_per_sec: 1234.0,
                spike_bits: vec![0b101],
                non_grid_glow: vec![0, 200, 90],
                watched_v: vec![(3, vec![-70.0, -55.0])],
                grid_delta: vec![GridDelta {
                    grid: 0,
                    texels: vec![(5, 255), (9, 12)],
                }],
                input_values: vec![0.7],
            })),
            ServerMsg::GridTraces {
                base_seq: 9,
                grids: vec![vec![0u8; 16]],
            },
            ServerMsg::Clipboard("{\"neurons\":[]}".into()),
            ServerMsg::Error("stale edit".into()),
        ] {
            round_trip(&m);
        }
    }
}
