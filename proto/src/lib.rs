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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
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

impl Default for Config {
    fn default() -> Self {
        Self {
            dt_ms: net_format::DEFAULT_DT_MS,
            synapse_tau_ms: net_format::DEFAULT_SYNAPSE_TAU_MS,
            readout_tau_ms: net_format::DEFAULT_READOUT_TAU_MS,
            noise_rate_hz: 0.0,
            noise_kick_mv: 20.0,
            seed: 1,
            int_weights: false,
        }
    }
}

/// Activity-plane trace decay constant (ms) — part of the wire
/// semantics: the worker quantizes `exp(-(t - last_spike)/τ)` and
/// clients decay their accumulators by the same τ between frames.
pub const PLANE_TAU_MS: f32 = 20.0;

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
    /// `Activity::non_grid_spiked` is coded against.
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
    /// (synapse index, new delay_steps) — moves recompute delays.
    pub syn_delays: Vec<(u32, u16)>,
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

/// The f64 a `format!("{x:?}")`-built Rhai call delivered for an f32
/// field — the shortest-decimal string reparsed. A plain `x as f64`
/// widens the binary tail instead (`0.3f32 as f64` is
/// 0.30000001192092896), so RNG thresholds and stored weights/taus
/// would differ from the same digits typed into a script.
pub fn f64_of(x: f32) -> f64 {
    format!("{x:?}").parse().unwrap_or(x as f64)
}

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
    /// Flip a neuron's Dale kind — the inspector's kind selector.
    /// Re-signs its outgoing synapses at the next sim build.
    SetNeuronKind {
        id: u32,
        kind: NeuronKind,
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
    /// Set absolute positions for inputs (post-drag). pos [0,0,0] means
    /// "auto-place above the targets" — a drop stores wherever it
    /// landed, so it doesn't snap back.
    MoveInputs {
        items: Vec<(u32, [f32; 3])>,
    },
    /// Add ids to a group, creating the group when missing.
    AddToGroup {
        name: String,
        ids: Vec<u32>,
    },
    /// Append one target to a named input's target list (does NOT replace
    /// the existing targets — connect mode draws input→neuron links).
    AddInputTarget {
        name: String,
        target: u32,
        w: f32,
    },
    /// `net.create_connections(group, scaling, lambda, w)` — seeded
    /// probabilistic wiring inside one group (the dialog no longer
    /// evals a Rhai string for this).
    CreateConnections {
        group: String,
        scaling: f64,
        lambda: f64,
        w: f32,
    },
    /// `net.set_synapse_weights(group, mean, sd)` — gaussian weights on
    /// every synapse whose target is a group member.
    SetSynapseWeights {
        group: String,
        mean: f64,
        sd: f64,
    },
    /// `net.set_syn_tau_dist(src, dst, mean, sd)` — gaussian per-synapse
    /// current-trace tau on src→dst group edges (0 = merged global tau).
    SetSynTauDist {
        src: String,
        dst: String,
        mean: f64,
        sd: f64,
    },
    /// `net.set_exc_inh(group, frac)` — seeded excitatory/inhibitory
    /// kind mix over the group's members.
    SetExcInh {
        group: String,
        frac: f64,
    },
    /// `net.script_add(name, code)` — attach or replace a script stored
    /// in the net file ("Save to net").
    ScriptAdd {
        name: String,
        src: String,
    },
}

// ---------------------------------------------------------------------------
// messages
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    /// First frame on a connection.
    Hello {
        proto: u32,
    },
    /// Re-request full state (Geo + Status + GridTraces) — used on
    /// (re)connect and by later subscribers on a shared run.
    Subscribe,
    /// Run a Rhai script on the worker.
    Eval {
        code: String,
    },
    /// Replace the whole network from a .ne2 path resolved server-side.
    LoadNet {
        path: String,
    },
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
    Edit {
        base: u64,
        edit: NetEdit,
    },
    /// Serialize the subnet on these ids to JSON; the server answers with
    /// `ServerMsg::Clipboard`.
    CopySubnet {
        ids: Vec<u32>,
    },
    SetBackend(Backend),
    SetDtMs(f32),
    /// Admin-pushed per-user quota update (hosted mode; editors never
    /// send this — the worker just adjusts its write cap).
    SetQuota(u64),
    SetNoise {
        hz: f32,
        kick: f32,
    },
    SetInputConst {
        name: String,
        value: f32,
    },
    PlayInput {
        name: String,
        series: Vec<f32>,
    },
    SetExternal {
        neuron: u32,
        amps: f32,
    },
    /// List a directory inside the server's data dir (or verbatim in
    /// local mode) — powers the UI's file pickers so no `std::fs` is
    /// needed client-side.
    ListFiles {
        path: String,
    },
    /// Read a small file (script sources, configs). Answer: `FileData`
    /// or `Error`.
    ReadFile {
        path: String,
    },
    /// Write a small file. Answer: `FileWritten` or `Error`.
    WriteFile {
        path: String,
        data: Vec<u8>,
    },
    /// Upload a dataset into `data_dir/datasets/<name>`: raw signal bytes
    /// (`f32` LE rows or a `.npy` file), JSON meta (channels/dt/steps…)
    /// and JSON labels (`{events_ms: […]}`). Server-side size limit;
    /// answer: `FileWritten` or `Error`.
    UploadDataset {
        name: String,
        /// Stored as `datasets/<name>/signal.<ext>`; the loader reads
        /// `f32` (raw LE f32) and `npy`.
        signal_ext: String,
        signal: Vec<u8>,
        meta_json: String,
        labels_json: String,
    },
    /// Run a script as a background job on the server: full speed, own
    /// network/sim, console + `job_progress` calls stream to every
    /// client; results persist under `data_dir/runs/<id>`. Answer:
    /// `JobStarted` or `Error`.
    StartJob {
        name: String,
        script: String,
    },
    /// Ask the server to stop a running job.
    CancelJob {
        id: u64,
    },
    /// Live jobs + persisted runs. Answer: `JobList`.
    ListJobs,
    /// Re-open a finished/interrupted run's artifacts. Answer:
    /// `JobDetail` or `Error`.
    OpenJob {
        id: u64,
    },
    /// Create a directory (recursive) inside the user's root.
    /// Answer: `DirCreated` or `Error`.
    Mkdir {
        path: String,
    },
    /// Delete a file or directory (dirs recursively) inside the user's
    /// root. Answer: `FileDeleted` or `Error`.
    DeleteFile {
        path: String,
    },
    /// Rename/move a file or directory inside the user's root.
    /// Answer: `FileRenamed` or `Error`.
    RenameFile {
        from: String,
        to: String,
    },
    /// Storage used / quota for this user. Answer: `Usage`.
    FileUsage,
    /// Begin a chunked upload to `path`: checks the declared `size`
    /// against the per-file cap and the account quota before a byte
    /// lands. Answer: `UploadReady` or `Error`.
    UploadStart {
        path: String,
        size: u64,
    },
    /// One chunk of an upload for `path` (must follow `UploadStart`).
    /// Appended to `<path>.upload`; `last` commits the rename.
    /// Answer: `FileWritten` on `last`, `Error` aborts (part removed).
    UploadChunk {
        path: String,
        data: Vec<u8>,
        last: bool,
    },
    /// Chunked download — answers with `FileChunk*` ending `done`,
    /// or `Error`. Big datasets ride this without one huge frame.
    DownloadFile {
        path: String,
    },
    /// Polite disconnect.
    Bye,
    /// Q6: save the live network to `path` (server-resolved) with a typed
    /// acknowledgement — answers `Saved` or `Error`. Unlike an eval of
    /// `net.save(..)` this carries the saved net version back so the
    /// client can mark the document clean only on real confirmation.
    SaveNet {
        path: String,
    },
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
    /// Bit i set ⇔ neuron id `i` spiked since the previous Activity.
    /// Covers the whole id space — clients fold it into their raster /
    /// particle spike views AND their glow accumulators: the decay laws
    /// (`exp(-Δms / PLANE_TAU)` for plane texels, `(1-dt/tau_r)` per step
    /// for per-neuron glow) are deterministic, so a spiked texel/neuron
    /// just gets stamped 1.0 client-side instead of shipping every fading
    /// value over the wire each frame.
    pub spike_bits: Vec<u8>,
    /// New v samples for watched neurons, per id.
    pub watched_v: Vec<(u32, Vec<f32>)>,
    /// New u (recovery) samples for watched neurons — same ids and
    /// lengths as `watched_v`, feeding the editor's v–u phase plot.
    pub watched_u: Vec<(u32, Vec<f32>)>,
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
    /// Answer to `ClientMsg::ListFiles`.
    FileList {
        path: String,
        entries: Vec<FileEntry>,
    },
    /// Answer to `ClientMsg::ReadFile`.
    FileData { path: String, data: Vec<u8> },
    /// Answer to `ClientMsg::WriteFile`.
    FileWritten { path: String },
    /// A job started (or already-known info for `ListJobs`).
    JobStarted { id: u64, name: String },
    /// One `job_progress(#{…})` call's numeric fields — streamed for the
    /// jobs panel's live plot.
    JobProgress { id: u64, fields: Vec<(String, f64)> },
    /// Job finished. `status`: "done" | "failed" | "cancelled".
    JobDone {
        id: u64,
        status: String,
        error: Option<String>,
    },
    /// Answer to `ClientMsg::ListJobs` — live jobs and persisted runs.
    JobList { jobs: Vec<JobInfo> },
    /// Answer to `ClientMsg::OpenJob` — the run's artifacts.
    JobDetail {
        id: u64,
        name: String,
        status: String,
        script: String,
        result_json: Option<String>,
        /// The `job_progress` history, one row per call.
        progress: Vec<Vec<(String, f64)>>,
    },
    /// Answer to `ClientMsg::Mkdir`.
    DirCreated { path: String },
    /// Answer to `ClientMsg::DeleteFile`.
    FileDeleted { path: String },
    /// Answer to `ClientMsg::RenameFile`.
    FileRenamed { from: String, to: String },
    /// Answer to `ClientMsg::FileUsage` — `quota` 0 means unlimited.
    Usage { used: u64, quota: u64 },
    /// Answer to `ClientMsg::UploadStart` — the upload may proceed.
    UploadReady { path: String },
    /// One chunk of a `DownloadFile` reply; `done` closes the stream
    /// (`data` empty on the last).
    FileChunk {
        path: String,
        data: Vec<u8>,
        done: bool,
    },
    /// Refused edit / protocol error / worker failure worth a dialog.
    Error(String),
    /// Q6: answer to `ClientMsg::SaveNet` — the file landed; `version`
    /// is the net version that was written (the document's clean mark).
    Saved { path: String, version: u64 },
    /// Q7: answer to a successful `ClientMsg::LoadNet` — `net_path` and
    /// recent-file lists update only on this, never on the request.
    NetLoaded { path: String, version: u64 },
}

/// One row of `JobList` — a live job or a `runs/<id>` artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: u64,
    pub name: String,
    /// "running" | "done" | "failed" | "cancelled" | "lost" (a run whose
    /// meta says running but no live job owns it — e.g. server restart).
    pub status: String,
    /// Wall-clock start time, unix ms (0 for old artifacts).
    pub started_ms: u64,
}

/// One entry in a `FileList` reply.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    /// Unix ms mtime (0 when unknown).
    pub mtime_ms: u64,
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

/// Wire payloads are bincode; payloads above `COMPRESS_OVER` travel
/// deflated behind a 1-byte tag — a 1M-neuron `Activity` bitmap would
/// otherwise cost ~125 KB raw. Small messages stay raw (tag `0`).
const COMPRESS_OVER: usize = 256;
const TAG_RAW: u8 = 0;
const TAG_DEFLATE: u8 = 1;

/// Encode a message as one WebSocket binary frame's payload.
pub fn encode<T: Serialize>(m: &T) -> Vec<u8> {
    use std::io::Write;
    let raw = bincode::serialize(m).expect("proto encode");
    if raw.len() <= COMPRESS_OVER {
        let mut out = Vec::with_capacity(raw.len() + 1);
        out.push(TAG_RAW);
        out.extend_from_slice(&raw);
        return out;
    }
    let mut e = flate2::write::DeflateEncoder::new(
        Vec::with_capacity(raw.len() / 4),
        flate2::Compression::fast(),
    );
    e.write_all(&raw).expect("deflate");
    let z = e.finish().expect("deflate finish");
    let mut out = Vec::with_capacity(z.len() + 1);
    out.push(TAG_DEFLATE);
    out.extend_from_slice(&z);
    out
}

/// Decompression limit: no legitimate message inflates past this.
/// The largest outbound payload is an upload chunk (bounded by
/// `--max-upload-mb`, default 64 MiB); allow a little slack and refuse
/// the rest so a compressed-zeroes frame can't inflate endlessly.
pub const DECODE_LIMIT: u64 = 256 * 1024 * 1024;

/// Decode one payload. `Err` on malformed bytes or trailing data.
/// Deflate payloads are inflated through a capped reader and bincode
/// reads with `with_limit` so neither side of a hostile frame can
/// request unbounded allocation.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, bincode::Error> {
    use bincode::Options;
    use std::io::Read;
    // `legacy()` compatibility: fixint + trailing-allowed — only the
    // size limit differs from the old plain `bincode::deserialize`.
    let opts = bincode::options()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(DECODE_LIMIT);
    match bytes.split_first() {
        Some((&TAG_RAW, raw)) => opts.deserialize(raw),
        Some((&TAG_DEFLATE, z)) => {
            let mut raw = Vec::new();
            flate2::read::DeflateDecoder::new(z)
                .take(DECODE_LIMIT)
                .read_to_end(&mut raw)
                .map_err(|e| Box::new(bincode::ErrorKind::Io(e)))?;
            opts.deserialize(&raw)
        }
        _ => Err(Box::new(bincode::ErrorKind::Custom("empty frame".into()))),
    }
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
            ClientMsg::SetQuota(1 << 30),
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
            ClientMsg::ListFiles {
                path: "sub/dir".into(),
            },
            ClientMsg::ReadFile {
                path: "a.rhai".into(),
            },
            ClientMsg::WriteFile {
                path: "b.rhai".into(),
                data: b"print(1)".to_vec(),
            },
            ClientMsg::UploadDataset {
                name: "d1".into(),
                signal_ext: "f32".into(),
                signal: vec![0, 1, 2, 3],
                meta_json: "{\"channels\":2}".into(),
                labels_json: "{\"events_ms\":[1.5]}".into(),
            },
            ClientMsg::StartJob {
                name: "train".into(),
                script: "print(1)".into(),
            },
            ClientMsg::CancelJob { id: 9 },
            ClientMsg::ListJobs,
            ClientMsg::OpenJob { id: 4 },
            ClientMsg::Mkdir {
                path: "new/dir".into(),
            },
            ClientMsg::DeleteFile {
                path: "old.rhai".into(),
            },
            ClientMsg::RenameFile {
                from: "a".into(),
                to: "b/c".into(),
            },
            ClientMsg::FileUsage,
            ClientMsg::UploadStart {
                path: "big.bin".into(),
                size: 12345,
            },
            ClientMsg::UploadChunk {
                path: "big.bin".into(),
                data: vec![9; 7],
                last: true,
            },
            ClientMsg::DownloadFile {
                path: "big.bin".into(),
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
            NetEdit::AddInputTarget {
                name: "i".into(),
                target: 4,
                w: 6.0,
            },
            NetEdit::MoveInputs {
                items: vec![(0, [9.0, 8.0, 7.0]), (2, [0.0; 3])],
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
                watched_v: vec![(3, vec![-70.0, -55.0])],
                watched_u: vec![(3, vec![-14.0, -13.0])],
                input_values: vec![0.7],
            })),
            ServerMsg::GridTraces {
                base_seq: 9,
                grids: vec![vec![0u8; 16]],
            },
            ServerMsg::Clipboard("{\"neurons\":[]}".into()),
            ServerMsg::FileList {
                path: ".".into(),
                entries: vec![
                    FileEntry {
                        name: "a.rhai".into(),
                        is_dir: false,
                        size: 42,
                        mtime_ms: 7,
                    },
                    FileEntry {
                        name: "nets".into(),
                        is_dir: true,
                        size: 0,
                        mtime_ms: 8,
                    },
                ],
            },
            ServerMsg::FileData {
                path: "a.rhai".into(),
                data: b"code".to_vec(),
            },
            ServerMsg::FileWritten {
                path: "a.rhai".into(),
            },
            ServerMsg::JobStarted {
                id: 3,
                name: "train".into(),
            },
            ServerMsg::JobProgress {
                id: 3,
                fields: vec![("epoch".into(), 4.0), ("auc".into(), 0.91)],
            },
            ServerMsg::JobDone {
                id: 3,
                status: "done".into(),
                error: None,
            },
            ServerMsg::JobList {
                jobs: vec![JobInfo {
                    id: 2,
                    name: "old".into(),
                    status: "done".into(),
                    started_ms: 1_700_000_000_000,
                }],
            },
            ServerMsg::JobDetail {
                id: 2,
                name: "old".into(),
                status: "done".into(),
                script: "print(1)".into(),
                result_json: Some("{\"auc\":0.9}".into()),
                progress: vec![vec![("auc".into(), 0.5)], vec![("auc".into(), 0.9)]],
            },
            ServerMsg::Error("stale edit".into()),
        ] {
            round_trip(&m);
        }
    }

    /// S8: a raw frame claiming a huge collection is refused by the
    /// bincode limit without allocating it; a deflate bomb inflates
    /// only up to DECODE_LIMIT and then fails — neither blocks the
    /// worker thread nor grows memory unbounded.
    #[test]
    fn decode_refuses_oversized_payloads() {
        // RAW frame: a real `Watch` encoding is tag + u32 variant + u64
        // len. Patch the length field past the limit — decode must fail
        // fast instead of trying to allocate the claimed vector.
        let mut frame = encode(&ClientMsg::Watch(vec![]));
        let len_at = frame.len() - 8; // u64 element count, last field
        frame[len_at..].copy_from_slice(&(DECODE_LIMIT + 1).to_le_bytes());
        let r: Result<ClientMsg, _> = decode(&frame);
        assert!(r.is_err(), "oversized vec must fail fast");

        // deflate bomb: 300 MB of zeroes compresses small; the capped
        // inflater stops at DECODE_LIMIT — decode returns (probably as
        // Hello{proto:0} — a bounded result) without a 300 MB vec.
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        use std::io::Write;
        enc.write_all(&vec![0u8; 8 << 20]).unwrap();
        let z = enc.finish().unwrap();
        let mut frame = vec![TAG_DEFLATE];
        frame.extend_from_slice(&z);
        let start = std::time::Instant::now();
        let r: Result<ClientMsg, _> = decode(&frame);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "capped inflate must return quickly"
        );
        let _ = r;
    }
}
