// neuro_edit v2 — single WGSL module for all pipelines.

struct Camera {
    view_proj: mat4x4f,
    cam_pos: vec4f,
    // exc_color, inh_color as rgb + unused
    exc_color: vec4f,
    inh_color: vec4f,
    right: vec4f,
    up: vec4f,
    misc: vec4f, // x: neuron size, y: activity glow, z: time, w: pick mode
    lod: vec4f,  // x: synapse alpha mul, y: keep 1-of-N, z: hovered inst +1
    preview: vec4f, // xyz: live move delta for selected, w: drag active
}

@group(0) @binding(0) var<uniform> cam: Camera;

// ---------- neurons (instanced cubes) ----------

struct Inst {
    pos_scale: vec4f, // xyz = pos, w = scale
    aux: u32,         // bits 0..1 kind, bit 2 input marker
}

@group(0) @binding(1) var<storage, read> inst: array<Inst>;
@group(0) @binding(2) var<storage, read> activity: array<f32>;
@group(0) @binding(3) var<storage, read> sel_bits: array<u32>;
// Per-instance hide bitfield (same packing as sel_bits): set for cube
// instances whose group currently renders as an activity plane, and for
// synapse-endpoint lookup in vs_syn (shared buffer, binding 6 in both
// layouts — each pipeline sees only its own layout).
@group(0) @binding(6) var<storage, read> hide_bits: array<u32>;

// Unit cube, 6 faces x 2 tris, all triangles wound CCW seen from OUTSIDE
// (back-face culling is on; a face wound inward is invisible from outside).
const CUBE: array<vec3f, 36> = array<vec3f, 36>(
    // -z
    vec3f(-1,-1,-1), vec3f(-1,1,-1), vec3f(1,1,-1),
    vec3f(-1,-1,-1), vec3f(1,1,-1), vec3f(1,-1,-1),
    // +x
    vec3f(1,-1,-1), vec3f(1,1,-1), vec3f(1,1,1),
    vec3f(1,-1,-1), vec3f(1,1,1), vec3f(1,-1,1),
    // +y
    vec3f(-1,1,-1), vec3f(-1,1,1), vec3f(1,1,1),
    vec3f(-1,1,-1), vec3f(1,1,1), vec3f(1,1,-1),
    // -x
    vec3f(-1,-1,1), vec3f(-1,1,1), vec3f(-1,1,-1),
    vec3f(-1,-1,-1), vec3f(-1,-1,1), vec3f(-1,1,-1),
    // +z
    vec3f(-1,-1,1), vec3f(1,1,1), vec3f(-1,1,1),
    vec3f(-1,-1,1), vec3f(1,-1,1), vec3f(1,1,1),
    // -y
    vec3f(-1,-1,-1), vec3f(1,-1,-1), vec3f(1,-1,1),
    vec3f(-1,-1,-1), vec3f(1,-1,1), vec3f(-1,-1,1),
);

const CUBE_N: array<vec3f, 36> = array<vec3f, 36>(
    vec3f(0,0,-1), vec3f(0,0,-1), vec3f(0,0,-1),
    vec3f(0,0,-1), vec3f(0,0,-1), vec3f(0,0,-1),
    vec3f(1,0,0), vec3f(1,0,0), vec3f(1,0,0),
    vec3f(1,0,0), vec3f(1,0,0), vec3f(1,0,0),
    vec3f(0,1,0), vec3f(0,1,0), vec3f(0,1,0),
    vec3f(0,1,0), vec3f(0,1,0), vec3f(0,1,0),
    vec3f(-1,0,0), vec3f(-1,0,0), vec3f(-1,0,0),
    vec3f(-1,0,0), vec3f(-1,0,0), vec3f(-1,0,0),
    vec3f(0,0,1), vec3f(0,0,1), vec3f(0,0,1),
    vec3f(0,0,1), vec3f(0,0,1), vec3f(0,0,1),
    vec3f(0,-1,0), vec3f(0,-1,0), vec3f(0,-1,0),
    vec3f(0,-1,0), vec3f(0,-1,0), vec3f(0,-1,0),
);

struct NeurOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
    @location(1) nrm: vec3f,
    @location(2) @interpolate(flat) id: u32,
}

fn selected(i: u32) -> bool {
    let w = sel_bits[i >> 5u];
    return ((w >> (i & 31u)) & 1u) != 0u;
}

fn hidden(i: u32) -> bool {
    let w = hide_bits[i >> 5u];
    return ((w >> (i & 31u)) & 1u) != 0u;
}

@vertex
fn vs_neuron(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> NeurOut {
    var o: NeurOut;
    let it = inst[ii];
    if hidden(ii) {
        // Group renders as a plane: collapse the cube to a degenerate
        // triangle — zero fragments, still one draw call for all cubes.
        o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
        o.nrm = vec3f(0.0);
        o.id = ii;
        o.color = vec4f(0.0);
        return o;
    }
    // misc.w = pick mode: draw the same cubes 75% bigger for easier clicks.
    let size = it.pos_scale.w * (1.0 + cam.misc.w * 0.75);
    let sel = selected(ii);
    let moving = sel && cam.preview.w > 0.5;
    // Live move preview: selected cubes (and, in vs_syn, their synapse
    // endpoints) render at the drag offset — the real positions only
    // change when the drag is committed on release.
    var wpos = it.pos_scale.xyz;
    if moving {
        wpos += cam.preview.xyz;
    }
    let wp = wpos + CUBE[vi] * size;
    o.clip = cam.view_proj * vec4f(wp, 1.0);
    o.nrm = CUBE_N[vi];
    o.id = ii;

    let kind = it.aux & 1u;
    let is_input = (it.aux & 2u) != 0u;
    var base = select(cam.exc_color.rgb, cam.inh_color.rgb, kind == 1u);
    if is_input {
        base = vec3f(0.95, 0.75, 0.2);
    }
    let act = clamp(activity[ii], 0.0, 1.0);
    var col = mix(base * 0.45, base * 1.6, act);
    if sel {
        col = mix(col, vec3f(1.0, 0.35, 0.1), 0.75);
    }
    // Hover feedback: the cube under the cursor brightens a notch.
    if ii + 1u == u32(cam.lod.z) {
        col = col * 1.4 + vec3f(0.06, 0.06, 0.06);
    }
    o.color = vec4f(col, select(1.0, 0.5, moving));
    return o;
}

@fragment
fn fs_neuron(o: NeurOut) -> @location(0) vec4f {
    let l = normalize(vec3f(0.5, 0.8, 0.4));
    let d = 0.55 + 0.45 * abs(dot(normalize(o.nrm), l));
    return vec4f(o.color.rgb * d, o.color.a);
}

// Pick pass: flat id+1 into an R32Uint target.
@fragment
fn fs_pick(o: NeurOut) -> @location(0) u32 {
    return o.id + 1u;
}

// ---------- synapse / misc lines (index-pair pull) ----------

@group(0) @binding(4) var<storage, read> lpos: array<vec4f>;
@group(0) @binding(5) var<storage, read> lpairs: array<vec2u>;

struct LineOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_syn(@builtin(vertex_index) vi: u32) -> LineOut {
    var o: LineOut;
    let si = vi >> 1u;          // synapse index (local to this draw chunk)
    let keep = u32(cam.lod.y + 0.5);
    if keep > 1u && (si % keep) != 0u {
        // LOD-culled: degenerate line off-screen, zero fragments
        o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
        o.color = vec4f(0.0);
        return o;
    }
    let end = vi & 1u;          // 0 = src, 1 = tgt
    let raw = select(lpairs[si].x, lpairs[si].y, end == 1u);
    let p = raw & 0x7fffffffu;  // neuron index (top bit is the src-kind flag)
    let q = lpairs[si].y & 0x7fffffffu;
    if hidden(p) && hidden(q) {
        // Both endpoints are plane members: the plane surface shows the
        // group's internal wiring already — cull the member-member edges.
        o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
        o.color = vec4f(0.0);
        return o;
    }
    // Move preview: endpoints belonging to selected neurons ride along.
    var lpt = lpos[p].xyz;
    if selected(p) && cam.preview.w > 0.5 {
        lpt += cam.preview.xyz;
    }
    o.clip = cam.view_proj * vec4f(lpt, 1.0);
    let act = clamp(activity[p], 0.0, 1.0);
    // Per-synapse colour by source kind travels in lpairs[si].x's top bit:
    // no separate 16B/synapse colour buffer (10M synapses stay bindable).
    let inh = (lpairs[si].x & 0x80000000u) != 0u;
    let base_rgb = select(vec3f(1.0, 0.55, 0.45), vec3f(0.5, 0.55, 1.0), inh);
    // Direction cue: the line brightens toward the target end.
    let dir = select(0.5, 1.45, end == 1u);
    var rgb = mix(base_rgb * 0.35, base_rgb, act) * dir;
    var alpha = 0.4 * cam.lod.x;
    // Synapses touching a selected neuron glow golden and denser.
    let src_i = lpairs[si].x & 0x7fffffffu;
    let tgt_i = lpairs[si].y & 0x7fffffffu;
    if selected(src_i) || selected(tgt_i) {
        rgb = mix(rgb, vec3f(1.0, 0.85, 0.3), 0.65);
        alpha = min(alpha * 3.0, 1.0);
    }
    o.color = vec4f(rgb, alpha);
    return o;
}

@fragment
fn fs_line(o: LineOut) -> @location(0) vec4f {
    return o.color;
}

// ---------- simple polyline vertices (grid, boxes, connect line) ----------

struct Pv {
    @location(0) pos: vec3f,
    @location(1) color: vec4f,
}

@vertex
fn vs_pv(v: Pv) -> LineOut {
    var o: LineOut;
    o.clip = cam.view_proj * vec4f(v.pos, 1.0);
    o.color = v.color;
    return o;
}

// ---------- particles (camera-facing quads) ----------

struct Part {
    pos: vec4f,   // xyz + size
    color: vec4f,
}

@group(0) @binding(7) var<storage, read> parts: array<Part>;

const QUAD: array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-1,-1), vec2f(1,-1), vec2f(1,1),
    vec2f(-1,-1), vec2f(1,1), vec2f(-1,1),
);

@vertex
fn vs_part(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LineOut {
    var o: LineOut;
    let c = QUAD[vi];
    let p = parts[ii];
    let wp = p.pos.xyz + (cam.right.xyz * c.x + cam.up.xyz * c.y) * p.pos.w;
    o.clip = cam.view_proj * vec4f(wp, 1.0);
    o.color = p.color;
    return o;
}

// ---------- blit (scene texture -> egui viewport) ----------

struct BlitOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    o.clip = vec4f(x, y, 0.0, 1.0);
    o.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return o;
}

@group(0) @binding(8) var scene_t: texture_2d<f32>;
@group(0) @binding(9) var scene_s: sampler;

@fragment
fn fs_blit(o: BlitOut) -> @location(0) vec4f {
    return textureSample(scene_t, scene_s, o.uv);
}

// ---------- anti-aliased floor grid ----------
//
// One camera-following quad at y = cam.lod.w; the fragment shader draws
// minor (20u) and major (200u) lines analytically with fwidth-based AA,
// fading out with distance — no 1px aliasing, no moiré, no hard tile edge.

struct GridOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
}

const GRID_EXT: f32 = 3000.0;
const GRID_QUAD: array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32) -> GridOut {
    var o: GridOut;
    let c = GRID_QUAD[vi] * GRID_EXT;
    // The quad follows the camera in x/z (lines come from world coords,
    // so the pattern doesn't swim).
    let wp = vec3f(cam.cam_pos.x + c.x, cam.lod.w, cam.cam_pos.z + c.y);
    o.clip = cam.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    return o;
}

fn grid_line(coord: vec2f) -> f32 {
    // AA line intensity in [0,1] — fwidth widens the line to a screen pixel
    // so far cells dissolve into a smooth glow instead of moiré.
    let g = abs(fract(coord - vec2f(0.5)) - vec2f(0.5)) / fwidth(coord);
    return 1.0 - min(min(g.x, g.y), 1.0);
}

@fragment
fn fs_grid(o: GridOut) -> @location(0) vec4f {
    let minor = grid_line(o.wp.xz / 20.0);
    let major = grid_line(o.wp.xz / 200.0);
    let dist = distance(o.wp.xz, cam.cam_pos.xz);
    let fade = 1.0 - smoothstep(GRID_EXT * 0.55, GRID_EXT * 0.95, dist);
    let col = mix(vec3f(0.24, 0.27, 0.32), vec3f(0.42, 0.46, 0.55), major);
    let alpha = (minor * 0.45 + major * 0.4) * fade;
    if alpha < 0.004 {
        discard;
    }
    return vec4f(col, alpha);
}

// ---------- grid activity planes (one quad per grid group) ----------
//
// Each plane is a textured quad: the app reduces the grid's f32 spike
// trace to a 2D image (slice or max/mean projection for 3D grids) and
// uploads it to an R32Float texture; a parallel u32 buffer maps each
// displayed texel back to its member neuron id for picking.

struct PlaneParams {
    origin: vec4f,  // world pos of the quad corner (texel -0.5,-0.5 cell edge)
    du: vec4f,      // world step per texel along u
    dv: vec4f,      // world step per texel along v
    wh: vec4u,      // x = w texels, y = h texels
}

@group(0) @binding(10) var<storage, read> pparams: array<PlaneParams>;
@group(0) @binding(11) var ptex: texture_2d<f32>;
@group(0) @binding(12) var<storage, read> ppick: array<u32>;

struct PlaneOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f, // texel coordinates 0..w, 0..h
    @location(1) @interpolate(flat) id: u32,
}

const QUAD6: array<vec2f, 6> = array<vec2f, 6>(
    vec2f(0, 0), vec2f(1, 0), vec2f(1, 1),
    vec2f(0, 0), vec2f(1, 1), vec2f(0, 1),
);

@vertex
fn vs_plane(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PlaneOut {
    var o: PlaneOut;
    let p = pparams[ii];
    let c = QUAD6[vi];
    let wp = p.origin.xyz + (c.x * f32(p.wh.x)) * p.du.xyz + (c.y * f32(p.wh.y)) * p.dv.xyz;
    o.clip = cam.view_proj * vec4f(wp, 1.0);
    o.uv = c * vec2f(f32(p.wh.x), f32(p.wh.y));
    o.id = ii;
    return o;
}

// Perceptually-uniform colormap (viridis stops) on sqrt(v) — the trace is
// a decaying 0..1 value; the sqrt lift keeps low activity visible.
fn cmap(v: f32) -> vec3f {
    let t = clamp(v, 0.0, 1.0);
    let c0 = vec3f(0.267, 0.005, 0.329);
    let c1 = vec3f(0.284, 0.141, 0.458);
    let c2 = vec3f(0.254, 0.265, 0.530);
    let c3 = vec3f(0.207, 0.372, 0.553);
    let c4 = vec3f(0.164, 0.471, 0.558);
    let c5 = vec3f(0.128, 0.567, 0.551);
    let c6 = vec3f(0.135, 0.659, 0.518);
    let c7 = vec3f(0.267, 0.749, 0.441);
    let c8 = vec3f(0.478, 0.821, 0.318);
    let c9 = vec3f(0.741, 0.873, 0.150);
    let c10 = vec3f(0.993, 0.906, 0.144);
    let x = t * 10.0;
    let i = u32(min(floor(x), 9.0));
    let f = x - f32(i);
    var stops: array<vec3f, 11>;
    stops[0] = c0; stops[1] = c1; stops[2] = c2; stops[3] = c3;
    stops[4] = c4; stops[5] = c5; stops[6] = c6; stops[7] = c7;
    stops[8] = c8; stops[9] = c9; stops[10] = c10;
    return mix(stops[i], stops[i + 1u], f);
}

@fragment
fn fs_plane(o: PlaneOut) -> @location(0) vec4f {
    let p = pparams[o.id];
    let t = vec2i(clamp(o.uv, vec2f(0.0), vec2f(f32(p.wh.x) - 0.5, f32(p.wh.y) - 0.5)));
    let v = textureLoad(ptex, t, 0).r;
    let col = cmap(sqrt(clamp(v, 0.0, 1.0)));
    // Dim the quiet end instead of going fully black so the lattice still
    // reads as a surface; active texels pop.
    return vec4f(col * (0.35 + 0.65 * clamp(v, 0.0, 1.0)), 0.92);
}

@fragment
fn fs_plane_pick(o: PlaneOut) -> @location(0) u32 {
    let p = pparams[o.id];
    let t = vec2i(clamp(o.uv, vec2f(0.0), vec2f(f32(p.wh.x) - 0.5, f32(p.wh.y) - 0.5)));
    let tid = u32(t.y) * p.wh.x + u32(t.x);
    return ppick[tid] + 1u;
}
