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
    lod: vec4f,  // x: synapse alpha multiplier, y: keep 1-of-N (<=1 draws all)
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

@vertex
fn vs_neuron(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> NeurOut {
    var o: NeurOut;
    let it = inst[ii];
    // misc.w = pick mode: draw the same cubes 75% bigger for easier clicks.
    let size = it.pos_scale.w * (1.0 + cam.misc.w * 0.75);
    let wp = it.pos_scale.xyz + CUBE[vi] * size;
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
    if selected(ii) {
        col = mix(col, vec3f(1.0, 0.35, 0.1), 0.75);
    }
    o.color = vec4f(col, 1.0);
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
@group(0) @binding(6) var<storage, read> lcol: array<vec4f>;

struct LineOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_syn(@builtin(vertex_index) vi: u32) -> LineOut {
    var o: LineOut;
    let si = vi >> 1u;          // synapse index
    let keep = u32(cam.lod.y + 0.5);
    if keep > 1u && (si % keep) != 0u {
        // LOD-culled: degenerate line off-screen, zero fragments
        o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
        o.color = vec4f(0.0);
        return o;
    }
    let end = vi & 1u;          // 0 = src, 1 = tgt
    let p = select(lpairs[si].x, lpairs[si].y, end == 1u);
    o.clip = cam.view_proj * vec4f(lpos[p].xyz, 1.0);
    let act = clamp(activity[p], 0.0, 1.0);
    let base = lcol[si];
    o.color = vec4f(mix(base.rgb * 0.35, base.rgb, act), base.a * cam.lod.x);
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
