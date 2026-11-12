//! wgpu renderer: instanced neuron cubes, index-pulled synapse lines,
//! billboard spike particles, plain polyline overlay, and a GPU pick pass
//! (R32Uint id buffer) — all from one WGSL module.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::snap::GeoSnap;

/// Cap on one synapse-pairs bind range — keeps every binding comfortably
/// under the 128MB `max_storage_buffer_binding_size` minimum, whatever the
/// adapter reports above it.
const SYN_BIND_MAX_BYTES: usize = 64 << 20;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CamU {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    exc_color: [f32; 4],
    inh_color: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    misc: [f32; 4],
    /// x: synapse alpha multiplier, y: draw 1-of-N synapses (<=1 draws all).
    lod: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Inst {
    pos_scale: [f32; 4],
    aux: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Part {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Pv {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub _pad: f32,
}

/// GPU state for one network geometry (rebuilt when geo version changes).
/// Buffers are held to keep the allocations alive for the bind groups.
#[allow(dead_code)]
pub struct Scene {
    pub n: usize,
    pub entity_of_inst: Vec<Entity>,
    inst_buf: wgpu::Buffer,
    activity_buf: wgpu::Buffer,
    sel_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    pairs_buf: wgpu::Buffer,
    /// Synapse vertices drawn this scene (2 per synapse, after truncation).
    pub syn_verts: u32,
    /// Mean nearest-neighbour scale: cbrt(bounds volume / neuron count).
    avg_spacing: f32,
    /// Mean synapse length in world units.
    avg_edge_len: f32,
    /// Bounds silhouette proxy: product of the two largest extents.
    silhouette: f32,
    pub bounds: (Vec3, Vec3),
}

/// What an instance index refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entity {
    Neuron(u32),
    Input(u32),
    Readout(u32),
}

#[allow(dead_code)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    cam_buf: wgpu::Buffer,
    bgl_neur: wgpu::BindGroupLayout,
    bgl_syn: wgpu::BindGroupLayout,
    bgl_pv: wgpu::BindGroupLayout,
    bgl_part: wgpu::BindGroupLayout,
    pip_neur: wgpu::RenderPipeline,
    pip_pick: wgpu::RenderPipeline,
    pip_syn: wgpu::RenderPipeline,
    pip_pv: wgpu::RenderPipeline,
    pip_part: wgpu::RenderPipeline,
    pip_blit: wgpu::RenderPipeline,
    bgl_blit: wgpu::BindGroupLayout,
    blit_bg: Option<wgpu::BindGroup>,
    blit_sampler: wgpu::Sampler,
    scene_tex: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    depth: Option<(wgpu::TextureView, u32, u32)>,
    pick: Option<PickTarget>,
    pub scene: Option<Scene>,
    neuron_bg: Option<wgpu::BindGroup>,
    /// One bind group per synapse-pairs chunk: each binding stays under
    /// `max_storage_buffer_binding_size`, so huge nets still draw.
    syn_bgs: Vec<(wgpu::BindGroup, u32)>,
    part_bg: Option<wgpu::BindGroup>,
    part_buf: wgpu::Buffer,
    part_cap: usize,
    pv_buf: wgpu::Buffer,
    pv_cap: usize,
    // pick readback
    pick_pending: Option<PendingPick>,
    pick_buf: wgpu::Buffer,
    /// Last camera uniform written by `write_cam`; pick passes rewrite it
    /// with `misc.w = 1` (enlarged pick footprints).
    cam_u: CamU,
    pv_bg: wgpu::BindGroup,
    /// Draw counts for this frame, set by the UI before rendering.
    pub draw_params: Option<DrawParams>,
}

/// What to draw this frame (set per frame by the app).
pub struct DrawParams {
    pub pv_count: u32,
    pub part_count: u32,
    pub show_syn: bool,
}

struct PickTarget {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
    w: u32,
    h: u32,
}

struct PendingPick {
    result: Arc<std::sync::Mutex<Option<u32>>>,
    hover: bool,
    // kept to hold the staging buffer alive until the map completes
    _buf: Arc<wgpu::Buffer>,
}

impl Gpu {
    /// `surface_format` must match the render target `blit` writes into —
    /// the egui surface in GUI mode (Bgra8Unorm on many drivers), the
    /// internal Rgba8UnormSrgb texture headless.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Gpu {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cam"),
            size: std::mem::size_of::<CamU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_neur = bgl(
            &device,
            "neur",
            &[
                (
                    0,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                ),
                (1, storage(), wgpu::ShaderStages::VERTEX),
                (2, storage(), wgpu::ShaderStages::VERTEX_FRAGMENT),
                (3, storage(), wgpu::ShaderStages::VERTEX),
            ],
        );
        let bgl_syn = bgl(
            &device,
            "syn",
            &[
                (
                    0,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::VERTEX,
                ),
                (2, storage(), wgpu::ShaderStages::VERTEX),
                (4, storage(), wgpu::ShaderStages::VERTEX),
                (5, storage(), wgpu::ShaderStages::VERTEX),
            ],
        );
        let bgl_pv = bgl(
            &device,
            "pv",
            &[(
                0,
                wgpu::BufferBindingType::Uniform,
                wgpu::ShaderStages::VERTEX,
            )],
        );
        let bgl_part = bgl(
            &device,
            "part",
            &[
                (
                    0,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::VERTEX,
                ),
                (7, storage(), wgpu::ShaderStages::VERTEX),
            ],
        );

        let no_cull = wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };
        let _ = no_cull;

        let mk_pl = |bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };

        let col = |blend: bool| wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let depth_st = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        };

        let mk = |label, vs, fs, targets: &[Option<wgpu::ColorTargetState>], pl, topo, cull| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: topo,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(depth_st.clone()),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };

        let pl_neur = mk_pl(&bgl_neur);
        let pl_syn = mk_pl(&bgl_syn);
        let pl_pv = mk_pl(&bgl_pv);
        let pl_part = mk_pl(&bgl_part);

        let pip_neur = mk(
            "neur",
            "vs_neuron",
            "fs_neuron",
            &[Some(col(true))],
            &pl_neur,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
        );
        let pip_pick = mk(
            "pick",
            "vs_neuron",
            "fs_pick",
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            &pl_neur,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
        );
        let pip_syn = mk(
            "syn",
            "vs_syn",
            "fs_line",
            &[Some(col(true))],
            &pl_syn,
            wgpu::PrimitiveTopology::LineList,
            None,
        );
        let pip_part = mk(
            "part",
            "vs_part",
            "fs_line",
            &[Some(col(true))],
            &pl_part,
            wgpu::PrimitiveTopology::TriangleList,
            None,
        );
        // pv pipeline needs a vertex buffer layout — separate descriptor.
        let pv_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Pv>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        };
        let pip_pv = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pv"),
            layout: Some(&pl_pv),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_pv"),
                compilation_options: Default::default(),
                buffers: &[pv_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                compilation_options: Default::default(),
                targets: &[Some(col(true))],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_st),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let pick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick rb"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pv bg"),
            layout: &bgl_pv,
            entries: &[entry(0, cam_buf.as_entire_binding())],
        });

        // blit pipeline (texture + sampler, no cam)
        let bgl_blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl_blit = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl blit"),
            bind_group_layouts: &[&bgl_blit],
            push_constant_ranges: &[],
        });
        let pip_blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pl_blit),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_blit"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let part_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("part"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pv"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Gpu {
            device,
            queue,
            cam_buf,
            bgl_neur,
            bgl_syn,
            bgl_pv,
            bgl_part,
            pip_neur,
            pip_pick,
            pip_syn,
            pip_pv,
            pip_part,
            pip_blit,
            bgl_blit,
            blit_bg: None,
            blit_sampler,
            scene_tex: None,
            depth: None,
            pick: None,
            scene: None,
            neuron_bg: None,
            syn_bgs: Vec::new(),
            part_bg: None,
            part_buf,
            part_cap: 0,
            pv_buf,
            pv_cap: 0,
            pick_pending: None,
            pick_buf,
            cam_u: CamU::zeroed(),
            pv_bg,
            draw_params: None,
        }
    }

    /// Rebuild all static geometry buffers from a new GeoSnap.
    pub fn set_scene(&mut self, geo: &GeoSnap) {
        let n = geo.pos.len();
        // instances: neurons, then inputs, then readouts
        let mut inst = Vec::with_capacity(n + geo.inputs.len() + geo.readouts.len());
        let mut entity_of_inst = Vec::with_capacity(inst.capacity());
        for (i, p) in geo.pos.iter().enumerate() {
            inst.push(Inst {
                pos_scale: [p[0], p[1], p[2], 4.0],
                aux: geo.kind.get(i).copied().unwrap_or(0) as u32,
                _pad: [0; 3],
            });
            entity_of_inst.push(Entity::Neuron(i as u32));
        }
        for (k, inp) in geo.inputs.iter().enumerate() {
            // inputs without a position: centroid of their targets
            let mut p = inp.pos;
            if p == [0.0; 3] && !inp.targets.is_empty() {
                let mut c = [0f32; 3];
                let mut m = 0f32;
                for &t in &inp.targets {
                    if let Some(&tp) = geo.pos.get(t as usize) {
                        c[0] += tp[0];
                        c[1] += tp[1];
                        c[2] += tp[2];
                        m += 1.0;
                    }
                }
                if m > 0.0 {
                    p = [c[0] / m, c[1] / m + 60.0, c[2] / m];
                }
            }
            inst.push(Inst {
                pos_scale: [p[0], p[1], p[2], 7.0],
                aux: 2,
                _pad: [0; 3],
            });
            entity_of_inst.push(Entity::Input(k as u32));
        }
        for (k, (name, _, _)) in geo.readouts.iter().enumerate() {
            let _ = name;
            // readout marker at centroid + offset
            let units = &geo.readouts[k].1;
            let mut c = [0f32; 3];
            if !units.is_empty() {
                let mut m = 0f32;
                for &u in units {
                    if let Some(&tp) = geo.pos.get(u as usize) {
                        c[0] += tp[0];
                        c[1] += tp[1];
                        c[2] += tp[2];
                        m += 1.0;
                    }
                }
                if m > 0.0 {
                    c = [c[0] / m, c[1] / m + 120.0, c[2] / m];
                }
            }
            inst.push(Inst {
                pos_scale: [c[0], c[1], c[2], 7.0],
                aux: 3,
                _pad: [0; 3],
            });
            entity_of_inst.push(Entity::Readout(k as u32));
        }
        let inst_total = inst.len();
        // An empty network (the editor starts on one) would make a zero-sized
        // storage binding, which wgpu rejects. Draws use `entity_of_inst.len()`,
        // so the placeholder is never drawn.
        if inst.is_empty() {
            inst.push(Inst {
                pos_scale: [0.0; 4],
                aux: 0,
                _pad: [0; 3],
            });
        }

        let inst_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inst"),
                contents: bytemuck::cast_slice(&inst),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let activity_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act"),
            size: (inst_total.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sel_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sel"),
            size: ((inst_total.max(1) / 32 + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut pos4 = Vec::with_capacity(n);
        for p in &geo.pos {
            pos4.push([p[0], p[1], p[2], 0.0]);
        }
        if pos4.is_empty() {
            pos4.push([0.0; 4]);
        }
        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pos"),
                contents: bytemuck::cast_slice(&pos4),
                usage: wgpu::BufferUsages::STORAGE,
            });
        // Each synapse is one u32 pair; the source neuron's kind flag rides
        // in the top bit of `a` (src < 2^31 always). That keeps per-synapse
        // GPU data at 8B — a 10M-synapse net is an 80MB buffer.
        // `max_buffer_size` still caps the absolute size; past it we draw a
        // truncated prefix rather than trip device validation.
        let limits = self.device.limits();
        let cap_syns = (limits.max_buffer_size / 8) as usize;
        let n_syns = geo.syn.len();
        if n_syns > cap_syns {
            eprintln!(
                "scene: {} synapses exceed max_buffer_size — drawing first {} only",
                n_syns, cap_syns
            );
        }
        let draw_syns = n_syns.min(cap_syns).max(1); // [0,0] placeholder if empty
        let mut pairs = Vec::with_capacity(draw_syns);
        for &(a, b) in geo.syn.iter().take(draw_syns) {
            let flag = u32::from(geo.kind.get(a as usize) == Some(&1)) << 31;
            pairs.push([a | flag, b]);
        }
        if pairs.is_empty() {
            pairs.push([0, 0]);
        }
        let pairs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pairs"),
                contents: bytemuck::cast_slice(&pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.neuron_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neur bg"),
            layout: &self.bgl_neur,
            entries: &[
                entry(0, self.cam_buf.as_entire_binding()),
                entry(1, inst_buf.as_entire_binding()),
                entry(2, activity_buf.as_entire_binding()),
                entry(3, sel_buf.as_entire_binding()),
            ],
        }));
        // Chunk the pairs binding under max_storage_buffer_binding_size
        // (128MB on the least capable adapters). Each chunk becomes one
        // bind group + one draw call.
        let bind_cap =
            (limits.max_storage_buffer_binding_size as usize).min(SYN_BIND_MAX_BYTES) / 8;
        let chunk_syns = bind_cap.max(1);
        self.syn_bgs = (0..pairs.len())
            .step_by(chunk_syns)
            .map(|off| {
                let len = (pairs.len() - off).min(chunk_syns);
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("syn bg"),
                    layout: &self.bgl_syn,
                    entries: &[
                        entry(0, self.cam_buf.as_entire_binding()),
                        entry(2, activity_buf.as_entire_binding()),
                        entry(4, pos_buf.as_entire_binding()),
                        entry(
                            5,
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &pairs_buf,
                                offset: (off * 8) as u64,
                                size: std::num::NonZero::new((len * 8) as u64),
                            }),
                        ),
                    ],
                });
                (bg, (len * 2) as u32)
            })
            .collect();

        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for p in &geo.pos {
            let v = Vec3::from(*p);
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if mn.x > mx.x {
            mn = Vec3::splat(-100.0);
            mx = Vec3::splat(100.0);
        }

        // Stats for the screen-space synapse LOD in `syn_lod`.
        let ext = (mx - mn).max(Vec3::splat(1.0));
        let mut dims = [ext.x, ext.y, ext.z];
        dims.sort_by(|a, b| b.total_cmp(a));
        let silhouette = dims[0] * dims[1];
        let avg_spacing = (ext.x * ext.y * ext.z / n.max(1) as f32).cbrt();
        let mut edge_sum = 0.0f32;
        for &(a, b) in &geo.syn {
            if let (Some(&pa), Some(&pb)) = (geo.pos.get(a as usize), geo.pos.get(b as usize)) {
                edge_sum += (Vec3::from(pb) - Vec3::from(pa)).length();
            }
        }
        let avg_edge_len = edge_sum / geo.syn.len().max(1) as f32;

        self.scene = Some(Scene {
            n,
            entity_of_inst,
            inst_buf,
            activity_buf,
            sel_buf,
            pos_buf,
            pairs_buf,
            syn_verts: (draw_syns.min(n_syns) * 2) as u32,
            avg_spacing,
            avg_edge_len,
            silhouette,
            bounds: (mn, mx),
        });
    }

    fn ensure_targets(&mut self, w: u32, h: u32) {
        let need = match &self.depth {
            Some((_, dw, dh)) => *dw != w || *dh != h,
            None => true,
        };
        if need {
            let st = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let sv = st.create_view(&Default::default());
            self.blit_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit bg"),
                layout: &self.bgl_blit,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&sv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            }));
            self.scene_tex = Some((st, sv, w, h));
            let t = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((t.create_view(&Default::default()), w, h));
        }
        let pick_need = match &self.pick {
            Some(p) => p.w != w || p.h != h,
            None => true,
        };
        if pick_need {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pick"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let dtex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pick depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.pick = Some(PickTarget {
                view: tex.create_view(&Default::default()),
                tex,
                depth: dtex.create_view(&Default::default()),
                w,
                h,
            });
        }
    }

    pub fn write_cam(&mut self, cam: &Camera, aspect: f32, vp_h_px: f32, t: f32) {
        let vp = cam.view_proj(aspect);
        let eye = cam.eye();
        let fwd = (cam.target - eye).normalize_or(Vec3::Z);
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd).normalize_or(Vec3::Y);
        let (alpha_mul, keep) = self.syn_lod(cam, aspect, vp_h_px);
        let u = CamU {
            view_proj: vp.to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            exc_color: [0.55, 0.8, 1.0, 1.0],
            inh_color: [0.85, 0.6, 1.0, 1.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            misc: [4.0, 1.0, t, 0.0],
            lod: [alpha_mul, keep as f32, 0.0, 0.0],
        };
        self.cam_u = u;
        self.queue
            .write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&u));
    }

    /// Screen-space synapse LOD: returns (alpha multiplier, keep-1-of-N).
    ///
    /// Alpha: estimate mean line overdraw inside the projected silhouette —
    /// drawn_lines × mean length px ÷ max(screen, silhouette px) — and pick a
    /// per-line alpha that accumulates to a translucent volume instead of an
    /// opaque slab. Sparse views keep alpha 1.
    ///
    /// Subsampling: once mean neuron spacing drops below a few pixels,
    /// individual edges are unresolvable — the shader draws only every
    /// `keep`-th synapse; the alpha calc uses the reduced count so total
    /// coverage is preserved.
    fn syn_lod(&self, cam: &Camera, aspect: f32, vp_h_px: f32) -> (f32, u32) {
        let Some(sc) = &self.scene else {
            return (1.0, 1);
        };
        let n_syn = (sc.syn_verts / 2) as f32;
        if n_syn < 1.0 || vp_h_px <= 0.0 {
            return (1.0, 1);
        }
        // vertical fov is 60° → px per world unit at the target depth
        let px_per_unit = vp_h_px * 0.5 / (cam.dist * 0.577_350_3);
        let screen_px = (vp_h_px * vp_h_px * aspect).max(1.0);

        let spacing_px = sc.avg_spacing * px_per_unit;
        let keep = if spacing_px < 3.0 {
            ((3.0 / spacing_px.max(0.05)).powi(2).ceil() as u32).clamp(1, 64)
        } else {
            1
        };

        let drawn = n_syn / keep as f32;
        let len_px = (sc.avg_edge_len * px_per_unit).max(1.0);
        let sil_px = sc.silhouette * px_per_unit * px_per_unit;
        let overdraw = drawn * len_px / screen_px.max(sil_px);
        let alpha_mul = if overdraw > 1.0 {
            // per-line alpha so k overlaps accumulate to ~55% coverage:
            // a = 1 - (1-T)^(1/k), divided by the baked-in base alpha 0.25
            const T: f32 = 0.55;
            (1.0 - (1.0 - T).powf(1.0 / overdraw)) / 0.25
        } else {
            1.0
        };
        (alpha_mul.clamp(0.01, 1.0), keep)
    }

    /// Per-frame activity upload (one f32 per entity; extras zero).
    pub fn write_activity(&self, act: &[f32]) {
        if let Some(sc) = &self.scene {
            let total = sc.entity_of_inst.len();
            let mut buf = vec![0f32; total];
            let m = act.len().min(total);
            buf[..m].copy_from_slice(&act[..m]);
            self.queue
                .write_buffer(&sc.activity_buf, 0, bytemuck::cast_slice(&buf));
        }
    }

    pub fn write_selection(&self, inst_ids: &[u32]) {
        if let Some(sc) = &self.scene {
            let total = sc.entity_of_inst.len();
            let mut bits = vec![0u32; total / 32 + 1];
            for &i in inst_ids {
                if (i as usize) < total {
                    bits[i as usize / 32] |= 1 << (i % 32);
                }
            }
            self.queue
                .write_buffer(&sc.sel_buf, 0, bytemuck::cast_slice(&bits));
        }
    }

    /// Upload overlay lines (grid, boxes, drag helpers) for this frame.
    pub fn write_pv(&mut self, verts: &[Pv]) {
        let need = verts.len().max(1);
        if need > self.pv_cap {
            self.pv_cap = need.next_power_of_two();
            self.pv_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pv"),
                size: (self.pv_cap * std::mem::size_of::<Pv>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !verts.is_empty() {
            self.queue
                .write_buffer(&self.pv_buf, 0, bytemuck::cast_slice(verts));
        }
    }

    /// Upload spike-dot particle instances; returns how many fit.
    pub fn write_parts(&mut self, parts: &[Part]) -> usize {
        let need = parts.len().max(1);
        if need > self.part_cap {
            self.part_cap = need.next_power_of_two();
            self.part_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("part"),
                size: (self.part_cap * std::mem::size_of::<Part>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.part_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("part bg"),
                layout: &self.bgl_part,
                entries: &[
                    entry(0, self.cam_buf.as_entire_binding()),
                    entry(7, self.part_buf.as_entire_binding()),
                ],
            }));
        }
        if self.part_bg.is_none() {
            self.part_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("part bg"),
                layout: &self.bgl_part,
                entries: &[
                    entry(0, self.cam_buf.as_entire_binding()),
                    entry(7, self.part_buf.as_entire_binding()),
                ],
            }));
        }
        if !parts.is_empty() {
            self.queue
                .write_buffer(&self.part_buf, 0, bytemuck::cast_slice(parts));
        }
        parts.len()
    }

    /// Draw the whole scene into `rpass` (neurons, synapses, parts, overlay).
    pub fn render(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        pv_count: u32,
        part_count: u32,
        draw_syn: bool,
    ) {
        let Some(sc) = &self.scene else { return };
        if draw_syn && sc.syn_verts > 0 {
            rpass.set_pipeline(&self.pip_syn);
            for (bg, verts) in &self.syn_bgs {
                rpass.set_bind_group(0, bg, &[]);
                rpass.draw(0..*verts, 0..1);
            }
        }
        if part_count > 0 {
            rpass.set_pipeline(&self.pip_part);
            rpass.set_bind_group(0, self.part_bg.as_ref().unwrap(), &[]);
            rpass.draw(0..6, 0..part_count);
        }
        if sc.n > 0 {
            rpass.set_pipeline(&self.pip_neur);
            rpass.set_bind_group(0, self.neuron_bg.as_ref().unwrap(), &[]);
            rpass.draw(0..36, 0..sc.entity_of_inst.len() as u32);
        }
        if pv_count > 0 {
            rpass.set_pipeline(&self.pip_pv);
            rpass.set_bind_group(0, &self.pv_bg, &[]);
            rpass.set_vertex_buffer(0, self.pv_buf.slice(..));
            rpass.draw(0..pv_count, 0..1);
        }
    }

    /// Depth view for the current size (ensures allocation).
    pub fn depth_view(&mut self, w: u32, h: u32) -> wgpu::TextureView {
        self.ensure_targets(w, h);
        self.depth.as_ref().unwrap().0.clone()
    }

    /// Render the scene into the internal viewport texture (own encoder).
    pub fn render_offscreen(&mut self, w_px: u32, h_px: u32) {
        if self.scene.is_none() {
            return;
        }
        self.ensure_targets(w_px.max(1), h_px.max(1));
        let (pv, parts, syn) = match &self.draw_params {
            Some(p) => (p.pv_count, p.part_count, p.show_syn),
            None => (0, 0, true),
        };
        let (view, depth) = (
            self.scene_tex.as_ref().unwrap().1.clone(),
            self.depth.as_ref().unwrap().0.clone(),
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene"),
            });
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.035,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.render(&mut rpass, pv, parts, syn);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Blit the scene texture into egui's pass at `viewport_px` (x,y,w,h).
    pub fn blit(&self, rpass: &mut wgpu::RenderPass<'_>, vp: [u32; 4]) {
        let Some(bg) = &self.blit_bg else { return };
        rpass.set_viewport(
            vp[0] as f32,
            vp[1] as f32,
            vp[2] as f32,
            vp[3] as f32,
            0.0,
            1.0,
        );
        rpass.set_pipeline(&self.pip_blit);
        rpass.set_bind_group(0, bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Queue a single-pixel entity pick at (px,py). `hover` tags the result
    /// so the UI can tell hover picks (context menu, status bar) apart from
    /// click picks (selection).
    pub fn request_pick(&mut self, px: u32, py: u32, vw: u32, vh: u32, hover: bool) {
        if self.pick_pending.is_some() || self.scene.is_none() {
            return;
        }
        self.ensure_targets(vw.max(1), vh.max(1));
        // Pick mode: vs_neuron fattens cubes when misc.w != 0, so thin
        // neurons are clickable. The visible pass rewrites misc.w = 0
        // every frame in `write_cam`, so this never shows on screen.
        let mut u = self.cam_u;
        u.misc[3] = 1.0;
        self.queue
            .write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&u));
        let pick = self.pick.as_ref().unwrap();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pick"),
            });
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &pick.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let sc = self.scene.as_ref().unwrap();
            rpass.set_pipeline(&self.pip_pick);
            rpass.set_bind_group(0, self.neuron_bg.as_ref().unwrap(), &[]);
            rpass.draw(0..36, 0..sc.entity_of_inst.len() as u32);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);

        let result = Arc::new(std::sync::Mutex::new(None::<u32>));
        let res2 = result.clone();
        let buf = Arc::new(self.pick_buf.clone());
        let b2 = buf.clone();
        buf.map_async(wgpu::MapMode::Read, .., move |r| {
            let v = if r.is_ok() {
                let v = {
                    let data = b2.get_mapped_range(..4);
                    let words: &[u32] = bytemuck::cast_slice(&data);
                    words[0]
                };
                b2.unmap();
                v
            } else {
                0
            };
            *res2.lock().unwrap() = Some(v);
        });
        self.pick_pending = Some(PendingPick {
            result,
            hover,
            _buf: buf,
        });
    }

    /// Poll the pick result: outer None = still pending; inner None = miss.
    /// Returns (was_hover_request, picked_instance).
    pub fn poll_pick(&mut self) -> Option<(bool, Option<u32>)> {
        let p = self.pick_pending.as_ref()?;
        let v = *p.result.lock().unwrap();
        match v {
            None => None,
            Some(raw) => {
                let hover = p.hover;
                self.pick_pending = None;
                Some((hover, if raw == 0 { None } else { Some(raw - 1) }))
            }
        }
    }
}

fn storage() -> wgpu::BufferBindingType {
    wgpu::BufferBindingType::Storage { read_only: true }
}

fn bgl(
    device: &wgpu::Device,
    label: &str,
    binds: &[(u32, wgpu::BufferBindingType, wgpu::ShaderStages)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = binds
        .iter()
        .map(|(i, ty, st)| wgpu::BindGroupLayoutEntry {
            binding: *i,
            visibility: *st,
            ty: wgpu::BindingType::Buffer {
                ty: *ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn entry(i: u32, b: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: i,
        resource: b,
    }
}
