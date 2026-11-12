//! A 10M-synapse scene must build its GPU buffers and draw without
//! tripping wgpu validation: the pairs binding is chunked under
//! `max_storage_buffer_binding_size` and the per-synapse payload is 8B.

use editor::{
    gpu::{DrawParams, Gpu, PlaneParamsGpu},
    plane::{self, PlaneCfg},
    snap::{GeoSnap, GridSnap},
};

fn adapter_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    Some(pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default()),
    ))
    .map(|t| t.unwrap())
}

fn big_geo(n_neurons: u32, n_syns: usize) -> GeoSnap {
    let pos: Vec<[f32; 3]> = (0..n_neurons)
        .map(|i| [i as f32 % 100.0, (i / 100) as f32 * 8.0, 0.0])
        .collect();
    let kind: Vec<u8> = (0..n_neurons).map(|i| (i % 5 == 4) as u8).collect();
    let mut syn = Vec::with_capacity(n_syns);
    for i in 0..n_syns {
        let a = (i as u32) % n_neurons;
        let b = ((i as u32) * 7 + 1) % n_neurons;
        syn.push((a, b));
    }
    GeoSnap {
        version: 0,
        pos,
        kind,
        names: Vec::new(),
        syn,
        syn_delay_steps: Vec::new(),
        disp_edges: Vec::new(),
        disp_ptr: Vec::new(),
        groups: Vec::new(),
        inputs: Vec::new(),
        readouts: Vec::new(),
        samples: Vec::new(),
        scripts: Vec::new(),
        grids: Vec::new(),
        dt_ms: 0.1,
    }
}

#[test]
fn ten_million_synapses_build_and_draw() {
    let Some((device, queue)) = adapter_device() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut gpu = Gpu::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    gpu.set_scene(&big_geo(1000, 10_000_000));
    let sc = gpu.scene.as_ref().unwrap();
    assert_eq!(sc.syn_verts, 20_000_000);
    // A frame draw: wgpu validation failures panic here if the chunked
    // bindings were wrong.
    gpu.render_offscreen(64, 64);
    gpu.device.poll(wgpu::PollType::Wait).unwrap();
}

/// A 1000x1000-neuron grid renders as ONE textured quad: scene upload,
/// trace texture upload (with 256B-aligned rows), hide bits and the draw
/// all stay inside a modest budget. Rendering the same neurons as cubes
/// would be 36M instanced verts.
#[test]
fn million_neuron_grid_plane_draws() {
    let Some((device, queue)) = adapter_device() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    const W: u32 = 1000;
    const H: u32 = 1000;
    let n = (W * H) as usize;

    let t0 = std::time::Instant::now();
    let mut geo = big_geo(n as u32, 0);
    // Regular lattice along +x / +y, one group covering everything.
    geo.pos = (0..n as u32)
        .map(|i| [(i % W) as f32 * 4.0, (i / W) as f32 * 4.0, 0.0])
        .collect();
    geo.grids = vec![GridSnap {
        group: 0,
        dims: [W, H, 1],
        axes: [0, 1, 2],
        origin: [0.0, 0.0, 0.0],
        spacing: 4.0,
        members: (0..n as u32).collect(),
    }];

    let mut gpu = Gpu::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    gpu.set_scene(&geo);
    let build_s = t0.elapsed().as_secs_f64();

    // Fake a trace: bright checkerboard so the draw is non-trivial.
    let t1 = std::time::Instant::now();
    let gs = &geo.grids[0];
    let tr: Vec<f32> = (0..n)
        .map(|k| ((k % W as usize + k / W as usize) % 7 == 0) as u8 as f32)
        .collect();
    let (w, h, pix, pick) = plane::reduce(gs, &tr, &PlaneCfg::default());
    assert_eq!((w, h), (W, H));
    gpu.ensure_plane_tex(0, w, h);
    gpu.write_plane(0, &pix, &pick);
    let (ua, va) = plane::plane_dims(gs, &PlaneCfg::default());
    let (origin, du, dv) = plane::quad(gs, &PlaneCfg::default());
    gpu.write_plane_params(&[PlaneParamsGpu {
        origin,
        du,
        dv,
        wh: [gs.dims[ua], gs.dims[va], 0, 0],
    }]);
    let mut bits = vec![0u32; n / 32 + 1];
    for &m in &gs.members {
        bits[m as usize >> 5] |= 1 << (m & 31);
    }
    gpu.write_hide_bits(&bits);
    let upload_s = t1.elapsed().as_secs_f64();

    let t2 = std::time::Instant::now();
    gpu.draw_params = Some(DrawParams {
        pv_count: 0,
        part_count: 0,
        show_syn: false,
        planes_on: None,
    });
    gpu.render_offscreen(1280, 800);
    gpu.device.poll(wgpu::PollType::Wait).unwrap();
    let draw_s = t2.elapsed().as_secs_f64();

    eprintln!("1M-grid plane: build {build_s:.2}s upload {upload_s:.2}s draw {draw_s:.2}s");
    // Generous ceilings — CI GPUs vary wildly; the point is that this
    // completes at all (the cube path would not fit comfortably).
    assert!(build_s < 30.0, "scene build {build_s:.1}s");
    assert!(upload_s < 30.0, "plane upload {upload_s:.1}s");
    assert!(draw_s < 10.0, "draw {draw_s:.1}s");
}
