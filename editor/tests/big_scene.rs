//! A 10M-synapse scene must build its GPU buffers and draw without
//! tripping wgpu validation: the pairs binding is chunked under
//! `max_storage_buffer_binding_size` and the per-synapse payload is 8B.

use editor::{gpu::Gpu, snap::GeoSnap};

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
