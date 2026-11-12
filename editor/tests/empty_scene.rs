//! The editor starts on an empty network: building the GPU scene for it must
//! not create zero-sized storage bindings (wgpu validation panic on launch).

use editor::{gpu::Gpu, simctl::geo_from_net};
use net_format::Network;

#[test]
fn set_scene_accepts_empty_network() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
    else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let mut gpu = Gpu::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    gpu.set_scene(&editor::snap::geo_snap_from(&geo_from_net(
        &Network::new(),
        0.1,
    )));
}
