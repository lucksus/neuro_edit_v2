//! Headless screenshot mode: render the 3D scene to a PNG with no window —
//! `editor --screenshot out.png --net x.ne2 --after-ms 500`.

use anyhow::{bail, Context};
use net_format::Network;
use sim::{BackendKind, Sim, SimConfig};

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::scene;
use crate::simctl::geo_from_net;
use crate::snap::GeoSnap;

pub struct ShotOpts {
    pub net_path: Option<String>,
    pub script: Option<String>,
    pub after_ms: f32,
    pub width: u32,
    pub height: u32,
    pub out: String,
    pub backend: BackendKind,
    pub seed: u64,
    pub show_syn: bool,
    pub int_weights: bool,
    /// Optional camera overrides applied after auto-framing.
    pub cam_dist: Option<f32>,
    pub cam_yaw: Option<f32>,
    pub cam_pitch: Option<f32>,
}

pub async fn render_headless(o: ShotOpts) -> anyhow::Result<()> {
    // ---- build network ----
    let mut net = match &o.net_path {
        Some(p) => net_format::io::load(p).with_context(|| format!("load {p}"))?,
        None => Network::new(),
    };

    // ---- optional script builds/drives the net ----
    if let Some(path) = &o.script {
        let mut env = sim::script::Env::new(std::mem::take(&mut net), o.backend, o.seed)?;
        let _ = env.eval_file(path).map_err(|e| anyhow::anyhow!(e))?;
        net = std::mem::take(&mut *env.net.borrow_mut());
    }
    if net.neurons.is_empty() {
        bail!("empty network — nothing to render");
    }

    // ---- run the sim a bit so the frame shows activity ----
    let geo: GeoSnap = geo_from_net(&net, 0.1);
    let dt = 0.1f32;
    let mut sim = Sim::new(
        &net,
        SimConfig {
            seed: o.seed,
            record_spikes: true,
            int_weights: o.int_weights,
            ..Default::default()
        },
        o.backend,
    )?;
    let steps = (o.after_ms / dt).round().max(0.0) as usize;
    sim.run_steps(steps)?;
    let activity = sim.trace();

    // ---- wgpu headless ----
    let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = inst
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("no GPU adapter for screenshot")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    let mut gpu = Gpu::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    gpu.set_scene(&geo);
    gpu.write_activity(&activity);

    let w = o.width.max(8);
    let h = o.height.max(8);
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let depth = gpu.depth_view(w, h);

    let (mn, mx) = gpu.scene.as_ref().unwrap().bounds;
    let mut cam = Camera::default();
    cam.frame(mn, mx);
    if let Some(d) = o.cam_dist {
        cam.dist = d;
    }
    if let Some(y) = o.cam_yaw {
        cam.yaw_deg = y;
    }
    if let Some(p) = o.cam_pitch {
        cam.pitch_deg = p;
    }
    cam.clamp_above_floor();
    gpu.write_cam(&cam, w as f32 / h as f32, h as f32, 0.0);

    // overlay: floor grid + group boxes
    let mut pvs = Vec::new();
    scene::floor_grid(cam.target, &mut pvs);
    scene::group_boxes(&geo, &mut pvs);
    gpu.write_pv(&pvs);
    let pv_count = pvs.len() as u32;

    // particles from the last 300 steps of the run
    let mut parts = scene::Particles::new();
    let counts = sim.spike_counts((300.0 / dt) as usize).unwrap_or_default();
    let now_ms = sim.time_ms();
    parts.feed(&geo, &counts, now_ms);
    let pinst = parts.instances(now_ms + 1.0);
    let part_count = gpu.write_parts(&pinst) as u32;

    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shot"),
        });
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shot"),
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
        gpu.render(&mut rpass, pv_count, part_count, o.show_syn);
    }
    gpu.queue.submit([enc.finish()]);

    // time the scene render alone (re-render into the same target)
    let t0 = std::time::Instant::now();
    {
        let mut enc2 = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t") });
        {
            let mut rpass = enc2.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("timed"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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
            gpu.render(&mut rpass, pv_count, part_count, o.show_syn);
        }
        gpu.queue.submit([enc2.finish()]);
        let _ = gpu.device.poll(wgpu::PollType::Wait);
    }
    let frame_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!("scene render: {frame_ms:.2} ms");

    // ---- readback ----
    let rb_size = (w * h * 4) as u64;
    let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb"),
        size: rb_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rb") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([enc.finish()]);

    let (tx, rx) = std::sync::mpsc::channel();
    rb.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::wait()).unwrap();
    rx.recv()
        .context("map failed")?
        .map_err(|e| anyhow::anyhow!("map: {e:?}"))?;

    let mut png = Vec::with_capacity((w * h * 4) as usize);
    {
        let data = rb.get_mapped_range(..);
        // BGRA? no — Rgba8UnormSrgb is RGBA already.
        let mut row = vec![0u8; (w * 4) as usize];
        for y in 0..h {
            let s = (y * w * 4) as usize;
            row.copy_from_slice(&data[s..s + (w * 4) as usize]);
            png.extend_from_slice(&row);
        }
    }
    rb.unmap();

    image::save_buffer(&o.out, &png, w, h, image::ColorType::Rgba8)
        .with_context(|| format!("save {}", o.out))?;
    println!(
        "wrote {} ({}x{}, {} neurons, {} synapses, t={}ms)",
        o.out,
        w,
        h,
        net.neurons.len(),
        net.synapses.len(),
        sim.time_ms()
    );
    Ok(())
}
