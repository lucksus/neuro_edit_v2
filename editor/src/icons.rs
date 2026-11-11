//! Toolbar/menu artwork reused verbatim from the original C++/Qt
//! neuro_edit (`NeuroEdit/Resources`, GPL-3.0, same author).

use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};

fn tex(ctx: &Context, name: &str, png: &[u8]) -> TextureHandle {
    let img = image::load_from_memory(png).expect(name).to_rgba8();
    let (w, h) = img.dimensions();
    let ci = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
    ctx.load_texture(name, ci, TextureOptions::LINEAR)
}

pub struct Icons {
    pub add_neuron: TextureHandle,
    pub connect: TextureHandle,
    pub play: TextureHandle,
    pub pause: TextureHandle,
    pub load: TextureHandle,
    pub save: TextureHandle,
    pub about: TextureHandle,
}

impl Icons {
    pub fn load(ctx: &Context) -> Self {
        Self {
            add_neuron: tex(
                ctx,
                "ic_add_neuron",
                include_bytes!("../assets/add_neuron.png"),
            ),
            connect: tex(ctx, "ic_connect", include_bytes!("../assets/connect.png")),
            play: tex(ctx, "ic_play", include_bytes!("../assets/play.png")),
            pause: tex(ctx, "ic_pause", include_bytes!("../assets/pause.png")),
            load: tex(ctx, "ic_load", include_bytes!("../assets/load.png")),
            save: tex(ctx, "ic_save", include_bytes!("../assets/save.png")),
            about: tex(ctx, "ic_about", include_bytes!("../assets/about.png")),
        }
    }
}
