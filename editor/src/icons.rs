//! Toolbar, plot-tab and menu icons plus the About banner (Stage 10
//! artwork; SVG masters in `assets/src/`).
//!
//! Tool icons are white 48 px PNGs. egui-wgpu 0.32 creates every managed
//! texture with one mip level and ignores `TextureOptions::mipmap_mode`,
//! so drawing 48 px art at 16 px would alias. Instead each icon is
//! box-filtered on the CPU to exactly 16 px × `pixels_per_point` and
//! reloaded when that changes. White art × tint = tint, so the buttons
//! colour the icons with their own text colour.

use eframe::egui::{
    self, Button, Color32, ColorImage, Context, Image, Response, TextureHandle, TextureOptions, Ui,
    WidgetText,
};

/// Icon size in points.
pub const SIZE: f32 = 16.0;

fn tex(ctx: &Context, name: &str, png: &[u8], px: Option<u32>) -> TextureHandle {
    let mut img = image::load_from_memory(png).expect(name).to_rgba8();
    if let Some(px) = px {
        if img.width() != px {
            // area average: every source pixel lands in exactly one target pixel
            img = image::imageops::thumbnail(&img, px, px);
        }
    }
    let (w, h) = img.dimensions();
    let ci = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
    ctx.load_texture(name, ci, TextureOptions::LINEAR)
}

macro_rules! icons {
    ($($field:ident = $file:literal),* $(,)?) => {
        #[derive(Clone)]
        pub struct Icons {
            /// `pixels_per_point` the tool icons were scaled for
            pub ppp: f32,
            pub banner: TextureHandle,
            $(pub $field: TextureHandle,)*
        }

        impl Icons {
            pub fn load(ctx: &Context) -> Self {
                Self::load_at(ctx, ctx.pixels_per_point())
            }

            /// Tool icons scaled for `ppp` (16 px at 1x, 32 px at 2x, at
            /// most the 48 px source).
            pub fn load_at(ctx: &Context, ppp: f32) -> Self {
                let px = Some(((SIZE * ppp).round() as u32).clamp(SIZE as u32, 48));
                Self {
                    ppp,
                    banner: tex(ctx, "ic_banner", include_bytes!("../assets/about.png"), None),
                    $($field: tex(
                        ctx,
                        concat!("ic_", $file),
                        include_bytes!(concat!("../assets/icons/", $file, ".png")),
                        px,
                    ),)*
                }
            }
        }
    };
}

icons! {
    // sim
    run = "run", stop = "stop", step = "step", step_long = "step_long",
    reset = "reset", speed = "speed", timestep = "timestep", noise = "noise",
    backend_gpu = "backend_gpu", backend_cpu = "backend_cpu",
    // edit modes
    navigate = "navigate", insert = "insert", connect = "connect", mv = "move",
    // view
    plane_xz = "plane_xz", plane_xy = "plane_xy", plane_yz = "plane_yz",
    synapses = "synapses", grid = "grid", frame = "frame",
    view_auto = "view_auto", view_cubes = "view_cubes", view_planes = "view_planes",
    // panels
    console = "console", plots = "plots", inspector = "inspector", script = "script",
    // plot tabs
    voltage = "voltage", raster = "raster", readout = "readout", samples = "samples",
    phase = "phase", jobs = "jobs",
    // File / Help menus
    new = "new", open = "open", recent = "recent", save = "save", save_as = "save_as",
    quit = "quit", about = "about",
}

impl Icons {
    /// Load once, and again when the window moves to a screen with a
    /// different scale.
    pub fn ensure(slot: &mut Option<Self>, ctx: &Context) {
        if slot
            .as_ref()
            .is_some_and(|i| (i.ppp - ctx.pixels_per_point()).abs() > 0.01)
        {
            *slot = None;
        }
        slot.get_or_insert_with(|| Self::load(ctx));
    }
}

/// A tool icon at [`SIZE`], untinted (white).
pub fn img(t: &TextureHandle) -> Image<'static> {
    Image::new((t.id(), egui::vec2(SIZE, SIZE))).fit_to_exact_size(egui::vec2(SIZE, SIZE))
}

/// A tool icon tinted with `color` (for widgets that don't tint images
/// themselves, e.g. checkboxes and labels).
pub fn tinted(t: &TextureHandle, color: Color32) -> Image<'static> {
    img(t).tint(color)
}

/// Icon + text button; the icon follows the button's text colour
/// (hover, selected and disabled included).
pub fn button<'a>(t: &TextureHandle, text: impl Into<WidgetText>) -> Button<'a> {
    Button::image_and_text(img(t), text).image_tint_follows_text_color(true)
}

/// Icon-only button; say what it does with `on_hover_text`.
pub fn icon_button<'a>(t: &TextureHandle) -> Button<'a> {
    Button::image(img(t)).image_tint_follows_text_color(true)
}

/// Icon + text selectable (frame only when selected), icon tinted like
/// the text.
pub fn selectable<'a>(
    selected: bool,
    t: &TextureHandle,
    text: impl Into<WidgetText>,
) -> Button<'a> {
    Button::selectable(selected, (img(t), text.into())).image_tint_follows_text_color(true)
}

/// `ui.selectable_value` with an icon.
pub fn selectable_value<V: PartialEq>(
    ui: &mut Ui,
    current: &mut V,
    value: V,
    t: &TextureHandle,
    text: impl Into<WidgetText>,
) -> Response {
    let mut r = ui.add(selectable(*current == value, t, text));
    if r.clicked() && *current != value {
        *current = value;
        r.mark_changed();
    }
    r
}

/// `ui.toggle_value` with an icon: on/off for a show-X flag.
pub fn toggle(
    ui: &mut Ui,
    on: &mut bool,
    t: &TextureHandle,
    text: impl Into<WidgetText>,
) -> Response {
    let mut r = ui.add(selectable(*on, t, text));
    if r.clicked() {
        *on = !*on;
        r.mark_changed();
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_icons_are_scaled_to_16_points() {
        let ctx = Context::default();
        for (ppp, px) in [(1.0, 16), (1.5, 24), (2.0, 32), (3.0, 48), (4.0, 48)] {
            let ic = Icons::load_at(&ctx, ppp);
            assert_eq!(ic.run.size(), [px, px], "ppp {ppp}");
            assert_eq!(ic.about.size(), [px, px], "ppp {ppp}");
        }
        // the banner is never rescaled: 2x art for a 370×123 slot
        assert_eq!(Icons::load_at(&ctx, 1.0).banner.size(), [740, 246]);
    }
}
