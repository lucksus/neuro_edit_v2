//! In-app manual: `docs/manual/*.md` baked into the binary (native and
//! wasm both get it for free) plus a deliberately tiny markdown→egui
//! renderer — headings, fenced code, bullets, bold/code spans.
//! Not a general markdown engine; the pages are written to fit it.

use eframe::egui;

pub struct Page {
    pub title: &'static str,
    pub md: &'static str,
}

/// The manual's right-hand body column: a bounded-width scroll area. Without
/// `set_max_width` the parent `ui.horizontal` hands down an infinite x bound
/// and prose never wraps (runs off the window's right edge).
pub fn body_scroll(ui: &mut egui::Ui, width: f32, page: usize) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_max_width(width.max(80.0));
            render(ui, PAGES[page.min(PAGES.len() - 1)].md);
        });
}

pub const PAGES: &[Page] = &[
    Page {
        title: "Getting started",
        md: include_str!("../../docs/manual/getting-started.md"),
    },
    Page {
        title: "Controls",
        md: include_str!("../../docs/manual/controls.md"),
    },
    Page {
        title: "Console",
        md: include_str!("../../docs/manual/console.md"),
    },
    Page {
        title: "Plots",
        md: include_str!("../../docs/manual/plots.md"),
    },
];

/// Render one page. `**bold**` and `` `code` `` spans are honoured
/// inline; everything else is plain wrapped text.
pub fn render(ui: &mut egui::Ui, md: &str) {
    let mut in_code = false;
    for raw in md.lines() {
        let line = raw.trim_end();
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if line.is_empty() {
            ui.add_space(6.0);
            continue;
        }
        if in_code {
            ui.add(egui::Label::new(egui::RichText::new(line).monospace()).wrap());
            continue;
        }
        let (h, rest) = match line.strip_prefix('#') {
            Some(l) if line.starts_with("### ") => (3, &l[2..]),
            Some(l) if line.starts_with("## ") => (2, &l[1..]),
            Some(l) if line.starts_with("# ") => (1, &l[0..]),
            _ => (0, line),
        };
        let bullet = rest.strip_prefix("- ").or_else(|| rest.strip_prefix("* "));
        let text = bullet.unwrap_or(rest);
        match h {
            1 => {
                ui.add(egui::Label::new(egui::RichText::new(text).heading()).wrap());
            }
            2 | 3 => {
                ui.add(egui::Label::new(egui::RichText::new(text).strong()).wrap());
            }
            _ => {
                rich_line(ui, text, bullet.is_some());
            }
        }
    }
}

/// Split `**bold**` / `` `code` `` spans into ONE wrapped LayoutJob.
/// A single job wraps as a whole paragraph; multiple wrapping Labels
/// inside `horizontal_wrapped` get squeezed into the row's leftover
/// sliver and wrap into word-per-line columns instead.
fn rich_line(ui: &mut egui::Ui, s: &str, bullet: bool) {
    let mut job = egui::text::LayoutJob::default();
    let body =
        egui::TextFormat::simple(egui::FontId::proportional(14.0), ui.visuals().text_color());
    let mut bold_f = body.clone();
    bold_f.color = ui.visuals().strong_text_color();
    let mut code_f = body.clone();
    code_f.font_id = egui::FontId::monospace(13.0);
    code_f.background = ui.visuals().code_bg_color;
    if bullet {
        job.append("\u{2022} ", 0.0, body.clone());
    }
    let mut buf = String::new();
    let mut bold = false;
    let mut code = false;
    let mut chars = s.chars().peekable();
    macro_rules! flush {
        () => {
            if !buf.is_empty() {
                let f = if code {
                    code_f.clone()
                } else if bold {
                    bold_f.clone()
                } else {
                    body.clone()
                };
                job.append(&std::mem::take(&mut buf), 0.0, f);
            }
        };
    }
    while let Some(c) = chars.next() {
        if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            flush!();
            bold = !bold;
        } else if c == '`' {
            flush!();
            code = !code;
        } else {
            buf.push(c);
        }
    }
    flush!();
    ui.add(egui::Label::new(job).wrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embedded page must have a title heading and body text —
    /// guards against a moved/renamed docs path silently baking an
    /// empty manual.
    #[test]
    fn manual_pages_are_embedded_and_non_empty() {
        assert!(PAGES.len() >= 4);
        for p in PAGES {
            assert!(!p.title.is_empty());
            assert!(p.md.lines().any(|l| l.starts_with("# ")), "{}", p.title);
            assert!(p.md.len() > 200, "{} looks empty", p.title);
        }
        // Stage 13 K: the controls page must document the move keys.
        let controls = PAGES.iter().find(|p| p.title == "Controls").unwrap();
        assert!(controls.md.contains("Shift"));
        assert!(controls.md.contains("X / Y / Z"));
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// The rendered page must wrap inside the window — a long body line
    /// must never produce a text galley wider than the available width
    /// (egui's Shape::Text galley size is pre-clip, so overflow shows up
    /// here even when the window's own clip rect would hide it on screen).
    #[test]
    fn body_text_wraps_to_window() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(360.0, 480.0),
            )),
            ..Default::default()
        };
        // Mirror app::manual_window: a `ui.horizontal` (unbounded child x!)
        // around the shared `body_scroll` column, inside a Window.
        let frame = |page: usize| {
            ctx.run(input.clone(), |ctx| {
                egui::Window::new("Manual").show(ctx, |ui| {
                    let w = (ui.clip_rect().width() - 110.0 - 40.0).max(80.0);
                    let h = ui.available_height().max(60.0);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.set_width(110.0);
                        });
                        ui.separator();
                        ui.allocate_ui_with_layout(
                            egui::vec2(w, h),
                            egui::Layout::top_down(egui::Align::LEFT),
                            |ui| body_scroll(ui, w, page),
                        );
                    });
                });
            })
        };
        for page in 0..PAGES.len() {
            let _ = frame(page); // first frame: layout pass
            let out = frame(page); // second frame: painted content
            let mut widest = 0.0f32;
            for shape in &out.shapes {
                if let egui::epaint::Shape::Text(t) = &shape.shape {
                    widest = widest.max(t.galley.size().x);
                }
            }
            assert!(
                widest > 0.0,
                "no text was painted for {}",
                PAGES[page].title
            );
            assert!(
                widest <= 340.0,
                "{}: a line painted {widest}px wide in a 360px window — text is not wrapping",
                PAGES[page].title
            );
        }
    }
    /// Any non-ASCII char in a user-visible string must exist in the bundled
    /// fonts, or egui paints a tofu box. Manual prose renders proportional,
    /// fenced code monospace; `src/*.rs` lines containing a `"` are treated
    /// as string-bearing and checked against the proportional family (the
    /// strict one). Comment-only lines carry no `"` and are skipped.
    #[test]
    fn ui_strings_stay_in_font_coverage() {
        let fonts = egui::epaint::Fonts::new(
            1.0,
            1024,
            egui::epaint::AlphaFromCoverage::Linear,
            egui::FontDefinitions::default(),
        );
        let prop = egui::FontId::proportional(14.0);
        let mono = egui::FontId::monospace(13.0);
        let mut bad = Vec::new();

        for page in PAGES {
            let mut fenced = false;
            for (ln, line) in page.md.lines().enumerate() {
                if line.trim_start().starts_with("```") {
                    fenced = !fenced;
                    continue;
                }
                let id = if fenced { &mono } else { &prop };
                for c in line.chars().filter(|c| !c.is_ascii()) {
                    if !fonts.has_glyph(id, c) {
                        bad.push(format!("{}:{} {c:?}", page.title, ln + 1));
                    }
                }
            }
        }

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for e in std::fs::read_dir(&src).unwrap() {
            let e = e.unwrap();
            if e.path().extension().and_then(|x| x.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(e.path()).unwrap();
            for (ln, line) in text.lines().enumerate() {
                if !line.contains('"') {
                    continue;
                }
                for c in line.chars().filter(|c| !c.is_ascii()) {
                    if !fonts.has_glyph(&prop, c) {
                        bad.push(format!("{}:{} {c:?}", e.path().display(), ln + 1));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "glyphs missing from bundled fonts:\n{}",
            bad.join("\n")
        );
    }

    #[test]
    fn probe_window_widths() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            ..Default::default()
        };
        for i in 0..3 {
            let out = ctx.run(input.clone(), |ctx| {
                egui::Window::new("Manual")
                    .default_size([640.0, 520.0])
                    .default_pos([90.0, 60.0])
                    .show(ctx, |ui| {
                        let body_w = (ui.clip_rect().width() - 150.0).clamp(80.0, 480.0);
                        let body_h = ui.available_height().max(60.0);
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(110.0);
                                for p in PAGES {
                                    ui.selectable_label(false, p.title);
                                }
                            });
                            ui.separator();
                            ui.allocate_ui_with_layout(
                                egui::vec2(body_w, body_h),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| body_scroll(ui, body_w, 0),
                            );
                        });
                    });
            });
            let mut widest = 0.0f32;
            for shape in &out.shapes {
                if let egui::epaint::Shape::Text(t) = &shape.shape {
                    widest = widest.max(t.galley.size().x);
                }
            }
            let mut clipr = None;
            for shape in &out.shapes {
                if let egui::epaint::Shape::Text(t) = &shape.shape {
                    if t.galley.size().x >= widest {
                        clipr = Some(shape.clip_rect);
                    }
                }
            }
            eprintln!("frame{i}: widest_text={widest:.0} clip={clipr:?}");
        }
    }
}
