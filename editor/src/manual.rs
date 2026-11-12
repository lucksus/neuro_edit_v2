//! In-app manual: `docs/manual/*.md` baked into the binary (native and
//! wasm both get it for free) plus a deliberately tiny markdown→egui
//! renderer — headings, fenced code, bullets, bold/code spans.
//! Not a general markdown engine; the pages are written to fit it.

use eframe::egui;

pub struct Page {
    pub title: &'static str,
    pub md: &'static str,
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
        ui.horizontal_wrapped(|ui| {
            if bullet.is_some() {
                ui.label("• ");
            }
            let text = bullet.unwrap_or(rest);
            match h {
                1 => {
                    ui.add(egui::Label::new(egui::RichText::new(text).heading()).wrap());
                }
                2 | 3 => {
                    ui.add(egui::Label::new(egui::RichText::new(text).strong()).wrap());
                }
                _ => {
                    rich_line(ui, text);
                }
            }
        });
    }
}

/// Split `**bold**` / `` `code` `` spans into RichText fragments.
/// Each fragment label wraps, so a long paragraph can't run off the
/// window edge (`horizontal_wrapped` gives a wrapped label the rest
/// of the row and it breaks there).
fn rich_line(ui: &mut egui::Ui, s: &str) {
    let mut buf = String::new();
    let mut bold = false;
    let mut code = false;
    let flush = |ui: &mut egui::Ui, buf: &mut String, bold: bool, code: bool| {
        if buf.is_empty() {
            return;
        }
        let mut t = egui::RichText::new(std::mem::take(buf));
        if bold {
            t = t.strong();
        }
        if code {
            t = t.monospace();
        }
        ui.add(egui::Label::new(t).wrap());
    };
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            flush(ui, &mut buf, bold, code);
            bold = !bold;
        } else if c == '`' {
            flush(ui, &mut buf, bold, code);
            code = !code;
        } else {
            buf.push(c);
        }
    }
    flush(ui, &mut buf, bold, code);
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
        let md = "One deliberately long body sentence that cannot possibly fit on a single row at this narrow window width, so wrapping is the only correct outcome for the renderer to choose.";
        let mut out = ctx.run(input.clone(), |ctx| {
            egui::Window::new("Manual").show(ctx, |ui| render(ui, md));
        });
        // a second frame so the window has laid out and painted content
        out = ctx.run(input, |ctx| {
            egui::Window::new("Manual").show(ctx, |ui| render(ui, md));
        });
        let mut widest = 0.0f32;
        for shape in &out.shapes {
            if let egui::epaint::Shape::Text(t) = &shape.shape {
                widest = widest.max(t.galley.size().x);
            }
        }
        assert!(widest > 0.0, "no text was painted at all");
        assert!(
            widest <= 340.0,
            "a body line painted {widest}px wide in a 360px window — text is not wrapping"
        );
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
}