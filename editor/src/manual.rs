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
            ui.monospace(line);
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
                    ui.heading(text);
                }
                2 | 3 => {
                    ui.label(egui::RichText::new(text).strong());
                }
                _ => {
                    rich_line(ui, text);
                }
            }
        });
    }
}

/// Split `**bold**` / `` `code` `` spans into RichText fragments.
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
        ui.label(t);
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
