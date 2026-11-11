//! Minimal Rhai syntax highlighting for the script editor and console.
//! Single-pass tokenizer -> egui LayoutJob; block comments span lines
//! because the layouter always sees the whole buffer.

use eframe::egui;
use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, Galley, TextStyle, Ui};
use std::sync::Arc;

const KEYWORDS: &[&str] = &[
    "let",
    "const",
    "fn",
    "if",
    "else",
    "for",
    "while",
    "loop",
    "do",
    "until",
    "in",
    "return",
    "break",
    "continue",
    "true",
    "false",
    "null",
    "import",
    "export",
    "global",
    "match",
    "switch",
    "go",
    "spawn",
    "try",
    "catch",
    "throw",
    "private",
    "protected",
    "this",
];

/// Globals injected into the script scope by `sim::script`.
const GLOBALS: &[&str] = &["net", "sim", "rng", "print"];

fn is_id_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}
fn is_id_char(c: char) -> bool {
    is_id_start(c) || c.is_ascii_digit()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tok {
    Plain,
    Keyword,
    Global,
    Comment,
    Str,
    Num,
}

/// Tokenize `src` into (byte_range, class) spans covering the whole input.
fn tokens(src: &str) -> Vec<(usize, usize, Tok)> {
    let mut out: Vec<(usize, usize, Tok)> = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    macro_rules! tok {
        ($end:expr, $t:expr, $next:expr) => {{
            let from = out.last().map(|l| l.1).unwrap_or(0);
            if from < i {
                out.push((from, i, Tok::Plain));
            }
            out.push((i, $end, $t));
            i = $next;
        }};
    }
    while i < b.len() {
        let c = b[i] as char;
        // line comment
        if c == '/' && b.get(i + 1) == Some(&b'/') {
            let end = src[i..].find('\n').map(|k| i + k).unwrap_or(b.len());
            tok!(end, Tok::Comment, end);
            continue;
        }
        // block comment
        if c == '/' && b.get(i + 1) == Some(&b'*') {
            let end = src[i + 2..]
                .find("*/")
                .map(|k| i + 2 + k + 2)
                .unwrap_or(b.len());
            tok!(end, Tok::Comment, end);
            continue;
        }
        // string / char literal
        if c == '"' || c == '\'' {
            let q = b[i];
            let mut j = i + 1;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 1;
                } else if b[j] == q {
                    j += 1;
                    break;
                }
                j += 1;
            }
            tok!(j.min(b.len()), Tok::Str, j.min(b.len()));
            continue;
        }
        // number (digit run, or .<digit> not following an identifier)
        if c.is_ascii_digit()
            || (c == '.'
                && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())
                && (i == 0 || !is_id_char(b[i - 1] as char)))
        {
            let mut j = i;
            while j < b.len() && (is_id_char(b[j] as char) || b[j] == b'.') {
                j += 1;
            }
            tok!(j, Tok::Num, j);
            continue;
        }
        // identifier / keyword / global
        if is_id_start(c) {
            let mut j = i;
            while j < b.len() && is_id_char(b[j] as char) {
                j += 1;
            }
            let word = &src[i..j];
            let t = if KEYWORDS.contains(&word) {
                Tok::Keyword
            } else if GLOBALS.contains(&word) {
                Tok::Global
            } else {
                Tok::Plain
            };
            tok!(j, t, j);
            continue;
        }
        i += 1;
    }
    let from = out.last().map(|l| l.1).unwrap_or(0);
    if from < b.len() {
        out.push((from, b.len(), Tok::Plain));
    }
    out
}

/// Layouter for `TextEdit`: colorizes `src` as Rhai at `wrap` px width.
pub fn highlight(ui: &Ui, src: &str, wrap: f32) -> Arc<Galley> {
    let v = ui.visuals();
    let plain = v.text_color();
    let color_of = |t: Tok| match t {
        Tok::Plain => plain,
        Tok::Keyword => Color32::from_rgb(0xC5, 0x86, 0xC0),
        Tok::Global => Color32::from_rgb(0x4E, 0xC9, 0xB0),
        Tok::Comment => Color32::from_rgb(0x6A, 0x99, 0x55),
        Tok::Str => Color32::from_rgb(0xCE, 0x91, 0x78),
        Tok::Num => Color32::from_rgb(0xB5, 0xCE, 0xA8),
    };
    let font = TextStyle::Monospace.resolve(ui.style());
    let mut job = LayoutJob::default();
    job.wrap.max_width = wrap;
    for (a, z, t) in tokens(src) {
        job.append(
            &src[a..z],
            0.0,
            TextFormat {
                font_id: font.clone(),
                color: color_of(t),
                ..Default::default()
            },
        );
    }
    ui.fonts(|f| f.layout_job(job))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class_at(src: &str, needle: &str) -> Tok {
        let off = src.find(needle).expect("needle");
        tokens(src)
            .iter()
            .find(|(a, z, _)| off >= *a && off + needle.len() <= *z)
            .expect("span")
            .2
    }

    #[test]
    fn covers_whole_input() {
        for s in [
            "let x = sim.run(10.0); // hi",
            "/* a\nb */ net.connect(1,2)",
            "let s = \"esc\\\"q\" + 'c'",
            "x = 1.5e3.foo",
            "",
            "// only a comment",
        ] {
            let t = tokens(s);
            let mut off = 0;
            for (a, z, _) in &t {
                assert_eq!(*a, off, "gap before span at {a} in {s:?}");
                off = *z;
            }
            assert_eq!(off, s.len());
        }
    }

    #[test]
    fn classifies() {
        let s = "let x = 0.5; // c\nsim.run(x) + \"txt\"";
        assert_eq!(class_at(s, "let"), Tok::Keyword);
        assert_eq!(class_at(s, "0.5"), Tok::Num);
        assert_eq!(class_at(s, "// c"), Tok::Comment);
        assert_eq!(class_at(s, "sim"), Tok::Global);
        assert_eq!(class_at(s, "run"), Tok::Plain);
        assert_eq!(class_at(s, "\"txt\""), Tok::Str);
        // block comment spanning a newline
        let s = "a /* x\ny */ b";
        let t = tokens(s);
        assert_eq!(t[2], (2usize, 11usize, Tok::Comment));
    }
}
