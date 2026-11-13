# Stage 19 — split editor/src/app.rs (no behaviour change)

Repo: ~/code/neuro_edit_v2 on miguel. Commit to `main` only. Do NOT push, do NOT touch
~/deploy/neuroedit or the running `neuroedit` service.

## Why
Nico approved the refactors (2026-11-13). The next stages put the model abstraction and a new
guided UI (Build / Run / Script states, docked console, inspector plots, undo) into the editor.
`editor/src/app.rs` (~4500 lines, one struct with ~110 fields) makes that work slow and risky.

## Spec
Follow `review/fable-review-1112.md` section 3.4 exactly: layout table and steps 1-5.
Line numbers there are from an older commit; find the code by function name.

1. `app.rs` -> `app/mod.rs` unchanged; free functions + their tests -> `app/layout.rs`.
2. One window per commit into its own file as `impl EditorApp` blocks (`pub(super)`).
3. `dialog()` -> one function per `Dialog` variant in `app/dialogs.rs`.
4. Group fields into sub-structs (`SimControls`, `PlotsUi`, `FilesUi`, `LoginUi`, `JobsUi`,
   `ScriptUi`, `ConsoleUi`, `Windows`), one group per commit, files first.
5. Default window rectangles into one table in `layout.rs`, plus a test that no two
   default-open windows overlap at 800x700 and 1440x900.

Also fix the two items listed at the end of 3.4:
- `sync_shared`: copy what is needed out of `Shared`, drop the lock once, then act.
- Dialogs that build Rhai with `format!` and send `Eval` (e.g. `net.set_exc_inh(..)`): replace
  with typed `NetEdit` variants handled in `sim/src/serve.rs`. One test per new variant that
  sends it over the protocol and checks the network changed. This is the only behaviour-
  touching part; keep the old script functions working.

## Rules
- Every commit compiles and keeps all tests green: `cargo test --workspace` (CUDA available
  on this box), plus the editor wasm build `editor/web/build.sh`.
- Moves are cut-and-paste. Do not rename, restyle or "improve" code you move.
- Before the first commit take screenshots at 800 and 1440 px with the existing shot tool
  (`editor/src/shot.rs`); after the last commit take them again. Save both sets under
  `review/stage19_*` and state in NOTES whether they differ (they must not).
- No file in `editor/src/app/` above ~700 lines unless you explain why in NOTES.
- No Co-Authored-By trailers.

## Report
Append a `## Stage 19` section to `NOTES-for-data.md`: commit list, final file sizes, test
counts before/after, screenshot comparison, anything you could not move and why.
