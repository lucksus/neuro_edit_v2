// Check WebGPU BEFORE the wasm bundle loads: without it the editor
// can't start anyway, and a clear how-to page beats a dead canvas.
// webgpu-help.mjs is a plain ES module — no build step.
// Kept out of index.html so the hosted CSP can stay free of
// 'unsafe-inline' scripts (stage 16B).
import { boot } from './webgpu-help.mjs';
import init from './editor.js';
boot(() => init());
