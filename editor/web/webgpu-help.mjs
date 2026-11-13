// webgpu-help.js — plain ES module, no build step, no browser globals
// at import time (node --test imports this file directly).
//
// Runs BEFORE the wasm bundle loads: checks navigator.gpu /
// requestAdapter, and when WebGPU can't be had, renders a per-
// browser+OS "how to turn it on" page instead of a dead canvas.
//
//   detectBrowser(userAgent, isBrave, platform) -> {browser, os, scheme}
//   stepsFor(env, gpuState, host) -> {title, steps}
//   checkWebGPU()                       (browser only)
//   renderHelp(document, gpuState, detail)   (browser only)
//   boot(loadEditor)                    (browser only)
//
// gpuState: 'missing' (navigator.gpu absent — WebGPU is OFF here),
//           'no-adapter' (on, but no usable GPU/driver),
//           'device-failed' (adapter found, device request died —
//            the Rust fallback path calls renderHelp with this),
//           'connect-failed' (not a GPU state: the editor could not
//            open its WebSocket to the server — Rust calls
//            neWebgpuHelp.showConnect so it never reads as WebGPU).
//
// Steps are arrays of parts: {t: "text"} or {c: "copyable"}. Browsers
// refuse to open chrome:// etc. from page JS, so copyable parts render
// as text + a Copy button — "paste into the address bar".

const SCHEMES = {
    brave: 'brave', chrome: 'chrome', edge: 'edge', vivaldi: 'vivaldi',
    opera: 'opera', chromium: 'chrome',
};

export function detectBrowser(ua, isBrave, platform) {
    ua = ua || '';
    platform = platform || '';
    const os =
        /Android/i.test(ua) ? 'android' :
        /iPhone|iPad|iPod/.test(ua) ? 'ios' :
        /Win/.test(platform) ? 'windows' :
        /Mac/.test(platform) ? 'macos' :
        /Linux/.test(platform) ? 'linux' :
        /CrOS/.test(ua) ? 'chromeos' : 'other';
    let browser;
    if (isBrave) browser = 'brave';
    else if (/Edg\//.test(ua)) browser = 'edge';
    else if (/OPR\/|Opera\//.test(ua)) browser = 'opera';
    else if (/Vivaldi\//.test(ua)) browser = 'vivaldi';
    else if (/Chromium\//.test(ua) && !/Chrome\//.test(ua)) browser = 'chromium';
    else if (/Chrome\//.test(ua)) browser = 'chrome';
    else if (/Firefox\//.test(ua)) browser = 'firefox';
    else if (/Safari\//.test(ua) && /Version\//.test(ua)) browser = 'safari';
    else browser = 'unknown';
    return { browser, os, scheme: SCHEMES[browser] || 'chrome' };
}

// ---- step builders (each returns an array of steps; a step is an
// array of {t}|{c} parts) -------------------------------------------

const T = (t) => ({ t });
const C = (c) => ({ c });
const copyLine = (code) => [C(code), T(' — paste into the address bar')];

function chromiumSteps(env, state, host) {
    const s = env.scheme;
    const desktop = [
        [T('Or use the desktop editor: '), C(`editor --connect ${host}`)],
    ];
    if (state === 'device-failed') {
        return {
            title: 'WebGPU device request failed',
            steps: [
                [T('Your browser offered a WebGPU adapter but refused the device.')],
                [T('Check '), C(`${s}://gpu`), T(' — "WebGPU" should read "Hardware accelerated".')],
                [T('Update the GPU driver and relaunch the browser.')],
                [T('If the status says "Blocklisted", try '), C(`${s}://flags/#ignore-gpu-blocklist`), T(' set to Enabled.')],
                ...desktop,
            ],
        };
    }
    if (state === 'no-adapter') {
        const steps = [
            [T('WebGPU is enabled in this browser, but no GPU adapter was offered.')],
            [T('Open '), C(`${s}://settings/system`), T(' and turn on "Use graphics acceleration when available".')],
            [T('Check '), C(`${s}://gpu`), T(' — "WebGPU" should read "Hardware accelerated".')],
            [T('If it says "Blocklisted" or "Disabled", set '), C(`${s}://flags/#ignore-gpu-blocklist`), T(' to Enabled and relaunch.')],
            [T('Update the GPU driver (and on Linux make sure a Vulkan driver is installed).')],
            ...desktop,
        ];
        if (env.os === 'linux') {
            steps.splice(2, 0,
                [T('On Linux also enable '), C(`${s}://flags/#enable-unsafe-webgpu`),
                 T(' and '), C(`${s}://flags/#enable-vulkan`), T(', then relaunch.')]);
        }
        return { title: 'WebGPU found no GPU', steps };
    }
    // state === 'missing'
    if (env.os === 'linux' || env.os === 'android') {
        return {
            title: 'WebGPU is off in this browser',
            steps: [
                [T(`${env.browser === 'android' ? 'Chrome on Android' : 'Chromium browsers on Linux'} ship with WebGPU off — turn it on:`)],
                copyLine(`${s}://flags/#enable-unsafe-webgpu`),
                [T('Set it to Enabled. WebGPU on Linux rides Vulkan, so also enable:')],
                copyLine(`${s}://flags/#enable-vulkan`),
                [T('Relaunch the browser completely (all windows).')],
                [T('Alternative: launch the browser with '), C(`--enable-unsafe-webgpu --enable-features=Vulkan`), T(' on the command line.')],
                [T('Verify on '), C(`${s}://gpu`), T(' — "WebGPU" should read "Hardware accelerated".')],
                ...desktop,
            ],
        };
    }
    // windows / macos / chromeos / ios / other
    return {
        title: 'WebGPU is off in this browser',
        steps: [
            [T('WebGPU has been on by default in Chrome-family browsers since Chrome 113 — an off WebGPU here usually means an old version or a policy.')],
            [T('Update the browser to the latest version and relaunch.')],
            [T('If it stays off, open '), C(`${s}://flags/#enable-unsafe-webgpu`), T(' and set it to Enabled, then relaunch.')],
            [T('Check '), C(`${s}://gpu`), T(' — "WebGPU" should read "Hardware accelerated".')],
            ...desktop,
        ],
    };
}

function firefoxSteps(env, state, host) {
    const desktop = [
        [T('Or use the desktop editor: '), C(`editor --connect ${host}`)],
    ];
    if (state === 'device-failed' || state === 'no-adapter') {
        return {
            title: 'WebGPU found no usable device',
            steps: [
                [T('WebGPU is enabled, but Firefox could not get a device.')],
                [T('Update the GPU driver.')],
                [T('Open '), C('about:config'), T(' — paste into the address bar — and set '), C('gfx.webgpu.ignore-blocklist'), T(' to true, then restart.')],
                ...desktop,
            ],
        };
    }
    return {
        title: 'WebGPU is off in this browser',
        steps: [
            [T('Firefox ships WebGPU on by default on Windows since version 141 and on macOS since 147 — on Linux and Android it is not on in stable releases yet.')],
            [T('Update Firefox first — a newer version may already have it on for your OS.')],
            [T('If it is still off: open '), C('about:config'), T(' — paste into the address bar, accept the warning.')],
            [T('Set '), C('dom.webgpu.enabled'), T(' to true and restart Firefox.')],
            [T('(On Linux this is still unfinished in Firefox — if it runs badly, Chrome with the flags below is the easier route.)')],
            ...desktop,
        ],
    };
}

function safariSteps(env, state, host) {
    const desktop = [
        [T('Or use the desktop editor: '), C(`editor --connect ${host}`)],
    ];
    if (state === 'missing') {
        return {
            title: 'WebGPU is off in this browser',
            steps: [
                [T('WebGPU is on by default in Safari 26 (macOS Tahoe / iOS 26) and newer — the simplest fix is to update Safari / macOS.')],
                [T('On older Safari: Settings → Advanced → turn on "Show features for web developers", then Feature Flags → enable WebGPU.')],
                ...desktop,
            ],
        };
    }
    return {
        title: 'WebGPU found no usable device',
        steps: [
            [T('WebGPU is enabled, but Safari could not get a device — update macOS and the GPU support that comes with it.')],
            ...desktop,
        ],
    };
}

function unknownSteps(env, state, host) {
    const desktop = [
        [T('Or use the desktop editor: '), C(`editor --connect ${host}`)],
    ];
    return {
        title: 'WebGPU is not available in this browser',
        steps: [
            [T('This browser was not recognised. neuro_edit v2 needs WebGPU, which ships in Chrome/Edge ≥ 113, Safari ≥ 26, and Firefox ≥ 141 (Windows) / ≥ 147 (macOS).')],
            [T('On a Chromium-family browser on Linux: open '), C('chrome://flags/#enable-unsafe-webgpu'), T(' and '), C('chrome://flags/#enable-vulkan'), T(' (use your browser\'s scheme — brave://, edge://, vivaldi://, opera://), set both Enabled, relaunch.')],
            [T('Check '), C('chrome://gpu'), T(' for "WebGPU: Hardware accelerated".')],
            ...desktop,
        ],
    };
}

export function stepsFor(env, gpuState, host) {
    if (gpuState === 'connect-failed') return connectSteps(env, host);
    if (env.browser === 'firefox') return firefoxSteps(env, gpuState, host);
    if (env.browser === 'safari') return safariSteps(env, gpuState, host);
    if (env.browser === 'unknown') return unknownSteps(env, gpuState, host);
    return chromiumSteps(env, gpuState, host);
}

// Same for every browser: the socket failing says nothing about WebGPU.
function connectSteps(env, host) {
    return {
        title: "can't connect to the server",
        steps: [
            [T('The editor loaded, but opening its connection to the server failed.')],
            [T('Reload without the cache (Ctrl+Shift+R, Cmd+Shift+R on macOS) — an editor bundle cached from an older version may connect the wrong way.')],
            [T('If you were logged out, log in again on the start page.')],
            [T('Or use the desktop editor: '), C(`editor --connect ${host}`)],
        ],
    };
}

// ---- browser-only part (never touched by node --test) --------------

async function isBraveBrowser() {
    try {
        return !!(navigator.brave && (await navigator.brave.isBrave()));
    } catch (_) {
        return false;
    }
}

export async function checkWebGPU() {
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
        return { state: 'missing' };
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { state: 'no-adapter' };
        let info = '';
        try {
            const i = adapter.info;
            if (i) info = [i.vendor, i.architecture].filter(Boolean).join(' ');
        } catch (_) { /* adapter.info not exposed */ }
        return { state: 'ok', info };
    } catch (e) {
        return { state: 'no-adapter', detail: String(e && e.message ? e.message : e) };
    }
}

function stepEl(doc, parts) {
    const li = doc.createElement('li');
    for (const p of parts) {
        if (p.c !== undefined) {
            const code = doc.createElement('code');
            code.textContent = p.c;
            code.style.cssText =
                'background:#23232b;padding:2px 6px;border-radius:4px;' +
                'color:#9ecbff;user-select:all';
            li.appendChild(code);
            const btn = doc.createElement('button');
            btn.textContent = 'copy';
            btn.style.cssText =
                'margin-left:6px;font:11px monospace;background:#2a2a33;' +
                'color:#bbb;border:1px solid #444;border-radius:3px;cursor:pointer';
            btn.onclick = () => {
                try { navigator.clipboard.writeText(p.c); btn.textContent = 'copied'; }
                catch (_) { btn.textContent = 'select+copy'; }
            };
            li.appendChild(btn);
        } else {
            li.appendChild(doc.createTextNode(p.t));
        }
    }
    return li;
}

/// Render the help page into <body>. `detail` (optional raw error
/// string) goes into a <details> line.
export function renderHelp(doc, env, gpuState, detail) {
    const { title, steps } = stepsFor(env, gpuState, location.host);
    doc.body.innerHTML = '';
    doc.body.style.cssText = 'background:#101014;color:#ccc;font:14px/1.65 monospace';
    const box = doc.createElement('div');
    box.style.cssText = 'max-width:46em;margin:8vh auto;padding:0 2em';
    const h = doc.createElement('h2');
    h.style.cssText = 'color:#eee;font-size:17px';
    h.textContent = `neuro_edit v2 — ${title}`;
    box.appendChild(h);
    const sub = doc.createElement('p');
    sub.style.cssText = 'color:#888';
    sub.textContent = (gpuState === 'connect-failed'
        ? 'WebGPU is fine here — the problem is the connection. '
        : 'The 3D view uses WebGPU storage buffers — the page needs WebGPU on. ') +
        `Detected: ${env.browser} on ${env.os}.`;
    box.appendChild(sub);
    const ol = doc.createElement('ol');
    ol.style.cssText = 'padding-left:1.4em';
    for (const s of steps) {
        const li = stepEl(doc, s);
        li.style.margin = '0.7em 0';
        ol.appendChild(li);
    }
    box.appendChild(ol);
    if (detail) {
        const d = doc.createElement('details');
        d.style.cssText = 'color:#777;margin-top:2em';
        const sum = doc.createElement('summary');
        sum.textContent = 'details';
        d.appendChild(sum);
        const pre = doc.createElement('pre');
        pre.style.cssText = 'white-space:pre-wrap;color:#888';
        pre.textContent = String(detail);
        d.appendChild(pre);
        box.appendChild(d);
    }
    doc.body.appendChild(box);
}

/// Write the loading line into #loading-status, so the logo next to it
/// in #loading stays. Falls back to #loading for an older index.html.
export function setLoadingStatus(doc, text) {
    const el = doc.getElementById('loading-status') ?? doc.getElementById('loading');
    if (el) {
        el.textContent = text;
    }
}

export async function boot(loadEditor) {
    const env = detectBrowser(
        navigator.userAgent, await isBraveBrowser(), navigator.platform);
    // exposed so the wasm fallback path can render the same page when a
    // device request still fails after this check passed
    window.neWebgpuHelp = {
        env,
        show: (detail) => renderHelp(document, env, 'device-failed', detail),
        showConnect: (detail) => renderHelp(document, env, 'connect-failed', detail),
    };
    const r = await checkWebGPU();
    if (r.state !== 'ok') {
        renderHelp(document, env, r.state, r.detail);
        return;
    }
    setLoadingStatus(document,
        `loading neuro_edit v2… WebGPU adapter${r.info ? `: ${r.info}` : ''}`);
    try {
        await loadEditor();
    } finally {
        document.getElementById('loading')?.remove();
    }
}
