// node --test editor/web/webgpu-help.test.mjs
// Pure-function tests for the WebGPU help page: real UA strings in,
// {browser, os, steps} out.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { detectBrowser, setLoadingStatus, stepsFor } from './webgpu-help.mjs';

const CHROME_LINUX_UA =
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) ' +
    'Chrome/124.0.0.0 Safari/537.36';
const EDGE_WIN_UA =
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0';
const FF_LINUX_UA =
    'Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0';
const SAFARI_MAC_UA =
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 ' +
    '(KHTML, like Gecko) Version/17.5 Safari/605.1.15';
const VIVALDI_UA =
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) ' +
    'Chrome/124.0.0.0 Safari/537.36 Vivaldi/6.7.3329';

const HOST = '127.0.0.1:3090';
// flatten a steps array to searchable text (copyable parts included)
const flat = (steps) =>
    steps.flat().map((p) => (p.c !== undefined ? p.c : p.t)).join(' ');

test('brave on linux: isBrave flag wins, brave:// scheme + linux flags', () => {
    const env = detectBrowser(CHROME_LINUX_UA, true, 'Linux x86_64');
    assert.equal(env.browser, 'brave');
    assert.equal(env.os, 'linux');
    assert.equal(env.scheme, 'brave');
    const { title, steps } = stepsFor(env, 'missing', HOST);
    assert.match(title, /off in this browser/i);
    const txt = flat(steps);
    assert.ok(txt.includes('brave://flags/#enable-unsafe-webgpu'), txt);
    assert.ok(txt.includes('brave://flags/#enable-vulkan'), txt);
    assert.ok(txt.includes('brave://gpu'), txt);
    assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
});

test('chrome on linux: chrome:// scheme + unsafe-webgpu flag', () => {
    const env = detectBrowser(CHROME_LINUX_UA, false, 'Linux x86_64');
    assert.equal(env.browser, 'chrome');
    assert.equal(env.os, 'linux');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.ok(txt.includes('chrome://flags/#enable-unsafe-webgpu'), txt);
    assert.ok(txt.includes('--enable-unsafe-webgpu'), txt);
});

test('edge on windows: missing → update (on by default), no-adapter → settings/system', () => {
    const env = detectBrowser(EDGE_WIN_UA, false, 'Win32');
    assert.equal(env.browser, 'edge');
    assert.equal(env.os, 'windows');
    const off = flat(stepsFor(env, 'missing', HOST).steps);
    assert.match(off, /update/i);
    assert.ok(off.includes('edge://flags/#enable-unsafe-webgpu'), off);
    const noGpu = flat(stepsFor(env, 'no-adapter', HOST).steps);
    assert.ok(noGpu.includes('edge://settings/system'), noGpu);
    assert.ok(noGpu.includes('edge://flags/#ignore-gpu-blocklist'), noGpu);
});

test('vivaldi on linux gets the vivaldi:// scheme', () => {
    const env = detectBrowser(VIVALDI_UA, false, 'Linux x86_64');
    assert.equal(env.browser, 'vivaldi');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.ok(txt.includes('vivaldi://flags/#enable-unsafe-webgpu'), txt);
});

test('firefox on linux: about:config dom.webgpu.enabled', () => {
    const env = detectBrowser(FF_LINUX_UA, false, 'Linux x86_64');
    assert.equal(env.browser, 'firefox');
    assert.equal(env.os, 'linux');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.ok(txt.includes('about:config'), txt);
    assert.ok(txt.includes('dom.webgpu.enabled'), txt);
    assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
});

test('safari on macos: update / feature flags', () => {
    const env = detectBrowser(SAFARI_MAC_UA, false, 'MacIntel');
    assert.equal(env.browser, 'safari');
    assert.equal(env.os, 'macos');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.match(txt, /Safari 26|Feature Flags/i);
    assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
});

test('unknown browser: generic chromium steps + supported list', () => {
    const env = detectBrowser('MyBrowser/1.0', false, 'Linux x86_64');
    assert.equal(env.browser, 'unknown');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.ok(txt.includes('chrome://flags/#enable-unsafe-webgpu'), txt);
    assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
});

test('device-failed (rust fallback path): gpu page + driver', () => {
    const env = detectBrowser(CHROME_LINUX_UA, false, 'Linux x86_64');
    const { title, steps } = stepsFor(env, 'device-failed', HOST);
    assert.match(title, /device request failed/i);
    const txt = flat(steps);
    assert.ok(txt.includes('chrome://gpu'), txt);
    assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
});

test('connect-failed: a server page with no WebGPU steps, in every browser', () => {
    for (const [ua, brave] of [[CHROME_LINUX_UA, true], [FF_LINUX_UA, false], [SAFARI_MAC_UA, false]]) {
        const env = detectBrowser(ua, brave, 'Linux x86_64');
        const { title, steps } = stepsFor(env, 'connect-failed', HOST);
        assert.match(title, /connect to the server/i, env.browser);
        const txt = flat(steps);
        assert.ok(!/webgpu|:\/\/gpu|flags|driver/i.test(txt), txt);
        assert.ok(txt.includes('Ctrl+Shift+R'), txt);
        assert.ok(txt.includes(`editor --connect ${HOST}`), txt);
    }
});

test('android chromium: linux-style flag steps', () => {
    const ua =
        'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) ' +
        'Chrome/124.0.0.0 Mobile Safari/537.36';
    const env = detectBrowser(ua, false, 'Linux armv81');
    assert.equal(env.browser, 'chrome');
    assert.equal(env.os, 'android');
    const txt = flat(stepsFor(env, 'missing', HOST).steps);
    assert.ok(txt.includes('chrome://flags/#enable-unsafe-webgpu'), txt);
});

// minimal stand-in for `document`: elements by id, textContent recorded
const fakeDoc = (ids) => {
    const els = Object.fromEntries(ids.map((id) => [id, { id, textContent: `<${id}>` }]));
    return { els, getElementById: (id) => els[id] ?? null };
};

test('loading status goes to #loading-status, the logo in #loading stays', () => {
    const doc = fakeDoc(['loading', 'loading-status']);
    setLoadingStatus(doc, 'loading neuro_edit v2… WebGPU adapter: nvidia');
    assert.equal(doc.els['loading-status'].textContent, 'loading neuro_edit v2… WebGPU adapter: nvidia');
    assert.equal(doc.els.loading.textContent, '<loading>');
});

test('loading status falls back to #loading without a status span', () => {
    const doc = fakeDoc(['loading']);
    setLoadingStatus(doc, 'x');
    assert.equal(doc.els.loading.textContent, 'x');
});

test('index.html: #loading holds the logo and the #loading-status span', () => {
    const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
    const loading = html.match(/<div id="loading">([\s\S]*?)<\/div>/);
    assert.ok(loading, 'no #loading div');
    assert.match(loading[1], /<img src="loading-logo.svg"/);
    assert.match(loading[1], /<span id="loading-status">/);
    assert.match(html, /<link rel="icon" href="favicon.svg"/);
});
