// app.js: the live demo of Fast Splat Morphing.
//
// Pick object A on the left and object B on the right. The page loads both,
// computes the FSM assignment between them live (fsm-worker.js, with the
// paper's own cloud.js, pairings.js and trajectories.js), and shows A, the
// morph, and B side by side, the morph swinging back and forth in t.
//
// Rendering is renderer.js, the viewer the paper's figures were drawn with
// (adapted from antimatter15/splat, MIT): one WebGL2 canvas, one viewport per
// view, each view depth-sorted by its own worker. The views share one camera.
'use strict';

const SIGMA_K = 4;          // splat footprint; must match fsm-worker.js
const KERNEL2D = 0.3;       // screen-space dilation in px^2, as the paper's viewer
const PERIOD = 6;           // seconds for one full A -> B -> A swing
const SWAY = 0.45;          // radians the camera sways about its azimuth
const FIT = 1.08;           // margin around the unit sphere every cloud lives in

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');

// ---------------------------------------------------------------------------
// WebGL: one program, four texture units, one vertex array per view.
// ---------------------------------------------------------------------------
const canvas = $('gl');
const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true });
if (!gl) {
  $('noteMText').textContent = 'This demo needs WebGL2, which this browser does not provide.';
  throw new Error('WebGL2 not available');
}
function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vertSrc));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragSrc));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
gl.clearColor(0, 0, 0, 0);
gl.enable(gl.SCISSOR_TEST);

const U = {};
for (const n of ['projection', 'view', 'focal', 'viewport', 'u_t', 'u_traj', 'u_kernel', 'u_fit',
                 'u_textureA', 'u_textureB', 'u_textureW'])
  U[n] = gl.getUniformLocation(program, n);
gl.uniform3f(U.u_fit, 1, 0, 0);
gl.uniform1f(U.u_kernel, KERNEL2D * SIGMA_K);

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(program, 'position');
const aIdx = gl.getAttribLocation(program, 'index');

// Texture units: 0 = A, 1 = B in its own order, 2 = B arranged by the
// assignment (slot i holds A's partner), 3 = the path waypoints.
const TEX = { A: 0, B: 1, T: 2, W: 3 };
const texObj = {};
function upload(unit, t) {
  if (!texObj[unit]) texObj[unit] = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texObj[unit]);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, t.w, t.h, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, t.data);
}

const SORT_SRC = URL.createObjectURL(new Blob(['(', createWorker.toString(), ')(self)'],
  { type: 'application/javascript' }));

// A view: where it sits, which textures it reads, which t it shows.
function makeView(el, texA, texB, traj) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
  gl.enableVertexAttribArray(aIdx);
  gl.vertexAttribIPointer(aIdx, 1, gl.INT, false, 0, 0);
  gl.vertexAttribDivisor(aIdx, 1);
  gl.bindVertexArray(null);
  return { el, texA, texB, traj, vao, idxBuf, count: 0, sorter: null, ready: false };
}
const views = {
  A: makeView($('vA'), TEX.A, TEX.A, 0),
  M: makeView($('vM'), TEX.A, TEX.T, 1),
  B: makeView($('vB'), TEX.B, TEX.B, 0),
};

// A fresh sort worker per view and per pair, so an order computed for the
// previous clouds can never be drawn with the new ones.
function resetSorter(v, posA, posB, posW) {
  if (v.sorter) v.sorter.terminate();
  v.count = 0; v.ready = false;
  const w = new Worker(SORT_SRC);
  w.onmessage = (e) => {
    if (v.sorter !== w) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, v.idxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, e.data.order, gl.DYNAMIC_DRAW);
    v.count = e.data.count;
  };
  w.postMessage({ init: true, posA: posA.buffer, count: posA.length / 3 }, [posA.buffer]);
  const msg = { setB: posB.buffer };
  const tr = [posB.buffer];
  if (posW) { msg.setW = posW.buffer; tr.push(posW.buffer); }
  w.postMessage(msg, tr);
  v.sorter = w;
  v.ready = true;
}

// ---------------------------------------------------------------------------
// Camera: one orbit shared by the three views, swaying gently about the view
// the paper's figures use; a drag takes over.
// ---------------------------------------------------------------------------
let az0 = 3.14, el = -0.25, az = az0, sway = true, swayClock = 0;
const WORLD_UP = [0, 1, 0];
function viewMatrix(dist) {
  const ce = Math.cos(el), se = Math.sin(el);
  const eye = [ce * Math.sin(az) * dist, se * dist, ce * Math.cos(az) * dist];
  const fwd = norm3(sub3([0, 0, 0], eye));
  const right = norm3(cross3(WORLD_UP, fwd));
  const up = cross3(fwd, right);
  return invert4([right[0], right[1], right[2], 0, up[0], up[1], up[2], 0,
                  fwd[0], fwd[1], fwd[2], 0, eye[0], eye[1], eye[2], 1]);
}

const stage = $('stage');
let dragging = false, lastX = 0, lastY = 0;
stage.addEventListener('pointerdown', (e) => {
  dragging = true; sway = false; lastX = e.clientX; lastY = e.clientY;
  stage.classList.add('dragging'); stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  az += (e.clientX - lastX) * 0.006;
  el = Math.max(-1.45, Math.min(1.45, el + (e.clientY - lastY) * 0.006));
  lastX = e.clientX; lastY = e.clientY;
});
const endDrag = () => { dragging = false; stage.classList.remove('dragging'); };
stage.addEventListener('pointerup', endDrag);
stage.addEventListener('pointercancel', endDrag);
stage.addEventListener('dblclick', () => { az0 = 3.14; el = -0.25; sway = true; swayClock = 0; });
// The project page tells an embedded demo when it is promoted to full screen;
// only then does the canvas take vertical swipes too.
addEventListener('message', (e) => {
  if (e.data && typeof e.data.fullscreen === 'boolean')
    document.body.classList.toggle('fullscreen', e.data.fullscreen);
});

// ---------------------------------------------------------------------------
// t: a sine swing from A to B and back, or wherever the slider puts it.
// ---------------------------------------------------------------------------
let playing = true, clock = 0, t = 0;
const slider = $('t'), tval = $('tval'), playBtn = $('play');
function showT() {
  tval.textContent = `t = ${t.toFixed(2)}`;
  slider.value = String(Math.round(t * 1000));
}
function setPlaying(on) { playing = on; playBtn.textContent = on ? 'Pause' : 'Play'; }
slider.addEventListener('input', () => { setPlaying(false); t = Number(slider.value) / 1000; showT(); });
playBtn.addEventListener('click', () => {
  if (!playing) clock = Math.acos(1 - 2 * t) / (2 * Math.PI) * PERIOD;   // resume from here
  setPlaying(!playing);
});

// ---------------------------------------------------------------------------
// Objects, loading, and the live computation.
// ---------------------------------------------------------------------------
const compute = new Worker('fsm-worker.js');
let manifest = null;
const byId = {};
const loads = {};            // id -> Promise that resolves once the worker holds it
let sel = { A: null, B: null };
let job = 0, busy = false, queued = false, shown = { a: null, b: null };

function load(id, onProgress) {
  if (loads[id]) return loads[id];
  const o = byId[id];
  loads[id] = (async () => {
    const res = await fetch('assets/' + o.file);
    if (!res.ok) throw new Error(`${res.status} loading ${o.file}`);
    const total = Number(res.headers.get('content-length')) || 0;
    let buf;
    if (res.body && total) {
      const out = new Uint8Array(total), rd = res.body.getReader();
      let got = 0;
      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        out.set(value, got); got += value.length;
        onProgress && onProgress(got / total);
      }
      buf = out.buffer;
    } else buf = await res.arrayBuffer();
    compute.postMessage({ type: 'object', id, buf, yaw: o.yaw_deg || 0, pitch: o.pitch_deg || 0 }, [buf]);
  })();
  return loads[id];
}

function setNote(side, text, frac) {
  $('note' + side).hidden = text === null;
  if (text !== null) {
    $('note' + side + 'Text').textContent = text;
    const bar = $('prog' + side);
    if (bar) bar.style.width = `${Math.round(100 * (frac || 0))}%`;
  }
}

async function request() {
  if (busy) { queued = true; return; }
  const a = sel.A, b = sel.B;
  if (a === shown.a && b === shown.b) return;
  busy = true; queued = false;
  const my = ++job;
  const live = $('live');
  live.classList.add('busy');
  $('badgeText').textContent = 'COMPUTING';
  $('what').textContent = 'building the assignment in your browser…';
  views.M.ready = false;
  setNote('M', 'computing the assignment live…');
  for (const s of ['A', 'B']) {
    const id = sel[s];
    if (!loads[id]) setNote(s, `loading ${byId[id].label.toLowerCase()}…`, 0);
  }
  try {
    await Promise.all([
      load(a, (f) => setNote('A', `loading ${byId[a].label.toLowerCase()}…`, f)),
      load(b, (f) => setNote('B', `loading ${byId[b].label.toLowerCase()}…`, f)),
    ]);
    compute.postMessage({ type: 'pair', job: my, a, b });
  } catch (err) {
    busy = false;
    setNote('M', 'Could not load the objects: ' + err.message);
  }
}

compute.onmessage = (e) => {
  const d = e.data;
  if (d.job !== job) return;                 // a newer pair was asked for
  if (d.type === 'error') {
    busy = false; $('live').classList.remove('busy');
    setNote('M', 'Something went wrong: ' + d.message);
    return;
  }
  if (d.type === 'clouds') {
    upload(TEX.A, d.texA); upload(TEX.B, d.texB);
    resetSorter(views.A, d.posA.slice(), d.posA);
    resetSorter(views.B, d.posB.slice(), d.posB);
    $('countA').textContent = fmt(d.nA); $('countB').textContent = fmt(d.nB);
    $('nameA').textContent = byId[d.a].label; $('nameB').textContent = byId[d.b].label;
    setNote('A', null); setNote('B', null);
    return;
  }
  if (d.type === 'morph') {
    upload(TEX.T, d.texT); upload(TEX.W, d.texW);
    resetSorter(views.M, d.posA, d.posT, d.posW);
    shown = { a: d.a, b: d.b };
    setNote('M', null);
    const live = $('live');
    live.classList.remove('busy');
    $('badgeText').textContent = 'LIVE';
    $('ms').innerHTML = `${Math.round(d.ms)}<small>&nbsp;ms</small>`;
    $('what').textContent = `FSM assignment of ${fmt(d.N)} pairs, computed just now in your browser`;
    live.classList.remove('flash'); void live.offsetWidth; live.classList.add('flash');
    setTimeout(() => live.classList.remove('flash'), 60);
    if (playing) clock = 0;                  // start the swing at A
    busy = false;
    if (queued) request();
  }
};

function buildPicker(side) {
  const box = $('pick' + side);
  for (const o of manifest.objects) {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'obj'; btn.dataset.id = o.id;
    btn.title = o.label;
    const img = document.createElement('img');
    img.src = `assets/thumbs/${o.id}.jpg`; img.alt = '';
    img.onerror = () => img.remove();
    const lab = document.createElement('span'); lab.textContent = o.label;
    btn.append(img, lab);
    btn.addEventListener('click', () => { sel[side] = o.id; refreshPickers(); request(); });
    box.appendChild(btn);
  }
}
function refreshPickers() {
  for (const side of ['A', 'B']) {
    const other = side === 'A' ? sel.B : sel.A;
    for (const btn of $('pick' + side).children) {
      btn.setAttribute('aria-pressed', String(btn.dataset.id === sel[side]));
      btn.disabled = btn.dataset.id === other;
    }
  }
}

// ---------------------------------------------------------------------------
// Frame loop: advance t and the sway, then draw each view in its own viewport.
// ---------------------------------------------------------------------------
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (playing) { clock += dt; t = 0.5 - 0.5 * Math.cos(2 * Math.PI * clock / PERIOD); showT(); }
  if (sway) { swayClock += dt; az = az0 + SWAY * Math.sin(swayClock * 0.35); }

  const dpr = Math.min(devicePixelRatio || 1, 2);
  const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  gl.viewport(0, 0, cw, ch); gl.scissor(0, 0, cw, ch);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const cr = canvas.getBoundingClientRect();
  for (const key of ['A', 'M', 'B']) {
    const v = views[key];
    const r = v.el.getBoundingClientRect();
    const x = Math.round((r.left - cr.left) * dpr), w = Math.round(r.width * dpr);
    const h = Math.round(r.height * dpr), y = Math.round((cr.bottom - r.bottom) * dpr);
    if (w < 2 || h < 2) continue;
    const dist = FIT * Math.max(2, 2 * h / w);
    const view = viewMatrix(dist);
    const proj = getProjectionMatrix(h, h, w, h);
    const tv = key === 'M' ? t : 0;
    if (v.sorter) v.sorter.postMessage({ view: multiply4(proj, view), t: tv });
    if (!v.ready || !v.count) continue;
    gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
    gl.uniformMatrix4fv(U.projection, false, proj);
    gl.uniformMatrix4fv(U.view, false, view);
    gl.uniform2fv(U.focal, [h, h]);
    gl.uniform2fv(U.viewport, [w, h]);
    gl.uniform1f(U.u_t, tv);
    gl.uniform1f(U.u_traj, v.traj);
    gl.uniform1i(U.u_textureA, v.texA);
    gl.uniform1i(U.u_textureB, v.texB);
    gl.uniform1i(U.u_textureW, v.traj ? TEX.W : v.texA);
    gl.bindVertexArray(v.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, v.count);
  }
  gl.bindVertexArray(null);
  requestAnimationFrame(frame);
}

(async () => {
  const res = await fetch('assets/objects.json');
  manifest = await res.json();
  for (const o of manifest.objects) byId[o.id] = o;
  compute.postMessage({ type: 'config', omega: manifest.omega, seed: manifest.seed });
  az0 = az = manifest.camera.az; el = manifest.camera.el;
  const q = new URLSearchParams(location.search);
  sel.A = byId[q.get('a')] ? q.get('a') : manifest.default[0];
  sel.B = byId[q.get('b')] && q.get('b') !== sel.A ? q.get('b') : manifest.default[1];
  if (sel.B === sel.A) sel.B = manifest.objects.find((o) => o.id !== sel.A).id;
  if (q.get('play') === '0') { setPlaying(false); t = Number(q.get('t')) || 0; }
  if (q.get('sway') === '0') sway = false;
  if (q.has('az')) az0 = az = Number(q.get('az'));
  if (q.has('el')) el = Number(q.get('el'));
  buildPicker('A'); buildPicker('B'); refreshPickers();
  showT();
  request();
  requestAnimationFrame(frame);
})();
