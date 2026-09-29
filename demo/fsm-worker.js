// fsm-worker.js: the live computation of the demo, off the page's main thread.
//
// It runs the paper's pipeline unchanged. cloud.js prepares both splat clouds
// at full density (outlier cull, unit-sphere normalization, zero-alpha clone
// padding of the smaller side), pairings.js builds the Fast Splat Morphing
// assignment, and trajectories.js bends each pair's path into a
// motion-coherent one. Those three files are byte-identical to the ones the
// paper's timings were measured with.
//
// The reported time covers exactly what the paper times: packing both clouds
// into the assignment's input (positions plus the CIELAB lightness at weight
// omega) and the co-bisection itself. Preparing the clouds and building the
// paths happen outside that window and are reported apart.
'use strict';

// packSplatTexture (renderer.js) reads this global; it must match the page.
self.SIGMA_K = 4;
importScripts('trajectories.js', 'cloud.js', 'pairings.js', 'renderer.js');

let OMEGA = 0.25, SEED = 7;
const objects = new Map();           // id -> { buf, n }, already turned to its yaw

function mulberry32(s) {
  let a = (s || 1) >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

// The rotation an object is shown in: a turn by `yaw` about the up axis (y),
// with the sign convention of the paper's import tool, then a tilt by `pitch`
// about x, the horizontal axis of the demo's camera. Row-major 3x3.
function turnMatrix(yawDeg, pitchDeg) {
  const a = yawDeg * Math.PI / 180, b = pitchDeg * Math.PI / 180;
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const Y = [ca, 0, -sa, 0, 1, 0, sa, 0, ca];
  const X = [1, 0, 0, 0, cb, -sb, 0, sb, cb];
  const R = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    for (let k = 0; k < 3; k++) R[3 * i + j] += X[3 * i + k] * Y[3 * k + j];
  return R;
}
// The unit quaternion (w,x,y,z) of a rotation matrix, in the convention
// renderer.js decodes, so that composing it on the left turns every Gaussian by
// exactly the matrix that moves its center.
function matToQuat(R) {
  const tr = R[0] + R[4] + R[8];
  let w, x, y, z;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]);
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]);
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]);
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  return [w, x, y, z];
}
// Turn a .splat buffer in place. Each Gaussian's orientation is turned by the
// SAME rotation as its center, so every ellipsoid keeps its place on the
// surface. Composing a unit quaternion preserves each stored quaternion's norm,
// so the covariance scale is unchanged.
function rotateSplats(buf, yawDeg, pitchDeg) {
  if (!yawDeg && !pitchDeg) return;
  const R = turnMatrix(yawDeg || 0, pitchDeg || 0);
  const [qw, qx, qy, qz] = matToQuat(R);
  const f = new Float32Array(buf), u = new Uint8Array(buf);
  const n = (buf.byteLength / 32) | 0;
  const q8 = (v) => Math.max(0, Math.min(255, Math.round(v * 128 + 128)));
  for (let i = 0; i < n; i++) {
    const o = 8 * i, x = f[o], y = f[o + 1], z = f[o + 2];
    f[o]     = R[0] * x + R[1] * y + R[2] * z;
    f[o + 1] = R[3] * x + R[4] * y + R[5] * z;
    f[o + 2] = R[6] * x + R[7] * y + R[8] * z;
    const b = 32 * i + 28;
    const w = (u[b] - 128) / 128, X = (u[b + 1] - 128) / 128,
          Y = (u[b + 2] - 128) / 128, Z = (u[b + 3] - 128) / 128;
    u[b]     = q8(qw * w - qx * X - qy * Y - qz * Z);
    u[b + 1] = q8(qw * X + qx * w + qy * Z - qz * Y);
    u[b + 2] = q8(qw * Y - qx * Z + qy * w + qz * X);
    u[b + 3] = q8(qw * Z + qx * Y - qy * X + qz * w);
  }
}

function packed(cloud) {
  const p = packSplatTexture(cloud.buf, cloud.N);
  return { data: p.texdata, w: p.texwidth, h: p.texheight };
}

function run(job, a, b) {
  const A = objects.get(a), B = objects.get(b);
  if (!A || !B) throw new Error('object not loaded: ' + (A ? b : a));
  // A fresh, seeded generator per pair: the clone padding draws its hosts from
  // it, so the same pair always gets the same clouds (seed 7, as the figures).
  CLOUD_RAND = mulberry32(SEED);

  // Full density: every splat that survives the cull is kept, and the smaller
  // side is padded with zero-alpha clones so the assignment is a bijection.
  const pA = prepareCloud(A, A.n), pB = prepareCloud(B, B.n);
  const nA = pA.kept.length, nB = pB.kept.length, N = Math.max(nA, nB);
  const cloudA = buildBuffer(A, pA.kept, pA.bs, sizeBoost(nA, A.n), N - nA);
  const cloudB = buildBuffer(B, pB.kept, pB.bs, sizeBoost(nB, B.n), N - nB);

  const texA = packed(cloudA), texB = packed(cloudB);
  const posA = cloudA.pos.slice(), posB = cloudB.pos.slice();
  self.postMessage({ type: 'clouds', job, a, b, N, nA, nB, texA, texB, posA, posB },
    [texA.data.buffer, texB.data.buffer, posA.buffer, posB.buffer]);

  // THE TIMED STEP, exactly as in the paper: pack both clouds (with the
  // lightness channel at omega) and build the co-bisection.
  const t0 = performance.now();
  const map = hierPairing(packCloud(cloudA, OMEGA, N), packCloud(cloudB, OMEGA, N), N);
  const ms = performance.now() - t0;

  const tgt = buildTargetFromMap(cloudA, cloudB, map);
  const wp = buildWaypointCloud(cloudA, tgt, 'coherent', { beta: 0.12, kappa: 0.8, iters: 2 });
  const texT = packed(tgt), texW = packed(wp);
  const posT = tgt.pos, posW = wp.pos, posA2 = cloudA.pos;
  self.postMessage({ type: 'morph', job, a, b, N, ms, pathMs: wp.ms, texT, texW, posA: posA2, posT, posW },
    [texT.data.buffer, texW.data.buffer, posA2.buffer, posT.buffer, posW.buffer]);
}

self.onmessage = (e) => {
  const d = e.data;
  try {
    if (d.type === 'config') { OMEGA = d.omega; SEED = d.seed; return; }
    if (d.type === 'object') {
      rotateSplats(d.buf, d.yaw, d.pitch);
      objects.set(d.id, { buf: d.buf, n: (d.buf.byteLength / 32) | 0 });
      return;
    }
    if (d.type === 'pair') run(d.job, d.a, d.b);
  } catch (err) {
    self.postMessage({ type: 'error', job: d.job, message: String(err && err.message || err) });
  }
};
