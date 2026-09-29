// trajectories.js — per-pair trajectory models beyond straight-line lerp.
//
// The renderer draws each matched pair along a quadratic Bezier through a
// WAYPOINT cloud W: attr(t) = mix(mix(A, W, t), mix(W, B, t), t). With W = the
// straight midpoint this is exactly the old lerp; a better W bends the path
// while the endpoints stay EXACT (t=0 -> A, t=1 -> B) for any W. One waypoint
// cloud is built per (mapping, trajectory-mode) and cached; it reuses the
// standard cloud/packCloud machinery, so position, covariance, and color all
// upgrade at once. Two modes are provided on top of `linear` (no waypoint):
//
//   geodesic — per-pair appearance/shape midpoint, positions stay straight:
//     * covariance: quaternion slerp midpoint + mean scales — a cheap
//       approximation of the Bures-Wasserstein geodesic of the two Gaussians
//       (exact at both endpoints; see knowledge/joint-metric-and-trajectories.md
//       §2.1). Rotationally mismatched splats morph through a rotating
//       ellipsoid instead of a degenerate pancake.
//     * color: the waypoint is reflected so the quadratic passes exactly
//       through the CIELAB-midpoint color at t=1/2 (clamped to gamut) —
//       the path visits the perceptual halfway color instead of the
//       RGB-linear one.
//   coherent — geodesic PLUS CPD-style motion coherence for positions
//     (Myronenko & Song, TPAMI 2010): the per-splat displacement field is
//     kernel-smoothed over spatial neighbors (Gaussian kernel, bandwidth
//     `beta`), and the waypoint is bent toward the coherent field by strength
//     `kappa` while endpoints stay exact — the residual ramps in linearly via
//     the Bezier. Nearby splats follow similar mid-morph paths, attacking the
//     block-choppiness of partition maps and the color-stream interleaving of
//     joint-metric maps.
//   rigid — geodesic appearance PLUS registered-arc positions: given a global
//     similarity transform x' = s*R*x + t from code/prealign.py (opts.xform,
//     loaded via ?xform=), each splat follows the FACTORED path
//     x(t) = R(t) s(t) p + tT + t*r  (r = q - (sRp + T) the non-rigid residual)
//     and the Bezier waypoint is placed so the curve passes through x(1/2)
//     exactly. Near-rigid motion travels an ARC instead of a chord (no
//     mid-morph contraction/shear), residuals ramp in linearly. See
//     knowledge/registration-and-rigidity.md paragraph 3.
//
// Loaded as a plain script before morph.js.

// ---------------------------------------------------------------------------
// sRGB <-> CIELAB (D65). L in [0,100], a/b roughly [-100,100]; distances in
// this space are classic Delta-E 76.
// ---------------------------------------------------------------------------
const LAB_DELTA = 6 / 29;
function _srgbLin(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
function _linSrgb(v) {
  const c = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(v, 0), 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(c * 255)));
}
function _fLab(t) { return t > LAB_DELTA ** 3 ? Math.cbrt(t) : t / (3 * LAB_DELTA * LAB_DELTA) + 4 / 29; }
function _fLabInv(t) { return t > LAB_DELTA ? t * t * t : 3 * LAB_DELTA * LAB_DELTA * (t - 4 / 29); }

function rgbToLab(r, g, b, out) {
  const R = _srgbLin(r), G = _srgbLin(g), B = _srgbLin(b);
  const fx = _fLab((0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047);
  const fy = _fLab( 0.2126729 * R + 0.7151522 * G + 0.0721750 * B);
  const fz = _fLab((0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883);
  out[0] = 116 * fy - 16;
  out[1] = 500 * (fx - fy);
  out[2] = 200 * (fy - fz);
}
function labToRgb(L, a, b, out) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const X = 0.95047 * _fLabInv(fx), Y = _fLabInv(fy), Z = 1.08883 * _fLabInv(fz);
  out[0] = _linSrgb( 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
  out[1] = _linSrgb(-0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z);
  out[2] = _linSrgb( 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
}

// Mean Delta-E 76 between the paired colors of two slot-aligned clouds — the
// recolor cost of a mapping, the appearance analog of meanTravel.
function meanDeltaE(bufA, bufT, N) {
  const uA = new Uint8Array(bufA), uT = new Uint8Array(bufT);
  const la = [0, 0, 0], lb = [0, 0, 0];
  let s = 0;
  for (let i = 0; i < N; i++) {
    const oa = 32 * i + 24, ot = 32 * i + 24;
    rgbToLab(uA[oa], uA[oa + 1], uA[oa + 2], la);
    rgbToLab(uT[ot], uT[ot + 1], uT[ot + 2], lb);
    const d0 = la[0] - lb[0], d1 = la[1] - lb[1], d2 = la[2] - lb[2];
    s += Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
  }
  return s / N;
}

// ---------------------------------------------------------------------------
// CPD-style motion coherence: kernel regression of the displacement field.
// Grid-accelerated Nadaraya-Watson with a Gaussian kernel of bandwidth beta
// (unit-sphere units): each cell stores its members' mean position and mean
// displacement; each splat averages the 3x3x3 neighboring cell means with
// weight count * exp(-||x_i - cellMeanPos||^2 / (2 beta^2)). O(N * 27) per
// iteration regardless of density — the linear-time stand-in for CPD's
// low-rank RKHS solve, keeping the same artistic knob (beta = coherence
// length). Returns the smoothed field v (Float32Array, 3N).
// ---------------------------------------------------------------------------
function smoothDisplacementField(srcPos, tgtPos, N, beta, iters) {
  let d = new Float32Array(3 * N);
  for (let i = 0; i < 3 * N; i++) d[i] = tgtPos[i] - srcPos[i];
  const R = Math.max(4, Math.min(96, Math.round(2 / beta)));
  const cellW = 2 / R;
  const clampI = (v) => (v < 0 ? 0 : (v >= R ? R - 1 : v));
  const ci = (x) => clampI(((x + 1) / cellW) | 0);
  const nC = R * R * R;
  const cellIdx = new Int32Array(N);
  for (let i = 0; i < N; i++)
    cellIdx[i] = (ci(srcPos[3 * i]) * R + ci(srcPos[3 * i + 1])) * R + ci(srcPos[3 * i + 2]);
  const inv2b2 = 1 / (2 * beta * beta);

  for (let it = 0; it < Math.max(1, iters); it++) {
    const cnt = new Float64Array(nC);
    const sp = new Float64Array(3 * nC);   // sum of member positions
    const sd = new Float64Array(3 * nC);   // sum of member displacements
    for (let i = 0; i < N; i++) {
      const c = cellIdx[i];
      cnt[c]++;
      sp[3 * c] += srcPos[3 * i]; sp[3 * c + 1] += srcPos[3 * i + 1]; sp[3 * c + 2] += srcPos[3 * i + 2];
      sd[3 * c] += d[3 * i];      sd[3 * c + 1] += d[3 * i + 1];      sd[3 * c + 2] += d[3 * i + 2];
    }
    const v = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      const x = srcPos[3 * i], y = srcPos[3 * i + 1], z = srcPos[3 * i + 2];
      const cx = ci(x), cy = ci(y), cz = ci(z);
      let wsum = 0, vx = 0, vy = 0, vz = 0;
      for (let ix = Math.max(0, cx - 1); ix <= Math.min(R - 1, cx + 1); ix++)
      for (let iy = Math.max(0, cy - 1); iy <= Math.min(R - 1, cy + 1); iy++)
      for (let iz = Math.max(0, cz - 1); iz <= Math.min(R - 1, cz + 1); iz++) {
        const c = (ix * R + iy) * R + iz;
        const n = cnt[c];
        if (!n) continue;
        const mx = sp[3 * c] / n - x, my = sp[3 * c + 1] / n - y, mz = sp[3 * c + 2] / n - z;
        const w = n * Math.exp(-(mx * mx + my * my + mz * mz) * inv2b2);
        wsum += w;
        vx += w * sd[3 * c] / n; vy += w * sd[3 * c + 1] / n; vz += w * sd[3 * c + 2] / n;
      }
      if (wsum > 1e-12) { v[3 * i] = vx / wsum; v[3 * i + 1] = vy / wsum; v[3 * i + 2] = vz / wsum; }
      else { v[3 * i] = d[3 * i]; v[3 * i + 1] = d[3 * i + 1]; v[3 * i + 2] = d[3 * i + 2]; }
    }
    d = v;
  }
  return d;
}

// Row-major 3x3 rotation matrix of a unit quaternion (w,x,y,z).
function quatToMat(w, x, y, z) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

// ---------------------------------------------------------------------------
// Waypoint cloud builder. cloudA and cloudT are slot-aligned standard clouds
// ({buf, pos, N}); returns a standard cloud usable by packCloud/uploadTexture.
// mode: 'geodesic' (straight positions), 'coherent' (CPD-bent positions), or
// 'rigid' (registered-arc positions; needs opts.xform).
// ---------------------------------------------------------------------------
function buildWaypointCloud(cloudA, cloudT, mode, opts) {
  const t0 = performance.now();
  const N = cloudA.N;
  const fA = new Float32Array(cloudA.buf), uA = new Uint8Array(cloudA.buf);
  const fT = new Float32Array(cloudT.buf), uT = new Uint8Array(cloudT.buf);
  const out = new ArrayBuffer(N * 32);
  const fo = new Float32Array(out), uo = new Uint8Array(out);
  const pos = new Float32Array(3 * N);

  // Position waypoint: straight midpoint, optionally bent toward the
  // kernel-smoothed coherent field. Bezier bend b enters as W = mid + b/2
  // (the quadratic term is 2t(1-t)(W - mid) = t(1-t) * b).
  let bend = null;
  if (mode === 'coherent') {
    const v = smoothDisplacementField(cloudA.pos, cloudT.pos, N, opts.beta, opts.iters);
    bend = v;                              // reuse as bend after the loop below
    const k = opts.kappa;
    for (let i = 0; i < 3 * N; i++) bend[i] = 0.5 * k * (v[i] - (cloudT.pos[i] - cloudA.pos[i]));
  }
  // Rigid mode: full transform RF (for the residual r) and half transform RH
  // (rotation slerp midpoint, sqrt of the scale) for the arc's t=1/2 point.
  // The transform lives in the solver's unit frames; the demo's map-mode
  // frames differ by <1% (pair-row vs kept-row bounding sphere) — a
  // second-order effect on the BEND only (endpoints come from the textures).
  let RF = null, RH = null, sFull = 1, sHalf = 1, TX = 0, TY = 0, TZ = 0;
  if (mode === 'rigid' && opts.xform) {
    let [w, x, y, z] = opts.xform.quat;
    if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
    RF = quatToMat(w, x, y, z);
    const hl = Math.hypot(w + 1, x, y, z);          // normalize(q + identity)
    RH = quatToMat((w + 1) / hl, x / hl, y / hl, z / hl); // = slerp(I, q; 1/2)
    sFull = opts.xform.scale > 0 ? opts.xform.scale : 1;  // guard: sqrt(neg) = NaN waypoints
    sHalf = Math.sqrt(sFull);
    [TX, TY, TZ] = opts.xform.t;
  }
  const la = [0, 0, 0], lb = [0, 0, 0], rgb = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    const o8 = 8 * i, o32 = 32 * i;
    // positions
    if (RF) {
      const px = fA[o8], py = fA[o8 + 1], pz = fA[o8 + 2];
      const qx = fT[o8], qy = fT[o8 + 1], qz = fT[o8 + 2];
      const ax = sFull * (RF[0] * px + RF[1] * py + RF[2] * pz) + TX;   // sRp + T
      const ay = sFull * (RF[3] * px + RF[4] * py + RF[5] * pz) + TY;
      const az = sFull * (RF[6] * px + RF[7] * py + RF[8] * pz) + TZ;
      // x(1/2) = R_half s_half p + T/2 + r/2,  r = q - (sRp + T)
      const hx = sHalf * (RH[0] * px + RH[1] * py + RH[2] * pz) + 0.5 * TX + 0.5 * (qx - ax);
      const hy = sHalf * (RH[3] * px + RH[4] * py + RH[5] * pz) + 0.5 * TY + 0.5 * (qy - ay);
      const hz = sHalf * (RH[6] * px + RH[7] * py + RH[8] * pz) + 0.5 * TZ + 0.5 * (qz - az);
      // Bezier waypoint that makes the curve pass through x(1/2) exactly
      fo[o8] = 2 * hx - 0.5 * (px + qx); pos[3 * i] = fo[o8];
      fo[o8 + 1] = 2 * hy - 0.5 * (py + qy); pos[3 * i + 1] = fo[o8 + 1];
      fo[o8 + 2] = 2 * hz - 0.5 * (pz + qz); pos[3 * i + 2] = fo[o8 + 2];
    } else
    for (let c = 0; c < 3; c++) {
      let m = 0.5 * (fA[o8 + c] + fT[o8 + c]);
      if (bend) m += bend[3 * i + c];
      fo[o8 + c] = m; pos[3 * i + c] = m;
    }
    // scales: arithmetic midpoint (ghost targets have scale 0 -> half-size)
    fo[o8 + 3] = 0.5 * (fA[o8 + 3] + fT[o8 + 3]);
    fo[o8 + 4] = 0.5 * (fA[o8 + 4] + fT[o8 + 4]);
    fo[o8 + 5] = 0.5 * (fA[o8 + 5] + fT[o8 + 5]);
    // color: waypoint REFLECTED so the quadratic passes exactly through the
    // CIELAB-midpoint color at t=1/2 (B(1/2) = (cA+cB)/4 + w/2, so
    // w = 2*labmid - (cA+cB)/2; same trick as the rigid position waypoint),
    // clamped to gamut. Alpha: arithmetic midpoint.
    rgbToLab(uA[o32 + 24], uA[o32 + 25], uA[o32 + 26], la);
    rgbToLab(uT[o32 + 24], uT[o32 + 25], uT[o32 + 26], lb);
    labToRgb(0.5 * (la[0] + lb[0]), 0.5 * (la[1] + lb[1]), 0.5 * (la[2] + lb[2]), rgb);
    for (let c = 0; c < 3; c++) {
      const w = 2 * rgb[c] - 0.5 * (uA[o32 + 24 + c] + uT[o32 + 24 + c]);
      uo[o32 + 24 + c] = Math.max(0, Math.min(255, Math.round(w)));
    }
    uo[o32 + 27] = (uA[o32 + 27] + uT[o32 + 27]) >> 1;
    // rotation: shorter-arc slerp midpoint = normalize(qA + qB'), the exact
    // t=0.5 slerp — no trig needed. Bytes decode as (b - 128)/128, (w,x,y,z).
    let qw = uA[o32 + 28] - 128, qx = uA[o32 + 29] - 128,
        qy = uA[o32 + 30] - 128, qz = uA[o32 + 31] - 128;
    let pw = uT[o32 + 28] - 128, px = uT[o32 + 29] - 128,
        py = uT[o32 + 30] - 128, pz = uT[o32 + 31] - 128;
    if (qw * pw + qx * px + qy * py + qz * pz < 0) { pw = -pw; px = -px; py = -py; pz = -pz; }
    let mw = qw + pw, mx = qx + px, my = qy + py, mz = qz + pz;
    const ml = Math.hypot(mw, mx, my, mz);
    if (ml > 1e-6) { mw /= ml; mx /= ml; my /= ml; mz /= ml; } else { mw = 1; mx = my = mz = 0; }
    uo[o32 + 28] = Math.max(0, Math.min(255, Math.round(mw * 128 + 128)));
    uo[o32 + 29] = Math.max(0, Math.min(255, Math.round(mx * 128 + 128)));
    uo[o32 + 30] = Math.max(0, Math.min(255, Math.round(my * 128 + 128)));
    uo[o32 + 31] = Math.max(0, Math.min(255, Math.round(mz * 128 + 128)));
  }
  return { buf: out, pos, N, ms: performance.now() - t0 };
}
