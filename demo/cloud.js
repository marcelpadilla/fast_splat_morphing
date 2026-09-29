// cloud.js — splat cloud preparation: sample, cull, normalize, build, pad.
//
// SINGLE SOURCE OF TRUTH, loaded by BOTH consumers:
//   * the browser demo  — as a plain <script> before morph.js (see index.html)
//   * code/bench_hier.mjs — via vm.runInThisContext, the same way it already
//     loads pairings.js and trajectories.js
//
// This file exists because the bench used to hand-copy this pipeline out of
// morph.js under a "if morph.js changes, change this too" warning. That mirror
// was the single biggest correctness hazard in the repo: any drift silently
// changed the travel/deltaE numbers the paper reports, because the bench is
// what produces them. Now there is one copy and no way to drift.
//
// Everything here is pure computation on ArrayBuffers -- no DOM, no fetch, no
// URL params -- so it runs identically in a browser and in Node. The three
// things that DO differ between the two hosts are lifted into the mutable
// config below; each host assigns them after loading this file.
//
// A "cloud" is the shape the renderer, the sort worker and the assignment
// builders all consume:
//     { buf: ArrayBuffer (N * ROW bytes), pos: Float32Array(3N), N, rows }
// `pos` is the stride-3 normalized position array, `rows` the source FILE row
// each slot came from.

// --- host-configurable knobs (defaults are the bench/headless behaviour) ----
var ROW = 32;                 // bytes per splat in the .splat format
// Random source. Called dynamically so a host that seeds Math.random (the
// bench does, for reproducibility) is honoured without reassigning anything.
// morph.js overrides this with its own ?seed=-aware generator.
var CLOUD_RAND = function () { return Math.random(); };
// Outlier culling. morph.js exposes ?cull=0 to disable it for debugging.
var CLOUD_CULL = true;
// Upper bound on the subsample size-boost. morph.js exposes ?boost=.
var BOOST_CAP = 2.5;

// --- sampling ---------------------------------------------------------------
function sampleIndices(want, total) {
  const idx = new Uint32Array(want);
  if (want >= total) {
    for (let i = 0; i < total; i++) idx[i] = i;
    for (let i = total; i < want; i++) idx[i] = (CLOUD_RAND() * total) | 0; // pad with replacement
  } else {
    const pool = new Uint32Array(total);
    for (let i = 0; i < total; i++) pool[i] = i;
    for (let i = 0; i < want; i++) {
      const j = i + ((CLOUD_RAND() * (total - i)) | 0);
      const t = pool[i]; pool[i] = pool[j]; pool[j] = t;
      idx[i] = pool[i];
    }
  }
  return idx;
}

function median(arr) { const s = arr.slice().sort(); return s[s.length >> 1]; }

// Robust center / radius / typical splat size from a candidate index set.
// Median center + 90th-percentile radius ignore stray floaters; the median of
// each splat's largest axis gives a size scale for culling giant blob splats.
function robustStats(f, idx) {
  const M = idx.length;
  const xs = new Float32Array(M), ys = new Float32Array(M), zs = new Float32Array(M), ms = new Float32Array(M);
  for (let i = 0; i < M; i++) {
    const o = 8 * idx[i];
    xs[i] = f[o]; ys[i] = f[o+1]; zs[i] = f[o+2];
    ms[i] = Math.max(f[o+3], f[o+4], f[o+5]);
  }
  const cx = median(xs), cy = median(ys), cz = median(zs);
  const dist = new Float32Array(M);
  for (let i = 0; i < M; i++) dist[i] = Math.hypot(xs[i]-cx, ys[i]-cy, zs[i]-cz);
  const ds = dist.slice().sort();
  const radius = Math.max(ds[Math.floor(M * 0.9)], 1e-4);
  const medScale = Math.max(median(ms), 1e-6);
  return { cx, cy, cz, radius, medScale };
}

// Raw captures carry two kinds of junk that dominate a naive render: far-away
// background/floater splats and a handful of enormous ground/sky blob splats.
// Drop both (position beyond 1.8x radius, largest axis beyond 6x the median
// splat size) so the actual object survives the normalization.
function passesCuts(f, i, st) {
  const o = 8 * i;
  const d = Math.hypot(f[o]-st.cx, f[o+1]-st.cy, f[o+2]-st.cz);
  const s = Math.max(f[o+3], f[o+4], f[o+5]);
  return d <= st.radius * 1.8 && s <= st.medScale * 6;
}
function keepIndices(f, idx, st) {
  const kept = [];
  for (let i = 0; i < idx.length; i++) if (passesCuts(f, idx[i], st)) kept.push(idx[i]);
  return Uint32Array.from(kept);
}

// Exact bounding sphere of the kept splats: bbox center + max distance. Used
// for normalization so the WHOLE object (post-cull) fits in the unit sphere at
// the origin — the camera's default distance then always frames all of it.
function boundingSphere(f, idx) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < idx.length; i++) {
    const o = 8 * idx[i];
    const x = f[o], y = f[o+1], z = f[o+2];
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }
  const cx = (xmin+xmax)/2, cy = (ymin+ymax)/2, cz = (zmin+zmax)/2;
  let r2 = 0;
  for (let i = 0; i < idx.length; i++) {
    const o = 8 * idx[i];
    const dx = f[o]-cx, dy = f[o+1]-cy, dz = f[o+2]-cz;
    const d2 = dx*dx + dy*dy + dz*dz;
    if (d2 > r2) r2 = d2;
  }
  return { cx, cy, cz, radius: Math.max(Math.sqrt(r2), 1e-4) };
}

function prepareCloud(src, want) {
  const f = new Float32Array(src.buf);
  // Generous candidate pool (2x want so enough survive culling to reach N), but
  // capped: robust stats + culling scan the whole pool, and the paired cloud may
  // cap N far below `want` anyway, so a huge pool is wasted load-time work.
  const poolSize = Math.min(src.n, Math.max(want * 2, 50000), 1200000);
  const cand = sampleIndices(poolSize, src.n);
  const st = robustStats(f, cand);          // robust stats drive the culling…
  const kept = CLOUD_CULL ? keepIndices(f, cand, st) : cand;
  // Callers take a PREFIX of `kept` to reach their target count, so `kept` has
  // to be in random order for that prefix to be a uniform sample of the object.
  // It usually is, because sampleIndices shuffled the pool. It is NOT when the
  // pool reached the whole file: that branch returns the identity, and the
  // prefix then becomes the first rows of the .splat file, which is training
  // order and not spatially uniform. Measured on shell->plant (2026-08-04): at
  // that LOD the destination field moved 10.7% of the object against 1.0-1.4%
  // everywhere else, and two different seeds agreed EXACTLY, which is the tell,
  // since a random subsample cannot be seed-independent.
  //
  // Only this branch is reshuffled. The full-density runs every published
  // number uses take ALL of `kept`, where a prefix is the whole array and the
  // order it arrives in is the storage order the assignment has always seen;
  // shuffling there would perturb every table for no gain.
  if (poolSize >= src.n && kept.length > want) shuffleInPlace(kept);
  const bs = boundingSphere(f, kept);       // …the exact sphere drives centering
  return { kept, bs };
}

// Fisher-Yates through the CLOUD_RAND hook, so a seeded host stays reproducible.
function shuffleInPlace(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = (CLOUD_RAND() * (i + 1)) | 0;
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}

// Center on the bounding-sphere center and scale so its radius is 1: both
// clouds land exactly inside the unit sphere at the origin, so one fixed
// camera frames either endpoint (and everything in between) in full.
//
// `boost` scales every splat's size to compensate for subsampling. A 3DGS
// surface is opaque only because many overlapping semi-transparent splats
// accumulate; render a fraction phi of them and each surface point is covered
// by phi-times fewer splats, so it looks translucent and holey. Growing each
// splat's linear size by 1/sqrt(phi) makes it cover the area of the ones we
// dropped, restoring both coverage and accumulated opacity — and boost -> 1 as
// n -> the full cloud, so quality converges to the exact render.
// `clones` is either a COUNT (hosts drawn at random — the pre-match padding)
// or a Uint32Array of host slots, one per clone (the nearest-source padding
// picks them; see nearestHosts). Either way a clone is an exact zero-alpha copy
// of a host splat, so this cloud's own endpoint render is unchanged.
function buildBuffer(src, idx, bs, boost, clones = 0) {
  const f = new Float32Array(src.buf), u = new Uint8Array(src.buf);
  const hosts = typeof clones === 'number' ? null : clones;
  const nClone = hosts ? hosts.length : clones;
  // CAP bounds a single splat's extent (of the unit radius) as a last guard
  // against giant blob splats that survive culling.
  const R = idx.length, N = R + nClone, inv = boost / bs.radius, CAP = 0.25;
  const out = new ArrayBuffer(N * ROW);
  const fo = new Float32Array(out), uo = new Uint8Array(out);
  const pos = new Float32Array(N * 3);
  const rows = new Uint32Array(N);         // source FILE row per slot
  const posInv = 1 / bs.radius;
  for (let i = 0; i < R; i++) {
    const si = 8 * idx[i], di = 8 * i;
    const x = (f[si]-bs.cx)*posInv, y = (f[si+1]-bs.cy)*posInv, z = (f[si+2]-bs.cz)*posInv;
    fo[di] = x; fo[di+1] = y; fo[di+2] = z;
    pos[3*i] = x; pos[3*i+1] = y; pos[3*i+2] = z;
    rows[i] = idx[i];
    fo[di+3] = Math.min(f[si+3]*inv, CAP);   // inv already folds in the size boost
    fo[di+4] = Math.min(f[si+4]*inv, CAP);
    fo[di+5] = Math.min(f[si+5]*inv, CAP);
    const sb = 32*idx[i], db = 32*i;
    for (let b = 24; b < 32; b++) uo[db+b] = u[sb+b];
  }
  // Count-mismatch padding: each clone duplicates an already-built splat of
  // this cloud (same position/covariance/color) with alpha 0, so this cloud's
  // endpoint render is unchanged; along the morph the clone fades in/out while
  // traveling to/from its partner. The HOST choice is the whole difference
  // between the two padding modes: a random splat of this cloud (hosts = null,
  // the clone then starts wherever chance put it) or the splat nearest to the
  // surplus partner this clone was created for (hosts[k], short local travel).
  for (let k = 0; k < nClone; k++) {
    const s = hosts ? hosts[k] : (CLOUD_RAND() * R) | 0, di = R + k;
    fo.copyWithin(8*di, 8*s, 8*s + 6);                       // pos + scales
    uo.set(uo.subarray(32*s + 24, 32*s + 32), 32*di + 24);   // rgb + alpha + quat
    uo[32*di + 27] = 0;                                      // alpha 0
    pos[3*di] = pos[3*s]; pos[3*di+1] = pos[3*s+1]; pos[3*di+2] = pos[3*s+2];
    rows[di] = rows[s];                                      // clone inherits host row
  }
  return { buf: out, pos, N, rows };
}

// --- nearest-source clone hosts (the `nearest` padding mode) ----------------
// Uniform-grid nearest neighbour over the first `n` slots of a stride-3
// position array. Both clouds are normalized into the SAME unit sphere, so the
// nearest source splat of a target position is meaningful across the two.
//
// A grid, not a kd-tree: the points are a dense surface sample in a bounded box,
// so one counting sort (O(n)) buys constant-time cell lookup. Two details carry
// the performance at scene scale (1.2M points, 250k queries), both learned by
// measurement:
//   * the coordinates are STORED CELL-SORTED (px/py/pz below, with `ids` mapping
//     back). Reading a cell is then a contiguous scan instead of two dependent
//     random loads per point; that alone is the difference between a ~19 s and a
//     ~4 s host search on bee -> kitty.
//   * the stop test uses the true distance from the query to the OUTSIDE of the
//     already-scanned box, not the conservative (r-1)*cell. One saved ring is a
//     ~40% saving, because the work per radius grows cubically.
function buildGrid(pos, n) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = pos[3*i], y = pos[3*i+1], z = pos[3*i+2];
    if (x<x0)x0=x; if (x>x1)x1=x; if (y<y0)y0=y; if (y>y1)y1=y; if (z<z0)z0=z; if (z>z1)z1=z;
  }
  // ~2 points per cell if they filled the box; they lie on a surface, so cells
  // are fuller than that in practice. Measured optimum is shallow and this end
  // of it is the right one: finer grids pay more in ring expansion than they
  // save in per-cell scanning.
  const g = Math.max(1, Math.min(96, Math.round(Math.cbrt(n / 2))));
  const cell = Math.max(x1-x0, y1-y0, z1-z0, 1e-6) / g, inv = 1 / cell;
  const nx = Math.max(1, Math.min(g, Math.ceil((x1-x0) * inv))),
        ny = Math.max(1, Math.min(g, Math.ceil((y1-y0) * inv))),
        nz = Math.max(1, Math.min(g, Math.ceil((z1-z0) * inv)));
  const nc = nx * ny * nz;
  const cellOf = new Uint32Array(n), start = new Uint32Array(nc + 1);
  const cl = (v, m) => (v < 0 ? 0 : v > m ? m : v);
  for (let i = 0; i < n; i++) {
    const ix = cl(((pos[3*i]  -x0) * inv) | 0, nx-1),
          iy = cl(((pos[3*i+1]-y0) * inv) | 0, ny-1),
          iz = cl(((pos[3*i+2]-z0) * inv) | 0, nz-1);
    const c = (iz * ny + iy) * nx + ix;
    cellOf[i] = c; start[c + 1]++;
  }
  for (let c = 0; c < nc; c++) start[c + 1] += start[c];
  const px = new Float32Array(n), py = new Float32Array(n), pz = new Float32Array(n);
  const ids = new Uint32Array(n), cur = start.slice(0, nc);
  for (let i = 0; i < n; i++) {
    const s = cur[cellOf[i]]++;
    px[s] = pos[3*i]; py[s] = pos[3*i+1]; pz[s] = pos[3*i+2]; ids[s] = i;
  }
  return { x0, y0, z0, inv, cell, nx, ny, nz, start, px, py, pz, ids };
}
// Index (into the ORIGINAL position array) of the grid point closest to
// (x,y,z). Expanding cubic shells around the query cell, stopping as soon as the
// best hit is closer than anything the next shell could hold.
function nearestInGrid(G, x, y, z) {
  const cl = (v, m) => (v < 0 ? 0 : v > m ? m : v);
  const ix = cl(((x-G.x0) * G.inv) | 0, G.nx-1),
        iy = cl(((y-G.y0) * G.inv) | 0, G.ny-1),
        iz = cl(((z-G.z0) * G.inv) | 0, G.nz-1);
  const rmax = Math.max(G.nx, G.ny, G.nz);
  let best = 0, bd = Infinity;
  for (let r = 0; r <= rmax; r++) {
    if (bd < Infinity) {
      // What is left unscanned lies outside the box of cells ALREADY covered,
      // [i-(r-1), i+(r-1)]^3, so the distance from the query to that box's
      // exterior bounds anything still to come. (Negative while the query sits
      // outside the box, i.e. off the grid: nothing can be excluded yet and the
      // search simply keeps growing.) Using the query's true offset inside its
      // cell rather than a flat (r-1)*cell is what saves the extra ring.
      const p = r - 1;
      const s = Math.min(
        x - (G.x0 + (ix - p) * G.cell), (G.x0 + (ix + p + 1) * G.cell) - x,
        y - (G.y0 + (iy - p) * G.cell), (G.y0 + (iy + p + 1) * G.cell) - y,
        z - (G.z0 + (iz - p) * G.cell), (G.z0 + (iz + p + 1) * G.cell) - z);
      if (s > 0 && bd <= s * s) break;
    }
    for (let k = Math.max(0, iz-r); k <= Math.min(G.nz-1, iz+r); k++) {
      const kEdge = Math.abs(k - iz) === r;
      for (let j = Math.max(0, iy-r); j <= Math.min(G.ny-1, iy+r); j++) {
        const jEdge = Math.abs(j - iy) === r;
        // On a non-edge (j,k) row only the two i-faces of the shell are new, so
        // jump the interior in one step instead of walking and skipping it. The
        // jump must be a jump-TO (not a fixed stride): at the grid border iLo is
        // clamped and a stride from there would step straight past the far face.
        const iLo = Math.max(0, ix-r), iHi = Math.min(G.nx-1, ix+r);
        const faceOnly = r > 0 && !kEdge && !jEdge;
        for (let i = iLo; i <= iHi; i++) {
          if (faceOnly && i !== ix-r && i !== ix+r) { i = ix + r - 1; continue; }
          const c = (k * G.ny + j) * G.nx + i;
          for (let t = G.start[c], e = G.start[c+1]; t < e; t++) {
            const dx = G.px[t]-x, dy = G.py[t]-y, dz = G.pz[t]-z;
            const d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < bd) { bd = d2; best = t; }
          }
        }
      }
    }
  }
  return G.ids[best];
}
// Clone hosts for the SHORT cloud: one per surplus slot [core, N) of the long
// cloud, each the short cloud's real splat (slots [0, core)) closest to that
// surplus partner. POSITION ONLY, deliberately: the hosts must not depend on
// the assignment method's weight ω, or the padded cloud would differ per
// method and the comparison would stop being like-for-like.
function nearestHosts(shortPos, core, longPos, N) {
  const G = buildGrid(shortPos, core);
  const hosts = new Uint32Array(N - core);
  for (let k = core; k < N; k++)
    hosts[k - core] = nearestInGrid(G, longPos[3*k], longPos[3*k+1], longPos[3*k+2]);
  return hosts;
}

// Size boost for a cloud rendered at `rendered` of its `original` splats.
// The cap keeps heavy subsampling from over-blurring; ?boost= raises it for
// scenes that render a small fraction of a large cloud (e.g. map mode, where
// only the offline pair rows draw) and prefer coverage over crispness.
function sizeBoost(rendered, original) {
  const phi = Math.max(rendered / Math.max(original, 1), 1e-4);
  return Math.min(1 / Math.sqrt(phi), BOOST_CAP);
}

// Build a target arrangement aligned to source cloud A: slot i holds target
// splat map[i]. cloudA and cloudTgt are both in the shared normalized frame; the
// result is a standard cloud (same N slots as A) the renderer and sort worker
// consume unchanged. With the bijective maps used here every target appears
// exactly once, so at t=1 the slots reconstruct the entire target cloud (= B).
// The map[i] < 0 branch is a defensive GHOST (copy of source splat i with zero
// scale/alpha at its own spot, so it fades in place) for any future partial map.
function buildTargetFromMap(cloudA, cloudTgt, map) {
  const N = cloudA.N;
  const uA = new Uint8Array(cloudA.buf), uT = new Uint8Array(cloudTgt.buf);
  const out = new ArrayBuffer(N * ROW);
  const uOut = new Uint8Array(out), fOut = new Float32Array(out);
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const j = map[i];
    if (j >= 0) {                                          // real target splat
      uOut.set(uT.subarray(j*ROW, j*ROW + ROW), i*ROW);
      pos[3*i] = cloudTgt.pos[3*j]; pos[3*i+1] = cloudTgt.pos[3*j+1]; pos[3*i+2] = cloudTgt.pos[3*j+2];
    } else {                                               // ghost at source's own spot
      uOut.set(uA.subarray(i*ROW, i*ROW + ROW), i*ROW);    // copy source record...
      fOut[8*i+3] = 0; fOut[8*i+4] = 0; fOut[8*i+5] = 0;   // ...zero scale floats...
      uOut[i*ROW + 27] = 0;                                // ...and zero alpha
      pos[3*i] = cloudA.pos[3*i]; pos[3*i+1] = cloudA.pos[3*i+1]; pos[3*i+2] = cloudA.pos[3*i+2];
    }
  }
  return { buf: out, pos, N };
}

// Weighted CIELAB lightness channel of a built cloud — the 4th coordinate fed
// to pairings.js packPoints for the luminance-augmented assignment:
// w * (L*/50 - 1) maps L* in [0,100] onto [-w, w], the same units as w times
// the unit-sphere positions. Clones inherit their host's color, so they need
// no special case. Requires rgbToLab from trajectories.js, so load that first.
// `n` limits the channel to the first n slots (the CORE, when the clone tail is
// pre-paired and never reaches the matcher).
function lumChannel(cloud, w, n = cloud.N) {
  const u = new Uint8Array(cloud.buf), lab = [0, 0, 0];
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    rgbToLab(u[32*i + 24], u[32*i + 25], u[32*i + 26], lab);
    out[i] = w * (lab[0] / 50 - 1);
  }
  return out;
}

// Pack a built cloud into pairings.js's stride-4 assignment representation at
// lightness weight w (w = 0 -> geometry only, the omega = 0 case). Both hosts
// call this so the assignment input is identical in the browser and the bench.
// `n` packs only the first n slots, so a nearest-padded run hands the matcher
// exactly the real splats and the timing covers exactly what it matched.
function packCloud(cloud, w, n = cloud.N) {
  return packPoints(cloud.pos, w > 0 ? lumChannel(cloud, w, n) : null, n);
}
