// pairings.js — assignment (bijection) builders for the splat morph.
//
// A mapping is an Int32Array `map` of length N with map[i] = the target index
// paired to source slot i, and it is a PERMUTATION of [0, N): every target is
// used exactly once. Because the union of targets is the entire target cloud,
// at t=1 the render is EXACTLY B for ANY mapping — the mapping only shapes the
// in-between trajectories, never the endpoints. That is why an approximate
// assignment costs nothing at the endpoints.
//
// Loaded as a plain script before morph.js (no modules — keeps the demo
// servable by any static server), and loaded VERBATIM by code/bench_hier.mjs,
// so the numbers in the paper are the numbers the browser produces.
//
// ---------------------------------------------------------------------------
// ONE point representation, ONE weight knob
// ---------------------------------------------------------------------------
// Every builder here consumes the same packed stride-4 array
//     [x, y, z, omega-weighted lightness]
// produced by packPoints(). The 4th coordinate is the CIELAB L* channel scaled
// by the lightness weight omega (callers build it as w * (L*/50 - 1), so L* in
// [0,100] maps onto the position range [-1,1]).
//
// omega = 0 makes that coordinate identically zero, so the SAME code path
// reproduces the geometry-only assignment bit-for-bit (verified 2026-07-21 at
// N = 100k and N = 400k against the previous separate implementation). There is
// deliberately NO second geometry-only implementation: the method is one
// algorithm with a weight, exactly as the paper presents it (FSM at omega = 0
// and the default FSM at omega = 1/4). Swapping the weight is the only thing
// that distinguishes them.
//
// ---------------------------------------------------------------------------
// Method origins — where each assignment came from
// ---------------------------------------------------------------------------
//   index        No builder lives here; morph.js uses the identity permutation
//                (slot i -> slot i). It is the naive floor: an arbitrary
//                bijection with no geometric meaning, kept only as the
//                worst-case reference every other method is scored against.
//
//   hierPairing  OURS. This is the paper's method, Fast Splat Morphing (FSM).
//                It is Hierarchical Refinement optimal transport (HiRef;
//                Halmos et al., ICML 2025, arXiv:2503.03025) SPECIALIZED to its
//                rank-2 case, with HiRef's per-node low-rank OT solve replaced
//                by a closed-form geometric median split. That single
//                substitution is what buys the ~10^3x speedup, and what costs
//                the small measured optimality gap. Background and the
//                measured alternatives: knowledge/assignment-scaling.md sec. B.
//
//   hier4        REMOVED 2026-07-22. The identical recursion at arity 4 (2x2
//                co-partition, exact 4x4 block assignment) was built and
//                measured: same travel as hierPairing to 3 decimals, +16%
//                faster at N=100k but 21% SLOWER at N=400k. Marcel froze the
//                method at BINARY splitting for the paper, so it was deleted
//                here. Its measurements live on in evaluation/evaluation.html
//                and knowledge/decisions.md; do not reintroduce it.
//
//   sinkhorn     NOT built here. Entropic OT (Cuturi 2013; Peyre & Cuturi
//   exact        2019), solved offline in log domain on the GPU by
//   hiref        code/solve_ot.py, loaded as a .pairs file.
//                exact: the true optimum, Hungarian / Jonker-Volgenant LAP via
//                scipy linear_sum_assignment. O(n^3), so it only exists at
//                reduced n; it is the ground truth every optimality gap is
//                measured against.
//                hiref: the ORIGIN of our method. Official HiRef (FRLC low-rank
//                + rank annealing), vendored at code/vendor/hiref and driven by
//                code/compare_methods.py. We specialize it, and we also beat it
//                on transport cost and on time.
//                All three are offline references that exist to show where the
//                scaling wall is; they are loaded by morph.js, not computed in
//                the browser.

const PT_K = 4;                    // stride: [x, y, z, omega*lightness]

// Pack stride-3 positions plus an optional PRE-WEIGHTED lightness channel into
// the shared representation. Pass lum = null (or an all-zero channel, i.e.
// omega = 0) to get the geometry-only assignment.
function packPoints(pos, lum, N) {
  const P = new Float32Array(PT_K * N);
  for (let i = 0; i < N; i++) {
    P[PT_K*i]     = pos[3*i];
    P[PT_K*i + 1] = pos[3*i + 1];
    P[PT_K*i + 2] = pos[3*i + 2];
    P[PT_K*i + 3] = lum ? lum[i] : 0;
  }
  return P;
}

// --- shared primitives (all stride-4) --------------------------------------
// nth_element: reorder idx[lo..hi) so the (k-lo) smallest along `axis` come first.
function quickselect(idx, lo, hi, k, P, axis) {
  while (hi - lo > 1) {
    // median-of-3 pivot
    const a = P[PT_K*idx[lo]+axis], b = P[PT_K*idx[(lo+hi)>>1]+axis], c = P[PT_K*idx[hi-1]+axis];
    const p = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
    let i = lo, j = hi - 1;
    while (i <= j) {
      while (P[PT_K*idx[i]+axis] < p) i++;
      while (P[PT_K*idx[j]+axis] > p) j--;
      if (i <= j) { const t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
    }
    if (k <= j) hi = j + 1; else if (k >= i) lo = i; else return;
  }
}
function extents(idx, lo, hi, P, out) {
  let x0=Infinity, x1=-Infinity, y0=Infinity, y1=-Infinity,
      z0=Infinity, z1=-Infinity, l0=Infinity, l1=-Infinity;
  for (let t = lo; t < hi; t++) {
    const o = PT_K*idx[t], x = P[o], y = P[o+1], z = P[o+2], l = P[o+3];
    if (x<x0)x0=x; if (x>x1)x1=x; if (y<y0)y0=y; if (y>y1)y1=y;
    if (z<z0)z0=z; if (z>z1)z1=z; if (l<l0)l0=l; if (l>l1)l1=l;
  }
  out[0] = x1-x0; out[1] = y1-y0; out[2] = z1-z0; out[3] = l1-l0;
}
function centroid(idx, lo, hi, P, out) {
  let cx = 0, cy = 0, cz = 0, cl = 0; const n = hi - lo;
  for (let t = lo; t < hi; t++) {
    const o = PT_K*idx[t]; cx += P[o]; cy += P[o+1]; cz += P[o+2]; cl += P[o+3];
  }
  out[0] = cx/n; out[1] = cy/n; out[2] = cz/n; out[3] = cl/n;
}
// squared distance between two 4-vectors (centroids)
function dist2(p, q) {
  const dx=p[0]-q[0], dy=p[1]-q[1], dz=p[2]-q[2], dl=p[3]-q[3];
  return dx*dx + dy*dy + dz*dz + dl*dl;
}
// squared source-to-target cost between packed point x of A and y of B
function cost2(A, x, B, y) {
  const dx=A[PT_K*x]-B[PT_K*y], dy=A[PT_K*x+1]-B[PT_K*y+1],
        dz=A[PT_K*x+2]-B[PT_K*y+2], dl=A[PT_K*x+3]-B[PT_K*y+3];
  return dx*dx + dy*dy + dz*dz + dl*dl;
}

// --- hier: binary hierarchical co-bisection (the featured method) -----------
// Approximate min-cost assignment via recursive BALANCED co-partition: split
// BOTH clouds into two equal-count halves along ONE SHARED axis (the widest by
// combined extents, which may be the lightness axis), pair the halves straight
// or crossed by whichever makes the half-centroids travel less, and recurse to
// singletons. O(N log N).
//
// Sharing the split axis is essential and is not a detail: choosing each
// cloud's own widest axis independently pairs geometrically unrelated halves
// when extents are nearly tied (measured: mean travel 1.0 vs 0.018 on matched
// shells). Odd-n crossed pairing re-splits B at the floor half so the two sides
// still match in size; leaves of 2 are solved exactly.
function hierPairing(A, B, N) {
  const idxA = new Uint32Array(N), idxB = new Uint32Array(N);
  for (let i = 0; i < N; i++) { idxA[i] = i; idxB[i] = i; }
  const map = new Int32Array(N);
  if (N < 1) return map;                     // see below; never taken for N >= 1
  const eA = [0,0,0,0], eB = [0,0,0,0];
  const cAlo = [0,0,0,0], cAhi = [0,0,0,0], cBlo = [0,0,0,0], cBhi = [0,0,0,0];
  // The empty node is the one size the recursion cannot end on. A node of n >= 3
  // splits into ceil(n/2) and floor(n/2), both >= 1, so no SUBNODE is ever empty
  // and this guard is unreachable for N >= 1 (every published number is
  // bit-identical with and without it). The TOP node can be empty, though, if a
  // scene payload arrives truncated or empty: centroid() of nothing is NaN, the
  // crossed test is then false, and the straight branch pushes two more empty
  // nodes forever. That froze the tab and exhausted the heap in ~1 s rather than
  // reporting anything (2026-09-11).
  const stack = [[0, N, 0, N]];              // [loA, hiA, loB, hiB], equal sizes
  while (stack.length) {
    const [la, ha, lb, hb] = stack.pop();
    const n = ha - la;
    if (n === 1) { map[idxA[la]] = idxB[lb]; continue; }
    if (n === 2) {                           // exact 2-point assignment
      const a0 = idxA[la], a1 = idxA[la+1], b0 = idxB[lb], b1 = idxB[lb+1];
      if (cost2(A,a0,B,b0) + cost2(A,a1,B,b1) <= cost2(A,a0,B,b1) + cost2(A,a1,B,b0)) {
        map[a0] = b0; map[a1] = b1;
      } else { map[a0] = b1; map[a1] = b0; }
      continue;
    }
    // ONE shared split axis: widest by the summed extents of both subsets. When
    // the weighted lightness spread exceeds the spatial extents, the node
    // splits dark-vs-bright first and space second.
    extents(idxA, la, ha, A, eA); extents(idxB, lb, hb, B, eB);
    let ax = 0, ebest = eA[0] + eB[0];
    for (let c = 1; c < PT_K; c++) { const e = eA[c] + eB[c]; if (e > ebest) { ebest = e; ax = c; } }
    const kHalf = (n + 1) >> 1;              // ceil half
    const ka = la + kHalf, kb = lb + kHalf;
    quickselect(idxA, la, ha, ka, A, ax);
    quickselect(idxB, lb, hb, kb, B, ax);
    // Straight (lo<->lo) or crossed (lo<->hi) half pairing, by centroid travel.
    centroid(idxA, la, ka, A, cAlo); centroid(idxA, ka, ha, A, cAhi);
    centroid(idxB, lb, kb, B, cBlo); centroid(idxB, kb, hb, B, cBhi);
    if (dist2(cAlo,cBhi) + dist2(cAhi,cBlo) < dist2(cAlo,cBlo) + dist2(cAhi,cBhi)) {
      let kb2 = kb;
      // Crossed halves must still match sizes: for odd n re-split B at the
      // floor half so B_hi has ceil elements (pairs A_lo) and B_lo floor (A_hi).
      if (n & 1) { kb2 = lb + (n >> 1); quickselect(idxB, lb, hb, kb2, B, ax); }
      stack.push([la, ka, kb2, hb]);
      stack.push([ka, ha, lb, kb2]);
    } else {
      stack.push([la, ka, lb, kb]);
      stack.push([ka, ha, kb, hb]);
    }
  }
  return map;
}

// Mean travel distance of a mapping (unit-sphere units) — the pure geometric
// objective the mappings are compared on. Takes the STRIDE-3 position arrays,
// not the packed points: travel is always position-only, whatever omega the
// assignment itself was built with, so hier and hier+lum stay comparable.
function meanTravel(map, srcPos, tgtPos, N) {
  let s = 0;
  for (let i = 0; i < N; i++) {
    const j = map[i];
    const dx = tgtPos[3*j]-srcPos[3*i], dy = tgtPos[3*j+1]-srcPos[3*i+1], dz = tgtPos[3*j+2]-srcPos[3*i+2];
    s += Math.sqrt(dx*dx + dy*dy + dz*dz);
  }
  return s / N;
}
