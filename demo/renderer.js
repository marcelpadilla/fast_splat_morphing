// renderer.js — GPU-side plumbing for the splat viewer: camera math, the
// half-float texture packing, the depth-sort worker, and the two shaders.
//
// Loaded as a plain script before morph.js (see index.html). Nothing here knows
// about morphing, assignments or the UI: this is the part that would be the
// same in any 3D Gaussian-splat viewer, and it is NOT a contribution of the
// paper. The splat projection math and the counting-sort worker are adapted
// from antimatter15/splat (MIT); see ../repo/THIRD-PARTY-NOTICES.md.
//
// It is split out of morph.js so that file is only the morph itself:
// orchestration, the method registry, metrics and UI.
//
// Contents:
//   multiply4 / invert4 / getProjectionMatrix / sub3 / cross3 / norm3
//                       camera + projection matrices (column-major flat arrays)
//   floatToHalf / packHalf2x16 / packSplatTexture
//                       pack a cloud into the uint32 texture the shader reads.
//                       NOTE the name: cloud.js owns `packCloud` for packing a
//                       cloud into the ASSIGNMENT input. Distinct jobs, and
//                       these are plain scripts sharing one global scope.
//   createWorker        counting-sort worker source (back-to-front order for
//                       correct alpha compositing; stringified into a Blob)
//   vertSrc / fragSrc   the vertex shader that interpolates every per-splat
//                       attribute by the single uniform u_t (this is where the
//                       morph is actually evaluated, per frame, on the GPU)

// ---------------------------------------------------------------------------
// 4x4 matrix helpers (column-major flat arrays), from antimatter15/splat.
// ---------------------------------------------------------------------------
function multiply4(a, b) {
  return [
    b[0]*a[0]+b[1]*a[4]+b[2]*a[8]+b[3]*a[12], b[0]*a[1]+b[1]*a[5]+b[2]*a[9]+b[3]*a[13],
    b[0]*a[2]+b[1]*a[6]+b[2]*a[10]+b[3]*a[14], b[0]*a[3]+b[1]*a[7]+b[2]*a[11]+b[3]*a[15],
    b[4]*a[0]+b[5]*a[4]+b[6]*a[8]+b[7]*a[12], b[4]*a[1]+b[5]*a[5]+b[6]*a[9]+b[7]*a[13],
    b[4]*a[2]+b[5]*a[6]+b[6]*a[10]+b[7]*a[14], b[4]*a[3]+b[5]*a[7]+b[6]*a[11]+b[7]*a[15],
    b[8]*a[0]+b[9]*a[4]+b[10]*a[8]+b[11]*a[12], b[8]*a[1]+b[9]*a[5]+b[10]*a[9]+b[11]*a[13],
    b[8]*a[2]+b[9]*a[6]+b[10]*a[10]+b[11]*a[14], b[8]*a[3]+b[9]*a[7]+b[10]*a[11]+b[11]*a[15],
    b[12]*a[0]+b[13]*a[4]+b[14]*a[8]+b[15]*a[12], b[12]*a[1]+b[13]*a[5]+b[14]*a[9]+b[15]*a[13],
    b[12]*a[2]+b[13]*a[6]+b[14]*a[10]+b[15]*a[14], b[12]*a[3]+b[13]*a[7]+b[14]*a[11]+b[15]*a[15],
  ];
}
function invert4(a) {
  const b00=a[0]*a[5]-a[1]*a[4], b01=a[0]*a[6]-a[2]*a[4], b02=a[0]*a[7]-a[3]*a[4];
  const b03=a[1]*a[6]-a[2]*a[5], b04=a[1]*a[7]-a[3]*a[5], b05=a[2]*a[7]-a[3]*a[6];
  const b06=a[8]*a[13]-a[9]*a[12], b07=a[8]*a[14]-a[10]*a[12], b08=a[8]*a[15]-a[11]*a[12];
  const b09=a[9]*a[14]-a[10]*a[13], b10=a[9]*a[15]-a[11]*a[13], b11=a[10]*a[15]-a[11]*a[14];
  const det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (!det) return null;
  return [
    (a[5]*b11-a[6]*b10+a[7]*b09)/det, (a[2]*b10-a[1]*b11-a[3]*b09)/det,
    (a[13]*b05-a[14]*b04+a[15]*b03)/det, (a[10]*b04-a[9]*b05-a[11]*b03)/det,
    (a[6]*b08-a[4]*b11-a[7]*b07)/det, (a[0]*b11-a[2]*b08+a[3]*b07)/det,
    (a[14]*b02-a[12]*b05-a[15]*b01)/det, (a[8]*b05-a[10]*b02+a[11]*b01)/det,
    (a[4]*b10-a[5]*b08+a[7]*b06)/det, (a[1]*b08-a[0]*b10-a[3]*b06)/det,
    (a[12]*b04-a[13]*b02+a[15]*b00)/det, (a[9]*b02-a[8]*b04-a[11]*b00)/det,
    (a[5]*b07-a[4]*b09-a[6]*b06)/det, (a[0]*b09-a[1]*b07+a[2]*b06)/det,
    (a[13]*b01-a[12]*b03-a[14]*b00)/det, (a[8]*b03-a[9]*b01+a[10]*b00)/det,
  ];
}
function getProjectionMatrix(fx, fy, w, h) {
  const znear = 0.2, zfar = 200;
  return [
    (2*fx)/w, 0, 0, 0,
    0, -(2*fy)/h, 0, 0,
    0, 0, zfar/(zfar-znear), 1,
    0, 0, -(zfar*znear)/(zfar-znear), 0,
  ];
}
// vec helpers
const sub3 = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const cross3 = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

// ---------------------------------------------------------------------------
// Pack a subsampled 32-byte splat buffer into the antimatter15 uint32 texture
// (position float-bits in .xyz of the even texel; packed covariance sigma + RGBA
// in the odd texel). One texel pair per splat, texture width fixed at 2048.
// ---------------------------------------------------------------------------
const _f32 = new Float32Array(1), _i32 = new Int32Array(_f32.buffer);
function floatToHalf(f) {
  _f32[0] = f; const x = _i32[0];
  const sign = (x >> 31) & 1; let exp = (x >> 23) & 0xff; let frac = x & 0x7fffff;
  let ne;
  if (exp === 0) ne = 0;
  else if (exp < 113) { ne = 0; frac |= 0x800000; frac >>= (113 - exp); if (frac & 0x1000000) { ne = 1; frac = 0; } }
  else if (exp < 142) ne = exp - 112;
  else { ne = 31; frac = 0; }
  return (sign << 15) | (ne << 10) | (frac >> 13);
}
const packHalf2x16 = (x, y) => (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;

// NOTE: named packSplatTexture, not packCloud -- cloud.js owns `packCloud`
// for packing a cloud into the ASSIGNMENT input. These are plain scripts
// sharing one global scope, so a duplicate name would silently shadow it.
function packSplatTexture(buf, n) {
  const f = new Float32Array(buf), u = new Uint8Array(buf);
  const texwidth = 2048;
  const texheight = Math.ceil((2 * n) / texwidth);
  const texdata = new Uint32Array(texwidth * texheight * 4);
  const tc = new Uint8Array(texdata.buffer);
  const tf = new Float32Array(texdata.buffer);
  for (let i = 0; i < n; i++) {
    tf[8*i+0] = f[8*i+0]; tf[8*i+1] = f[8*i+1]; tf[8*i+2] = f[8*i+2];
    tc[4*(8*i+7)+0] = u[32*i+24]; tc[4*(8*i+7)+1] = u[32*i+25];
    tc[4*(8*i+7)+2] = u[32*i+26]; tc[4*(8*i+7)+3] = u[32*i+27];
    const s0 = f[8*i+3], s1 = f[8*i+4], s2 = f[8*i+5];
    const r0 = (u[32*i+28]-128)/128, r1 = (u[32*i+29]-128)/128,
          r2 = (u[32*i+30]-128)/128, r3 = (u[32*i+31]-128)/128;
    // M = S * R  (rotation matrix rows scaled by the per-axis scale)
    const M = [
      1-2*(r2*r2+r3*r3), 2*(r1*r2+r0*r3), 2*(r1*r3-r0*r2),
      2*(r1*r2-r0*r3), 1-2*(r1*r1+r3*r3), 2*(r2*r3+r0*r1),
      2*(r1*r3+r0*r2), 2*(r2*r3-r0*r1), 1-2*(r1*r1+r2*r2),
    ];
    M[0]*=s0; M[1]*=s0; M[2]*=s0; M[3]*=s1; M[4]*=s1; M[5]*=s1; M[6]*=s2; M[7]*=s2; M[8]*=s2;
    const sigma = [
      M[0]*M[0]+M[3]*M[3]+M[6]*M[6], M[0]*M[1]+M[3]*M[4]+M[6]*M[7], M[0]*M[2]+M[3]*M[5]+M[6]*M[8],
      M[1]*M[1]+M[4]*M[4]+M[7]*M[7], M[1]*M[2]+M[4]*M[5]+M[7]*M[8], M[2]*M[2]+M[5]*M[5]+M[8]*M[8],
    ];
    texdata[8*i+4] = packHalf2x16(SIGMA_K*sigma[0], SIGMA_K*sigma[1]);
    texdata[8*i+5] = packHalf2x16(SIGMA_K*sigma[2], SIGMA_K*sigma[3]);
    texdata[8*i+6] = packHalf2x16(SIGMA_K*sigma[4], SIGMA_K*sigma[5]);
  }
  return { texdata, texwidth, texheight };
}

// ---------------------------------------------------------------------------
// Sort worker: holds both clouds' positions (plus, in Bezier trajectory modes,
// the active waypoint positions) and depth-sorts by the interpolated position
// each time the view or t changes.
// ---------------------------------------------------------------------------
function createWorker(self) {
  let posA = null, posB = null, posW = null, count = 0;
  let running = false, pending = null, lastT = -1, lastProj = null, lastGen;

  function runSort(viewProj, t, gen) {
    if (!posA || !posB) return;
    const s = 1 - t;
    // straight lerp weights, or quadratic-Bezier weights through the waypoint
    const w0 = posW ? s * s : s, w1 = posW ? 2 * s * t : 0, w2 = posW ? t * t : t;
    let maxD = -Infinity, minD = Infinity;
    const depth = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      let x = posA[3*i]*w0 + posB[3*i]*w2;
      let y = posA[3*i+1]*w0 + posB[3*i+1]*w2;
      let z = posA[3*i+2]*w0 + posB[3*i+2]*w2;
      if (posW) { x += posW[3*i]*w1; y += posW[3*i+1]*w1; z += posW[3*i+2]*w1; }
      const d = ((viewProj[2]*x + viewProj[6]*y + viewProj[10]*z) * 4096) | 0;
      depth[i] = d;
      if (d > maxD) maxD = d;
      if (d < minD) minD = d;
    }
    const inv = (65535) / (maxD - minD || 1);
    const counts = new Uint32Array(65536);
    for (let i = 0; i < count; i++) { depth[i] = ((depth[i]-minD)*inv)|0; counts[depth[i]]++; }
    const starts = new Uint32Array(65536);
    for (let i = 1; i < 65536; i++) starts[i] = starts[i-1] + counts[i-1];
    const order = new Uint32Array(count);
    for (let i = 0; i < count; i++) order[starts[depth[i]]++] = i;
    self.postMessage({ order, count, gen }, [order.buffer]);
  }

  function schedule() {
    if (running || !pending) return;
    running = true;
    const { viewProj, t, gen } = pending; pending = null;
    runSort(viewProj, t, gen);
    setTimeout(() => { running = false; schedule(); }, 0);
  }

  self.onmessage = (e) => {
    const d = e.data;
    if (d.init) { posA = new Float32Array(d.posA); count = d.count; return; }
    // setB and setW arrive FUSED in one message on a mapping switch, so no
    // sort can execute against a mixed (new targets, stale waypoints) state;
    // a trajectory toggle sends setW alone. Either forces a re-sort.
    if (d.setB || 'setW' in d) {
      if (d.setB) posB = new Float32Array(d.setB);
      if ('setW' in d) posW = d.setW ? new Float32Array(d.setW) : null; // null = linear
      lastProj = null;
      return;
    }
    if (d.view) {
      // Skip only if the view, t AND capture generation are all unchanged since
      // the last queued sort. gen is undefined in interactive use (a harmless
      // no-op: undefined === undefined); in capture mode it bumps once per pose,
      // forcing a fresh sort even if the camera happens to match, so the driver's
      // "settled" signal can never be satisfied by a stale earlier frame.
      if (lastProj && d.t === lastT && d.gen === lastGen) {
        let same = true;
        for (let k = 0; k < 16; k++) if (Math.abs(lastProj[k]-d.view[k]) > 1e-9) { same = false; break; }
        if (same) return;
      }
      lastProj = d.view; lastT = d.t; lastGen = d.gen;
      pending = { viewProj: d.view, t: d.t, gen: d.gen };
      schedule();
    }
  };
}

// ---------------------------------------------------------------------------
// Shaders. Vertex shader fetches both clouds and interpolates by u_t — either
// a straight lerp (u_traj = 0) or a quadratic Bezier through the waypoint
// texture u_textureW (u_traj = 1; see trajectories.js). bez2() applies to
// positions, packed covariance halves, and colors alike; at t=0/1 it always
// returns the exact endpoint, whatever the waypoint holds.
// ---------------------------------------------------------------------------
const vertSrc = `#version 300 es
precision highp float; precision highp int;
uniform highp usampler2D u_textureA;
uniform highp usampler2D u_textureB;
uniform highp usampler2D u_textureW;
uniform mat4 projection, view;
uniform vec2 focal, viewport;
uniform float u_t;
uniform float u_traj;
uniform float u_kernel;
// Capture auto-frame: a screen-space (x = zoom, yz = pan) transform of the FINAL
// clip xy. Applied to the whole gl_Position — center AND splat extent — so a zoom
// magnifies splats too and never opens gaps. Default (1,0,0) = identity.
uniform vec3 u_fit;
in vec2 position;
in int index;
out vec4 vColor;
out vec2 vPosition;

vec4 rgbaOf(uint w) {
  return vec4(float(w & 0xffu), float((w >> 8) & 0xffu),
              float((w >> 16) & 0xffu), float((w >> 24) & 0xffu)) / 255.0;
}
vec3 bez2(vec3 a, vec3 w, vec3 b, float t) { return mix(mix(a, w, t), mix(w, b, t), t); }
vec2 bez2v(vec2 a, vec2 w, vec2 b, float t) { return mix(mix(a, w, t), mix(w, b, t), t); }
vec4 bez4(vec4 a, vec4 w, vec4 b, float t) { return mix(mix(a, w, t), mix(w, b, t), t); }

void main() {
  ivec2 cenUV = ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10);
  ivec2 covUV = ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10);

  uvec4 cenA = texelFetch(u_textureA, cenUV, 0);
  uvec4 cenB = texelFetch(u_textureB, cenUV, 0);
  vec3 pa = uintBitsToFloat(cenA.xyz), pb = uintBitsToFloat(cenB.xyz);
  vec3 p;
  if (u_traj > 0.5) {
    vec3 pw = uintBitsToFloat(texelFetch(u_textureW, cenUV, 0).xyz);
    p = bez2(pa, pw, pb, u_t);
  } else {
    p = mix(pa, pb, u_t);
  }

  vec4 cam = view * vec4(p, 1.0);
  vec4 pos2d = projection * cam;
  float clip = 1.2 * pos2d.w;
  if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return;
  }

  uvec4 covA = texelFetch(u_textureA, covUV, 0);
  uvec4 covB = texelFetch(u_textureB, covUV, 0);
  vec2 a1 = unpackHalf2x16(covA.x), a2 = unpackHalf2x16(covA.y), a3 = unpackHalf2x16(covA.z);
  vec2 b1 = unpackHalf2x16(covB.x), b2 = unpackHalf2x16(covB.y), b3 = unpackHalf2x16(covB.z);
  vec2 m1, m2, m3;
  vec4 rgba;
  if (u_traj > 0.5) {
    uvec4 covW = texelFetch(u_textureW, covUV, 0);
    vec2 w1 = unpackHalf2x16(covW.x), w2 = unpackHalf2x16(covW.y), w3 = unpackHalf2x16(covW.z);
    m1 = bez2v(a1, w1, b1, u_t); m2 = bez2v(a2, w2, b2, u_t); m3 = bez2v(a3, w3, b3, u_t);
    rgba = bez4(rgbaOf(covA.w), rgbaOf(covW.w), rgbaOf(covB.w), u_t);
  } else {
    m1 = mix(a1, b1, u_t); m2 = mix(a2, b2, u_t); m3 = mix(a3, b3, u_t);
    rgba = mix(rgbaOf(covA.w), rgbaOf(covB.w), u_t);
  }
  // The 3x3 covariance cone is convex and lerp/Bezier weights are positive
  // (Bernstein), so any interpolant of valid covariances is valid.
  mat3 Vrk = mat3(m1.x, m1.y, m2.x, m1.y, m2.y, m3.x, m2.x, m3.x, m3.y);

  mat3 J = mat3(
    focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
    0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
    0., 0., 0.);
  mat3 T = transpose(mat3(view)) * J;
  mat3 cov2d = transpose(T) * Vrk * T;

  // Screen-space low-pass dilation (GaussianSplats3D kernel2DSize): guarantees
  // every splat covers >= ~1px so sub-pixel splats don't alias into speckle.
  cov2d[0][0] += u_kernel;
  cov2d[1][1] += u_kernel;

  float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
  float rad = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
  float lambda1 = mid + rad, lambda2 = mid - rad;
  if (lambda2 < 0.0) return;
  vec2 diag = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
  vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diag;
  vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diag.y, -diag.x);

  vColor = clamp(pos2d.z / pos2d.w + 1.0, 0.0, 1.0) * rgba;
  vPosition = position;
  vec2 vCenter = vec2(pos2d) / pos2d.w;
  vec2 ndc = vCenter + position.x * majorAxis / viewport + position.y * minorAxis / viewport;
  gl_Position = vec4(ndc * u_fit.x + u_fit.yz, 0.0, 1.0);
}`;

const fragSrc = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPosition;
out vec4 fragColor;
void main() {
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  fragColor = vec4(B * vColor.rgb, B);
}`;
