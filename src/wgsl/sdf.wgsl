// gen3d SDF library. Units are millimetres, +Z is up (build plate is z = 0).
// Most formulas from Inigo Quilez: https://iquilezles.org/articles/distfunctions/
// Include with:  #include <sdf.wgsl>

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// ---------------------------------------------------------------- primitives
fn sdSphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }

// b = half-extents
fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 { return sdBox(p, b - r) - r; }

// torus in XY plane: R = major radius, r = tube radius
fn sdTorus(p: vec3f, R: f32, r: f32) -> f32 {
    let q = vec2f(length(p.xy) - R, p.z);
    return length(q) - r;
}

// Z-aligned cylinder centred at origin, radius r, half-height h
fn sdCylinder(p: vec3f, r: f32, h: f32) -> f32 {
    let d = abs(vec2f(length(p.xy), p.z)) - vec2f(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
fn sdRoundCylinder(p: vec3f, r: f32, h: f32, rr: f32) -> f32 {
    return sdCylinder(p, r - rr, h - rr) - rr;
}

// capsule between points a and b
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Z-aligned capped cone from z=-h (radius r1) to z=+h (radius r2)
fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
    let q = vec2f(length(p.xy), p.z);
    let k1 = vec2f(r2, h);
    let k2 = vec2f(r2 - r1, 2.0 * h);
    let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

// regular hexagonal prism along Z: h.x = inner radius, h.y = half-height
fn sdHexPrism(p: vec3f, h: vec2f) -> f32 {
    let k = vec3f(-0.8660254, 0.5, 0.57735);
    let a = abs(p);
    let xy = a.xy - 2.0 * min(dot(k.xy, a.xy), 0.0) * k.xy;
    let d = vec2f(length(xy - vec2f(clamp(xy.x, -k.z * h.x, k.z * h.x), h.x)) * sign(xy.y - h.x),
                  a.z - h.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdOctahedron(p: vec3f, s: f32) -> f32 {
    let a = abs(p);
    return (a.x + a.y + a.z - s) * 0.57735027;
}

// infinite plane with normal n (normalized) at offset h
fn sdPlane(p: vec3f, n: vec3f, h: f32) -> f32 { return dot(p, n) + h; }

// ---------------------------------------------------------------- 2D -> 3D
fn sd2Circle(p: vec2f, r: f32) -> f32 { return length(p) - r; }
fn sd2Box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}
// extrude a 2D distance (in XY) along Z with half-height h
fn opExtrude(p: vec3f, d2: f32, h: f32) -> f32 {
    let w = vec2f(d2, abs(p.z) - h);
    return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
}
// revolve around Z: returns 2D coords (radial distance - o, z) for a 2D profile
fn opRevolve(p: vec3f, o: f32) -> vec2f { return vec2f(length(p.xy) - o, p.z); }

// ---------------------------------------------------------------- booleans
fn opUnion(a: f32, b: f32) -> f32 { return min(a, b); }
fn opSubtract(a: f32, b: f32) -> f32 { return max(a, -b); } // a minus b
fn opIntersect(a: f32, b: f32) -> f32 { return max(a, b); }

fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
fn opSmoothSubtract(a: f32, b: f32, k: f32) -> f32 { // a minus b
    let h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}
fn opSmoothIntersect(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}
// 45-degree chamfered union
fn opChamferUnion(a: f32, b: f32, r: f32) -> f32 {
    return min(min(a, b), (a - r + b) * 0.70710678);
}

// ---------------------------------------------------------------- modifiers
fn opRound(d: f32, r: f32) -> f32 { return d - r; }
fn opShell(d: f32, t: f32) -> f32 { return abs(d) - 0.5 * t; } // hollow, wall thickness t

// ---------------------------------------------------------------- transforms (apply to p)
// GLSL-style modulo (result has the sign of y); WGSL's `%` truncates instead.
fn floorMod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }
fn deg(d: f32) -> f32 { return d * PI / 180.0; }
fn rot2(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }
fn rotX(p: vec3f, a: f32) -> vec3f { let yz = rot2(a) * p.yz; return vec3f(p.x, yz); }
fn rotY(p: vec3f, a: f32) -> vec3f { let xz = rot2(a) * p.xz; return vec3f(xz.x, p.y, xz.y); }
fn rotZ(p: vec3f, a: f32) -> vec3f { let xy = rot2(a) * p.xy; return vec3f(xy, p.z); }

// twist around Z: `rate` radians per mm
fn opTwist(p: vec3f, rate: f32) -> vec3f { let xy = rot2(rate * p.z) * p.xy; return vec3f(xy, p.z); }
// bend in XZ plane
fn opBend(p: vec3f, k: f32) -> vec3f { let xz = rot2(k * p.x) * p.xz; return vec3f(xz.x, p.y, xz.y); }

// infinite repetition with period s (per axis; use 0 to disable an axis)
fn opRepeat(p: vec3f, s: vec3f) -> vec3f {
    let safe = select(vec3f(1.0), s, s > vec3f(0.0));
    return select(p, p - safe * round(p / safe), s > vec3f(0.0));
}
// limited repetition: copies from -l to +l (per axis, integer counts)
fn opRepeatLim(p: vec3f, s: f32, l: vec3f) -> vec3f { return p - s * clamp(round(p / s), -l, l); }

// polar repetition around Z: n copies
fn opPolar(p: vec3f, n: f32) -> vec3f {
    let a = TAU / n;
    let ang = floorMod(atan2(p.y, p.x) + 0.5 * a, a) - 0.5 * a;
    let r = length(p.xy);
    return vec3f(r * cos(ang), r * sin(ang), p.z);
}

// mirror across planes
fn opMirrorX(p: vec3f) -> vec3f { return vec3f(abs(p.x), p.yz); }
fn opMirrorY(p: vec3f) -> vec3f { return vec3f(p.x, abs(p.y), p.z); }

// ---------------------------------------------------------------- TPMS lattices (period s mm)
// Use as infill: opIntersect(shape, opShell(sdGyroid(p, 10.0), 1.0))
fn sdGyroid(p: vec3f, s: f32) -> f32 {
    let q = p * (TAU / s);
    let g = dot(sin(q), cos(q.yzx));
    return g * s / (TAU * 1.5); // approx. scaling to mm
}
fn sdSchwarzP(p: vec3f, s: f32) -> f32 {
    let q = p * (TAU / s);
    return (cos(q.x) + cos(q.y) + cos(q.z)) * s / (TAU * 1.7);
}

// ---------------------------------------------------------------- noise (for displacement)
fn hash13(p3: vec3f) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
fn valueNoise(p: vec3f) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3f(0, 0, 0)), hash13(i + vec3f(1, 0, 0)), f.x),
                   mix(hash13(i + vec3f(0, 1, 0)), hash13(i + vec3f(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3f(0, 0, 1)), hash13(i + vec3f(1, 0, 1)), f.x),
                   mix(hash13(i + vec3f(0, 1, 1)), hash13(i + vec3f(1, 1, 1)), f.x), f.y), f.z);
}
fn fbm(p: vec3f) -> f32 {
    var a = 0.5;
    var s = 0.0;
    var q = p;
    for (var i = 0; i < 5; i++) { s += a * valueNoise(q); q *= 2.03; a *= 0.5; }
    return s;
}
