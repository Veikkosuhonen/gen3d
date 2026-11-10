// gen3d SDF library. Units are millimetres, +Z is up (build plate is z = 0).
// Most formulas from Inigo Quilez: https://iquilezles.org/articles/distfunctions/
// Include with:  #include <sdf.glsl>

#define PI  3.14159265359
#define TAU 6.28318530718

// ---------------------------------------------------------------- primitives
float sdSphere(vec3 p, float r) { return length(p) - r; }

// b = half-extents
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdRoundBox(vec3 p, vec3 b, float r) { return sdBox(p, b - r) - r; }

// torus in XY plane: R = major radius, r = tube radius
float sdTorus(vec3 p, float R, float r) {
    vec2 q = vec2(length(p.xy) - R, p.z);
    return length(q) - r;
}

// Z-aligned cylinder centred at origin, radius r, half-height h
float sdCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sdRoundCylinder(vec3 p, float r, float h, float rr) {
    return sdCylinder(p, r - rr, h - rr) - rr;
}

// capsule between points a and b
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Z-aligned capped cone from z=-h (radius r1) to z=+h (radius r2)
float sdCappedCone(vec3 p, float h, float r1, float r2) {
    vec2 q = vec2(length(p.xy), p.z);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

// regular hexagonal prism along Z: h.x = inner radius, h.y = half-height
float sdHexPrism(vec3 p, vec2 h) {
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
                  p.z - h.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

// infinite plane with normal n (normalized) at offset h
float sdPlane(vec3 p, vec3 n, float h) { return dot(p, n) + h; }

// ---------------------------------------------------------------- 2D -> 3D
float sd2Circle(vec2 p, float r) { return length(p) - r; }
float sd2Box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
// extrude a 2D distance (in XY) along Z with half-height h
float opExtrude(vec3 p, float d2, float h) {
    vec2 w = vec2(d2, abs(p.z) - h);
    return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
}
// revolve around Z: returns 2D coords (radial distance - o, z) for a 2D profile
vec2 opRevolve(vec3 p, float o) { return vec2(length(p.xy) - o, p.z); }

// ---------------------------------------------------------------- booleans
float opUnion(float a, float b) { return min(a, b); }
float opSubtract(float a, float b) { return max(a, -b); } // a minus b
float opIntersect(float a, float b) { return max(a, b); }

float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float opSmoothSubtract(float a, float b, float k) { // a minus b
    float h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}
float opSmoothIntersect(float a, float b, float k) {
    float h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}
// 45-degree chamfered union
float opChamferUnion(float a, float b, float r) {
    return min(min(a, b), (a - r + b) * 0.70710678);
}

// ---------------------------------------------------------------- modifiers
float opRound(float d, float r) { return d - r; }
float opShell(float d, float t) { return abs(d) - 0.5 * t; } // hollow, wall thickness t

// ---------------------------------------------------------------- transforms (apply to p)
mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
vec3 rotX(vec3 p, float a) { p.yz = rot2(a) * p.yz; return p; }
vec3 rotY(vec3 p, float a) { p.xz = rot2(a) * p.xz; return p; }
vec3 rotZ(vec3 p, float a) { p.xy = rot2(a) * p.xy; return p; }
float deg(float d) { return d * PI / 180.0; }

// twist around Z: `rate` radians per mm
vec3 opTwist(vec3 p, float rate) { p.xy = rot2(rate * p.z) * p.xy; return p; }
// bend in XZ plane
vec3 opBend(vec3 p, float k) { p.xz = rot2(k * p.x) * p.xz; return p; }

// infinite repetition with period s (per axis; use 0 to disable an axis)
vec3 opRepeat(vec3 p, vec3 s) {
    vec3 q = p;
    if (s.x > 0.0) q.x = p.x - s.x * round(p.x / s.x);
    if (s.y > 0.0) q.y = p.y - s.y * round(p.y / s.y);
    if (s.z > 0.0) q.z = p.z - s.z * round(p.z / s.z);
    return q;
}
// limited repetition: copies from -l to +l (per axis, integer counts)
vec3 opRepeatLim(vec3 p, float s, vec3 l) { return p - s * clamp(round(p / s), -l, l); }

// polar repetition around Z: n copies
vec3 opPolar(vec3 p, float n) {
    float a = TAU / n;
    float ang = atan(p.y, p.x) + 0.5 * a;
    ang = mod(ang, a) - 0.5 * a;
    float r = length(p.xy);
    return vec3(r * cos(ang), r * sin(ang), p.z);
}

// mirror across planes
vec3 opMirrorX(vec3 p) { p.x = abs(p.x); return p; }
vec3 opMirrorY(vec3 p) { p.y = abs(p.y); return p; }

// ---------------------------------------------------------------- TPMS lattices (period s mm)
// Use as infill: opIntersect(shape, opShell(sdGyroid(p, 10.0), 1.0))
float sdGyroid(vec3 p, float s) {
    vec3 q = p * (TAU / s);
    float g = dot(sin(q), cos(q.yzx));
    return g * s / (TAU * 1.5); // approx. scaling to mm
}
float sdSchwarzP(vec3 p, float s) {
    vec3 q = p * (TAU / s);
    return (cos(q.x) + cos(q.y) + cos(q.z)) * s / (TAU * 1.7);
}

// ---------------------------------------------------------------- noise (for displacement)
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float valueNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), f.x),
                   mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
                   mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 5; i++) { s += a * valueNoise(p); p *= 2.03; a *= 0.5; }
    return s;
}
