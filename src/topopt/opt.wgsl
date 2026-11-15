// GPU kernels for the optimizer's per-element steps (see gpu_opt.rs; mirrors the CPU code in mod.rs).
// Uses the prelude of mg.wgsl. Element flags: bit 0 domain, bit 1 kept solid (passive), bit 2 design
// (domain and not passive). Elements are in the fine grid's brick layout (P.m.x of them stored);
// kernels that look at neighbours get its page table and slot coordinates. Elements of absent bricks
// are either outside the domain or dropped (fixed at x = 0, see gpu_opt.rs): both read as 0. The
// filter's weight sums (`wsum`, Σ rmin - distance over domain elements within rmin) are worked out
// on the host, so they count dropped elements.

//// lib elem_grid
#include lookup NAME=fat PAGE=page

// grid coordinates of stored element e
fn elem_point(e: u32) -> vec3<i32> {
    return vec3<i32>(coords[e / BV].xyz * bshape() + unflat(e % BV, bshape()));
}

// the mirror image of p in the planes P.m.z (bits x, y, z) at the grid's low faces
fn mirror(p: vec3<i32>) -> vec3<i32> {
    var q = p;
    for (var a = 0u; a < 3u; a++) {
        if q[a] < 0 && ((P.m.z >> a) & 1u) != 0u { q[a] = -q[a] - 1; }
    }
    return q;
}

//// kernel stiffness
// ee = (E_min + x^p (E0 - E_min)) h on the domain, 0 elsewhere. P.f = (E_min, E0, p, h)
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> xphys: array<f32>;
@group(0) @binding(3) var<storage, read_write> ee: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    if e >= P.m.x { return; }
    var v = 0.0;
    if (flags[e] & 1u) != 0u {
        let x = xphys[e];
        let xp = select(0.0, pow(x, P.f.z), x > 0.0);
        v = (P.f.x + xp * (P.f.y - P.f.x)) * P.f.w;
    }
    ee[e] = v;
}

//// kernel sens
// Compliance and volume sensitivities of the design elements through the projection, divided by
// the filter weight sums (ready for the filter's transpose); 0 elsewhere. With the AM filter
// (P.m.w = 1) not divided: `am_adj` continues the chain rule and divides. `xtilde` is the
// projection's input (x̂ with the AM filter).
// P.f = (E_min, E0, p, h), P.g.y = η, β = S[P.m.y]
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> xphys: array<f32>;
@group(0) @binding(3) var<storage, read> xtilde: array<f32>;
@group(0) @binding(4) var<storage, read> ce: array<f32>;
@group(0) @binding(5) var<storage, read> S: array<f32>;
@group(0) @binding(6) var<storage, read_write> dcs: array<f32>;
@group(0) @binding(7) var<storage, read_write> dvs: array<f32>;
@group(0) @binding(8) var<storage, read> wsum: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    if e >= P.m.x { return; }
    var dc = 0.0;
    var dv = 0.0;
    if (flags[e] & 4u) != 0u {
        let beta = S[P.m.y];
        let eta = P.g.y;
        var dproj = 1.0;
        if beta > 0.0 {
            let t = tanh(beta * (xtilde[e] - eta));
            dproj = beta * (1.0 - t * t) / (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
        }
        let x = xphys[e];
        let xp = select(0.0, pow(x, P.f.z - 1.0), x > 0.0);
        var hs = 1.0;
        if P.m.w == 0u { hs = wsum[e]; }
        dc = -P.f.z * xp * (P.f.y - P.f.x) * P.f.w * ce[e] * dproj / hs;
        dv = dproj / hs;
    }
    dcs[e] = dc;
    dvs[e] = dv;
}

//// kernel filter bricks
// Cone filter over domain elements within P.f.x voxels (weights rmin - distance). P.m.y: 0 the raw
// weighted sum on domain elements; 1 densities: kept-solid 1, domain the weighted mean, else 0.
// P.m.z: axes (bits x, y, z) whose low face is a mirror plane: neighbours past it are the mirror
// images (index -i - 1). Densities divide by the weight sums `wsum`.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read> page: array<u32>;
@group(0) @binding(5) var<storage, read> coords: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> wsum: array<f32>;

#include elem_grid

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    if e >= P.m.x { return; }
    let f = flags[e];
    if (f & 1u) == 0u {
        outp[e] = 0.0;
        return;
    }
    if P.m.y == 1u && (f & 2u) != 0u {
        outp[e] = 1.0;
        return;
    }
    let p = elem_point(e);
    let n = vec3<i32>(P.n.xyz);
    let nb = nbricks(P.n.xyz);
    let rmin = P.f.x;
    let r = i32(ceil(rmin));
    var s = 0.0;
    for (var dk = -r; dk <= r; dk++) {
        for (var dj = -r; dj <= r; dj++) {
            for (var di = -r; di <= r; di++) {
                let w = rmin - sqrt(f32(di * di + dj * dj + dk * dk));
                if w <= 0.0 { continue; }
                let q = mirror(p + vec3<i32>(di, dj, dk));
                if any(q >= n) { continue; }
                let m = fat(q, nb);
                if m == NONE { continue; }
                if (flags[m] & 1u) != 0u { s += w * inp[m]; }
            }
        }
    }
    if P.m.y == 1u { s /= wsum[e]; }
    outp[e] = s;
}

//// kernel oc
// Optimality-criteria update of the design elements for multiplier λ = S[P.m.y], with the compliance
// sensitivity scaled by 1 / (|S[P.m.z]| P.f.y) (P.f.y: the whole part's compliance per the reduced
// model's, see symmetry.rs) and move limit P.f.x; other elements keep x.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> dc: array<f32>;
@group(0) @binding(4) var<storage, read> dv: array<f32>;
@group(0) @binding(5) var<storage, read> S: array<f32>;
@group(0) @binding(6) var<storage, read_write> xnew: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    if e >= P.m.x { return; }
    let xe = x[e];
    if (flags[e] & 4u) == 0u {
        xnew[e] = xe;
        return;
    }
    let lam = S[P.m.y];
    let cscale = 1.0 / max(abs(S[P.m.z]) * P.f.y, 1e-30);
    let be = max(-dc[e] * cscale, 0.0) / max(dv[e] * lam, 1e-30);
    let mv = P.f.x;
    xnew[e] = clamp(xe * sqrt(be), max(xe - mv, 0.0), min(xe + mv, 1.0));
}

//// kernel project
// Heaviside projection of the design elements' filtered densities, β = S[P.m.y] (0 = identity),
// threshold η = P.g.y; other elements copy.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> xt: array<f32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> xphys: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    if e >= P.m.x { return; }
    let v = xt[e];
    let beta = S[P.m.y];
    if (flags[e] & 4u) == 0u || beta == 0.0 {
        xphys[e] = v;
        return;
    }
    let eta = P.g.y;
    xphys[e] = (tanh(beta * eta) + tanh(beta * (v - eta))) / (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
}

//// kernel am_fwd bricks
// The AM filter (overhang.rs) on layer l = P.m.y (in build order): printed densities xi (design
// elements: smin(x, smax of the supports' xi in the layer below), kept-solid 1, else 0; x itself on
// the plate layer P.m.w and before) and per element its supports' largest xi m and r = Σ (xi / m)^P
// (`mr`). One thread per element of the layer (P.m.x of them). P.f = (P, 1/Q, ε, P/Q)
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> xi: array<f32>;
@group(0) @binding(4) var<storage, read_write> mr: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> page: array<u32>;

#include lookup NAME=fat PAGE=page

// The build axis b = P.n.w & 3, built towards -b when P.n.w & 4. Layers l in build order, each
// element in a layer at lateral (u, v) along the two other axes (in order).
fn lat_axes() -> vec2<u32> {
    let b = P.n.w & 3u;
    if b == 0u { return vec2<u32>(1u, 2u); }
    if b == 1u { return vec2<u32>(0u, 2u); }
    return vec2<u32>(0u, 1u);
}

fn lat_dims() -> vec2<i32> {
    let n = vec3<i32>(P.n.xyz);
    let a = lat_axes();
    return vec2<i32>(n[a.x], n[a.y]);
}

// the stored element at lateral c of layer l, or NONE
fn elem_at(c: vec2<i32>, l: u32) -> u32 {
    let n = vec3<i32>(P.n.xyz);
    let b = P.n.w & 3u;
    let a = lat_axes();
    var p = vec3<i32>(0, 0, 0);
    p[a.x] = c.x;
    p[a.y] = c.y;
    var w = i32(l);
    if (P.n.w & 4u) != 0u { w = n[b] - 1 - w; }
    p[b] = w;
    return fat(p, nbricks(P.n.xyz));
}

// xi at lateral c of layer l (0 when not stored)
fn xi_at(c: vec2<i32>, l: u32) -> f32 {
    let g = elem_at(c, l);
    if g == NONE { return 0.0; }
    return xi[g];
}

// support st (0..5: below, -u, +u, -v, +v) of lateral position c: its position, or x = -1. The
// mirror planes (P.m.z, bits x y z) on the lateral axes' low faces reflect it.
fn support(c: vec2<i32>, st: u32) -> vec2<i32> {
    var d = vec2<i32>(0, 0);
    switch st {
        case 1u: { d = vec2<i32>(-1, 0); }
        case 2u: { d = vec2<i32>(1, 0); }
        case 3u: { d = vec2<i32>(0, -1); }
        case 4u: { d = vec2<i32>(0, 1); }
        default: {}
    }
    let a = lat_axes();
    let nl = lat_dims();
    var q = c + d;
    if q.x < 0 && ((P.m.z >> a.x) & 1u) != 0u { q.x = -q.x - 1; }
    if q.y < 0 && ((P.m.z >> a.y) & 1u) != 0u { q.y = -q.y - 1; }
    if q.x < 0 || q.y < 0 || q.x >= nl.x || q.y >= nl.y { return vec2<i32>(-1, -1); }
    return q;
}

fn powp(v: f32, e: f32) -> f32 {
    return select(0.0, pow(v, e), v > 0.0);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let t = flat(wg, nwg, li);
    if t >= P.m.x { return; }
    let l = P.m.y;
    let nl = lat_dims();
    let c = vec2<i32>(i32(t) % nl.x, i32(t) / nl.x);
    let e = elem_at(c, l);
    if e == NONE { return; }
    let f = flags[e];
    var v = 0.0;
    var m = 0.0;
    var r = 0.0;
    if (f & 2u) != 0u {
        v = 1.0;
    } else if (f & 1u) != 0u {
        if l <= P.m.w {
            v = x[e];
        } else {
            for (var st = 0u; st < 5u; st++) {
                let q = support(c, st);
                if q.x >= 0 { m = max(m, xi_at(q, l - 1u)); }
            }
            var b = 0.0;
            if m > 0.0 {
                for (var st = 0u; st < 5u; st++) {
                    let q = support(c, st);
                    if q.x >= 0 { r += powp(xi_at(q, l - 1u) / m, P.f.x); }
                }
                b = pow(m, P.f.w) * pow(r, P.f.y);
            }
            let a = x[e];
            let eps = P.f.z;
            // ≥ 0 but for rounding (as on the CPU)
            v = max(0.5 * (a + b - sqrt((a - b) * (a - b) + eps) + sqrt(eps)), 0.0);
        }
    }
    xi[e] = v;
    mr[e] = vec2<f32>(m, r);
}

//// kernel am_adj bricks
// The AM filter's chain rule on layer l = P.m.y, run from the last layer back (overhang.rs's
// `sensitivities`): dc and dv hold df/dxi of the design elements and become df/dx divided by the
// filter weight sums (ready for the density filter's transpose, as `sens` leaves them without the
// AM filter). cc / cv carry each element's λ ∂ξ/∂smax (P/Q) smax / (m r) to the layer below,
// where a support multiplies it by (ξ / m)^(P-1) (its share of ∂smax). P as in `am_fwd`;
// P.g.z = rmin
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> xi: array<f32>;
@group(0) @binding(4) var<storage, read> mr: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> dc: array<f32>;
@group(0) @binding(6) var<storage, read_write> dv: array<f32>;
@group(0) @binding(7) var<storage, read_write> cc: array<f32>;
@group(0) @binding(8) var<storage, read_write> cv: array<f32>;
@group(0) @binding(9) var<storage, read> page: array<u32>;
@group(0) @binding(10) var<storage, read> wsum: array<f32>;

#include lookup NAME=fat PAGE=page

// The build axis b = P.n.w & 3, built towards -b when P.n.w & 4. Layers l in build order, each
// element in a layer at lateral (u, v) along the two other axes (in order).
fn lat_axes() -> vec2<u32> {
    let b = P.n.w & 3u;
    if b == 0u { return vec2<u32>(1u, 2u); }
    if b == 1u { return vec2<u32>(0u, 2u); }
    return vec2<u32>(0u, 1u);
}

fn lat_dims() -> vec2<i32> {
    let n = vec3<i32>(P.n.xyz);
    let a = lat_axes();
    return vec2<i32>(n[a.x], n[a.y]);
}

// the stored element at lateral c of layer l, or NONE
fn elem_at(c: vec2<i32>, l: u32) -> u32 {
    let n = vec3<i32>(P.n.xyz);
    let b = P.n.w & 3u;
    let a = lat_axes();
    var p = vec3<i32>(0, 0, 0);
    p[a.x] = c.x;
    p[a.y] = c.y;
    var w = i32(l);
    if (P.n.w & 4u) != 0u { w = n[b] - 1 - w; }
    p[b] = w;
    return fat(p, nbricks(P.n.xyz));
}

// xi at lateral c of layer l (0 when not stored)
fn xi_at(c: vec2<i32>, l: u32) -> f32 {
    let g = elem_at(c, l);
    if g == NONE { return 0.0; }
    return xi[g];
}

// support st (0..5: below, -u, +u, -v, +v) of lateral position c: its position, or x = -1. The
// mirror planes (P.m.z, bits x y z) on the lateral axes' low faces reflect it.
fn support(c: vec2<i32>, st: u32) -> vec2<i32> {
    var d = vec2<i32>(0, 0);
    switch st {
        case 1u: { d = vec2<i32>(-1, 0); }
        case 2u: { d = vec2<i32>(1, 0); }
        case 3u: { d = vec2<i32>(0, -1); }
        case 4u: { d = vec2<i32>(0, 1); }
        default: {}
    }
    let a = lat_axes();
    let nl = lat_dims();
    var q = c + d;
    if q.x < 0 && ((P.m.z >> a.x) & 1u) != 0u { q.x = -q.x - 1; }
    if q.y < 0 && ((P.m.z >> a.y) & 1u) != 0u { q.y = -q.y - 1; }
    if q.x < 0 || q.y < 0 || q.x >= nl.x || q.y >= nl.y { return vec2<i32>(-1, -1); }
    return q;
}

fn powp(v: f32, e: f32) -> f32 {
    return select(0.0, pow(v, e), v > 0.0);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let t = flat(wg, nwg, li);
    if t >= P.m.x { return; }
    let n = vec3<i32>(P.n.xyz);
    let l = P.m.y;
    let nl = lat_dims();
    let c = vec2<i32>(i32(t) % nl.x, i32(t) / nl.x);
    let e = elem_at(c, l);
    if e == NONE { return; }
    let f = flags[e];
    if (f & 4u) == 0u {
        dc[e] = 0.0;
        dv[e] = 0.0;
        cc[e] = 0.0;
        cv[e] = 0.0;
        return;
    }
    var lc = dc[e];
    var lv = dv[e];
    // the elements of the next layer whose supports include this one
    if i32(l) + 1 < n[P.n.w & 3u] {
        var gc = 0.0;
        var gv = 0.0;
        for (var nb = 0u; nb < 5u; nb++) {
            // the stencil is symmetric, so the positions whose supports include c are among its own
            // supports; on a mirror plane its own position appears twice there (and includes it
            // twice, counted by `mult`)
            let q = support(c, nb);
            if q.x < 0 || ((nb == 1u || nb == 3u) && all(q == support(c, 0u))) {
                continue;
            }
            var mult = 0.0;
            for (var st = 0u; st < 5u; st++) {
                if all(support(q, st) == c) { mult += 1.0; }
            }
            if mult == 0.0 { continue; }
            let a = elem_at(q, l + 1u);
            if a == NONE { continue; }
            let m = mr[a].x;
            if m <= 0.0 { continue; }
            let w = mult * powp(xi[e] / m, P.f.x - 1.0);
            gc += cc[a] * w;
            gv += cv[a] * w;
        }
        lc += gc;
        lv += gv;
    }
    var da = 1.0;
    var occ = 0.0;
    var ocv = 0.0;
    if l > P.m.w {
        let m = mr[e].x;
        let r = mr[e].y;
        var b = 0.0;
        if m > 0.0 { b = pow(m, P.f.w) * pow(r, P.f.y); }
        let a = x[e];
        let tt = (a - b) / sqrt((a - b) * (a - b) + P.f.z);
        da = 0.5 * (1.0 - tt);
        if m > 0.0 {
            let g = 0.5 * (1.0 + tt) * P.f.w * b / (m * r);
            occ = lc * g;
            ocv = lv * g;
        }
    }
    let hs = wsum[e];
    dc[e] = lc * da / hs;
    dv[e] = lv * da / hs;
    cc[e] = occ;
    cv[e] = ocv;
}
