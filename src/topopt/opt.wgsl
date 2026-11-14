// GPU kernels for the optimizer's per-element steps (see gpu_opt.rs; mirrors the CPU code in mod.rs).
// Uses the prelude of mg.wgsl. Element flags: bit 0 domain, bit 1 kept solid (passive), bit 2 design
// (domain and not passive). Elements are x-fastest on a grid of P.n elements.

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
// P.f = (E_min, E0, p, h), P.g.y = η, P.g.z = rmin, β = S[P.m.y]
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> xphys: array<f32>;
@group(0) @binding(3) var<storage, read> xtilde: array<f32>;
@group(0) @binding(4) var<storage, read> ce: array<f32>;
@group(0) @binding(5) var<storage, read> S: array<f32>;
@group(0) @binding(6) var<storage, read_write> dcs: array<f32>;
@group(0) @binding(7) var<storage, read_write> dvs: array<f32>;

// the filter's weight sum at element e (Σ rmin - distance over domain elements within rmin)
fn weight_sum(e: u32) -> f32 {
    let n = vec3<i32>(P.n.xyz);
    let ie = i32(e);
    let p = vec3<i32>(ie % n.x, (ie / n.x) % n.y, ie / (n.x * n.y));
    let rmin = P.g.z;
    let r = i32(ceil(rmin));
    var s = 0.0;
    let mir = P.m.z;
    for (var dk = -r; dk <= r; dk++) {
        var k = p.z + dk;
        if k < 0 && (mir & 4u) != 0u { k = -k - 1; }
        if k < 0 || k >= n.z { continue; }
        for (var dj = -r; dj <= r; dj++) {
            var j = p.y + dj;
            if j < 0 && (mir & 2u) != 0u { j = -j - 1; }
            if j < 0 || j >= n.y { continue; }
            for (var di = -r; di <= r; di++) {
                var i = p.x + di;
                if i < 0 && (mir & 1u) != 0u { i = -i - 1; }
                if i < 0 || i >= n.x { continue; }
                let w = rmin - sqrt(f32(di * di + dj * dj + dk * dk));
                if w <= 0.0 { continue; }
                if (flags[u32((k * n.y + j) * n.x + i)] & 1u) != 0u { s += w; }
            }
        }
    }
    return s;
}

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
        if P.m.w == 0u { hs = weight_sum(e); }
        dc = -P.f.z * xp * (P.f.y - P.f.x) * P.f.w * ce[e] * dproj / hs;
        dv = dproj / hs;
    }
    dcs[e] = dc;
    dvs[e] = dv;
}

//// kernel filter
// Cone filter over domain elements within P.f.x voxels (weights rmin - distance). P.m.y: 0 the raw
// weighted sum on domain elements; 1 densities: kept-solid 1, domain the weighted mean, else 0.
// P.m.z: axes (bits x, y, z) whose low face is a mirror plane: neighbours past it are the mirror
// images (index -i - 1). The same in `sens`'s weight sums.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    let n = vec3<i32>(P.n.xyz);
    if e >= u32(n.x * n.y * n.z) { return; }
    let f = flags[e];
    if (f & 1u) == 0u {
        outp[e] = 0.0;
        return;
    }
    if P.m.y == 1u && (f & 2u) != 0u {
        outp[e] = 1.0;
        return;
    }
    let ie = i32(e);
    let p = vec3<i32>(ie % n.x, (ie / n.x) % n.y, ie / (n.x * n.y));
    let rmin = P.f.x;
    let r = i32(ceil(rmin));
    var s = 0.0;
    var ws = 0.0;
    let mir = P.m.z;
    for (var dk = -r; dk <= r; dk++) {
        var k = p.z + dk;
        if k < 0 && (mir & 4u) != 0u { k = -k - 1; }
        if k < 0 || k >= n.z { continue; }
        for (var dj = -r; dj <= r; dj++) {
            var j = p.y + dj;
            if j < 0 && (mir & 2u) != 0u { j = -j - 1; }
            if j < 0 || j >= n.y { continue; }
            for (var di = -r; di <= r; di++) {
                var i = p.x + di;
                if i < 0 && (mir & 1u) != 0u { i = -i - 1; }
                if i < 0 || i >= n.x { continue; }
                let w = rmin - sqrt(f32(di * di + dj * dj + dk * dk));
                if w <= 0.0 { continue; }
                let m = u32((k * n.y + j) * n.x + i);
                if (flags[m] & 1u) != 0u {
                    s += w * inp[m];
                    ws += w;
                }
            }
        }
    }
    if P.m.y == 1u { s /= ws; }
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

//// kernel am_fwd
// The AM filter (overhang.rs) on layer k = P.m.y: printed densities xi (design elements:
// smin(x, smax of the supports' xi in the layer below), kept-solid 1, else 0; x itself on the plate
// layer P.m.w and below) and per element its supports' largest xi m and r = Σ (xi / m)^P (`mr`). One
// thread per element of the layer (P.m.x of them). P.m.z: mirror axes (bits x, y).
// P.f = (P, 1/Q, ε, P/Q)
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> xi: array<f32>;
@group(0) @binding(4) var<storage, read_write> mr: array<vec2<f32>>;

// support st (0..5: below, -x, +x, -y, +y) of column (i, j): its column in the layer, or -1
fn support(i: i32, j: i32, st: u32) -> i32 {
    let n = vec3<i32>(P.n.xyz);
    var d = vec2<i32>(0, 0);
    switch st {
        case 1u: { d = vec2<i32>(-1, 0); }
        case 2u: { d = vec2<i32>(1, 0); }
        case 3u: { d = vec2<i32>(0, -1); }
        case 4u: { d = vec2<i32>(0, 1); }
        default: {}
    }
    var a = i + d.x;
    var b = j + d.y;
    if a < 0 && (P.m.z & 1u) != 0u { a = -a - 1; }
    if b < 0 && (P.m.z & 2u) != 0u { b = -b - 1; }
    if a < 0 || b < 0 || a >= n.x || b >= n.y { return -1; }
    return b * n.x + a;
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
    let k = P.m.y;
    let e = k * P.m.x + t;
    let f = flags[e];
    var v = 0.0;
    var m = 0.0;
    var r = 0.0;
    if (f & 2u) != 0u {
        v = 1.0;
    } else if (f & 1u) != 0u {
        if k <= P.m.w {
            v = x[e];
        } else {
            let below = (k - 1u) * P.m.x;
            let i = i32(t) % n.x;
            let j = i32(t) / n.x;
            for (var st = 0u; st < 5u; st++) {
                let c = support(i, j, st);
                if c >= 0 { m = max(m, xi[below + u32(c)]); }
            }
            var b = 0.0;
            if m > 0.0 {
                for (var st = 0u; st < 5u; st++) {
                    let c = support(i, j, st);
                    if c >= 0 { r += powp(xi[below + u32(c)] / m, P.f.x); }
                }
                b = pow(m, P.f.w) * pow(r, P.f.y);
            }
            let a = x[e];
            let eps = P.f.z;
            v = 0.5 * (a + b - sqrt((a - b) * (a - b) + eps) + sqrt(eps));
        }
    }
    xi[e] = v;
    mr[e] = vec2<f32>(m, r);
}

//// kernel am_adj
// The AM filter's chain rule on layer k = P.m.y, run from the top layer down (overhang.rs's
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

fn support(i: i32, j: i32, st: u32) -> i32 {
    let n = vec3<i32>(P.n.xyz);
    var d = vec2<i32>(0, 0);
    switch st {
        case 1u: { d = vec2<i32>(-1, 0); }
        case 2u: { d = vec2<i32>(1, 0); }
        case 3u: { d = vec2<i32>(0, -1); }
        case 4u: { d = vec2<i32>(0, 1); }
        default: {}
    }
    var a = i + d.x;
    var b = j + d.y;
    if a < 0 && (P.m.z & 1u) != 0u { a = -a - 1; }
    if b < 0 && (P.m.z & 2u) != 0u { b = -b - 1; }
    if a < 0 || b < 0 || a >= n.x || b >= n.y { return -1; }
    return b * n.x + a;
}

fn powp(v: f32, e: f32) -> f32 {
    return select(0.0, pow(v, e), v > 0.0);
}

// the density filter's weight sum at element e (as in `sens`)
fn weight_sum(e: u32) -> f32 {
    let n = vec3<i32>(P.n.xyz);
    let ie = i32(e);
    let p = vec3<i32>(ie % n.x, (ie / n.x) % n.y, ie / (n.x * n.y));
    let rmin = P.g.z;
    let r = i32(ceil(rmin));
    var s = 0.0;
    let mir = P.m.z;
    for (var dk = -r; dk <= r; dk++) {
        var k = p.z + dk;
        if k < 0 && (mir & 4u) != 0u { k = -k - 1; }
        if k < 0 || k >= n.z { continue; }
        for (var dj = -r; dj <= r; dj++) {
            var j = p.y + dj;
            if j < 0 && (mir & 2u) != 0u { j = -j - 1; }
            if j < 0 || j >= n.y { continue; }
            for (var di = -r; di <= r; di++) {
                var i = p.x + di;
                if i < 0 && (mir & 1u) != 0u { i = -i - 1; }
                if i < 0 || i >= n.x { continue; }
                let w = rmin - sqrt(f32(di * di + dj * dj + dk * dk));
                if w <= 0.0 { continue; }
                if (flags[u32((k * n.y + j) * n.x + i)] & 1u) != 0u { s += w; }
            }
        }
    }
    return s;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let t = flat(wg, nwg, li);
    if t >= P.m.x { return; }
    let n = vec3<i32>(P.n.xyz);
    let k = P.m.y;
    let e = k * P.m.x + t;
    let f = flags[e];
    if (f & 4u) == 0u {
        dc[e] = 0.0;
        dv[e] = 0.0;
        cc[e] = 0.0;
        cv[e] = 0.0;
        return;
    }
    let i = i32(t) % n.x;
    let j = i32(t) / n.x;
    var lc = dc[e];
    var lv = dv[e];
    // the elements above whose supports include this one (mirrored ones may include it twice)
    if i32(k) + 1 < n.z {
        var gc = 0.0;
        var gv = 0.0;
        let above = (k + 1u) * P.m.x;
        for (var nb = 0u; nb < 5u; nb++) {
            // the stencil is symmetric, so the columns whose supports include (i, j) are the ones
            // among its own supports; on a mirror plane its own column appears twice there (and
            // includes it twice, counted by `mult`)
            let c = support(i, j, nb);
            if c < 0 || (nb == 1u && c == support(i, j, 0u)) || (nb == 3u && c == support(i, j, 0u)) {
                continue;
            }
            let ci = c % n.x;
            let cj = c / n.x;
            var mult = 0.0;
            for (var st = 0u; st < 5u; st++) {
                if support(ci, cj, st) == j * n.x + i { mult += 1.0; }
            }
            if mult == 0.0 { continue; }
            let a = above + u32(c);
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
    if k > P.m.w {
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
    let hs = weight_sum(e);
    dc[e] = lc * da / hs;
    dv[e] = lv * da / hs;
    cc[e] = occ;
    cv[e] = ocv;
}
