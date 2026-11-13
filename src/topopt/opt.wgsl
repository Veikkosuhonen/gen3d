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
// the filter weight sums (ready for the filter's transpose); 0 elsewhere.
// P.f = (E_min, E0, p, h), P.g.y = η, β = S[P.m.y]
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> xphys: array<f32>;
@group(0) @binding(3) var<storage, read> xtilde: array<f32>;
@group(0) @binding(4) var<storage, read> ce: array<f32>;
@group(0) @binding(5) var<storage, read> hs: array<f32>;
@group(0) @binding(6) var<storage, read> S: array<f32>;
@group(0) @binding(7) var<storage, read_write> dcs: array<f32>;
@group(0) @binding(8) var<storage, read_write> dvs: array<f32>;

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
        dc = -P.f.z * xp * (P.f.y - P.f.x) * P.f.w * ce[e] * dproj / hs[e];
        dv = dproj / hs[e];
    }
    dcs[e] = dc;
    dvs[e] = dv;
}

//// kernel filter
// Cone filter over domain elements within P.f.x voxels (weights rmin - distance). P.m.y: 0 the raw
// weighted sum on domain elements; 1 densities: kept-solid 1, domain sum / hs, else 0.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> hs: array<f32>;
@group(0) @binding(3) var<storage, read> inp: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

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
    for (var dk = -r; dk <= r; dk++) {
        let k = p.z + dk;
        if k < 0 || k >= n.z { continue; }
        for (var dj = -r; dj <= r; dj++) {
            let j = p.y + dj;
            if j < 0 || j >= n.y { continue; }
            for (var di = -r; di <= r; di++) {
                let i = p.x + di;
                if i < 0 || i >= n.x { continue; }
                let w = rmin - sqrt(f32(di * di + dj * dj + dk * dk));
                if w <= 0.0 { continue; }
                let m = u32((k * n.y + j) * n.x + i);
                if (flags[m] & 1u) != 0u { s += w * inp[m]; }
            }
        }
    }
    if P.m.y == 1u { s /= hs[e]; }
    outp[e] = s;
}

//// kernel oc
// Optimality-criteria update of the design elements for multiplier λ = S[P.m.y], with the compliance
// sensitivity scaled by 1 / |S[P.m.z]| and move limit P.f.x; other elements keep x.
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
    let cscale = 1.0 / max(abs(S[P.m.z]), 1e-30);
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
