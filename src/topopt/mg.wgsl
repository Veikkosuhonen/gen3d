// GPU kernels for multigrid-preconditioned CG (see gpu.rs and mg.rs; this mirrors the CPU path in
// f32). The file is split at the `//// kernel` lines; each kernel becomes its own module with the
// prelude in front, so each can number its bindings from 1. Binding 0 is always the parameters.
//
// Vectors hold 3 floats per node, nodes x-fastest. Grids are given as elements per axis (P.n.xyz);
// a grid of n elements has n + 1 nodes per axis.

//// prelude
struct Params {
    n: vec4<u32>,
    m: vec4<u32>,
    f: vec4<f32>,
    g: vec4<f32>,
}
@group(0) @binding(0) var<uniform> P: Params;

const WG: u32 = 128u;
const NONE: u32 = 0xffffffffu;

// thread index over a 2D dispatch of 1D workgroups
fn flat(wg: vec3<u32>, nwg: vec3<u32>, li: u32) -> u32 {
    return (wg.y * nwg.x + wg.x) * WG + li;
}

fn wg_flat(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.y * nwg.x + wg.x;
}

// node count of a grid of n elements
fn node_count(n: vec3<u32>) -> u32 {
    return (n.x + 1u) * (n.y + 1u) * (n.z + 1u);
}

// node index of local node b (bits x, y, z) of the element at (i, j, k)
fn elem_node(i: u32, j: u32, k: u32, b: u32, nx: u32, ny: u32) -> u32 {
    return ((k + ((b >> 2u) & 1u)) * ny + j + ((b >> 1u) & 1u)) * nx + i + (b & 1u);
}

// position of entry (r, c) of a symmetric 24×24 matrix stored as its upper triangle (300 entries)
fn sym(r: u32, c: u32) -> u32 {
    let a = min(r, c);
    let b = max(r, c);
    return a * (49u - a) / 2u + (b - a);
}

// trilinear weight of coarse local node b at fine local node a of child c
fn wgt(c: u32, a: u32, b: u32) -> f32 {
    var w = 1.0;
    for (var ax = 0u; ax < 3u; ax++) {
        let p = f32(((c >> ax) & 1u) + ((a >> ax) & 1u)) * 0.5;
        if ((b >> ax) & 1u) == 1u { w *= p; } else { w *= 1.0 - p; }
    }
    return w;
}

//// kernel apply_fine
// y = K u (y -= K u when P.m.x == 1) on the fine grid: per node, the rows of its up to 8 elements
// (E_e h · ke). Non-free nodes get 0.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> ke: array<f32>;
@group(0) @binding(4) var<storage, read> u: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    let n = P.n.xyz;
    if node >= node_count(n) { return; }
    let nx = n.x + 1u;
    let ny = n.y + 1u;
    var acc = vec3<f32>(0.0);
    if free[node] != 0u {
        let i = node % nx;
        let j = (node / nx) % ny;
        let k = node / (nx * ny);
        for (var a = 0u; a < 8u; a++) {
            let di = a & 1u;
            let dj = (a >> 1u) & 1u;
            let dk = (a >> 2u) & 1u;
            if i < di || j < dj || k < dk { continue; }
            let ei = i - di;
            let ej = j - dj;
            let ek = k - dk;
            if ei >= n.x || ej >= n.y || ek >= n.z { continue; }
            let s = ee[(ek * n.y + ej) * n.x + ei];
            if s == 0.0 { continue; }
            // ke annihilates translations, so apply it to displacements relative to the element's
            // first node: identical in exact arithmetic, but f32 rounding then scales with the
            // element's deformation rather than its (much larger) absolute displacement
            let m0 = elem_node(ei, ej, ek, 0u, nx, ny);
            let u0 = vec3<f32>(u[3u * m0], u[3u * m0 + 1u], u[3u * m0 + 2u]);
            for (var b = 1u; b < 8u; b++) {
                let m = elem_node(ei, ej, ek, b, nx, ny);
                let um = vec3<f32>(u[3u * m], u[3u * m + 1u], u[3u * m + 2u]) - u0;
                let r = (3u * a) * 24u + 3u * b;
                acc.x += s * dot(vec3<f32>(ke[r], ke[r + 1u], ke[r + 2u]), um);
                acc.y += s * dot(vec3<f32>(ke[r + 24u], ke[r + 25u], ke[r + 26u]), um);
                acc.z += s * dot(vec3<f32>(ke[r + 48u], ke[r + 49u], ke[r + 50u]), um);
            }
        }
    }
    // P.m.x == 1: y -= K u
    if P.m.x == 1u {
        y[3u * node] -= acc.x;
        y[3u * node + 1u] -= acc.y;
        y[3u * node + 2u] -= acc.z;
    } else {
        y[3u * node] = acc.x;
        y[3u * node + 1u] = acc.y;
        y[3u * node + 2u] = acc.z;
    }
}

//// kernel diag_fine
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> ke: array<f32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    let n = P.n.xyz;
    if node >= node_count(n) { return; }
    let nx = n.x + 1u;
    let ny = n.y + 1u;
    var acc = vec3<f32>(0.0);
    if free[node] != 0u {
        let i = node % nx;
        let j = (node / nx) % ny;
        let k = node / (nx * ny);
        for (var a = 0u; a < 8u; a++) {
            let di = a & 1u;
            let dj = (a >> 1u) & 1u;
            let dk = (a >> 2u) & 1u;
            if i < di || j < dj || k < dk { continue; }
            let ei = i - di;
            let ej = j - dj;
            let ek = k - dk;
            if ei >= n.x || ej >= n.y || ek >= n.z { continue; }
            let s = ee[(ek * n.y + ej) * n.x + ei];
            let r = (3u * a) * 25u;
            acc += s * vec3<f32>(ke[r], ke[r + 25u], ke[r + 50u]);
        }
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel diag_l1
// Level 1's diagonal from the fine `ee` and the Pᵀ (D ke D) P table (by clamped-node mask and child
// position). P.n = fine grid.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> T: array<f32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;

// clamped-node mask of fine element (i, j, k): bit b set when its local node b is not free
fn fine_mask(i: u32, j: u32, k: u32, nf: vec3<u32>) -> u32 {
    var m = 0u;
    for (var b = 0u; b < 8u; b++) {
        if free[elem_node(i, j, k, b, nf.x + 1u, nf.y + 1u)] == 0u { m |= 1u << b; }
    }
    return m;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    let nf = P.n.xyz;
    let n = (nf + vec3<u32>(1u)) / 2u;
    if node >= node_count(n) { return; }
    let nx = n.x + 1u;
    let ny = n.y + 1u;
    let i = node % nx;
    let j = (node / nx) % ny;
    let k = node / (nx * ny);
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let di = a & 1u;
        let dj = (a >> 1u) & 1u;
        let dk = (a >> 2u) & 1u;
        if i < di || j < dj || k < dk { continue; }
        let ei = i - di;
        let ej = j - dj;
        let ek = k - dk;
        if ei >= n.x || ej >= n.y || ek >= n.z { continue; }
        for (var c = 0u; c < 8u; c++) {
            let fi = 2u * ei + (c & 1u);
            let fj = 2u * ej + ((c >> 1u) & 1u);
            let fk = 2u * ek + ((c >> 2u) & 1u);
            if fi >= nf.x || fj >= nf.y || fk >= nf.z { continue; }
            let e = (fk * nf.y + fj) * nf.x + fi;
            let base = (fine_mask(fi, fj, fk, nf) * 8u + c) * 576u + (3u * a) * 25u;
            acc += ee[e] * vec3<f32>(T[base], T[base + 25u], T[base + 50u]);
        }
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel apply_elem
// y = K u on a level stored as symmetric 24×24 element matrices: upper triangles (`sym`),
// entry-major (mats[entry * elements + e]) so that neighbouring threads read neighbouring addresses.
@group(0) @binding(1) var<storage, read> mats: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    let n = P.n.xyz;
    if node >= node_count(n) { return; }
    let ne = n.x * n.y * n.z;
    let nx = n.x + 1u;
    let ny = n.y + 1u;
    let i = node % nx;
    let j = (node / nx) % ny;
    let k = node / (nx * ny);
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let di = a & 1u;
        let dj = (a >> 1u) & 1u;
        let dk = (a >> 2u) & 1u;
        if i < di || j < dj || k < dk { continue; }
        let ei = i - di;
        let ej = j - dj;
        let ek = k - dk;
        if ei >= n.x || ej >= n.y || ek >= n.z { continue; }
        let e = (ek * n.y + ej) * n.x + ei;
        for (var b = 0u; b < 8u; b++) {
            let m = elem_node(ei, ej, ek, b, nx, ny);
            let um = vec3<f32>(u[3u * m], u[3u * m + 1u], u[3u * m + 2u]);
            for (var i = 0u; i < 3u; i++) {
                let row = 3u * a + i;
                let row3 = vec3<f32>(
                    mats[sym(row, 3u * b) * ne + e],
                    mats[sym(row, 3u * b + 1u) * ne + e],
                    mats[sym(row, 3u * b + 2u) * ne + e]
                );
                acc[i] += dot(row3, um);
            }
        }
    }
    // P.m.x == 1: y -= K u
    if P.m.x == 1u {
        y[3u * node] -= acc.x;
        y[3u * node + 1u] -= acc.y;
        y[3u * node + 2u] -= acc.z;
    } else {
        y[3u * node] = acc.x;
        y[3u * node + 1u] = acc.y;
        y[3u * node + 2u] = acc.z;
    }
}

//// kernel diag_elem
@group(0) @binding(1) var<storage, read> mats: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    let n = P.n.xyz;
    if node >= node_count(n) { return; }
    let nx = n.x + 1u;
    let ny = n.y + 1u;
    let i = node % nx;
    let j = (node / nx) % ny;
    let k = node / (nx * ny);
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let di = a & 1u;
        let dj = (a >> 1u) & 1u;
        let dk = (a >> 2u) & 1u;
        if i < di || j < dj || k < dk { continue; }
        let ei = i - di;
        let ej = j - dj;
        let ek = k - dk;
        if ei >= n.x || ej >= n.y || ek >= n.z { continue; }
        let ne = n.x * n.y * n.z;
        let e = (ek * n.y + ej) * n.x + ei;
        let r = 3u * a;
        acc += vec3<f32>(mats[sym(r, r) * ne + e], mats[sym(r + 1u, r + 1u) * ne + e], mats[sym(r + 2u, r + 2u) * ne + e]);
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel assemble
// Galerkin coarsening, one workgroup per coarse element: out_E = Σ_c P_cᵀ K_c P_c (stored
// entry-major). With P.m.x == 1 the
// children are level-1 elements computed on the fly from the fine grid (P.n = fine grid, the output
// is level 2); otherwise they are stored matrices (P.n = the child level's grid).
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> T: array<f32>;
@group(0) @binding(4) var<storage, read> child: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

// clamped-node mask of fine element (i, j, k): bit b set when its local node b is not free
fn fine_mask(i: u32, j: u32, k: u32, nf: vec3<u32>) -> u32 {
    var m = 0u;
    for (var b = 0u; b < 8u; b++) {
        if free[elem_node(i, j, k, b, nf.x + 1u, nf.y + 1u)] == 0u { m |= 1u << b; }
    }
    return m;
}

var<workgroup> K: array<f32, 576>;
var<workgroup> KP: array<f32, 576>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let ce = wg_flat(wg, nwg);
    let virt = P.m.x == 1u;
    let nf = P.n.xyz;
    // the child level's grid, and the output's
    var nl = nf;
    if virt { nl = (nf + vec3<u32>(1u)) / 2u; }
    let nc = (nl + vec3<u32>(1u)) / 2u;
    if ce >= nc.x * nc.y * nc.z { return; }
    let ci = ce % nc.x;
    let cj = (ce / nc.x) % nc.y;
    let ck = ce / (nc.x * nc.y);
    var acc: array<f32, 9>;
    for (var t = 0u; t < 9u; t++) { acc[t] = 0.0; }
    for (var c2 = 0u; c2 < 8u; c2++) {
        let li_ = 2u * ci + (c2 & 1u);
        let lj = 2u * cj + ((c2 >> 1u) & 1u);
        let lk = 2u * ck + ((c2 >> 2u) & 1u);
        let inside = li_ < nl.x && lj < nl.y && lk < nl.z;
        // the child's matrix
        for (var t = 0u; t < 9u; t++) {
            let idx = li + 64u * t;
            var s = 0.0;
            if inside {
                if virt {
                    for (var c1 = 0u; c1 < 8u; c1++) {
                        let fi = 2u * li_ + (c1 & 1u);
                        let fj = 2u * lj + ((c1 >> 1u) & 1u);
                        let fk = 2u * lk + ((c1 >> 2u) & 1u);
                        if fi >= nf.x || fj >= nf.y || fk >= nf.z { continue; }
                        let e = (fk * nf.y + fj) * nf.x + fi;
                        let w = ee[e];
                        if w != 0.0 {
                            s += w * T[(fine_mask(fi, fj, fk, nf) * 8u + c1) * 576u + idx];
                        }
                    }
                } else {
                    s = child[sym(idx / 24u, idx % 24u) * (nl.x * nl.y * nl.z) + (lk * nl.y + lj) * nl.x + li_];
                }
            }
            K[idx] = s;
        }
        workgroupBarrier();
        // KP = K P
        for (var t = 0u; t < 9u; t++) {
            let idx = li + 64u * t;
            let r = idx / 24u;
            let col = idx % 24u;
            let b = col / 3u;
            let jj = col % 3u;
            var s = 0.0;
            for (var a = 0u; a < 8u; a++) {
                s += K[r * 24u + 3u * a + jj] * wgt(c2, a, b);
            }
            KP[idx] = s;
        }
        workgroupBarrier();
        // acc += Pᵀ KP
        for (var t = 0u; t < 9u; t++) {
            let idx = li + 64u * t;
            let row = idx / 24u;
            let col = idx % 24u;
            let b = row / 3u;
            let ii = row % 3u;
            var s = 0.0;
            for (var a = 0u; a < 8u; a++) {
                s += wgt(c2, a, b) * KP[(3u * a + ii) * 24u + col];
            }
            acc[t] += s;
        }
        workgroupBarrier();
    }
    // upper triangle, entry-major, like apply_elem reads it
    let nce = nc.x * nc.y * nc.z;
    for (var t = 0u; t < 9u; t++) {
        let idx = li + 64u * t;
        let row = idx / 24u;
        let col = idx % 24u;
        if row <= col {
            out[sym(row, col) * nce + ce] = acc[t];
        }
    }
}

//// kernel restrict
// rc = Pᵀ r from the grid P.n to its coarse grid.
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> rc: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let m = flat(wg, nwg, li);
    let nf = P.n.xyz;
    let nc = (nf + vec3<u32>(1u)) / 2u;
    if m >= node_count(nc) { return; }
    let f = vec3<i32>(nf + vec3<u32>(1u));
    let cx = nc.x + 1u;
    let cy = nc.y + 1u;
    let ci = i32(m % cx);
    let cj = i32((m / cx) % cy);
    let ck = i32(m / (cx * cy));
    var acc = vec3<f32>(0.0);
    for (var dk = -1; dk <= 1; dk++) {
        let k = 2 * ck + dk;
        if k < 0 || k >= f.z { continue; }
        for (var dj = -1; dj <= 1; dj++) {
            let j = 2 * cj + dj;
            if j < 0 || j >= f.y { continue; }
            for (var di = -1; di <= 1; di++) {
                let i = 2 * ci + di;
                if i < 0 || i >= f.x { continue; }
                let w = select(0.5, 1.0, di == 0) * select(0.5, 1.0, dj == 0) * select(0.5, 1.0, dk == 0);
                let q = 3u * u32((k * f.y + j) * f.x + i);
                acc += w * vec3<f32>(r[q], r[q + 1u], r[q + 2u]);
            }
        }
    }
    // P.m.x == 1: rc -= Pᵀ r
    if P.m.x == 1u {
        rc[3u * m] -= acc.x;
        rc[3u * m + 1u] -= acc.y;
        rc[3u * m + 2u] -= acc.z;
    } else {
        rc[3u * m] = acc.x;
        rc[3u * m + 1u] = acc.y;
        rc[3u * m + 2u] = acc.z;
    }
}

//// kernel prolong
// x += P xc on the grid P.n (x = P xc when P.m.x == 1), kept to DOFs with a positive diagonal.
@group(0) @binding(1) var<storage, read> xc: array<f32>;
@group(0) @binding(2) var<storage, read> diag: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let m = flat(wg, nwg, li);
    let nf = P.n.xyz;
    if m >= node_count(nf) { return; }
    let nc = (nf + vec3<u32>(1u)) / 2u;
    let fx = nf.x + 1u;
    let fy = nf.y + 1u;
    let cx = nc.x + 1u;
    let cy = nc.y + 1u;
    let p = vec3<u32>(m % fx, (m / fx) % fy, m / (fx * fy));
    var acc = vec3<f32>(0.0);
    for (var c = 0u; c < 8u; c++) {
        var w = 1.0;
        var q = vec3<u32>(0u);
        for (var ax = 0u; ax < 3u; ax++) {
            let v = p[ax];
            let hi = (c >> ax) & 1u;
            if v % 2u == 0u {
                if hi == 1u { w = 0.0; }
                q[ax] = v / 2u;
            } else {
                w *= 0.5;
                q[ax] = v / 2u + hi;
            }
        }
        if w == 0.0 { continue; }
        let o = 3u * ((q.z * cy + q.y) * cx + q.x);
        acc += w * vec3<f32>(xc[o], xc[o + 1u], xc[o + 2u]);
    }
    for (var d = 0u; d < 3u; d++) {
        let keep = diag[3u * m + d] > 0.0;
        if P.m.x == 1u {
            x[3u * m + d] = select(0.0, acc[d], keep);
        } else if keep {
            x[3u * m + d] += acc[d];
        }
    }
}

//// kernel lin
// y = a y + c x', with x' = x, or x / w where w > 0 (else 0) when P.m.w == 1. a = P.f.x (× S[P.m.y]),
// c = P.f.y (× S[P.m.z]). a == 0 overwrites y without reading it.
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let i = flat(wg, nwg, li);
    if i >= P.m.x { return; }
    var a = P.f.x;
    if P.m.y != NONE { a *= S[P.m.y]; }
    var c = P.f.y;
    if P.m.z != NONE { c *= S[P.m.z]; }
    var xv = x[i];
    if P.m.w == 1u {
        let d = w[i];
        xv = select(0.0, xv / d, d > 0.0);
    }
    if a == 0.0 {
        y[i] = c * xv;
    } else {
        y[i] = a * y[i] + c * xv;
    }
}

//// kernel dot_partial
// partials[g] = a reduction over a grid-stride share of the P.m.x entries. P.m.y: 0 Σ a b, 1 Σ a b w,
// 2 max |a - b| where w > 0, 3 count of 0.1 < a < 0.9 where w > 0, 4 max |a_i| of 3-vectors (P.m.x =
// vectors).
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let n = P.m.x;
    let mode = P.m.y;
    let is_max = mode == 2u || mode == 4u;
    var s = 0.0;
    for (var i = wg.x * WG + li; i < n; i += nwg.x * WG) {
        switch mode {
            case 0u: { s += a[i] * b[i]; }
            case 1u: { s += a[i] * b[i] * w[i]; }
            case 2u: { if w[i] > 0.0 { s = max(s, abs(a[i] - b[i])); } }
            case 3u: { if w[i] > 0.0 && a[i] > 0.1 && a[i] < 0.9 { s += 1.0; } }
            default: { s = max(s, length(vec3<f32>(a[3u * i], a[3u * i + 1u], a[3u * i + 2u]))); }
        }
    }
    sh[li] = s;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) {
        if li < k {
            if is_max { sh[li] = max(sh[li], sh[li + k]); } else { sh[li] += sh[li + k]; }
        }
        workgroupBarrier();
    }
    if li == 0u { partials[wg.x] = sh[0]; }
}

//// kernel dot_final
// S[P.m.y] = Σ (or max, when P.m.z == 1) of partials[0 .. P.m.x]
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32) {
    let is_max = P.m.z == 1u;
    var s = 0.0;
    for (var i = li; i < P.m.x; i += WG) {
        if is_max { s = max(s, partials[i]); } else { s += partials[i]; }
    }
    sh[li] = s;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) {
        if li < k {
            if is_max { sh[li] = max(sh[li], sh[li + k]); } else { sh[li] += sh[li + k]; }
        }
        workgroupBarrier();
    }
    if li == 0u { S[P.m.y] = sh[0]; }
}

//// kernel scalar
// One scalar operation on S: op P.m.x on S[P.m.y], S[P.m.z] into S[P.m.w].
@group(0) @binding(1) var<storage, read_write> S: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let a = S[P.m.y];
    let b = S[P.m.z];
    var r = 0.0;
    switch P.m.x {
        case 0u: { r = select(0.0, a / b, b != 0.0); }    // a / b
        case 1u: { r = a; }                                // copy
        case 2u: { r = select(0.0, inverseSqrt(a), a > 0.0); }
        default: { r = select(0.0, 1.0 / a, a != 0.0); }  // 1 / a
    }
    S[P.m.w] = r;
}

//// kernel fill_random
// v = pseudo-random in [-0.5, 0.5) where diag > 0, else 0 (P.m.x entries)
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let i = flat(wg, nwg, li);
    if i >= P.m.x { return; }
    var h = i * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    v[i] = select(0.0, f32(h >> 8u) / 16777216.0 - 0.5, diag[i] > 0.0);
}

//// kernel bottom
// x = A⁻¹ b with a dense inverse (P.m.x rows)
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let r = flat(wg, nwg, li);
    let n = P.m.x;
    if r >= n { return; }
    var s = 0.0;
    for (var j = 0u; j < n; j++) { s += inv[r * n + j] * b[j]; }
    x[r] = s;
}

//// kernel energy
// ce = u_eᵀ ke u_e per fine element with stiffness (added to ce when P.m.x == 1)
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> ke: array<f32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> ce: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let e = flat(wg, nwg, li);
    let n = P.n.xyz;
    if e >= n.x * n.y * n.z { return; }
    var s = 0.0;
    if ee[e] != 0.0 {
        let nx = n.x + 1u;
        let ny = n.y + 1u;
        let i = e % n.x;
        let j = (e / n.x) % n.y;
        let k = e / (n.x * n.y);
        // relative to the first node, as in apply_fine
        var ue: array<f32, 24>;
        let m0 = elem_node(i, j, k, 0u, nx, ny);
        for (var b = 0u; b < 8u; b++) {
            let m = elem_node(i, j, k, b, nx, ny);
            ue[3u * b] = u[3u * m] - u[3u * m0];
            ue[3u * b + 1u] = u[3u * m + 1u] - u[3u * m0 + 1u];
            ue[3u * b + 2u] = u[3u * m + 2u] - u[3u * m0 + 2u];
        }
        for (var r = 0u; r < 24u; r++) {
            var row = 0.0;
            for (var q = 0u; q < 24u; q++) { row += ke[r * 24u + q] * ue[q]; }
            s += ue[r] * row;
        }
    }
    if P.m.x == 1u { ce[e] += s; } else { ce[e] = s; }
}

//// kernel scatter_add
// y[idx[i]] += val[i] for the P.m.x entries (indices distinct)
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> val: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let i = flat(wg, nwg, li);
    if i >= P.m.x { return; }
    y[idx[i]] += val[i];
}

//// kernel reduce_design
// Like dot_partial, over design elements only (bit 2 of the element flags). P.m.y: 0 Σ a, 2 max
// |a - b|, 3 count of 0.1 < a < 0.9.
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let n = P.m.x;
    let mode = P.m.y;
    let is_max = mode == 2u;
    var s = 0.0;
    for (var i = wg.x * WG + li; i < n; i += nwg.x * WG) {
        if (flags[i] & 4u) == 0u { continue; }
        switch mode {
            case 0u: { s += a[i]; }
            case 2u: { s = max(s, abs(a[i] - b[i])); }
            default: { if a[i] > 0.1 && a[i] < 0.9 { s += 1.0; } }
        }
    }
    sh[li] = s;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) {
        if li < k {
            if is_max { sh[li] = max(sh[li], sh[li + k]); } else { sh[li] += sh[li + k]; }
        }
        workgroupBarrier();
    }
    if li == 0u { partials[wg.x] = sh[0]; }
}
