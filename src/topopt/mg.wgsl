// GPU kernels for multigrid-preconditioned CG (see gpu.rs and mg.rs; this mirrors the CPU path in
// f32). The file is split at the `//// kernel` lines; each kernel becomes its own module with the
// prelude in front, so each can number its bindings from 1. Binding 0 is always the parameters.
//
// Vectors hold 3 floats per node, in the level's brick layout (see "Block-sparse grids" below).
// Grids are given as elements per axis (P.n.xyz); a grid of n elements has n + 1 nodes per axis.

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

// a node's free axes (bits) as a 0/1 mask per DOF
fn free3(f: u32) -> vec3<f32> {
    return vec3<f32>(f32(f & 1u), f32((f >> 1u) & 1u), f32((f >> 2u) & 1u));
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

// Block-sparse grids (bricks.rs). A level's node grid is tiled by bricks of BX×BY×BZ = BV points
// (4³, or 2 across a slab's thin axis: kernels tagged `bricks` are compiled per shape,
// gpu_engine::BRICK_SHAPES); element (i, j, k) is in node (i, j, k)'s brick. Only stored bricks have
// values: point (slot, local) at slot · BV + local, local x-fastest, and points past the grid in a
// stored brick hold 0. A level's page table maps its box bricks (x-fastest, `nbricks` per axis) to
// slots (NONE: absent, reads as 0); `coords` holds each slot's brick coordinates. Brick kernels get
// the number of slots they run over in P.n.w.
const BX: u32 = 4u;
const BY: u32 = 4u;
const BZ: u32 = 4u;
const BV: u32 = 64u;
// workgroup tile sizes, set with the shape (workgroup memory limits how many workgroups a GPU core
// runs at once: tiles sized for the largest shape made the fine stencils 20% slower): a brick with
// a one-point halo (B + 2)³, with a halo on the − sides (B + 1)³, and the coarse points under a
// halo'd brick (B / 2 + 2)³
const HN: u32 = 216u;
const EN: u32 = 125u;
const CN: u32 = 64u;

fn bshape() -> vec3<u32> {
    return vec3<u32>(BX, BY, BZ);
}

// bricks per axis over the nodes of a grid of n elements
fn nbricks(n: vec3<u32>) -> vec3<u32> {
    return (n + bshape()) / bshape();
}

// point i of a box of s points (x fastest), and back
fn unflat(i: u32, s: vec3<u32>) -> vec3<u32> {
    return vec3<u32>(i % s.x, (i / s.x) % s.y, i / (s.x * s.y));
}

fn flat3(q: vec3<u32>, s: vec3<u32>) -> u32 {
    return (q.z * s.y + q.y) * s.x + q.x;
}

// index of point p (≥ 0) in slot s
fn in_slot(s: u32, p: vec3<i32>) -> u32 {
    return s * BV + flat3(vec3<u32>(p) % bshape(), bshape());
}

// the box brick holding point p of a grid of nb bricks per axis, or NONE outside it
fn box_brick(p: vec3<i32>, nb: vec3<u32>) -> u32 {
    if any(p < vec3<i32>(0)) { return NONE; }
    let b = vec3<u32>(p) / bshape();
    if any(b >= nb) { return NONE; }
    return flat3(b, nb);
}

//// lib lookup
// sparse index of point p on the grid with page table $PAGE (nb bricks per axis), or NONE
fn $NAME(p: vec3<i32>, nb: vec3<u32>) -> u32 {
    let b = box_brick(p, nb);
    if b == NONE { return NONE; }
    let s = $PAGE[b];
    if s == NONE { return NONE; }
    return in_slot(s, p);
}

//// lib ke_uniform
// The 24×24 element matrix at binding $BIND (`#include ke BIND=n`), as a uniform or a storage buffer
// (`Tuning::ke_uniform`). Uniform data comes from the constant cache, and once the loops unroll its
// offsets are constants (on NVIDIA, operands of the FMAs themselves) instead of one load
// instruction per entry. `kv(i)` is entry i (row-major).
@group(0) @binding($BIND) var<uniform> ke: array<vec4<f32>, 144>;

fn kv(i: u32) -> f32 {
    return ke[i >> 2u][i & 3u];
}

//// lib ke_storage
@group(0) @binding($BIND) var<storage, read> ke: array<f32>;

fn kv(i: u32) -> f32 {
    return ke[i];
}

//// lib nbhd
// The slots of the 3×3×3 bricks around brick b0 of the grid with page table $PAGE, in workgroup
// memory: every thread calls `$NAME_load` (then a barrier); `$NAME_at(q)` is then the sparse index of
// point q, given relative to b0's first point (each axis in -B..2B-1), or NONE.
var<workgroup> $NAME: array<u32, 27>;

fn $NAME_at(q: vec3<i32>) -> u32 {
    // (unsigned: power-of-two division and remainder are shifts and masks)
    let qu = vec3<u32>(q + vec3<i32>(bshape()));
    let bo = qu / bshape();
    let s = $NAME[bo.x + 3u * bo.y + 9u * bo.z];
    if s == NONE { return NONE; }
    return s * BV + flat3(qu % bshape(), bshape());
}

fn $NAME_load(b0: vec3<u32>, nb: vec3<u32>, li: u32) {
    if li < 27u {
        let b = vec3<i32>(b0) + vec3<i32>(unflat(li, vec3<u32>(3u))) - vec3<i32>(1);
        var s = NONE;
        if all(b >= vec3<i32>(0)) && all(vec3<u32>(b) < nb) { s = $PAGE[flat3(vec3<u32>(b), nb)]; }
        $NAME[li] = s;
    }
}

//// lib fine_stage
// The fine stencil on one brick (with `nbhd` nf loaded for it), from workgroup tiles: su, $U
// (3-vectors) on the brick and a one-node halo ((B + 2)³ points), and se, `ee` on the brick and a
// one-element halo on its − sides ((B + 1)³), which `stage_u` and `stage_ee` load (every thread).
// (3-vectors as three arrays: a vec3 takes 16 bytes in workgroup memory)
var<workgroup> sux: array<f32, HN>;
var<workgroup> suy: array<f32, HN>;
var<workgroup> suz: array<f32, HN>;
var<workgroup> se: array<f32, EN>;

fn su(i: u32) -> vec3<f32> {
    return vec3<f32>(sux[i], suy[i], suz[i]);
}

fn su_set(i: u32, v: vec3<f32>) {
    sux[i] = v.x;
    suy[i] = v.y;
    suz[i] = v.z;
}

fn hshape() -> vec3<u32> {
    return bshape() + vec3<u32>(2u);
}

fn eshape() -> vec3<u32> {
    return bshape() + vec3<u32>(1u);
}

fn stage_u(li: u32) {
    let hs = hshape();
    for (var s = li; s < hs.x * hs.y * hs.z; s += BV) {
        let g = nf_at(vec3<i32>(unflat(s, hs)) - vec3<i32>(1));
        var v = vec3<f32>(0.0);
        if g != NONE { v = vec3<f32>($U[3u * g], $U[3u * g + 1u], $U[3u * g + 2u]); }
        su_set(s, v);
    }
}

fn stage_ee(li: u32) {
    let es = eshape();
    for (var s = li; s < es.x * es.y * es.z; s += BV) {
        let g = nf_at(vec3<i32>(unflat(s, es)) - vec3<i32>(1));
        var v = 0.0;
        if g != NONE { v = ee[g]; }
        se[s] = v;
    }
}

struct Kd {
    ku: vec3<f32>,
    dg: vec3<f32>,
}

// (K u) and diag(K) at local node li, over its up to 8 elements (E_e h · ke). K u works on the
// displacements relative to each element's first node: ke annihilates translations, so it is
// identical in exact arithmetic, but f32 rounding then scales with the element's deformation rather
// than its (much larger) absolute displacement.
fn stencil(li: u32, with_u: bool, with_dg: bool) -> Kd {
    let hs = hshape();
    let es = eshape();
    let l = unflat(li, bshape()) + vec3<u32>(1u);
    let h = flat3(l, hs);
    let he = flat3(l, es);
    let sy = hs.x;
    let sz = hs.x * hs.y;
    let ey = es.x;
    let ez = es.x * es.y;
    var r: Kd;
    r.ku = vec3<f32>(0.0);
    r.dg = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let di = a & 1u;
        let dj = (a >> 1u) & 1u;
        let dk = (a >> 2u) & 1u;
        let s = se[he - di - ey * dj - ez * dk];
        if s == 0.0 { continue; }
        if with_dg {
            let rd = (3u * a) * 25u;
            r.dg += s * vec3<f32>(kv(rd), kv(rd + 25u), kv(rd + 50u));
        }
        if !with_u { continue; }
        let base = h - di - sy * dj - sz * dk;
        let u0 = su(base);
        for (var c = 1u; c < 8u; c++) {
            let um = su(base + (c & 1u) + sy * ((c >> 1u) & 1u) + sz * ((c >> 2u) & 1u)) - u0;
            let rr = (3u * a) * 24u + 3u * c;
            r.ku.x += s * dot(vec3<f32>(kv(rr), kv(rr + 1u), kv(rr + 2u)), um);
            r.ku.y += s * dot(vec3<f32>(kv(rr + 24u), kv(rr + 25u), kv(rr + 26u)), um);
            r.ku.z += s * dot(vec3<f32>(kv(rr + 48u), kv(rr + 49u), kv(rr + 50u)), um);
        }
    }
    return r;
}

//// lib fine_free
// Fine element f's node masks (with `lookup` fat on the fine grid's page table).
// clamped-node mask: bit b set when its local node b is not free on all axes (the table's index;
// elements with partly free nodes don't use the table, see `fine_free`)
fn fine_mask(f: vec3<i32>, nb: vec3<u32>) -> u32 {
    var m = 0u;
    for (var b = 0u; b < 8u; b++) {
        let g = fat(f + vec3<i32>(vec3<u32>(b & 1u, (b >> 1u) & 1u, (b >> 2u) & 1u)), nb);
        var fv = 0u;
        if g != NONE { fv = free[g]; }
        if fv != 7u { m |= 1u << b; }
    }
    return m;
}

// the free axes of its 8 nodes, 3 bits each (node b at bits 3b..3b+2), with bit 31 set when one of
// them is partly free (a roller, or a symmetry plane): the table of masked element products only
// knows fully clamped nodes, so such an element's level-1 matrix is computed directly with per-axis
// masks
fn fine_free(f: vec3<i32>, nb: vec3<u32>) -> u32 {
    var m = 0u;
    var partial = false;
    for (var b = 0u; b < 8u; b++) {
        let g = fat(f + vec3<i32>(vec3<u32>(b & 1u, (b >> 1u) & 1u, (b >> 2u) & 1u)), nb);
        var fv = 0u;
        if g != NONE { fv = free[g]; }
        m |= fv << (3u * b);
        if fv != 0u && fv != 7u { partial = true; }
    }
    if partial { m |= 1u << 31u; }
    return m;
}

//// kernel apply_fine bricks
// y = K u (y -= K u when P.m.x == 1) on the fine grid: per node, the rows of its up to 8 elements
// (E_e h · ke). Clamped DOFs (`free` holds a node's free axes as bits) get 0. One workgroup per
// stored brick, from tiles in workgroup memory (`fine_stage`).
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
#include ke BIND=3
@group(0) @binding(4) var<storage, read> u: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> page: array<u32>;
@group(0) @binding(7) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nf PAGE=page
#include fine_stage U=u

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nf_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    stage_u(li);
    stage_ee(li);
    workgroupBarrier();
    let node = slot * BV + li;
    let fm = free[node];
    var acc = vec3<f32>(0.0);
    if fm != 0u {
        acc = stencil(li, true, false).ku * free3(fm);
    }
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

//// kernel diag_fine bricks
// diag(K) on the fine grid (0 on clamped DOFs), as `apply_fine`.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
#include ke BIND=3
@group(0) @binding(4) var<storage, read_write> d: array<f32>;
@group(0) @binding(5) var<storage, read> page: array<u32>;
@group(0) @binding(6) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nf PAGE=page
// (no displacements: `stage_u` isn't used)
#include fine_stage U=ee

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nf_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    stage_ee(li);
    workgroupBarrier();
    let node = slot * BV + li;
    let fm = free[node];
    var acc = vec3<f32>(0.0);
    if fm != 0u {
        acc = stencil(li, false, true).dg * free3(fm);
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel cheb_fine bricks
// One smoother step on the fine grid with neither its diagonal nor its residual stored (each would
// be a fine vector): d = α d + γ D⁻¹ (b − K x) on free DOFs (0 elsewhere), D = diag(K) worked out
// here from `ee`, in the loop that forms K x. Recomputing b − K x from x costs the same mat-vec as
// updating a stored residual. α = P.f.x (0: d isn't read), γ = P.f.y (× S[P.m.z] unless NONE).
// P.m.x == 1: x is zero (not read). P.m.y == 1: without D⁻¹ (d = α d + γ (b − K x)).
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> ee: array<f32>;
@group(0) @binding(3) var<storage, read> free: array<u32>;
#include ke BIND=4
@group(0) @binding(5) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read> b: array<f32>;
@group(0) @binding(7) var<storage, read_write> d: array<f32>;
@group(0) @binding(8) var<storage, read> page: array<u32>;
@group(0) @binding(9) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nf PAGE=page
#include fine_stage U=x

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    let zero_x = P.m.x == 1u;
    nf_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    if !zero_x { stage_u(li); }
    stage_ee(li);
    workgroupBarrier();
    let node = slot * BV + li;
    let fm = free[node];
    if fm == 0u {
        d[3u * node] = 0.0;
        d[3u * node + 1u] = 0.0;
        d[3u * node + 2u] = 0.0;
        return;
    }
    let k = stencil(li, !zero_x, P.m.y == 0u);
    var res = vec3<f32>(b[3u * node], b[3u * node + 1u], b[3u * node + 2u]) - k.ku;
    if P.m.y == 0u {
        res = select(vec3<f32>(0.0), res / k.dg, k.dg > vec3<f32>(0.0));
    }
    res *= free3(fm);
    var g = P.f.y;
    if P.m.z != NONE { g *= S[P.m.z]; }
    let al = P.f.x;
    for (var c = 0u; c < 3u; c++) {
        if al == 0.0 {
            d[3u * node + c] = g * res[c];
        } else {
            d[3u * node + c] = al * d[3u * node + c] + g * res[c];
        }
    }
}

//// kernel apply_p bricks
// y = D K D P uc on the fine grid P.n, uc on its coarse grid: with `restrict` after it, level 1's
// mat-vec Pᵀ D K D P without storing P uc (a fine vector). One workgroup per stored fine brick: it
// loads the coarse nodes under the brick and its one-node halo (B / 2 + 2 per axis), interpolates
// D P uc on the brick and halo from them, then each thread applies K at its node as `apply_fine`.
// D keeps the free DOFs.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
#include ke BIND=3
@group(0) @binding(4) var<storage, read> uc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> page: array<u32>;
@group(0) @binding(7) var<storage, read> coords: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> pagec: array<u32>;

#include nbhd NAME=nf PAGE=page
#include nbhd NAME=nc PAGE=pagec
// (su is filled here, by interpolation: `stage_u` isn't used)
#include fine_stage U=uc

// the coarse nodes under the tile (B / 2 + 2 per axis)
var<workgroup> scx: array<f32, CN>;
var<workgroup> scy: array<f32, CN>;
var<workgroup> scz: array<f32, CN>;

fn sc(i: i32) -> vec3<f32> {
    return vec3<f32>(scx[i], scy[i], scz[i]);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    let n = P.n.xyz;
    let b0 = coords[slot].xyz;
    let o = b0 * bshape();
    let cb0 = (o / 2u) / bshape();
    nf_load(b0, nbricks(n), li);
    nc_load(cb0, nbricks((n + vec3<u32>(1u)) / 2u), li);
    workgroupBarrier();
    // coarse nodes c0 .. c0 + cs - 1, c0 = o / 2 - 1: the fine halo o - 1 .. o + B lies over them
    let cs = bshape() / 2u + vec3<u32>(2u);
    let c0 = vec3<i32>(o / 2u) - vec3<i32>(1);
    let crel = c0 - vec3<i32>(cb0 * bshape());
    for (var s = li; s < cs.x * cs.y * cs.z; s += BV) {
        let g = nc_at(crel + vec3<i32>(unflat(s, cs)));
        var v = vec3<f32>(0.0);
        if g != NONE { v = vec3<f32>(uc[3u * g], uc[3u * g + 1u], uc[3u * g + 2u]); }
        scx[s] = v.x;
        scy[s] = v.y;
        scz[s] = v.z;
    }
    stage_ee(li);
    workgroupBarrier();
    let hs = hshape();
    for (var s = li; s < hs.x * hs.y * hs.z; s += BV) {
        let qr = vec3<i32>(unflat(s, hs)) - vec3<i32>(1);
        let g = nf_at(qr);
        var v = vec3<f32>(0.0);
        if g != NONE {
            let fq = free[g];
            if fq != 0u {
                let qu = vec3<u32>(vec3<i32>(o) + qr);
                // per axis: the coarse nodes (relative to c0) and weights; an even fine node sits
                // on one coarse node, an odd one halfway between two
                let lo = vec3<i32>(qu / 2u) - c0;
                let odd = (qu % 2u) == vec3<u32>(1u);
                let hi = lo + select(vec3<i32>(0), vec3<i32>(1), odd);
                let w = select(vec3<f32>(1.0), vec3<f32>(0.5), odd);
                let wz = vec2<f32>(w.z, select(0.0, w.z, odd.z));
                let wy = vec2<f32>(w.y, select(0.0, w.y, odd.y));
                let wx = vec2<f32>(w.x, select(0.0, w.x, odd.x));
                let zs = vec2<i32>(lo.z, hi.z) * i32(cs.x * cs.y);
                let ys = vec2<i32>(lo.y, hi.y) * i32(cs.x);
                let xs = vec2<i32>(lo.x, hi.x);
                for (var cz = 0u; cz < 2u; cz++) {
                    for (var cy = 0u; cy < 2u; cy++) {
                        let wzy = wz[cz] * wy[cy];
                        let row = zs[cz] + ys[cy];
                        v += wzy * (wx.x * sc(row + xs.x) + wx.y * sc(row + xs.y));
                    }
                }
                v *= free3(fq);
            }
        }
        su_set(s, v);
    }
    workgroupBarrier();
    let node = slot * BV + li;
    let fm = free[node];
    var acc = vec3<f32>(0.0);
    if fm != 0u {
        acc = stencil(li, true, false).ku * free3(fm);
    }
    y[3u * node] = acc.x;
    y[3u * node + 1u] = acc.y;
    y[3u * node + 2u] = acc.z;
}

//// kernel diag_l1 bricks
// Level 1's diagonal from the fine `ee` and the Pᵀ (D ke D) P table (by clamped-node mask and child
// position). P.n = fine grid; one thread per stored level-1 node (P.n.w level-1 slots).
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> T: array<f32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;
#include ke BIND=5
@group(0) @binding(6) var<storage, read> page: array<u32>;
@group(0) @binding(7) var<storage, read> coords: array<vec4<u32>>;

#include lookup NAME=fat PAGE=page
#include fine_free

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let node = flat(wg, nwg, li);
    if node >= P.n.w * BV { return; }
    let nf = P.n.xyz;
    let nbf = nbricks(nf);
    let n = (nf + vec3<u32>(1u)) / 2u;
    let p = vec3<i32>(coords[node / BV].xyz * bshape() + unflat(node % BV, bshape()));
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let e = p - vec3<i32>(vec3<u32>(a & 1u, (a >> 1u) & 1u, (a >> 2u) & 1u));
        if any(e < vec3<i32>(0)) || any(vec3<u32>(e) >= n) { continue; }
        for (var c = 0u; c < 8u; c++) {
            let f = 2 * e + vec3<i32>(vec3<u32>(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u));
            if any(vec3<u32>(f) >= nf) { continue; }
            let ge = fat(f, nbf);
            if ge == NONE { continue; }
            let w = ee[ge];
            if w == 0.0 { continue; }
            let fm = fine_free(f, nbf);
            if (fm >> 31u) != 0u {
                // per-axis masks: Σ P[n, a] ke[(n, x), (n', x)] P[n', a] over free (n, x), (n', x)
                var v = vec3<f32>(0.0);
                for (var n1 = 0u; n1 < 8u; n1++) {
                    let pa = wgt(c, n1, a);
                    if pa == 0.0 { continue; }
                    for (var n2 = 0u; n2 < 8u; n2++) {
                        let pb = wgt(c, n2, a);
                        if pb == 0.0 { continue; }
                        for (var x = 0u; x < 3u; x++) {
                            let free_both = ((fm >> (3u * n1 + x)) & (fm >> (3u * n2 + x)) & 1u) == 1u;
                            if free_both { v[x] += pa * kv((3u * n1 + x) * 24u + 3u * n2 + x) * pb; }
                        }
                    }
                }
                acc += w * v;
            } else {
                let base = (fine_mask(f, nbf) * 8u + c) * 576u + (3u * a) * 25u;
                acc += w * vec3<f32>(T[base], T[base + 25u], T[base + 50u]);
            }
        }
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel apply_elem bricks
// y = K u on a level stored as symmetric 24×24 element matrices: upper triangles (`sym`),
// entry-major (mats[entry · P.m.y + e], P.m.y the level's stored elements) so that neighbouring
// threads read neighbouring addresses. One workgroup per stored brick.
@group(0) @binding(1) var<storage, read> mats: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> page: array<u32>;
@group(0) @binding(5) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nl PAGE=page

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nl_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    let node = slot * BV + li;
    let ne = P.m.y;
    let l = vec3<i32>(unflat(li, bshape()));
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let el = l - vec3<i32>(vec3<u32>(a & 1u, (a >> 1u) & 1u, (a >> 2u) & 1u));
        let e = nl_at(el);
        if e == NONE { continue; }
        for (var b = 0u; b < 8u; b++) {
            let m = nl_at(el + vec3<i32>(vec3<u32>(b & 1u, (b >> 1u) & 1u, (b >> 2u) & 1u)));
            if m == NONE { continue; }
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

//// kernel diag_elem bricks
// diag(K) of a level stored as element matrices (as `apply_elem`)
@group(0) @binding(1) var<storage, read> mats: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@group(0) @binding(3) var<storage, read> page: array<u32>;
@group(0) @binding(4) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nl PAGE=page

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nl_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    let node = slot * BV + li;
    let ne = P.m.y;
    let l = vec3<i32>(unflat(li, bshape()));
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let e = nl_at(l - vec3<i32>(vec3<u32>(a & 1u, (a >> 1u) & 1u, (a >> 2u) & 1u)));
        if e == NONE { continue; }
        let r = 3u * a;
        acc += vec3<f32>(mats[sym(r, r) * ne + e], mats[sym(r + 1u, r + 1u) * ne + e], mats[sym(r + 2u, r + 2u) * ne + e]);
    }
    d[3u * node] = acc.x;
    d[3u * node + 1u] = acc.y;
    d[3u * node + 2u] = acc.z;
}

//// kernel assemble bricks
// Galerkin coarsening, one workgroup per stored coarse element (of the output level's P.n.w slots):
// out_E = Σ_c P_cᵀ K_c P_c (stored entry-major). With P.m.x == 1 the children are level-1 elements
// computed on the fly from the fine grid (P.n = fine grid, `page` its page table; the output is
// level 2); otherwise they are stored matrices (P.n = the child level's grid, `page` its page
// table, P.m.y its stored elements).
// Level 2 is linear in the fine weights E_e h: fine elements without a clamped or partly free node
// add E_e h · T2[c2 · 8 + c1] (`free_level2_table`, precomputed P_c2ᵀ T[0, c1] P_c2, upper
// triangles), and only level-1 children holding another fine element (near supports, rollers and
// symmetry planes) take the product K P, Pᵀ (K P) through workgroup memory, as stored children do.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
@group(0) @binding(2) var<storage, read> free: array<u32>;
@group(0) @binding(3) var<storage, read> T: array<f32>;
@group(0) @binding(4) var<storage, read> child: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
#include ke BIND=6
@group(0) @binding(7) var<storage, read> page: array<u32>;
@group(0) @binding(8) var<storage, read> coords: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> T2: array<f32>;

#include lookup NAME=fat PAGE=page
#include fine_free

var<workgroup> K: array<f32, 576>;
var<workgroup> KP: array<f32, 576>;
// virtual children: per fine element under this coarse element (index c2 · 8 + c1), its E_e h and
// the offset of its table matrix, loaded once rather than once per matrix entry. FW: the weight
// when it takes the product path, FW2: when it adds its T2 matrix (0 otherwise).
var<workgroup> FW: array<f32, 64>;
var<workgroup> FW2: array<f32, 64>;
// bit c2: level-1 child c2 takes the product path (stored children: all); FBU its uniform copy
var<workgroup> FB: atomic<u32>;
var<workgroup> FBU: u32;
var<workgroup> FT: array<u32, 64>;
// per fine element: its nodes' free axes when one is partly free (`fine_free`), else 0
var<workgroup> FM: array<u32, 64>;
// stored children: their indices (NONE: none)
var<workgroup> CI: array<u32, 8>;

// entry idx of the level-1 matrix of fine element child c1 with node free axes fm (per-axis masks)
fn partial_entry(c1: u32, fm: u32, idx: u32) -> f32 {
    let r = idx / 24u;
    let col = idx % 24u;
    let a = r / 3u;
    let x = r % 3u;
    let b = col / 3u;
    let y = col % 3u;
    var v = 0.0;
    for (var n = 0u; n < 8u; n++) {
        if ((fm >> (3u * n + x)) & 1u) == 0u { continue; }
        let pa = PW[(c1 * 8u + n) * 8u + a];
        if pa == 0.0 { continue; }
        for (var n2 = 0u; n2 < 8u; n2++) {
            if ((fm >> (3u * n2 + y)) & 1u) == 0u { continue; }
            v += pa * kv((3u * n + x) * 24u + 3u * n2 + y) * PW[(c1 * 8u + n2) * 8u + b];
        }
    }
    return v;
}
// wgt(c, a, b) at (c · 8 + a) · 8 + b, so the products below don't recompute it
var<workgroup> PW: array<f32, 512>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let ce = wg_flat(wg, nwg);
    let nce = P.n.w * BV;
    // uniform across the workgroup (barriers follow)
    if ce >= nce { return; }
    let virt = P.m.x == 1u;
    let nf = P.n.xyz;
    // the child level's grid, and the output's (nf: fine grid when the children are virtual)
    var nl = nf;
    if virt { nl = (nf + vec3<u32>(1u)) / 2u; }
    let nc = (nl + vec3<u32>(1u)) / 2u;
    let nbf = nbricks(nf);
    let cp = coords[ce / BV].xyz * bshape() + unflat(ce % BV, bshape());
    if any(cp >= nc) {
        // past the grid
        for (var t = 0u; t < 9u; t++) {
            let idx = li + 64u * t;
            let row = idx / 24u;
            let col = idx % 24u;
            if row <= col { out[sym(row, col) * nce + ce] = 0.0; }
        }
        return;
    }
    for (var t = 0u; t < 8u; t++) {
        let i = li * 8u + t;
        PW[i] = wgt(i / 64u, (i / 8u) % 8u, i % 8u);
    }
    if virt {
        // thread li loads fine element c1 = li % 8 of level-1 child c2 = li / 8
        let c2 = li / 8u;
        let c1 = li % 8u;
        let bits2 = vec3<u32>(c2 & 1u, (c2 >> 1u) & 1u, (c2 >> 2u) & 1u);
        let bits1 = vec3<u32>(c1 & 1u, (c1 >> 1u) & 1u, (c1 >> 2u) & 1u);
        let f = 2u * (2u * cp + bits2) + bits1;
        var w = 0.0;
        var off = 0u;
        var fm = 0u;
        if all(f < nf) {
            let ge = fat(vec3<i32>(f), nbf);
            if ge != NONE { w = ee[ge]; }
            if w != 0.0 {
                fm = fine_free(vec3<i32>(f), nbf);
                if (fm >> 31u) == 0u {
                    fm = 0u;
                    off = (fine_mask(vec3<i32>(f), nbf) * 8u + c1) * 576u;
                }
            }
        }
        let slow = w != 0.0 && (fm != 0u || off >= 8u * 576u);
        FW[li] = select(0.0, w, slow);
        FW2[li] = select(w, 0.0, slow);
        FT[li] = off;
        FM[li] = fm;
        if slow { atomicOr(&FB, 1u << c2); }
    } else if li < 8u {
        let lc = 2u * cp + vec3<u32>(li & 1u, (li >> 1u) & 1u, (li >> 2u) & 1u);
        var g = NONE;
        if all(lc < nl) { g = fat(vec3<i32>(lc), nbricks(nl)); }
        CI[li] = g;
    }
    workgroupBarrier();
    if li == 0u { FBU = select(0xffu, atomicLoad(&FB), virt); }
    let fb = workgroupUniformLoad(&FBU);
    // this thread's upper-triangle entries k = li + 64 t (t < 5, k < 300): row and column
    var rr: array<u32, 5>;
    var cc: array<u32, 5>;
    for (var t = 0u; t < 5u; t++) {
        let k = li + 64u * t;
        var start = 0u;
        rr[t] = 0u;
        for (var r = 0u; r < 24u; r++) {
            if k < start + 24u - r { rr[t] = r; break; }
            start += 24u - r;
        }
        cc[t] = rr[t] + k - start;
    }
    // stored children: their upper triangles, all loaded before any is used (the loads are
    // scattered, entry-major: in flight together they overlap)
    var kc: array<f32, 40>;
    if !virt {
        for (var c2 = 0u; c2 < 8u; c2++) {
            let g = CI[c2];
            for (var t = 0u; t < 5u; t++) {
                let k = li + 64u * t;
                var v = 0.0;
                if k < 300u && g != NONE { v = child[k * P.m.y + g]; }
                kc[c2 * 5u + t] = v;
            }
        }
    }
    if fb != 0u {
        var acc: array<f32, 9>;
        for (var t = 0u; t < 9u; t++) { acc[t] = 0.0; }
        for (var c2 = 0u; c2 < 8u; c2++) {
            if ((fb >> c2) & 1u) == 0u { continue; }
            // the child's matrix
            if virt {
                for (var t = 0u; t < 9u; t++) {
                    let idx = li + 64u * t;
                    var s = 0.0;
                    for (var c1 = 0u; c1 < 8u; c1++) {
                        let w = FW[c2 * 8u + c1];
                        if w != 0.0 {
                            let fm = FM[c2 * 8u + c1];
                            if fm != 0u {
                                s += w * partial_entry(c1, fm, idx);
                            } else {
                                s += w * T[FT[c2 * 8u + c1] + idx];
                            }
                        }
                    }
                    K[idx] = s;
                }
            } else {
                for (var t = 0u; t < 5u; t++) {
                    if li + 64u * t < 300u {
                        let v = kc[c2 * 5u + t];
                        K[rr[t] * 24u + cc[t]] = v;
                        K[cc[t] * 24u + rr[t]] = v;
                    }
                }
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
                    s += K[r * 24u + 3u * a + jj] * PW[(c2 * 8u + a) * 8u + b];
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
                    s += PW[(c2 * 8u + a) * 8u + b] * KP[(3u * a + ii) * 24u + col];
                }
                acc[t] += s;
            }
            workgroupBarrier();
        }
        // to K, for the upper-triangle pass below
        for (var t = 0u; t < 9u; t++) { K[li + 64u * t] = acc[t]; }
        workgroupBarrier();
    }
    // upper triangle, entry-major, like apply_elem reads it: entry k = sym(row, col)
    for (var t = 0u; t < 5u; t++) {
        let k = li + 64u * t;
        if k >= 300u { continue; }
        var s = 0.0;
        if virt {
            for (var e = 0u; e < 64u; e++) {
                let w = FW2[e];
                if w != 0.0 { s += w * T2[e * 300u + k]; }
            }
        }
        if fb != 0u { s += K[rr[t] * 24u + cc[t]]; }
        out[k * nce + ce] = s;
    }
}

//// kernel restrict bricks
// rc = Pᵀ D r from the grid P.n to its coarse grid (rc -= Pᵀ D r when P.m.x == 1), D keeping the
// DOFs with a positive diagonal (as `prolong`). `mask` is the grid's diagonal (f32), or on the fine
// grid (P.m.y == 1, its diagonal isn't stored) its per-node free axes: with E_min > 0 a free DOF has
// a positive diagonal. One workgroup per stored coarse brick (`page`: the fine grid's, `coords`:
// the coarse grid's): the fine nodes 2c - 1 .. 2c + 1 of its nodes c lie in the 3×3×3 fine bricks
// around fine brick 2b.
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> rc: array<f32>;
@group(0) @binding(4) var<storage, read> page: array<u32>;
@group(0) @binding(5) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nf PAGE=page

fn keep(node: u32) -> vec3<f32> {
    if P.m.y == 1u { return free3(mask[node]); }
    let q = 3u * node;
    return vec3<f32>(f32(bitcast<f32>(mask[q]) > 0.0), f32(bitcast<f32>(mask[q + 1u]) > 0.0),
                     f32(bitcast<f32>(mask[q + 2u]) > 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nf_load(2u * coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    let m = slot * BV + li;
    let c2 = 2 * vec3<i32>(unflat(li, bshape()));
    var acc = vec3<f32>(0.0);
    for (var dk = -1; dk <= 1; dk++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var di = -1; di <= 1; di++) {
                let g = nf_at(c2 + vec3<i32>(di, dj, dk));
                if g == NONE { continue; }
                let w = select(0.5, 1.0, di == 0) * select(0.5, 1.0, dj == 0) * select(0.5, 1.0, dk == 0);
                let q = 3u * g;
                acc += w * keep(g) * vec3<f32>(r[q], r[q + 1u], r[q + 2u]);
            }
        }
    }
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

//// kernel prolong bricks
// x += D P xc on the grid P.n (x = D P xc when P.m.x == 1), D keeping the DOFs with a positive
// diagonal (`mask` as in `restrict`, free axes when P.m.y == 1). (Level 1, applied as Pᵀ D K D P
// through the fine grid, is then exact for rollers too; its diagonal and level 2 come from the
// per-node table, which sees a roller node as clamped.) One workgroup per stored brick of the grid
// (`coords`; `pagec`: the coarse grid's page table): its coarse nodes lie in the 3×3×3 coarse
// bricks around b / 2.
@group(0) @binding(1) var<storage, read> xc: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> pagec: array<u32>;
@group(0) @binding(5) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nc PAGE=pagec

fn keep(node: u32) -> vec3<f32> {
    if P.m.y == 1u { return free3(mask[node]); }
    let q = 3u * node;
    return vec3<f32>(f32(bitcast<f32>(mask[q]) > 0.0), f32(bitcast<f32>(mask[q + 1u]) > 0.0),
                     f32(bitcast<f32>(mask[q + 2u]) > 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    let fb = coords[slot].xyz;
    let nf = P.n.xyz;
    nc_load(fb / 2u, nbricks((nf + vec3<u32>(1u)) / 2u), li);
    workgroupBarrier();
    let m = slot * BV + li;
    let p = fb * bshape() + unflat(li, bshape());
    let corig = vec3<i32>((fb / 2u) * bshape());
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
        let g = nc_at(vec3<i32>(q) - corig);
        if g == NONE { continue; }
        acc += w * vec3<f32>(xc[3u * g], xc[3u * g + 1u], xc[3u * g + 2u]);
    }
    let k = keep(m);
    for (var d = 0u; d < 3u; d++) {
        if P.m.x == 1u {
            x[3u * m + d] = select(0.0, acc[d], k[d] > 0.0);
        } else if k[d] > 0.0 {
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
// x = A⁻¹ b with a dense inverse over the coarsest grid's P.m.x DOFs (x-fastest nodes, 3 each);
// `map` holds each DOF's index in the brick layout (NONE: its brick isn't stored).
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> map: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let r = flat(wg, nwg, li);
    let n = P.m.x;
    if r >= n { return; }
    let mr = map[r];
    if mr == NONE { return; }
    var s = 0.0;
    for (var j = 0u; j < n; j++) {
        let mj = map[j];
        if mj != NONE { s += inv[r * n + j] * b[mj]; }
    }
    x[mr] = s;
}

//// kernel energy bricks
// ce = u_eᵀ ke u_e per fine element with stiffness (added to ce when P.m.x == 1). One workgroup per
// stored brick, u staged on the brick and a one-node halo on its + sides.
@group(0) @binding(1) var<storage, read> ee: array<f32>;
#include ke BIND=2
@group(0) @binding(3) var<storage, read> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> ce: array<f32>;
@group(0) @binding(5) var<storage, read> page: array<u32>;
@group(0) @binding(6) var<storage, read> coords: array<vec4<u32>>;

#include nbhd NAME=nf PAGE=page

var<workgroup> su: array<vec3<f32>, EN>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let slot = wg_flat(wg, nwg);
    if slot >= P.n.w { return; }
    nf_load(coords[slot].xyz, nbricks(P.n.xyz), li);
    workgroupBarrier();
    let es = bshape() + vec3<u32>(1u);
    for (var s = li; s < es.x * es.y * es.z; s += BV) {
        let g = nf_at(vec3<i32>(unflat(s, es)));
        var v = vec3<f32>(0.0);
        if g != NONE { v = vec3<f32>(u[3u * g], u[3u * g + 1u], u[3u * g + 2u]); }
        su[s] = v;
    }
    workgroupBarrier();
    let e = slot * BV + li;
    var s = 0.0;
    if ee[e] != 0.0 {
        let base = flat3(unflat(li, bshape()), es);
        // relative to the first node, as in apply_fine
        var ue: array<f32, 24>;
        let u0 = su[base];
        for (var b = 0u; b < 8u; b++) {
            let v = su[base + (b & 1u) + es.x * ((b >> 1u) & 1u) + es.x * es.y * ((b >> 2u) & 1u)] - u0;
            ue[3u * b] = v.x;
            ue[3u * b + 1u] = v.y;
            ue[3u * b + 2u] = v.z;
        }
        for (var r = 0u; r < 24u; r++) {
            var row = 0.0;
            for (var q = 0u; q < 24u; q++) { row += kv(r * 24u + q) * ue[q]; }
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
