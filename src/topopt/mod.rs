//! Topology optimization: minimum compliance (stiffest part for a given amount of material) with the
//! SIMP method on a voxel grid, in the style of top88/top3d.
//!
//! The problem comes from a shape file: `map` is the design domain, `field_fixed` (< 0) clamps nodes,
//! `field_load` (vec3f) gives the load direction where it is non-zero, and the optional `field_keep`
//! / `field_void` (< 0) force elements solid / empty. Primitives from the editor add to these with
//! their roles (`prims::Role`); each load primitive has its own force and load case. Each voxel is a
//! trilinear 8-node hex element.
//!
//! Load cases: loads in one case act together; separate cases are separate situations. The
//! objective is the sum of the cases' compliances (each solved separately).
//!
//! Each iteration solves K(ρ) u = f with a conjugate gradient (warm-started, matrix-free on the fine
//! grid) preconditioned by a geometric multigrid V-cycle. The whole iteration runs either on the GPU
//! in f32 (`gpu_opt`, `gpu_mg`) or on the CPU (`optimize` below with `mg`, multithreaded; the
//! reference the GPU path is tested against). Jacobi-preconditioned CG is kept as a fallback. It then computes the compliance sensitivities, applies a density filter and updates the
//! densities with the optimality-criteria method under a volume constraint. With overhang control
//! (`overhang.rs`) the AM filter sits between the density filter and the projection. The current design is
//! written to a .vti as it runs, so `gen3d view` can watch it live.
//!
//! Units: mm, N, MPa. With the default PLA-like modulus the displacements are real millimetres.

// small dense matrix code reads best with explicit indices
#![allow(clippy::needless_range_loop)]

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gpu::Gpu;
use crate::mesher::{Grid, sample_field};
use crate::shape::Shape;
use output::Snapshot;
use symmetry::Model;

mod checkpoint;
mod gpu_engine;
mod gpu_mg;
mod gpu_opt;
pub mod mg;
mod output;
mod overhang;
mod symmetry;

/// Linear solver for K u = f.
#[derive(Clone, Copy, Debug)]
pub enum Solver {
    /// Jacobi-preconditioned CG
    Jacobi,
    /// multigrid-preconditioned CG
    Multigrid(mg::MgSettings),
    /// multigrid-preconditioned CG on the GPU (f32)
    Gpu(mg::MgSettings),
}

#[derive(Clone)]
pub struct Settings {
    /// element size (mm)
    pub voxel: f64,
    /// fraction of the design elements' volume to keep
    pub volfrac: f64,
    /// SIMP penalty
    pub penal: f64,
    /// density filter radius, in voxels
    pub rmin: f64,
    pub iters: usize,
    /// stop when no density changes by more than this
    pub tol: f64,
    /// Young's modulus of the solid (MPa)
    pub e0: f64,
    pub nu: f64,
    /// total load (N), spread over the loaded nodes
    pub force: f64,
    /// Heaviside projection: final sharpness β (0 = no projection). β starts at 1 and doubles each
    /// time the design converges, up to this.
    pub beta_max: f64,
    pub solver: Solver,
    /// stiffness of empty elements, relative to `e0`
    pub emin: f64,
    /// relative residual at which CG stops (in the final projection stage)
    pub cg_tol: f64,
    /// solve less accurately while the design is still moving a lot: tolerance 0.05 × the last
    /// density change, between `cg_tol` and 1e-2, and `cg_tol` once β reaches its final value
    pub adaptive_tol: bool,
    /// the part stands on the ground: the bottom of the design domain is supported
    pub ground: Ground,
    /// `run`: do the β = 1 stage on a grid of twice the element size first
    pub coarse_start: bool,
    /// use the shape's mirror planes: solve one side of a symmetric problem (see `symmetry.rs`)
    pub symmetry: bool,
    /// keep overhangs to 45° or steeper when printed along `up` (the AM filter, `overhang.rs`)
    pub overhang: bool,
    /// the build direction: the side of the part that faces up on the printer
    pub up: crate::mesh::Up,
}

/// How the bottom of the design domain is supported (`--ground`). The optimizer decides where the
/// part touches down: the supported nodes may end up under empty space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Ground {
    /// no ground: only the fixed regions hold the part
    #[default]
    None,
    /// stands on a slippery surface: held vertically only, free to slide and spread sideways (the
    /// least pins against sliding and spinning are added)
    Slide,
    /// stands on a grippy surface (or is glued down): held in all directions
    Grip,
}

impl Ground {
    pub fn name(self) -> &'static str {
        match self {
            Ground::None => "none",
            Ground::Slide => "slide",
            Ground::Grip => "grip",
        }
    }
}

impl Settings {
    /// The usual E_min for a solver: multigrid copes with the 1e9 stiffness contrast, Jacobi doesn't.
    pub fn default_emin(solver: &Solver) -> f64 {
        match solver {
            Solver::Jacobi => 1e-6,
            Solver::Multigrid(_) | Solver::Gpu(_) => 1e-9,
        }
    }
}

/// A discretized problem. Elements and nodes are indexed x-fastest.
#[derive(Clone)]
pub struct Problem {
    /// elements per axis
    pub n: [usize; 3],
    pub h: f64,
    /// position of the first node (the bounds' minimum)
    pub origin: [f64; 3],
    pub domain: Vec<bool>,
    /// forced solid (design domain ∩ keep)
    pub passive: Vec<bool>,
    /// per node: the clamped axes, bit c for axis c (only nodes of domain elements); 7 = fully
    /// clamped, a single bit = a roller (held along that axis only)
    pub fixed: Vec<u8>,
    /// per load case, per node: force (N)
    pub loads: Vec<Vec<[f64; 3]>>,
    /// the load cases' numbers, in the order of `loads`
    pub cases: Vec<u32>,
}

impl Problem {
    fn nodes(&self) -> [usize; 3] {
        self.n.map(|v| v + 1)
    }

    fn node(&self, i: usize, j: usize, k: usize) -> usize {
        let [nx, ny, _] = self.nodes();
        (k * ny + j) * nx + i
    }

    /// Nodes that belong to at least one domain element (the others carry no stiffness).
    fn active_nodes(&self) -> Vec<bool> {
        let [ex, ey, ez] = self.n;
        let mut active = vec![false; self.nodes().iter().product()];
        for k in 0..ez {
            for j in 0..ey {
                for i in 0..ex {
                    if self.domain[(k * ey + j) * ex + i] {
                        for b in 0..8 {
                            active[self.node(i + (b & 1), j + (b >> 1 & 1), k + (b >> 2 & 1))] = true;
                        }
                    }
                }
            }
        }
        active
    }

    fn node_pos(&self, i: usize) -> [f64; 3] {
        let [nx, ny, _] = self.nodes();
        let ijk = [i % nx, (i / nx) % ny, i / (nx * ny)];
        std::array::from_fn(|c| self.origin[c] + ijk[c] as f64 * self.h)
    }

    /// Per node: the free axes (bit c for axis c), 0 for nodes without stiffness.
    fn free_mask(&self) -> Vec<u8> {
        let active = self.active_nodes();
        (0..active.len())
            .map(|i| if active[i] { 7 & !self.fixed[i] } else { 0 })
            .collect()
    }

    /// Which load cases need a solve. All cases share K, so a case whose load is a linear
    /// combination of other cases' loads has the same combination of their displacements: the
    /// rocked variants of a load (f ± rock·F_axis) span only 3 directions for 5 cases. Cases are
    /// taken in order; one is solved when its load (on free DOFs) is not within 1e-6 of the span of
    /// the cases solved before it.
    fn load_basis(&self) -> LoadBasis {
        let free = self.free_mask();
        let g: Vec<Vec<f64>> = self
            .loads
            .iter()
            .map(|l| {
                l.iter()
                    .zip(&free)
                    .flat_map(|(v, &f)| (0..3).map(move |c| if f >> c & 1 == 1 { v[c] } else { 0.0 }))
                    .collect()
            })
            .collect();
        let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let mut solved: Vec<usize> = vec![];
        // orthonormal basis of the solved loads (Gram–Schmidt, twice for stability)
        let mut q: Vec<Vec<f64>> = vec![];
        let mut cases = vec![];
        for (c, gc) in g.iter().enumerate() {
            let norm = dot(gc, gc).sqrt();
            let mut r = gc.clone();
            for _ in 0..2 {
                for qk in &q {
                    let d = dot(qk, &r);
                    r.par_iter_mut().zip(qk).for_each(|(a, b)| *a -= d * b);
                }
            }
            let rn = dot(&r, &r).sqrt();
            if rn > 1e-6 * norm {
                r.par_iter_mut().for_each(|v| *v /= rn);
                q.push(r);
                cases.push(CaseSolve::Solved(solved.len()));
                solved.push(c);
            } else {
                // least squares over the solved cases' loads: (BᵀB) a = Bᵀ g
                let k = solved.len();
                let mut m: Vec<Vec<f64>> = (0..k)
                    .map(|i| {
                        let mut row: Vec<f64> = (0..k).map(|j| dot(&g[solved[i]], &g[solved[j]])).collect();
                        row.push(dot(&g[solved[i]], gc));
                        row
                    })
                    .collect();
                let a = solve_dense(&mut m);
                cases.push(CaseSolve::Combined(
                    a.into_iter().enumerate().filter(|(_, v)| *v != 0.0).collect(),
                ));
            }
        }
        LoadBasis { solved, cases }
    }
}

/// How the load cases are solved (`Problem::load_basis`).
pub(super) struct LoadBasis {
    /// the solved cases (by index into `loads`), in order
    pub solved: Vec<usize>,
    /// per load case
    pub cases: Vec<CaseSolve>,
}

pub(super) enum CaseSolve {
    /// solved, into solution slot i (the position in `LoadBasis::solved`)
    Solved(usize),
    /// u = Σ a · u_slot
    Combined(Vec<(usize, f64)>),
}

impl LoadBasis {
    /// A line when some cases (over all groups) are combinations of others.
    fn describe_all(bases: &[LoadBasis]) -> Option<String> {
        let n: usize = bases.iter().map(|b| b.cases.len()).sum();
        let solved: usize = bases.iter().map(|b| b.solved.len()).sum();
        let what = if bases.len() > 1 { "load parts" } else { "load cases" };
        (solved < n).then(|| format!("{solved} of {n} {what} solved, the others are linear combinations of them"))
    }
}

/// Solve the small dense system [A | b] in place (Gaussian elimination, partial pivoting).
fn solve_dense(m: &mut [Vec<f64>]) -> Vec<f64> {
    let k = m.len();
    for i in 0..k {
        let piv = (i..k).max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs())).unwrap();
        m.swap(i, piv);
        for r in i + 1..k {
            let f = m[r][i] / m[i][i];
            for c in i..=k {
                m[r][c] -= f * m[i][c];
            }
        }
    }
    let mut a = vec![0.0; k];
    for i in (0..k).rev() {
        let s: f64 = (i + 1..k).map(|j| m[i][j] * a[j]).sum();
        a[i] = (m[i][k] - s) / m[i][i];
    }
    a
}

/// The active nodes of the lowest layer that has any.
fn ground_nodes(p: &Problem, active: &[bool]) -> Vec<usize> {
    let [nx, ny, _] = p.nodes();
    let layer = nx * ny;
    let Some(first) = active.iter().position(|a| *a) else {
        return vec![];
    };
    let k = first / layer;
    (k * layer..(k + 1) * layer).filter(|&i| active[i]).collect()
}

/// The rigid-body motions (3 translations, 3 rotations) at node position `q` (relative to `c`,
/// scaled by `scale`) along axis `a`.
fn rigid_row(q: [f64; 3], c: [f64; 3], scale: f64, a: usize) -> [f64; 6] {
    let r: [f64; 3] = std::array::from_fn(|k| (q[k] - c[k]) / scale);
    let mut row = [0.0; 6];
    row[a] = 1.0;
    // (ω × r)_a for ω = e_x, e_y, e_z
    for (w, out) in row[3..].iter_mut().enumerate() {
        let mut e = [0.0; 3];
        e[w] = 1.0;
        let cross = [
            e[1] * r[2] - e[2] * r[1],
            e[2] * r[0] - e[0] * r[2],
            e[0] * r[1] - e[1] * r[0],
        ];
        *out = cross[a];
    }
    row
}

/// An orthonormal basis of the rigid-body motions the supports block, grown row by row.
struct RigidBasis(Vec<[f64; 6]>);

impl RigidBasis {
    /// Adds a row; whether it blocked another motion.
    fn add(&mut self, row: [f64; 6]) -> bool {
        if self.0.len() == 6 {
            return false;
        }
        let mut v = row;
        for b in &self.0 {
            let d: f64 = (0..6).map(|k| v[k] * b[k]).sum();
            for k in 0..6 {
                v[k] -= d * b[k];
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 1e-6 {
            return false;
        }
        self.0.push(v.map(|x| x / n));
        true
    }
}

/// Centre and size of the supported nodes (the frame the rigid motions are measured in).
fn support_frame(p: &Problem) -> ([f64; 3], f64) {
    let held: Vec<[f64; 3]> = (0..p.fixed.len())
        .filter(|&i| p.fixed[i] != 0)
        .map(|i| p.node_pos(i))
        .collect();
    if held.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let c: [f64; 3] = std::array::from_fn(|k| held.iter().map(|q| q[k]).sum::<f64>() / held.len() as f64);
    let scale = held
        .iter()
        .map(|q| (0..3).map(|k| (q[k] - c[k]).powi(2)).sum::<f64>().sqrt())
        .fold(p.h, f64::max);
    (c, scale)
}

fn rigid_basis(p: &Problem, c: [f64; 3], scale: f64) -> RigidBasis {
    let mut b = RigidBasis(vec![]);
    for i in 0..p.fixed.len() {
        for a in 0..3 {
            if p.fixed[i] >> a & 1 == 1 {
                b.add(rigid_row(p.node_pos(i), c, scale, a));
                if b.0.len() == 6 {
                    return b;
                }
            }
        }
    }
    b
}

/// How many of the 6 rigid-body motions the supports block.
fn rigid_rank(p: &Problem) -> usize {
    let (c, scale) = support_frame(p);
    rigid_basis(p, c, scale).0.len()
}

/// The fewest extra pins (node, axes) that stop the part sliding or spinning on its supports (e.g.
/// on rollers): the supported node nearest their centre, then the one farthest from it, then the
/// one farthest from the line through both, each held only along the axes still needed.
fn stabilize(p: &Problem) -> Vec<(usize, u8)> {
    let (c, scale) = support_frame(p);
    let mut basis = rigid_basis(p, c, scale);
    if basis.0.len() == 6 {
        return vec![];
    }
    let held: Vec<usize> = (0..p.fixed.len()).filter(|&i| p.fixed[i] != 0).collect();
    if held.is_empty() {
        return vec![];
    }
    let pos = |i: usize| p.node_pos(i);
    let dist2 = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let a = *held
        .iter()
        .min_by(|&&x, &&y| dist2(pos(x), c).total_cmp(&dist2(pos(y), c)))
        .unwrap();
    let b = *held
        .iter()
        .max_by(|&&x, &&y| dist2(pos(x), pos(a)).total_cmp(&dist2(pos(y), pos(a))))
        .unwrap();
    let (pa, pb) = (pos(a), pos(b));
    let ab: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
    let ab2 = dist2(pa, pb).max(1e-30);
    let off_line = |q: [f64; 3]| {
        let t = (0..3).map(|k| (q[k] - pa[k]) * ab[k]).sum::<f64>() / ab2;
        dist2(q, std::array::from_fn(|k| pa[k] + t * ab[k]))
    };
    let cc = *held
        .iter()
        .max_by(|&&x, &&y| off_line(pos(x)).total_cmp(&off_line(pos(y))))
        .unwrap();
    let mut out: Vec<(usize, u8)> = vec![];
    for node in [a, b, cc] {
        // the axes most across the line to the first pin first: they block spinning best
        let mut axes = [0, 1, 2];
        if node != a {
            axes.sort_by(|&x, &y| ab[x].abs().total_cmp(&ab[y].abs()));
        }
        let mut add = 0u8;
        for ax in axes {
            if p.fixed[node] >> ax & 1 == 0 && basis.add(rigid_row(pos(node), c, scale, ax)) {
                add |= 1 << ax;
            }
        }
        if add != 0 {
            out.push((node, add));
        }
        if basis.0.len() == 6 {
            break;
        }
    }
    out
}

/// Whether DOF `i` (3 per node) is free in a per-node axis mask.
fn dof_free(free: &[u8], i: usize) -> bool {
    free[i / 3] >> (i % 3) & 1 == 1
}

/// How many elements the automatic element size aims for: the GPU solver runs ~300k elements in a
/// few seconds per optimization; the CPU solvers stay at 40k to be as quick.
pub fn auto_elements(gpu_solver: bool) -> f64 {
    if gpu_solver { 300_000.0 } else { 40_000.0 }
}

/// Element size giving at most about `target` elements in the shape's bounds, rounded up to 2
/// significant digits. (It may be finer than the shape's `//! voxel`, which is its meshing and
/// preview resolution; the optimized result is a grid of its own.)
pub fn auto_voxel(shape: &Shape, target: f64) -> f32 {
    let vol: f64 = shape.size().iter().map(|s| (*s as f64).max(1e-3)).product();
    let v = (vol / target).cbrt();
    let mag = 10f64.powf(v.log10().floor() - 1.0);
    ((v / mag).ceil() * mag) as f32
}

/// z-fastest (the GPU sampler's order) to x-fastest.
fn x_fastest(v: &[f32], d: [usize; 3], comps: usize) -> Vec<f32> {
    let [nx, ny, nz] = d;
    let mut out = vec![0f32; v.len()];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let s = ((x * ny + y) * nz + z) * comps;
                let t = ((z * ny + y) * nx + x) * comps;
                out[t..t + comps].copy_from_slice(&v[s..s + comps]);
            }
        }
    }
    out
}

/// The most cells `setup` samples the shape's bounds at (before cropping to the domain).
const MAX_SAMPLED: usize = 200_000_000;

/// Empty cells kept around the domain when the grid is cropped to it.
const CROP_MARGIN: usize = 1;

/// The cell range [a, b) per axis that the grid is cropped to: the cells `inside` plus
/// `CROP_MARGIN`, symmetric about the middle on the `mirror` axes (so a mirror plane stays the
/// middle node layer). None if nothing is inside.
fn crop_box(inside: &[bool], n: [usize; 3], mirror: [bool; 3]) -> Option<([usize; 3], [usize; 3])> {
    let (mut a, mut b) = (n, [0usize; 3]);
    for (e, _) in inside.iter().enumerate().filter(|(_, i)| **i) {
        let ijk = [e % n[0], (e / n[0]) % n[1], e / (n[0] * n[1])];
        for k in 0..3 {
            a[k] = a[k].min(ijk[k]);
            b[k] = b[k].max(ijk[k] + 1);
        }
    }
    if b[0] == 0 {
        return None;
    }
    for k in 0..3 {
        a[k] = a[k].saturating_sub(CROP_MARGIN);
        b[k] = (b[k] + CROP_MARGIN).min(n[k]);
        if mirror[k] {
            let c = n[k] / 2;
            let half = (c - a[k].min(c)).max(b[k].max(c) - c);
            (a[k], b[k]) = (c - half, c + half);
        }
    }
    Some((a, b))
}

/// The cells [a, b) of the x-fastest cell values `v` on an `n` grid.
fn crop_cells(v: &[f32], n: [usize; 3], a: [usize; 3], b: [usize; 3]) -> Vec<f32> {
    let mut out = Vec::with_capacity((0..3).map(|k| b[k] - a[k]).product());
    for z in a[2]..b[2] {
        for y in a[1]..b[1] {
            let row = (z * n[1] + y) * n[0];
            out.extend_from_slice(&v[row + a[0]..row + b[0]]);
        }
    }
    out
}

/// Sample the shape's domain and boundary-condition fields on the element grid.
pub fn setup(shape: &Shape, s: &Settings, gpu: &Gpu) -> Result<Problem> {
    let h = s.voxel;
    if h <= 0.0 {
        bail!("voxel size must be positive");
    }
    // boundary-condition primitives reach 1.5 elements past their surface (so a hole's wall is clamped)
    let mut shape = shape.clone();
    if shape.bc {
        shape.defines.push(("g3d_reach".into(), 1.5 * h));
    }
    let shape = &shape;
    let fields = shape.fields()?;
    let field = |name: &str, comps: usize| fields.iter().find(|f| f.name == name && f.comps == comps);
    let supports = shape.support_specs();
    if supports.is_empty() && s.ground == Ground::None {
        bail!(
            "optimize needs supports: `fn field_fixed(p: vec3f) -> f32` (< 0 where the part is clamped), \
             a primitive with the fixed role, or --ground"
        );
    }
    let specs = shape.load_specs();
    if specs.is_empty() {
        bail!(
            "optimize needs loads: `fn field_load(p: vec3f) -> vec3f` (the load direction where it is \
             applied), or a primitive with the load role"
        );
    }
    let (lo, hi) = shape.bounds;
    let mut n = [0, 1, 2].map(|k| ((hi[k] - lo[k]) as f64 / h).ceil().max(1.0) as usize);
    let mut origin = lo.map(|v| v as f64);
    // a mirror plane becomes the node layer in the middle of the grid (also with --no-symmetry,
    // so that only changes how it is solved, not the discretization)
    let mirror = if shape.editor {
        shape.mirror.map(|m| m.is_some())
    } else {
        [false; 3]
    };
    for a in (0..3).filter(|&a| mirror[a]) {
        let c = shape.mirror[a].unwrap() as f64;
        let half = ((c - lo[a] as f64).max(hi[a] as f64 - c) / h).ceil().max(1.0) as usize;
        n[a] = 2 * half;
        origin[a] = c - half as f64 * h;
    }
    // the bounds are sampled once to find the domain; past this the sampling alone is too much
    let full: usize = n.iter().product();
    if full > MAX_SAMPLED {
        bail!(
            "{full} elements in the shape's bounds is too many to sample (at most {MAX_SAMPLED}); use a \
             larger --voxel"
        );
    }
    let cells_at = |n: [usize; 3], origin: [f64; 3]| Grid {
        vol: vec![],
        dims: n,
        origin: origin.map(|o| o + 0.5 * h),
        step: h,
    };
    let at = |grid: &Grid, func: &str| -> Result<Vec<f32>> {
        Ok(x_fastest(&sample_field(shape, grid, func, 1, gpu)?, grid.dims, 1))
    };
    let inside = |sdf: &[f32], void: &Option<Vec<f32>>| -> Vec<bool> {
        (0..sdf.len())
            .map(|e| sdf[e] < 0.0 && void.as_ref().is_none_or(|v| v[e] >= 0.0))
            .collect()
    };
    let cells = cells_at(n, origin);
    let mut sdf = at(&cells, "map")?;
    let mut void = field("void", 1).map(|f| at(&cells, &f.func)).transpose()?;
    // crop the grid to the domain: empty cells only cost memory and time (the bounds can be much
    // larger than the part, e.g. a declared `//! bounds`)
    if let Some((a, b)) = crop_box(&inside(&sdf, &void), n, mirror) {
        let m: [usize; 3] = std::array::from_fn(|k| b[k] - a[k]);
        if m != n {
            println!(
                "grid cropped to the design domain: {}x{}x{} -> {}x{}x{}",
                n[0], n[1], n[2], m[0], m[1], m[2]
            );
            sdf = crop_cells(&sdf, n, a, b);
            void = void.map(|v| crop_cells(&v, n, a, b));
            origin = std::array::from_fn(|k| origin[k] + a[k] as f64 * h);
            n = m;
        }
    }
    let ne: usize = n.iter().product();
    // the CPU multigrid needs ~1.2 kB per element (mostly the level-1 element matrices)
    let max = match s.solver {
        Solver::Jacobi => 3_000_000,
        Solver::Multigrid(_) => 8_000_000,
        Solver::Gpu(_) => 100_000_000,
    };
    if ne > max {
        bail!("{ne} elements is too many for this solver (at most {max}); use a larger --voxel");
    }
    let cells = cells_at(n, origin);
    let pts = n.map(|v| v + 1);
    let nodes = Grid {
        vol: vec![],
        dims: pts,
        origin,
        step: h,
    };
    let keep = field("keep", 1).map(|f| at(&cells, &f.func)).transpose()?;
    let domain = inside(&sdf, &void);
    let passive: Vec<bool> = (0..ne)
        .map(|e| domain[e] && keep.as_ref().is_some_and(|k| k[e] < 0.0))
        .collect();
    let mut p = Problem {
        n,
        h,
        origin,
        domain,
        passive,
        fixed: vec![],
        loads: vec![],
        cases: vec![],
    };
    let active = p.active_nodes();
    p.fixed = vec![0; active.len()];
    for spec in &supports {
        let r = x_fastest(&sample_field(shape, &nodes, &spec.func, 1, gpu)?, pts, 1);
        let mut hit = 0;
        for i in 0..active.len() {
            if active[i] && r[i] < 0.0 {
                p.fixed[i] |= spec.axes;
                hit += 1;
            }
        }
        if hit == 0 {
            bail!(
                "no supported nodes for `{}`: its region reaches no node of the design domain (try a \
                 smaller --voxel, a thicker region, or move it onto the part)",
                spec.name
            );
        }
    }
    if s.ground != Ground::None {
        let n = ground_nodes(&p, &active);
        let axes = if s.ground == Ground::Grip { 7 } else { 4 };
        for &i in &n {
            p.fixed[i] |= axes;
        }
        println!("ground ({}): {} nodes on the bottom layer", s.ground.name(), n.len());
    }
    let mut pins = stabilize(&p);
    // with mirror planes, the pins' mirror images too (so the supports stay symmetric)
    for (a, _) in mirror.iter().enumerate().filter(|(_, m)| **m) {
        for k in 0..pins.len() {
            let (i, axes) = pins[k];
            let mut ijk = [i % pts[0], (i / pts[0]) % pts[1], i / (pts[0] * pts[1])];
            ijk[a] = pts[a] - 1 - ijk[a];
            let m = (ijk[2] * pts[1] + ijk[1]) * pts[0] + ijk[0];
            if m != i {
                pins.push((m, axes));
            }
        }
    }
    for (i, axes) in pins {
        p.fixed[i] |= axes;
        let q = p.node_pos(i);
        let names: String = (0..3)
            .filter(|c| axes >> c & 1 == 1)
            .map(|c| ['X', 'Y', 'Z'][c])
            .collect();
        println!(
            "pinned the node at ({:.1}, {:.1}, {:.1}) in {names}: the supports alone let the part move",
            q[0], q[1], q[2]
        );
    }
    if rigid_rank(&p) < 6 {
        bail!("the supports can't hold the part in place (all on one line?): add or widen a support");
    }
    let nfixed = p.fixed.iter().filter(|b| **b != 0).count();
    // each load is spread over its own nodes so its magnitudes sum to its force
    let nn = active.len();
    let mut cases: std::collections::BTreeMap<u32, Vec<[f64; 3]>> = Default::default();
    let mut loaded = vec![false; nn];
    for spec in &specs {
        let raw: Vec<[f64; 3]> = match spec.dir {
            None => {
                let v = x_fastest(&sample_field(shape, &nodes, &spec.func, 3, gpu)?, pts, 3);
                (0..nn).map(|i| [0, 1, 2].map(|c| v[3 * i + c] as f64)).collect()
            }
            Some(d) => {
                let r = x_fastest(&sample_field(shape, &nodes, &spec.func, 1, gpu)?, pts, 1);
                let d = d.map(|c| c as f64);
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if l == 0.0 {
                    bail!("load `{}` has no direction", spec.name);
                }
                (0..nn)
                    .map(|i| if r[i] < 0.0 { d.map(|c| c / l) } else { [0.0; 3] })
                    .collect()
            }
        };
        let raw: Vec<[f64; 3]> = raw
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                if active[i] && p.fixed[i] != 7 && v.iter().all(|x| x.is_finite()) {
                    v
                } else {
                    [0.0; 3]
                }
            })
            .collect();
        let total: f64 = raw
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum();
        if total == 0.0 {
            bail!(
                "no loaded nodes for load `{}`: its region reaches no free node of the design domain \
                 (try a smaller --voxel, a thicker region, or move it onto the part)",
                spec.name
            );
        }
        let force = spec.force.unwrap_or(s.force);
        match &spec.label {
            Some(l) => println!("load `{}`: {force:.1} N, case {l}", spec.name),
            None => println!("load `{}`: {force} N, case {}", spec.name, spec.case),
        }
        let case = cases.entry(spec.case).or_insert_with(|| vec![[0.0; 3]; nn]);
        for (i, v) in raw.iter().enumerate() {
            if v.iter().any(|x| *x != 0.0) {
                loaded[i] = true;
                for c in 0..3 {
                    case[i][c] += v[c] * force / total;
                }
            }
        }
    }
    let nload = loaded.iter().filter(|b| **b).count();
    p.cases = cases.keys().copied().collect();
    p.loads = cases.into_values().collect();
    let ndomain = p.domain.iter().filter(|b| **b).count();
    let npassive = p.passive.iter().filter(|b| **b).count();
    println!(
        "grid {}x{}x{} @ {} mm: {ndomain} domain elements ({} design, {npassive} kept solid), {nfixed} supported nodes, {nload} loaded nodes, {} load case{}",
        n[0],
        n[1],
        n[2],
        h as f32,
        ndomain - npassive,
        p.loads.len(),
        if p.loads.len() == 1 { "" } else { "s" }
    );
    Ok(p)
}

/// Stiffness matrix of a unit-cube trilinear hex with E = 1 (2x2x2 Gauss). A cube of size h with
/// modulus E has stiffness `E * h * ke`. DOFs are 3 per node, nodes ordered by bits (x, y, z).
pub fn hex_ke(nu: f64) -> [[f64; 24]; 24] {
    let lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 1.0 / (2.0 * (1.0 + nu));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lam;
        }
        d[i][i] = lam + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    let g = 1.0 / 3f64.sqrt();
    let mut k = [[0.0; 24]; 24];
    for q in 0..8 {
        let (xi, eta, zeta) = (
            if q & 1 == 0 { -g } else { g },
            if q & 2 == 0 { -g } else { g },
            if q & 4 == 0 { -g } else { g },
        );
        let mut b = [[0.0; 24]; 6];
        for a in 0..8 {
            let (sx, sy, sz) = (
                (2 * (a & 1)) as f64 - 1.0,
                (2 * (a >> 1 & 1)) as f64 - 1.0,
                (2 * (a >> 2 & 1)) as f64 - 1.0,
            );
            // derivatives in the unit cube: x = (xi + 1) / 2, so d/dx = 2 d/dxi
            let dx = 2.0 * 0.125 * sx * (1.0 + eta * sy) * (1.0 + zeta * sz);
            let dy = 2.0 * 0.125 * sy * (1.0 + xi * sx) * (1.0 + zeta * sz);
            let dz = 2.0 * 0.125 * sz * (1.0 + xi * sx) * (1.0 + eta * sy);
            let c = 3 * a;
            b[0][c] = dx;
            b[1][c + 1] = dy;
            b[2][c + 2] = dz;
            b[3][c] = dy;
            b[3][c + 1] = dx;
            b[4][c + 1] = dz;
            b[4][c + 2] = dy;
            b[5][c] = dz;
            b[5][c + 2] = dx;
        }
        let det = 0.125; // unit cube from the [-1, 1]^3 reference cube; Gauss weights are 1
        let mut db = [[0.0; 24]; 6];
        for i in 0..6 {
            for j in 0..24 {
                db[i][j] = (0..6).map(|m| d[i][m] * b[m][j]).sum();
            }
        }
        for i in 0..24 {
            for j in 0..24 {
                k[i][j] += det * (0..6).map(|m| b[m][i] * db[m][j]).sum::<f64>();
            }
        }
    }
    k
}

/// Matrix-free finite element operator for the current element moduli.
struct Fem<'a> {
    p: &'a Problem,
    ke: &'a [[f64; 24]; 24],
    /// per element: E_e * h (0 outside the domain)
    ee: Vec<f64>,
    /// per node: the axes that have stiffness and are not clamped (bit c for axis c)
    free: Vec<u8>,
}

impl Fem<'_> {
    fn element_nodes(&self, e: usize) -> [usize; 8] {
        let [ex, ey, _] = self.p.n;
        let (i, j, k) = (e % ex, (e / ex) % ey, e / (ex * ey));
        std::array::from_fn(|b| self.p.node(i + (b & 1), j + (b >> 1 & 1), k + (b >> 2 & 1)))
    }

    /// y = K u, gathered per node (each node sums the rows of its up to 8 elements).
    fn matvec(&self, u: &[f64], y: &mut [f64]) {
        let [nx, ny, _] = self.p.nodes();
        let [ex, ey, ez] = self.p.n;
        y.par_chunks_mut(3).enumerate().for_each(|(n, yn)| {
            yn.fill(0.0);
            let free = self.free[n];
            if free == 0 {
                return;
            }
            let (i, j, k) = (n % nx, (n / nx) % ny, n / (nx * ny));
            for a in 0..8 {
                let (di, dj, dk) = (a & 1, a >> 1 & 1, a >> 2 & 1);
                if i < di || j < dj || k < dk || i - di >= ex || j - dj >= ey || k - dk >= ez {
                    continue;
                }
                let e = ((k - dk) * ey + (j - dj)) * ex + (i - di);
                let ee = self.ee[e];
                if ee == 0.0 {
                    continue;
                }
                let nodes = self.element_nodes(e);
                for (b, &m) in nodes.iter().enumerate() {
                    let um = &u[3 * m..3 * m + 3];
                    for c in 0..3 {
                        let row = &self.ke[3 * a + c][3 * b..3 * b + 3];
                        yn[c] += ee * (row[0] * um[0] + row[1] * um[1] + row[2] * um[2]);
                    }
                }
            }
            for (c, y) in yn.iter_mut().enumerate() {
                if free >> c & 1 == 0 {
                    *y = 0.0;
                }
            }
        });
    }

    fn inv_diag(&self) -> Vec<f64> {
        let nn = self.free.len();
        let mut d = vec![0.0; 3 * nn];
        for (e, &ee) in self.ee.iter().enumerate() {
            if ee == 0.0 {
                continue;
            }
            for (a, &n) in self.element_nodes(e).iter().enumerate() {
                for c in 0..3 {
                    d[3 * n + c] += ee * self.ke[3 * a + c][3 * a + c];
                }
            }
        }
        d.iter()
            .enumerate()
            .map(|(i, &v)| {
                if dof_free(&self.free, i) && v > 0.0 {
                    1.0 / v
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Preconditioned conjugate gradient, starting from `u`. `precond(r, z)` sets z = M⁻¹ r and must
    /// keep z zero on non-free DOFs. Returns the iteration count.
    fn solve(
        &self,
        f: &[f64],
        u: &mut [f64],
        tol: f64,
        maxit: usize,
        precond: &mut dyn FnMut(&[f64], &mut [f64]),
    ) -> usize {
        let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for (i, v) in u.iter_mut().enumerate() {
            if !dof_free(&self.free, i) {
                *v = 0.0;
            }
        }
        let fm: Vec<f64> = f
            .iter()
            .enumerate()
            .map(|(i, &v)| if dof_free(&self.free, i) { v } else { 0.0 })
            .collect();
        let fnorm = dot(&fm, &fm).sqrt().max(1e-300);
        let mut q = vec![0.0; u.len()];
        self.matvec(u, &mut q);
        let mut r: Vec<f64> = fm.iter().zip(&q).map(|(a, b)| a - b).collect();
        if dot(&r, &r).sqrt() <= tol * fnorm {
            return 0;
        }
        let mut z = vec![0.0; u.len()];
        precond(&r, &mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for it in 1..=maxit {
            self.matvec(&p, &mut q);
            let alpha = rz / dot(&p, &q);
            u.par_iter_mut().zip(&p).for_each(|(x, y)| *x += alpha * y);
            r.par_iter_mut().zip(&q).for_each(|(x, y)| *x -= alpha * y);
            if dot(&r, &r).sqrt() <= tol * fnorm {
                return it;
            }
            precond(&r, &mut z);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            p.par_iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
        }
        maxit
    }

    /// Per element: u_eᵀ ke u_e (unit modulus, unit size).
    fn element_energy(&self, u: &[f64]) -> Vec<f64> {
        (0..self.ee.len())
            .into_par_iter()
            .map(|e| {
                if !self.p.domain[e] {
                    return 0.0;
                }
                let nodes = self.element_nodes(e);
                let ue: [f64; 24] = std::array::from_fn(|i| u[3 * nodes[i / 3] + i % 3]);
                let mut s = 0.0;
                for i in 0..24 {
                    let row: f64 = (0..24).map(|j| self.ke[i][j] * ue[j]).sum();
                    s += ue[i] * row;
                }
                s
            })
            .collect()
    }
}

/// Density filter: weighted average over domain elements within `rmin` voxels (weights rmin - dist).
struct Filter {
    offsets: Vec<([isize; 3], f64)>,
    hs: Vec<f64>,
    /// axes whose low face is a mirror plane: neighbours past it are the mirror images
    mirror: [bool; 3],
}

impl Filter {
    fn new(p: &Problem, rmin: f64, mirror: [bool; 3]) -> Filter {
        let r = rmin.ceil() as isize;
        let mut offsets = vec![];
        for dk in -r..=r {
            for dj in -r..=r {
                for di in -r..=r {
                    let w = rmin - ((di * di + dj * dj + dk * dk) as f64).sqrt();
                    if w > 0.0 {
                        offsets.push(([di, dj, dk], w));
                    }
                }
            }
        }
        let mut f = Filter {
            offsets,
            hs: vec![],
            mirror,
        };
        let ones = vec![1.0; p.domain.len()];
        f.hs = f.apply_raw(p, &ones);
        f
    }

    fn apply_raw(&self, p: &Problem, x: &[f64]) -> Vec<f64> {
        let [ex, ey, ez] = p.n;
        (0..x.len())
            .into_par_iter()
            .map(|e| {
                if !p.domain[e] {
                    return 0.0;
                }
                let (i, j, k) = ((e % ex) as isize, ((e / ex) % ey) as isize, (e / (ex * ey)) as isize);
                let mut s = 0.0;
                let fold = |v: isize, ax: usize| if v < 0 && self.mirror[ax] { -v - 1 } else { v };
                for ([di, dj, dk], w) in &self.offsets {
                    let (a, b, c) = (fold(i + di, 0), fold(j + dj, 1), fold(k + dk, 2));
                    if a < 0 || b < 0 || c < 0 || a >= ex as isize || b >= ey as isize || c >= ez as isize {
                        continue;
                    }
                    let n = (c as usize * ey + b as usize) * ex + a as usize;
                    if p.domain[n] {
                        s += w * x[n];
                    }
                }
                s
            })
            .collect()
    }

    /// Filtered densities; kept-solid elements stay 1.
    fn densities(&self, p: &Problem, x: &[f64]) -> Vec<f64> {
        let s = self.apply_raw(p, x);
        (0..x.len())
            .map(|e| {
                if p.passive[e] {
                    1.0
                } else if p.domain[e] {
                    s[e] / self.hs[e]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Chain rule through the filter: d/dx_i = sum_e w_ie (d/dxphys_e) / hs_e.
    fn sensitivities(&self, p: &Problem, d: &[f64]) -> Vec<f64> {
        let scaled: Vec<f64> = (0..d.len())
            .map(|e| {
                if p.domain[e] && !p.passive[e] {
                    d[e] / self.hs[e]
                } else {
                    0.0
                }
            })
            .collect();
        self.apply_raw(p, &scaled)
    }
}

/// Smoothed Heaviside projection of filtered densities (Wang, Lazarov & Sigmund 2011): pushes them
/// towards 0 or 1 around the threshold `eta`, more sharply as `beta` grows. `beta == 0` is identity.
#[derive(Clone, Copy)]
struct Projection {
    beta: f64,
    eta: f64,
}

impl Projection {
    fn den(&self) -> f64 {
        (self.beta * self.eta).tanh() + (self.beta * (1.0 - self.eta)).tanh()
    }

    fn apply(&self, xt: f64) -> f64 {
        if self.beta == 0.0 {
            return xt;
        }
        ((self.beta * self.eta).tanh() + (self.beta * (xt - self.eta)).tanh()) / self.den()
    }

    fn deriv(&self, xt: f64) -> f64 {
        if self.beta == 0.0 {
            return 1.0;
        }
        let t = (self.beta * (xt - self.eta)).tanh();
        self.beta * (1.0 - t * t) / self.den()
    }

    /// Physical densities from filtered ones: design elements projected, kept-solid stay 1.
    fn densities(&self, p: &Problem, xt: &[f64]) -> Vec<f64> {
        (0..xt.len())
            .map(|e| {
                if p.domain[e] && !p.passive[e] {
                    self.apply(xt[e])
                } else {
                    xt[e]
                }
            })
            .collect()
    }
}

/// Timings (s) and solver work of one optimizer iteration.
#[derive(Clone, Debug, Default)]
pub struct IterStats {
    /// building the preconditioner (multigrid hierarchy or Jacobi diagonal)
    pub setup: f64,
    /// CG solves and element energies, all load cases
    pub solve: f64,
    /// sensitivities, filter, projection and the OC update
    pub update: f64,
    /// writing the .vti
    pub io: f64,
    /// CG iterations per load case
    pub cg: Vec<usize>,
    /// volume evaluations (filter + projection) in the OC update
    pub oc_steps: usize,
    /// CG tolerance used
    pub cg_tol: f64,
    pub compliance: f64,
}

pub struct Outcome {
    pub iterations: usize,
    pub compliance: f64,
    pub max_disp: f64,
    pub stats: Vec<IterStats>,
    /// the design variables x after the last update, per element
    pub design: Vec<f64>,
    /// the projection's β for the next iteration
    pub beta: f64,
    /// the loop's state for the next iteration (to resume a stopped run)
    pub(crate) state: LoopState,
    /// Stop was requested: the run ended before converging
    pub stopped: bool,
}

/// What an optimizer loop carries from one iteration to the next besides the design: enough to
/// resume a stopped run where it would have continued.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct LoopState {
    control: Control,
    /// OC's Lagrange multiplier (the next bisection brackets it)
    lambda: f64,
    /// the last maximum density change (sets the adaptive CG tolerance)
    last_change: f64,
}

impl Default for LoopState {
    fn default() -> Self {
        LoopState {
            control: Control::default(),
            lambda: 1.0,
            last_change: 1.0,
        }
    }
}

/// Where an optimizer stage starts, and how it reports.
#[derive(Default)]
pub(crate) struct Stage {
    /// design variables per element and the projection's β (None: uniform `volfrac`, β = 1)
    start: Option<(Vec<f64>, f64)>,
    /// iterations done by earlier stages (the iteration lines continue their numbering)
    it0: usize,
    /// an earlier stage: no "done" line (the app reads it as the end of the run)
    quiet: bool,
    /// the loop's state (resuming a stopped run; otherwise fresh)
    state: LoopState,
}

const CG_MAXIT: usize = 20_000;

/// z = M⁻¹ r
type Precond<'a> = dyn FnMut(&[f64], &mut [f64]) + 'a;

/// Run the optimization, writing the design to `out` (cell `density`, cell `density_filtered` (x̂,
/// what gets meshed), cell `energy` in MPa, point `displacement` in mm). The file is rewritten as often as keeps writing under ~10% of the run
/// time (every iteration on small grids), and always after the last iteration.
pub fn optimize(p: &Problem, s: &Settings, out: &Path, gpu: Option<&Gpu>) -> Result<Outcome> {
    optimize_stage(&Model::single(p.clone()), s, out, gpu, &Stage::default())
}

/// The mirror planes the run uses: the shape's (editor) planes, unless `--no-symmetry`.
fn mirror_axes(shape: &Shape, s: &Settings) -> [bool; 3] {
    if !s.symmetry || !shape.editor {
        return [false; 3];
    }
    let mut axes = shape.mirror.map(|m| m.is_some());
    // printing breaks the symmetry about a plane across the build direction
    let b = s.up.axis();
    if s.overhang && axes[b] {
        println!(
            "symmetry: not using the mirror plane in {} (overhang control prints along {})",
            ["x", "y", "z"][b],
            s.up.name()
        );
        axes[b] = false;
    }
    axes
}

/// Fewest domain elements (of the whole part) for solving one side of the mirror planes. Below
/// this each hierarchy's fixed costs (dispatches, the coarsest level's readback) outweigh the
/// smaller grid: the table's 26k-element coarse stage took 5.8 s as a quarter, 3.6 s whole.
const SYMMETRY_MIN_ELEMENTS: usize = 100_000;

/// The model of a set-up problem: reduced to one side of its mirror planes when it is symmetric
/// about them (see `symmetry.rs`) and large enough, else the problem itself.
fn model(full: Problem, shape: &Shape, s: &Settings) -> Model {
    let axes = mirror_axes(shape, s);
    if !axes.iter().any(|a| *a) {
        return Model::single(full);
    }
    if full.domain.iter().filter(|d| **d).count() < SYMMETRY_MIN_ELEMENTS {
        return Model::single(full);
    }
    match symmetry::reduce(&full, axes) {
        Ok(m) => {
            if let Some(d) = m.describe() {
                println!("{d}");
            }
            m
        }
        Err(why) => {
            println!("symmetry not used: {why}; solving the whole part");
            Model::single(full)
        }
    }
}

/// The checkpoint a stopped run writing `out` leaves (it exists while there is a run to resume).
pub fn resume_path(out: &Path) -> std::path::PathBuf {
    checkpoint::path(out)
}

/// Fewest domain elements of the coarse grid for a coarse start. Below this the coarse grid loses
/// thin features, and supports and loads (1.5 elements thick) get much thicker than on the fine grid.
const COARSE_MIN_ELEMENTS: usize = 10_000;

/// Set up and optimize `shape`.
///
/// With `coarse_start` and projection continuation, the β = 1 stage (where the layout forms, about
/// half the iterations) runs on a grid of twice the element size first: 1/8 of the elements, until
/// its compliance stalls. The fine grid then starts from that design (interpolated) at β = 2. The
/// filter radius stays `rmin` voxels on both grids, so it is twice as wide on the coarse one.
pub fn run(shape: &Shape, s: &Settings, out: &Path, gpu: &Gpu, resume: bool) -> Result<Outcome> {
    let t_start = Instant::now();
    let ck = if resume {
        Some(checkpoint::read(out, s)?)
    } else {
        checkpoint::remove(out);
        None
    };
    let fine = model(setup(shape, s, gpu)?, shape, s);
    // the whole part's elements (the coarse grid's resolution is what matters, not the model's size)
    let ndomain = (fine.p().domain.iter().filter(|d| **d).count() as f64 * fine.scale()) as usize;
    let coarse_start = s.coarse_start && s.beta_max > 1.0 && ndomain / 8 >= COARSE_MIN_ELEMENTS;
    let resumed = |c: &checkpoint::Checkpoint, p: &Model, grid: &str| -> Result<Stage> {
        c.check(p)?;
        let h = &c.header;
        println!(
            "resuming after iteration {} ({grid} grid, beta {})",
            h.iterations, h.beta
        );
        Ok(Stage {
            start: Some((c.x.clone(), h.beta)),
            it0: h.iterations,
            quiet: true,
            state: h.state.clone(),
        })
    };
    // --iters caps the whole run
    let left = |done: usize| Settings {
        iters: s.iters.saturating_sub(done).max(1),
        ..s.clone()
    };
    let mut earlier = vec![];
    let stage = match &ck {
        Some(c) if !c.header.coarse => resumed(c, &fine, "fine")?,
        _ if coarse_start => {
            let cs = Settings {
                voxel: 2.0 * s.voxel,
                beta_max: 1.0,
                ..s.clone()
            };
            println!("coarse start: the beta 1 stage at {:.3} mm", cs.voxel);
            let coarse = model(setup(shape, &cs, gpu)?, shape, &cs);
            let stage = match &ck {
                Some(c) => resumed(c, &coarse, "coarse")?,
                None => Stage {
                    quiet: true,
                    ..Default::default()
                },
            };
            let cs = Settings {
                iters: left(stage.it0).iters,
                ..cs
            };
            let mut rc = optimize_stage(&coarse, &cs, out, Some(gpu), &stage)?;
            let done = stage.it0 + rc.iterations;
            if rc.stopped {
                checkpoint::write(out, s, &coarse, true, done, rc.beta, &rc.state, &rc.design)?;
                rc.iterations = done;
                print_done(&rc, t_start, out);
                return Ok(rc);
            }
            let x = prolong_design(coarse.p(), &rc.design, fine.p(), s.volfrac);
            println!("fine grid: continuing from the coarse design");
            earlier = rc.stats;
            Stage {
                start: Some((x, 2.0)),
                it0: done,
                quiet: true,
                ..Default::default()
            }
        }
        Some(_) => bail!("the checkpoint is from a coarse start, which no longer applies; run without --resume"),
        None => Stage {
            quiet: true,
            ..Default::default()
        },
    };
    let mut rf = optimize_stage(&fine, &left(stage.it0), out, Some(gpu), &stage)?;
    let done = stage.it0 + rf.iterations;
    if rf.stopped {
        checkpoint::write(out, s, &fine, false, done, rf.beta, &rf.state, &rf.design)?;
    } else {
        checkpoint::remove(out);
    }
    rf.iterations = done;
    rf.stats.splice(0..0, earlier);
    print_done(&rf, t_start, out);
    Ok(rf)
}

/// Design variables on `fine` from those on `coarse` (each cropped on its own, so their origins
/// may differ): trilinear in the coarse cell
/// centres, over the coarse domain cells only. Kept-solid elements get 1, others outside the domain 0.
fn prolong_design(coarse: &Problem, xc: &[f64], fine: &Problem, volfrac: f64) -> Vec<f64> {
    let [cx, cy, cz] = coarse.n;
    let r = fine.h / coarse.h;
    let [fx, fy, _] = fine.n;
    (0..fine.domain.len())
        .map(|e| {
            if fine.passive[e] {
                return 1.0;
            }
            if !fine.domain[e] {
                return 0.0;
            }
            let ijk = [e % fx, (e / fx) % fy, e / (fx * fy)];
            // the fine centre in coarse cell-centre coordinates, split into a base cell and weights
            let mut base = [0isize; 3];
            let mut t = [0.0; 3];
            for c in 0..3 {
                let pos = fine.origin[c] - coarse.origin[c];
                let u = (pos / fine.h + ijk[c] as f64 + 0.5) * r - 0.5;
                base[c] = u.floor() as isize;
                t[c] = u - u.floor();
            }
            let (mut sum, mut wsum) = (0.0, 0.0);
            for b in 0..8 {
                let o = [b & 1, b >> 1 & 1, b >> 2 & 1];
                let q: [isize; 3] = std::array::from_fn(|c| base[c] + o[c] as isize);
                if q.iter().zip([cx, cy, cz]).any(|(&v, n)| v < 0 || v >= n as isize) {
                    continue;
                }
                let ce = (q[2] as usize * cy + q[1] as usize) * cx + q[0] as usize;
                if !coarse.domain[ce] {
                    continue;
                }
                let w: f64 = (0..3).map(|c| if o[c] == 1 { t[c] } else { 1.0 - t[c] }).product();
                sum += w * xc[ce];
                wsum += w;
            }
            if wsum > 1e-9 { sum / wsum } else { volfrac }
        })
        .collect()
}

fn optimize_stage(m: &Model, s: &Settings, out: &Path, gpu: Option<&Gpu>, stage: &Stage) -> Result<Outcome> {
    output::catch_stop();
    let p = m.p();
    let bases: Vec<LoadBasis> = m.groups.iter().map(|g| g.load_basis()).collect();
    if let Some(d) = LoadBasis::describe_all(&bases) {
        println!("{d}");
    }
    if let Solver::Gpu(ms) = &s.solver {
        let Some(gpu) = gpu else {
            bail!("the GPU solver needs a GPU");
        };
        return gpu_opt::optimize_gpu(m, &bases, s, ms, out, gpu, stage);
    }
    let ke = hex_ke(s.nu);
    let ne = p.domain.len();
    let design: Vec<usize> = (0..ne).filter(|&e| p.domain[e] && !p.passive[e]).collect();
    if design.is_empty() {
        bail!("no design elements: the domain is empty or entirely kept solid");
    }
    // per group (one without symmetry): free DOFs, loads per case
    let frees: Vec<Vec<u8>> = m.groups.iter().map(|g| g.free_mask()).collect();
    let fs: Vec<Vec<Vec<f64>>> = m
        .groups
        .iter()
        .map(|g| g.loads.iter().map(|l| l.iter().flatten().copied().collect()).collect())
        .collect();
    let emin = s.e0 * s.emin;
    let filter = Filter::new(p, s.rmin, m.mirror_axes());
    let am = s.overhang.then(|| overhang::Am::new(p, m.mirror_axes(), s.up));
    // x̃ → x̂: the AM filter's printed densities (with its forward pass), or x̃ itself
    let print = |xtilde: Vec<f64>| -> (Vec<f64>, Option<overhang::Printed>) {
        match &am {
            Some(am) => {
                let fw = am.forward(p, &xtilde);
                (fw.xi.clone(), Some(fw))
            }
            None => (xtilde, None),
        }
    };
    let mut proj = Projection {
        beta: if s.beta_max > 0.0 { 1.0 } else { 0.0 },
        eta: 0.5,
    };

    let mut x: Vec<f64> = (0..ne)
        .map(|e| {
            if p.passive[e] {
                1.0
            } else if p.domain[e] {
                s.volfrac
            } else {
                0.0
            }
        })
        .collect();
    if let Some((x0, beta)) = &stage.start {
        x.copy_from_slice(x0);
        proj.beta = *beta;
    }
    let mut xtilde = filter.densities(p, &x);
    let (mut xhat, mut printed) = print(xtilde.clone());
    let mut xphys = proj.densities(p, &xhat);
    let mut us: Vec<Vec<Vec<f64>>> = fs
        .iter()
        .map(|g| g.iter().map(|f| vec![0.0; f.len()]).collect())
        .collect();
    let t_start = Instant::now();
    let mut control = stage.state.control.clone();
    let mut result = Outcome {
        iterations: 0,
        compliance: 0.0,
        max_disp: 0.0,
        stats: vec![],
        design: vec![],
        beta: 0.0,
        state: LoopState::default(),
        stopped: false,
    };
    println!(
        "SIMP: volume fraction {}, penalty {}, filter radius {} voxels, E = {} MPa, nu = {}, {} load case{}, {}",
        s.volfrac,
        s.penal,
        s.rmin,
        s.e0,
        s.nu,
        p.loads.len(),
        if p.loads.len() == 1 {
            ""
        } else {
            "s (summed compliance)"
        },
        if s.beta_max > 0.0 {
            format!("Heaviside projection up to beta {}", s.beta_max)
        } else {
            "no projection".into()
        }
    );
    if let Some(am) = &am {
        println!(
            "overhang: at most 45°, printed {} up from layer {}",
            s.up.name(),
            am.base
        );
    }
    // the last iteration's output, until it is written
    // the last iteration's full fields, until written
    let mut pending: Option<Snapshot> = None;
    let mut output = output::Output::new(m, out);
    // OC's Lagrange multiplier, for the next iteration's bracket
    let mut lambda = stage.state.lambda;
    let mut last_change = stage.state.last_change;
    for it in 1..=s.iters {
        let t0 = Instant::now();
        let mut st = IterStats::default();
        let ee: Vec<f64> = (0..ne)
            .map(|e| {
                if p.domain[e] {
                    (emin + xphys[e].powf(s.penal) * (s.e0 - emin)) * p.h
                } else {
                    0.0
                }
            })
            .collect();
        let final_stage = proj.beta > 0.0 && proj.beta >= s.beta_max;
        let tol = if !s.adaptive_tol || final_stage {
            s.cg_tol
        } else {
            (0.05 * last_change).clamp(s.cg_tol, 1e-2)
        };
        st.cg_tol = tol;
        let mut ce = vec![0.0; ne];
        let t1 = Instant::now();
        // per group (its own supports): one preconditioner, shared by its load cases
        for (g, group) in m.groups.iter().enumerate() {
            let t0 = Instant::now();
            let fem = Fem {
                p: group,
                ke: &ke,
                ee: ee.clone(),
                free: frees[g].clone(),
            };
            let (basis, fs, us) = (&bases[g], &fs[g], &mut us[g]);
            let mut precond: Box<Precond> = match &s.solver {
                Solver::Multigrid(ms) => {
                    let mut m = mg::Multigrid::new(&fem, ms);
                    if it == 1 && g == 0 {
                        let c = m.coarsest();
                        println!(
                            "multigrid: {} levels, coarsest {}x{}x{} elements, {:?}",
                            m.levels(),
                            c[0],
                            c[1],
                            c[2],
                            ms.smoother
                        );
                    }
                    Box::new(move |r: &[f64], z: &mut [f64]| m.precondition(r, z))
                }
                _ => {
                    let inv = fem.inv_diag();
                    Box::new(move |r: &[f64], z: &mut [f64]| {
                        z.par_iter_mut().zip(r).zip(&inv).for_each(|((z, r), d)| *z = r * d)
                    })
                }
            };
            st.setup += t0.elapsed().as_secs_f64();
            // each load case separately; the objective is their summed compliance
            for &c in &basis.solved {
                let n = fem.solve(&fs[c], &mut us[c], tol, CG_MAXIT, &mut *precond);
                if n >= CG_MAXIT {
                    println!("warning: CG did not converge in {CG_MAXIT} iterations");
                }
                st.cg.push(n);
            }
            for (c, how) in basis.cases.iter().enumerate() {
                if let CaseSolve::Combined(a) = how {
                    let mut u = vec![0.0; us[c].len()];
                    for &(slot, w) in a {
                        let us_ = &us[basis.solved[slot]];
                        u.par_iter_mut().zip(us_).for_each(|(v, s)| *v += w * s);
                    }
                    us[c] = u;
                }
                for (a, b) in ce.iter_mut().zip(fem.element_energy(&us[c])) {
                    *a += b;
                }
            }
        }
        st.solve = t1.elapsed().as_secs_f64() - st.setup;
        let t2 = Instant::now();
        let compliance: f64 = (0..ne).map(|e| ee[e] * ce[e]).sum();
        // reported for the whole part (the reduced model's is 1/2^planes of it)
        let full_compliance = compliance * m.scale();
        st.compliance = full_compliance;
        let dc: Vec<f64> = (0..ne)
            .map(|e| {
                if p.domain[e] && !p.passive[e] {
                    -s.penal * xphys[e].powf(s.penal - 1.0) * (s.e0 - emin) * p.h * ce[e]
                } else {
                    0.0
                }
            })
            .collect();
        // chain rule: projection, AM filter, then density filter (the volume sensitivity changes
        // with the projection and the AM filter)
        let dproj: Vec<f64> = (0..ne).map(|e| proj.deriv(xhat[e])).collect();
        let dc: Vec<f64> = (0..ne).map(|e| dc[e] * dproj[e]).collect();
        let (dc, dv) = match (&am, &printed) {
            (Some(am), Some(fw)) => (
                am.sensitivities(p, &xtilde, fw, &dc),
                am.sensitivities(p, &xtilde, fw, &dproj),
            ),
            _ => (dc, dproj),
        };
        let dc = filter.sensitivities(p, &dc);
        let dv = filter.sensitivities(p, &dv);
        let used_xphys = xphys.clone();
        let used_xhat = xhat.clone();

        // optimality criteria: bisect the Lagrange multiplier to meet the volume. With dc scaled by
        // the compliance, λ is O(1) and moves little between iterations, so the search starts from
        // a bracket around the previous λ.
        let target = s.volfrac * design.len() as f64;
        let mv = 0.2;
        let mut xnew = x.clone();
        // by the whole part's compliance, so λ is the same with or without symmetry
        let cscale = 1.0 / full_compliance.abs().max(1e-300);
        let mut volume_at = |lam: f64, xnew: &mut Vec<f64>| {
            st.oc_steps += 1;
            for &e in &design {
                let be = (-dc[e] * cscale).max(0.0) / (dv[e] * lam).max(1e-300);
                xnew[e] = (x[e] * be.sqrt()).clamp((x[e] - mv).max(0.0), (x[e] + mv).min(1.0));
            }
            let trial = proj.densities(p, &print(filter.densities(p, xnew)).0);
            design.iter().map(|&e| trial[e]).sum::<f64>()
        };
        // volume falls as λ grows: find l1 < l2 with vol(l1) > target >= vol(l2)
        let (mut l1, mut l2) = (lambda / 2.0, lambda * 2.0);
        for _ in 0..60 {
            if volume_at(l1, &mut xnew) > target {
                break;
            }
            l2 = l1;
            l1 /= 8.0;
        }
        for _ in 0..60 {
            if volume_at(l2, &mut xnew) <= target {
                break;
            }
            l1 = l2;
            l2 *= 8.0;
        }
        while (l2 - l1) / (l1 + l2) > 1e-4 {
            let lmid = 0.5 * (l1 + l2);
            if volume_at(lmid, &mut xnew) > target {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
        lambda = 0.5 * (l1 + l2);
        volume_at(lambda, &mut xnew);
        let change = design.iter().map(|&e| (xnew[e] - x[e]).abs()).fold(0.0, f64::max);
        last_change = change;
        xtilde = filter.densities(p, &xnew);
        (xhat, printed) = print(xtilde.clone());
        xphys = proj.densities(p, &xhat);
        x = xnew;

        // per case and mirror side: the groups' parts with their signs there
        let mut max_disp = 0.0f64;
        for c in 0..m.cases() {
            for r in m.regions() {
                let signs = m.signs(r);
                let u0 = &us[0][c];
                let mx = (0..u0.len() / 3)
                    .into_par_iter()
                    .map(|i| {
                        let mut v = [0.0; 3];
                        for (g, sg) in signs.iter().enumerate() {
                            for (k, vk) in v.iter_mut().enumerate() {
                                *vk += sg * us[g][c][3 * i + k];
                            }
                        }
                        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
                    })
                    .reduce(|| 0.0, f64::max);
                max_disp = max_disp.max(mx);
            }
        }
        let grey = design
            .iter()
            .filter(|&&e| used_xphys[e] > 0.1 && used_xphys[e] < 0.9)
            .count() as f64
            / design.len() as f64;
        st.update = t2.elapsed().as_secs_f64();

        // the densities this solve used, with their energy density and displacements
        let t3 = Instant::now();
        let snap = Snapshot {
            density: used_xphys.iter().map(|&v| v as f32).collect(),
            filtered: output::filtered_out(p, used_xhat.iter().map(|&v| v as f32)),
            energy: (0..ne).map(|e| (0.5 * ee[e] * ce[e] / p.h.powi(3)) as f32).collect(),
            // per group, per case (the output combines them)
            disps: us
                .iter()
                .flat_map(|g| g.iter().map(|u| u.iter().map(|&v| v as f32).collect()))
                .collect(),
        };
        if output.due()? {
            if output.live_is_full() {
                output.live(snap);
                pending = None;
            } else {
                output.live(Snapshot {
                    density: snap.density.clone(),
                    filtered: vec![],
                    energy: vec![],
                    disps: vec![],
                });
                pending = Some(snap);
            }
        } else {
            pending = Some(snap);
        }
        st.io = t3.elapsed().as_secs_f64();

        print_iteration(
            stage.it0 + it,
            full_compliance,
            max_disp,
            change,
            grey,
            proj.beta,
            &st,
            t0,
        );
        result.iterations = it;
        result.compliance = full_compliance;
        result.max_disp = max_disp;
        result.stats.push(st);
        match control.next(s, proj.beta, compliance, change) {
            Next::Go => {}
            Next::Beta(b) => {
                proj.beta = b;
                xphys = proj.densities(p, &xhat);
            }
            Next::Stop => break,
        }
        // after the stopping rule, so a resumed run continues exactly where this one would have
        if output::stop_requested() {
            println!("stopped: writing the result of iteration {}", stage.it0 + it);
            result.stopped = true;
            break;
        }
    }
    // also without overhang control (when it can tell: no mirror plane across the build direction)
    if !m.mirror_axes()[s.up.axis()] {
        overhang::Am::new(p, m.mirror_axes(), s.up).report(p, &xphys);
    }
    result.beta = proj.beta;
    result.state = LoopState {
        control,
        lambda,
        last_change,
    };
    match pending {
        Some(snap) => {
            output.finish(snap)?;
        }
        None => output.wait()?,
    }
    result.design = x;
    if !stage.quiet {
        print_done(&result, t_start, out);
    }
    Ok(result)
}

fn print_done(result: &Outcome, t_start: Instant, out: &Path) {
    println!(
        "done in {:.1} s: {} iterations, compliance {:.4e} N·mm, max displacement {:.4} mm",
        t_start.elapsed().as_secs_f64(),
        result.iterations,
        result.compliance,
        result.max_disp
    );
    println!("wrote {} (iteration {})", out.display(), result.iterations);
}

/// The iteration line (`jobs::opt_progress` parses it).
#[allow(clippy::too_many_arguments)]
fn print_iteration(
    it: usize,
    compliance: f64,
    max_disp: f64,
    change: f64,
    grey: f64,
    beta: f64,
    st: &IterStats,
    t0: Instant,
) {
    let beta = if beta > 0.0 {
        format!(", beta {beta}")
    } else {
        String::new()
    };
    println!(
        "it {it:3}: compliance {compliance:.4e} N·mm, max displacement {max_disp:.4} mm, change {change:.3}, grey {:.0}%{beta}, {} CG its, {:.1} s [setup {:.2} solve {:.2} update {:.2} io {:.2}]",
        grey * 100.0,
        st.cg.iter().sum::<usize>(),
        t0.elapsed().as_secs_f64(),
        st.setup,
        st.solve,
        st.update,
        st.io,
    );
}

/// What to do after an iteration.
enum Next {
    Go,
    /// sharpen the projection to this β
    Beta(f64),
    Stop,
}

/// Convergence and β continuation, shared by the CPU and GPU loops.
#[derive(Default, Clone, Serialize, Deserialize)]
struct Control {
    stalled: usize,
    stage_its: usize,
    last_compliance: f64,
    /// the final stage's lowest compliance, and the iterations since it last improved by over 0.1%
    best: f64,
    since_best: usize,
}

impl Control {
    fn next(&mut self, s: &Settings, beta: f64, compliance: f64, change: f64) -> Next {
        // the max density change often hovers at the move limit while the design no longer
        // improves, so also stop once compliance has stagnated
        let improvement = (self.last_compliance - compliance) / compliance;
        self.last_compliance = compliance;
        self.stalled = if self.stage_its > 0 && improvement.abs() < 1e-3 {
            self.stalled + 1
        } else {
            0
        };
        // at the final β (sharp projection), OC tends to oscillate: compliance alternates by ~0.1%,
        // which keeps resetting `stalled` without the design improving
        if self.stage_its == 0 || compliance < self.best * (1.0 - 1e-3) {
            self.best = compliance;
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        self.stage_its += 1;
        let converged = change < s.tol || self.stalled >= 3;
        // continuation: sharpen the projection each time the design settles (or after 25 iterations)
        if beta > 0.0 && beta < s.beta_max && (converged || self.stage_its >= 25) {
            let b = (beta * 2.0).min(s.beta_max);
            println!("beta -> {b}");
            self.stalled = 0;
            self.stage_its = 0;
            return Next::Beta(b);
        }
        if change < s.tol {
            println!("converged (density change < {})", s.tol);
            return Next::Stop;
        }
        if self.stalled >= 3 {
            println!("converged (compliance changed < 0.1% for 3 iterations)");
            return Next::Stop;
        }
        let final_stage = beta == 0.0 || beta >= s.beta_max;
        if final_stage && self.since_best >= 3 {
            println!("converged (compliance hasn't improved by 0.1% in 3 iterations)");
            return Next::Stop;
        }
        Next::Go
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// An all-domain grid with no supports or loads.
    pub(in crate::topopt) fn grid(n: [usize; 3], h: f64) -> Problem {
        Problem {
            n,
            h,
            origin: [0.0; 3],
            domain: vec![true; n.iter().product()],
            passive: vec![false; n.iter().product()],
            fixed: vec![0; n.map(|v| v + 1).iter().product()],
            loads: vec![],
            cases: vec![],
        }
    }

    #[test]
    fn prolonged_design_interpolates_inside_the_domain() {
        // coarse 4x2x2 at h = 2, fine 8x4x4 at h = 1, same origin; the coarse design is a ramp in x
        let mut coarse = grid([4, 2, 2], 2.0);
        let mut fine = grid([8, 4, 4], 1.0);
        let xc: Vec<f64> = (0..16).map(|e| 0.1 + 0.2 * (e % 4) as f64).collect();
        let x = prolong_design(&coarse, &xc, &fine, 0.3);
        // inside, linear in x: fine centres 1.5 .. 5.5 lie between coarse centres 1 .. 7
        for i in 0..8 {
            let want = (0.1 + 0.2 * ((i as f64 + 0.5) / 2.0 - 0.5)).clamp(0.1, 0.7);
            assert!((x[i] - want).abs() < 1e-12, "i {i}: {} vs {want}", x[i]);
        }
        // coarse cells outside the domain don't pull the values down; fine kept-solid stays 1 and
        // fine non-domain 0
        coarse.domain[3] = false;
        fine.passive[1] = true;
        fine.domain[2] = false;
        let x = prolong_design(&coarse, &xc, &fine, 0.3);
        // fine 6 (centre 6.5) sits between coarse 2 and the removed 3: all of coarse 2's value
        assert!((x[6] - 0.5).abs() < 1e-12, "{}", x[6]);
        // fine 7 has no coarse domain cell nearby: volfrac
        assert_eq!(x[7], 0.3);
        assert_eq!((x[1], x[2]), (1.0, 0.0));
        // no coarse domain cell nearby: volfrac
        coarse.domain.fill(false);
        assert_eq!(prolong_design(&coarse, &xc, &fine, 0.3)[0], 0.3);
    }

    /// A beam on supports at both ends of its bottom, symmetric about the middle planes x and y
    /// (an even number of elements on both). Its load cases are a down load at the top middle
    /// (symmetric), a sideways push in y there (antisymmetric about the y plane) and a push in x
    /// and y on one end's top (no symmetry), each with its mirror images as further cases, so the
    /// problem as a whole (and so the optimized design) is symmetric while each single case isn't:
    /// the reduced model splits them into parts of all four patterns.
    pub(in crate::topopt) fn symmetric_beam() -> Problem {
        let n = [16, 6, 6];
        let mut p = grid(n, 1.0);
        let [nx, ny, nz] = n.map(|v| v + 1);
        let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let mut base = vec![vec![[0.0; 3]; nx * ny * nz]; 3];
        for j in 0..ny {
            for i in [0, 1, nx - 2, nx - 1] {
                p.fixed[node(i, j, 0)] = 7;
            }
            for i in nx / 2 - 1..=nx / 2 + 1 {
                base[0][node(i, j, nz - 1)] = [0.0, 0.0, -1.0];
                base[1][node(i, j, nz - 1)] = [0.0, 0.5, 0.0];
            }
            base[2][node(nx - 3, j, nz - 1)] = [0.3, 0.2 * (j + 1) as f64 / ny as f64, -0.2];
        }
        // each case and its distinct mirror images about x = 8 and y = 3
        let image = |f: &Vec<[f64; 3]>, g: usize| -> Vec<[f64; 3]> {
            let mut out = vec![[0.0; 3]; f.len()];
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let (si, sj) = (
                            if g & 1 == 1 { nx - 1 - i } else { i },
                            if g & 2 == 2 { ny - 1 - j } else { j },
                        );
                        let v = f[node(si, sj, k)];
                        out[node(i, j, k)] = [
                            if g & 1 == 1 { -v[0] } else { v[0] },
                            if g & 2 == 2 { -v[1] } else { v[1] },
                            v[2],
                        ];
                    }
                }
            }
            out
        };
        for f in &base {
            for g in 0..4 {
                let im = image(f, g);
                if !p.loads.contains(&im) {
                    p.loads.push(im);
                }
            }
        }
        p.cases = (1..=p.loads.len() as u32).collect();
        p
    }

    pub(in crate::topopt) fn beam_settings(solver: Solver) -> Settings {
        Settings {
            voxel: 1.0,
            volfrac: 0.3,
            penal: 3.0,
            rmin: 1.5,
            iters: 10,
            tol: 0.0,
            e0: 3500.0,
            nu: 0.35,
            force: 0.0,
            beta_max: 8.0,
            solver,
            emin: 1e-9,
            cg_tol: 1e-9,
            adaptive_tol: false,
            ground: Ground::None,
            coarse_start: false,
            symmetry: true,
            overhang: false,
            up: crate::mesh::Up::PlusZ,
        }
    }

    /// Solving a quarter of the symmetric beam (mirrored in x and y; the loads split into
    /// symmetric and antisymmetric parts) gives the same run as solving all of it: compliance and
    /// maximum displacement per iteration, the written density and displacements.
    #[test]
    fn symmetric_quarter_matches_the_whole_cpu() {
        let full = symmetric_beam();
        let m = symmetry::reduce(&full, [true, true, false]).unwrap();
        let sym = m.sym.as_ref().unwrap();
        assert_eq!(
            (sym.n, m.groups.len(), m.cases()),
            ([8, 3, 6], 4, 7),
            "all four patterns have a part"
        );
        let dir = std::env::temp_dir().join(format!("gen3d-sym-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = beam_settings(Solver::Multigrid(mg::MgSettings::default()));
        let whole = optimize_stage(
            &Model::single(full),
            &s,
            &dir.join("whole.vti"),
            None,
            &Stage::default(),
        )
        .unwrap();
        let quarter = optimize_stage(&m, &s, &dir.join("quarter.vti"), None, &Stage::default()).unwrap();
        compare_runs(&whole, &quarter, &dir, 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two runs of the same problem agree per iteration, and in the written fields.
    pub(in crate::topopt) fn compare_runs(a: &Outcome, b: &Outcome, dir: &Path, tol: f64) {
        assert_eq!(a.iterations, b.iterations);
        for (it, (x, y)) in a.stats.iter().zip(&b.stats).enumerate() {
            let rel = (x.compliance - y.compliance).abs() / x.compliance;
            assert!(
                rel < tol,
                "iteration {}: compliance {} vs {}",
                it + 1,
                x.compliance,
                y.compliance
            );
        }
        let rel = (a.max_disp - b.max_disp).abs() / a.max_disp;
        assert!(rel < tol.max(1e-5), "max displacement {} vs {}", a.max_disp, b.max_disp);
        let read = |f: &str| crate::vti::read(&dir.join(f)).unwrap();
        let (va, vb) = (read("whole.vti"), read("quarter.vti"));
        for name in [
            "density",
            "displacement",
            "displacement_2",
            "displacement_3",
            "displacement_7",
        ] {
            let get = |v: &crate::vti::VtiFile| v.arrays.iter().find(|a| a.name == name).unwrap().data.clone();
            let (da, db) = (get(&va), get(&vb));
            assert_eq!(da.len(), db.len(), "{name}");
            let scale = da.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = da.iter().zip(&db).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(
                diff <= (tol as f32).max(1e-5) * 10.0 * scale,
                "{name} differs by {diff} (of {scale})"
            );
        }
    }

    /// Rocked variants (f ± r·g) are formed from the upright case and the sideways push; loads on
    /// clamped DOFs don't count.
    #[test]
    fn load_basis_solves_only_independent_cases() {
        let n = [2, 1, 1];
        let nn = 3 * 2 * 2;
        let mut f = vec![[0.0; 3]; nn];
        let mut g = vec![[0.0; 3]; nn];
        f[1] = [0.0, 0.0, -2.0];
        f[5] = [0.0, 0.0, -1.0];
        g[1] = [1.0, 0.0, 0.0];
        g[5] = [0.5, 0.0, 0.0];
        let rocked = |s: f64| -> Vec<[f64; 3]> {
            f.iter()
                .zip(&g)
                .map(|(a, b)| std::array::from_fn(|c| a[c] + s * 0.5 * b[c]))
                .collect()
        };
        // a load on a clamped node only: no load at all
        let mut clamped = vec![[0.0; 3]; nn];
        clamped[0] = [0.0, 0.0, 5.0];
        let mut fixed = vec![0; nn];
        fixed[0] = 7;
        let p = Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            domain: vec![true; 2],
            passive: vec![false; 2],
            fixed,
            loads: vec![f.clone(), rocked(1.0), rocked(-1.0), g.clone(), clamped],
            cases: vec![1, 2, 3, 4, 5],
        };
        let b = p.load_basis();
        assert_eq!(b.solved, vec![0, 1]);
        let combo = |c: usize| match &b.cases[c] {
            CaseSolve::Combined(a) => a.clone(),
            CaseSolve::Solved(_) => panic!("case {c} is solved"),
        };
        let close = |a: Vec<(usize, f64)>, want: &[(usize, f64)]| {
            a.len() == want.len() && a.iter().zip(want).all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-9)
        };
        // f - 0.5 g = 2 f - (f + 0.5 g); g = 2 ((f + 0.5 g) - f)
        assert!(close(combo(2), &[(0, 2.0), (1, -1.0)]), "{:?}", combo(2));
        assert!(close(combo(3), &[(0, -2.0), (1, 2.0)]), "{:?}", combo(3));
        assert!(combo(4).is_empty());
        assert!(matches!(b.cases[1], CaseSolve::Solved(1)));
    }

    #[test]
    fn oscillation_at_the_final_beta_stops_the_run() {
        let s = Settings {
            voxel: 1.0,
            volfrac: 0.3,
            penal: 3.0,
            rmin: 1.5,
            iters: 100,
            tol: 0.01,
            e0: 1.0,
            nu: 0.3,
            force: 1.0,
            beta_max: 32.0,
            solver: Solver::Jacobi,
            emin: 1e-6,
            cg_tol: 1e-4,
            adaptive_tol: false,
            ground: Ground::None,
            coarse_start: false,
            symmetry: true,
            overhang: false,
            up: crate::mesh::Up::PlusZ,
        };
        // iterations until `next` says Stop at β, for a compliance sequence
        let run = |beta: f64, seq: &dyn Fn(usize) -> f64| {
            let mut c = Control::default();
            for i in 0..40 {
                match c.next(&s, beta, seq(i), 0.2) {
                    Next::Go => {}
                    Next::Beta(_) => return None,
                    Next::Stop => return Some(i + 1),
                }
            }
            Some(usize::MAX)
        };
        // alternating by ~0.1% around a flat level: no new best after the first iteration
        let osc = |i: usize| if i.is_multiple_of(2) { 8.28 } else { 8.29 };
        assert_eq!(run(32.0, &osc), Some(4));
        // a slow steady decline (0.06% per iteration) keeps going until the old stall rule
        let slow = |i: usize| 10.0 * 0.9994f64.powi(i as i32);
        assert_eq!(run(32.0, &slow), Some(4));
        let slower = |i: usize| 10.0 * (0.998f64).powi(i as i32);
        assert_eq!(run(32.0, &slower), Some(usize::MAX));
        // an intermediate stage never stops on it: it waits for the 25-iteration cap
        assert_eq!(run(4.0, &osc), None);
    }

    #[test]
    fn projection_endpoints_and_derivative() {
        for beta in [0.0, 1.0, 8.0, 32.0] {
            let pr = Projection { beta, eta: 0.5 };
            assert!(pr.apply(0.0).abs() < 1e-12 && (pr.apply(1.0) - 1.0).abs() < 1e-12);
            assert!((pr.apply(0.5) - 0.5).abs() < 1e-12);
            for x in [0.1, 0.4, 0.5, 0.62, 0.9] {
                let h = 1e-6;
                let fd = (pr.apply(x + h) - pr.apply(x - h)) / (2.0 * h);
                assert!((fd - pr.deriv(x)).abs() < 1e-5 * (1.0 + fd.abs()), "beta {beta} x {x}");
            }
        }
        // sharp at high beta
        let pr = Projection { beta: 32.0, eta: 0.5 };
        assert!(pr.apply(0.4) < 0.05 && pr.apply(0.6) > 0.95);
    }

    #[test]
    fn ke_is_symmetric_and_rigid_body_free() {
        let k = hex_ke(0.3);
        for i in 0..24 {
            for j in 0..24 {
                assert!((k[i][j] - k[j][i]).abs() < 1e-12);
            }
            for c in 0..3 {
                // translation along c
                let s: f64 = (0..8).map(|a| k[i][3 * a + c]).sum();
                assert!(s.abs() < 1e-12, "row {i} translation {c}: {s}");
            }
            assert!(k[i][i] > 0.0);
        }
    }

    /// A 1x1x4 column, clamped at the bottom and pulled at the top, with nu = 0: the trilinear hex
    /// Supports and loads from editor primitives: a fixed subtracted bolt hole, a pin hole loaded
    /// down (case 1) and a region at the pin loaded sideways (case 2). Skipped without a GPU.
    #[test]
    fn primitives_set_up_supports_and_load_cases() {
        use crate::prims::{Kind, Op, Prim, Role, save, sidecar};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-bc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bar.wgsl");
        std::fs::write(
            &path,
            "//! bounds -20 -6 0 20 6 20\nfn map(p: vec3f) -> f32 {\n  \
             let q = abs(p - vec3f(0, 0, 10)) - vec3f(20, 6, 10);\n  \
             return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n}\n",
        )
        .unwrap();
        let hole = |name: &str, x: f32, role| Prim {
            op: Op::Subtract,
            rot: [90.0, 0.0, 0.0],
            role,
            ..Prim::new(Kind::Cylinder, name.into(), [x, 0.0, 10.0], 5.0)
        };
        let bolt = hole("bolt", -15.0, Role::Fixed);
        let mut pin = hole("pin", 15.0, Role::Load);
        pin.dims = [2.5, 14.0, 0.0];
        pin.force = 50.0;
        let side = Prim {
            op: Op::Region,
            role: Role::Load,
            load: [0.0, 2.0, 0.0], // normalized when used
            force: 30.0,
            case: 2,
            ..Prim::new(Kind::Sphere, "side".into(), [15.0, 0.0, 10.0], 8.0)
        };
        save(&sidecar(&path), &[bolt, pin, side], &[None; 3]).unwrap();
        let shape = crate::shape::load_shape(&path, &[]).unwrap();
        assert!(shape.bc && shape.script_load.is_none());
        let names: Vec<_> = shape.fields().unwrap().into_iter().map(|f| f.name).collect();
        assert_eq!(names, ["fixed", "load", "keep"]);
        let s = Settings {
            voxel: 1.0,
            volfrac: 0.3,
            penal: 3.0,
            rmin: 1.5,
            iters: 1,
            tol: 0.01,
            e0: 3500.0,
            nu: 0.3,
            force: 999.0, // only for a script's field_load: unused here
            beta_max: 0.0,
            solver: Solver::Jacobi,
            emin: 1e-6,
            cg_tol: 1e-4,
            adaptive_tol: false,
            ground: Ground::None,
            coarse_start: false,
            symmetry: true,
            overhang: false,
            up: crate::mesh::Up::PlusZ,
        };
        let p = setup(&shape, &s, &gpu).unwrap();
        assert_eq!(p.cases, [1, 2]);
        let mag = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let total = |l: &Vec<[f64; 3]>| l.iter().map(mag).sum::<f64>();
        assert!((total(&p.loads[0]) - 50.0).abs() < 1e-6, "{}", total(&p.loads[0]));
        assert!((total(&p.loads[1]) - 30.0).abs() < 1e-6, "{}", total(&p.loads[1]));
        assert!(p.loads[0].iter().all(|v| v[0] == 0.0 && v[1] == 0.0 && v[2] <= 0.0));
        assert!(p.loads[1].iter().all(|v| v[0] == 0.0 && v[2] == 0.0 && v[1] >= 0.0));
        // clamped nodes: on the bolt hole's wall (radius 2.5 + reach 1.5), nowhere else
        let [nx, ny, _] = p.nodes();
        let pos = |i: usize| {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            [x, y, z]
                .map(|c| c as f64)
                .iter()
                .zip(p.origin)
                .map(|(c, o)| o + c * p.h)
                .collect::<Vec<_>>()
        };
        let clamped: Vec<usize> = (0..p.fixed.len()).filter(|&i| p.fixed[i] != 0).collect();
        assert!(clamped.len() > 20, "{}", clamped.len());
        for &i in &clamped {
            let q = pos(i);
            let r = ((q[0] + 15.0).powi(2) + (q[2] - 10.0).powi(2)).sqrt();
            assert!(r < 4.0 + 1e-6, "clamped node {q:?} is {r} from the bolt axis");
        }
        // the shells around the holes are kept solid
        assert!(p.passive.iter().filter(|b| **b).count() > 50);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The cropped range covers the domain plus the margin, and stays centred on mirror axes.
    #[test]
    fn crop_box_fits_the_domain() {
        let n = [10, 8, 6];
        let mut inside = vec![false; 480];
        let at = |i: usize, j: usize, k: usize| (k * 8 + j) * 10 + i;
        inside[at(6, 5, 2)] = true;
        inside[at(7, 6, 3)] = true;
        assert_eq!(crop_box(&inside, n, [false; 3]), Some(([5, 4, 1], [9, 8, 5])));
        // about the middle (x = 5, y = 4): x 5..9 -> 1..9, y 4..8 -> 0..8
        assert_eq!(crop_box(&inside, n, [true, true, false]), Some(([1, 0, 1], [9, 8, 5])));
        assert_eq!(crop_box(&[false; 480], n, [false; 3]), None);
    }

    /// Declared bounds much larger than the part: the grid shrinks to the part, and the problem is
    /// the same as with tight bounds (only the origin moves).
    #[test]
    fn setup_crops_to_the_domain() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-crop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let body = "fn map(p: vec3f) -> f32 {\n  \
                    let q = abs(p - vec3f(0, 0, 5)) - vec3f(10, 3, 5);\n  \
                    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n}\n\
                    fn field_fixed(p: vec3f) -> f32 { return p.x + 9.0; }\n\
                    fn field_load(p: vec3f) -> vec3f { if (p.x > 9.0) { return vec3f(0, 0, -1); } \
                    return vec3f(0.0); }\n";
        let load = |name: &str, bounds: &str| {
            let path = dir.join(name);
            std::fs::write(&path, format!("//! bounds {bounds}\n{body}")).unwrap();
            crate::shape::load_shape(&path, &[]).unwrap()
        };
        let s = Settings {
            force: 10.0,
            ..beam_settings(Solver::Jacobi)
        };
        let tight = setup(&load("tight.wgsl", "-10 -3 0 10 3 10"), &s, &gpu).unwrap();
        let wide = setup(&load("wide.wgsl", "-40 -20 -10 30 25 50"), &s, &gpu).unwrap();
        assert_eq!(tight.n, [20, 6, 10]);
        assert_eq!(wide.n, [22, 8, 12], "the part plus one empty cell around it");
        assert_eq!(wide.origin, [-11.0, -4.0, -1.0]);
        let ne = |p: &Problem| p.domain.iter().filter(|d| **d).count();
        assert_eq!(ne(&tight), ne(&wide));
        let held = |p: &Problem| p.fixed.iter().filter(|f| **f != 0).count();
        assert_eq!(held(&tight), held(&wide));
        let sum = |p: &Problem| p.loads[0].iter().map(|v| v[2]).sum::<f64>();
        assert!((sum(&tight) + 10.0).abs() < 1e-9 && (sum(&tight) - sum(&wide)).abs() < 1e-9);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bar standing on slippery ground, carrying a 5 kg weight that also rocks: the bottom layer
    /// is held in Z, three pins stop it sliding and spinning, and the rocking adds four cases.
    #[test]
    fn ground_weight_and_rocking() {
        use crate::prims::{Kind, Op, Prim, Role, save, sidecar};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-ground-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stand.wgsl");
        std::fs::write(&path, "").unwrap();
        let block = Prim::new(Kind::Box, "block".into(), [0.0, 0.0, 10.0], 20.0);
        let top = Prim {
            op: Op::Region,
            role: Role::Load,
            force: 5.0,
            kg: true,
            rock: 0.2,
            ..Prim::new(Kind::Sphere, "book".into(), [0.0, 0.0, 20.0], 4.0)
        };
        save(&sidecar(&path), &[block, top], &[None; 3]).unwrap();
        let shape = crate::shape::load_shape(&path, &[]).unwrap();
        let s = Settings {
            voxel: 2.0,
            volfrac: 0.3,
            penal: 3.0,
            rmin: 1.5,
            iters: 1,
            tol: 0.01,
            e0: 3500.0,
            nu: 0.3,
            force: 999.0,
            beta_max: 0.0,
            solver: Solver::Jacobi,
            emin: 1e-6,
            cg_tol: 1e-4,
            adaptive_tol: false,
            ground: Ground::None,
            coarse_start: false,
            symmetry: true,
            overhang: false,
            up: crate::mesh::Up::PlusZ,
        };
        let err = setup(&shape, &s, &gpu).err().unwrap().to_string();
        assert!(err.contains("needs supports"), "{err}");
        let p = setup(
            &shape,
            &Settings {
                ground: Ground::Slide,
                ..s
            },
            &gpu,
        )
        .unwrap();
        // 5 kg upright, and tilted 20% both ways along X and Y
        assert_eq!(p.cases, [1, 1001, 1002, 1003, 1004]);
        let sum = |l: &Vec<[f64; 3]>| [0, 1, 2].map(|c| l.iter().map(|v| v[c]).sum::<f64>());
        let w = 5.0 * crate::prims::G;
        let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|c| (a[c] - b[c]).abs() < 1e-6 * w);
        assert!(close(sum(&p.loads[0]), [0.0, 0.0, -w]));
        assert!(close(sum(&p.loads[1]), [0.2 * w, 0.0, -w]), "{:?}", sum(&p.loads[1]));
        assert!(close(sum(&p.loads[4]), [0.0, -0.2 * w, -w]), "{:?}", sum(&p.loads[4]));
        // the bottom layer rests on rollers; a few of its nodes are pinned sideways too
        let bottom = p.node_pos(p.fixed.iter().position(|f| *f != 0).unwrap())[2];
        let held: Vec<usize> = (0..p.fixed.len()).filter(|&i| p.fixed[i] != 0).collect();
        assert!(held.iter().all(|&i| p.fixed[i] & 4 != 0 && p.node_pos(i)[2] == bottom));
        let pinned: usize = held.iter().map(|&i| (p.fixed[i] & 3).count_ones() as usize).sum();
        assert_eq!(pinned, 3, "X/Y pins");
        assert_eq!(rigid_rank(&p), 6);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A block standing on rollers (bottom held in Z only, plus the least X/Y pins against sliding
    /// and spinning), squeezed by a uniform pressure on top: the exact solution is uniform
    /// compression with free sideways expansion, which trilinear elements reproduce.
    pub(super) fn block_on_rollers() -> (Problem, f64, f64, f64) {
        let (h, force, nu) = (1.5, 80.0, 0.3);
        let n = [4, 4, 8];
        let ne: usize = n.iter().product();
        let nn: usize = n.iter().map(|v| v + 1).product();
        let mut p = Problem {
            n,
            h,
            origin: [0.0; 3],
            domain: vec![true; ne],
            passive: vec![false; ne],
            fixed: vec![0; nn],
            loads: vec![vec![[0.0; 3]; nn]],
            cases: vec![1],
        };
        for j in 0..=n[1] {
            for i in 0..=n[0] {
                let b = p.node(i, j, 0);
                p.fixed[b] = 4;
            }
        }
        let b = p.node(0, 0, 0);
        p.fixed[b] = 7;
        let b = p.node(n[0], 0, 0);
        p.fixed[b] = 4 | 2;
        // consistent nodal loads of a uniform pressure: a quarter of each top face's share per corner
        let per_face = force / (n[0] * n[1]) as f64;
        for j in 0..n[1] {
            for i in 0..n[0] {
                for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let t = p.node(i + di, j + dj, n[2]);
                    p.loads[0][t][2] -= per_face / 4.0;
                }
            }
        }
        (p, force, nu, 1000.0)
    }

    /// Checks a displacement field against `block_on_rollers`' exact solution.
    pub(super) fn check_block_on_rollers(p: &Problem, u: &[f64], force: f64, nu: f64, e0: f64, tol: f64) {
        let [w, d, l] = p.n.map(|v| v as f64 * p.h);
        let stress = force / (w * d);
        let top = p.node(p.n[0], p.n[1], p.n[2]);
        let want_z = -stress / e0 * l;
        let want_x = nu * stress / e0 * w;
        let want_y = nu * stress / e0 * d;
        let got = &u[3 * top..3 * top + 3];
        for (c, want) in [want_x, want_y, want_z].into_iter().enumerate() {
            assert!(
                (got[c] - want).abs() < tol * want.abs(),
                "axis {c}: {} vs {want}",
                got[c]
            );
        }
    }

    #[test]
    fn block_on_rollers_cpu() {
        let (p, force, nu, e0) = block_on_rollers();
        let ke = hex_ke(nu);
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: vec![e0 * p.h; p.domain.len()],
            free: p.free_mask(),
        };
        let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
        let inv = fem.inv_diag();
        let mut u = vec![0.0; f.len()];
        let mut jacobi = |r: &[f64], z: &mut [f64]| {
            for i in 0..r.len() {
                z[i] = r[i] * inv[i];
            }
        };
        fem.solve(&f, &mut u, 1e-12, 5000, &mut jacobi);
        check_block_on_rollers(&p, &u, force, nu, e0, 1e-6);
        // multigrid: its coarse levels see the rollers as clamped, still a good preconditioner
        let mut m = mg::Multigrid::new(&fem, &mg::MgSettings::default());
        let mut u = vec![0.0; f.len()];
        let its = fem.solve(&f, &mut u, 1e-12, 500, &mut |r, z| m.precondition(r, z));
        check_block_on_rollers(&p, &u, force, nu, e0, 1e-6);
        assert!(its < 100, "{its} MGCG iterations");
    }

    /// reproduces uniaxial tension exactly, so the top moves F L / (E A).
    #[test]
    fn column_in_tension() {
        let (h, e0, force) = (2.0, 1000.0, 50.0);
        let n = [1, 1, 4];
        let mut p = Problem {
            n,
            h,
            origin: [0.0; 3],
            domain: vec![true; 4],
            passive: vec![false; 4],
            fixed: vec![0; 2 * 2 * 5],
            loads: vec![vec![[0.0; 3]; 2 * 2 * 5]],
            cases: vec![1],
        };
        for j in 0..2 {
            for i in 0..2 {
                let b = p.node(i, j, 0);
                p.fixed[b] = 7;
                let t = p.node(i, j, 4);
                p.loads[0][t] = [0.0, 0.0, force / 4.0];
            }
        }
        let ke = hex_ke(0.0);
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: vec![e0 * h; 4],
            free: p.free_mask(),
        };
        let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
        let mut u = vec![0.0; f.len()];
        let inv = fem.inv_diag();
        let mut jacobi = |r: &[f64], z: &mut [f64]| {
            for i in 0..r.len() {
                z[i] = r[i] * inv[i];
            }
        };
        fem.solve(&f, &mut u, 1e-12, 1000, &mut jacobi);
        let want = force * (4.0 * h) / (e0 * h * h);
        let top = p.node(1, 1, 4);
        assert!(
            (u[3 * top + 2] - want).abs() < 1e-9 * want,
            "{} vs {want}",
            u[3 * top + 2]
        );
        assert!(u[3 * top].abs() < 1e-12);
    }
}
