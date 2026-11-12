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
//! Each iteration solves K(ρ) u = f with a conjugate gradient (multithreaded, warm-started,
//! matrix-free on the fine grid) preconditioned by a geometric multigrid V-cycle (`mg`), or by
//! Jacobi. It then computes the compliance sensitivities, applies a density filter and updates the
//! densities with the optimality-criteria method under a volume constraint. The current design is
//! written to a .vti as it runs, so `gen3d view` can watch it live.
//!
//! Units: mm, N, MPa. With the default PLA-like modulus the displacements are real millimetres.

// small dense matrix code reads best with explicit indices
#![allow(clippy::needless_range_loop)]

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::gpu::Gpu;
use crate::mesher::{Grid, sample_field};
use crate::shape::Shape;

pub mod mg;

/// Linear solver for K u = f.
#[derive(Clone, Copy, Debug)]
pub enum Solver {
    /// Jacobi-preconditioned CG
    Jacobi,
    /// multigrid-preconditioned CG
    Multigrid(mg::MgSettings),
}

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
}

impl Settings {
    /// The usual E_min for a solver: multigrid copes with the 1e9 stiffness contrast, Jacobi doesn't.
    pub fn default_emin(solver: &Solver) -> f64 {
        match solver {
            Solver::Jacobi => 1e-6,
            Solver::Multigrid(_) => 1e-9,
        }
    }
}

/// A discretized problem. Elements and nodes are indexed x-fastest.
pub struct Problem {
    /// elements per axis
    pub n: [usize; 3],
    pub h: f64,
    /// position of the first node (the bounds' minimum)
    pub origin: [f64; 3],
    pub domain: Vec<bool>,
    /// forced solid (design domain ∩ keep)
    pub passive: Vec<bool>,
    /// per node: clamped (only nodes of domain elements)
    pub fixed: Vec<bool>,
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
}

/// Element size giving at most about `target` elements in the shape's bounds (never finer than the
/// shape's own voxel), rounded up to 2 significant digits.
pub fn auto_voxel(shape: &Shape, target: f64) -> f32 {
    let vol: f64 = shape.size().iter().map(|s| (*s as f64).max(1e-3)).product();
    let v = (shape.voxel as f64).max((vol / target).cbrt());
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
    let Some(fixed_f) = field("fixed", 1) else {
        bail!(
            "optimize needs supports: `fn field_fixed(p: vec3f) -> f32` (< 0 where the part is clamped), \
             or a primitive with the fixed role"
        );
    };
    let specs = shape.load_specs();
    if specs.is_empty() {
        bail!(
            "optimize needs loads: `fn field_load(p: vec3f) -> vec3f` (the load direction where it is \
             applied), or a primitive with the load role"
        );
    }
    let (lo, hi) = shape.bounds;
    let n = [0, 1, 2].map(|k| ((hi[k] - lo[k]) as f64 / h).ceil().max(1.0) as usize);
    let origin = lo.map(|v| v as f64);
    let ne: usize = n.iter().product();
    // the CPU multigrid needs ~1.2 kB per element (mostly the level-1 element matrices)
    let max = match s.solver {
        Solver::Jacobi => 3_000_000,
        Solver::Multigrid(_) => 8_000_000,
    };
    if ne > max {
        bail!("{ne} elements is too many for this solver (at most {max}); use a larger --voxel");
    }
    let cells = Grid {
        vol: vec![],
        dims: n,
        origin: origin.map(|o| o + 0.5 * h),
        step: h,
    };
    let pts = n.map(|v| v + 1);
    let nodes = Grid {
        vol: vec![],
        dims: pts,
        origin,
        step: h,
    };
    let at_cells =
        |func: &str| -> Result<Vec<f32>> { Ok(x_fastest(&sample_field(shape, &cells, func, 1, gpu)?, n, 1)) };
    let sdf = at_cells("map")?;
    let keep = field("keep", 1).map(|f| at_cells(&f.func)).transpose()?;
    let void = field("void", 1).map(|f| at_cells(&f.func)).transpose()?;
    let fixed = x_fastest(&sample_field(shape, &nodes, &fixed_f.func, 1, gpu)?, pts, 1);

    let domain: Vec<bool> = (0..ne)
        .map(|e| sdf[e] < 0.0 && void.as_ref().is_none_or(|v| v[e] >= 0.0))
        .collect();
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
    p.fixed = (0..active.len()).map(|i| active[i] && fixed[i] < 0.0).collect();
    let nfixed = p.fixed.iter().filter(|b| **b).count();
    if nfixed == 0 {
        bail!(
            "no clamped nodes: the fixed region (field_fixed < 0, or a fixed primitive) reaches no node of \
             the design domain (try a smaller --voxel, a thicker region, or move it onto the part)"
        );
    }
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
                if active[i] && !p.fixed[i] && v.iter().all(|x| x.is_finite()) {
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
        println!("load `{}`: {force} N, case {}", spec.name, spec.case);
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
        "grid {}x{}x{} @ {} mm: {ndomain} domain elements ({} design, {npassive} kept solid), {nfixed} clamped nodes, {nload} loaded nodes, {} load case{}",
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
    /// per node: has stiffness and is not clamped
    free: Vec<bool>,
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
            if !self.free[n] {
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
            .map(|(i, &v)| if self.free[i / 3] && v > 0.0 { 1.0 / v } else { 0.0 })
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
            if !self.free[i / 3] {
                *v = 0.0;
            }
        }
        let fm: Vec<f64> = f
            .iter()
            .enumerate()
            .map(|(i, &v)| if self.free[i / 3] { v } else { 0.0 })
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
}

impl Filter {
    fn new(p: &Problem, rmin: f64) -> Filter {
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
        let mut f = Filter { offsets, hs: vec![] };
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
                for ([di, dj, dk], w) in &self.offsets {
                    let (a, b, c) = (i + di, j + dj, k + dk);
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
}

const CG_MAXIT: usize = 20_000;

/// z = M⁻¹ r
type Precond<'a> = dyn FnMut(&[f64], &mut [f64]) + 'a;

/// Run the optimization, writing the design to `out` (cell `density`, cell `energy` in MPa, point
/// `displacement` in mm). The file is rewritten as often as keeps writing under ~10% of the run
/// time (every iteration on small grids), and always after the last iteration.
pub fn optimize(p: &Problem, s: &Settings, out: &Path) -> Result<Outcome> {
    let ke = hex_ke(s.nu);
    let ne = p.domain.len();
    let design: Vec<usize> = (0..ne).filter(|&e| p.domain[e] && !p.passive[e]).collect();
    if design.is_empty() {
        bail!("no design elements: the domain is empty or entirely kept solid");
    }
    let active = p.active_nodes();
    let free: Vec<bool> = (0..active.len()).map(|i| active[i] && !p.fixed[i]).collect();
    let fs: Vec<Vec<f64>> = p.loads.iter().map(|l| l.iter().flatten().copied().collect()).collect();
    let emin = s.e0 * s.emin;
    let filter = Filter::new(p, s.rmin);
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
    let mut xtilde = filter.densities(p, &x);
    let mut xphys = proj.densities(p, &xtilde);
    let mut us: Vec<Vec<f64>> = fs.iter().map(|f| vec![0.0; f.len()]).collect();
    let t_start = Instant::now();
    let mut stalled = 0;
    let mut stage_its = 0;
    let mut result = Outcome {
        iterations: 0,
        compliance: 0.0,
        max_disp: 0.0,
        stats: vec![],
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
    // the last iteration's output, until it is written
    let mut pending: Option<Snapshot> = None;
    let mut last_write: Option<(Instant, f64)> = None;
    // OC's Lagrange multiplier, for the next iteration's bracket
    let mut lambda = 1.0;
    let mut last_change = 1.0f64;
    for it in 1..=s.iters {
        let t0 = Instant::now();
        let mut st = IterStats::default();
        let fem = Fem {
            p,
            ke: &ke,
            ee: (0..ne)
                .map(|e| {
                    if p.domain[e] {
                        (emin + xphys[e].powf(s.penal) * (s.e0 - emin)) * p.h
                    } else {
                        0.0
                    }
                })
                .collect(),
            free: free.clone(),
        };
        // one preconditioner, shared by the load cases
        let mut precond: Box<Precond> = match &s.solver {
            Solver::Jacobi => {
                let inv = fem.inv_diag();
                Box::new(move |r: &[f64], z: &mut [f64]| {
                    z.par_iter_mut().zip(r).zip(&inv).for_each(|((z, r), d)| *z = r * d)
                })
            }
            Solver::Multigrid(ms) => {
                let mut m = mg::Multigrid::new(&fem, ms);
                if it == 1 {
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
        };
        st.setup = t0.elapsed().as_secs_f64();
        // each load case separately; the objective is their summed compliance
        let t1 = Instant::now();
        let mut ce = vec![0.0; ne];
        let final_stage = proj.beta > 0.0 && proj.beta >= s.beta_max;
        let tol = if !s.adaptive_tol || final_stage {
            s.cg_tol
        } else {
            (0.05 * last_change).clamp(s.cg_tol, 1e-2)
        };
        st.cg_tol = tol;
        for (f, u) in fs.iter().zip(us.iter_mut()) {
            let n = fem.solve(f, u, tol, CG_MAXIT, &mut *precond);
            if n >= CG_MAXIT {
                println!("warning: CG did not converge in {CG_MAXIT} iterations");
            }
            st.cg.push(n);
            for (a, b) in ce.iter_mut().zip(fem.element_energy(u)) {
                *a += b;
            }
        }
        drop(precond);
        st.solve = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let compliance: f64 = (0..ne).map(|e| fem.ee[e] * ce[e]).sum();
        st.compliance = compliance;
        let dc: Vec<f64> = (0..ne)
            .map(|e| {
                if p.domain[e] && !p.passive[e] {
                    -s.penal * xphys[e].powf(s.penal - 1.0) * (s.e0 - emin) * p.h * ce[e]
                } else {
                    0.0
                }
            })
            .collect();
        // chain rule: projection, then filter (the volume sensitivity changes with the projection)
        let dproj: Vec<f64> = (0..ne).map(|e| proj.deriv(xtilde[e])).collect();
        let dc: Vec<f64> = (0..ne).map(|e| dc[e] * dproj[e]).collect();
        let dc = filter.sensitivities(p, &dc);
        let dv = filter.sensitivities(p, &dproj);
        let used_xphys = xphys.clone();

        // optimality criteria: bisect the Lagrange multiplier to meet the volume. With dc scaled by
        // the compliance, λ is O(1) and moves little between iterations, so the search starts from
        // a bracket around the previous λ.
        let target = s.volfrac * design.len() as f64;
        let mv = 0.2;
        let mut xnew = x.clone();
        let cscale = 1.0 / compliance.abs().max(1e-300);
        let mut volume_at = |lam: f64, xnew: &mut Vec<f64>| {
            st.oc_steps += 1;
            for &e in &design {
                let be = (-dc[e] * cscale).max(0.0) / (dv[e] * lam).max(1e-300);
                xnew[e] = (x[e] * be.sqrt()).clamp((x[e] - mv).max(0.0), (x[e] + mv).min(1.0));
            }
            let trial = proj.densities(p, &filter.densities(p, xnew));
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
        xphys = proj.densities(p, &xtilde);
        x = xnew;

        let max_disp = us
            .iter()
            .flat_map(|u| u.as_chunks::<3>().0.iter())
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0, f64::max);
        let grey = design
            .iter()
            .filter(|&&e| used_xphys[e] > 0.1 && used_xphys[e] < 0.9)
            .count() as f64
            / design.len() as f64;
        st.update = t2.elapsed().as_secs_f64();

        // the densities this solve used, with their energy density and displacements
        let t3 = Instant::now();
        pending = Some(Snapshot {
            density: used_xphys.iter().map(|&v| v as f32).collect(),
            energy: (0..ne)
                .map(|e| (0.5 * fem.ee[e] * ce[e] / p.h.powi(3)) as f32)
                .collect(),
            disps: us.iter().map(|u| u.iter().map(|&v| v as f32).collect()).collect(),
            iteration: it,
        });
        // keep writing under ~10% of the time
        let due = last_write.is_none_or(|(t, dur)| t.elapsed().as_secs_f64() >= 10.0 * dur);
        if due {
            let bytes = pending.take().unwrap().write(p, out)?;
            last_write = Some((Instant::now(), t3.elapsed().as_secs_f64()));
            if it == 1 {
                println!("wrote {} ({:.1} MB, updated as it runs)", out.display(), bytes as f64 / 1e6);
            }
        }
        st.io = t3.elapsed().as_secs_f64();

        let beta = if proj.beta > 0.0 {
            format!(", beta {}", proj.beta)
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
        // the max density change often hovers at the move limit while the design no longer
        // improves, so also stop once compliance has stagnated
        let improvement = (result.compliance - compliance) / compliance;
        stalled = if stage_its > 0 && improvement.abs() < 1e-3 {
            stalled + 1
        } else {
            0
        };
        stage_its += 1;
        result.iterations = it;
        result.compliance = compliance;
        result.max_disp = max_disp;
        result.stats.push(st);
        let converged = change < s.tol || stalled >= 3;
        // continuation: sharpen the projection each time the design settles (or after 25 iterations)
        if proj.beta > 0.0 && proj.beta < s.beta_max && (converged || stage_its >= 25) {
            proj.beta = (proj.beta * 2.0).min(s.beta_max);
            println!("beta -> {}", proj.beta);
            xphys = proj.densities(p, &xtilde);
            stalled = 0;
            stage_its = 0;
            continue;
        }
        if change < s.tol {
            println!("converged (density change < {})", s.tol);
            break;
        }
        if stalled >= 3 {
            println!("converged (compliance changed < 0.1% for 3 iterations)");
            break;
        }
    }
    if let Some(snap) = pending {
        snap.write(p, out)?;
    }
    println!(
        "done in {:.1} s: {} iterations, compliance {:.4e} N·mm, max displacement {:.4} mm",
        t_start.elapsed().as_secs_f64(),
        result.iterations,
        result.compliance,
        result.max_disp
    );
    println!("wrote {} (iteration {})", out.display(), result.iterations);
    Ok(result)
}

/// One iteration's output fields.
struct Snapshot {
    density: Vec<f32>,
    energy: Vec<f32>,
    disps: Vec<Vec<f32>>,
    #[allow(dead_code)]
    iteration: usize,
}

impl Snapshot {
    fn write(&self, p: &Problem, out: &Path) -> Result<u64> {
        let names: Vec<String> = p
            .cases
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if k == 0 {
                    "displacement".to_string()
                } else {
                    format!("displacement_{c}")
                }
            })
            .collect();
        let mut arrays = vec![
            crate::vti::GridArray {
                name: "density",
                cell: true,
                comps: 1,
                data: &self.density,
            },
            crate::vti::GridArray {
                name: "energy",
                cell: true,
                comps: 1,
                data: &self.energy,
            },
        ];
        for (name, data) in names.iter().zip(&self.disps) {
            arrays.push(crate::vti::GridArray {
                name,
                cell: false,
                comps: 3,
                data,
            });
        }
        crate::vti::write_grid(out, p.origin, p.h, p.nodes(), &arrays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        save(&sidecar(&path), &[bolt, pin, side]).unwrap();
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
        let clamped: Vec<usize> = (0..p.fixed.len()).filter(|&i| p.fixed[i]).collect();
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
            fixed: vec![false; 2 * 2 * 5],
            loads: vec![vec![[0.0; 3]; 2 * 2 * 5]],
            cases: vec![1],
        };
        for j in 0..2 {
            for i in 0..2 {
                let b = p.node(i, j, 0);
                p.fixed[b] = true;
                let t = p.node(i, j, 4);
                p.loads[0][t] = [0.0, 0.0, force / 4.0];
            }
        }
        let ke = hex_ke(0.0);
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: vec![e0 * h; 4],
            free: p.fixed.iter().map(|f| !f).collect(),
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
