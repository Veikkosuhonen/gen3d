//! Topology optimization: minimum compliance (stiffest part for a given amount of material) with the
//! SIMP method on a voxel grid, in the style of top88/top3d.
//!
//! The problem comes from a shape file: `map` is the design domain, `field_fixed` (< 0) clamps nodes,
//! `field_load` (vec3f) gives the load direction where it is non-zero, and the optional `field_keep`
//! / `field_void` (< 0) force elements solid / empty. Each voxel is a trilinear 8-node hex element.
//!
//! Each iteration solves K(ρ) u = f with a matrix-free, Jacobi-preconditioned conjugate gradient
//! (multithreaded, warm-started), computes the compliance sensitivities, applies a density filter
//! and updates the densities with the optimality-criteria method under a volume constraint. The
//! current design is written to a .vti after every iteration, so `gen3d view` can watch it live.
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
    /// per node: force (N)
    pub load: Vec<[f64; 3]>,
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
    let fields = shape.fields()?;
    let field = |name: &str, comps: usize| fields.iter().find(|f| f.name == name && f.comps == comps);
    let Some(fixed_f) = field("fixed", 1) else {
        bail!("optimize needs `fn field_fixed(p: vec3f) -> f32`, < 0 where the part is clamped");
    };
    let Some(load_f) = field("load", 3) else {
        bail!("optimize needs `fn field_load(p: vec3f) -> vec3f`, the load direction where it is applied");
    };
    let h = s.voxel;
    if h <= 0.0 {
        bail!("voxel size must be positive");
    }
    let (lo, hi) = shape.bounds;
    let n = [0, 1, 2].map(|k| ((hi[k] - lo[k]) as f64 / h).ceil().max(1.0) as usize);
    let origin = lo.map(|v| v as f64);
    let ne: usize = n.iter().product();
    if ne > 3_000_000 {
        bail!("{ne} elements is too many for this solver; use a larger --voxel");
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
    let load = x_fastest(&sample_field(shape, &nodes, &load_f.func, 3, gpu)?, pts, 3);

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
        load: vec![],
    };
    let active = p.active_nodes();
    p.fixed = (0..active.len()).map(|i| active[i] && fixed[i] < 0.0).collect();
    let raw: Vec<[f64; 3]> = (0..active.len())
        .map(|i| {
            let v = [0, 1, 2].map(|c| load[3 * i + c] as f64);
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
    let nfixed = p.fixed.iter().filter(|b| **b).count();
    let nload = raw.iter().filter(|v| v.iter().any(|x| *x != 0.0)).count();
    if nfixed == 0 {
        bail!(
            "no clamped nodes: field_fixed is never < 0 at a node of the design domain (try a smaller --voxel or a thicker region)"
        );
    }
    if nload == 0 {
        bail!(
            "no loaded nodes: field_load is zero at every free node of the design domain (try a smaller --voxel or a thicker region)"
        );
    }
    // spread the total force so the load magnitudes sum to `force`
    p.load = raw.iter().map(|v| v.map(|x| x * s.force / total)).collect();
    let ndomain = p.domain.iter().filter(|b| **b).count();
    let npassive = p.passive.iter().filter(|b| **b).count();
    println!(
        "grid {}x{}x{} @ {} mm: {ndomain} domain elements ({} design, {npassive} kept solid), {nfixed} clamped nodes, {nload} loaded nodes",
        n[0],
        n[1],
        n[2],
        h as f32,
        ndomain - npassive
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

    /// Jacobi-preconditioned conjugate gradient, starting from `u`. Returns the iteration count.
    fn solve(&self, f: &[f64], u: &mut [f64], tol: f64, maxit: usize) -> usize {
        let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let inv = self.inv_diag();
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
        let mut z: Vec<f64> = r.iter().zip(&inv).map(|(a, b)| a * b).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for it in 0..maxit {
            if dot(&r, &r).sqrt() <= tol * fnorm {
                return it;
            }
            self.matvec(&p, &mut q);
            let alpha = rz / dot(&p, &q);
            u.par_iter_mut().zip(&p).for_each(|(x, y)| *x += alpha * y);
            r.par_iter_mut().zip(&q).for_each(|(x, y)| *x -= alpha * y);
            z.par_iter_mut().zip(&r).zip(&inv).for_each(|((z, r), d)| *z = r * d);
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

pub struct Outcome {
    pub iterations: usize,
    pub compliance: f64,
    pub max_disp: f64,
}

/// Run the optimization, writing the design to `out` (cell `density`, cell `energy` in MPa, point
/// `displacement` in mm) after every iteration.
pub fn optimize(p: &Problem, s: &Settings, out: &Path) -> Result<Outcome> {
    let ke = hex_ke(s.nu);
    let ne = p.domain.len();
    let design: Vec<usize> = (0..ne).filter(|&e| p.domain[e] && !p.passive[e]).collect();
    if design.is_empty() {
        bail!("no design elements: the domain is empty or entirely kept solid");
    }
    let active = p.active_nodes();
    let free: Vec<bool> = (0..active.len()).map(|i| active[i] && !p.fixed[i]).collect();
    let f: Vec<f64> = p.load.iter().flatten().copied().collect();
    let emin = s.e0 * 1e-6;
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
    let mut u = vec![0.0; f.len()];
    let t_start = Instant::now();
    let mut stalled = 0;
    let mut stage_its = 0;
    let mut result = Outcome {
        iterations: 0,
        compliance: 0.0,
        max_disp: 0.0,
    };
    println!(
        "SIMP: volume fraction {}, penalty {}, filter radius {} voxels, E = {} MPa, nu = {}, load {} N, {}",
        s.volfrac,
        s.penal,
        s.rmin,
        s.e0,
        s.nu,
        s.force,
        if s.beta_max > 0.0 {
            format!("Heaviside projection up to beta {}", s.beta_max)
        } else {
            "no projection".into()
        }
    );
    for it in 1..=s.iters {
        let t0 = Instant::now();
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
        let cg = fem.solve(&f, &mut u, 1e-4, 20_000);
        let ce = fem.element_energy(&u);
        let compliance: f64 = (0..ne).map(|e| fem.ee[e] * ce[e]).sum();
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

        // optimality criteria: bisect the Lagrange multiplier to meet the volume
        let target = s.volfrac * design.len() as f64;
        let (mut l1, mut l2, mv) = (0.0f64, 1e9f64, 0.2);
        let mut xnew = x.clone();
        while (l2 - l1) / (l1 + l2).max(1e-300) > 1e-4 && l2 > 1e-40 {
            let lmid = 0.5 * (l1 + l2);
            for &e in &design {
                let be = (-dc[e]).max(0.0) / (dv[e] * lmid).max(1e-300);
                xnew[e] = (x[e] * be.sqrt()).clamp((x[e] - mv).max(0.0), (x[e] + mv).min(1.0));
            }
            let trial = proj.densities(p, &filter.densities(p, &xnew));
            let vol: f64 = design.iter().map(|&e| trial[e]).sum();
            if vol > target {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
        let change = design.iter().map(|&e| (xnew[e] - x[e]).abs()).fold(0.0, f64::max);
        xtilde = filter.densities(p, &xnew);
        xphys = proj.densities(p, &xtilde);
        x = xnew;

        let max_disp = u
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0, f64::max);
        let grey = design
            .iter()
            .filter(|&&e| used_xphys[e] > 0.1 && used_xphys[e] < 0.9)
            .count() as f64
            / design.len() as f64;
        let beta = if proj.beta > 0.0 {
            format!(", beta {}", proj.beta)
        } else {
            String::new()
        };
        println!(
            "it {it:3}: compliance {compliance:.4e} N·mm, max displacement {max_disp:.4} mm, change {change:.3}, grey {:.0}%{beta}, {cg} CG its, {:.1} s",
            grey * 100.0,
            t0.elapsed().as_secs_f64()
        );
        // the densities this solve used, with their energy density and displacements
        let density: Vec<f32> = used_xphys.iter().map(|&v| v as f32).collect();
        let energy: Vec<f32> = (0..ne)
            .map(|e| (0.5 * fem.ee[e] * ce[e] / p.h.powi(3)) as f32)
            .collect();
        let disp: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let arrays = [
            crate::vti::GridArray {
                name: "density",
                cell: true,
                comps: 1,
                data: &density,
            },
            crate::vti::GridArray {
                name: "energy",
                cell: true,
                comps: 1,
                data: &energy,
            },
            crate::vti::GridArray {
                name: "displacement",
                cell: false,
                comps: 3,
                data: &disp,
            },
        ];
        let bytes = crate::vti::write_grid(out, p.origin, p.h, p.nodes(), &arrays)?;
        if it == 1 {
            println!(
                "wrote {} ({:.1} MB, updated every iteration)",
                out.display(),
                bytes as f64 / 1e6
            );
        }
        // the max density change often hovers at the move limit while the design no longer
        // improves, so also stop once compliance has stagnated
        let improvement = (result.compliance - compliance) / compliance;
        stalled = if stage_its > 0 && improvement.abs() < 1e-3 {
            stalled + 1
        } else {
            0
        };
        stage_its += 1;
        result = Outcome {
            iterations: it,
            compliance,
            max_disp,
        };
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
            load: vec![[0.0; 3]; 2 * 2 * 5],
        };
        for j in 0..2 {
            for i in 0..2 {
                let b = p.node(i, j, 0);
                p.fixed[b] = true;
                let t = p.node(i, j, 4);
                p.load[t] = [0.0, 0.0, force / 4.0];
            }
        }
        let ke = hex_ke(0.0);
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: vec![e0 * h; 4],
            free: p.fixed.iter().map(|f| !f).collect(),
        };
        let f: Vec<f64> = p.load.iter().flatten().copied().collect();
        let mut u = vec![0.0; f.len()];
        fem.solve(&f, &mut u, 1e-12, 1000);
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
