//! Geometric multigrid preconditioner for the stiffness system (MGCG: Amir, Aage & Lazarov 2014).
//!
//! This is the CPU reference, written in the shape a GPU version would take: a structured grid,
//! point smoothers (damped Jacobi or Chebyshev, no Gauss–Seidel), and Galerkin coarse levels.
//!
//! - Each level coarsens the voxel grid 2× per axis. Odd sizes round up, so a coarse grid can reach
//!   half an element past the finer one; no padding is needed.
//! - Interpolation is trilinear. Each fine element then lies inside one coarse element, and the
//!   Galerkin operator Pᵀ K P is again a sum of 24×24 element matrices: K_E = Σ_c P_cᵀ K_c P_c over
//!   the (up to 8) children c of E.
//! - The fine level stays matrix-free. Coarse levels store their element matrices.
//! - Clamped DOFs are masked out of the fine element matrices first (K̃_e = D K_e D), so the coarse
//!   operators are Galerkin projections of the constrained system. Coarse DOFs whose diagonal is
//!   zero (they only touch clamped or empty space) are inactive.
//! - The coarsest level is solved exactly with a banded Cholesky factorization.
//!
//! The V-cycle uses the same smoother before and after the coarse correction, so it is a symmetric
//! preconditioner and plain CG can use it.

use rayon::prelude::*;

use super::Fem;

/// How the levels are smoothed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smoother {
    /// `sweeps` damped Jacobi sweeps with ω = `omega` / λmax(D⁻¹K)
    Jacobi { sweeps: usize, omega: f64 },
    /// Chebyshev polynomial of this degree in D⁻¹K, on [0.1, 1.1]·λmax
    Chebyshev { degree: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct MgSettings {
    pub smoother: Smoother,
    /// keep coarsening while the coarsest level's Cholesky would cost more than this (flops)
    pub max_direct_flops: f64,
}

impl Default for MgSettings {
    fn default() -> Self {
        MgSettings {
            smoother: Smoother::Chebyshev { degree: 2 },
            max_direct_flops: 2e7,
        }
    }
}

/// Nodes of element `e` on a grid of `n` elements (x-fastest), ordered by bits (x, y, z).
fn element_nodes(n: [usize; 3], e: usize) -> [usize; 8] {
    let [ex, ey, _] = n;
    let (nx, ny) = (ex + 1, ey + 1);
    let (i, j, k) = (e % ex, (e / ex) % ey, e / (ex * ey));
    std::array::from_fn(|b| ((k + (b >> 2 & 1)) * ny + j + (b >> 1 & 1)) * nx + i + (b & 1))
}

fn coarse_dims(n: [usize; 3]) -> [usize; 3] {
    n.map(|v| v.div_ceil(2))
}

/// Interpolation weights from a coarse element's nodes to the nodes of its child `c` (bits x, y, z
/// of the child's position): `w[a][b]` is fine local node a's weight on coarse local node b.
fn child_weights(c: usize) -> [[f64; 8]; 8] {
    let mut w = [[0.0; 8]; 8];
    for a in 0..8 {
        for b in 0..8 {
            w[a][b] = (0..3)
                .map(|ax| {
                    let p = ((c >> ax & 1) + (a >> ax & 1)) as f64 * 0.5;
                    if b >> ax & 1 == 1 { p } else { 1.0 - p }
                })
                .product();
        }
    }
    w
}

/// out += scale · Pᵀ k P, for a 24×24 element matrix `k` (row-major) and child weights `w`.
fn add_ptkp(k: &[f64], w: &[[f64; 8]; 8], scale: f64, out: &mut [f64]) {
    // t = k P
    let mut t = [0.0; 576];
    for r in 0..24 {
        let kr = &k[r * 24..r * 24 + 24];
        for b in 0..8 {
            for j in 0..3 {
                let mut s = 0.0;
                for a in 0..8 {
                    s += kr[3 * a + j] * w[a][b];
                }
                t[r * 24 + 3 * b + j] = s;
            }
        }
    }
    // out += Pᵀ t
    for b in 0..8 {
        for i in 0..3 {
            let o = &mut out[(3 * b + i) * 24..(3 * b + i) * 24 + 24];
            for a in 0..8 {
                let wab = w[a][b];
                if wab == 0.0 {
                    continue;
                }
                let tr = &t[(3 * a + i) * 24..(3 * a + i) * 24 + 24];
                for col in 0..24 {
                    o[col] += scale * wab * tr[col];
                }
            }
        }
    }
}

/// A coarse level: the Galerkin operator as element matrices.
struct Coarse {
    /// elements per axis
    n: [usize; 3],
    /// 24×24 per element, row-major
    mats: Vec<f64>,
    /// element has any stiffness
    nz: Vec<bool>,
    /// per DOF: 1 / diagonal, 0 where inactive
    inv_diag: Vec<f64>,
}

impl Coarse {
    fn nodes(&self) -> usize {
        self.n.iter().map(|v| v + 1).product()
    }

    /// Galerkin coarsening of the fine level.
    fn from_fine(fem: &Fem) -> Coarse {
        let n = fem.p.n;
        let ke: Vec<f64> = fem.ke.iter().flatten().copied().collect();
        let weights: Vec<[[f64; 8]; 8]> = (0..8).map(child_weights).collect();
        // unmasked children share Pᵀ ke P per child position
        let ptkp: Vec<Vec<f64>> = weights
            .iter()
            .map(|w| {
                let mut m = vec![0.0; 576];
                add_ptkp(&ke, w, 1.0, &mut m);
                m
            })
            .collect();
        Self::build(n, |e, out| {
            let ee = fem.ee[e.0];
            if ee == 0.0 {
                return false;
            }
            let nodes = fem.element_nodes(e.0);
            let free = |m: usize| fem.free[m] == 7;
            // per-DOF mask: a roller node keeps its free axes
            let dof = |r: usize| fem.free[nodes[r / 3]] >> (r % 3) & 1 == 1;
            if nodes.iter().all(|&m| free(m)) {
                for (o, m) in out.iter_mut().zip(&ptkp[e.1]) {
                    *o += ee * m;
                }
            } else {
                let mut km = [0.0; 576];
                for r in 0..24 {
                    if !dof(r) {
                        continue;
                    }
                    for s in 0..24 {
                        if dof(s) {
                            km[r * 24 + s] = ee * ke[r * 24 + s];
                        }
                    }
                }
                add_ptkp(&km, &weights[e.1], 1.0, out);
            }
            true
        })
    }

    /// Galerkin coarsening of a coarse level.
    fn from_coarse(l: &Coarse) -> Coarse {
        let weights: Vec<[[f64; 8]; 8]> = (0..8).map(child_weights).collect();
        Self::build(l.n, |e, out| {
            if !l.nz[e.0] {
                return false;
            }
            add_ptkp(&l.mats[576 * e.0..576 * e.0 + 576], &weights[e.1], 1.0, out);
            true
        })
    }

    /// Coarse level of a grid of `n` elements; `child((element, position), out)` adds Pᵀ K P of a
    /// child element to `out` and says whether it had any stiffness.
    fn build(n: [usize; 3], child: impl Fn((usize, usize), &mut [f64]) -> bool + Sync) -> Coarse {
        let nc = coarse_dims(n);
        let [cx, cy, _] = nc;
        let nec: usize = nc.iter().product();
        let mut mats = vec![0.0; 576 * nec];
        let nz: Vec<bool> = mats
            .par_chunks_mut(576)
            .enumerate()
            .map(|(ce, out)| {
                let (ci, cj, ck) = (ce % cx, (ce / cx) % cy, ce / (cx * cy));
                let mut any = false;
                for c in 0..8 {
                    let (i, j, k) = (2 * ci + (c & 1), 2 * cj + (c >> 1 & 1), 2 * ck + (c >> 2 & 1));
                    if i >= n[0] || j >= n[1] || k >= n[2] {
                        continue;
                    }
                    any |= child(((k * n[1] + j) * n[0] + i, c), out);
                }
                any
            })
            .collect();
        let mut l = Coarse {
            n: nc,
            mats,
            nz,
            inv_diag: vec![],
        };
        l.inv_diag = l
            .diagonal()
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();
        l
    }

    /// Calls `f(element, local node)` for each element around node `node`.
    fn around(n: [usize; 3], node: usize, mut f: impl FnMut(usize, usize)) {
        let [ex, ey, ez] = n;
        let (nx, ny) = (ex + 1, ey + 1);
        let (i, j, k) = (node % nx, (node / nx) % ny, node / (nx * ny));
        for a in 0..8 {
            let (di, dj, dk) = (a & 1, a >> 1 & 1, a >> 2 & 1);
            if i < di || j < dj || k < dk || i - di >= ex || j - dj >= ey || k - dk >= ez {
                continue;
            }
            f(((k - dk) * ey + (j - dj)) * ex + (i - di), a);
        }
    }

    fn diagonal(&self) -> Vec<f64> {
        let mut d = vec![0.0; 3 * self.nodes()];
        d.par_chunks_mut(3).enumerate().for_each(|(node, dn)| {
            Self::around(self.n, node, |e, a| {
                if self.nz[e] {
                    for c in 0..3 {
                        dn[c] += self.mats[576 * e + (3 * a + c) * 25];
                    }
                }
            });
        });
        d
    }

    fn apply(&self, u: &[f64], y: &mut [f64]) {
        y.par_chunks_mut(3).enumerate().for_each(|(node, yn)| {
            yn.fill(0.0);
            if self.inv_diag[3 * node..3 * node + 3].iter().all(|&v| v == 0.0) {
                return;
            }
            Self::around(self.n, node, |e, a| {
                if !self.nz[e] {
                    return;
                }
                let m = &self.mats[576 * e..576 * e + 576];
                let nodes = element_nodes(self.n, e);
                for c in 0..3 {
                    let row = &m[(3 * a + c) * 24..(3 * a + c) * 24 + 24];
                    let mut s = 0.0;
                    for (b, &nb) in nodes.iter().enumerate() {
                        let ub = &u[3 * nb..3 * nb + 3];
                        s += row[3 * b] * ub[0] + row[3 * b + 1] * ub[1] + row[3 * b + 2] * ub[2];
                    }
                    yn[c] += s;
                }
            });
        });
    }
}

/// Lower band of a symmetric positive definite matrix, Cholesky-factored in place.
struct Banded {
    n: usize,
    /// half bandwidth
    bw: usize,
    /// row i holds L[i][i - bw ..= i]
    l: Vec<f64>,
    /// node index -> position in the band ordering
    perm: Vec<usize>,
    /// per position: left out of the solve (a mechanism, see `factor`)
    dropped: Vec<bool>,
}

/// Node ordering for a small band: the axis with the most nodes slowest.
fn band_order(n: [usize; 3]) -> ([usize; 3], usize) {
    let nn = n.map(|v| v + 1);
    let mut axes = [0, 1, 2];
    axes.sort_by_key(|&a| nn[a]);
    let bw = 3 * (nn[axes[0]] * nn[axes[1]] + nn[axes[0]] + 1) + 2;
    (axes, bw)
}

fn direct_flops(n: [usize; 3]) -> f64 {
    let (_, bw) = band_order(n);
    let dofs = 3.0 * n.iter().map(|v| (v + 1) as f64).product::<f64>();
    dofs * (bw as f64).powi(2)
}

impl Banded {
    /// Factor a level. A DOF whose pivot falls to `drop` × its diagonal or below is a mechanism
    /// (in the matrix as given: rounding can make one, e.g. the GPU's f32 sums losing a void
    /// element's 1e-9 stiffness next to solid ones): it is left out of the solve (its row and
    /// column zeroed, the solution 0 there) rather than divided by ~0, which put 1e150 into the
    /// inverse and inf into its f32 copy (a 2D slab's GPU run after a coarse start).
    fn factor(level: &Coarse, drop: f64) -> Banded {
        let (axes, bw) = band_order(level.n);
        let nn = level.n.map(|v| v + 1);
        let nodes = level.nodes();
        let perm: Vec<usize> = (0..nodes)
            .map(|m| {
                let c = [m % nn[0], (m / nn[0]) % nn[1], m / (nn[0] * nn[1])];
                (c[axes[2]] * nn[axes[1]] + c[axes[1]]) * nn[axes[0]] + c[axes[0]]
            })
            .collect();
        let n = 3 * nodes;
        let w = bw + 1;
        let mut l = vec![0.0; n * w];
        for e in 0..level.nz.len() {
            if !level.nz[e] {
                continue;
            }
            let m = &level.mats[576 * e..576 * e + 576];
            let en = element_nodes(level.n, e);
            for r in 0..24 {
                let gi = 3 * perm[en[r / 3]] + r % 3;
                for s in 0..24 {
                    let gj = 3 * perm[en[s / 3]] + s % 3;
                    if gj <= gi {
                        l[gi * w + bw - (gi - gj)] += m[r * 24 + s];
                    }
                }
            }
        }
        // inactive DOFs: identity rows
        for i in 0..n {
            if l[i * w + bw] <= 0.0 {
                l[i * w..i * w + w].fill(0.0);
                l[i * w + bw] = 1.0;
            }
        }
        let mut dropped = vec![false; n];
        for i in 0..n {
            let j0 = i.saturating_sub(bw);
            let diag = l[i * w + bw];
            for j in j0..=i {
                if dropped[j] && j < i {
                    l[i * w + bw - (i - j)] = 0.0;
                    continue;
                }
                let k0 = j0.max(j.saturating_sub(bw));
                let mut s = l[i * w + bw - (i - j)];
                for k in k0..j {
                    s -= l[i * w + bw - (i - k)] * l[j * w + bw - (j - k)];
                }
                if i == j {
                    if s <= drop * diag {
                        dropped[i] = true;
                        l[i * w..i * w + w].fill(0.0);
                        l[i * w + bw] = 1.0;
                    } else {
                        l[i * w + bw] = s.sqrt();
                    }
                } else {
                    l[i * w + bw - (i - j)] = s / l[j * w + bw];
                }
            }
        }
        Banded {
            n,
            bw,
            l,
            perm,
            dropped,
        }
    }

    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let (n, bw, w) = (self.n, self.bw, self.bw + 1);
        let mut y = vec![0.0; n];
        for (m, &p) in self.perm.iter().enumerate() {
            y[3 * p..3 * p + 3].copy_from_slice(&b[3 * m..3 * m + 3]);
        }
        for i in 0..n {
            let mut s = y[i];
            for k in i.saturating_sub(bw)..i {
                s -= self.l[i * w + bw - (i - k)] * y[k];
            }
            y[i] = s / self.l[i * w + bw];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..(i + bw + 1).min(n) {
                s -= self.l[k * w + bw - (k - i)] * y[k];
            }
            y[i] = s / self.l[i * w + bw];
        }
        for (i, d) in self.dropped.iter().enumerate() {
            if *d {
                y[i] = 0.0;
            }
        }
        for (m, &p) in self.perm.iter().enumerate() {
            x[3 * m..3 * m + 3].copy_from_slice(&y[3 * p..3 * p + 3]);
        }
    }
}

/// Pᵀ (D ke D) P for each clamped-node mask (8 bits: a set bit zeroes that local node's rows and
/// columns) and child position: 256 × 8 matrices of 576, row-major. The GPU builds level 1 from it.
pub(super) fn masked_ptkp_table(ke: &[[f64; 24]; 24]) -> Vec<f64> {
    let ke: Vec<f64> = ke.iter().flatten().copied().collect();
    let weights: Vec<[[f64; 8]; 8]> = (0..8).map(child_weights).collect();
    let mut t = vec![0.0; 256 * 8 * 576];
    t.par_chunks_mut(8 * 576).enumerate().for_each(|(m, out)| {
        let mut km = [0.0; 576];
        for r in 0..24 {
            for s in 0..24 {
                if m >> (r / 3) & 1 == 0 && m >> (s / 3) & 1 == 0 {
                    km[r * 24 + s] = ke[r * 24 + s];
                }
            }
        }
        for c in 0..8 {
            add_ptkp(&km, &weights[c], 1.0, &mut out[576 * c..576 * c + 576]);
        }
    });
    t
}

/// Cost of the exact solve on a grid of `n` elements (for choosing the number of levels).
pub(super) fn direct_cost(n: [usize; 3]) -> f64 {
    direct_flops(n)
}

/// Dense inverse (row-major) of a level given as element matrices; inactive DOFs get identity rows.
pub(super) fn dense_inverse(n: [usize; 3], mats: Vec<f64>) -> Vec<f64> {
    let nz = mats.chunks(576).map(|m| m.iter().any(|&v| v != 0.0)).collect();
    let mut l = Coarse {
        n,
        mats,
        nz,
        inv_diag: vec![],
    };
    l.inv_diag = l
        .diagonal()
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
        .collect();
    // (the GPU's levels are summed in f32: a pivot that loses 6 digits is rounding)
    let chol = Banded::factor(&l, 1e-6);
    let nd = 3 * l.nodes();
    let mut inv = vec![0.0; nd * nd];
    // A⁻¹ is symmetric, so its columns are its rows
    inv.par_chunks_mut(nd).enumerate().for_each(|(j, row)| {
        let mut e = vec![0.0; nd];
        e[j] = 1.0;
        chol.solve(&e, row);
    });
    inv
}

/// Coarse correction: rc = Pᵀ D r, from a grid of `nf` elements to its coarse grid, with D zeroing the
/// DOFs where `mask` (an inverse diagonal) is zero.
fn restrict(nf: [usize; 3], r: &[f64], rc: &mut [f64], mask: &[f64]) {
    let [fx, fy, fz] = nf.map(|v| v + 1);
    let [cx, cy, _] = coarse_dims(nf).map(|v| v + 1);
    rc.par_chunks_mut(3).enumerate().for_each(|(m, out)| {
        let (ci, cj, ck) = (m % cx, (m / cx) % cy, m / (cx * cy));
        out.fill(0.0);
        for dk in -1isize..=1 {
            let k = (2 * ck) as isize + dk;
            if k < 0 || k >= fz as isize {
                continue;
            }
            let wk = if dk == 0 { 1.0 } else { 0.5 };
            for dj in -1isize..=1 {
                let j = (2 * cj) as isize + dj;
                if j < 0 || j >= fy as isize {
                    continue;
                }
                let wj = wk * if dj == 0 { 1.0 } else { 0.5 };
                for di in -1isize..=1 {
                    let i = (2 * ci) as isize + di;
                    if i < 0 || i >= fx as isize {
                        continue;
                    }
                    let w = wj * if di == 0 { 1.0 } else { 0.5 };
                    let f = 3 * ((k as usize * fy + j as usize) * fx + i as usize);
                    for c in 0..3 {
                        if mask[f + c] != 0.0 {
                            out[c] += w * r[f + c];
                        }
                    }
                }
            }
        }
    });
}

/// x += D P xc, D keeping the DOFs where `mask` (an inverse diagonal) is non-zero.
fn prolong_add(nf: [usize; 3], xc: &[f64], x: &mut [f64], mask: &[f64]) {
    let [fx, fy, _] = nf.map(|v| v + 1);
    let [cx, cy, _] = coarse_dims(nf).map(|v| v + 1);
    let split = |i: usize| -> [(usize, f64); 2] {
        if i.is_multiple_of(2) {
            [(i / 2, 1.0), (i / 2, 0.0)]
        } else {
            [(i / 2, 0.5), (i / 2 + 1, 0.5)]
        }
    };
    x.par_chunks_mut(3).enumerate().for_each(|(m, out)| {
        let (i, j, k) = (m % fx, (m / fx) % fy, m / (fx * fy));
        let mut s = [0.0; 3];
        for (ck, wk) in split(k) {
            if wk == 0.0 {
                continue;
            }
            for (cj, wj) in split(j) {
                if wj == 0.0 {
                    continue;
                }
                for (ci, wi) in split(i) {
                    if wi == 0.0 {
                        continue;
                    }
                    let w = wk * wj * wi;
                    let c = 3 * ((ck * cy + cj) * cx + ci);
                    for d in 0..3 {
                        s[d] += w * xc[c + d];
                    }
                }
            }
        }
        for d in 0..3 {
            if mask[3 * m + d] != 0.0 {
                out[d] += s[d];
            }
        }
    });
}

/// Scratch vectors for one level.
struct Work {
    r: Vec<f64>,
    t: Vec<f64>,
    d: Vec<f64>,
    /// the next coarser level's right-hand side and solution
    cb: Vec<f64>,
    cx: Vec<f64>,
}

pub(super) struct Multigrid<'a> {
    fem: &'a Fem<'a>,
    fine_inv: Vec<f64>,
    coarse: Vec<Coarse>,
    bottom: Banded,
    /// per level (fine first): estimated λmax(D⁻¹K)
    lmax: Vec<f64>,
    smoother: Smoother,
    work: Vec<Work>,
}

impl<'a> Multigrid<'a> {
    pub(super) fn new(fem: &'a Fem<'a>, s: &MgSettings) -> Multigrid<'a> {
        let mut coarse = vec![Coarse::from_fine(fem)];
        loop {
            let last = coarse.last().unwrap();
            if direct_flops(last.n) <= s.max_direct_flops || last.n.iter().all(|&v| v <= 1) {
                break;
            }
            let next = Coarse::from_coarse(last);
            coarse.push(next);
        }
        let bottom = Banded::factor(coarse.last().unwrap(), 1e-12);
        let mut mg = Multigrid {
            fem,
            fine_inv: fem.inv_diag(),
            coarse,
            bottom,
            lmax: vec![],
            smoother: s.smoother,
            work: vec![],
        };
        let sizes: Vec<usize> = (0..=mg.coarse.len()).map(|l| mg.len(l)).collect();
        mg.work = (0..mg.coarse.len())
            .map(|l| Work {
                r: vec![0.0; sizes[l]],
                t: vec![0.0; sizes[l]],
                d: vec![0.0; sizes[l]],
                cb: vec![0.0; sizes[l + 1]],
                cx: vec![0.0; sizes[l + 1]],
            })
            .collect();
        mg.lmax = (0..mg.coarse.len()).map(|l| mg.estimate_lmax(l)).collect();
        mg
    }

    /// Number of levels, including the fine one.
    pub(super) fn levels(&self) -> usize {
        self.coarse.len() + 1
    }

    /// Elements per axis of the coarsest level.
    pub(super) fn coarsest(&self) -> [usize; 3] {
        self.coarse.last().unwrap().n
    }

    fn len(&self, l: usize) -> usize {
        if l == 0 {
            self.fine_inv.len()
        } else {
            3 * self.coarse[l - 1].nodes()
        }
    }

    fn dims(&self, l: usize) -> [usize; 3] {
        if l == 0 { self.fem.p.n } else { self.coarse[l - 1].n }
    }

    fn inv(&self, l: usize) -> &[f64] {
        if l == 0 {
            &self.fine_inv
        } else {
            &self.coarse[l - 1].inv_diag
        }
    }

    fn apply(&self, l: usize, u: &[f64], y: &mut [f64]) {
        if l == 0 {
            self.fem.matvec(u, y)
        } else {
            self.coarse[l - 1].apply(u, y)
        }
    }

    /// λmax(D⁻¹K) by power iteration (D⁻¹K is self-adjoint in the D inner product).
    fn estimate_lmax(&self, l: usize) -> f64 {
        let inv = self.inv(l);
        let n = inv.len();
        // a fixed pseudo-random start, zero on inactive DOFs
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
                if inv[i] != 0.0 {
                    (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                } else {
                    0.0
                }
            })
            .collect();
        let mut kv = vec![0.0; n];
        let mut lambda = 0.0;
        for _ in 0..12 {
            self.apply(l, &v, &mut kv);
            // Rayleigh quotient vᵀKv / vᵀDv
            let num: f64 = v.par_iter().zip(&kv).map(|(a, b)| a * b).sum();
            let den: f64 = v
                .par_iter()
                .zip(inv)
                .map(|(a, d)| if *d != 0.0 { a * a / d } else { 0.0 })
                .sum();
            lambda = num / den.max(1e-300);
            // v <- D⁻¹ K v, normalized in the D norm
            v.par_iter_mut().zip(&kv).zip(inv).for_each(|((v, k), d)| *v = k * d);
            let norm: f64 = v
                .par_iter()
                .zip(inv)
                .map(|(a, d)| if *d != 0.0 { a * a / d } else { 0.0 })
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                break;
            }
            v.par_iter_mut().for_each(|x| *x /= norm);
        }
        lambda
    }

    /// z = M⁻¹ r: one V-cycle.
    pub(super) fn precondition(&mut self, r: &[f64], z: &mut [f64]) {
        let mut work = std::mem::take(&mut self.work);
        self.vcycle(0, r, z, &mut work);
        self.work = work;
    }

    fn vcycle(&self, l: usize, b: &[f64], x: &mut [f64], work: &mut [Work]) {
        if l == self.coarse.len() {
            self.bottom.solve(b, x);
            return;
        }
        let (w, rest) = work.split_first_mut().unwrap();
        x.fill(0.0);
        self.smooth(l, b, x, w, true);
        // coarse correction of the residual
        self.apply(l, x, &mut w.r);
        w.r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
        restrict(self.dims(l), &w.r, &mut w.cb, self.inv(l));
        self.vcycle(l + 1, &w.cb, &mut w.cx, rest);
        prolong_add(self.dims(l), &w.cx, x, self.inv(l));
        self.smooth(l, b, x, w, false);
    }

    fn smooth(&self, l: usize, b: &[f64], x: &mut [f64], w: &mut Work, zero_start: bool) {
        let inv = self.inv(l);
        let lmax = self.lmax[l];
        // w.r = b - K x
        let residual = |x: &[f64], r: &mut Vec<f64>| {
            if zero_start {
                r.copy_from_slice(b);
            } else {
                self.apply(l, x, r);
                r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
            }
        };
        match self.smoother {
            Smoother::Jacobi { sweeps, omega } => {
                let om = omega / lmax;
                for s in 0..sweeps {
                    if s == 0 {
                        residual(x, &mut w.r);
                    } else {
                        self.apply(l, x, &mut w.r);
                        w.r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
                    }
                    x.par_iter_mut()
                        .zip(&w.r)
                        .zip(inv)
                        .for_each(|((x, r), d)| *x += om * d * r);
                }
            }
            Smoother::Chebyshev { degree } => {
                let (lo, hi) = (0.1 * lmax, 1.1 * lmax);
                let (theta, delta) = (0.5 * (hi + lo), 0.5 * (hi - lo));
                let sigma = theta / delta;
                let mut rho = 1.0 / sigma;
                residual(x, &mut w.r);
                w.d.par_iter_mut()
                    .zip(&w.r)
                    .zip(inv)
                    .for_each(|((d, r), i)| *d = r * i / theta);
                for k in 1..=degree {
                    x.par_iter_mut().zip(&w.d).for_each(|(x, d)| *x += d);
                    if k == degree {
                        break;
                    }
                    self.apply(l, &w.d, &mut w.t);
                    w.r.par_iter_mut().zip(&w.t).for_each(|(r, t)| *r -= t);
                    let rho_new = 1.0 / (2.0 * sigma - rho);
                    let (a, c) = (rho_new * rho, 2.0 * rho_new / delta);
                    w.d.par_iter_mut()
                        .zip(&w.r)
                        .zip(inv)
                        .for_each(|((d, r), i)| *d = a * *d + c * r * i);
                    rho = rho_new;
                }
            }
        }
    }
}

/// CPU reference pieces for the GPU tests.
#[cfg(test)]
pub(super) mod tests_api {
    use super::*;

    pub fn restrict(nf: [usize; 3], r: &[f64], rc: &mut [f64], mask: &[f64]) {
        super::restrict(nf, r, rc, mask)
    }

    pub fn prolong_add(nf: [usize; 3], xc: &[f64], x: &mut [f64], mask: &[f64]) {
        super::prolong_add(nf, xc, x, mask)
    }

    type Apply = Box<dyn Fn(&[f64]) -> Vec<f64>>;

    /// Level 1's mat-vec and diagonal.
    pub fn level1(fem: &Fem) -> (Apply, Vec<f64>) {
        let c = Coarse::from_fine(fem);
        let d = c.diagonal();
        (
            Box::new(move |v| {
                let mut y = vec![0.0; v.len()];
                c.apply(v, &mut y);
                y
            }),
            d,
        )
    }

    pub fn level2(fem: &Fem) -> Apply {
        let c = Coarse::from_coarse(&Coarse::from_fine(fem));
        Box::new(move |v| {
            let mut y = vec![0.0; v.len()];
            c.apply(v, &mut y);
            y
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_weights_partition_unity() {
        for c in 0..8 {
            let w = child_weights(c);
            for a in 0..8 {
                let s: f64 = w[a].iter().sum();
                assert!((s - 1.0).abs() < 1e-15);
            }
        }
        // child 0's node 0 is coarse node 0; its far node is the coarse element's centre
        let w = child_weights(0);
        assert_eq!(w[0][0], 1.0);
        assert!(w[7].iter().all(|&v| v == 0.125));
    }

    /// Restriction is the transpose of prolongation, on an odd-sized grid.
    #[test]
    fn restrict_is_prolong_transpose() {
        let nf = [3, 2, 5];
        let nfn = 3 * nf.iter().map(|v| v + 1).product::<usize>();
        let ncn = 3 * coarse_dims(nf).iter().map(|v| v + 1).product::<usize>();
        let f: Vec<f64> = (0..nfn).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
        let c: Vec<f64> = (0..ncn).map(|i| ((i * 104729) % 11) as f64 - 5.0).collect();
        let mut pc = vec![0.0; nfn];
        prolong_add(nf, &c, &mut pc, &vec![1.0; nfn]);
        let mut rf = vec![0.0; ncn];
        restrict(nf, &f, &mut rf, &vec![1.0; nfn]);
        let a: f64 = f.iter().zip(&pc).map(|(x, y)| x * y).sum();
        let b: f64 = c.iter().zip(&rf).map(|(x, y)| x * y).sum();
        assert!((a - b).abs() < 1e-9 * a.abs().max(1.0), "{a} vs {b}");
    }

    /// A small problem with holes, clamped nodes and a stiffness contrast.
    fn problem(n: [usize; 3]) -> super::super::Problem {
        let ne: usize = n.iter().product();
        let nn: usize = n.iter().map(|v| v + 1).product();
        let mut p = super::super::Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            // a box with a square hole through it along y
            domain: (0..ne)
                .map(|e| {
                    let (i, k) = (e % n[0], e / (n[0] * n[1]));
                    !(i > n[0] / 3 && i < n[0] / 2 && k > n[2] / 3 && k < n[2] / 2)
                })
                .collect(),
            passive: vec![false; ne],
            fixed: vec![0; nn],
            loads: vec![vec![[0.0; 3]; nn]],
            cases: vec![1],
            extrude: None,
        };
        let [nx, ny, _] = p.nodes();
        for j in 0..ny {
            for i in 0..nx {
                let m = p.node(i, j, 0);
                p.fixed[m] = 7;
            }
        }
        for j in 0..ny {
            let m = p.node(nx - 1, j, n[2]);
            p.loads[0][m] = [0.3, -1.0, 0.5];
        }
        p
    }

    fn fem<'a>(p: &'a super::super::Problem, ke: &'a [[f64; 24]; 24]) -> Fem<'a> {
        Fem {
            p,
            ke,
            // solid diagonal struts in a soft (1e-9) matrix, a design midway through an optimization
            ee: (0..p.domain.len())
                .map(|e| {
                    let [ex, ey, _] = p.n;
                    let (i, j, k) = (e % ex, (e / ex) % ey, e / (ex * ey));
                    let strut = (i + k) % 6 < 2 || (i + 2 * ex - k % (2 * ex)) % 5 < 2 || j == 0;
                    if !p.domain[e] {
                        0.0
                    } else if strut {
                        1.0
                    } else {
                        1e-9
                    }
                })
                .collect(),
            free: p.free_mask(),
        }
    }

    fn pseudo(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104729) % 23) as f64 / 11.0 - 1.0)
            .collect()
    }

    /// The stored coarse operators equal Pᵀ (D K D) P applied explicitly, on two levels.
    #[test]
    fn galerkin_matches_explicit_product() {
        let p = problem([5, 4, 7]);
        let ke = super::super::hex_ke(0.3);
        let fem = fem(&p, &ke);
        let mask: Vec<f64> = (0..3 * fem.free.len())
            .map(|i| if fem.free[i / 3] >> (i % 3) & 1 == 1 { 1.0 } else { 0.0 })
            .collect();
        let c1 = Coarse::from_fine(&fem);
        let n1 = 3 * c1.nodes();
        let v = pseudo(n1, 1);
        let mut pv = vec![0.0; mask.len()];
        prolong_add(p.n, &v, &mut pv, &mask);
        let mut kpv = vec![0.0; mask.len()];
        fem.matvec(&pv, &mut kpv);
        let mut want = vec![0.0; n1];
        restrict(p.n, &kpv, &mut want, &mask);
        let mut got = vec![0.0; n1];
        c1.apply(&v, &mut got);
        let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for i in 0..n1 {
            assert!(
                (got[i] - want[i]).abs() < 1e-12 * scale,
                "level 1 dof {i}: {} vs {}",
                got[i],
                want[i]
            );
        }

        let c2 = Coarse::from_coarse(&c1);
        let n2 = 3 * c2.nodes();
        let v = pseudo(n2, 2);
        let mut pv = vec![0.0; n1];
        prolong_add(c1.n, &v, &mut pv, &vec![1.0; n1]);
        let mut kpv = vec![0.0; n1];
        c1.apply(&pv, &mut kpv);
        let mut want = vec![0.0; n2];
        restrict(c1.n, &kpv, &mut want, &vec![1.0; n1]);
        let mut got = vec![0.0; n2];
        c2.apply(&v, &mut got);
        let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for i in 0..n2 {
            assert!(
                (got[i] - want[i]).abs() < 1e-12 * scale,
                "level 2 dof {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    /// The banded Cholesky solves its level exactly.
    #[test]
    fn banded_cholesky_solves() {
        let p = problem([6, 3, 5]);
        let ke = super::super::hex_ke(0.3);
        let fem = fem(&p, &ke);
        let c = Coarse::from_fine(&fem);
        let chol = Banded::factor(&c, 1e-12);
        let n = 3 * c.nodes();
        let b: Vec<f64> = pseudo(n, 3)
            .iter()
            .zip(&c.inv_diag)
            .map(|(b, d)| if *d != 0.0 { *b } else { 0.0 })
            .collect();
        let mut x = vec![0.0; n];
        chol.solve(&b, &mut x);
        let mut kx = vec![0.0; n];
        c.apply(&x, &mut kx);
        let err = kx.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let bmax = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-8 * bmax, "residual {err}");
    }

    /// MGCG reaches the Jacobi-CG solution in far fewer iterations, with either smoother.
    #[test]
    fn mgcg_matches_jacobi_cg() {
        let p = problem([12, 6, 16]);
        let ke = super::super::hex_ke(0.3);
        let fem = fem(&p, &ke);
        let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
        let inv = fem.inv_diag();
        let mut jacobi = |r: &[f64], z: &mut [f64]| {
            for i in 0..r.len() {
                z[i] = r[i] * inv[i];
            }
        };
        let mut uj = vec![0.0; f.len()];
        let nj = fem.solve(&f, &mut uj, 1e-10, 100_000, &mut jacobi);
        let umax = uj.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for smoother in [
            Smoother::Chebyshev { degree: 2 },
            Smoother::Jacobi { sweeps: 2, omega: 1.2 },
        ] {
            let settings = MgSettings {
                smoother,
                max_direct_flops: 1e5,
            };
            let mut m = Multigrid::new(&fem, &settings);
            assert!(m.levels() >= 3, "{}", m.levels());
            let mut u = vec![0.0; f.len()];
            let n = fem.solve(&f, &mut u, 1e-10, 1000, &mut |r, z| m.precondition(r, z));
            let err = u.iter().zip(&uj).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-6 * umax, "{smoother:?}: error {err} of {umax}");
            eprintln!("{smoother:?}: {n} MGCG vs {nj} Jacobi-CG iterations");
            assert!(n * 5 < nj, "{smoother:?}: {n} MGCG vs {nj} Jacobi-CG iterations");
        }
    }
}
