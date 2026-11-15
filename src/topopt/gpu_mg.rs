//! Multigrid-preconditioned CG on the GPU, in f32: the same method as `mg.rs`, with every vector
//! resident on the GPU.
//!
//! - Level 0 (fine) is matrix-free.
//! - Level 1 is never stored: K1 u = Pᵀ (D K D) P u is applied through the fine grid (prolong,
//!   fine mat-vec, restrict), which costs about a fine mat-vec and no memory per element. Its
//!   diagonal and level 2 are built from a table of Pᵀ (D ke D) P per clamped-node mask and child
//!   position (`mg::masked_ptkp_table`).
//! - Stored matrices keep their upper triangle (300 of 576 entries), entry-major
//!   (`mats[entry · elements + e]`) so neighbouring threads read neighbouring addresses.
//! - Levels ≥ 2 are assembled on the GPU as 24×24 element matrices (Galerkin, one workgroup per
//!   coarse element).
//! - The coarsest level is solved with a dense inverse, computed on the CPU once per optimizer
//!   iteration and applied on the GPU, so nothing round-trips inside a V-cycle.
//! - CG's scalars (α, β, ρ) live in a small GPU buffer; the residual norm is read back every
//!   few iterations to test convergence.
//!
//! Each phase (setup, CG start, CG iteration, energies) is recorded once as a list of dispatches
//! with fixed bind groups and replayed.

use anyhow::{Result, bail};

use super::gpu_engine::{Engine, NONE, Params, Step, dims4, groups, threads};
use super::mg::{self, MgSettings, Smoother};
use super::{CaseSolve, LoadBasis, Problem};

/// Power iteration steps for the smoothers' λmax estimates, and for a careful re-estimate when a
/// solve stalls right after one (`reestimate_long`).
const POWER_STEPS: usize = 12;
const POWER_STEPS_LONG: usize = 40;

/// CG broke down (a non-finite residual): the preconditioner wasn't positive definite, which an
/// underestimated smoother λmax causes. The solution was reset to zero, so the solve can be
/// retried after a careful re-estimate.
#[derive(Debug)]
pub(super) struct Diverged(pub f32);

impl std::fmt::Display for Diverged {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GPU CG diverged (residual {})", self.0)
    }
}

impl std::error::Error for Diverged {}

/// Entries of a symmetric 24×24 element matrix's upper triangle (how coarse levels are stored).
const SYM: usize = 300;

/// CG iterations between convergence checks.
const CHECK_EVERY: usize = 4;

// scalar slots
const RR: u32 = 0;
const RZ: u32 = 2;
const RZN: u32 = 3;
const PQ: u32 = 4;
const ALPHA: u32 = 5;
const BETA: u32 = 6;
const NUM: u32 = 7;
const DEN: u32 = 8;
const NRM: u32 = 9;
const SC: u32 = 10;
/// λmax of level l at LAM + l, its inverse at INVL + l (the first hierarchy; the others of a
/// symmetric model keep theirs at `lam_slots`)
const LAM: u32 = 16;
const INVL: u32 = 48;

/// The scalar slots of hierarchy `group`'s λmax estimates and their inverses (each hierarchy has
/// its own: a group's smoothers mustn't use another's estimates).
fn lam_slots(group: usize) -> (u32, u32) {
    match group {
        0 => (LAM, INVL),
        g => {
            let base = 320 + 32 * (g as u32 - 1);
            (base, base + 16)
        }
    }
}

/// Most hierarchies (groups of a symmetric model: 2^planes).
pub(super) const MAX_GROUPS: usize = 8;

// scalar kernel ops
const OP_DIV: u32 = 0;
const OP_COPY: u32 = 1;
const OP_RSQRT: u32 = 2;
const OP_RECIP: u32 = 3;

fn nodes(n: [usize; 3]) -> usize {
    n.iter().map(|v| v + 1).product()
}

fn coarse(n: [usize; 3]) -> [usize; 3] {
    n.map(|v| v.div_ceil(2))
}

/// `apply_p`'s kernel and tile of nodes (one workgroup, at most 128): 8×4×4, or 2×8×8 across a slab
/// one element thick (a 2D problem), where 8 nodes along the thin axis would leave three quarters of
/// the threads idle (+50% solve time on `monitor_table`). The tiles must match the kernels' `override`
/// constants (`gpu_engine::VARIANTS`); the workgroup arrays fit both (halo 400 nodes, coarse block 108).
fn apply_p_tile(n: [usize; 3]) -> (&'static str, [usize; 3]) {
    match n.iter().position(|&v| v == 1) {
        Some(0) => ("apply_p_slab_x", [2, 8, 8]),
        Some(1) => ("apply_p_slab_y", [8, 2, 8]),
        Some(_) => ("apply_p_slab_z", [8, 8, 2]),
        None => ("apply_p", [8, 4, 4]),
    }
}

pub(super) struct GpuMg<'a> {
    e: &'a Engine<'a>,
    /// this hierarchy's λmax slots (level l at + l) and their inverses
    lam: u32,
    invl: u32,
    /// elements per axis, per level (0 = fine)
    dims: Vec<[usize; 3]>,
    smoother: Smoother,
    pub(super) ee: wgpu::Buffer,
    free: wgpu::Buffer,
    ke: wgpu::Buffer,
    table: wgpu::Buffer,
    pub(super) ce: wgpu::Buffer,
    diag: Vec<wgpu::Buffer>,
    /// element matrices, levels ≥ 2 (index = level)
    mats: Vec<Option<wgpu::Buffer>>,
    /// right-hand side and solution, levels ≥ 1
    lb: Vec<Option<wgpu::Buffer>>,
    lx: Vec<Option<wgpu::Buffer>>,
    /// smoother scratch (residual, direction), smoothed levels
    lr: Vec<wgpu::Buffer>,
    ld: Vec<wgpu::Buffer>,
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    p: wgpu::Buffer,
    q: wgpu::Buffer,
    /// per solved load case (`LoadBasis::solved`): the displacement, kept for warm starts
    pub(super) us: Vec<wgpu::Buffer>,
    /// per load case: its slot in `us`, or None when it is a combination of solved cases
    slot: Vec<Option<usize>>,
    /// a combined case's displacement, formed when needed (None if every case is solved)
    uc: Option<wgpu::Buffer>,
    /// per load case: a combined case's coefficients over the solved slots (empty when solved)
    combos: Vec<Vec<(usize, f64)>>,
    /// per solved load case: DOF indices, values, count
    fs: Vec<(wgpu::Buffer, wgpu::Buffer, usize)>,
    /// per solved load case: |f|²
    ff: Vec<f64>,
    bottom: wgpu::Buffer,
    /// diagonals and Galerkin coarse levels
    setup: Vec<Step>,
    /// the smoothers' λmax(D⁻¹K) estimates (power iteration), and a longer one
    estimate: Vec<Step>,
    estimate_long: Vec<Step>,
    init: Vec<Vec<Step>>,
    iter: Vec<Vec<Step>>,
}

impl<'a> GpuMg<'a> {
    /// Buffers and programs for a problem. `free` per node (free axes as bits), as in `Fem`.
    pub(super) fn new(
        e: &'a Engine<'a>,
        p: &Problem,
        basis: &LoadBasis,
        ke: &[[f64; 24]; 24],
        free: &[u8],
        s: &MgSettings,
        shared: Option<(&GpuMg<'a>, usize)>,
    ) -> Result<Self> {
        let (lam, invl) = lam_slots(shared.map_or(0, |(_, g)| g));
        let shared = shared.map(|(o, _)| o);
        // levels: at least fine, level 1 (virtual) and one stored level
        let mut dims = vec![p.n, coarse(p.n), coarse(coarse(p.n))];
        loop {
            let last = *dims.last().unwrap();
            if mg::direct_cost(last) <= s.max_direct_flops || last.iter().all(|&v| v <= 1) {
                break;
            }
            dims.push(coarse(last));
        }
        let last = dims.len() - 1;
        let ne: usize = p.n.iter().product();
        let nf = 3 * nodes(p.n);
        let limit = e.max_binding();
        let bytes2 = (SYM * 4 * dims[2].iter().product::<usize>()) as u64;
        if (nf * 4) as u64 > limit || bytes2 > limit {
            bail!(
                "grid too large for the GPU solver ({} MB per vector, {} MB for level 2; limit {} MB)",
                (nf * 4) >> 20,
                bytes2 >> 20,
                limit >> 20
            );
        }

        let buf = |label: &str, len: usize| e.buf(label, len);
        let init = |label: &str, data: &[u8]| e.init(label, data);
        let ke32: Vec<f32> = ke.iter().flatten().map(|&v| v as f32).collect();
        let table: Vec<f32> = mg::masked_ptkp_table(ke).iter().map(|&v| v as f32).collect();
        let free32: Vec<u32> = free.iter().map(|&f| f as u32).collect();
        let len = |l: usize| 3 * nodes(dims[l]);
        // loads touch few nodes: kept as (DOF, value) lists, with |f|² for the stopping test
        let mut fs = vec![];
        let mut ff = vec![];
        for l in basis.solved.iter().map(|&c| &p.loads[c]) {
            let (mut idx, mut val, mut sq) = (vec![], vec![], 0.0);
            for (m, v) in l.iter().enumerate() {
                for (c, &x) in v.iter().enumerate() {
                    if free[m] >> c & 1 == 1 && x != 0.0 {
                        idx.push((3 * m + c) as u32);
                        val.push(x as f32);
                        sq += x * x;
                    }
                }
            }
            fs.push((
                init("f dofs", bytemuck::cast_slice(&idx)),
                init("f values", bytemuck::cast_slice(&val)),
                idx.len(),
            ));
            ff.push(sq);
        }
        let bdofs = len(last);
        // another group's hierarchy on the same grid (symmetry): its element stiffnesses and
        // energies, element tables and CG scratch vectors are used here too (the groups are solved
        // one after another)
        let share = |f: for<'b> fn(&'b GpuMg<'a>) -> &'b wgpu::Buffer, make: &dyn Fn() -> wgpu::Buffer| match shared {
            Some(o) => f(o).clone(),
            None => make(),
        };
        let q = share(|o| &o.q, &|| buf("cg q", nf));
        let z = share(|o| &o.z, &|| buf("cg z", nf));
        let p_ = share(|o| &o.p, &|| buf("cg p", nf));
        let mut g = GpuMg {
            e,
            lam,
            invl,
            smoother: s.smoother,
            ee: share(|o| &o.ee, &|| buf("ee", ne)),
            free: init("free", bytemuck::cast_slice(&free32)),
            ke: share(|o| &o.ke, &|| init("ke", bytemuck::cast_slice(&ke32))),
            table: share(|o| &o.table, &|| init("ptkp table", bytemuck::cast_slice(&table))),
            ce: share(|o| &o.ce, &|| buf("ce", ne)),
            // level 0's diagonal isn't kept (the smoother works it out, `cheb_fine`): only the λmax
            // estimate needs it, between solves, in CG's p
            diag: (0..=last)
                .map(|l| if l == 0 { p_.clone() } else { buf("diag", len(l)) })
                .collect(),
            mats: (0..=last)
                .map(|l| (l >= 2).then(|| buf("mats", SYM * dims[l].iter().product::<usize>())))
                .collect(),
            lb: (0..=last).map(|l| (l >= 1).then(|| buf("b", len(l)))).collect(),
            lx: (0..=last).map(|l| (l >= 1).then(|| buf("x", len(l)))).collect(),
            // level 0's smoother residual shares CG's q, which is free during the V-cycle
            lr: (0..last)
                .map(|l| if l == 0 { q.clone() } else { buf("r", len(l)) })
                .collect(),
            // level 0's smoother direction is CG's q too (`smooth`); this one is only the λmax
            // estimate's, between solves, in CG's z
            ld: (0..last)
                .map(|l| if l == 0 { z.clone() } else { buf("d", len(l)) })
                .collect(),
            r: share(|o| &o.r, &|| buf("cg r", nf)),
            z,
            p: p_,
            q,
            us: basis.solved.iter().map(|_| buf("u", nf)).collect(),
            slot: basis
                .cases
                .iter()
                .map(|c| match c {
                    CaseSolve::Solved(i) => Some(*i),
                    CaseSolve::Combined(_) => None,
                })
                .collect(),
            uc: (basis.solved.len() < basis.cases.len()).then(|| buf("u combined", nf)),
            combos: basis
                .cases
                .iter()
                .map(|c| match c {
                    CaseSolve::Combined(a) => a.clone(),
                    CaseSolve::Solved(_) => vec![],
                })
                .collect(),
            fs,
            ff,
            bottom: buf("bottom inverse", bdofs * bdofs),
            dims,
            setup: vec![],
            estimate: vec![],
            estimate_long: vec![],
            init: vec![],
            iter: vec![],
        };
        g.setup = g.record_setup();
        g.estimate = g.record_estimate(POWER_STEPS);
        g.estimate_long = g.record_estimate(POWER_STEPS_LONG);
        for c in 0..g.us.len() {
            let i = g.record_init(c);
            g.init.push(i);
            let i = g.record_iter(c);
            g.iter.push(i);
        }
        Ok(g)
    }

    /// Record: form load case `c`'s displacement (if it is a combination) and add its element
    /// energies u_eᵀ ke u_e to `ce` (overwriting it when not `accumulate`). `case_u(c)` holds the
    /// displacement afterwards, until another case is formed.
    pub(super) fn record_case(&self, out: &mut Vec<Step>, c: usize, accumulate: bool) {
        let ne: usize = self.dims[0].iter().product();
        out.extend(self.form_steps(c));
        self.e.step(
            out,
            "energy",
            Params {
                n: dims4(self.dims[0]),
                m: [accumulate as u32, 0, 0, 0],
                ..Default::default()
            },
            &[&self.ee, &self.ke, self.case_u(c), &self.ce],
            threads(ne),
        );
    }

    /// The steps that form combined case `c` in `uc`: Σ a u_slot, the first term overwriting it
    /// (no terms: a zero load). None for a solved case.
    fn form_steps(&self, c: usize) -> Vec<Step> {
        let mut out = vec![];
        let nf = self.len(0);
        if let (Some(uc), None) = (&self.uc, self.slot[c]) {
            self.e.lin(&mut out, uc, &self.r, None, (0.0, NONE), (0.0, NONE), nf);
            for &(s, w) in &self.combos[c] {
                self.e
                    .lin(&mut out, uc, &self.us[s], None, (1.0, NONE), (w as f32, NONE), nf);
            }
        }
        out
    }

    /// Load case `c`'s displacement buffer (for a combined case, valid after its form steps ran).
    pub(super) fn case_u(&self, c: usize) -> &wgpu::Buffer {
        match self.slot[c] {
            Some(s) => &self.us[s],
            None => self.uc.as_ref().unwrap(),
        }
    }

    pub(super) fn levels(&self) -> usize {
        self.dims.len()
    }

    pub(super) fn coarsest(&self) -> [usize; 3] {
        *self.dims.last().unwrap()
    }

    fn last(&self) -> usize {
        self.dims.len() - 1
    }

    fn len(&self, l: usize) -> usize {
        3 * nodes(self.dims[l])
    }

    /// y = K_l u
    fn apply(&self, out: &mut Vec<Step>, l: usize, u: &wgpu::Buffer, y: &wgpu::Buffer) {
        self.apply_mode(out, l, u, y, false)
    }

    /// y = K_l u, or y -= K_l u when `sub`
    fn apply_mode(&self, out: &mut Vec<Step>, l: usize, u: &wgpu::Buffer, y: &wgpu::Buffer, sub: bool) {
        let t = threads(nodes(self.dims[l]));
        let m = [sub as u32, 0, 0, 0];
        match l {
            0 => self.e.step(
                out,
                "apply_fine",
                Params {
                    n: dims4(self.dims[0]),
                    m,
                    ..Default::default()
                },
                &[&self.ee, &self.free, &self.ke, u, y],
                t,
            ),
            1 => {
                // Pᵀ (D K D) P u through the fine grid (`apply_p` prolongs on the fly), in a fine
                // vector that is free whenever level 1 is worked on: level 0's residual, already
                // restricted. Cheaper than stored or on-the-fly element matrices.
                let kpu = &self.lr[0];
                let (kernel, tile) = apply_p_tile(self.dims[0]);
                self.e.step(
                    out,
                    kernel,
                    Params {
                        n: dims4(self.dims[0]),
                        ..Default::default()
                    },
                    &[&self.ee, &self.free, &self.ke, u, kpu],
                    groups(
                        self.dims[0]
                            .iter()
                            .zip(tile)
                            .map(|(v, t)| (v + 1).div_ceil(t))
                            .product(),
                    ),
                );
                self.e.step(
                    out,
                    "restrict",
                    Params {
                        n: dims4(self.dims[0]),
                        m: [sub as u32, 1, 0, 0],
                        ..Default::default()
                    },
                    &[kpu, &self.free, y],
                    t,
                );
            }
            _ => self.e.step(
                out,
                "apply_elem",
                Params {
                    n: dims4(self.dims[l]),
                    m,
                    ..Default::default()
                },
                &[self.mats[l].as_ref().unwrap(), u, y],
                t,
            ),
        }
    }

    fn record_setup(&self) -> Vec<Step> {
        let mut out = vec![];
        let last = self.last();
        self.e.step(
            &mut out,
            "diag_l1",
            Params {
                n: dims4(self.dims[0]),
                ..Default::default()
            },
            &[&self.ee, &self.free, &self.table, &self.diag[1], &self.ke],
            threads(nodes(self.dims[1])),
        );
        for l in 2..=last {
            let mats = self.mats[l].as_ref().unwrap();
            let (n, virt, child) = if l == 2 {
                (self.dims[0], 1, &self.table)
            } else {
                (self.dims[l - 1], 0, self.mats[l - 1].as_ref().unwrap())
            };
            self.e.step(
                &mut out,
                "assemble",
                Params {
                    n: dims4(n),
                    m: [virt, 0, 0, 0],
                    ..Default::default()
                },
                &[&self.ee, &self.free, &self.table, child, mats, &self.ke],
                groups(self.dims[l].iter().product()),
            );
            self.e.step(
                &mut out,
                "diag_elem",
                Params {
                    n: dims4(self.dims[l]),
                    ..Default::default()
                },
                &[mats, &self.diag[l]],
                threads(nodes(self.dims[l])),
            );
        }
        out
    }

    /// λmax(D⁻¹K) per smoothed level, by power iteration from a random start. This is most of the
    /// setup's cost (12 mat-vecs on levels 0 and 1), and λmax drifts only a few % per optimizer
    /// iteration, so the optimizer re-runs it only now and then (`build(estimate)`).
    fn record_estimate(&self, steps: usize) -> Vec<Step> {
        let mut out = vec![];
        // level 0's diagonal, in CG's p (see `new`)
        self.e.step(
            &mut out,
            "diag_fine",
            Params {
                n: dims4(self.dims[0]),
                ..Default::default()
            },
            &[&self.ee, &self.free, &self.ke, &self.diag[0]],
            threads(nodes(self.dims[0])),
        );
        for l in 0..self.last() {
            let (v, kv, d, n) = (&self.lr[l], &self.ld[l], &self.diag[l], self.len(l));
            self.e.step(
                &mut out,
                "fill_random",
                Params {
                    m: [n as u32, 0, 0, 0],
                    ..Default::default()
                },
                &[d, v],
                threads(n),
            );
            for _ in 0..steps {
                self.apply(&mut out, l, v, kv);
                self.e.dot(&mut out, v, kv, None, n, NUM);
                self.e.dot(&mut out, v, v, Some(d), n, DEN);
                self.e.scalar(&mut out, OP_DIV, NUM, DEN, self.lam + l as u32);
                self.e.lin(&mut out, v, kv, Some(d), (0.0, NONE), (1.0, NONE), n);
                self.e.dot(&mut out, v, v, Some(d), n, NRM);
                self.e.scalar(&mut out, OP_RSQRT, NRM, NRM, SC);
                self.e.lin(&mut out, v, kv, None, (1.0, SC), (0.0, NONE), n);
            }
            self.e
                .scalar(&mut out, OP_RECIP, self.lam + l as u32, 0, self.invl + l as u32);
        }
        out
    }

    fn smooth(&self, out: &mut Vec<Step>, l: usize, b: &wgpu::Buffer, x: &wgpu::Buffer, zero_start: bool) {
        if l == 0 {
            return self.smooth_fine(out, b, x, zero_start);
        }
        let (r, d, diag, n) = (&self.lr[l], &self.ld[l], &self.diag[l], self.len(l));
        let invl = self.invl + l as u32;
        // r = b - K x
        let residual = |out: &mut Vec<Step>, zero: bool| {
            if zero {
                self.e.lin(out, r, b, None, (0.0, NONE), (1.0, NONE), n);
            } else {
                self.apply(out, l, x, r);
                self.e.lin(out, r, b, None, (-1.0, NONE), (1.0, NONE), n);
            }
        };
        match self.smoother {
            Smoother::Jacobi { sweeps, omega } => {
                for k in 0..sweeps {
                    let zero = zero_start && k == 0;
                    residual(out, zero);
                    let a = if zero { 0.0 } else { 1.0 };
                    self.e.lin(out, x, r, Some(diag), (a, NONE), (omega as f32, invl), n);
                }
            }
            Smoother::Chebyshev { degree } => {
                // on [0.1, 1.1]·λmax: θ = 0.6 λ, δ = 0.5 λ, σ = θ / δ
                let sigma = 1.2f64;
                let mut rho = 1.0 / sigma;
                residual(out, zero_start);
                self.e
                    .lin(out, d, r, Some(diag), (0.0, NONE), ((1.0 / 0.6) as f32, invl), n);
                for k in 1..=degree {
                    let a = if zero_start && k == 1 { 0.0 } else { 1.0 };
                    self.e.lin(out, x, d, None, (a, NONE), (1.0, NONE), n);
                    if k == degree {
                        break;
                    }
                    // r -= K d, fused
                    self.apply_mode(out, l, d, r, true);
                    let rho_new = 1.0 / (2.0 * sigma - rho);
                    self.e.lin(
                        out,
                        d,
                        r,
                        Some(diag),
                        ((rho_new * rho) as f32, NONE),
                        ((2.0 * rho_new / 0.5) as f32, invl),
                        n,
                    );
                    rho = rho_new;
                }
            }
        }
    }

    /// d = α d + γ D⁻¹ (b − K x) on the fine grid (`cheb_fine`; the plain residual when `scaled` is
    /// false), γ = c.0 × S[c.1].
    #[allow(clippy::too_many_arguments)]
    fn cheb_fine(
        &self,
        out: &mut Vec<Step>,
        b: &wgpu::Buffer,
        x: &wgpu::Buffer,
        d: &wgpu::Buffer,
        alpha: f32,
        c: (f32, u32),
        zero_x: bool,
        scaled: bool,
    ) {
        self.e.step(
            out,
            "cheb_fine",
            Params {
                n: dims4(self.dims[0]),
                m: [zero_x as u32, !scaled as u32, c.1, 0],
                f: [alpha, c.0, 0.0, 0.0],
                ..Default::default()
            },
            &[&self.e.scalars, &self.ee, &self.free, &self.ke, x, b, d],
            threads(nodes(self.dims[0])),
        );
    }

    /// `smooth` on the fine grid, the same iteration with neither the diagonal nor a residual
    /// stored: each step recomputes D⁻¹ (b − K x) from x (`cheb_fine`, one mat-vec as before), and
    /// the direction lives in CG's q, which is free during the V-cycle.
    fn smooth_fine(&self, out: &mut Vec<Step>, b: &wgpu::Buffer, x: &wgpu::Buffer, zero_start: bool) {
        let (d, n) = (&self.q, self.len(0));
        let invl = self.invl;
        match self.smoother {
            Smoother::Jacobi { sweeps, omega } => {
                for k in 0..sweeps {
                    let zero = zero_start && k == 0;
                    self.cheb_fine(out, b, x, d, 0.0, (omega as f32, invl), zero, true);
                    let a = if zero { 0.0 } else { 1.0 };
                    self.e.lin(out, x, d, None, (a, NONE), (1.0, NONE), n);
                }
            }
            Smoother::Chebyshev { degree } => {
                // as `smooth`
                let sigma = 1.2f64;
                let mut rho = 1.0 / sigma;
                self.cheb_fine(out, b, x, d, 0.0, ((1.0 / 0.6) as f32, invl), zero_start, true);
                for k in 1..=degree {
                    let a = if zero_start && k == 1 { 0.0 } else { 1.0 };
                    self.e.lin(out, x, d, None, (a, NONE), (1.0, NONE), n);
                    if k == degree {
                        break;
                    }
                    let rho_new = 1.0 / (2.0 * sigma - rho);
                    self.cheb_fine(
                        out,
                        b,
                        x,
                        d,
                        (rho_new * rho) as f32,
                        ((2.0 * rho_new / 0.5) as f32, invl),
                        false,
                        true,
                    );
                    rho = rho_new;
                }
            }
        }
    }

    fn vcycle(&self, out: &mut Vec<Step>, l: usize, b: &wgpu::Buffer, x: &wgpu::Buffer) {
        if l == self.last() {
            let n = self.len(l);
            self.e.step(
                out,
                "bottom",
                Params {
                    m: [n as u32, 0, 0, 0],
                    ..Default::default()
                },
                &[&self.bottom, b, x],
                threads(n),
            );
            return;
        }
        let n = self.len(l);
        self.smooth(out, l, b, x, true);
        let r = &self.lr[l];
        if l == 0 {
            self.cheb_fine(out, b, x, r, 0.0, (1.0, NONE), false, false);
        } else {
            self.apply(out, l, x, r);
            self.e.lin(out, r, b, None, (-1.0, NONE), (1.0, NONE), n);
        }
        // the fine grid's transfer mask is its free axes (its diagonal isn't stored)
        let (mask, bits) = if l == 0 { (&self.free, 1) } else { (&self.diag[l], 0) };
        let (cb, cx) = (self.lb[l + 1].as_ref().unwrap(), self.lx[l + 1].as_ref().unwrap());
        self.e.step(
            out,
            "restrict",
            Params {
                n: dims4(self.dims[l]),
                m: [0, bits, 0, 0],
                ..Default::default()
            },
            &[r, mask, cb],
            threads(nodes(self.dims[l + 1])),
        );
        self.vcycle(out, l + 1, cb, cx);
        self.e.step(
            out,
            "prolong",
            Params {
                n: dims4(self.dims[l]),
                m: [0, bits, 0, 0],
                ..Default::default()
            },
            &[cx, mask, x],
            threads(nodes(self.dims[l])),
        );
        self.smooth(out, l, b, x, false);
    }

    fn record_init(&self, c: usize) -> Vec<Step> {
        let mut out = vec![];
        let n = self.len(0);
        let (u, (fi, fv, fn_)) = (&self.us[c], &self.fs[c]);
        // r = f - K u
        self.apply(&mut out, 0, u, &self.q);
        self.e
            .lin(&mut out, &self.r, &self.q, None, (0.0, NONE), (-1.0, NONE), n);
        self.e.step(
            &mut out,
            "scatter_add",
            Params {
                m: [*fn_ as u32, 0, 0, 0],
                ..Default::default()
            },
            &[fi, fv, &self.r],
            threads(*fn_),
        );
        self.e.dot(&mut out, &self.r, &self.r, None, n, RR);
        self.vcycle(&mut out, 0, &self.r, &self.z);
        self.e
            .lin(&mut out, &self.p, &self.z, None, (0.0, NONE), (1.0, NONE), n);
        self.e.dot(&mut out, &self.r, &self.z, None, n, RZ);
        out
    }

    fn record_iter(&self, c: usize) -> Vec<Step> {
        let mut out = vec![];
        let n = self.len(0);
        let u = &self.us[c];
        self.apply(&mut out, 0, &self.p, &self.q);
        self.e.dot(&mut out, &self.p, &self.q, None, n, PQ);
        self.e.scalar(&mut out, OP_DIV, RZ, PQ, ALPHA);
        self.e.lin(&mut out, u, &self.p, None, (1.0, NONE), (1.0, ALPHA), n);
        self.e
            .lin(&mut out, &self.r, &self.q, None, (1.0, NONE), (-1.0, ALPHA), n);
        self.vcycle(&mut out, 0, &self.r, &self.z);
        self.e.dot(&mut out, &self.r, &self.z, None, n, RZN);
        self.e.scalar(&mut out, OP_DIV, RZN, RZ, BETA);
        self.e.scalar(&mut out, OP_COPY, RZN, RZN, RZ);
        self.e
            .lin(&mut out, &self.p, &self.z, None, (1.0, BETA), (1.0, NONE), n);
        self.e.dot(&mut out, &self.r, &self.r, None, n, RR);
        out
    }

    /// Build the hierarchy for element stiffnesses `ee` (E_e h per fine element).
    #[cfg(test)]
    pub(super) fn set_stiffness(&self, ee: &[f64]) -> Result<()> {
        let ee32: Vec<f32> = ee.iter().map(|&v| v as f32).collect();
        self.e.write_f32(&self.ee, &ee32);
        self.build(true)
    }

    /// Build the hierarchy for the element stiffnesses already in `ee`. With `estimate` false the
    /// smoothers keep their previous λmax estimates: the Chebyshev interval reaches 1.1 λmax, so a
    /// few % of drift is harmless, and the preconditioner stays symmetric either way.
    pub(super) fn build(&self, estimate: bool) -> Result<()> {
        self.e.run(&[&self.setup])?;
        if estimate {
            self.e.run(&[&self.estimate])?;
        }
        // exact solve on the coarsest level
        let last = self.last();
        let n = self.dims[last];
        let ne: usize = n.iter().product();
        let soa = self.e.read_f32(self.mats[last].as_ref().unwrap(), SYM * ne)?;
        // upper triangles entry-major on the GPU, full element-major matrices on the CPU
        let mut mats = vec![0.0; 576 * ne];
        let mut k = 0;
        for r in 0..24 {
            for c in r..24 {
                for e in 0..ne {
                    let v = soa[k * ne + e] as f64;
                    mats[e * 576 + r * 24 + c] = v;
                    mats[e * 576 + c * 24 + r] = v;
                }
                k += 1;
            }
        }
        let inv = mg::dense_inverse(n, mats);
        let inv32: Vec<f32> = inv.iter().map(|&v| v as f32).collect();
        self.e.write_f32(&self.bottom, &inv32);
        Ok(())
    }

    /// Re-estimate the smoothers' λmax with more power steps. 12 steps can underestimate it by
    /// over 10% (seen on the 1.2M bracket's first fine-grid iteration after a coarse start, a
    /// high-contrast design): the Chebyshev smoother then amplifies the top of the spectrum, the
    /// preconditioner is indefinite, and CG stalls at a few times its tolerance.
    pub(super) fn reestimate_long(&self) -> Result<()> {
        self.e.run(&[&self.estimate_long])
    }

    /// Solve solved-case slot `c` (`LoadBasis::solved`) from its previous solution to relative residual `tol`. Returns the CG
    /// iterations (a multiple of the check interval).
    pub(super) fn solve(&self, c: usize, tol: f64, maxit: usize) -> Result<usize> {
        self.e.run(&[&self.init[c]])?;
        let rr0 = self.e.read_scalar(RR)?;
        let target = (tol * tol * self.ff[c]) as f32;
        if rr0 <= target {
            return Ok(0);
        }
        let mut its = 0;
        let batch: Vec<&[Step]> = vec![&self.iter[c]; CHECK_EVERY];
        while its < maxit {
            self.e.run(&batch)?;
            its += CHECK_EVERY;
            let rr = self.e.read_scalar(RR)?;
            if !rr.is_finite() {
                // start the next attempt from zero, not from the NaNs
                self.e.write_f32(&self.us[c], &vec![0.0; self.len(0)]);
                return Err(Diverged(rr).into());
            }
            if rr <= target {
                break;
            }
        }
        Ok(its)
    }

    /// Load case `c`'s displacement (all cases, solved or combined).
    pub(super) fn displacement(&self, c: usize) -> Result<Vec<f32>> {
        let form = self.form_steps(c);
        if !form.is_empty() {
            self.e.run(&[&form])?;
        }
        self.e.read_f32(self.case_u(c), self.len(0))
    }

    /// Scalar slot (tests): λmax estimate of level l.
    #[cfg(test)]
    fn lambda(&self, l: usize) -> Result<f32> {
        Ok(self.e.read_scalars()?[(self.lam + l as u32) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topopt::Fem;

    fn problem(n: [usize; 3]) -> Problem {
        let ne: usize = n.iter().product();
        let nn = nodes(n);
        let mut p = Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            domain: (0..ne).map(|e| e % 11 != 4).collect(),
            passive: vec![false; ne],
            fixed: vec![0; nn],
            loads: vec![vec![[0.0; 3]; nn]],
            cases: vec![1],
            extrude: None,
        };
        let [nx, ny, _] = n.map(|v| v + 1);
        for j in 0..ny {
            for i in 0..nx / 2 {
                let m = p.node(i, j, 0);
                p.fixed[m] = 7;
            }
            let m = p.node(nx - 1, j, n[2]);
            p.loads[0][m] = [0.3, -1.0, 0.5];
        }
        p
    }

    fn pseudo(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104729) % 23) as f64 / 11.0 - 1.0)
            .collect()
    }

    fn close(got: &[f32], want: &[f64], what: &str) {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max);
        assert!(err <= 2e-5 * scale, "{what}: max error {err} of {scale}");
        eprintln!("{what}: max error {:.2e} relative", err / scale);
    }

    /// Fine and level-1 operators, diagonals and transfers agree with the CPU path.
    #[test]
    fn kernels_match_cpu() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        // a 3D grid, and slabs one element thick along each axis (2D problems: `apply_p`'s other
        // tile shape)
        for n in [[7, 5, 9], [1, 9, 10], [9, 1, 10], [9, 10, 1]] {
            kernels_match_cpu_on(&gpu, n);
        }
    }

    fn kernels_match_cpu_on(gpu: &crate::gpu::Gpu, n: [usize; 3]) {
        let p = problem(n);
        let ke = super::super::hex_ke(0.3);
        let free = p.free_mask();
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: (0..p.domain.len())
                .map(|e| {
                    if p.domain[e] {
                        if e % 3 == 0 { 1e-3 } else { 1.0 + (e % 5) as f64 }
                    } else {
                        0.0
                    }
                })
                .collect(),
            free: free.clone(),
        };
        let eng = Engine::new(gpu).unwrap();
        let g = GpuMg::new(&eng, &p, &p.load_basis(), &ke, &free, &MgSettings::default(), None).unwrap();
        g.set_stiffness(&fem.ee).unwrap();
        let write = |b: &wgpu::Buffer, v: &[f64]| {
            let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            gpu.queue.write_buffer(b, 0, bytemuck::cast_slice(&v32));
        };
        let masked = |v: Vec<f64>| -> Vec<f64> {
            v.iter()
                .enumerate()
                .map(|(i, x)| if free[i / 3] >> (i % 3) & 1 == 1 { *x } else { 0.0 })
                .collect()
        };
        let n0 = g.len(0);
        let u = masked(pseudo(n0, 1));

        // fine diagonal (left in CG's p by the λmax estimate), then the mat-vec
        let inv = fem.inv_diag();
        let d: Vec<f64> = inv.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();
        close(&g.e.read_f32(&g.diag[0], n0).unwrap(), &d, "fine diagonal");
        write(&g.p, &u);
        let mut prog = vec![];
        g.apply(&mut prog, 0, &g.p, &g.q);
        g.e.run(&[&prog]).unwrap();
        let mut want = vec![0.0; n0];
        fem.matvec(&u, &mut want);
        close(&g.e.read_f32(&g.q, n0).unwrap(), &want, "fine K u");

        // the fused smoother step: d = 0.5 d + 2 D⁻¹ (b − K u), and the plain residual
        let b = masked(pseudo(n0, 5));
        let d0 = masked(pseudo(n0, 6));
        write(&g.r, &b);
        write(&g.z, &u);
        write(&g.q, &d0);
        let mut prog = vec![];
        g.cheb_fine(&mut prog, &g.r, &g.z, &g.q, 0.5, (2.0, NONE), false, true);
        g.e.run(&[&prog]).unwrap();
        let mut ku = vec![0.0; n0];
        fem.matvec(&u, &mut ku);
        let want: Vec<f64> = (0..n0).map(|i| 0.5 * d0[i] + 2.0 * (b[i] - ku[i]) * inv[i]).collect();
        close(&g.e.read_f32(&g.q, n0).unwrap(), &want, "fused smoother step");
        let mut prog = vec![];
        g.cheb_fine(&mut prog, &g.r, &g.z, &g.q, 0.0, (1.0, NONE), false, false);
        g.e.run(&[&prog]).unwrap();
        let want: Vec<f64> = (0..n0).map(|i| b[i] - ku[i]).collect();
        close(&g.e.read_f32(&g.q, n0).unwrap(), &want, "fused residual");

        // level 1: Pᵀ K P against the CPU's assembled level
        let c1 = mg::tests_api::level1(&fem);
        let n1 = g.len(1);
        let v = pseudo(n1, 2);
        write(g.lb[1].as_ref().unwrap(), &v);
        let mut prog = vec![];
        g.apply(&mut prog, 1, g.lb[1].as_ref().unwrap(), g.lx[1].as_ref().unwrap());
        g.e.run(&[&prog]).unwrap();
        close(
            &g.e.read_f32(g.lx[1].as_ref().unwrap(), n1).unwrap(),
            &c1.0(&v),
            "level-1 K u",
        );
        close(&g.e.read_f32(&g.diag[1], n1).unwrap(), &c1.1, "level-1 diagonal");

        // level 2, assembled on the GPU
        let c2 = mg::tests_api::level2(&fem);
        let n2 = g.len(2);
        let v = pseudo(n2, 3);
        write(g.lb[2].as_ref().unwrap(), &v);
        let mut prog = vec![];
        g.apply(&mut prog, 2, g.lb[2].as_ref().unwrap(), g.lx[2].as_ref().unwrap());
        g.e.run(&[&prog]).unwrap();
        close(
            &g.e.read_f32(g.lx[2].as_ref().unwrap(), n2).unwrap(),
            &c2(&v),
            "level-2 K u",
        );

        // transfers
        write(&g.lr[0], &u);
        let mut prog = vec![];
        g.e.step(
            &mut prog,
            "restrict",
            Params {
                n: dims4(g.dims[0]),
                m: [0, 1, 0, 0],
                ..Default::default()
            },
            &[&g.lr[0], &g.free, g.lb[1].as_ref().unwrap()],
            threads(nodes(g.dims[1])),
        );
        g.e.run(&[&prog]).unwrap();
        let mut want = vec![0.0; n1];
        mg::tests_api::restrict(p.n, &u, &mut want, &inv);
        close(&g.e.read_f32(g.lb[1].as_ref().unwrap(), n1).unwrap(), &want, "restrict");
        let v = pseudo(n1, 4);
        write(g.lx[1].as_ref().unwrap(), &v);
        write(&g.z, &u);
        let mut prog = vec![];
        g.e.step(
            &mut prog,
            "prolong",
            Params {
                n: dims4(g.dims[0]),
                m: [0, 1, 0, 0],
                ..Default::default()
            },
            &[g.lx[1].as_ref().unwrap(), &g.free, &g.z],
            threads(nodes(g.dims[0])),
        );
        g.e.run(&[&prog]).unwrap();
        let mut want = u.clone();
        mg::tests_api::prolong_add(p.n, &v, &mut want, &inv);
        close(&g.e.read_f32(&g.z, n0).unwrap(), &want, "prolong");

        // λmax estimates are positive and of the CPU's size
        for l in 0..g.last() {
            let lam = g.lambda(l).unwrap();
            assert!(lam > 0.5 && lam < 10.0, "level {l}: λmax {lam}");
        }
    }

    /// The GPU solve reaches the CPU solution, at a realistic scale: E·h in the thousands, small
    /// nodal loads, a long lever arm (large rigid-body-like displacements), and 1e-9 contrast.
    /// Before the element operator worked on relative displacements, f32 rounding of ke made this
    /// cantilever 0.35% too soft.
    #[test]
    fn gpu_solve_matches_cpu() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let n = [40, 4, 6];
        let mut p = problem(n);
        // clamp the x = 0 face only, load the far end lightly
        p.fixed.fill(0);
        p.loads[0].iter_mut().for_each(|l| *l = [0.0; 3]);
        let [nx, ny, nz] = n.map(|v| v + 1);
        for k in 0..nz {
            for j in 0..ny {
                let m = p.node(0, j, k);
                p.fixed[m] = 7;
                let m = p.node(nx - 1, j, k);
                p.loads[0][m] = [0.0, 0.0, -0.01];
            }
        }
        p.domain.fill(true);
        let ke = super::super::hex_ke(0.35);
        let free = p.free_mask();
        let fem = Fem {
            p: &p,
            ke: &ke,
            // a hollow beam: solid skin, near-void core
            ee: (0..p.domain.len())
                .map(|e| {
                    let (j, k) = ((e / n[0]) % n[1], e / (n[0] * n[1]));
                    let skin = j == 0 || j == n[1] - 1 || k == 0 || k == n[2] - 1;
                    4550.0 * if skin { 1.0 } else { 1e-9 }
                })
                .collect(),
            free: free.clone(),
        };
        let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
        let mut m = mg::Multigrid::new(&fem, &MgSettings::default());
        let mut u = vec![0.0; f.len()];
        fem.solve(&f, &mut u, 1e-10, 1000, &mut |r, z| m.precondition(r, z));
        let eng = Engine::new(&gpu).unwrap();
        let g = GpuMg::new(&eng, &p, &p.load_basis(), &ke, &free, &MgSettings::default(), None).unwrap();
        g.set_stiffness(&fem.ee).unwrap();
        let its = g.solve(0, 1e-5, 1000).unwrap();
        let ug = g.displacement(0).unwrap();
        let fu = |u: &dyn Fn(usize) -> f64| (0..f.len()).map(|i| f[i] * u(i)).sum::<f64>();
        let (c_cpu, c_gpu) = (fu(&|i| u[i]), fu(&|i| ug[i] as f64));
        eprintln!("{its} GPU CG iterations; compliance {c_gpu:.6e} vs {c_cpu:.6e}");
        assert!((c_gpu - c_cpu).abs() < 1e-4 * c_cpu, "compliance {c_gpu} vs {c_cpu}");
        close(&ug, &u, "GPU u");
    }

    /// Rollers on the GPU: the exact solution of `block_on_rollers`.
    #[test]
    fn gpu_block_on_rollers() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let (p, force, nu, e0) = crate::topopt::tests::block_on_rollers();
        let ke = super::super::hex_ke(nu);
        let free = p.free_mask();
        let eng = Engine::new(&gpu).unwrap();
        let g = GpuMg::new(&eng, &p, &p.load_basis(), &ke, &free, &MgSettings::default(), None).unwrap();
        g.set_stiffness(&vec![e0 * p.h; p.domain.len()]).unwrap();
        let its = g.solve(0, 1e-6, 1000).unwrap();
        let u: Vec<f64> = g.displacement(0).unwrap().iter().map(|&v| v as f64).collect();
        crate::topopt::tests::check_block_on_rollers(&p, &u, force, nu, e0, 1e-3);
        assert!(its < 200, "{its} GPU CG iterations");
    }

    /// Timings of the solver's pieces on a 9.6M-element box (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn profile_kernels() {
        let gpu = crate::gpu::headless().unwrap();
        let n = [200, 120, 400];
        let mut p = problem(n);
        p.domain.fill(true);
        let ke = super::super::hex_ke(0.3);
        let free = p.free_mask();
        let ee: Vec<f64> = (0..p.domain.len())
            .map(|e| if e % 3 == 0 { 1e-6 } else { 1.0 })
            .collect();
        let eng = Engine::new(&gpu).unwrap();
        let g = GpuMg::new(&eng, &p, &p.load_basis(), &ke, &free, &MgSettings::default(), None).unwrap();
        let t = std::time::Instant::now();
        g.set_stiffness(&ee).unwrap();
        eprintln!("levels {:?}", g.dims);
        eprintln!(
            "setup (incl. upload, coarsest inverse): {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        let time = |name: &str, prog: &[Step]| {
            let reps = 10;
            let batch: Vec<&[Step]> = vec![prog; reps];
            g.e.run(&batch).unwrap();
            g.e.read_scalars().unwrap();
            let t = std::time::Instant::now();
            g.e.run(&batch).unwrap();
            g.e.read_scalars().unwrap();
            eprintln!(
                "{name:>24}: {:7.2} ms ({} dispatches)",
                t.elapsed().as_secs_f64() * 1e3 / reps as f64,
                prog.len()
            );
        };
        time("setup program", &g.setup);
        for l in 0..g.levels() - 1 {
            let mut prog = vec![];
            g.apply(&mut prog, l, &g.lr[l], &g.ld[l]);
            time(&format!("apply level {l}"), &prog);
        }
        let n0 = g.len(0);
        let mut prog = vec![];
        g.e.lin(&mut prog, &g.p, &g.z, None, (1.0, NONE), (1.0, NONE), n0);
        time("lin (fine)", &prog);
        let mut prog = vec![];
        g.e.dot(&mut prog, &g.p, &g.z, None, n0, 20);
        time("dot (fine)", &prog);
        for l in 0..g.levels() - 1 {
            let mut prog = vec![];
            g.smooth(
                &mut prog,
                l,
                if l == 0 { &g.r } else { g.lb[l].as_ref().unwrap() },
                if l == 0 { &g.z } else { g.lx[l].as_ref().unwrap() },
                false,
            );
            time(&format!("smooth level {l}"), &prog);
        }
        let mut prog = vec![];
        g.vcycle(&mut prog, 0, &g.r, &g.z);
        time("V-cycle", &prog);
        time("CG iteration", &g.iter[0]);
    }
}
