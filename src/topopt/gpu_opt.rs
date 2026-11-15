//! The optimizer loop with everything on the GPU: element stiffnesses, the MGCG solve (`gpu_mg`),
//! energies, sensitivities, density filter, projection and the OC update. Per iteration only
//! scalars come back (compliance, volume during the λ search, change, grey share, max
//! displacement), plus the fields when the .vti is written. Mirrors `optimize` in mod.rs.
//!
//! Element arrays and vectors are block-sparse (bricks.rs): only the bricks around the domain are
//! stored, and readback is scattered to the dense grid.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use super::bricks::{self, Bricks};
use super::gpu_engine::{Engine, NONE, Params, Reduce, Step, dims4, threads};
use super::gpu_mg::GpuMg;
use super::mg::MgSettings;
use super::output::{self, Output, Snapshot};
use super::symmetry::Model;
use super::{CG_MAXIT, IterStats, LoadBasis, LoopState, Next, Outcome, Settings, Stage};
use crate::gpu::Gpu;

/// Re-estimate the smoothers' λmax at least every this many optimizer iterations.
const REESTIMATE_EVERY: usize = 5;
/// CG iterations after which an unconverged solve refreshes the smoothers' λmax estimate (a long
/// re-estimate if it was fresh) and carries on (MGCG usually needs 4–16).
const STALE_CG_ITS: usize = 32;

// scalar slots (gpu_mg uses those below 80)
const COMPL: u32 = 96;
const LAMBDA: u32 = 97;
const VOL: u32 = 98;
const CHANGE: u32 = 99;
const GREY: u32 = 100;
const BETA: u32 = 101;
/// max displacement of load case c on mirror side r (index into `Model::regions`) at
/// MAXD + c · regions + r
const MAXD: u32 = 104;
const MAX_CASES: usize = 24;

/// Element arrays on the GPU.
struct Fields {
    flags: wgpu::Buffer,
    x: wgpu::Buffer,
    xnew: wgpu::Buffer,
    xtilde: wgpu::Buffer,
    xphys: wgpu::Buffer,
    /// the densities the last solve used (for output)
    xused: wgpu::Buffer,
    /// the sensitivities before filtering (dc's shares `xnew`, which is free until the OC step)
    dvs: wgpu::Buffer,
    dc: wgpu::Buffer,
    dv: wgpu::Buffer,
    /// the AM filter's (overhang control)
    am: Option<AmFields>,
}

/// The AM filter's arrays (see `am_fwd`, `am_adj` in opt.wgsl).
struct AmFields {
    /// printed densities x̂ (the projection's input)
    xhat: wgpu::Buffer,
    /// per element its supports' largest ξ (m) and Σ (ξ / m)^P (r)
    mr: wgpu::Buffer,
    /// the adjoint's carries to the layer below, compliance and volume
    cc: wgpu::Buffer,
    cv: wgpu::Buffer,
}

pub(super) fn optimize_gpu(
    m: &Model,
    bases: &[LoadBasis],
    s: &Settings,
    ms: &MgSettings,
    out: &Path,
    gpu: &Gpu,
    stage: &Stage,
) -> Result<Outcome> {
    let p = m.p();
    let ke = super::hex_ke(s.nu);
    let ne = p.domain.len();
    let ndesign = (0..ne).filter(|&e| p.domain[e] && !p.passive[e]).count();
    if ndesign == 0 {
        bail!("no design elements: the domain is empty or entirely kept solid");
    }
    let ncases = m.cases();
    let regions = m.regions();
    if ncases > MAX_CASES {
        bail!("at most {MAX_CASES} load cases on the GPU");
    }
    assert!(m.groups.len() <= super::gpu_mg::MAX_GROUPS);
    let e = Engine::new(gpu)?;
    let elements = ne as f64 / 1e6;
    // the bricks holding the domain's elements and nodes; `nes` elements (and nodes) stored
    let bricks = Bricks::from_elements(p.n, bricks::shape_for(p.n), &p.domain);
    let nes = bricks.len();
    let v = bricks.variant();
    let bk = |kernel: &str| format!("{kernel}{v}");
    let oom = |err: anyhow::Error| {
        anyhow::anyhow!("{err}: {elements:.1}M elements don't fit on this GPU; use a larger --voxel")
    };
    // one hierarchy per group (its own supports); the others share the first one's element and
    // scratch buffers
    let mgs = gpu
        .alloc_scoped(|| -> Result<Vec<GpuMg>> {
            let mut out: Vec<GpuMg> = vec![];
            for (g, group) in m.groups.iter().enumerate() {
                let free = group.free_mask();
                let made = GpuMg::new(
                    &e,
                    group,
                    &bases[g],
                    &ke,
                    &free,
                    ms,
                    out.first().map(|o| (o, g)),
                    &bricks,
                )?;
                out.push(made);
            }
            Ok(out)
        })
        .map_err(oom)??;
    let mg = &mgs[0];
    let c = mg.coarsest();
    println!(
        "GPU multigrid: {} levels, coarsest {}x{}x{} elements, {:?}",
        mg.levels(),
        c[0],
        c[1],
        c[2],
        ms.smoother
    );

    // element arrays
    let flags: Vec<u32> = (0..ne)
        .map(|i| {
            let (d, k) = (p.domain[i], p.passive[i]);
            d as u32 | (k as u32) << 1 | ((d && !k) as u32) << 2
        })
        .collect();
    let flags = bricks.gather(&flags, p.n, 1, 0);
    let x0: Vec<f32> = (0..ne)
        .map(|i| {
            if let Some((x, _)) = &stage.start {
                x[i] as f32
            } else if p.passive[i] {
                1.0
            } else if p.domain[i] {
                s.volfrac as f32
            } else {
                0.0
            }
        })
        .collect();
    let x0 = bricks.gather(&x0, p.n, 1, 0.0);
    let ne = nes;
    let (page, coords) = mg.fine_tables();
    let f = gpu
        .alloc_scoped(|| Fields {
            flags: e.init("flags", bytemuck::cast_slice(&flags)),
            x: e.init("x", bytemuck::cast_slice(&x0)),
            xnew: e.buf("xnew", ne),
            xtilde: e.buf("xtilde", ne),
            xphys: e.buf("xphys", ne),
            xused: e.buf("xused", ne),
            dvs: e.buf("dvs", ne),
            dc: e.buf("dc", ne),
            dv: e.buf("dv", ne),
            am: s.overhang.then(|| AmFields {
                xhat: e.buf("xhat", ne),
                mr: e.buf("am m, r", 2 * ne),
                cc: e.buf("am cc", ne),
                cv: e.buf("am cv", ne),
            }),
        })
        .map_err(oom)?;
    drop((flags, x0));
    let ndomain = p.domain.iter().filter(|&&d| d).count();
    println!(
        "GPU memory: {:.0} MB ({:.0} B per domain element; {} of {} bricks stored, {:.0}% of their elements in the domain)",
        e.allocated() as f64 / 1e6,
        e.allocated() as f64 / ndomain as f64,
        bricks.slots(),
        bricks.page.len(),
        100.0 * ndomain as f64 / nes as f64
    );
    let emin = s.e0 * s.emin;
    let elem = |out: &mut Vec<Step>, kernel: &str, pf: [f32; 4], m: [u32; 4], bufs: &[&wgpu::Buffer]| {
        e.step(
            out,
            kernel,
            Params {
                n: dims4(p.n),
                m: [ne as u32, m[1], m[2], m[3]],
                f: pf,
                g: [0.0, 0.5, s.rmin as f32, 0.0], // η = 0.5, rmin
            },
            bufs,
            threads(ne),
        )
    };
    let material = [emin as f32, s.e0 as f32, s.penal as f32, p.h as f32];
    let rmin = [s.rmin as f32, 0.0, 0.0, 0.0];
    // filter: mode 0 raw sums, 1 densities
    // the mirror planes (at the grid's low faces) reflect the filter
    let mirror = (0..3).filter(|&a| m.mirror_axes()[a]).fold(0u32, |b, a| b | 1 << a);
    let filter = |out: &mut Vec<Step>, mode: u32, inp: &wgpu::Buffer, outp: &wgpu::Buffer| {
        elem(
            out,
            &bk("filter"),
            rmin,
            [0, mode, mirror, 0],
            &[&f.flags, inp, outp, page, coords],
        )
    };
    // the AM filter, one layer per step: forward from the plate up, the chain rule from the top down.
    // n.w: the build axis, | 4 when built towards its negative side
    let am = s.overhang.then(|| super::overhang::Am::new(p, m.mirror_axes(), s.up));
    let [l0, l1] = s.up.lateral();
    let layer = p.n[l0] * p.n[l1];
    let layers = p.n[s.up.axis()];
    let am_params = |l: usize| Params {
        n: {
            let mut n = dims4(p.n);
            n[3] = s.up.axis() as u32 | (s.up.negative() as u32) << 2;
            n
        },
        m: [layer as u32, l as u32, mirror, am.as_ref().map_or(0, |a| a.base as u32)],
        f: {
            let q = am.as_ref().map_or(1.0, |a| a.q);
            let pw = super::overhang::P;
            [
                pw as f32,
                (1.0 / q) as f32,
                super::overhang::EPS as f32,
                (pw / q) as f32,
            ]
        },
        g: [0.0, 0.5, s.rmin as f32, 0.0],
    };
    // the projection's input: x̂ with the AM filter, else x̃
    let xhat = f.am.as_ref().map_or(&f.xtilde, |a| &a.xhat);
    // xin -> xtilde (-> xhat) -> xphys
    let densities = |out: &mut Vec<Step>, xin: &wgpu::Buffer| {
        filter(out, 1, xin, &f.xtilde);
        if let Some(a) = &f.am {
            for l in 0..layers {
                e.step(
                    out,
                    &bk("am_fwd"),
                    am_params(l),
                    &[&f.flags, &f.xtilde, &a.xhat, &a.mr, page],
                    threads(layer),
                );
            }
        }
        elem(
            out,
            "project",
            [0.0; 4],
            [0, BETA, 0, 0],
            &[&f.flags, xhat, &e.scalars, &f.xphys],
        );
    };

    let mut prog_init = vec![];
    densities(&mut prog_init, &f.x);
    let mut prog_stiffness = vec![];
    elem(
        &mut prog_stiffness,
        "stiffness",
        material,
        [0; 4],
        &[&f.flags, &f.xphys, &mg.ee],
    );
    e.lin(
        &mut prog_stiffness,
        &f.xused,
        &f.xphys,
        None,
        (0.0, NONE),
        (1.0, NONE),
        ne,
    );
    let mut prog_sens = vec![];
    e.dot(&mut prog_sens, &mg.ee, &mg.ce, None, ne, COMPL);
    elem(
        &mut prog_sens,
        &bk("sens"),
        material,
        [0, BETA, mirror, f.am.is_some() as u32],
        &[
            &f.flags, &f.xphys, xhat, &mg.ce, &e.scalars, &f.xnew, &f.dvs, page, coords,
        ],
    );
    if let Some(a) = &f.am {
        for l in (0..layers).rev() {
            e.step(
                &mut prog_sens,
                &bk("am_adj"),
                am_params(l),
                &[
                    &f.flags, &f.xtilde, &a.xhat, &a.mr, &f.xnew, &f.dvs, &a.cc, &a.cv, page, coords,
                ],
                threads(layer),
            );
        }
    }
    filter(&mut prog_sens, 0, &f.xnew, &f.dc);
    filter(&mut prog_sens, 0, &f.dvs, &f.dv);
    e.reduce_design(&mut prog_sens, Reduce::Grey, &f.xphys, &f.xphys, &f.flags, ne, GREY);
    let mut prog_oc = vec![];
    elem(
        &mut prog_oc,
        "oc",
        // move limit; the whole part's compliance per the model's (OC divides by the former, so
        // λ is the same with or without symmetry)
        [0.2, m.scale() as f32, 0.0, 0.0],
        [0, LAMBDA, COMPL, 0],
        &[&f.flags, &f.x, &f.dc, &f.dv, &e.scalars, &f.xnew],
    );
    densities(&mut prog_oc, &f.xnew);
    e.reduce_design(&mut prog_oc, Reduce::Dot, &f.xphys, &f.xphys, &f.flags, ne, VOL);
    let mut prog_accept = vec![];
    e.reduce_design(&mut prog_accept, Reduce::MaxDiff, &f.xnew, &f.x, &f.flags, ne, CHANGE);
    e.lin(&mut prog_accept, &f.x, &f.xnew, None, (0.0, NONE), (1.0, NONE), ne);
    // energies of every group's part of every case (summed into `ce`), and each case's largest
    // displacement on every mirror side (the groups' parts with their signs there)
    // stored nodes
    let nn = nes;
    let ur = (mgs.len() > 1).then(|| e.buf("u, one mirror side", 3 * nn));
    let mut prog_energy = vec![];
    for c in 0..ncases {
        for (g, mg) in mgs.iter().enumerate() {
            mg.record_case(&mut prog_energy, c, g > 0 || c > 0);
        }
        for (ri, &r) in regions.iter().enumerate() {
            let slot = MAXD + (c * regions.len() + ri) as u32;
            let u = match &ur {
                None => mgs[0].case_u(c),
                Some(ur) => {
                    for (g, (mg, sign)) in mgs.iter().zip(m.signs(r)).enumerate() {
                        let a = if g == 0 { 0.0 } else { 1.0 };
                        e.lin(
                            &mut prog_energy,
                            ur,
                            mg.case_u(c),
                            None,
                            (a, NONE),
                            (sign as f32, NONE),
                            3 * nn,
                        );
                    }
                    ur
                }
            };
            e.reduce(&mut prog_energy, Reduce::MaxNorm, u, u, u, nn, slot);
        }
    }
    let mut prog_project = vec![];
    elem(
        &mut prog_project,
        "project",
        [0.0; 4],
        [0, BETA, 0, 0],
        &[&f.flags, xhat, &e.scalars, &f.xphys],
    );

    let mut beta = match &stage.start {
        Some((_, b)) => *b,
        None if s.beta_max > 0.0 => 1.0,
        None => 0.0,
    };
    e.set_scalar(BETA, beta as f32);
    e.run(&[&prog_init])?;

    println!(
        "SIMP: volume fraction {}, penalty {}, filter radius {} voxels, E = {} MPa, nu = {}, {} load case{}, {}",
        s.volfrac,
        s.penal,
        s.rmin,
        s.e0,
        s.nu,
        ncases,
        if ncases == 1 { "" } else { "s (summed compliance)" },
        if s.beta_max > 0.0 {
            format!("Heaviside projection up to beta {}", s.beta_max)
        } else {
            "no projection".into()
        }
    );
    let t_start = Instant::now();
    let mut control = stage.state.control.clone();
    let mut output = Output::new(m, out, s.up);
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
    // the last iteration whose fields are not written yet
    let mut pending = false;
    let mut lambda = stage.state.lambda;
    let mut last_change = stage.state.last_change;
    let target = s.volfrac * ndesign as f64;
    // per group: optimizer iterations since the smoothers' λmax was estimated (None: estimate next
    // time)
    let mut estimate_age: Vec<Option<usize>> = vec![None; mgs.len()];
    // an element array, on the dense grid
    let read_dense = |b: &wgpu::Buffer| -> Result<Vec<f32>> { Ok(bricks.scatter(&e.read_f32(b, ne)?, p.n, 1, 0.0)) };
    let snapshot = || -> Result<Snapshot> {
        let ee = read_dense(&mg.ee)?;
        let ce = read_dense(&mg.ce)?;
        let h3 = (p.h * p.h * p.h) as f32;
        Ok(Snapshot {
            density: read_dense(&f.xused)?,
            // x̂ (x̃) of the same densities during the run (the OC step hasn't replaced it yet); after
            // the loop it is one OC update newer than `xused`, a small difference
            filtered: output::filtered_out(p, read_dense(xhat)?.into_iter()),
            energy: ee.iter().zip(&ce).map(|(a, b)| 0.5 * a * b / h3).collect(),
            // per group, per case (the output combines them)
            disps: mgs
                .iter()
                .flat_map(|mg| (0..ncases).map(move |c| mg.displacement(c)))
                .collect::<Result<_>>()?,
        })
    };
    for it in 1..=s.iters {
        let t0 = Instant::now();
        let mut st = IterStats::default();
        e.run(&[&prog_stiffness])?;
        for (mg, age) in mgs.iter().zip(&mut estimate_age) {
            let estimate = age.is_none_or(|a| a >= REESTIMATE_EVERY);
            mg.build(estimate)?;
            *age = Some(if estimate { 0 } else { age.unwrap() + 1 });
        }
        st.setup = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let final_stage = beta > 0.0 && beta >= s.beta_max;
        let tol = if !s.adaptive_tol || final_stage {
            s.cg_tol
        } else {
            (0.05 * last_change).clamp(s.cg_tol, 1e-2)
        };
        st.cg_tol = tol;
        let mut rebuilt = 0.0;
        for (mg, age) in mgs.iter().zip(&mut estimate_age) {
            for c in 0..mg.us.len() {
                let mut n = match mg.solve(c, tol, STALE_CG_ITS) {
                    // seen on a 2D slab's first fine iteration after a coarse start: as a stall,
                    // but from zero (a second breakdown is an error)
                    Err(e) if e.is::<super::gpu_mg::Diverged>() => STALE_CG_ITS,
                    r => r?,
                };
                if n >= STALE_CG_ITS {
                    // not converged yet (or broke down): an old λmax estimate may have weakened
                    // the smoother, or a fresh one come out too low; refresh it (carefully if it
                    // was fresh) and carry on from where CG got to
                    let t = Instant::now();
                    if *age == Some(0) {
                        mg.reestimate_long()?;
                    } else {
                        mg.build(true)?;
                    }
                    *age = Some(0);
                    rebuilt += t.elapsed().as_secs_f64();
                    n += mg.solve(c, tol, CG_MAXIT)?;
                }
                if n >= CG_MAXIT {
                    println!("warning: CG did not converge in {CG_MAXIT} iterations");
                }
                st.cg.push(n);
            }
        }
        e.run(&[&prog_energy])?;
        st.setup += rebuilt;
        st.solve = t1.elapsed().as_secs_f64() - rebuilt;

        let t2 = Instant::now();
        e.run(&[&prog_sens])?;
        let sc = e.read_scalars()?;
        // for the whole part (the reduced model's is 1/2^planes of it)
        let compliance = sc[COMPL as usize] as f64 * m.scale();
        let grey = sc[GREY as usize] as f64 / ndesign as f64;
        let max_disp = (0..ncases * regions.len())
            .map(|i| sc[MAXD as usize + i] as f64)
            .fold(0.0, f64::max);
        st.compliance = compliance;
        st.update = t2.elapsed().as_secs_f64();

        // the fields of this solve, before the update replaces the densities
        let t3 = Instant::now();
        if output.due()? {
            if output.live_is_full() {
                output.live(snapshot()?);
                pending = false;
            } else {
                // a preview: only the densities come back
                output.live(Snapshot {
                    density: read_dense(&f.xused)?,
                    filtered: vec![],
                    energy: vec![],
                    disps: vec![],
                });
                pending = true;
            }
        } else {
            pending = true;
        }
        st.io = t3.elapsed().as_secs_f64();

        // optimality criteria: as on the CPU, bracket λ around the previous value and bisect
        let t4 = Instant::now();
        let mut volume_at = |lam: f64| -> Result<f64> {
            st.oc_steps += 1;
            e.set_scalar(LAMBDA, lam as f32);
            e.run(&[&prog_oc])?;
            Ok(e.read_scalars()?[VOL as usize] as f64)
        };
        let (mut l1, mut l2) = (lambda / 2.0, lambda * 2.0);
        for _ in 0..60 {
            if volume_at(l1)? > target {
                break;
            }
            l2 = l1;
            l1 /= 8.0;
        }
        for _ in 0..60 {
            if volume_at(l2)? <= target {
                break;
            }
            l1 = l2;
            l2 *= 8.0;
        }
        while (l2 - l1) / (l1 + l2) > 1e-4 {
            let lmid = 0.5 * (l1 + l2);
            if volume_at(lmid)? > target {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
        lambda = 0.5 * (l1 + l2);
        volume_at(lambda)?;
        e.run(&[&prog_accept])?;
        let change = e.read_scalars()?[CHANGE as usize] as f64;
        last_change = change;
        st.update += t4.elapsed().as_secs_f64();

        super::print_iteration(stage.it0 + it, compliance, max_disp, change, grey, beta, &st, t0);
        result.iterations = it;
        result.compliance = compliance;
        result.max_disp = max_disp;
        result.stats.push(st);
        match control.next(s, beta, compliance, change) {
            Next::Go => {}
            Next::Beta(b) => {
                beta = b;
                // a sharper projection changes the densities at once
                estimate_age.fill(None);
                e.set_scalar(BETA, beta as f32);
                e.run(&[&prog_project])?;
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
    if !m.mirror_axes()[s.up.axis()] {
        let am = super::overhang::Am::new(p, m.mirror_axes(), s.up);
        let xphys: Vec<f64> = read_dense(&f.xphys)?.into_iter().map(f64::from).collect();
        am.report(p, &xphys);
    }
    result.beta = beta;
    result.state = LoopState {
        control,
        lambda,
        last_change,
    };
    if pending {
        output.finish(snapshot()?)?;
    } else {
        output.wait()?;
    }
    result.design = read_dense(&f.x)?.into_iter().map(f64::from).collect();
    if !stage.quiet {
        super::print_done(&result, t_start, out);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::{Ground, Problem, Settings, Solver, optimize};
    use super::*;

    /// A cantilever: clamped at x = 0, a spread load down at the far top edge, a kept-solid strip.
    /// The third load case is a combination of the first two (as a rocked load is), so it isn't
    /// solved but formed from their displacements.
    fn cantilever() -> Problem {
        let n = [24, 6, 10];
        let ne: usize = n.iter().product();
        let nn: usize = n.iter().map(|v| v + 1).product();
        let mut p = Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            domain: vec![true; ne],
            passive: (0..ne).map(|e| e % n[0] == n[0] - 1).collect(),
            fixed: vec![0; nn],
            loads: vec![vec![[0.0; 3]; nn]; 3],
            cases: vec![1, 2, 3],
            extrude: None,
        };
        let [nx, ny, nz] = n.map(|v| v + 1);
        for k in 0..nz {
            for j in 0..ny {
                let m = p.node(0, j, k);
                p.fixed[m] = 7;
            }
        }
        for j in 0..ny {
            let m = p.node(nx - 1, j, nz - 1);
            p.loads[0][m] = [0.0, 0.0, -10.0 / ny as f64];
            let m = p.node(nx - 1, j, nz / 2);
            p.loads[1][m] = [0.0, 3.0 / ny as f64, 0.0];
        }
        p.loads[2] = (0..nn)
            .map(|m| std::array::from_fn(|c| p.loads[0][m][c] - 0.5 * p.loads[1][m][c]))
            .collect();
        p
    }

    /// On the GPU too, a quarter of the symmetric beam (four load patterns, so four hierarchies
    /// sharing their buffers) gives the same run as the whole beam.
    #[test]
    fn symmetric_quarter_matches_the_whole_gpu() {
        use super::super::symmetry::{self, Model};
        use super::super::tests::{beam_settings, compare_runs, symmetric_beam};
        use super::super::{Stage, optimize_stage};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        if gpu.is_software() {
            eprintln!("software GPU; skipped");
            return;
        }
        let full = symmetric_beam();
        let m = symmetry::reduce(&full, [true, true, false]).unwrap();
        let dir = std::env::temp_dir().join(format!("gen3d-symgpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = Settings {
            cg_tol: 1e-6,
            ..beam_settings(Solver::Gpu(MgSettings::default()))
        };
        let st = Stage::default();
        let whole = optimize_stage(&Model::single(full), &s, &dir.join("whole.vti"), Some(&gpu), &st).unwrap();
        let quarter = optimize_stage(&m, &s, &dir.join("quarter.vti"), Some(&gpu), &st).unwrap();
        compare_runs(&whole, &quarter, &dir, 2e-3);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The GPU loop follows the CPU loop: same compliance, iteration by iteration.
    #[test]
    fn gpu_loop_matches_cpu_loop() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        if gpu.is_software() {
            eprintln!("software GPU; skipped");
            return;
        }
        let p = cantilever();
        let dir = std::env::temp_dir().join(format!("gen3d-gpuloop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use crate::mesh::Up;
        let runs = [
            (0.0, false, Up::PlusZ),
            (8.0, false, Up::PlusZ),
            (0.0, true, Up::PlusZ),
            (8.0, true, Up::PlusZ),
            (8.0, true, Up::MinusX),
            (0.0, true, Up::PlusY),
            (8.0, true, Up::MinusZ),
        ];
        for (beta_max, overhang, up) in runs {
            let settings = |solver| Settings {
                voxel: 1.0,
                volfrac: 0.3,
                penal: 3.0,
                rmin: 1.5,
                iters: 12,
                tol: 0.0,
                e0: 3500.0,
                nu: 0.35,
                force: 0.0,
                beta_max,
                solver,
                emin: 1e-9,
                cg_tol: 1e-6,
                adaptive_tol: false,
                ground: Ground::None,
                coarse_start: false,
                symmetry: true,
                overhang,
                up,
                plane2d: None,
            };
            let ms = MgSettings::default();
            let cpu = optimize(&p, &settings(Solver::Multigrid(ms)), &dir.join("cpu.vti"), None).unwrap();
            let gpu_r = optimize(&p, &settings(Solver::Gpu(ms)), &dir.join("gpu.vti"), Some(&gpu)).unwrap();
            assert_eq!(cpu.iterations, gpu_r.iterations);
            // the overhang runs amplify f32 rounding over 12 iterations: -X up with beta 8 ends
            // 1.7e-3 to 2.1e-3 apart depending on how the smoother rounds (per-iteration
            // compliances stay within 1.3e-3)
            let rel = (cpu.max_disp - gpu_r.max_disp).abs() / cpu.max_disp;
            assert!(
                rel < 3e-3,
                "max displacement {} (CPU) vs {} (GPU)",
                cpu.max_disp,
                gpu_r.max_disp
            );
            for (it, (a, b)) in cpu.stats.iter().zip(&gpu_r.stats).enumerate() {
                let rel = (a.compliance - b.compliance).abs() / a.compliance;
                assert!(
                    rel < 2e-3,
                    "beta_max {beta_max} overhang {overhang} {up:?}, iteration {}: compliance {} (CPU) vs {} (GPU)",
                    it + 1,
                    a.compliance,
                    b.compliance
                );
            }
            // the written fields agree too
            let read = |f: &str| crate::vti::read(&dir.join(f)).unwrap();
            let (c, g) = (read("cpu.vti"), read("gpu.vti"));
            let dens = |v: &crate::vti::VtiFile| v.arrays.iter().find(|a| a.name == "density").unwrap().data.clone();
            let (dc, dg) = (dens(&c), dens(&g));
            let diff = dc.iter().zip(&dg).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(
                diff < 0.05,
                "beta_max {beta_max} overhang {overhang} {up:?}: density differs by {diff}"
            );
            // and the combined case's displacement
            let disp = |v: &crate::vti::VtiFile| {
                v.arrays
                    .iter()
                    .find(|a| a.name == "displacement_3")
                    .map(|a| a.data.clone())
            };
            let (a, b) = (disp(&c).unwrap(), disp(&g).unwrap());
            let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(
                diff < 1e-2 * scale,
                "beta_max {beta_max}: case 3 displacement differs by {diff}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
