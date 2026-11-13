//! The optimizer loop with everything on the GPU: element stiffnesses, the MGCG solve (`gpu_mg`),
//! energies, sensitivities, density filter, projection and the OC update. Per iteration only
//! scalars come back (compliance, volume during the λ search, change, grey share, max
//! displacement), plus the fields when the .vti is written. Mirrors `optimize` in mod.rs.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use super::gpu_engine::{Engine, NONE, Params, Reduce, Step, dims4, threads};
use super::gpu_mg::GpuMg;
use super::mg::MgSettings;
use super::{CG_MAXIT, Control, IterStats, Next, Outcome, Problem, Settings, Snapshot, Throttle};
use crate::gpu::Gpu;

// scalar slots (gpu_mg uses those below 80)
const COMPL: u32 = 96;
const LAMBDA: u32 = 97;
const VOL: u32 = 98;
const CHANGE: u32 = 99;
const GREY: u32 = 100;
const BETA: u32 = 101;
/// max displacement of load case c at MAXD + c
const MAXD: u32 = 104;
const MAX_CASES: usize = 24;

/// Element arrays on the GPU.
struct Fields {
    flags: wgpu::Buffer,
    design: wgpu::Buffer,
    hs: wgpu::Buffer,
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
}

pub(super) fn optimize_gpu(p: &Problem, s: &Settings, ms: &MgSettings, out: &Path, gpu: &Gpu) -> Result<Outcome> {
    let ke = super::hex_ke(s.nu);
    let ne = p.domain.len();
    let ndesign = (0..ne).filter(|&e| p.domain[e] && !p.passive[e]).count();
    if ndesign == 0 {
        bail!("no design elements: the domain is empty or entirely kept solid");
    }
    if p.loads.len() > MAX_CASES {
        bail!("at most {MAX_CASES} load cases on the GPU");
    }
    let active = p.active_nodes();
    let free: Vec<bool> = (0..active.len()).map(|i| active[i] && !p.fixed[i]).collect();
    let e = Engine::new(gpu)?;
    let elements = ne as f64 / 1e6;
    let oom = |err: anyhow::Error| {
        anyhow::anyhow!("{err}: {elements:.1}M elements don't fit on this GPU; use a larger --voxel")
    };
    let mg = gpu.alloc_scoped(|| GpuMg::new(&e, p, &ke, &free, ms)).map_err(oom)??;
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
    let design: Vec<f32> = flags.iter().map(|f| (f & 4 != 0) as u32 as f32).collect();
    let x0: Vec<f32> = (0..ne)
        .map(|i| {
            if p.passive[i] {
                1.0
            } else if p.domain[i] {
                s.volfrac as f32
            } else {
                0.0
            }
        })
        .collect();
    let f = gpu
        .alloc_scoped(|| Fields {
            flags: e.init("flags", bytemuck::cast_slice(&flags)),
            design: e.init("design", bytemuck::cast_slice(&design)),
            hs: e.buf("hs", ne),
            x: e.init("x", bytemuck::cast_slice(&x0)),
            xnew: e.buf("xnew", ne),
            xtilde: e.buf("xtilde", ne),
            xphys: e.buf("xphys", ne),
            xused: e.buf("xused", ne),
            dvs: e.buf("dvs", ne),
            dc: e.buf("dc", ne),
            dv: e.buf("dv", ne),
        })
        .map_err(oom)?;
    drop((flags, design, x0));
    let emin = s.e0 * s.emin;
    let elem = |out: &mut Vec<Step>, kernel: &str, pf: [f32; 4], m: [u32; 4], bufs: &[&wgpu::Buffer]| {
        e.step(
            out,
            kernel,
            Params {
                n: dims4(p.n),
                m: [ne as u32, m[1], m[2], m[3]],
                f: pf,
                g: [0.0, 0.5, 0.0, 0.0], // η = 0.5
            },
            bufs,
            threads(ne),
        )
    };
    let material = [emin as f32, s.e0 as f32, s.penal as f32, p.h as f32];
    let rmin = [s.rmin as f32, 0.0, 0.0, 0.0];
    // filter: mode 0 raw sums, 1 densities
    let filter = |out: &mut Vec<Step>, mode: u32, inp: &wgpu::Buffer, outp: &wgpu::Buffer| {
        elem(out, "filter", rmin, [0, mode, 0, 0], &[&f.flags, &f.hs, inp, outp])
    };
    // xin -> xtilde -> xphys
    let densities = |out: &mut Vec<Step>, xin: &wgpu::Buffer| {
        filter(out, 1, xin, &f.xtilde);
        elem(
            out,
            "project",
            [0.0; 4],
            [0, BETA, 0, 0],
            &[&f.flags, &f.xtilde, &e.scalars, &f.xphys],
        );
    };

    // filter weight sums: the raw filter of ones
    {
        let ones = e.init("ones", bytemuck::cast_slice(&vec![1f32; ne]));
        let mut prog = vec![];
        // mode 0 doesn't read hs; bind `ones` in its place so hs is only written
        elem(&mut prog, "filter", rmin, [0; 4], &[&f.flags, &ones, &ones, &f.hs]);
        e.run(&[&prog])?;
    }
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
        "sens",
        material,
        [0, BETA, 0, 0],
        &[
            &f.flags, &f.xphys, &f.xtilde, &mg.ce, &f.hs, &e.scalars, &f.xnew, &f.dvs,
        ],
    );
    filter(&mut prog_sens, 0, &f.xnew, &f.dc);
    filter(&mut prog_sens, 0, &f.dvs, &f.dv);
    e.reduce(&mut prog_sens, Reduce::Grey, &f.xphys, &f.xphys, &f.design, ne, GREY);
    let nn = active.len();
    for (c, u) in mg.us.iter().enumerate() {
        e.reduce(&mut prog_sens, Reduce::MaxNorm, u, u, u, nn, MAXD + c as u32);
    }
    let mut prog_oc = vec![];
    elem(
        &mut prog_oc,
        "oc",
        [0.2, 0.0, 0.0, 0.0], // move limit
        [0, LAMBDA, COMPL, 0],
        &[&f.flags, &f.x, &f.dc, &f.dv, &e.scalars, &f.xnew],
    );
    densities(&mut prog_oc, &f.xnew);
    e.dot(&mut prog_oc, &f.xphys, &f.design, None, ne, VOL);
    let mut prog_accept = vec![];
    e.reduce(&mut prog_accept, Reduce::MaxDiff, &f.xnew, &f.x, &f.design, ne, CHANGE);
    e.lin(&mut prog_accept, &f.x, &f.xnew, None, (0.0, NONE), (1.0, NONE), ne);
    let mut prog_project = vec![];
    elem(
        &mut prog_project,
        "project",
        [0.0; 4],
        [0, BETA, 0, 0],
        &[&f.flags, &f.xtilde, &e.scalars, &f.xphys],
    );

    let mut beta = if s.beta_max > 0.0 { 1.0 } else { 0.0 };
    e.set_scalar(BETA, beta as f32);
    e.run(&[&prog_init])?;

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
    let t_start = Instant::now();
    let mut control = Control::default();
    let mut throttle = Throttle::default();
    let mut result = Outcome {
        iterations: 0,
        compliance: 0.0,
        max_disp: 0.0,
        stats: vec![],
    };
    // the last iteration whose fields are not written yet
    let mut pending = false;
    let mut lambda = 1.0f64;
    let mut last_change = 1.0f64;
    let target = s.volfrac * ndesign as f64;
    let snapshot = || -> Result<Snapshot> {
        let ee = e.read_f32(&mg.ee, ne)?;
        let ce = e.read_f32(&mg.ce, ne)?;
        let h3 = (p.h * p.h * p.h) as f32;
        Ok(Snapshot {
            density: e.read_f32(&f.xused, ne)?,
            energy: ee.iter().zip(&ce).map(|(a, b)| 0.5 * a * b / h3).collect(),
            disps: (0..mg.us.len()).map(|c| mg.displacement(c)).collect::<Result<_>>()?,
            iteration: 0,
        })
    };
    for it in 1..=s.iters {
        let t0 = Instant::now();
        let mut st = IterStats::default();
        e.run(&[&prog_stiffness])?;
        mg.build()?;
        st.setup = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let final_stage = beta > 0.0 && beta >= s.beta_max;
        let tol = if !s.adaptive_tol || final_stage {
            s.cg_tol
        } else {
            (0.05 * last_change).clamp(s.cg_tol, 1e-2)
        };
        st.cg_tol = tol;
        for c in 0..mg.us.len() {
            let n = mg.solve(c, tol, CG_MAXIT)?;
            if n >= CG_MAXIT {
                println!("warning: CG did not converge in {CG_MAXIT} iterations");
            }
            st.cg.push(n);
        }
        mg.compute_energies()?;
        st.solve = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        e.run(&[&prog_sens])?;
        let sc = e.read_scalars()?;
        let compliance = sc[COMPL as usize] as f64;
        let grey = sc[GREY as usize] as f64 / ndesign as f64;
        let max_disp = (0..mg.us.len())
            .map(|c| sc[MAXD as usize + c] as f64)
            .fold(0.0, f64::max);
        st.compliance = compliance;
        st.update = t2.elapsed().as_secs_f64();

        // the fields of this solve, before the update replaces the densities
        let t3 = Instant::now();
        if throttle.due() {
            let bytes = snapshot()?.write(p, out)?;
            throttle.wrote(t3);
            pending = false;
            if it == 1 {
                println!(
                    "wrote {} ({:.1} MB, updated as it runs)",
                    out.display(),
                    bytes as f64 / 1e6
                );
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

        super::print_iteration(it, compliance, max_disp, change, grey, beta, &st, t0);
        result.iterations = it;
        result.compliance = compliance;
        result.max_disp = max_disp;
        result.stats.push(st);
        match control.next(s, beta, compliance, change) {
            Next::Go => {}
            Next::Beta(b) => {
                beta = b;
                e.set_scalar(BETA, beta as f32);
                e.run(&[&prog_project])?;
            }
            Next::Stop => break,
        }
    }
    if pending {
        snapshot()?.write(p, out)?;
    }
    super::print_done(&result, t_start, out);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::{Settings, Solver, optimize};
    use super::*;

    /// A cantilever: clamped at x = 0, a spread load down at the far top edge, a kept-solid strip.
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
            fixed: vec![false; nn],
            loads: vec![vec![[0.0; 3]; nn], vec![[0.0; 3]; nn]],
            cases: vec![1, 2],
        };
        let [nx, ny, nz] = n.map(|v| v + 1);
        for k in 0..nz {
            for j in 0..ny {
                let m = p.node(0, j, k);
                p.fixed[m] = true;
            }
        }
        for j in 0..ny {
            let m = p.node(nx - 1, j, nz - 1);
            p.loads[0][m] = [0.0, 0.0, -10.0 / ny as f64];
            let m = p.node(nx - 1, j, nz / 2);
            p.loads[1][m] = [0.0, 3.0 / ny as f64, 0.0];
        }
        p
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
        for beta_max in [0.0, 8.0] {
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
            };
            let ms = MgSettings::default();
            let cpu = optimize(&p, &settings(Solver::Multigrid(ms)), &dir.join("cpu.vti"), None).unwrap();
            let gpu_r = optimize(&p, &settings(Solver::Gpu(ms)), &dir.join("gpu.vti"), Some(&gpu)).unwrap();
            assert_eq!(cpu.iterations, gpu_r.iterations);
            for (it, (a, b)) in cpu.stats.iter().zip(&gpu_r.stats).enumerate() {
                let rel = (a.compliance - b.compliance).abs() / a.compliance;
                assert!(
                    rel < 2e-3,
                    "beta_max {beta_max}, iteration {}: compliance {} (CPU) vs {} (GPU)",
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
            assert!(diff < 0.05, "beta_max {beta_max}: density differs by {diff}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
