//! Resuming a stopped run. When a run is stopped before it converges, `run` writes a checkpoint next
//! to the result (`<result>.resume`): the design variables x, the projection's β, the iteration
//! count and the loop's state (`LoopState`), so `--resume` continues where the run would have gone
//! on. A run that finishes, or a fresh run, removes it.
//!
//! Format: one line of JSON (`Header`), then x as little-endian f32 per element of the grid the run
//! was on (the coarse grid if it was stopped during the coarse start).

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::symmetry::Model;
use super::{LoopState, Problem, Settings};

const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub(super) struct Header {
    version: u32,
    /// stopped during the coarse start (x is on the coarse grid)
    pub coarse: bool,
    /// the settings that shape the run (`settings_key`)
    settings: String,
    /// elements per axis, and a fingerprint of the problem (domain, supports, loads)
    n: [usize; 3],
    problem: String,
    /// iterations done, over the whole run
    pub iterations: usize,
    /// β for the next iteration
    pub beta: f64,
    pub state: LoopState,
}

pub(super) struct Checkpoint {
    pub header: Header,
    pub x: Vec<f64>,
}

/// Where the checkpoint of the run writing `out` goes.
pub(super) fn path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".resume");
    out.with_file_name(name)
}

/// The settings a resumed run must share with the stopped one. The iteration limit, the stopping
/// tolerance and the solver may change.
fn settings_key(s: &Settings) -> String {
    format!(
        "voxel {} volfrac {} penal {} rmin {} E {} nu {} force {} beta_max {} emin {} ground {} coarse_start {} symmetry {} overhang {} up {}",
        s.voxel,
        s.volfrac,
        s.penal,
        s.rmin,
        s.e0,
        s.nu,
        s.force,
        s.beta_max,
        s.emin,
        s.ground.name(),
        s.coarse_start,
        s.symmetry,
        s.overhang,
        s.up.name()
    ) + &s
        .plane2d
        .map_or(String::new(), |a| format!(" 2d {}", ["x", "y", "z"][a]))
}

/// FNV-1a over the model's grid, domain, supports and loads, every group's (stable across builds).
fn fingerprint(m: &Model) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in &m.groups {
        hash_problem(&mut h, p);
    }
    format!("{h:016x}")
}

fn hash_problem(h: &mut u64, p: &Problem) {
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for v in p.n {
        eat(&(v as u64).to_le_bytes());
    }
    eat(&p.h.to_le_bytes());
    for v in p.origin {
        eat(&v.to_le_bytes());
    }
    for (&d, &k) in p.domain.iter().zip(&p.passive) {
        eat(&[d as u8 | (k as u8) << 1]);
    }
    eat(&p.fixed);
    eat(&p.contact);
    for case in &p.loads {
        for v in case.iter().flatten() {
            eat(&v.to_le_bytes());
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write(
    out: &Path,
    s: &Settings,
    m: &Model,
    coarse: bool,
    iterations: usize,
    beta: f64,
    state: &LoopState,
    x: &[f64],
) -> Result<()> {
    let header = Header {
        version: VERSION,
        coarse,
        settings: settings_key(s),
        n: m.p().n,
        problem: fingerprint(m),
        iterations,
        beta,
        state: state.clone(),
    };
    let target = path(out);
    let tmp = target.with_extension("resume.tmp");
    let mut buf = serde_json::to_vec(&header)?;
    buf.push(b'\n');
    buf.reserve(4 * x.len());
    for &v in x {
        buf.extend((v as f32).to_le_bytes());
    }
    let mut f = std::fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    f.write_all(&buf)?;
    drop(f);
    std::fs::rename(&tmp, &target)?;
    println!(
        "saved {} (continue with --resume)",
        target.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(())
}

/// Read the checkpoint of the run writing `out`, checking that it was made with the same settings.
pub(super) fn read(out: &Path, s: &Settings) -> Result<Checkpoint> {
    let target = path(out);
    if !target.exists() {
        bail!(
            "nothing to resume: no {} (a run that is stopped before it converges leaves one)",
            target.display()
        );
    }
    let bytes = std::fs::read(&target).with_context(|| format!("reading {}", target.display()))?;
    let nl = bytes
        .iter()
        .position(|&b| b == b'\n')
        .context("damaged checkpoint: no header")?;
    let header: Header = serde_json::from_slice(&bytes[..nl]).context("damaged checkpoint header")?;
    if header.version != VERSION {
        bail!("the checkpoint is from another version of gen3d; run without --resume to start over");
    }
    if header.settings != settings_key(s) {
        bail!(
            "the checkpoint was made with other settings; run without --resume to start over\n  \
             checkpoint: {}\n  now:        {}",
            header.settings,
            settings_key(s)
        );
    }
    let data = &bytes[nl + 1..];
    if !data.len().is_multiple_of(4) {
        bail!("damaged checkpoint: truncated design");
    }
    let x = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c) as f64)
        .collect();
    Ok(Checkpoint { header, x })
}

impl Checkpoint {
    /// Check that the checkpoint belongs to problem `p` (the grid it was stopped on).
    pub(super) fn check(&self, m: &Model) -> Result<()> {
        let (n, p) = (self.header.n, m.p());
        if n != p.n || self.x.len() != p.domain.len() || self.header.problem != fingerprint(m) {
            bail!(
                "the shape, its supports or its loads changed since the run was stopped (grid {}x{}x{}, now \
                 {}x{}x{}); run without --resume to start over",
                n[0],
                n[1],
                n[2],
                p.n[0],
                p.n[1],
                p.n[2]
            );
        }
        Ok(())
    }
}

/// Remove the checkpoint of the run writing `out`, if any.
pub(super) fn remove(out: &Path) {
    let _ = std::fs::remove_file(path(out));
}

#[cfg(test)]
mod tests {
    use super::super::{Ground, Settings, Solver, tests::grid};
    use super::*;

    fn settings(volfrac: f64) -> Settings {
        Settings {
            voxel: 1.0,
            volfrac,
            penal: 3.0,
            rmin: 1.5,
            iters: 50,
            tol: 0.01,
            e0: 3500.0,
            nu: 0.35,
            force: 1.0,
            beta_max: 32.0,
            solver: Solver::Jacobi,
            emin: 1e-9,
            cg_tol: 1e-4,
            adaptive_tol: true,
            ground: Ground::None,
            coarse_start: true,
            symmetry: true,
            overhang: false,
            up: crate::mesh::Up::PlusZ,
            plane2d: None,
        }
    }

    /// The design, β, iteration count and loop state come back; other settings or another problem
    /// are refused.
    #[test]
    fn checkpoint_round_trip() {
        let dir = std::env::temp_dir().join(format!("gen3d-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("r.vti");
        let p = grid([4, 2, 2], 1.0);
        let x: Vec<f64> = (0..p.domain.len()).map(|e| e as f64 / 16.0).collect();
        let state = LoopState {
            lambda: 0.25,
            last_change: 0.125,
            ..Default::default()
        };
        let s = settings(0.3);
        write(&out, &s, &Model::single(p.clone()), true, 17, 4.0, &state, &x).unwrap();
        assert_eq!(path(&out), dir.join("r.vti.resume"));
        let c = read(&out, &s).unwrap();
        assert!(c.header.coarse);
        assert_eq!((c.header.iterations, c.header.beta), (17, 4.0));
        assert_eq!((c.header.state.lambda, c.header.state.last_change), (0.25, 0.125));
        assert_eq!(c.x, x);
        c.check(&Model::single(p.clone())).unwrap();
        // the iteration limit may change, the volume fraction not
        read(&out, &Settings { iters: 80, ..s.clone() }).unwrap();
        assert!(read(&out, &settings(0.25)).is_err());
        // another load
        let mut q = grid([4, 2, 2], 1.0);
        q.loads = vec![vec![[0.0, 0.0, -1.0]; q.fixed.len()]];
        assert!(c.check(&Model::single(q)).is_err());
        remove(&out);
        assert!(read(&out, &s).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
