//! Writing the result while the optimizer runs.
//!
//! - Live writes are throttled to ~10% of the run time and happen on a background thread, so the
//!   solver doesn't wait for the disk.
//! - On large grids (more than `PREVIEW_CELLS` elements) the live file is a preview: density only,
//!   averaged over k³ blocks. A full-resolution file with every field is gigabytes there, and the
//!   viewer would reload it each time.
//! - The full fields (density, energy, displacements) are written at the end, and when the run is
//!   stopped: the app's Stop sends SIGTERM, which is caught so the current iteration finishes and
//!   the full result is written. A second SIGTERM ends the process at once.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Result, anyhow};
use rayon::prelude::*;

use super::symmetry::{Model, Sym};

/// Grids with more elements than this get a decimated, density-only live preview.
pub(super) const PREVIEW_CELLS: usize = 4_000_000;

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
    // a second SIGTERM kills as usual
    unsafe {
        libc::signal(libc::SIGTERM, libc::SIG_DFL);
    }
}

/// Catch SIGTERM (the app's Stop) so the run can end cleanly with its full result.
pub(super) fn catch_stop() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
    }
}

/// x̃ as written: NaN outside the domain (see `Snapshot::filtered`).
pub(super) fn filtered_out(p: &super::Problem, xtilde: impl Iterator<Item = f32>) -> Vec<f32> {
    xtilde
        .zip(&p.domain)
        .map(|(v, &d)| if d { v } else { f32::NAN })
        .collect()
}

pub(super) fn stop_requested() -> bool {
    STOP.load(Ordering::SeqCst)
}

/// What the .vti needs to know about the grid.
#[derive(Clone)]
struct Grid {
    n: [usize; 3],
    origin: [f64; 3],
    h: f64,
    /// displacement array names, per load case
    names: Vec<String>,
    /// a reduced (symmetric) model: its fields are mirrored onto this grid, and each case's
    /// displacement is combined from its parts in the groups (`Snapshot::disps` is per group, per
    /// case)
    sym: Option<Sym>,
}

/// One iteration's output fields (`filtered`, `energy` and `disps` empty for a density-only
/// preview).
pub(super) struct Snapshot {
    pub density: Vec<f32>,
    /// the densities before projection (x̃, or with overhang control the AM filter's x̂): smooth where `density` is nearly binary, with
    /// the same 0.5 surface (the projection's threshold), so it is what the mesher polygonizes.
    /// NaN outside the design domain, which tells the mesher where the domain ends.
    pub filtered: Vec<f32>,
    pub energy: Vec<f32>,
    /// per load case (with symmetry: per group, per case)
    pub disps: Vec<Vec<f32>>,
}

impl Grid {
    fn of(m: &Model) -> Grid {
        let p = m.p();
        let (n, origin) = match &m.sym {
            Some(s) => (s.full_n, s.full_origin),
            None => (p.n, p.origin),
        };
        Grid {
            n,
            origin,
            h: p.h,
            sym: m.sym.clone(),
            names: p
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
                .collect(),
        }
    }

    /// A cell array on this grid (mirrored from the reduced one).
    fn cells<'a>(&self, v: &'a [f32]) -> std::borrow::Cow<'a, [f32]> {
        match &self.sym {
            Some(sym) if !v.is_empty() => sym.expand_cells(v).into(),
            _ => v.into(),
        }
    }

    fn write(&self, s: &Snapshot, out: &Path) -> Result<u64> {
        let (density, filtered, energy) = (self.cells(&s.density), self.cells(&s.filtered), self.cells(&s.energy));
        let disps: Vec<std::borrow::Cow<[f32]>> = match &self.sym {
            Some(sym) if !s.disps.is_empty() => {
                let cases = self.names.len();
                (0..cases)
                    .map(|c| {
                        let parts: Vec<&[f32]> = (0..sym.patterns.len()).map(|g| &s.disps[g * cases + c][..]).collect();
                        sym.expand_disp(&parts).into()
                    })
                    .collect()
            }
            _ => s.disps.iter().map(|d| d[..].into()).collect(),
        };
        let mut arrays = vec![crate::vti::GridArray {
            name: "density",
            cell: true,
            comps: 1,
            data: &density,
        }];
        if !s.filtered.is_empty() {
            arrays.push(crate::vti::GridArray {
                name: crate::volume::FILTERED,
                cell: true,
                comps: 1,
                data: &filtered,
            });
        }
        if !s.energy.is_empty() {
            arrays.push(crate::vti::GridArray {
                name: "energy",
                cell: true,
                comps: 1,
                data: &energy,
            });
        }
        for (name, data) in self.names.iter().zip(&disps) {
            arrays.push(crate::vti::GridArray {
                name,
                cell: false,
                comps: 3,
                data,
            });
        }
        crate::vti::write_grid(out, self.origin, self.h, self.n.map(|v| v + 1), &arrays)
    }

    /// The density averaged over k³ blocks of cells, on a grid k× coarser.
    fn decimate(&self, density: &[f32], k: usize) -> (Grid, Vec<f32>) {
        let n = self.n;
        let nc = n.map(|v| v.div_ceil(k));
        let mut out = vec![0f32; nc.iter().product()];
        out.par_chunks_mut(nc[0]).enumerate().for_each(|(row, o)| {
            let (cj, ck) = (row % nc[1], row / nc[1]);
            for (ci, v) in o.iter_mut().enumerate() {
                let (mut s, mut c) = (0.0f32, 0);
                for k2 in ck * k..((ck + 1) * k).min(n[2]) {
                    for j in cj * k..((cj + 1) * k).min(n[1]) {
                        for i in ci * k..((ci + 1) * k).min(n[0]) {
                            s += density[(k2 * n[1] + j) * n[0] + i];
                            c += 1;
                        }
                    }
                }
                *v = s / c.max(1) as f32;
            }
        });
        let g = Grid {
            n: nc,
            origin: self.origin,
            h: self.h * k as f64,
            names: vec![],
            sym: None,
        };
        (g, out)
    }
}

pub(super) struct Output {
    grid: Grid,
    path: PathBuf,
    /// decimation of live writes (1: full fields)
    k: usize,
    /// end and duration of the last live write
    last: Option<(Instant, f64)>,
    writer: Option<JoinHandle<Result<(u64, f64)>>>,
    announced: bool,
}

impl Output {
    pub(super) fn new(m: &Model, path: &Path) -> Output {
        let grid = Grid::of(m);
        let ne: usize = grid.n.iter().product();
        let k = if ne > PREVIEW_CELLS {
            (ne as f64 / PREVIEW_CELLS as f64).cbrt().ceil() as usize
        } else {
            1
        };
        if k > 1 {
            println!(
                "live preview: density averaged over {k}x{k}x{k} blocks; the full fields are written at the end (or on Stop)"
            );
        }
        Output {
            grid,
            path: path.to_path_buf(),
            k,
            last: None,
            writer: None,
            announced: false,
        }
    }

    /// Whether live writes carry all fields (else only the density is used).
    pub(super) fn live_is_full(&self) -> bool {
        self.k == 1
    }

    /// Collect a finished background write.
    fn reap(&mut self) -> Result<()> {
        if self.writer.as_ref().is_some_and(|w| w.is_finished()) {
            let (bytes, secs) = self
                .writer
                .take()
                .unwrap()
                .join()
                .map_err(|_| anyhow!("the .vti writer panicked"))??;
            self.last = Some((Instant::now(), secs));
            if !self.announced {
                self.announced = true;
                println!(
                    "wrote {} ({:.1} MB, updated as it runs)",
                    self.path.display(),
                    bytes as f64 / 1e6
                );
            }
        }
        Ok(())
    }

    /// Whether a live write should be made now: the writer is idle and writing stays under ~10% of
    /// the time.
    pub(super) fn due(&mut self) -> Result<bool> {
        self.reap()?;
        Ok(self.writer.is_none() && self.last.is_none_or(|(t, d)| t.elapsed().as_secs_f64() >= 10.0 * d))
    }

    /// Write `s` in the background (decimated to a density-only preview on large grids).
    pub(super) fn live(&mut self, s: Snapshot) {
        let (grid, path, k) = (self.grid.clone(), self.path.clone(), self.k);
        self.writer = Some(std::thread::spawn(move || {
            let t = Instant::now();
            let bytes = if k > 1 {
                let (g, d) = grid.decimate(&grid.cells(&s.density), k);
                g.write(
                    &Snapshot {
                        density: d,
                        filtered: vec![],
                        energy: vec![],
                        disps: vec![],
                    },
                    &path,
                )?
            } else {
                grid.write(&s, &path)?
            };
            Ok((bytes, t.elapsed().as_secs_f64()))
        }));
    }

    /// Write the full fields, after any live write still running.
    pub(super) fn finish(&mut self, s: Snapshot) -> Result<u64> {
        if let Some(w) = self.writer.take() {
            w.join().map_err(|_| anyhow!("the .vti writer panicked"))??;
        }
        self.grid.write(&s, &self.path)
    }

    /// Wait for a live write still running (when the last one was already the full result).
    pub(super) fn wait(&mut self) -> Result<()> {
        if let Some(w) = self.writer.take() {
            w.join().map_err(|_| anyhow!("the .vti writer panicked"))??;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_averages_blocks() {
        let g = Grid {
            n: [5, 2, 3],
            origin: [1.0, 2.0, 3.0],
            h: 0.5,
            names: vec![],
            sym: None,
        };
        let d: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let (c, v) = g.decimate(&d, 2);
        assert_eq!(c.n, [3, 1, 2]);
        assert_eq!(c.h, 1.0);
        // block (0,0,0): cells x 0-1, y 0-1, z 0-1
        let want: f32 = [0, 1, 5, 6, 10, 11, 15, 16].iter().map(|&i| i as f32).sum::<f32>() / 8.0;
        assert_eq!(v[0], want);
        // the last x block holds one column of cells: x = 4
        let want: f32 = [4, 9, 14, 19].iter().map(|&i| i as f32).sum::<f32>() / 4.0;
        assert_eq!(v[2], want);
    }
}
