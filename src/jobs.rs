//! Background `gen3d <command>` runs started from the app (optimize, export, slice, open). They run as
//! subprocesses of the same binary, so the CLI and the app share one code path, a job can be stopped
//! (killed) without corrupting the app, and its output is captured for the log.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default, Clone)]
pub struct Job {
    /// subcommand, e.g. "slice"
    pub cmd: String,
    /// the main file the job writes (to match a job to what the UI shows)
    pub target: PathBuf,
    /// arguments after the subcommand, for display
    pub args: String,
    pub running: bool,
    pub ok: Option<bool>,
    pub log: String,
    pub started: Option<Instant>,
    pub took: Duration,
    /// process id while running, for Stop
    pub pid: Option<u32>,
    pub stopped: bool,
}

impl Job {
    pub fn elapsed(&self) -> f32 {
        if self.running {
            self.started.map_or(0.0, |s| s.elapsed().as_secs_f32())
        } else {
            self.took.as_secs_f32()
        }
    }
}

/// One job slot: at most one job runs in it at a time.
#[derive(Clone, Default)]
pub struct Slot(Arc<Mutex<Job>>);

impl Slot {
    pub fn get(&self) -> Job {
        self.0.lock().unwrap().clone()
    }

    pub fn running(&self) -> bool {
        self.0.lock().unwrap().running
    }

    /// Run `gen3d <cmd> <args...>`; does nothing if a job is already running in this slot.
    pub fn start(&self, cmd: &str, args: Vec<String>, target: PathBuf) {
        {
            let mut job = self.0.lock().unwrap();
            if job.running {
                return;
            }
            *job = Job {
                cmd: cmd.to_string(),
                target,
                args: args.join(" "),
                running: true,
                started: Some(Instant::now()),
                ..Default::default()
            };
        }
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("gen3d"));
        let mut c = std::process::Command::new(exe);
        c.arg(cmd).args(&args);
        c.stdout(Stdio::piped()).stderr(Stdio::piped());
        println!("$ gen3d {cmd} {}", args.join(" "));
        let job = self.0.clone();
        std::thread::spawn(move || {
            let push = |job: &Mutex<Job>, line: &str| {
                let line = strip_ansi(line.rsplit('\r').next().unwrap_or(line));
                let mut j = job.lock().unwrap();
                j.log.push_str(&line);
                j.log.push('\n');
            };
            let result = (|| -> std::io::Result<std::process::ExitStatus> {
                let mut child = c.spawn()?;
                job.lock().unwrap().pid = Some(child.id());
                let (out, err) = (child.stdout.take(), child.stderr.take());
                let j = job.clone();
                let reader = std::thread::spawn(move || {
                    for l in out
                        .into_iter()
                        .flat_map(|o| BufReader::new(o).lines().map_while(|l| l.ok()))
                    {
                        println!("{l}");
                        push(&j, &l);
                    }
                });
                for l in err
                    .into_iter()
                    .flat_map(|e| BufReader::new(e).lines().map_while(|l| l.ok()))
                {
                    eprintln!("{l}");
                    push(&job, &l);
                }
                let _ = reader.join();
                child.wait()
            })();
            let mut j = job.lock().unwrap();
            j.running = false;
            j.pid = None;
            j.took = j.started.map_or_else(Duration::default, |s| s.elapsed());
            j.ok = Some(match result {
                Ok(s) => s.success(),
                Err(e) => {
                    let msg = format!("could not run gen3d {}: {e}\n", j.cmd);
                    j.log.push_str(&msg);
                    false
                }
            });
        });
    }

    /// Stop the running job (it keeps what it already wrote).
    pub fn stop(&self) {
        let mut j = self.0.lock().unwrap();
        if let Some(pid) = j.pid {
            j.stopped = true;
            let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
        }
    }
}

/// Drop ANSI escape sequences (colours) from a line of job output.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            if it.peek() == Some(&'[') {
                it.next();
                for c in it.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Progress of a `gen3d optimize` run, from its log.
#[derive(Default, Debug, PartialEq)]
pub struct OptProgress {
    pub iteration: usize,
    pub compliance: Option<String>,
    pub grey: Option<String>,
    pub beta: Option<String>,
    /// the final "converged ..." / "done in ..." line, once there is one
    pub finished: Option<String>,
    /// wall time of each iteration, s
    pub iter_secs: Vec<f32>,
    /// iterations after which the projection sharpened (`beta -> ...`)
    pub beta_steps: Vec<usize>,
}

/// Estimated time left of an optimization run, s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eta {
    /// best guess, if the run's course can be predicted (with projection)
    pub likely: Option<f32>,
    /// if it runs to the iteration limit
    pub max: f32,
}

impl OptProgress {
    /// Time left, from the recent iteration times. `beta_max` is the projection's final sharpness
    /// (0 without projection); each sharpening stage ends on convergence or after 25 iterations.
    pub fn eta(&self, maxit: usize, beta_max: f64) -> Option<Eta> {
        // the first iteration includes setup, so leave it out once there are others
        let times = match self.iter_secs.len() {
            0 => return None,
            1 => &self.iter_secs[..],
            _ => &self.iter_secs[1..],
        };
        let recent = &times[times.len().saturating_sub(5)..];
        let per = recent.iter().sum::<f32>() / recent.len() as f32;
        let left_max = maxit.saturating_sub(self.iteration);
        let max = left_max as f32 * per;
        let beta = self.beta.as_deref().and_then(|b| b.parse::<f64>().ok());
        let likely = beta.filter(|_| beta_max > 0.0).map(|beta| {
            let mut bounds = vec![0];
            bounds.extend(&self.beta_steps);
            let done: Vec<usize> = bounds.windows(2).map(|w| w[1] - w[0]).collect();
            let typical = if done.is_empty() {
                25.0
            } else {
                done.iter().sum::<usize>() as f32 / done.len() as f32
            };
            let in_stage = self.iteration.saturating_sub(*bounds.last().unwrap()) as f32;
            let later = (beta_max / beta).log2().round().max(0.0) as f32;
            let final_stage = later == 0.0;
            let mut cur = (typical - in_stage).max(1.0);
            if !final_stage {
                cur = cur.min((25.0 - in_stage).max(1.0));
            }
            let its = (cur + later * typical).min(left_max as f32);
            its * per
        });
        Some(Eta { likely, max })
    }
}

pub fn opt_progress(log: &str) -> OptProgress {
    let mut p = OptProgress::default();
    for l in log.lines() {
        let l = l.trim();
        if let Some(rest) = l.strip_prefix("it ") {
            let Some((n, rest)) = rest.split_once(':') else {
                continue;
            };
            p.iteration = n.trim().parse().unwrap_or(p.iteration);
            for part in rest.split(", ") {
                let part = part.trim();
                if let Some(v) = part.strip_prefix("compliance ") {
                    p.compliance = Some(v.to_string());
                } else if let Some(v) = part.strip_prefix("grey ") {
                    p.grey = Some(v.to_string());
                } else if let Some(v) = part.strip_prefix("beta ") {
                    p.beta = Some(v.to_string());
                } else if let Some((secs, _)) = part.split_once(" s [")
                    && let Ok(t) = secs.trim().parse()
                {
                    p.iter_secs.push(t);
                }
            }
        } else if l.starts_with("beta -> ") {
            p.beta_steps.push(p.iteration);
        } else if l.starts_with("done in ") {
            p.finished = Some(l.to_string());
        }
    }
    p
}

/// The `size ... volume ... watertight=..., bodies=...` line `mesher::report` prints.
pub fn mesh_report(log: &str) -> Option<String> {
    log.lines()
        .rev()
        .find(|l| l.starts_with("size ") && l.contains("watertight="))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_opt_progress() {
        let log = "SIMP: volume fraction 0.3\n\
                   it   1: compliance 9.6000e1 N·mm, max displacement 1.17 mm, change 0.200, grey 24%, beta 1, 900 CG its, 1.8 s [setup 0.10 solve 1.50 update 0.10 io 0.10]\n\
                   beta -> 2\n\
                   it  12: compliance 1.0400e1 N·mm, max displacement 0.124 mm, change 0.050, grey 10%, beta 2, 950 CG its, 1.7 s [setup 0.10 solve 1.40 update 0.10 io 0.10]\n";
        let p = opt_progress(log);
        assert_eq!(p.iteration, 12);
        assert_eq!(p.compliance.as_deref(), Some("1.0400e1 N·mm"));
        assert_eq!(p.grey.as_deref(), Some("10%"));
        assert_eq!(p.beta.as_deref(), Some("2"));
        assert!(p.finished.is_none());
        assert_eq!(p.iter_secs, vec![1.8, 1.7]);
        assert_eq!(p.beta_steps, vec![1]);
        let p = opt_progress(&format!("{log}done in 52.0 s: 29 iterations\n"));
        assert!(p.finished.unwrap().starts_with("done in"));
    }

    #[test]
    fn opt_eta() {
        let p = OptProgress {
            iteration: 10,
            beta: Some("4".into()),
            iter_secs: vec![5.0, 2.0, 2.0],
            beta_steps: vec![4, 8],
            ..Default::default()
        };
        // stages of 4 iterations, 2 into the beta 4 one, then 8, 16, 32: 2 + 3 * 4 = 14 its at 2 s
        let e = p.eta(100, 32.0).unwrap();
        assert_eq!(e.likely, Some(28.0));
        assert_eq!(e.max, 180.0);
        // capped by the iteration limit
        assert_eq!(p.eta(12, 32.0).unwrap().likely, Some(4.0));
        // without projection only the upper bound is known
        assert_eq!(p.eta(100, 0.0).unwrap().likely, None);
        assert!(OptProgress::default().eta(100, 32.0).is_none());
    }

    #[test]
    fn strips_ansi() {
        assert_eq!(strip_ansi("\x1b[31merror:\x1b[0m bad"), "error: bad");
    }
}
