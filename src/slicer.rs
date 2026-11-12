//! Find a slicer and drive it: open a model in the GUI, or slice to G-code headlessly.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Result, anyhow, bail};

const PRUSA_NAMES: &[&str] = &["prusa-slicer", "PrusaSlicer", "prusaslicer"];
const ORCA_NAMES: &[&str] = &["orca-slicer", "OrcaSlicer", "orcaslicer"];
const APPIMAGE_DIRS: &[&str] = &["~/Applications", "~/.local/bin", "~/Downloads", "./tools"];

const NOT_FOUND: &str = "no slicer found. Install one of:
  sudo pacman -S prusa-slicer          (official repo, recommended: full CLI slicing)
  yay -S orca-slicer-bin               (AUR)
or drop a PrusaSlicer/OrcaSlicer AppImage in ~/Applications, or set GEN3D_SLICER=/path/to/slicer";

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Kind {
    Prusa,
    Orca,
}

fn kind_of(path: &Path) -> Kind {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.contains("orca") { Kind::Orca } else { Kind::Prusa }
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(name))
        .find(|p| p.is_file())
}

fn expand_home(d: &str) -> PathBuf {
    match (d.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => Path::new(&home).join(rest),
        _ => PathBuf::from(d),
    }
}

/// Return (kind, executable). Honors $GEN3D_SLICER.
pub fn find_slicer(prefer: Option<Kind>) -> Option<(Kind, PathBuf)> {
    if let Some(env) = std::env::var_os("GEN3D_SLICER") {
        let p = PathBuf::from(env);
        return Some((kind_of(&p), p));
    }
    let order: Vec<&str> = if prefer == Some(Kind::Orca) {
        ORCA_NAMES.iter().chain(PRUSA_NAMES).copied().collect()
    } else {
        PRUSA_NAMES.iter().chain(ORCA_NAMES).copied().collect()
    };
    for name in order {
        if let Some(exe) = which(name) {
            return Some((kind_of(&exe), exe));
        }
    }
    for d in APPIMAGE_DIRS {
        let Ok(entries) = std::fs::read_dir(expand_home(d)) else {
            continue;
        };
        let mut hits: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let n = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                n.ends_with(".appimage") && n.contains("slice") && (n.contains("rusa") || n.contains("rca"))
            })
            .collect();
        hits.sort();
        if let Some(p) = hits.pop() {
            return Some((kind_of(&p), p));
        }
    }
    None
}

pub fn open_in_gui(model: &Path, prefer: Option<Kind>) -> Result<PathBuf> {
    let (_, exe) = find_slicer(prefer).ok_or_else(|| anyhow!(NOT_FOUND))?;
    let mut cmd = Command::new(&exe);
    cmd.arg(model)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // keep the GUI alive after we exit
    }
    cmd.spawn()
        .map_err(|e| anyhow!("could not start {}: {e}", exe.display()))?;
    Ok(exe)
}

pub fn slice_to_gcode(
    model: &Path,
    out: &Path,
    configs: &[PathBuf],
    extra: &[String],
    prefer: Option<Kind>,
) -> Result<()> {
    let (kind, exe) = find_slicer(prefer).ok_or_else(|| anyhow!(NOT_FOUND))?;
    let dir = out.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let mut args: Vec<String> = vec![];
    let s = |p: &Path| p.display().to_string();
    match kind {
        Kind::Prusa => {
            args.extend(["--export-gcode".into(), "--output".into(), s(out)]);
            for c in configs {
                args.extend(["--load".into(), s(c)]);
            }
        }
        Kind::Orca => {
            // OrcaSlicer: configs are JSON presets; first = machine, then process, filaments last
            let name = |c: &PathBuf| {
                c.file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default()
            };
            let machine: Vec<_> = configs
                .iter()
                .filter(|c| name(c).contains("machine") || name(c).contains("printer"))
                .collect();
            let filament: Vec<_> = configs.iter().filter(|c| name(c).contains("filament")).collect();
            let process: Vec<_> = configs
                .iter()
                .filter(|c| !machine.contains(c) && !filament.contains(c))
                .collect();
            args.extend(["--slice".into(), "0".into(), "--outputdir".into(), s(dir)]);
            let settings: Vec<String> = machine.iter().chain(&process).map(|c| s(c)).collect();
            if !settings.is_empty() {
                args.extend(["--load-settings".into(), settings.join(";")]);
            }
            if !filament.is_empty() {
                args.extend([
                    "--load-filaments".into(),
                    filament.iter().map(|c| s(c)).collect::<Vec<_>>().join(";"),
                ]);
            }
        }
    }
    args.extend(extra.iter().cloned());
    args.push(s(model));

    println!("$ {} {}", exe.display(), args.join(" "));
    let started = std::time::SystemTime::now();
    let r = Command::new(&exe)
        .args(&args)
        .output()
        .map_err(|e| anyhow!("could not run {}: {e}", exe.display()))?;
    let logs = || {
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&r.stdout),
            String::from_utf8_lossy(&r.stderr)
        )
    };
    if !r.status.success() {
        bail!("slicer failed ({}):\n{}", r.status, logs());
    }
    if kind == Kind::Orca {
        let newest = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.starts_with("plate_") && n.ends_with(".gcode")
            })
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .filter(|(t, _)| *t >= started)
            .max();
        if let Some((_, p)) = newest {
            std::fs::rename(p, out)?;
        }
    }
    if !out.exists() {
        bail!("slicer finished but {} was not written:\n{}", out.display(), logs());
    }
    summarize(out)
}

/// The settings and estimates worth showing from a G-code file's comments (print time, filament,
/// layer height, ...), as `key = value` lines. Slicers write them at the end of the file.
pub fn gcode_summary(gcode: &Path) -> Result<Vec<String>> {
    const KEYS: &[&str] = &[
        "estimated printing time (normal mode)",
        "filament used [g]",
        "filament used [mm]",
        "layer_height",
        "fill_density",
        "support_material",
        "nozzle_diameter",
        "printer_model",
    ];
    let mut found: Vec<Option<String>> = vec![None; KEYS.len()];
    let f = std::fs::File::open(gcode)?;
    for line in BufReader::new(f).split(b'\n').map_while(Result::ok) {
        let line = String::from_utf8_lossy(&line);
        let Some(rest) = line.strip_prefix("; ") else { continue };
        let key = rest.split(" =").next().unwrap_or("").trim();
        if let Some(i) = KEYS.iter().position(|k| *k == key)
            && found[i].is_none()
        {
            found[i] = Some(rest.trim().to_string());
        }
    }
    Ok(found.into_iter().flatten().collect())
}

fn summarize(gcode: &Path) -> Result<()> {
    let size = std::fs::metadata(gcode)?.len();
    println!("wrote {} ({:.1} MB)", gcode.display(), size as f64 / 1e6);
    for f in gcode_summary(gcode)? {
        println!("  {f}");
    }
    Ok(())
}

/// Slicer profiles in `printers/` under the working directory (PrusaSlicer .ini, Orca .json).
pub fn profiles() -> Vec<PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let Ok(entries) = std::fs::read_dir(cwd.join("printers")) else {
        return vec![];
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("ini") || e.eq_ignore_ascii_case("json"))
        })
        .collect();
    out.sort();
    out
}

/// Open a file or folder with the desktop's default application (detached).
pub fn open_with_desktop(path: &Path) -> Result<()> {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    let mut cmd = Command::new(opener);
    cmd.arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().map_err(|e| anyhow!("could not run {opener}: {e}"))?;
    Ok(())
}

/// Open a shape in the user's editor: `$GEN3D_EDITOR` (a command, e.g. `code` or `kitty nvim`) if
/// set, else the desktop's default application for the file.
pub fn open_in_editor(path: &Path) -> Result<()> {
    let Some(editor) = std::env::var("GEN3D_EDITOR").ok().filter(|e| !e.trim().is_empty()) else {
        return open_with_desktop(path);
    };
    let mut words = editor.split_whitespace();
    let exe = words.next().unwrap_or_default();
    let mut cmd = Command::new(exe);
    cmd.args(words)
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
        .map_err(|e| anyhow!("could not run GEN3D_EDITOR ({editor}): {e}"))?;
    Ok(())
}
