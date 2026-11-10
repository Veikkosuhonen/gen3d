"""Find a slicer and drive it: open a model in the GUI, or slice to G-code headlessly."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

PRUSA_NAMES = ["prusa-slicer", "PrusaSlicer", "prusaslicer"]
ORCA_NAMES = ["orca-slicer", "OrcaSlicer", "orcaslicer"]
APPIMAGE_DIRS = ["~/Applications", "~/.local/bin", "~/Downloads", "./tools"]


def _kind(path: str) -> str:
    return "orca" if "orca" in Path(path).name.lower() else "prusa"


def find_slicer(prefer: str | None = None) -> tuple[str, str] | None:
    """Return (kind, executable). Honors $GEN3D_SLICER."""
    env = os.environ.get("GEN3D_SLICER")
    if env:
        return _kind(env), env
    order = ORCA_NAMES + PRUSA_NAMES if prefer == "orca" else PRUSA_NAMES + ORCA_NAMES
    for name in order:
        if exe := shutil.which(name):
            return _kind(exe), exe
    for d in APPIMAGE_DIRS:
        for pat in ("*rusa*lice*.AppImage", "*rca*lice*.AppImage"):
            hits = sorted(glob.glob(str(Path(d).expanduser() / pat)))
            if hits:
                return _kind(hits[-1]), hits[-1]
    return None


NOT_FOUND = """no slicer found. Install one of:
  sudo pacman -S prusa-slicer          (official repo, recommended: full CLI slicing)
  yay -S orca-slicer-bin               (AUR)
or drop a PrusaSlicer/OrcaSlicer AppImage in ~/Applications, or set GEN3D_SLICER=/path/to/slicer"""


def open_in_gui(model: Path, prefer: str | None = None):
    found = find_slicer(prefer)
    if not found:
        raise RuntimeError(NOT_FOUND)
    _, exe = found
    subprocess.Popen([exe, str(model)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return exe


def slice_to_gcode(model: Path, out: Path, configs: list[Path], extra: list[str] | None = None,
                   prefer: str | None = None, log=print) -> Path:
    found = find_slicer(prefer)
    if not found:
        raise RuntimeError(NOT_FOUND)
    kind, exe = found
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or []

    if kind == "prusa":
        cmd = [exe, "--export-gcode", "--output", str(out)]
        for c in configs:
            cmd += ["--load", str(c)]
        cmd += extra + [str(model)]
    else:
        # OrcaSlicer: configs are JSON presets; first = machine, then process, filaments last
        machine = [c for c in configs if "machine" in c.name.lower() or "printer" in c.name.lower()]
        filament = [c for c in configs if "filament" in c.name.lower()]
        process = [c for c in configs if c not in machine and c not in filament]
        cmd = [exe, "--slice", "0", "--outputdir", str(out.parent)]
        settings = [str(c) for c in machine + process]
        if settings:
            cmd += ["--load-settings", ";".join(settings)]
        if filament:
            cmd += ["--load-filaments", ";".join(str(c) for c in filament)]
        cmd += extra + [str(model)]

    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"slicer failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    if kind == "orca":
        produced = sorted(out.parent.glob("plate_*.gcode"), key=os.path.getmtime)
        if produced:
            produced[-1].replace(out)
    if not out.exists():
        raise RuntimeError(f"slicer finished but {out} was not written:\n{r.stdout}\n{r.stderr}")
    _summarize(out, log)
    return out


def _summarize(gcode: Path, log=print):
    keys = ("estimated printing time (normal mode)", "filament used [mm]", "filament used [g]",
            "layer_height", "fill_density", "nozzle_diameter", "printer_model")
    found = {}
    with gcode.open(errors="replace") as f:
        for line in f:
            if line.startswith("; "):
                k = line[2:].split(" =", 1)[0].strip()
                if k in keys and k not in found:
                    found[k] = line[2:].strip()
    log(f"wrote {gcode} ({gcode.stat().st_size / 1e6:.1f} MB)")
    for k in keys:
        if k in found:
            log("  " + found[k])
