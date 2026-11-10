"""gen3d command line: view / export / slice / open / new."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .shader import ShaderError, load_shape

TEMPLATE = """// {name}.glsl — gen3d shape. Units: mm, +Z up, build plate at z = 0.
// Export region (xmin ymin zmin  xmax ymax zmax) and mesh resolution in mm:
//! bounds -30 -30 0  30 30 50
//! voxel 0.2
#include <sdf.glsl>

float map(vec3 p) {{
    float body = sdRoundBox(p - vec3(0, 0, 15), vec3(20, 20, 15), 3.0);
    float hole = sdCylinder(rotX(p - vec3(0, 0, 13), deg(90.0)), 7.0, 25.0);  // through-hole along Y
    float ball = sdSphere(p - vec3(0, 0, 30), 12.0);
    return opSmoothUnion(opSmoothSubtract(body, hole, 2.0), ball, 4.0);
}}
"""


def parse_defines(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip() or "1"
    return out


def default_out(shape_path: Path, ext: str, defines: dict) -> Path:
    suffix = "".join(f"_{k}-{v}" for k, v in defines.items())
    return Path.cwd() / "out" / f"{shape_path.stem}{suffix}{ext}"


def resolve_configs(args, shape) -> list[Path]:
    if args.config:
        return [Path(c) for c in args.config]
    if d := shape.directives.get("slicer"):
        return [(shape.path.parent / c.strip()).resolve() for c in d.split(",")]
    printers = Path.cwd() / "printers"
    for name in ("default.ini",):
        if (printers / name).exists():
            return [printers / name]
    return []


def cmd_view(args):
    from .viewer import view

    view(Path(args.shape), parse_defines(args.define))


def _export(args, ext=None):
    from . import mesher

    defines = parse_defines(args.define)
    shape = load_shape(args.shape, defines)
    if args.bounds:
        shape.bounds = (tuple(args.bounds[:3]), tuple(args.bounds[3:]))
    decimate = args.decimate if args.decimate is not None else float(shape.directives.get("decimate", 0) or 0)
    mesh = mesher.build_mesh(shape, args.voxel, decimate or None)
    mesher.report(mesh)
    if ext is None:  # `export` command: -o is the mesh path
        out = Path(args.output) if args.output else default_out(shape.path, "." + args.format, defines)
    else:  # slice/open: mesh goes to out/, -o (if any) means something else
        out = default_out(shape.path, ext, defines)
    mesher.save_mesh(mesh, out)
    return shape, out


def cmd_export(args):
    _export(args)


def cmd_slice(args):
    from . import slicer

    shape, model = _export(args, ".stl")
    gcode = Path(args.output) if args.output else model.with_suffix(".gcode")
    configs = resolve_configs(args, shape)
    if not configs:
        print("note: no slicer config given (--config, //! slicer, or printers/default.ini); "
              "using the slicer's built-in defaults")
    slicer.slice_to_gcode(model, gcode, configs, args.slicer_args, prefer=args.slicer)


def cmd_open(args):
    from . import slicer

    _, model = _export(args, ".3mf" if args.format == "3mf" else ".stl")
    exe = slicer.open_in_gui(model, prefer=args.slicer)
    print(f"opened in {exe}")


def cmd_new(args):
    path = Path(args.name)
    if path.suffix != ".glsl":
        path = Path("shapes") / f"{args.name}.glsl"
    if path.exists():
        sys.exit(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.format(name=path.stem))
    print(f"created {path}\nrun: uv run gen3d view {path}")


def cmd_render(args):
    from .viewer import render_png

    defines = parse_defines(args.define)
    out = Path(args.output) if args.output else default_out(Path(args.shape), ".png", defines)
    render_png(Path(args.shape), defines, out, tuple(args.size), args.section, args.yaw, args.pitch)
    print(f"wrote {out}")


def cmd_check(args):
    from . import mesher

    shape = load_shape(args.shape, parse_defines(args.define))
    ctx = mesher.create_context()
    mesher.compile_compute(ctx, shape)
    print(f"ok: {shape.path.name} compiles; bounds {shape.bounds}, voxel {shape.voxel} mm")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gen3d", description="Procedural SDF modelling workbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def shape_args(sp):
        sp.add_argument("shape", help="path to .glsl shape file")
        sp.add_argument("-D", "--define", action="append", metavar="NAME=VALUE",
                        help="inject #define (repeatable), e.g. -D RADIUS=12.0")

    def mesh_args(sp):
        sp.add_argument("--voxel", type=float, help="voxel size in mm (overrides //! voxel)")
        sp.add_argument("--bounds", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
        sp.add_argument("--decimate", type=float, help="keep this fraction of triangles (0-1)")
        sp.add_argument("--format", choices=["stl", "3mf", "obj", "ply"], default="stl")

    sp = sub.add_parser("view", help="live preview window with hot reload")
    shape_args(sp)
    sp.set_defaults(fn=cmd_view)

    sp = sub.add_parser("export", help="mesh the SDF and write STL/3MF")
    shape_args(sp)
    mesh_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("slice", help="export and slice to G-code (PrusaSlicer/OrcaSlicer CLI)")
    shape_args(sp)
    mesh_args(sp)
    sp.add_argument("-o", "--output", help="G-code output path")
    sp.add_argument("-c", "--config", action="append",
                    help="slicer config (PrusaSlicer .ini, or Orca .json presets); repeatable")
    sp.add_argument("--slicer", choices=["prusa", "orca"], help="prefer this slicer")
    sp.set_defaults(fn=cmd_slice)

    sp = sub.add_parser("open", help="export and open in the slicer GUI")
    shape_args(sp)
    mesh_args(sp)
    sp.add_argument("--slicer", choices=["prusa", "orca"])
    sp.set_defaults(fn=cmd_open)

    sp = sub.add_parser("new", help="create a new shape from a template")
    sp.add_argument("name")
    sp.set_defaults(fn=cmd_new)

    sp = sub.add_parser("render", help="render a preview image (PNG) headlessly")
    shape_args(sp)
    sp.add_argument("-o", "--output")
    sp.add_argument("--size", type=int, nargs=2, default=[1024, 768], metavar=("W", "H"))
    sp.add_argument("--section", action="store_true", help="show cross-section")
    sp.add_argument("--yaw", type=float, default=-60.0)
    sp.add_argument("--pitch", type=float, default=28.0)
    sp.set_defaults(fn=cmd_render)

    sp = sub.add_parser("check", help="compile a shape and report errors")
    shape_args(sp)
    sp.set_defaults(fn=cmd_check)

    p.epilog = "slice: arguments after `--` are passed to the slicer, e.g. -- --layer-height 0.2"
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    args.slicer_args = passthrough
    try:
        args.fn(args)
    except (ShaderError, ValueError, RuntimeError) as e:
        sys.exit(f"\033[31merror:\033[0m {e}")
