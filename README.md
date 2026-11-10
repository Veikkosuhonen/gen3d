# gen3d: procedural SDF workbench for 3D printing

Write shapes as WGSL signed distance functions, preview them live (with hot reload), mesh them on the GPU,
and slice them to G-code. A single Rust binary built on [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12 or GL).

```
shape.wgsl ──► gen3d view    live raymarched preview, reloads on save
           ──► gen3d export  GPU compute-shader sampling → marching cubes → watertight STL/3MF
           ──► gen3d slice   export + PrusaSlicer/OrcaSlicer CLI → .gcode
           ──► gen3d open    export + open in the slicer GUI
```

## Quick start

```sh
cargo install --path .              # installs the `gen3d` binary to ~/.cargo/bin
gen3d new mything                   # creates shapes/mything.wgsl from a template
gen3d view shapes/mything.wgsl      # open the preview, then edit the file in your editor
gen3d export shapes/mything.wgsl    # -> out/mything.stl
gen3d slice shapes/mything.wgsl     # -> out/mything.gcode (needs a slicer, see below)
```

(Or use `cargo run --release -- view shapes/knob.wgsl` without installing.)

Slicer: `sudo pacman -S prusa-slicer` (full CLI slicing support). Then export your printer profile to
`printers/default.ini`; see [printers/README.md](printers/README.md).

## Writing shapes

A shape file defines `fn map(p: vec3f) -> f32`: the signed distance (mm) from point `p` to the surface,
negative inside. Coordinates are **millimetres, +Z up, build plate at z = 0**.

```wgsl
//! bounds -30 -30 0  30 30 40     // export region: xmin ymin zmin  xmax ymax zmax
//! voxel 0.2                      // mesh resolution (mm); smaller = finer & slower
#include <sdf.wgsl>

override R: f32 = 12.0;            // parameter: override with -D R=15

fn map(p: vec3f) -> f32 {
    let body = sdRoundBox(p - vec3f(0, 0, 15), vec3f(20, 20, 15), 3.0);
    let ball = sdSphere(p - vec3f(0, 0, 30), R);
    return opSmoothUnion(body, ball, 4.0);
}
```

Directives (`//! key value`, anywhere in the main file):

| directive | meaning |
|---|---|
| `bounds x0 y0 z0 x1 y1 z1` | region that gets meshed; anything outside is cut off (the viewer shows this box) |
| `voxel 0.2` | grid spacing in mm for export |
| `step 0.6` | raymarch step factor for the viewer; lower it if a non-exact SDF (twist, noise, displacement) shows holes or artifacts |
| `decimate 0.5` | keep this fraction of triangles after meshing |
| `slicer ../printers/x.ini` | slicer config(s) for this shape |

`#include "file.wgsl"` resolves relative to the shape first, then the built-in library (compiled into the
binary). Each file is included once. Shapes are validated with [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
and errors are reported as `yourfile.wgsl:LINE:COL` with the offending line.

**Parameters** are WGSL [pipeline-overridable constants](https://www.w3.org/TR/WGSL/#override-decls):
declare `override NAME: f32 = default;` and set it with `-D NAME=VALUE` on any command. Values are applied
when the pipeline is created, so changing them never edits the source. An unknown `-D` name is an error
that lists the shape's parameters.

### Library (`src/wgsl/sdf.wgsl`)

- **Primitives:** `sdSphere sdBox sdRoundBox sdTorus sdCylinder sdRoundCylinder sdCapsule sdCappedCone sdHexPrism sdOctahedron sdPlane`
- **2D → 3D:** `sd2Circle sd2Box opExtrude opRevolve`
- **Booleans:** `opUnion opSubtract opIntersect opSmoothUnion opSmoothSubtract opSmoothIntersect opChamferUnion`
- **Modifiers:** `opRound opShell` (hollow with wall thickness)
- **Domain:** `rotX rotY rotZ rot2 deg opTwist opBend opRepeat opRepeatLim opPolar opMirrorX opMirrorY`
- **Lattices:** `sdGyroid sdSchwarzP` (use with `opShell` + `opIntersect` for TPMS infill)
- **Noise:** `hash13 valueNoise fbm`
- **Constants & helpers:** `PI TAU floorMod` (GLSL-style `mod`; WGSL's `%` truncates)

Distance functions are mostly from [Inigo Quilez](https://iquilezles.org/articles/distfunctions/).

### Coming from GLSL

| GLSL | WGSL |
|---|---|
| `float map(vec3 p)` | `fn map(p: vec3f) -> f32` |
| `vec3(0, 0, 1)`, `vec3(20)` | `vec3f(0, 0, 1)`, `vec3f(20.0)` |
| `float d = ...;` / reassigned later | `let d = ...;` / `var d = ...;` |
| `#ifndef R` / `#define R 12.0` | `override R: f32 = 12.0;` |
| `p.xy = rot2(a) * p.xy;` (swizzle assign) | `let xy = rot2(a) * p.xy; let q = vec3f(xy, p.z);` |
| `atan(y, x)`, `mod(x, y)` | `atan2(y, x)`, `floorMod(x, y)` |
| `for (int i = 0; ...)`, `float(i)` | `for (var i = 0; ...)`, `f32(i)` |
| `max(v, 0.0)` on a vector | `max(v, vec3f(0.0))` |
| `c ? a : b` | `select(b, a, c)` |

## Viewer controls

| | |
|---|---|
| left drag / right drag / wheel | orbit / pan / zoom |
| `1` `2` `3` `4` | front / side / top / iso view |
| `C`, `[` `]` | toggle cross-section, move the cut plane (the section face shows internal distance bands) |
| `B` / `V` | toggle clipping to export bounds / bounds box |
| `E` / `G` / `O` | export mesh / slice to G-code / open in slicer GUI (runs in the background) |
| `R`, `F5`, `Q` | reset camera, force reload, quit |

Keys are matched by physical position (US layout), so `[` `]` work without AltGr on Nordic layouts.

## Command reference

```
gen3d view   SHAPE [-D NAME=VAL]...
gen3d export SHAPE [-o out.stl] [--format stl|3mf|obj|ply] [--voxel MM] [--bounds X0 Y0 Z0 X1 Y1 Z1] [--decimate F]
gen3d slice  SHAPE [-o out.gcode] [-c config.ini]... [--slicer prusa|orca] [mesh opts] [-- extra slicer args]
gen3d open   SHAPE [--slicer prusa|orca] [--format stl|3mf]
gen3d render SHAPE [-o img.png] [--size W H] [--section] [--yaw DEG] [--pitch DEG]
gen3d check  SHAPE
gen3d new    NAME
```

Outputs go to `out/`; `-D` values are appended to the filename (`out/knob_DIAM-30.stl`).
`check` only needs naga, not a GPU. `WGPU_BACKEND=vulkan|gl` and `WGPU_ADAPTER_NAME=...` pick the GPU.

## Mesh quality notes

- Meshes are closed by construction (the grid is padded one voxel past `bounds`, and the marching-cubes
  case table is generated so neighbouring cells always agree on ambiguous faces). `export` prints
  `watertight=` and the volume; if a part touches the bounds it is cut flat there.
- Export resolution is `voxel`: features thinner than ~2 voxels may disappear or break apart. 0.1–0.2 mm
  is plenty for FDM; triangle counts grow with 1/voxel².
- Non-exact SDFs (smooth booleans, twist, noise) are fine for meshing; only the preview's raymarcher cares.
  Scale the field down (`d *= 0.7`) or set `//! step`.
- `--decimate` uses meshoptimizer's quadric simplifier. If a collapse pinches a thin feature, the area is
  locked and simplification is redone; if the result still isn't watertight the full mesh is kept.

## Development

```
cargo test            # marching-cubes table and watertightness, shape loading / error mapping
cargo clippy --all-targets
```

| file | role |
|---|---|
| `src/main.rs` | CLI |
| `src/shape.rs` | `#include` expansion, `//!` directives, naga validation, error → `file:line:col` |
| `src/gpu.rs` | adapter selection (hardware first, software fallback with a warning) |
| `src/mesher.rs` | compute-shader sampling harness, grid padding, polygonize + report |
| `src/mc.rs` | marching cubes with a generated crack-free case table |
| `src/mesh.rs` | volume / watertight / bodies, decimation, STL/3MF/OBJ/PLY writers |
| `src/viewer.rs` | raymarching preview harness, winit window, headless PNG render |
| `src/slicer.rs` | PrusaSlicer / OrcaSlicer discovery and CLI driving |
| `src/wgsl/sdf.wgsl` | the SDF library |
