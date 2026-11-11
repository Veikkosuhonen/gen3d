# gen3d: procedural SDF workbench for 3D printing

Write shapes as WGSL signed distance functions, preview them live (with hot reload), mesh them on the GPU,
and slice them to G-code. A single Rust binary built on [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12 or GL).

```
shape.wgsl ──► gen3d view    live raymarched preview, reloads on save
           ──► gen3d export  GPU compute-shader sampling → marching cubes → watertight STL/3MF
           ──► gen3d slice   export + PrusaSlicer/OrcaSlicer CLI → .gcode
           ──► gen3d open    export + open in the slicer GUI
           ──► gen3d export --format vti   SDF + boundary-condition fields as a VTK grid (simulation)

result.vti ──► gen3d view    threshold surface of a grid (e.g. a topology-optimization density), live reload
           ──► gen3d export  marching cubes → STL/3MF, or slice / open like a shape
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
| `F` | cycle `field_*` overlays: all, each field alone, off (see [Extra fields](#extra-fields-boundary-conditions-and-regions)) |
| `E` / `G` / `O` | export mesh / slice to G-code / open in slicer GUI (runs in the background) |
| `R`, `F5`, `Q` | reset camera, force reload, quit |

Keys are matched by physical position (US layout), so `[` `]` work without AltGr on Nordic layouts.

## Command reference

```
gen3d view   SHAPE [-D NAME=VAL]...     (SHAPE is a .wgsl file, or a .vti grid with the options below)
gen3d export SHAPE [-o out.stl] [--format stl|3mf|obj|ply|vti] [--voxel MM] [--bounds X0 Y0 Z0 X1 Y1 Z1] [--decimate F]
gen3d slice  SHAPE [-o out.gcode] [-c config.ini]... [--slicer prusa|orca] [mesh opts] [-- extra slicer args]
gen3d open   SHAPE [--slicer prusa|orca] [--format stl|3mf]
gen3d render SHAPE [-o img.png] [--size W H] [--section] [--yaw DEG] [--pitch DEG] [--no-fields]
gen3d check  SHAPE                     (for a .vti: lists its arrays and value ranges)

.vti options (view/render/export/slice/open): --array NAME  --iso X  --below|--above  --blocky  --color NAME
gen3d new    NAME
```

Outputs go to `out/`; `-D` values are appended to the filename (`out/knob_DIAM-30.stl`).
`check` only needs naga, not a GPU. `WGPU_BACKEND=vulkan|gl` and `WGPU_ADAPTER_NAME=...` pick the GPU.
Shaders are compiled without wgpu's loop-bounding and integer-division guards, which slow raymarching a lot
on Vulkan; an infinite loop in a shape can therefore stall the GPU until the driver resets it.
`GEN3D_SHADER_CHECKS=1` turns the guards back on.

## SDF grid export (VTI) for simulation

`gen3d export SHAPE --format vti` (or `-o field.vti`) skips meshing and writes the sampled SDF itself as VTK
ImageData: a Float32 point array `sdf` (mm, negative inside) plus any extra fields (see below), on a regular grid starting exactly at the
`bounds` minimum, with spacing `voxel`. Grid points sit at `min + i * voxel`, so a 0..22 mm range at 0.5 mm
gives 45 points. Use it as the design domain or initial density for topology optimization, for voxel FEA,
for lattice-Boltzmann/immersed-boundary CFD, and so on. It opens in ParaView and loads in Python with:

```python
import pyvista as pv
m = pv.read("out/knob.vti")
sdf = m["sdf"].reshape(m.dimensions[::-1])   # array indexed [z, y, x]
solid = sdf < 0                               # or a smooth density: 0.5 - 0.5*np.tanh(sdf / eps)
```

### Extra fields: boundary conditions and regions

Every function `fn field_<name>(p: vec3f) -> f32` or `-> vec3f` in the shape is sampled on the same grid
and written as an extra array `<name>` (1 or 3 components). This lets a single `.wgsl` file describe a
whole simulation or topology-optimization problem: `map` is the design domain, and the fields mark
supports, loads, and non-design regions. gen3d doesn't interpret the fields; the meaning is up to your
solver. [`shapes/bracket_domain.wgsl`](shapes/bracket_domain.wgsl) uses these conventions:

| field | type | meaning |
|---|---|---|
| `fixed` | f32 | `< 0` where nodes are clamped (the bolt hole walls) |
| `load` | vec3f | force direction × weight where load is applied, zero elsewhere; the solver scales it to the total force |
| `keep` | f32 | `< 0` where material is forced solid (mounting plate, pin boss) |

```python
fixed = m["fixed"] < 0                        # per-point masks
load = m["load"]                              # (N, 3) array, same point order as m.points
load *= 1000.0 / np.abs(load[:, 2]).sum()     # e.g. 1 kN total downwards
```

Write region functions as SDFs (shells via `max(d - t, -d)`, booleans, `-D` parameters), the same way as
`map`. Keep a region at least a voxel thick, or it may fall between grid points. `gen3d check` lists the
fields it finds.

`view` and `render` draw each field's region as a coloured overlay on the surface and on the section face,
and print a colour legend. Scalar fields are shown where they are `< 0`, and vector fields where they are
non-zero, with bands that move in the vector's direction. Overlapping regions alternate in diagonal
stripes. The lookup happens half a voxel inside the surface (where the exported grid samples the field),
so a shell written around a hole shows on the hole wall. Regions buried inside the part only appear in
the section view (`C`). `F` cycles the overlays.

The SDF is exact only if the shape's `map` is (smooth booleans, twist and noise give approximate distances;
the sign is always right).

## Viewing and meshing grids (.vti)

`view`, `render`, `export`, `slice` and `open` also take a `.vti` file (VTK ImageData), such as the density
your topology optimizer writes back. The viewer draws the region where one array is past a threshold:

```sh
gen3d check  result.vti                                   # list arrays: point/cell data, ranges
gen3d view   result.vti --array density --color stress    # live view; reloads when the file changes
gen3d export result.vti --array density --iso 0.5         # marching cubes -> out/result.stl
gen3d slice  result.vti --array density                   # straight to G-code
```

- **Arrays:** point data (values at grid points) and cell data (one value per voxel, as SIMP-style
  optimizers write) are both supported. Vector arrays are shown by magnitude. Data can be raw, base64 or
  ascii, optionally zlib-compressed, so files from gen3d, pyvista (`grid.save(...)`), ParaView and VTK
  load as written. LZ4/LZMA compression, multi-piece files and rotated grids (`Direction`) are not
  supported; a rotation is ignored with a warning.
- **Threshold:** by default, an SDF-like array (named `*sdf*`, or with values on both sides of 0) is
  solid where it is `< 0`. Anything else (densities, masks) is solid above the middle of its value range.
  `--iso`, `--below` and `--above` override this.
- **Smooth or blocky:** smooth draws the interpolated threshold surface, the same one `export` meshes.
  `--blocky` (or `M`) draws every inside sample as a cube; for cell data those are exactly the elements.
- **Colour:** `--color NAME` (or `F`) colours the surface and the section face by another array, using a
  viridis colour map. The range is the min to max of that array over the inside samples, and the window
  title shows it.
- **Live reload:** when the file changes on disk, the view reloads but keeps the array, threshold and
  colour you picked. If a write is caught half-finished, the old view stays until the next change.
- **Below the plate:** as with shapes, anything below z = 0 is hidden by the build plate. Grids centred
  on the origin show only their upper half, so shift the origin when you write the file.

Keys in the viewer for `.vti` files:

| | |
|---|---|
| `-` `=` | lower / raise the threshold (1/50 of the value range per press) |
| `M` | smooth / blocky |
| `I` | flip which side of the threshold is inside |
| `A` | next array |
| `F` | colour by: off, then each array in turn |

`E` / `G` / `O` export, slice or open with the current array and threshold. On a large grid (8M samples)
a threshold change takes about a second, because the distance field is recomputed on the CPU.

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
