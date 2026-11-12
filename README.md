# gen3d: procedural SDF workbench for 3D printing

Write shapes as WGSL signed distance functions, preview them live (with hot reload), optionally
topology-optimize them, mesh them on the GPU, and slice them to G-code. A single Rust binary built on
[wgpu](https://wgpu.rs) (Vulkan, Metal, DX12 or GL).

```
gen3d   ─►  1 Design    open or create a .wgsl, edit it in your editor (live reload), tune parameters,
                        add primitives (sphere, box, ...) with the mouse
            2 Optimize  optional: topology optimization of the shape as a design domain (live)
            3 Print     mesh the design or the optimized result, pick a printer profile, slice to G-code
```

Every step is also a command (`export`, `optimize`, `slice`, ...) for scripting; the app runs them.

## Quick start

```sh
cargo install --path .              # installs the `gen3d` binary to ~/.cargo/bin
gen3d                               # open the app: pick a shape in shapes/, or create one
```

(Or `cargo run --release` without installing. `gen3d view FILE` opens a file directly.)

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
| `range NAME min max` | slider range for parameter `NAME` in the viewer (one line per parameter) |

`#include "file.wgsl"` resolves relative to the shape first, then the built-in library (compiled into the
binary). Each file is included once. Shapes are validated with [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
and errors are reported as `yourfile.wgsl:LINE:COL` with the offending line.

**Parameters** are WGSL [pipeline-overridable constants](https://www.w3.org/TR/WGSL/#override-decls):
declare `override NAME: f32 = default;` and set it with `-D NAME=VALUE` on any command. Values are applied
when the pipeline is created, so changing them never edits the source. An unknown `-D` name is an error
that lists the shape's parameters (the app warns and ignores it instead).

In the app every parameter gets a slider (a checkbox for `bool`). A trailing `// comment` on the
declaration becomes its tooltip. The slider covers 0 to twice the default unless you give it a range
with `//! range NAME min max`; you can still type any value into the number box. Optimize and Print use
the values on screen.

### Primitives from the app (`NAME.prims.json`)

The Design step can also add **primitives**: sphere, box, cylinder, capsule, torus and cone, each with
a position, rotation (degrees about X, then Y, then Z), size, edge rounding (boxes, cylinders) and a
*blend* radius for a smooth union. They are stored next to the shape, `knob.wgsl` → `knob.prims.json`,
and are **unioned** with the script: when that file exists, gen3d renames the script's `fn map` to
`g3d_script_map` and generates

```wgsl
fn map(p: vec3f) -> f32 {
    var d = g3d_script_map(p);                        // your script (or nothing, if it has no map)
    d = g3d_ed_union(d, g3d_e0(p), g3d_e0_blend);     // each visible primitive, in list order
    ...
    return d;
}
```

so every command (`render`, `export`, `optimize`, `slice`) and the app see the combined shape. Other
functions that call `map(p)` (e.g. a `field_keep`) get the combined shape too. The bounds grow to fit the
primitives. A shape can be primitives only: a `.wgsl` without `fn map` works when its `.prims.json`
exists. Each primitive's numbers are `override` constants, so moving or resizing one only rebuilds the
pipeline; adding, removing, hiding or changing a kind regenerates the source. Only union for now (no
subtraction or intersection).

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

## The app

`gen3d` (or `gen3d view [FILE]`) opens one window ([egui](https://github.com/emilk/egui), drawn with the
same wgpu device as the raymarcher). The top bar has **Open…**, the file with an **Edit** button, and
the three steps; the sidebar shows the current step's controls, and the viewport has a toolbar for the
camera presets, section view, bounds box, clipping and render resolution.

- **Open** (`Ctrl+O`, and at startup without a file): browse folders for `.wgsl` shapes and `.vti`
  grids, newest first, with shortcuts to the project, `shapes/` and `out/`. **New shape** writes the
  template into the folder, opens it and starts your editor.
- **Edit** opens the shape in `$GEN3D_EDITOR` (a command, e.g. `code` or `kitty nvim`) or else the
  desktop's default app. The preview reloads whenever the file or an include is saved; errors show in a
  panel under the view, and the last version that compiled stays on screen.
- **1 Design:** sliders for the shape's [parameters](#writing-shapes), [primitives](#primitives-from-the-app-nameprimsjson),
  `field_*` overlay selection, and shape details. A parameter change rebuilds only the pipeline, on a
  background thread, so dragging stays smooth. **Primitives:** *Add* puts one at the centre of the view;
  click one in the view (or the list) to select it. It is tinted blue and gets arrows: drag an arrow to
  move along X/Y/Z, the centre to move in the view plane, `Ctrl` snaps to whole mm. The sidebar edits
  its numbers; `Del` deletes, `Ctrl+D` duplicates, `Esc` deselects, `Ctrl+Z` / `Ctrl+Shift+Z` undo and
  redo. Every change is saved to the `.prims.json` right away, so Optimize and Print use it. Picking only
  sees primitives (the script's shape isn't known on the CPU), so a primitive hidden inside script
  geometry can still be clicked.
- **2 Optimize** (for shapes with `field_fixed` and `field_load`, see
  [Topology optimization](#topology-optimization)): material share, load, element size, minimum member
  size, iterations, crisp 0/1. **Optimize** runs in the background with a progress bar; from the first
  iteration the viewport shows the result growing. Then pick the threshold on its histogram (with the
  material volume in cm³ and grams), optionally colour by `energy` or `displacement`, and continue
  with **Print the result**.
- **3 Print:** print the design or the optimized result. Mesh voxel and triangle reduction (for
  shapes), STL or 3MF, **Export mesh**. Pick a printer profile from `printers/` (or *auto*: the shape's
  `//! slicer`, else `printers/default.ini`), optionally override layer height, infill and supports
  (PrusaSlicer), then **Slice to G-code** or **Open in slicer**. The result shows print time,
  filament and settings, read from the G-code.

Outputs have fixed names in `out/` from the file and its parameter values (`knob_DIAM-30.stl`,
`bracket_domain_topopt.vti`), so reopening a shape finds its earlier result, mesh and G-code. Each
output says when it is older than its source. Jobs run as `gen3d` subprocesses: one optimization and
one export/slice at a time, each with **Stop** and a log.

The keyboard shortcuts work whenever no text field has focus:

| | |
|---|---|
| `Ctrl+O` | open a shape |
| left drag / right drag / wheel | orbit / pan / zoom (double-click resets the camera) |
| `1` `2` `3` `4` | front / side / top / iso view |
| `C`, `[` `]` | toggle cross-section, move the cut plane (the section face shows internal distance bands) |
| `B` / `V` | toggle clipping to export bounds / bounds box |
| `F` | cycle `field_*` overlays: all, each field alone, off (see [Extra fields](#extra-fields-boundary-conditions-and-regions)) |
| `R`, `F5` | reset camera, force reload |

Keys are matched by physical position (US layout), so `[` `]` work without AltGr on Nordic layouts.

## Command reference

```
gen3d                                   the app
gen3d view   [FILE] [-D NAME=VAL]...    the app with a .wgsl shape or .vti grid open
gen3d export SHAPE [-o out.stl] [--format stl|3mf|obj|ply|vti] [--voxel MM] [--bounds X0 Y0 Z0 X1 Y1 Z1] [--decimate F]
gen3d slice  SHAPE [-o out.gcode] [-c config.ini]... [--slicer prusa|orca] [mesh opts] [-- extra slicer args]
gen3d open   SHAPE [--slicer prusa|orca] [--format stl|3mf]
gen3d render SHAPE [-o img.png] [--size W H] [--section] [--yaw DEG] [--pitch DEG] [--no-fields]
gen3d check  SHAPE                     (for a .vti: lists its arrays and value ranges)
gen3d optimize SHAPE [--volfrac 0.3] [--voxel MM] [--force N] [--rmin VOXELS] [--iters N] [--beta-max 32 | --no-projection] [--modulus MPA] [--nu V] [-o out.vti]

.vti options (render/export/slice/open): --array NAME  --iso X  --below|--above  --blocky  --color NAME
gen3d new    NAME
gen3d snapshot [FILE] [--step design|optimize|print] [--result] [--browser] [-o png]   (hidden; see Development)
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

The app opens `.vti` files (VTK ImageData), such as the density an external topology optimizer writes
back, and `render`, `export`, `slice` and `open` take them too. The viewport draws the region where one
array is past a threshold; the Design step has the threshold and colour controls, and Print meshes it:

```sh
gen3d check  result.vti                                   # list arrays: point/cell data, ranges
gen3d view   result.vti                                   # live view; reloads when the file changes
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
  viridis colour map. The range is the 1st to 99th percentile of that array over the inside samples (so
  a few hotspots don't flatten it); the sidebar shows a colour bar and can fix the range.
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

The sidebar has the same controls, plus a histogram for picking the threshold and the material volume
(see [The app](#the-app)); Print meshes with the current array and threshold. On a large grid (8M
samples) a threshold change takes about a second, because the distance field is recomputed on the CPU.

## Topology optimization

`gen3d optimize` finds the stiffest part that uses a given share of the design domain (minimum
compliance), with the SIMP method on a voxel grid. The problem is a shape file written with the
[extra fields](#extra-fields-boundary-conditions-and-regions) conventions:

| | |
|---|---|
| `map` | the design domain: where material may go |
| `field_fixed` (f32) | `< 0` where the part is clamped (all three displacement components are held) |
| `field_load` (vec3f) | load direction where it is non-zero; the total `--force` is spread over these nodes so the magnitudes add up to it |
| `field_keep` (f32, optional) | `< 0`: forced solid (mounting plates, bosses, faces you need) |
| `field_void` (f32, optional) | `< 0`: forced empty (clearance for bolts, other parts) |

```sh
gen3d optimize shapes/bracket_domain.wgsl --volfrac 0.3    # -> out/bracket_domain_topopt.vti
gen3d view out/bracket_domain_topopt.vti                     # opens on `density`, threshold 0.5
gen3d export out/bracket_domain_topopt.vti                   # -> out/bracket_domain_topopt.stl
```

In the app, the **Optimize** step runs the same thing in the background and shows the result in the
viewport while it is still running. The file is rewritten after every iteration, atomically, so you
watch the design form; **Stop** ends the run and keeps the last iteration.

- **Output:** cell arrays `density` (0..1) and `energy` (strain energy density, MPa: where the part
  works hardest), and point array `displacement` (mm). Colour by `energy` or `displacement` to see how
  the result carries the load.
- **Units:** mm, N and MPa. The default material is PLA-like (E = 3500 MPa, ν = 0.35), so the printed
  maximum displacement is a rough real-world estimate for a solid print. It ignores layer adhesion and
  infill.
- **Method:**
  - Each voxel is a trilinear 8-node hex element.
  - Density is penalized with p = 3, and a density filter with radius `--rmin` voxels sets the minimum
    member size.
  - **Heaviside projection** (on by default) pushes the filtered densities towards 0 or 1 around 0.5.
    Its sharpness β starts at 1 and doubles each time the design settles (or after 25 iterations), up
    to `--beta-max` (32). Starting sharp would lock in a poor layout early.
  - The density update uses optimality criteria (move limit 0.2) under the volume constraint, which is
    measured on the projected densities.
  - It stops when compliance improves by less than 0.1% for 3 iterations at the final β, or at
    `--iters` (150).
- **Projection or not:** with projection the result is crisp (0% grey on the bracket against 24%
  without) and stiffer for the same material (compliance 8.96 against 10.36 N·mm). Thresholding at
  0.5 then matches `--volfrac` closely. The cost is about 1.7× the iterations. Because each voxel is now
  fully in or out, the threshold surface follows the voxel staircase; a finer `--voxel` gives a smoother
  part. `--no-projection` gives the faster, softer-edged grey result.
- **Speed:** the solver is matrix-free conjugate gradient with a Jacobi preconditioner, multithreaded and
  warm-started. The default voxel keeps the bounds at about 40k elements. On 12 cores, the bracket
  (34k domain elements) takes about 1–2 s per iteration: 49 iterations and 77 s with projection, 29
  iterations and 52 s without. Time grows quickly with resolution: halving the voxel is 8× the elements
  and more solver iterations.
- **Shortcuts in this first version:**
  - No multigrid, so fine grids are slow.
  - One load case.
  - No symmetry or overhang (printability) constraints.

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
gen3d snapshot shapes/bracket_domain.wgsl --step optimize -o out/ui.png   # the app's window, offscreen
```

`snapshot` draws the app's window into a PNG without opening one (same GPU path), to check the UI
headlessly. It can open a step (`--step`), the optimized result in Print (`--result`), the Open
dialog (`--browser`), or select a primitive (`--select N`).

| file | role |
|---|---|
| `src/main.rs` | CLI |
| `src/app.rs` | the app window: Design / Optimize / Print steps, jobs, viewport, `snapshot` |
| `src/browser.rs` | the Open dialog and New shape |
| `src/jobs.rs` | background `gen3d` subprocess jobs, log parsing (optimization progress, mesh report) |
| `src/shape.rs` | `#include` expansion, `//!` directives, naga validation, error → `file:line:col` |
| `src/prims.rs` | editor primitives: `.prims.json`, generated WGSL union, bounds, CPU SDFs for picking |
| `src/gpu.rs` | adapter selection (hardware first, software fallback with a warning) |
| `src/mesher.rs` | compute-shader sampling harness, grid padding, polygonize + report |
| `src/mc.rs` | marching cubes with a generated crack-free case table |
| `src/mesh.rs` | volume / watertight / bodies, decimation, STL/3MF/OBJ/PLY writers |
| `src/viewer.rs` | raymarching preview harness and pipelines, headless PNG render |
| `src/slicer.rs` | PrusaSlicer / OrcaSlicer discovery and CLI driving, G-code summary, profiles, opening files |
| `src/vti.rs` | VTK ImageData reader and writers |
| `src/volume.rs` | .vti grids as viewable shapes: distance fields, EDT, volume WGSL, meshing |
| `src/topopt.rs` | topology optimization: hex element, matrix-free PCG, filter, OC update |
| `src/wgsl/sdf.wgsl` | the SDF library |
