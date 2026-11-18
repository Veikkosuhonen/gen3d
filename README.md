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
| `2d y` | optimize in 2D, projected along this axis (`x` `y` `z`), see [2D optimization](#2d-optimization) (an editor shape keeps it in its sidecar's `"2d"` instead) |
| `up -x` | the side printed up (`+z` `-z` `+x` `-x` `+y` `-y`): Optimize's overhang control and Print's mesh orientation default to it (an editor shape keeps it in its sidecar's `"up"` instead) |
| `import NAME file.vti [options]` | a `.vti` grid's solid (e.g. an optimization result) as `fn NAME(p: vec3f) -> f32`, see [Importing a result](#importing-a-result-into-a-shape-import) (one line per grid) |

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

**Live vs. compiled parameters.** The type decides what a slider costs:

| type | in the preview | dragging the slider |
|---|---|---|
| `f32` | read from a uniform buffer (the `override` becomes a `var<private>` of the same name, set when the fragment shader starts) | instant: a buffer write |
| `i32`, `u32`, `bool` | stays an `override` constant the compiler sees | recompiles the shader (~100 ms, in the background) |

So use `f32` for sizes and offsets, and `i32`/`u32` for things the compiler should know: loop counts
(`for (var i = 0; i < N; i++)` can then be unrolled), array sizes, switch selectors. Compiled parameters
are shown in *italics* in the app. `render`, `export`, `optimize` etc. compile once, so there every
parameter is an `override`. If the preview can't make an `f32` live (one used where WGSL needs a constant,
e.g. `override B: f32 = A * 2.0;`), all of that shape's parameters stay compiled.

### Primitives from the app (`NAME.prims.json`)

The Design step can also add **primitives**: sphere, box, cylinder, capsule, torus, cone and
[import](#importing-a-result-into-a-shape-import) (a `.vti` result), each with
a position, rotation (degrees about X, then Y, then Z), size, edge rounding (boxes, cylinders), an
**operation** (*add*, *subtract* or *intersect*) and a *blend* radius that smooths it. They are stored
next to the shape, `knob.wgsl` → `knob.prims.json`, and applied to the script **top to bottom**: each
primitive combines with everything above it (the script first). A subtract carves the script and the
primitives before it; an add after it puts material back. When that file exists, gen3d renames the
script's `fn map` to `g3d_script_map` and generates

```wgsl
fn map(p: vec3f) -> f32 {
    var d = g3d_script_map(p);                                  // your script (or nothing, if it has no map)
    d = g3d_ed_combine(d, g3d_e0(p), g3d_ed[1].w, g3d_ed[3].w); // each visible primitive: blend, operation
    ...
    return d;
}
```

so every command (`render`, `export`, `optimize`, `slice`) and the app see the combined shape. Other
functions that call `map(p)` (e.g. a `field_keep`) get the combined shape too. The bounds grow to fit
added primitives (subtracting and intersecting ones never add material). A shape can be primitives only: a `.wgsl` without `fn map` works when its `.prims.json`
exists. The primitives' numbers live in one `array<vec4f>` (`g3d_ed`): in the preview it's a uniform
buffer, so moving or resizing a primitive, switching its operation, or selecting one is a buffer write
with no shader compile; in `render`, `export`, `optimize` etc. it's a `const`, baked in. Adding,
removing, hiding or changing a kind regenerates the source.

**Mirror planes** (the *Mirror* row under the primitives; `"mirror": [0, null, null]` in the
`.prims.json` for a plane at x = 0): every primitive is mirrored across each plane, becoming the union of
itself and its mirror images (`g3d_e<i>` is the min over them), so a subtracted hole is cut on both
sides and a support or loaded region gets its twin. A load keeps its direction on the mirrored region
(the same push over both; its force is spread over both). The bounds become symmetric about the planes.
The script's own `map` is not mirrored. A new plane goes through the middle of the bounds.

**Optimize settings** (`"optimize": {"volfrac": 0.3, "iters": 150, "force": 100, "rmin": 1.5,
"projection": true, "ground": "none", "overhang": true}`, plus `"voxel"` when the element size isn't
automatic): the app saves the Optimize step's settings here each time it starts a run and restores them
when the shape is opened again. A missing field takes its default. Only the app reads them; `gen3d
optimize` takes its settings from its flags. A script shape without a `.prims.json` keeps them for the
session only.

### Importing a result into a shape (`//! import`)

To combine an optimization result (or any `.vti` grid) with exact SDFs, import it. There are two ways:

- **As a primitive (in the app):** Design → *Add* → **import…** lists the `.vti` files in `out/` and
  next to the shape, newest first. The result becomes a primitive like the others: drag it with the
  arrows, rotate it, set its operation (add, subtract, intersect), blend and optimization role, and
  mirror it. Its *position* is the grid's centre, so a new one sits exactly where it was optimized and
  rotates about its middle; *reset* puts it back there. Clicking it in the view selects it. The
  sidecar stores the file relative to the shape (`"kind": "import", "file": "../out/x_topopt.vti"`).
  Start from an empty New shape to build on a result with primitives alone.
- **In the script:** `//! import NAME file.vti` gives the shape an SDF function to use in `map`:

```wgsl
//! import leg ../out/stand_leg_topopt.vti
#include <sdf.wgsl>

fn map(p: vec3f) -> f32 {
    let foot = sdRoundBox(p - vec3f(0, 0, 2), vec3f(20, 40, 2), 1.0);         // an exact base plate
    let bolt = sdCylinder(p - vec3f(0, 30, 0), 3.2, 30.0);                    // a clean bolt hole
    return opSubtract(opSmoothUnion(leg(p), foot, 2.0), bolt);
}
```

Either way:

- **The file** is looked up next to the shape, then relative to the working directory. When it changes
  (e.g. you rerun the optimizer), the shape reloads.
- **The surface** is the one Print would mesh from that file: an optimizer's `density` is
  [faired](#viewing-and-meshing-grids-vti) at 0.5, and other arrays get the view's interpolated
  threshold surface. Where the result ends at the design domain, the faces are flat. `//! import`
  takes options after the file: `array=NAME`, `iso=X`, `below` / `above`, `fair` / `no-fair`,
  `grow=MM` (as `export`'s `--array`, `--iso`, ... for `.vti` files); a primitive uses the defaults.
- **Without `fn map`** (a file with only directives), the shape *is* the `//! import`ed solid(s).
- Without `//! bounds`, the bounds are the `//! import`ed grid's box (grown to fit primitives as
  usual), and an import primitive grows them like any primitive. Without `//! up`, an `//! import`ed
  result's print orientation carries over, and `//! step` defaults to 0.8.
- The field is computed on the CPU when the shape loads (about 1.5 s for a 4M-sample result) and
  cached, so editing the script reloads quickly. On the GPU, all imported grids share one storage
  buffer (`g3d_imp`, group 0 binding 8). The field is sampled trilinearly, so features smaller than
  the result's voxel are as coarse as they are in the result. That is why the exact SDFs you add
  around it stay sharp.

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
  grids, newest first, with shortcuts to the project, `shapes/` and `out/`. **New shape** writes an
  empty `.wgsl` into the folder and opens it: build it from primitives in Design, or click **Edit** to
  script it. A file with no code (only comments, `//!` directives, includes) is a shape made of its
  primitives; a file with code still needs `fn map`. `gen3d new NAME` (CLI) still writes the scripted
  template.
- **Edit** opens the shape in `$GEN3D_EDITOR` (a command, e.g. `code` or `kitty nvim`) or else the
  desktop's default app. The preview reloads whenever the file or an include is saved; errors show in a
  panel under the view, and the last version that compiled stays on screen.
- **1 Design:** sliders for the shape's [parameters](#writing-shapes), [primitives](#primitives-from-the-app-nameprimsjson),
  `field_*` overlay selection, and shape details. `f32` parameters update live; `i32`/`u32`/`bool`
  ones (in *italics*) rebuild the pipeline on a background thread (see
  [live vs. compiled](#writing-shapes)). **Primitives:** *Add* puts one at the centre of the view;
  click one in the view (or the list) to select it. It is tinted blue and gets arrows: drag an arrow to
  move along X/Y/Z, the centre to move in the view plane, `Ctrl` snaps to whole mm. The sidebar edits
  its numbers; `Del` deletes, `Ctrl+D` duplicates, `Esc` deselects, `Ctrl+Z` / `Ctrl+Shift+Z` undo and
  redo. *Operation* sets add / subtract / intersect (the list shows `+`, `-`, `&`); **Up** / **Down**
  change the order they're applied in. The selected primitive is drawn as a translucent blue ghost
  wherever it isn't the visible surface (a cutter, or a part buried in the script's shape), so cutters
  can be seen and dragged. Every change is saved to the `.prims.json` right away, so Optimize and Print
  use it. Picking only sees primitives (the script's shape isn't known on the CPU), so a primitive
  hidden inside script geometry can still be clicked.
- **2 Optimize** (for shapes with `field_fixed` and `field_load`, see
  [Topology optimization](#topology-optimization)): material share, load, element size, minimum member
  size, iterations, crisp 0/1, printable without supports and which side is printed up (on by default,
  +z up; see [overhang control](#overhang-control)), and the design space: 3D or
  [2D along X, Y or Z](#2d-optimization) (saved with the shape). With another side up the viewport
  shows the build plate. **Optimize** runs in the background with a progress bar; from the first
  iteration the viewport shows the result growing. Then pick the threshold on its histogram (with the
  material volume in cm³ and grams), optionally colour by `energy` or `displacement`, and continue
  with **Print the result**.
- **3 Print:** print the design or the optimized result. Which side is printed up (the same setting as
  Optimize's; the mesh is turned so that side faces up and set on the plate), mesh voxel and triangle
  reduction (for shapes), STL or 3MF, **Export mesh**. Pick a printer profile from `printers/` (or *auto*: the shape's
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
gen3d export SHAPE [-o out.stl] [--format stl|3mf|obj|ply|vti] [--voxel MM] [--bounds X0 Y0 Z0 X1 Y1 Z1] [--decimate F] [--up=+z|-z|+x|-x|+y|-y]
            [--min-body PCT] [.vti: --smooth N] [--fair | --no-fair] [--grow MM]
gen3d slice  SHAPE [-o out.gcode] [-c config.ini]... [--slicer prusa|orca] [mesh opts] [-- extra slicer args]
gen3d open   SHAPE [--slicer prusa|orca] [--format stl|3mf] [mesh opts]
gen3d render SHAPE [-o img.png] [--size W H] [--section] [--yaw DEG] [--pitch DEG] [--no-fields]
gen3d check  SHAPE                     (for a .vti: lists its arrays and value ranges)
gen3d optimize SHAPE [--volfrac 0.3] [--voxel MM] [--force N] [--rmin VOXELS] [--iters N] [--beta-max 32 | --no-projection] [--no-coarse-start] [--no-symmetry] [--no-overhang] [--up=+z|...] [--2d x|y|z | --3d] [--modulus MPA] [--nu V] [-o out.vti] [--resume]

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
non-zero, with bands that move in the vector's direction (for 3 s after the view changes, then they
hold still so an idle window leaves the GPU alone). Overlapping regions alternate in diagonal stripes. The lookup happens half a voxel inside the surface (where the exported grid samples the field),
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
- **Smooth or blocky:** smooth draws the interpolated threshold surface. `--blocky` (or `M`) draws every
  inside sample as a cube; for cell data those are exactly the elements.
- **Meshing (`export`, Print):** marching cubes, then `--smooth N` Taubin passes (default 10). For an
  optimizer's `density` the surface is first *faired*: the smoothest surface that stays within 0.45
  voxel of the voxels' own (a thin-plate fit, `src/fairing.rs`). That removes the voxel terraces on
  shallow slopes, which Taubin smoothing hardly touches, and keeps thin members as thick as the
  voxels; where the part meets the design domain's boundary, the outline on the flat face follows the
  same smooth surface instead of the voxel staircase. `--no-fair` meshes the interpolated
  `density_filtered` instead (the view's surface); `--fair` fairs any other array (e.g. a binary mask).
  Fairing can't remove features of a whole voxel, such as the ridges of a coarse design's thin shells:
  a finer `--voxel` or larger `--rmin` in `optimize` can, or `--grow`.
- **Smoother than the design (`--grow MM`, Print's *grow to smooth*):** lets the faired surface move up
  to MM outward instead of half a voxel, so it can fill the ripples, terraces, dents and narrow pockets
  the optimizer itself leaves a voxel or two deep. Walls never get thinner, but thin ones get thicker
  (by about MM/2 a side), so the part is heavier and stiffer than optimized: about 2 voxels gave
  +8% (monitor table) to +25% (stand leg, 2-voxel shells) volume. Ridges sticking out of a shell
  stay, softened, unless MM is large enough to fill the pocket around them. Meshing takes 2-3x as
  long. (Letting the surface move inward too was tried: on optimized shells it mostly thins them and
  opens holes.)
- **Small pieces (`--min-body PCT`, Print's *drop small pieces*):** removes separate pieces and fills
  voids inside the part that enclose less than PCT % of the largest body's volume. Works for shapes too.
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
| (primitives) | supports, loads (with load cases) and kept regions can also come from primitives, see [below](#supports-and-loads-from-primitives) |
| `field_fixed` (f32) | `< 0` where the part is clamped (all three displacement components are held) |
| `field_load` (vec3f) | load direction where it is non-zero; the total `--force` is spread over these nodes so the magnitudes add up to it |
| `field_keep` (f32, optional) | `< 0`: forced solid (mounting plates, bosses, faces you need) |
| `field_void` (f32, optional) | `< 0`: forced empty (clearance for bolts, other parts) |

```sh
gen3d optimize shapes/bracket_domain.wgsl --volfrac 0.3    # -> out/bracket_domain_topopt.vti
gen3d view out/bracket_domain_topopt.vti                     # opens on `density`, threshold 0.5
gen3d export out/bracket_domain_topopt.vti                   # -> out/bracket_domain_topopt.stl
```

### 2D optimization

Many parts are a profile of constant thickness loaded in its own plane: a beam or a block pushed
from the top, a bracket plate, a shelf support. For those, `--2d AXIS` (the shape's `//! 2d y` or
sidecar `"2d": "y"`, the app's *Design space*) optimizes the profile and extrudes it back, at about
4× the in-plane resolution of a 3D run in the same time, with a cross-section that is the same all
through.

- **Into 2D:** the problem is sampled in 3D as usual, then projected along the axis. Looking
  through the part: an element of the profile is in the domain (kept solid) where any element
  behind it is; a node is held where any node behind it is; the loads' in-plane components are
  summed through the part and their component along the axis is dropped (a load with nothing
  left is an error).
- **The solve:** the profile is a slab one element thick, solved by the same optimizer (plane
  stress). It stands for the domain's full thickness T: its loads are scaled by element / T, so
  stresses and displacements are the thick part's, and the reported compliance is the part's.
  Overhang control works in the profile's plane (45° chamfers and arches, extruded); printed along
  the axis, the extrusion stands as a prism and needs nothing. Mirror planes in the plane are used;
  one across the axis is ignored.
- **Back to 3D:** the result (`density`, `density_filtered`, `energy`, `displacement` in the
  plane) is written on the 3D grid, the profile copied through the design domain: where the 3D
  domain is missing (a hole, a chamfer, a void) the part is too. Along the axis the grid has at most
  24 cells (finer isn't needed for a prism; the beam's result is 2.4M cells, not 9M). The file
  records the axis as FieldData `plane2d`.
- **Element size:** by default about 100k elements in the profile (GPU; 13k on the CPU), with at
  most 50M cells sampled in 3D.
- **When it's wrong:** anything that isn't the same through the thickness. A load along the axis,
  a support on one face only, a domain that changes along the axis: the projection makes a prism of
  them, so check the result in 3D. [`shapes/beam_2d.wgsl`](shapes/beam_2d.wgsl) is the clean case:
  the classic MBB beam, a 120 × 20 × 40 mm block on pads at both ends, loaded in the middle of its
  top: 2D reaches compliance 41.9 N·mm in 29 s (0.22 mm elements), 3D 41.6 in 23 s (0.69 mm).
  The bracket (`bracket_domain.wgsl --2d y`) is the other kind: its bolt holes run along x, so in
  2D the plate is clamped across its whole width, and the part can't be a closed 3D box; it ends at
  14.5 N·mm against 7.9 in 3D.

### Supports and loads from primitives

Instead of (or as well as) writing the fields, mark [primitives](#primitives-from-the-app-nameprimsjson)
in the Design step with **Optimize as**:

| role | effect |
|---|---|
| *design* (default) | part of the shape like any other: optimized |
| *keep solid* | its inside is forced solid (a mounting plate, a boss) |
| *fixed* | clamped: nodes inside it **or within a thin shell around it** don't move |
| *load* | its region (same rule) carries **force** N along **direction**, in **load case** N |

The shell is 1.5 elements thick (`g3d_reach`, set from the optimizer's element size). It makes the
natural thing work: a **subtracted** cylinder (a bolt hole) marked *fixed* clamps the hole's wall, and a
subtracted pin hole marked *load* loads its wall. Fixed and loaded regions are also kept solid, since a
bolt or pin needs material around it. A primitive whose operation is **region** marks a spot without
changing the shape, e.g. a box over the face that rests against a wall. Giving an *add* primitive the
*fixed* or *load* role switches it to *region* (and back to *add* when the role goes back to *design*
or *keep solid*). Set it to *add* again on purpose to make the region part of the shape too: its
whole volume is then kept solid, even where it sticks out of the design.

**Load cases:** loads with the same case number act together. Different cases are separate situations
(a bracket pulled down *or* pushed sideways), and the part is optimized for all of them: the objective
is the sum of their compliances, each case solved on its own. All cases share the supports, so a case
whose load is a linear combination of other cases' loads isn't solved: its displacement is the same
combination of theirs. Rocking gives 5 cases (upright, ±X, ±Y) but only 3 solves. Each load
primitive's force is spread over its own nodes. A script's `field_load` is load case 1 and uses
`--force` (the app's "script load"). The result has `displacement` for the first case and `displacement_<case>` for the others.

**Ground** (`--ground`, *Stands on the ground* in the app): the bottom layer of the design domain
rests on a table or floor, and the optimizer picks where the part touches down.
- `slide`: the ground pushes up only and the part is free to slide and spread sideways. A few small
  patches (4 mm radius) are pinned sideways so it can't slide or spin away.
- `grip`: where the part touches down it is held in all directions (no slip).
- `glued`: held in all directions everywhere, and the ground can pull (glued, screwed or bolted).

`slide` and `grip` are contact: the ground only pushes. Each load case has its own set of nodes
touching down: after a solve, nodes the ground would have to pull down lift off, lifted nodes that
sank below the ground touch down again, and the case is solved again until the set settles (at most
8 solves per case and iteration; each iteration starts from the last one's sets, so after the first
few most cases take one solve). The iteration line shows the solves and the touching nodes per case.
With contact every case is solved (none is a combination of others), and mirror symmetry is used only
when every case is symmetric. Before running, each case is checked against tipping: for each
separate piece of the domain, its load must press on the ground inside the piece's footprint (the
centre of pressure inside the convex hull of its ground nodes), or the run stops with the case and
the piece. A stand made of separate legs needs each leg to stand on its own. *The GPU solver doesn't
do contact yet: it holds the ground both ways, as `glued`, with a note.*

Script and primitives combine: when primitives use a role, the script's own `field_fixed`,
`field_load` or `field_keep` (if any) is renamed and wrapped, so the field is the script's region plus
the primitives'. Overlays, `export --format vti` and the Optimize step see the combined fields.
`shapes/bracket_editor.wgsl` is the bracket built this way: a domain box, a kept plate, two fixed bolt
holes, and the pin hole loaded down (case 1) and sideways (case 2). For both cases the optimizer finds a
hollow, closed box section; a closed section is what stands up to the sideways load's twist.

In the app, the **Optimize** step runs the same thing in the background and shows the result in the
viewport while it is still running. The file is rewritten as it runs (atomically, in the background,
at most ~10% of the time; every iteration on small grids), so you watch the design form. On grids
over 4M elements the live file is a preview: the density averaged over k×k×k blocks. The full fields
come at the end. **Stop** finishes the current iteration and writes its full result (stopping twice
ends it at once).

A run stopped before it converges also leaves a checkpoint next to the result (`NAME_topopt.vti.resume`:
the design variables, β, the iteration count and the stopping rule's state). **Resume** in the app, or
`--resume` with the same settings on the command line, continues the run where it would have gone on:
the iterations and the final design match an uninterrupted run. A different shape or setting is
refused with the difference shown; `--iters` (still counted from the run's start), the solver and its
tolerance may change. A run that finishes, or a fresh one (**Start over**), removes the checkpoint.

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
  - It stops at the final β when compliance changes by less than 0.1% for 3 iterations, or doesn't
    improve on its best by 0.1% in 3 iterations (at β = 32 the updates tend to oscillate around a
    level), or at `--iters` (150).
  - **Coarse start:** the β = 1 stage, where the layout forms, runs on a grid of twice the element
    size first (1/8 of the elements) until it settles. The fine grid then continues from that design
    at β = 2. This is 15–30% faster on most problems, with compliance within about 1% (sometimes
    better). It applies from 80k domain elements; `--no-coarse-start` turns it off. The output file
    shows the coarse design first. On the GPU the fine stage keeps only the part of the domain
    within a few voxels of that design (`continuing the design on N of M bricks`); the rest stays
    void, which saves memory and time (1.4× per iteration on the bracket, 3× on a 2D table).
  - **Symmetry:** with [mirror planes](#primitives-from-the-app-nameprimsjson) the grid is aligned so
    each plane is a node layer. When the domain, kept regions and supports are symmetric about them,
    only one side is solved (1/2, 1/4 or 1/8 of the elements) and the design is symmetric. The loads
    may be asymmetric (rocking, a sideways push): each case is split into parts that are symmetric
    or antisymmetric about each plane, and each part pattern is solved on the reduced grid with its
    own plane condition (the displacement across the plane held, or along it). The full fields are
    written as usual; the energy is averaged over mirror images. Same compliance per iteration as
    solving the whole part; bracket (1/2) 9.1 → 7.4 s, table (1/4) 15.9 → 12.2 s. It applies from
    100k domain elements (below that each pattern's fixed costs outweigh the smaller grid);
    `--no-symmetry` solves the whole part (on the same aligned grid). A problem that isn't symmetric
    after all (a script's own supports, `--ground` with a Z plane) is solved whole, with a note.
- **Overhang control** (on by default; `--no-overhang`, or untick *Printable without supports* in
  the app, turns it off): the result is printable without support material with the `--up` side up
  (+z by default: as the shape stands). No surface faces down more than 45° from vertical. The design
  domain's first layer along that direction is on the build plate (the lowest one for +z, the highest
  x for -x). This is Langelaar's additive-manufacturing filter (2017): going up layer by
  layer, an element is only as dense as a smooth maximum of the 5 elements under it (below, and the 4
  beside that one) allows. It sits between the density filter and the projection, and the optimizer
  differentiates through it, so it grows 45° chamfers and arches instead of ledges rather than having
  them cut off afterwards. `optimize` ends with the count of unsupported solid elements (0 for the
  optimized material; a few at the boundary are possible).
  - Kept-solid regions are printed as given: one that overhangs (a horizontal hole's ring arching
    over the hole, a tabletop marked keep solid) still needs supports or bridging, and is counted
    separately. Material can stand on them.
  - **Print orientation** (`--up`, *printed … up* in the app): any of the six axis directions, so the
    layers stay voxel layers. It is saved with the shape: the app writes the choice to an editor
    shape's `.prims.json` (`"up": "-x"`); a script shape takes `//! up -x` (the app says so when you
    change it). `optimize` without `--up` uses the shape's, and records it in the result (FieldData
    `print_up`, the up vector). `export`, `slice` and `open` without `--up` turn the mesh so the
    saved side (the shape's, or the one a result was made for) faces up and set it on z = 0; `--up=+z`
    keeps it as modelled. In the app Optimize and Print share the setting, and Print warns when a
    result was optimized for another side up. Other angles: rotate the shape. The orientation matters: the bracket printed on its wall plate (`--up=+x`) reaches
    compliance 7.83 in 58 iterations (8.5 s), against 7.88 in 115 (16.5 s) standing as modelled.
  - 45° is along the layer's two axes; diagonally the limit is a little steeper (about 55°). Bridges
    are not allowed either.
  - A mirror plane across the build direction isn't used (printing breaks that symmetry); the others
    are.
  - Cost: per iteration a few percent (two sweeps over the layers, one up and one down). The runs
    tend to take more iterations.
- **Projection or not:** with projection the result is crisp (0% grey on the bracket against 24%
  without) and stiffer for the same material (compliance 8.96 against 10.36 N·mm). Thresholding at
  0.5 then matches `--volfrac` closely. The cost is about 1.7× the iterations. Because each voxel is now
  fully in or out, the threshold surface follows the voxel staircase; `export` fairs it (see Meshing
  above), and a finer `--voxel` gives a smoother part. `--no-projection` gives the faster, softer-edged grey result.
- **Speed:** each iteration solves the stiffness system with conjugate gradient preconditioned by
  geometric multigrid (a V-cycle with Chebyshev smoothing and Galerkin coarse levels; the fine level
  is matrix-free), warm-started. With a hardware GPU (`--solver gpu`, the default) the whole iteration
  runs there in f32: solve, sensitivities, filter, projection and the OC update. Only a few numbers
  come back per iteration, plus the fields when the `.vti` is written. Without one, `--solver mg` runs
  the same on the CPU in f64, multithreaded. Solves are looser while the design still moves a lot and
  reach full accuracy at the final β. CG needs about 8–13 iterations per solve at any grid size.
  - The default element size gives about 300k elements in the bounds with the GPU solver (the bracket:
    0.64 mm, 50 iterations in 2.7 s) and 40k with the CPU ones (1.3 mm, 7.6 s). It can be finer than
    the shape's `//! voxel`, which is only its meshing and preview resolution.
  - The grid is cropped to the design domain plus one empty cell (symmetric about mirror planes), so
    bounds larger than the part cost nothing in the solve.
  - Per iteration on an RTX 3080: 0.14 s at 1.2M elements, 1.0 s at 9.5M, 2.7 s at 22M, 5.6 s at
    44M.
  - CPU multigrid: 7.7 s at 1.2M; the old Jacobi-preconditioned CG (`--solver jacobi`): 76 s.
  - The GPU path needs ~160 B per stored element (`optimize` prints it per domain element): it
    stores only the 4×4×4 bricks around the domain, so a part filling a fraction of its box costs
    about that fraction. A 10 GB card holds about 45M elements of a box-filling part. Past that,
    `optimize` says it doesn't fit. The CPU solver allows 8M (~1.2 kB each).
- **Shortcuts in this version:**
  - Overhang control: 45° only, and the print direction is one of the six axis directions.
  - 2D: the profile is extruded through the domain's thickest part; no per-column thickness, no
    axisymmetric (revolved) profiles.

## Mesh quality notes

- Meshes are closed by construction (the grid is padded one voxel past `bounds`, and the marching-cubes
  case table is generated so neighbouring cells always agree on ambiguous faces). `export` prints
  `watertight=` and the volume; if a part touches the bounds it is cut flat there.
- Export resolution is `voxel`: features thinner than ~2 voxels may disappear or break apart. 0.1–0.2 mm
  is plenty for FDM; triangle counts grow with 1/voxel².
- Non-exact SDFs (smooth booleans, twist, noise) are fine for meshing; only the preview's raymarcher cares.
  Scale the field down (`d *= 0.7`) or set `//! step`.
- Optimized results often have a few tiny voids inside or crumbs beside the part (`bodies=` above 1);
  `--min-body 1` removes them.
- `--decimate` uses meshoptimizer's quadric simplifier. If a collapse pinches a thin feature, the area is
  locked and simplification is redone; if the result still isn't watertight the full mesh is kept.

## Development

```
cargo test            # marching-cubes table and watertightness, shape loading / error mapping
cargo clippy --all-targets
gen3d snapshot shapes/bracket_domain.wgsl --step optimize -o out/ui.png   # the app's window, offscreen
gen3d bench-topopt [--voxels 1.3,0.8,0.5,0.4] [--iters 5] [--solver mg|jacobi]  # optimizer timing table
```

`bench-topopt` (hidden) runs `optimize` on the bracket at several element sizes and prints the mean
time per iteration, split into preconditioner setup, solve, update (filter + OC) and file output, with
CG iterations per solve. `optimize` prints the same split on each iteration line. Hidden solver flags
for experiments: `--smoother chebyshev|jacobi`, `--sweeps`, `--omega`, `--emin`, `--cg-tol`,
`--fixed-cg-tol`.

GPU kernel tuning: the optimizer's kernels take a few hardware-dependent choices from a profile
(`src/topopt/tuning.rs`), picked by GPU vendor: `nvidia` (measured on an RTX 3080, the primary
target), `apple` (the M1 Pro settings) or `generic`. `GEN3D_GPU_PROFILE=nvidia|apple|generic`
forces one; `GEN3D_GPU_TUNE="key=value,..."` overrides single fields (e.g.
`GEN3D_GPU_TUNE=ke_uniform=0`). `bench-topopt` prints the profile in use. For the GPU time of each
kernel in a CG iteration and the multigrid setup (timestamp queries), run
`cargo test --release profile_kernels -- --ignored --nocapture` (`PROFILE_BY_SIZE=1` splits each
kernel by dispatch size, i.e. by level).

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
| `src/prims.rs` | editor primitives: `.prims.json`, generated WGSL (add / subtract / intersect), bounds, CPU SDFs for picking |
| `src/gpu.rs` | adapter selection (hardware first, software fallback with a warning) |
| `src/mesher.rs` | compute-shader sampling harness, grid padding, polygonize + report |
| `src/mc.rs` | marching cubes with a generated crack-free case table |
| `src/mesh.rs` | volume / watertight / bodies (and dropping small ones), decimation, STL/3MF/OBJ/PLY writers |
| `src/viewer.rs` | raymarching preview harness and pipelines, headless PNG render |
| `src/slicer.rs` | PrusaSlicer / OrcaSlicer discovery and CLI driving, G-code summary, profiles, opening files |
| `src/vti.rs` | VTK ImageData reader and writers (per-axis spacing) |
| `src/volume.rs` | .vti grids as viewable shapes: distance fields, EDT, volume WGSL, meshing (domain faces, outlines), the faired SDF for imports |
| `src/import.rs` | `//! import`: .vti grids as SDF functions in a shape (parsing, cached fields, generated WGSL, the `g3d_imp` buffer) |
| `src/fairing.rs` | smooth surfaces from binary voxels: constrained thin-plate fairing of the field (`--grow` widens the band outward) |
| `src/topopt/mod.rs` | topology optimization: hex element, matrix-free PCG, filter, OC update |
| `src/topopt/mg.rs` | multigrid preconditioner: Galerkin coarse levels, Chebyshev/Jacobi smoothers, banded Cholesky |
| `src/topopt/gpu_mg.rs`, `mg.wgsl` | the same MGCG on the GPU (f32), on block-sparse grids |
| `src/topopt/gpu_opt.rs`, `opt.wgsl` | the optimizer loop on the GPU: stiffness, sensitivities, filter, AM filter, projection, OC |
| `src/topopt/gpu_engine.rs` | GPU kernels as recorded dispatch programs, reductions, scalars, readback, per-kernel timestamp profile |
| `src/topopt/tuning.rs` | hardware profiles for the GPU kernels (NVIDIA, Apple, generic) and their env overrides |
| `src/topopt/bricks.rs` | block-sparse grids for the GPU optimizer: 64-point bricks around the domain, page tables, gather/scatter, coarsening |
| `src/topopt/output.rs` | writing the result as it runs: background writes, decimated previews, full result on Stop |
| `src/topopt/overhang.rs` | overhang control: the AM filter (layer-by-layer smooth min/max), its adjoint, the unsupported-element count |
| `src/topopt/plane2d.rs` | 2D optimization: projecting the problem into a one-element slab, extruding the result back onto the 3D grid |
| `src/topopt/symmetry.rs` | mirror symmetry: the reduced model (one side), load parts per symmetric/antisymmetric pattern, expanding the fields back |
| `src/topopt/contact.rs` | a ground that only pushes: the per-load-case set of touching nodes (active set), updated from reactions and displacements |
| `src/topopt/checkpoint.rs` | stop and resume: the `.resume` checkpoint next to the result |
| `src/wgsl/sdf.wgsl` | the SDF library |
