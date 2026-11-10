# gen3d: procedural SDF workbench for 3D printing

Write shapes as GLSL signed distance functions, preview them live (with hot reload), mesh them on the GPU,
and slice them to G-code.

```
shape.glsl ──► gen3d view    live raymarched preview, reloads on save
           ──► gen3d export  GPU compute-shader sampling → marching cubes → watertight STL/3MF
           ──► gen3d slice   export + PrusaSlicer/OrcaSlicer CLI → .gcode
           ──► gen3d open    export + open in the slicer GUI
```

## Quick start

```sh
uv run gen3d new mything            # creates shapes/mything.glsl from a template
uv run gen3d view shapes/mything.glsl   # open the preview, then edit the file in your editor
uv run gen3d export shapes/mything.glsl # -> out/mything.stl
uv run gen3d slice shapes/mything.glsl  # -> out/mything.gcode (needs a slicer, see below)
```

Slicer: `sudo pacman -S prusa-slicer` (full CLI slicing support). Then export your printer profile to
`printers/default.ini`; see [printers/README.md](printers/README.md).

## Writing shapes

A shape file defines `float map(vec3 p)`: the signed distance (mm) from point `p` to the surface,
negative inside. Coordinates are **millimetres, +Z up, build plate at z = 0**.

```glsl
//! bounds -30 -30 0  30 30 40     // export region: xmin ymin zmin  xmax ymax zmax
//! voxel 0.2                      // mesh resolution (mm); smaller = finer & slower
#include <sdf.glsl>

#ifndef R                            // parameters: override with -D R=15.0
#define R 12.0
#endif

float map(vec3 p) {
    float body = sdRoundBox(p - vec3(0, 0, 15), vec3(20, 20, 15), 3.0);
    float ball = sdSphere(p - vec3(0, 0, 30), R);
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

`#include "file.glsl"` resolves relative to the shape first, then the built-in library dir.
Compile errors are reported as `yourfile.glsl:LINE`.

### Library (`src/gen3d/glsl/sdf.glsl`)

- **Primitives:** `sdSphere sdBox sdRoundBox sdTorus sdCylinder sdRoundCylinder sdCapsule sdCappedCone sdHexPrism sdOctahedron sdPlane`
- **2D → 3D:** `sd2Circle sd2Box opExtrude opRevolve`
- **Booleans:** `opUnion opSubtract opIntersect opSmoothUnion opSmoothSubtract opSmoothIntersect opChamferUnion`
- **Modifiers:** `opRound opShell` (hollow with wall thickness)
- **Domain:** `rotX rotY rotZ rot2 deg opTwist opBend opRepeat opRepeatLim opPolar opMirrorX opMirrorY`
- **Lattices:** `sdGyroid sdSchwarzP` (use with `opShell` + `opIntersect` for TPMS infill)
- **Noise:** `hash13 valueNoise fbm`

Distance functions are mostly from [Inigo Quilez](https://iquilezles.org/articles/distfunctions/).

## Viewer controls

| | |
|---|---|
| left drag / right drag / wheel | orbit / pan / zoom |
| `1` `2` `3` `4` | front / side / top / iso view |
| `C`, `[` `]` | toggle cross-section, move the cut plane (the section face shows internal distance bands) |
| `B` / `V` | toggle clipping to export bounds / bounds box |
| `E` / `G` / `O` | export mesh / slice to G-code / open in slicer GUI (runs in the background) |
| `R`, `F5`, `Q` | reset camera, force reload, quit |

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

## Mesh quality notes

- Meshes are closed by construction (the grid is padded one voxel past `bounds`). `export` prints
  `watertight=` and the volume; if a part touches the bounds it is cut flat there.
- Export resolution is `voxel`: features thinner than ~2 voxels may disappear or break apart. 0.1–0.2 mm
  is plenty for FDM; triangle counts grow with 1/voxel².
- Non-exact SDFs (smooth booleans, twist, noise) are fine for meshing; only the preview's raymarcher cares.
  Scale the field down (`d *= 0.7`) or set `//! step`.
- `--decimate` is skipped automatically if it would break watertightness (thin lattice sheets).
