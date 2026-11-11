# gen3d devlog

Newest entries first. Each entry covers what changed and why, the design decisions, how it was verified,
and what is still open.

## 2026-11-12: Topology optimization MVP (`gen3d optimize`)

**What:** minimum-compliance topology optimization with SIMP, in Rust inside gen3d (`src/topopt.rs`).
The problem is a shape file: `map` is the design domain, with `field_fixed`, `field_load` and the
optional `field_keep` / `field_void`. The result is a `.vti` (cell `density`, cell `energy`, point
`displacement`), rewritten atomically after every iteration so `gen3d view` shows the design forming.
The viewer has a Topology optimization section (volume fraction, load, voxel, filter radius,
iterations), and jobs now have **Stop** and a **View** button that works while they run.

**Why in gen3d (not a Python script):** the problem already lives in the shape file and the GPU samples
it. The live viewer and `.vti` → STL export were already in place, so `optimize` closes the loop from
problem to printable part in one binary.

**Design (shortcuts are deliberate for an MVP):**
- **Discretization:** one trilinear 8-node hex per voxel. The element stiffness matrix is computed by
  2×2×2 Gauss for a unit cube with E = 1 (`hex_ke`); a voxel of size h with modulus E has `E·h·ke`.
  Real units (mm, N, MPa), PLA-like defaults.
- **Sampling:** domain, keep and void at element centres; fixed and load at nodes. All on the GPU with
  the existing `sample_field`. Nodes outside the domain are dropped entirely, not given ρ_min, which
  keeps the system smaller and better conditioned.
- **Loads:** `field_load` vectors scaled so their magnitudes sum to `--force`.
- **Solver:** matrix-free K·u gathered per node (each node sums rows of its ≤8 elements), parallel with
  rayon. Jacobi-preconditioned CG, tolerance 1e-4, warm-started from the previous displacement.
  E_min = 1e-6·E0, a compromise for Jacobi conditioning.
- **Filter and update:** density filter (weights rmin - dist over domain elements, kept-solid elements
  pinned to 1, sensitivities by the chain rule). OC update with move 0.2 and a bisection on the
  multiplier for the volume of the design elements.
- **Stopping:** max change < 0.01, or compliance improving < 0.1% for 3 iterations. The max change
  hovers at the move limit (a few oscillating elements) long after compliance has settled, so the second
  rule is what usually triggers.
- **Also changed:**
  - The automatic colour range is now the 1st–99th percentile over inside samples; with min..max,
    energy hotspots at the bolts made everything one colour.
  - `vti::write_grid`: a general, atomic (tmp + rename) writer for point and cell data, with cell arrays
    first so the viewer opens on `density`.

**Verified:**
- 12 tests:
  - `hex_ke` symmetric with zero rigid-body translations.
  - A 1×1×4 column in tension with ν = 0 matches F·L/(E·A) to 1e-9, which validates the element and
    the solver together.
  - `write_grid` round trip.
- bracket_domain at 1.3 mm: 62×19×31 grid, 34k domain elements, 200 clamped and 399 loaded nodes.
  - Compliance falls from 96 to 10.4 N·mm (max displacement at 100 N: 1.17 → 0.124 mm).
  - Stops at iteration 29 in 52 s, about 1.8 s per iteration with roughly 1000 CG iterations each.
  - The design is a sensible cantilever truss: a tension chord to the pin boss, a compression strut, a
    web, and the plate.
- Thresholded at 0.5 and exported: watertight, one body, 27.6 cm³ against about 28.6 expected.
- The GUI's argument string was run from the CLI.

**Not verified:** clicking Optimize / Stop / View in the window. The screenshot meant for that caught
the user's active browser window instead of the viewer (they were using the desktop), so it was deleted;
UI checks now use headless renders unless the user is away.

**Next:**
- Heaviside projection with β continuation, for crisp 0/1 designs.
- A multigrid-preconditioned CG, for finer grids.
- Symmetry planes, several load cases, and an overhang constraint for printability.
- Use the optimized density inside a `.wgsl` shape (smooth it, add exact holes).

## 2026-11-12: .vti features in the egui sidebar

**What:** the sidebar now covers the `.vti` workflow in both directions.

- **Volume panel (viewing a grid):**
  - **Histogram** of the shown array over its range. Counts are log-scaled, so a SIMP density's
    0/1 peaks and its grey middle are both visible. Inside bars are drawn in the surface orange, and a
    line marks the threshold. Click or drag on the histogram to set the threshold; like the slider, it
    applies on release, since the distance field is recomputed on the CPU.
  - **Inside volume:** samples and % of the grid (the volume fraction), cm³ (inside samples × cell
    volume), and grams of PLA at 1.24 g/cm³ and 100% infill.
  - **Colour bar** (viridis, the same polynomial as the shader, in Rust), with the range under it.
    *Fixed range* plus two drag values override the automatic min..max over the inside samples, to
    compare iterations or ignore hotspots. It's kept across live reloads (`VolumeOpts::color_range`).
  - Smooth/blocky and colour-range edits upload only the uniform buffer (`edit_volume_uniforms`), not
    the grid buffers.
- **Grid section** (replaces Shape for `.vti`): point dims, spacing, first sample of the shown array,
  shown samples, and a table of every array (point/cell, components, range), as in `gen3d check`.
- **Output panel (making a grid from a shape):**
  - **Export VTI** runs `export --format vti --voxel V` with the current `-D` values.
  - The voxel defaults to the shape's voxel coarsened to about 1M samples, rounded up to 2 significant
    digits, because a print voxel (0.12 mm for the knob) makes a needlessly huge simulation grid.
  - The sample count, MB and array names are shown before exporting.
  - When it's done, **View** spawns `gen3d view out/x.vti` as a second window, taking the path from the
    job's last `wrote PATH (...)` line, so domain and result sit side by side.
- `run_bg_with` passes extra arguments; the job label shows them.

**Verified:**
- clippy clean; 9 tests, with new ones for `written_path` and the default VTI voxel.
- Screenshots (spectacle, RTX 3080, 165 fps):
  - density test grid: Volume panel with histogram, stats (3.67 cm³ as before) and stress colour bar;
    blocky mode; Grid section with cell-data offset and arrays table.
  - bracket_domain: Export VTI row, whose estimate of 81×25×41 = 83,025 samples, 2.0 MB matches the
    actual export.
- The exact command the button runs was checked from the CLI.

**Open:** clicking (histogram drag, fixed-range values, Export VTI → View) wasn't exercised by hand,
because KDE asks before allowing synthetic input. The histogram uses one bin layout per array, so after a
live reload with a new value range the bins rescale; fine for watching an optimizer, but it isn't a
fixed axis across iterations.

## 2026-11-12: egui sidebar for the viewer

**What:** the viewer gets a real UI instead of keys plus the window title. A left sidebar has
parameters, view options, field overlays, the `.vti` volume controls, output jobs, shape info and a key
reference. Shader errors show in a panel under the view. All keyboard shortcuts still work.

- **Parameter sliders:** `Shape::params()` reads each `override` with naga: its name, type
  (f32/i32/u32/bool) and default (a literal, possibly negated). A trailing `// comment` becomes the
  tooltip. The new directive `//! range NAME min max` sets the slider range; otherwise it's 0..2×default.
  The viewer keeps only values that differ from the defaults, as `-D` pairs, so export file names and
  **Copy -D flags** stay clean. An unknown `-D` in the viewer is now a dismissible warning, not a dead end.
- **Cheap rebuilds:** `build_pipeline` is split from `preview_module`, so a parameter change reuses the
  validated shader module and only recreates the pipeline with new constants. That still costs ~75 ms of
  driver compile (bracket_domain on an RTX 3080), so it runs on a worker thread (`Gpu` is now in an
  `Arc`). Changes made meanwhile collapse into one follow-up build, and a generation counter drops
  builds made stale by a reload. A failed build for a parameter value is reported separately from
  source errors, so it can't hide one.
- **Rendering:** the raymarch goes into an offscreen `Rgba8Unorm` texture sized to the central panel
  (times a resolution slider), registered with `egui_wgpu::Renderer` as a native texture. egui samples
  textures as gamma-encoded, so the shader's own gamma is applied, as for `render`. Camera input comes
  from the egui response (drag, wheel, double-click to reset).
- **Jobs:** export/slice/open still re-run the binary. Its stdout and stderr are now captured (ANSI
  stripped, `\r` progress collapsed), shown in the Output section with a spinner, status and timing,
  and still echoed to the terminal.

**Why egui and not gpui:** gpui (0.2.2) renders through its own blade/Vulkan layer on Linux. Its
`surface` element is macOS-only, so our wgpu raymarch could only reach it via a GPU→CPU→GPU copy every
frame. egui-wgpu 0.37 is on wgpu 30 and egui-winit on winit 0.30, the same versions we use, so it shares
our device and plugs into the existing `ApplicationHandler`.

**Gotchas:** egui 0.37 renamed things (`Context::run_ui`, `Panel::left`, `show` instead of
`show_inside`). Fonts are behind the `default_fonts` feature (without it every glyph is a box). A dropped
`TexturesDelta` panics in debug builds, so texture uploads are applied before any early return. Symbols
like ● ⟳ ↺ ✔ aren't in the default fonts, so status dots are painted and buttons use words. Sizing
sliders from `available_width()` inside a resizable panel feeds back and grows the panel each frame.

**Verified:** clippy clean; 7 tests pass, including a new one for `params()` (types, negative default,
note, range, slider range). By screenshot on the RTX 3080: knob, bracket_domain (fields + unknown `-D`
warning), a broken shape (error panel, dark red view, disabled export) and a 0.2M-sample `.vti` (Volume
panel, colour by `keep`), all at 165 fps. `render` still writes PNGs.

**Open:** slider dragging, the threshold slider and the Output buttons weren't exercised by hand. KDE
asks for permission before synthetic input, so they need a manual try. The `.vti` threshold slider
applies on release because the CPU recompute can take a second on big grids. Making parameters uniforms
instead of override constants would make dragging instant, but overrides can size arrays, so that needs
a naga-level rewrite.

## 2026-11-12: View and mesh .vti grids

**What:** `view`, `render`, `export`, `slice`, `open` and `check` accept `.vti` files. The goal is to
inspect what a topology optimizer (or any simulation) writes back, and to turn its density into a
printable mesh.

- **Reader (`src/vti.rs`):**
  - Any scalar or vector array, point or cell data.
  - Raw, base64 or ascii encoding, optionally zlib-compressed, with UInt32 or UInt64 headers.
  - Small hand-written tag scanner; no XML dependency. New dependencies: `flate2`, `base64`.
  - Gotcha: VTK puts `<InformationKey>` elements inside vector `DataArray`s after the data, so inline
    data ends at the next `<`, not at `</DataArray>`.
- **Viewer (`src/volume.rs`):** a grid becomes a generated shape whose `map()` reads storage buffers
  (bindings 1-4). The whole raymarching harness is reused: section view, bounds, shadows, `render` for
  headless checks, hot reload. Reload keeps the chosen array, threshold and colour, so an optimizer
  rewriting the file each iteration can be watched live.
  - **Smooth mode:** a signed distance to the threshold surface, computed on the CPU. Samples next to the
    surface get `(v - iso)/|∇v|`; the others get `EDT(other side) - cell diagonal`, a lower bound, so
    sphere tracing doesn't overshoot. Sampled trilinearly in the shader, with raymarch step factor 0.8.
  - **Blocky mode:** exact distance to the cubes in the 3×3×3 neighbourhood, plus a conservative far
    field: EDT to the inside samples, minus the offset to the nearest sample, minus half a cell diagonal.
    Its zero set is exactly the union of the cubes.
  - **EDT:** separable exact Euclidean distance transform (Felzenszwalb & Huttenlocher) with per-axis
    spacing; tested against brute force.
  - **Colour by array:** viridis colour map, range taken over the inside samples. It goes through the
    harness's existing `g3d_field_tint` hook, which the volume WGSL defines itself.
- **Defaults:** an SDF-like array (named *sdf* or crossing 0) is inside below 0. Anything else is inside
  above the middle of its range. Keys: `-`/`=` threshold, `M` smooth/blocky, `I` flip, `A` next array,
  `F` colour. Command line: `--array --iso --below/--above --blocky --color`.
- **Export:** marching cubes on `±(v - iso)`, padded to close, meshed in grid units and scaled per axis
  for anisotropic spacing. Decimation is shared with shape export (`mesher::decimate_checked`).

**Why raymarching and not a mesh:** reusing the harness gives the section view, bounds, shadows,
`render` and hot reload for free. The cost is CPU preprocessing on each threshold change.

**Verified:**
- **Files read:** gen3d's own export; pyvista zlib+base64 (its default), uncompressed and ascii files.
  LZ4 gives a clear error.
- **Renders:**
  - Smooth and blocky.
  - Section view.
  - Colour by a vector array (magnitude) and by a scalar.
  - Cell-data density with point-data stress.
  - Anisotropic 0.5/1/2 mm spacing.
- **Mesh volumes:**
  - Density test: 3.50 cm³ for 3,672 inside voxels of 1 mm³.
  - Anisotropic sphere: 14.07 cm³ against an exact 14.14.
  - Bracket SDF round trip (`.wgsl` → `.vti` → STL): 75.45 cm³, the box minus its holes.
- **Speed:** a 200³ (8M sample) grid reads in 0.2 s; a full render including preprocessing takes 1.7 s.

**Open / next:**
- Not tried in the interactive window (no display here): keys, live reload while a file is rewritten,
  frame rate on big grids. Headless renders use the same shader and code paths.
- A threshold change on 8M samples recomputes for about 1 s on one core; the EDT and band loop could be
  parallelized if it gets annoying.
- Blocky export (voxel cubes as STL) isn't implemented; `export` meshes the smooth surface only.
- Grids below z = 0 are hidden by the build plate, as with shapes.
- Possible next: sample a `.vti` inside a `.wgsl` shape (as a 3D texture), so an optimized density can be
  combined with exact holes, fillets and infill.

## 2026-11-12: Viewer flicker fixes (`95b978d`)

The bounds-box edges were drawn only where the box lay strictly in front of the surface. When a shape
fills its bounds exactly (a design domain), the edges lie on the surface and that test flipped from pixel
to pixel, which looked like z-fighting. Now a small depth tolerance applies. The field-overlap stripes
moved from screen space to world space, so they no longer slide over the surface while orbiting. Checked
with headless renders at nearby camera angles.

## 2026-11-12: Field overlays in the viewer (`052e46a`)

`field_*` regions are tinted on surfaces and the section face:
- scalar fields where they are `< 0`;
- vector fields where they are non-zero, with bands moving along the vector;
- overlapping regions as alternating stripes.

The lookup is half a voxel inside the surface, so shells around holes (`max(d - t, -d)`) show on the hole
wall, matching where the exported grid samples them. `F` cycles the overlays; `render --no-fields` hides
them.

## 2026-11-12: VTI export and `field_*` functions (`c0f4bb4`)

`export --format vti` writes the sampled SDF as VTK ImageData for simulation and topology optimization.
Each `fn field_<name>(p) -> f32|vec3f` is sampled on the same grid and written as an extra array, so one
`.wgsl` file can describe a whole problem: domain, supports, loads, non-design regions. The example is
`shapes/bracket_domain.wgsl`. The GPU sampler was generalized to evaluate any f32/vec3f function.
