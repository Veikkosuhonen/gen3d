# gen3d devlog

Newest entries first. Each entry covers what changed and why, the design decisions, how it was verified,
and what is still open.

## 2026-11-12: New shapes start empty

**What:** the app's **New shape** writes an empty `.wgsl` and opens it, without starting the external
editor. You build it from primitives, or click **Edit** to script it. An empty Design view says so
("Empty shape: add a primitive…"), and the Parameters how-to is hidden for primitives-only shapes
that have no parameters. `gen3d new NAME` on the command line still writes the scripted template (in a
terminal you're scripting anyway).

**How:** a `.wgsl` with **no code** counts as an editor shape even before its `.prims.json` exists.
No code means only comments (`//`, `/* */`), `//!` directives and `#include` lines (`has_code`). The
sidecar is created on the first Add. A file that has code but no `fn map` is still an error. The check
now runs before parsing, so it says "no `fn map` found" instead of naga's "unknown identifier `map`" from
inside the harness.

**Verified:** `cargo test` (24 pass): an empty file and a comments/directives/include-only file both
load as empty editor shapes and compile with the mesher and preview harnesses. Code without `map` still
gives the clear error. A `gen3d snapshot` of a fresh empty shape shows the hint and the Add buttons.
clippy clean.

## 2026-11-12: Supports, loads and load cases from editor primitives

**What:** a primitive has an **Optimize as** role: design (default), keep solid, fixed or load. Load
primitives have a force (N), a direction and a **load case**. A new operation, **region**, marks a
spot without changing the shape. The Optimize step lists the supports and loads; it is available as
soon as primitives give a fixed and a load region. `shapes/bracket_editor.wgsl` (+ `.prims.json`) is the
bracket set up this way, with two load cases.

**Design decisions** (aiming at "mark the feature, the way CAD does it")
- **Constraints go on features.** You fix *the bolt hole* and load *the pin hole*. So a role's region
  is the primitive's inside **plus a shell** `g3d_reach` thick around its surface. For a subtracted
  cylinder that is exactly its hole's wall, which is what `bracket_domain.wgsl` writes by hand
  (`max(bolts - WALL, -bolts)`). For an added box it is the box. `g3d_reach` is an override set by
  `topopt::setup` to 1.5 × the element size (≥ 1 layer of nodes at any voxel). The preview default is
  1.5 × the auto element size.
- **Region operation** for marking without geometry: a box over a face, a sphere on a spot. Invisible
  like a cutter, with the ghost when selected.
- **Supports and loads are kept solid** (the BC shells go into `field_keep`): a load on near-void
  elements gives degenerate SIMP results, and a printed bolt hole needs a boss. The bracket script does
  the same with its own keep regions.
- **Per-load force in N.** The old single `--force` spread over all of `field_load` can't express "100 N
  here, 50 N there", so each load primitive's force is spread over its own nodes. A script's
  `field_load` keeps `--force`.
- **Load cases** are real multi-load optimization: each case is solved separately per iteration
  (`K u_c = f_c`). The objective is Σ compliance, so the sensitivities sum the element energies. That is
  the standard formulation, and it differs from adding the loads (a down load and a sideways load
  shouldn't cancel or combine into one diagonal). Cost: one solve per case per iteration.
- **Script + editor combine** like `map`: a role used by primitives renames the script's `field_fixed`
  / `field_load` / `field_keep` to `g3d_script_*` and generates the union. Overlays, VTI export and the
  "optimizable" check then work unchanged. `Shape::load_specs()` gives the optimizer the loads: the
  script's field (vector, case 1) plus one per load primitive (region < 0, direction, force, case).
- Data per primitive grew to 4 vec4s (the 4th = load direction, force), so editing a load's direction
  updates the preview's load overlay without compiling. Role changes regenerate the source (they change
  which functions exist). Old `.prims.json` files load with role design, force 100 N, −Z, case 1.

**Verified:** `cargo test` (24 pass).
- `topopt::setup` on a bar with a fixed subtracted bolt hole, a loaded subtracted pin hole (50 N, −Z,
  case 1) and a region sphere (30 N, +Y, case 2) gives two cases summing to exactly 50 and 30 N, in the
  right directions. Every clamped node is on the bolt hole's wall (≤ r + reach from its axis), and the
  shells are kept solid.
- Script fields + a load primitive: the script's `field_fixed` is untouched, its `field_load` is wrapped
  (both loads present, case 1 and 2). Hiding the primitive restores the plain script. Old sidecars get
  the defaults.
- `gen3d optimize shapes/bracket_editor.wgsl`: 449 clamped / 646 loaded nodes, 2 load cases, converged
  in 48 iterations (82 s). The result is a hollow box section from plate to pin eye (section view).

**Open:** per-case weights; a load as a pressure on a face (a region's surface only); showing the load
direction as an arrow in the viewport; distributing a load by area instead of by node count (node
spreading slightly favours thicker regions).

## 2026-11-12: Subtract and intersect for editor primitives

**What:** every primitive has an **operation**: add (union), subtract or intersect, each smoothed by its
blend radius. They apply top to bottom: the script first, then each primitive with everything above it.
The sidebar has an operation selector, the list shows `+` / `-` / `&`, and **Up** / **Down** reorder
(order matters now). Existing `.prims.json` files load as before (`op` defaults to add).

**Design decisions**
- **The operation is data, not code:** its code goes in the spare 4th slot of the primitive's third vec4
  (`dims, op`), and the generated `map` calls `g3d_ed_combine(d, e, blend, op)`, which branches on it.
  The branch is uniform (same for every pixel) in the preview, and folded away in the baked-in `const`
  form for meshing/export. So switching add ↔ subtract is a buffer write like moving. Reordering
  primitives of the same kind is too; reordering different kinds regenerates the source.
- **Smooth formulas** are the same polynomial smin/smax as `sdf.wgsl`'s `opSmooth*`. A Rust copy
  (`Op::apply`, test-only) is the reference.
- **Ghost of the selected primitive:** a subtracting primitive is invisible apart from its hole. The
  preview now sphere-traces the selected primitive alone (only while one is selected) and blends a
  translucent blue ghost wherever it is nearer than the visible surface. That is the cutter in empty
  space, or a primitive buried in the script. It's a `g3d_overlay` hook in `g3d_fs`; every compile
  defines a no-op unless it's a preview with editor primitives (so .vti previews are untouched). The
  surface tint now uses `abs(distance)` so it also marks faces a cutter carved.
- **Bounds** grow only for added primitives.
- Sidebar operation changes and reorders start a new undo step (not merged with a preceding drag).

**Verified:** `cargo test` (22 pass). New GPU test: add box, smooth-subtract sphere, rotated intersect
cylinder, smooth-add torus, sampled on the GPU = the CPU fold within 1e-3, and the carved centre is
outside. The app test now also switches a primitive to subtract: no pipeline build, saved, bounds back
to the file's own, undo restores it. Visually: a smooth-subtracted cylinder makes a clean through-hole in
a tab, with the ghost when selected, and a subtracted sphere makes a dimple in the knob top. The
exported STL is watertight. clippy clean.

**Open:** rotation handles in the viewport; GPU picking so script geometry occludes; mirror/pattern
helpers; drag-to-reorder in the list.

## 2026-11-12: Live f32 parameters; ints and bools are the compiled ones

**What:** parameter sliders used to rebuild the pipeline on every change (overrides are applied at
pipeline creation; ~100 ms driver compile). Now the parameter's type decides: **`f32` parameters are
live** in the preview (a buffer write per change), while **`i32` / `u32` / `bool` stay `override`s** and
recompile. Those are the ones that can be loop counts, array sizes or switch selectors the compiler
should see. The app shows compiled parameters in italics, with a tooltip.

**How:** `Shape::compile_preview` rewrites each `f32` parameter's `override` keyword to `var<private>`
(same name, same line, so the script's code and error lines are untouched) and appends
`@group(0) @binding(5) var<uniform> g3d_par: array<vec4f, ⌈n/4⌉>` plus `g3d_load_params()`, which copies
the values into the privates. The preview's `g3d_fs` calls it first. Every other compile defines an
empty `g3d_load_params`, and batch paths (mesher, optimizer, export, `render`'s constants) keep all
parameters as overrides, fully constant-folded. If the live form fails to compile (an `f32` used in a
constant context, e.g. another override's initializer), the preview silently uses the all-override form,
so nothing that compiled before breaks. In the app, `sync_params` writes the buffer and only requests a
rebuild when the *override* values differ from what the pipeline was built with (`built_defines`).
`build_pipeline` now takes a `Bindings` struct (view, volume, params, editor buffers).

**Trade-off:** live values can't be constant-folded by the driver, so a shape whose `f32` parameter
feeds heavy math may raymarch slightly slower in the preview. Declare it `i32` (or derive an int) if
that matters.

**Verified:** `cargo test` (21 pass). New app test: three `f32` changes start zero pipeline builds, an
`i32` change starts exactly one. The shape test checks the rewrite (`R`, `OFF` live; `N` stays an override
with its `-D` value), the buffer layout, and the fallback for `override B: f32 = A * 2.0`. `gen3d render
knob -D DIAM=36` (live path) fills the bounds as expected; the exported STL is 35.99 mm across. All
bundled shapes check (all their parameters are `f32`, so all become live). clippy clean.

## 2026-11-12: Moving a primitive no longer compiles a shader

**What:** dragging a primitive showed "compiling…" and the frame rate dropped. Its numbers were WGSL
`override` constants, and overrides are applied when the *pipeline* is created, so every drag step
re-ran the driver's shader compile (~100 ms, on the background thread). Now the numbers (and the
selection index) live in one `array<vec4f, 1 + 3n>` named `g3d_ed`. The generated code reads
`g3d_ed[1 + 3i ..]`: (pos, blend), (rot, round), (dims, -), element 0 = selection.

- **Preview:** `Shape::compile_preview` declares `g3d_ed` as a uniform buffer at `@group(1)
  @binding(0)`. The app keeps the buffer next to the pipeline (`Win::ed`), and a move, resize or
  selection change is a `queue.write_buffer`. Structural changes (add/delete/hide/kind) still
  regenerate the source; a new buffer size comes with that.
- **Everything else** (`Shape::compile`: mesher, optimizer, VTI export, `render`) declares it as a WGSL
  `const` with the current values, so batch shaders stay fully constant-folded.
- `params()`/`fields()` parse the source with the const declaration added. Parameter overrides are
  unchanged and still rebuild the pipeline (they can change control flow, e.g. loop counts).

**Verified:** `cargo test` (20 pass). The app test now asserts a click-select plus a full gizmo drag
start **zero** pipeline builds (`App::rebuilds`). GPU/CPU SDF agreement still holds through the const
path. The snapshot and the exported STL are unchanged (47.28 × 32.40 × 19.99 mm, 9.34 cm³). clippy
clean.

## 2026-11-12: Primitive editor in the Design step (union with the script)

**What:** the Design step has a **Primitives** section: add a sphere, box, cylinder, capsule, torus or
cone, select it (click in the view or the list), move it with X/Y/Z arrows in the viewport (the centre
handle moves it in the view plane, Ctrl snaps to 1 mm), and edit position, rotation, size, edge
rounding and a smooth-union *blend* in the sidebar. Hide/show, duplicate (Ctrl+D), delete (Del),
deselect (Esc), undo/redo (Ctrl+Z / Ctrl+Shift+Z; a drag is one step). Only union for now.

**Design decisions**
- **Sidecar file, not the script.** Primitives are saved on every change to `NAME.prims.json` next to
  `NAME.wgsl` (serde_json, write-then-rename). The app never rewrites the user's .wgsl, which may be
  open in their editor. Because it's on disk, the CLI and the app's subprocess jobs (optimize, export,
  slice) pick primitives up with no extra plumbing.
- **Merged at load time, under the name `map`.** When the sidecar exists, `load_shape` renames the
  script's `fn map` (a comment-aware text scan that keeps line structure, so `file:line:col` errors
  still point into the script) to `g3d_script_map` and appends generated WGSL: a prefixed primitive
  library (`g3d_ed_*`, self-contained so it doesn't depend on `#include <sdf.wgsl>` or clash with the
  script's own `sdBox`), one function per primitive, and `map = script ∪ primitives`. Every harness
  (preview, mesher, topopt, VTI export) calls `map`, so none of them changed. Script functions that call
  `map(p)` get the combined shape. A .wgsl without `fn map` works when a sidecar exists (primitives
  only). Without a sidecar nothing changes.
- **Numbers are `override` constants** (`g3d_e{i}_px` …, hidden from the parameter sliders). Dragging
  goes through the same async pipeline rebuild as parameter sliders; only add/delete/hide/kind change
  regenerates the source. `Shape::constants()` now returns `-D` values plus the primitives' values.
  *(Replaced the same day by a uniform buffer; see the entry above.)*
- **Selection highlight:** `override g3d_ed_sel` and a generated `g3d_ed_selected(p)`; the preview
  harness tints surface points on the selected primitive in striped blue.
- **Bounds grow** to fit the primitives: a rotated local AABB plus the blend bulge plus 2 voxels. A
  bounding sphere put the tilted test torus 8 mm under the plate. The camera reframes only when the
  file's own `//! bounds` change, not when primitives grow them.
- **Picking on the CPU:** Rust copies of the six SDFs, sphere-traced along the same ray the shader uses
  (`Camera::basis`). It only sees primitives, not the script, so a primitive buried in script geometry is
  still clickable. Good enough for now; GPU picking would fix it.

**Verified:** `cargo test` (20 pass): the `fn map` rename, CPU SDF vs. the GPU-sampled generated WGSL
for all six kinds under rotation (within 1e-3), the AABB holds every inside point for random rotations,
shapes with primitives compile with every harness, `g3d_` overrides stay out of the parameter list,
script error lines stay correct, primitives-only shapes work. An app test drives egui with synthetic
pointer events: click-select, X-arrow drag (only x moves, camera doesn't orbit, sidecar saved), the drag
is one undo step, delete + undo. `gen3d snapshot --select N` of a knob with a blended box, a sphere and a
tilted torus looks right. `gen3d export` of it gives a watertight STL reaching the box's edge
(x = 32 mm). All bundled shapes still `check` unchanged. clippy clean.

**Open:** subtraction/intersection (per-primitive op, ordering against the script); rotation handles in
the viewport; GPU picking so script geometry occludes; snapping/alignment helpers (e.g. sit on the
plate); an "editor only" choice in New shape.

## 2026-11-12: One app, one flow: Design → Optimize → Print

**What:** the viewer's sidebar had grown a section per feature (Parameters, View, Field overlays,
Topology optimization, Output with Export VTI and a second "View" window, Grid, Shape). It's now one app
with three steps, so a shape goes from .wgsl to G-code without the terminal. `gen3d` with no arguments
opens it.

- **Top bar:** Open…, the file with an Edit button and a status dot, the steps (1 Design > 2 Optimize >
  3 Print), and running jobs. The sidebar shows only the current step.
- **Open dialog** (`src/browser.rs`, plain egui): folders and `.wgsl`/`.vti` files, newest first, with
  shortcuts to project/shapes/out. **New shape** writes the template (moved to `shape.rs`), opens the
  shape and starts the editor. It opens at startup when no file is given; `Ctrl+O` later.
- **Edit:** `$GEN3D_EDITOR` (split on whitespace, e.g. `kitty nvim`), else `xdg-open`/`open`.
- **Design:** parameters, field overlays, shape details. For a `.vti` opened directly, the threshold
  and colour controls instead; Optimize is disabled.
- **Optimize:** settings, Optimize/Stop, a progress bar parsed from the job log (iteration, compliance,
  grey, β). The viewport switches to the result after the first iteration (polled every 250 ms), and the
  threshold histogram and material volume sit under it. **Next: Print the result**.
- **Print:** design or optimized result. Mesh voxel (with a sample count) and decimation, STL/3MF,
  Export mesh. Printer profile from `printers/*.ini|json` or *auto* (what `slice` would pick anyway).
  Layer height, infill and supports overrides passed to PrusaSlicer after `--` (hidden for Orca). Slice to
  G-code / Open in slicer. The G-code summary is read from the file (`slicer::gcode_summary`, cached by
  mtime) and shown as print time, filament, layer height, infill, supports, nozzle.
- **Viewport toolbar:** camera presets, Section (+ cut slider), Bounds, Clip, resolution. These were a
  sidebar section.
- **Dropped:** Export VTI button and its voxel estimate (CLI `export --format vti` stays; the built-in
  optimizer made it an expert path), the second viewer window, Copy -D flags, the E/G/O keys (too easy
  to start a slow job by accident), Q/Esc quit (Esc should close the dialog), the resizable sidebar
  (its width jumped between steps), and `view`'s `.vti` display flags (`render` keeps them).

**Design decisions:**
- **Outputs are derived, not remembered.** Every output path is a function of the file and parameter
  values (`default_out`, new `topopt_out`), and the app passes `-o` explicitly. So reopening a shape
  finds its result, mesh and G-code with no project file. Changing a parameter correctly leaves "no
  result for these values". Each output line warns when the file is older than its sources (the
  design's files, or the result).
- **Design vs shown file.** `App.design` is what you opened; `App.shown` is what the viewport shows (the
  design or `out/NAME_topopt.vti`), picked from the step by `wanted_shown`. The design's shape, fields
  and parameters are kept while the result is shown, so the camera, Optimize settings and slider values
  survive switching back and forth. The grid view options (array, threshold) are saved when switching
  away and passed to export/slice as `--array/--iso/--above`.
- **Jobs stay subprocesses** (`src/jobs.rs`, moved out of the viewer): Stop is a plain kill, the log
  is the CLI's own output, and the CLI and app share one code path. There are two slots, so a mesh can be
  exported while an optimization runs.
- **Split:** `viewer.rs` is now only the raymarching core (harness, pipelines, `render_png`). The
  window is `app.rs`.

**Headless UI check (`gen3d snapshot`, hidden command):** there is no Xvfb here, and screenshotting the
desktop is off-limits while the user is on it. So the window's GPU state is split from the OS window
(`Win.os: Option<OsWindow>`), frames are `run_ui` + `paint(target)`, and `snapshot` paints the full UI
into an offscreen texture for a few frames, with time advancing 1 s per frame so fade-ins finish, then
writes a PNG (`viewer::save_texture_png`, shared with `render`). Options: `--step`, `--result`,
`--browser`, `--section`.

**Verified:**
- clippy clean; 15 tests pass. New tests: optimization progress parsing, ANSI stripping, the G-code
  summary labels, and result/mesh paths following parameter values.
- The exact command lines the app builds, run from the CLI:
  - Optimize with `-o`.
  - Export the result with `--array density --iso 0.5 --above -o` (watertight).
  - `slice` with `-- --layer-height 0.2 --fill-density 15% --no-support-material` (the G-code reports
    `support_material = 0`) and with build-plate-only supports (`= 1`).
- Snapshots on the RTX 3080:
  - Startup Open dialog.
  - Design: bracket_domain with field overlays.
  - A broken shape: error panel, red view, steps still usable.
  - Optimize with the existing result, histogram and volume.
  - Print of the result, with the G-code summary.
  - Print of the knob design.
  - A `.vti` opened directly.
- Found by the snapshots and fixed: `→` is not in egui's default fonts (it was already a box in the old
  Shape panel), and "blocky" overflowed the sidebar.

**Not verified:** real clicks and drags (KDE asks before synthetic input). Not exercised in a live
window: hot reload of the result while optimizing, the Edit/Show buttons launching apps, and Stop.
The logic for each is the same code as before or covered by the CLI runs above.

**Open / next:**
- A built-in WGSL editor panel (egui `TextEdit` with highlighting), for a truly single-window loop.
- Printer upload (PrusaLink/OctoPrint) as a last step.
- Show the mesh (not the SDF) in Print, with overhang shading, to catch printability problems before
  slicing.
- Remember recent files and the last profile across sessions.

## 2026-11-12: Heaviside projection for `gen3d optimize`

**What:** a smoothed Heaviside projection of the filtered densities (Wang, Lazarov & Sigmund 2011),
x̄ = (tanh βη + tanh β(x̃−η)) / (tanh βη + tanh β(1−η)) with η = 0.5, on by default.

- **Continuation:** β starts at 1 and doubles (to `--beta-max`, 32) each time a stage converges (the
  same stall rule: < 0.1% compliance change for 3 iterations, or density change < tol), or after 25
  iterations at one β. The final stop only happens at β_max. After each doubling, x̄ is recomputed from
  the current x̃, and the stall counter restarts, because compliance jumps when β changes.
- **Sensitivities:** compliance and volume both get the factor dx̄/dx̃ before the filter's chain rule.
  So the volume sensitivity is recomputed every iteration (before, it was a constant).
- **Volume:** measured on the projected densities inside the OC bisection.
- **Kept-solid elements** stay 1 and aren't projected.
- **Interface:** `--beta-max`, `--no-projection`, default `--iters` raised to 150 (runs stop early). The
  viewer's Topology optimization section has a "crisp 0/1 design" checkbox.

**Verified:**
- A new test: endpoints 0 and 1 fixed, x̄(η) = η, the derivative matches central differences for β in
  {0, 1, 8, 32}, and β = 32 is sharp. 13 tests pass, clippy is clean.
- bracket_domain at 1.3 mm:
  - Grey (0.1–0.9) went 24% → 20% (β 2) → 10% (β 4) → 2% (β 8) → 0% (β 16, 32).
  - Final compliance 8.96 against 10.36 N·mm without projection: 13% stiffer for the same material.
  - 49 iterations and 77 s, against 29 and 52 s without.
  - Same truss topology. Thresholded mesh watertight, one body, 27.6 cm³ against 28.6 for the exact
    voxel volume; marching cubes cuts the staircase corners.
- `--no-projection` reproduces the previous run exactly (compliance 40.976 at iteration 3).

**Tradeoff seen:** the crisp design's threshold surface is a voxel staircase. The grey design looked
smoother only because of its soft density gradient. Options: a finer voxel (costly), or smoothing the
density field before meshing (cheap; a small blur keeps the volume nearly unchanged), as a viewer and
export option.

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
