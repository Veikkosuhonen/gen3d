# gen3d devlog

Newest entries first. Each entry covers what changed and why, the design decisions, how it was verified,
and what is still open.

## 2026-11-14: Crop the optimizer grid to the design domain

**Why:** the optimizer's grid spanned the shape's bounds, which can be much larger than the part: a
generous `//! bounds`, a part off-centre from a mirror plane, a subtracted cut that removes most of a
box. Empty cells still cost memory and time on the GPU (its grid is dense) and in setup.

**What:** `setup` samples `map` (and `field_void`) over the bounds once, finds the cells inside the
domain, and crops to them plus one empty cell (`crop_box`, `CROP_MARGIN`), symmetric about the middle
on mirror axes so each plane stays the middle node layer. `field_keep`, supports and loads are then
sampled on the cropped grid. The solver's element limit applies after cropping; sampling the bounds is
capped at 200M cells. Coarse and fine grids are cropped separately; `prolong_design` already maps by
position. Elements outside the domain never entered the solve, filter or volume, so the problem is
the same.

**Result** (GPU, same voxel, before → after):

| Shape | Grid | Time | Compliance |
|---|---|---|---|
| stand_leg @ 1.9 mm | 54×106×54 → 12×106×54 | 29.0 → 12.6 s | 8.8615 → 8.8636 |
| book_stand1 @ 3.2 mm | 63×41×95 → 63×41×74 | 13.4 → 12.4 s | 3.0212 → 3.0238 |
| bracket_editor (auto) | 119×40×62 → 114×36×58 | 10.4 → 9.8 s | 16.951, same |

The compliance differences are the GPU multigrid's hierarchy changing with the grid size, within its
tolerance. (stand_leg's bounds were also oversized by the load-time bounds bug fixed just before.)

**Verified:** `crop_box_fits_the_domain` (margin, mirror centring, empty domain);
`setup_crops_to_the_domain` (a part in bounds ~4× too large per axis gives the same domain, supports
and loads as with tight bounds, one empty cell around it). All tests pass.

**Still open:** the automatic element size still comes from the bounds' volume, so oversized bounds
give a coarser grid than the 300k target; getting it from the domain needs the GPU sampling before
the size is chosen.

## 2026-11-14: Primitives-only shapes get bounds that fit them on load

**Why:** a `.wgsl` without `fn map` whose `.prims.json` holds the shape should have bounds that fit
its primitives. On load it got the default ±50 mm × 100 mm box added on top, so `optimize` and the
first view used a bigger box than the part; after any edit in the app the bounds shrank to fit. The
user's `stand_leg` (a 20 × 200 × 100 mm box) got a 102 mm wide grid.

**What:** `load_shape` set the bounds from the primitives before it knew whether the script has a
map of its own (`script_map` still held its default, `true`), so the default bounds counted as
declared. The bounds are now computed after.

**Result:** `stand_leg`'s bounds fit it: the automatic element size goes from 1.9 mm (56k elements) to
1.2 mm (223k), as intended for the GPU solver's 300k target.

**Verified:** the primitives-only test checks the bounds equal the primitives' own; an empty shape
still gets the default box.

**Still open:** —

## 2026-11-14: Fair the voxel surface of optimized parts

**Why:** exported optimization results still showed voxel terraces, even with `density_filtered` and
10 Taubin passes. On a shallow slope the steps are 1/slope voxels apart: a long wavelength that
Taubin (passband ~0.1) barely touches. What tells them apart from features is their amplitude, at
most half a voxel. Where the part meets a flat face of the design domain, the outline on that face
was also a staircase, because marching cubes only puts the snapped face vertices at lattice points.

**What:**
- `fairing.rs`: the smoothest field (thin-plate energy Σ(Δφ)²) whose value at each sample stays
  within 0.45 of its signed distance to the voxel surface (EDT − ½). Projected gradient descent with
  Nesterov momentum, 100 iterations on a 4-voxel band. Samples outside the domain are free (the
  field continues through the faces) and start from the mask extended by nearest neighbour.
- `Volume::mesh(&MeshOpts { fair, smooth })` does the whole sequence: field (faired or interpolated),
  marching cubes, snapping to domain faces, Taubin with the domain constraint (now in grid units), then
  scaling. With fairing, vertices on each domain face's outline slide within the face onto the
  faired surface, both before Taubin and again after it (that time only as far as flips no triangle).
  Vertices where two domain faces meet are left alone.
- On by default for `density` arrays; `export --no-fair` gives the old surface, and `--fair` fairs other
  arrays.

Tried and dropped (48³ test shapes, see below): longer Taubin (100 passes rounds everything and still
leaves terraces); Gaussian blur of x̃ (σ 1: thin rod −51% volume); membrane energy Σ|∇φ|² (shrinks
by mean curvature, −8 to −44%); sign constraints only (rod −24%: the field goes flat inside thin
members); asymmetric margins (fix the rod, inflate the rest 2–5%); a 2× finer field (10× the time,
small gains); thin-plate fairing of the mesh vertices (unstable near the domain, rod −20 to −35%);
bands wider than ½ voxel (rod −22 to −73%). Mirrored Laplacian vs. linear extrapolation at grid edges:
mirrored is better.

**Result** (mean vertex distance from the true surface in voxels / mean normal error / volume error;
x̃ marching cubes + Taubin 10 against faired + Taubin 10):

| shape | before | faired |
|---|---|---|
| sphere r11.3 | 0.077 / 5.3° / −0.7% | 0.025 / 1.9° / −0.4% |
| box tilted 9°/5°, faces | 0.158 / 9.0° / −0.9% | 0.166 / 4.4° / +2.8% |
| cylinder r7 tilted 6° | 0.107 / 6.7° / −0.7% | 0.046 / 2.1° / +1.1% |
| rod r1.9 tilted | 0.125 / 10.2° / −12.4% | 0.096 / 7.7° / −10.0% |

Volume against the voxels' own: bracket 26.31 → 26.92 cm³ (voxels 26.71: −1.5% → +0.8%), table
2009.8 → 2126.8 cm³ (voxels 2075.7: −3.2% → +2.5%). Visually, the bracket's ramp is now smooth, and
its outlines on the side and top faces are smooth curves. The table's pocket ridges stay: its
1.7 mm shells are 2–3 voxels thick and widen by a voxel at each step, a feature of the design
itself. Time: the 3.7M-cell table exports in 1.3 s instead of 0.5 s (+145 MB).

**Verified:** `fairing_beats_the_filtered_density` (the table above), `fairing_flattens_terraces`
(a tilted half-space comes back flat to 0.1 voxel, ±0.05 in the middle), and the slope `.vti` test for
both paths (faired: within 0.25 voxel of the plane on that small slab, watertight, inside the domain,
volume ±1%). Bracket and table meshes are watertight with the same body count.

**Still open:** sharp convex edges inside the domain are rounded and bulge out by up to 0.45 voxel
(the box's +2.8%); the preview still draws the interpolated x̃ surface, not the faired one; the
crease strip beside a domain face stays half a voxel wide and slightly faceted; one-voxel features
of the design (thin-shell ridges) need a finer or smoother optimization.

## 2026-11-14: Solve one side of mirror-symmetric problems

**Why:** with mirror planes declared (previous entry), a symmetric part needs only one side solved:
2^planes fewer elements, less memory, and a symmetric design by construction.

**What** (`topopt/symmetry.rs`, the module docs have the derivation):
- Setup aligns the grid so each plane is the middle node layer (also with `--no-symmetry`, so that
  flag changes only the solve), and mirrors the stabilizing pins so the supports stay symmetric.
- `symmetry::reduce` checks domain, kept regions and supports for symmetry and keeps the upper side.
  Each load case is split into 2^planes parts, symmetric or antisymmetric about each plane; each
  pattern with any load is a group with its own plane condition (S: the normal displacement held on
  the plane, A: the tangential ones) and its own multigrid hierarchy. Plane nodes carry half their
  load. Full compliance = 2^planes × the sum over groups; per element, the groups' energies sum to
  the full energy averaged over mirror images, which is the symmetric design's sensitivity.
- `Model { groups, sym }` replaces the single `Problem` in the CPU and GPU loops. GPU hierarchies after
  the first share its element stiffness/energy buffers, element tables and CG scratch vectors (solves
  are sequential); each has its own λmax scalar slots.
- The density filter (CPU, GPU `filter` and `sens` weight sums) reflects at the planes.
- OC divides the sensitivities by the full compliance, so λ (and the bisection's path) is the same
  with or without symmetry.
- Output: cell fields are mirrored, each case's displacement is combined from its parts with the
  signs of each mirror side; the max displacement is taken over every case and side.
- Used from 100k domain elements (the table's 26k-element coarse stage was slower as 3 quarter
  hierarchies: 5.8 s against 3.6 s whole). `--no-symmetry` turns it off.

**Result** (same aligned grid; time, iterations, compliance):

| | whole | reduced |
|---|---|---|
| bracket_editor mirrored in Y (2 patterns) | 9.1 s, 58, 16.802 | 7.4 s, 58, 16.801 |
| table mirrored in X, Y (3 patterns, 3 of 15 parts solved) | 15.9 s, 63, 0.029068 | 12.2 s, 63, 0.029069 |
| bracket_editor at 0.4 mm (1.2M) | 49.6 s, 53, 17.864 | 29.6 s, 53, 17.862 |

GPU memory: the bracket 59 → 46 MB.

**Verified:** `symmetric_quarter_matches_the_whole_cpu` / `_gpu`: a beam mirrored in X and Y whose cases
come with their mirror images (so the whole problem's design is symmetric too) while each case has
parts of all four patterns: compliance per iteration within 1e-6 (CPU) / 2e-3 (GPU), and the written
density and displacements of the asymmetric cases match. CPU and GPU CG counts match the whole
part's on the mirrored bracket.

**Still open:** each pattern builds its own hierarchy every iteration (setup × patterns); a block
solve over the patterns' cases; the script's own map isn't mirrored.

## 2026-11-14: GPU multigrid: per-axis masks for rollers, a careful λmax re-estimate

Two GPU solver problems found while testing symmetry, both older than it:

**Rollers on coarse levels.** The table of Pᵀ(D ke D)P that builds level 1's diagonal and level 2 is
indexed by which of an element's nodes are clamped; a partly held node (a roller, `--ground slide`,
a symmetry plane) counted as clamped on all axes, so the coarse levels didn't match the fine one along
such faces. A whole face of rollers (the symmetry plane) took CG from 8 to 28 iterations per solve.
Now an element with a partly free node gets its level-1 matrix computed in the kernel with per-axis
masks (`fine_free`, `partial_entry`; ke is bound to `diag_l1` and `assemble`); the others still use
the table. Result: the same CG counts as the CPU multigrid (8 against 8 on the bracket's symmetric
half, where it was 28).

**λmax underestimates.** On the 1.2M bracket's first fine-grid iteration after a coarse start, the
12-step power iteration came out low enough that Chebyshev amplified the top of the spectrum: CG
stalled at 3.5× its tolerance and ran to its 20,000-iteration limit (the run looked hung). 40 steps fix
it but cost ~20% on every run. Instead, each solve now first runs to `STALE_CG_ITS` (32) iterations;
if it hasn't converged it refreshes the estimate (a 40-step one when the estimate was fresh) and
carries on. That run: 49.6 s, as expected; bracket_editor and table unchanged (9.9–10.2 s,
16.7–17.2 s). CG's convergence checks now read back only the residual slot (the scalar buffer grew
to 544 slots for the per-group λ and per-side maximum displacements).

## 2026-11-14: Mirror planes in the editor

**Why:** symmetric parts are common (brackets, stands, tables), and the optimizer can solve only one
side of a symmetric problem. First step: a way to declare the symmetry.

**What:**
- `.prims.json` has an optional `"mirror": [x, y, z]` (a plane position per axis, or null), read with
  `prims::load_file` into `Shape::mirror`.
- Generated WGSL: each primitive's own SDF is `g3d_p<i>`, and `g3d_e<i>` is the min over its mirror
  images (2^planes points), so everything built on `g3d_e<i>` (the map, the selection highlight, the
  fixed / load / keep regions) is mirrored. Mirroring per primitive rather than folding the whole shape
  keeps subtracted holes on both sides and lets primitives sit on either side of the plane.
- Loads keep their direction on the mirrored region: a sideways push on a load region that spans the
  plane stays one push.
- Bounds are made symmetric about the planes; picking traces the mirror images too.
- Design step: a *Mirror* row (X / Y / Z toggles and plane positions; a position edit applies when the
  drag or typing ends, since it recompiles).

**Verified:** `mirrored_primitives` (sidecar round trip, symmetric bounds, the twins in the generated
code, compiles for meshing and preview, picking the twin); an offscreen snapshot of a mirrored beam.

**Still open:** the optimizer doesn't use the symmetry yet (next). The script's own map isn't mirrored.

## 2026-11-14: Animate the load bands for 3 s, then freeze

**Why:** even at 30 fps the moving load-direction bands kept the GPU 60–95% busy with the window
idle (a viewport raymarch takes 21–32 ms), on any shape with loads.

**What:** the preview animates for `ANIMATION_SECS` = 3 after the view changes, then stops.
"The view changed" means the viewport was marked out of date (`Win::touch`: pipeline, parameters,
primitives, volume, mesh, size) or its uniforms changed apart from the time (camera, options,
overlay mode), compared through a second copy of the uniforms with the time zeroed
(`Win::drawn_still`). The animation's own frames change only the time, so they don't extend it.
The pause during an optimization stays.

**Result** (GPU utilization per second after opening the window; baseline 0–4%):
- bracket_editor: 4 96 95 95 0 0 4 0 0 0
- bracket_domain: 60 63 61 49 5 0 4 0 0 0

**Verified:** the timelines above; clippy clean. Restarting on orbiting or changing the overlay
mode is by construction (both change the uniforms); not driven by hand here.

## 2026-11-14: Throttle the preview animation, and pause it while optimizing

**Why:** after on-demand drawing, the moving load-direction bands were the one thing still redrawing
the viewport at the refresh rate. A raymarch of the 2200×1670 px viewport takes ~32 ms on
bracket_editor and ~21 ms on bracket_domain (M1 Pro), so the animation alone kept the GPU busy,
also while an optimization (which needs the GPU) ran.

**What:** the animation redraws at `ANIMATION_FPS` = 30 (the bands move 1.5 bands per second, so it
looks the same). It holds still while an optimization job runs: `animating()` is false, so time
stops advancing and the viewport redraws only on changes.

**Result** (GPU utilization, idle window): bracket_domain 100% → 63%, bracket_editor 100% → 95%
(its raymarch takes about as long as a 30 fps frame); 0–4% (the baseline) during an optimization.

**Verified:** measured as above, with the render rate logged (30 fps). The pause during an
optimization follows from `animating()`; not measured with a job running from the app.

**Still open:** animating at all costs most of the GPU on heavier shapes (since then: it animates
only for 3 s after a change, see above).

## 2026-11-14: Draw the viewport on demand

**Why:** the window redrew every frame: `RedrawRequested` drew a frame and requested the next one, so
the app raymarched the preview at the display's refresh rate even when nothing changed. Idle, it
kept the GPU at 100% utilization, competing with optimizer jobs for the same GPU.

**What:**
- Frames on demand: a frame is drawn after input that egui wants repainted, when egui asks for one
  (its animations, the job spinner, tooltips), when the 250 ms poll sees a change (a reloaded file,
  the shown file switching, a job starting or ending, the mesh to show changing on disk), while a
  pipeline compiles (to pick it up), and continuously while the preview animates.
  `App::schedule` sets the next frame after each one; `about_to_wait` waits until it or the next
  poll.
- egui-winit answers `repaint: true` to every event, the redraw itself included, so that response
  is ignored for `RedrawRequested` (otherwise each frame requests the next).
- The viewport (drawn into its own texture, which egui shows) is re-rendered only when its view
  uniforms (camera, options, size, field mask) differ from those of the image it holds, or when
  `Win::touch` marked it out of date: the pipeline, live parameters, editor primitives, volume
  buffers, the shown mesh or the target size changed. UI-only frames reuse the image.
- The preview's time uniform advances only while it animates (a shown vector field overlay: the
  moving load-direction bands), so a still view keeps identical uniforms.
- The fps readout counts viewport renders and shows only while the viewport redraws.

**Result** (GPU utilization from `ioreg`, idle window, 6 samples; baseline without the app 0–4%):

| | before | after |
|---|---|---|
| gyroid_cube (no overlays) | 100% | 0–4% |
| vase | 100% | 0–4% |
| bracket_editor (load bands animate) | 100% | 100% |

**Verified:** idle measurements above; the offscreen snapshot still draws the preview; tests pass
(except the known `shapes_with_primitives_compile`); clippy clean. Interaction (orbit, zoom,
parameter sliders, editing primitives, the optimizer's live result) needs a check by hand: each
goes through egui input or `touch`.

**Still open:** the animated load bands keep the viewport redrawing at the refresh rate in the
Design step of any shape with loads.

## 2026-11-14: Stop and resume an optimization

**Why:** Stop kept the current design but threw away the run: optimizing again started from scratch.
On large grids a run takes minutes, and it is natural to stop one (to free the GPU, or to look at
the design) and carry on later.

**What:**
- On a user Stop before convergence, `run` writes `<result>.resume` (`topopt/checkpoint.rs`): a JSON
  header line (stage coarse/fine, grid size, settings key, problem fingerprint, iterations done, β,
  the loop state), then x as f32 per element.
- The loop state is what the CPU and GPU loops carry between iterations besides x: `Control` (the
  stopping rule's counters, now serializable), OC's λ (the next bisection's bracket) and the last
  density change (the adaptive CG tolerance). `Stage.state` starts a loop from it; `Outcome` returns
  it with β and `stopped`.
- The Stop check now comes after `Control::next`, so a stopped iteration still gets its β step or
  convergence test. A run that converges in the iteration it was stopped leaves no checkpoint.
- `--resume`: refused unless the settings key matches (voxel, volume fraction, penalty, rmin, E, ν,
  force, β max, E_min, ground, coarse start; not iterations, tolerances or solver) and the problem
  fingerprint (FNV-1a over grid, domain, kept regions, supports and loads) matches the grid it was
  stopped on. A coarse-stage checkpoint resumes the coarse stage, then continues to the fine grid as
  usual. Iteration numbering continues, and `--iters` still caps the whole run.
- A finished run, or a run without `--resume`, removes the checkpoint.
- App: while a checkpoint exists for the result, the Optimize step shows **Resume** (primary) and
  **Start over** instead of Optimize.

**Result** (bracket_domain, coarse start): uninterrupted 60 iterations, compliance 8.2021. Stopped at
iteration 11 (coarse stage) and resumed: iterations 12 and 13 match the uninterrupted run exactly,
60 iterations, 8.2020. Stopped at 41 (fine grid, β = 2) and resumed: iteration 42 within 0.01% (the
multigrid's warm start and λmax estimate are not saved), 60 iterations, 8.2019.

**Verified:** `checkpoint_round_trip` (design, β, iterations and state come back; other settings and
another load are refused; the iteration limit may change). By hand: the runs above, `--resume`
without a checkpoint, with `--volfrac` changed (shows both settings lines), and the app panel
(offscreen snapshot). topopt tests pass; clippy clean.

**Not done:** the solver's warm start (u per load case) isn't saved, so the first resumed iteration
needs a few more CG iterations. Continuing a run that ended at its iteration limit (rather than
stopped) isn't offered.

## 2026-11-14: Solve only the linearly independent load cases

**Why:** on the table (`item on top` with rocking: 5 cases) the solve was 88% of the run time. A
rocked variant's load is the case's load plus ±rock times a sideways push (the force is scaled so the
original component stays the same), so the 5 cases span only 3 load vectors. All cases share the
supports and so share K, and u is linear in f: 3 solves give all 5 displacements exactly.

**What:**
- `Problem::load_basis` goes through the cases in order. A case's load (on free DOFs) is solved when
  it is more than 1e-6 (relative) outside the span of the cases solved before it (Gram–Schmidt,
  twice). Otherwise its coefficients over the solved cases come from the normal equations. This is
  generic: any combination of cases is found, not only rocking. A load only on clamped DOFs is an
  empty combination (u = 0).
- CPU loop: solves the basis cases, forms the others, then energies and output as before.
- GPU: `GpuMg` keeps u, the load lists and the CG programs only for solved cases, plus one scratch
  vector for the combined ones. `compute_energies` forms each combined case in the scratch vector
  (`lin` steps), then adds its energy and takes its maximum displacement (moved from `gpu_opt`'s
  sensitivity program, since the scratch vector is reused). `displacement(c)` forms the case first.
- A line in the log when cases are skipped: "3 of 5 load cases solved, the others are linear
  combinations of them".

**Result:** table (`--ground grip`) 24.8 → 16.0 s, compliance 0.032862 → 0.032859; the iteration
lines match before and after (same compliance, same iteration count), with 3/5 of the CG iterations.
GPU memory 248 → 232 B per element (two fewer u vectors, one scratch vector). bracket_editor (two
independent cases) and book_stand1: unchanged.

**Verified:** `load_basis_solves_only_independent_cases` (rocked ± variants and the sideways push
are combinations with the expected coefficients; a load on a clamped node is empty).
`gpu_loop_matches_cpu_loop` now has a third case = case 1 − 0.5 · case 2: compliance per iteration,
max displacement and the combined case's written displacement agree between CPU and GPU. topopt
tests pass; clippy clean.

**Still open:** the remaining solves are still sequential. Solving all cases in one pass (one
operator read for all right-hand sides) needs a profile of the fine mat-vec first: at 9.6M elements
it reaches only ~2 GB/s, so it looks latency-bound, and sharing the reads might not help.

## 2026-11-14: Stop the final stage when compliance oscillates

**Why:** at β = 32 the OC updates oscillate: compliance alternates by about ±0.1% (8.28 ↔ 8.29 on the
1.8M bracket). Each swing resets the "changed < 0.1% for 3 iterations" counter, so the final stage ran
15–19 iterations, about 20 s of a 51 s run, without improving.

**What:** `Control` also tracks the stage's best compliance. At the final β (or without projection),
the run stops once compliance hasn't beaten that best by 0.1% in 3 iterations. The old rule still
stops it too, so no run takes more iterations than before. A steady decline keeps going even when
it is slow: 0.2% per iteration doesn't stop. Intermediate stages are unchanged; when they advance
affected compliance by up to 2.5% (see "Tried and not kept" on 2026-11-13).

**Result** (coarse start on; time, iterations, compliance):

| | before | after |
|---|---|---|
| bracket_domain 0.35 mm (1.8M) | 51.5 s, 73, 8.2887 | 30.5 s, 58, 8.2883 |
| table (--ground grip) | 26.4 s, 66, 0.032875 | 24.8 s, 64, 0.032862 |
| bracket_domain (291k) | 7.2 s, 62, 8.2035 | 6.4 s, 60, 8.2021 |
| bracket_editor, book_stand1, --no-projection, --no-coarse-start | unchanged | unchanged |

The final-stage iterations are the most expensive ones (16–24 CG iterations against 4–8), so the
time saved is larger than the share of iterations.

**Not done:** keeping the best iterate rather than the last. At the stop the last one is within the
oscillation's ~0.1% of the best, and keeping it would need copies of the density, filtered density
and displacement fields (about 20 B per element on the GPU).

**Possible next:** a smaller OC move limit at high β would probably remove the oscillation itself.

**Verified:** `oscillation_at_the_final_beta_stops_the_run` (oscillation stops after 4 iterations;
slow declines continue; an intermediate stage never stops the run). topopt tests pass; clippy clean.

## 2026-11-14: Coarse start: the first projection stage on a 2× coarser grid

**What:** `optimize` runs the β = 1 stage on a grid of twice the element size (1/8 of the elements)
until its compliance stalls, interpolates the design onto the fine grid, and continues there at β = 2.
On by default from 80k domain elements; `--no-coarse-start` turns it off. The app gets it through the
CLI and shows the coarse design about 6× sooner.

**How:**
- `topopt::run(shape, …)` sets up both grids with `setup`. The CLI calls it instead of
  `setup` + `optimize`; `bench-topopt` and the tests still call `optimize` directly.
- `optimize_stage` takes a `Stage`: a starting design and β, an iteration offset (the iteration lines
  continue their numbering, which `jobs::opt_progress` relies on), and `quiet` (only the last stage
  prints "done", which the app reads as the end). `Outcome.design` returns the final design variables.
  Both the CPU and GPU loops support it.
- `prolong_design`: trilinear in the coarse cell centres, over coarse domain cells only, so the
  domain's edge isn't pulled towards 0. Kept-solid elements get 1.
- The filter stays `rmin` voxels on both grids (twice as wide on the coarse one). `rmin / 2` would be
  0.75 coarse voxels: no filtering at all.
- `--iters` caps the whole run. A Stop during the coarse stage writes the coarse result.

**Result** (default settings unless noted; time, iterations, compliance):

| | fine only | coarse start |
|---|---|---|
| bracket_domain (291k) | 8.3 s, 50, 8.372 | 6.9 s, 62, 8.204 (−2.0%) |
| bracket_editor | 11.2 s, 50, 16.954 | 9.3 s, 57, 16.951 |
| book_stand1 | 7.8 s, 73, 0.08965 | 5.4 s, 76, 0.09033 (+0.8%) |
| table (--ground grip) | 26.3 s, 48, 0.03283 | 26.3 s, 66, 0.03288 |
| bracket_domain 0.35 mm (1.8M) | 59.9 s, 60, 8.436 | 51.3 s, 73, 8.289 (−1.7%) |
| bracket_domain 0.9 mm, CPU mg | 22.4 s, 48, 8.863 | 16.7 s, 62, 8.651 (−2.4%) |

**Why not more than 15–30%:**
- Coarse iterations cost 1/4 to 1/2.5 of fine ones, not 1/8: fixed per-dispatch costs on small
  grids, and more CG iterations (the table: 80–100 against ~40).
- After the switch, the β = 2 stage takes 11–17 iterations against 5–6 in a fine-only run: the design
  keeps refining to the finer resolution, at 0.1–0.3% per iteration.
- Continuing at β = 1 on the fine grid instead was slower, with about the same compliance.

**Still open:**
- **The final β = 32 stage oscillates** (fixed in the next entry): compliance alternates by about ±0.1% (8.28 ↔ 8.29 at 1.8M), so
  the "< 0.1% for 3 iterations" stop fires late. It ran 15–19 iterations at 1.8M, about 20–25 s of
  60, without improving. A stop on "no new best in N iterations", or on the change over two
  iterations, would end it sooner. Reporting the best iterate rather than the last would also help.
- Three levels (4h → 2h → h) for very large grids.

**Verified:** `prolonged_design_interpolates_inside_the_domain`; topopt tests pass; clippy clean.
SIGTERM during the coarse stage: exit 0, full coarse result written. CPU `--solver mg` path run end
to end. `prims::tests::shapes_with_primitives_compile` still fails, as before.

## 2026-11-13: GPU multigrid setup 3–4× faster

**Why:** per optimizer iteration, rebuilding the hierarchy took as long as the solve (4.8M elements:
1.47 s setup against 1.93 s solve). Timing each setup dispatch showed that the level-2 Galerkin
assembly (from the fine grid through the virtual level 1) took 956 ms of it, and the smoothers'
λmax power iteration most of the rest.

**What:**
- **Assembly (`mg.wgsl`, `assemble`):** every thread recomputed each fine element's clamped-node mask
  (8 reads of `free`) and re-read its `ee`, once per matrix entry: ~4,600 global reads per thread for 64
  values per workgroup. Now each of the 64 threads loads one fine element's weight and table offset
  into workgroup memory first. The trilinear weights `wgt(c, a, b)` (512 values, 2 × 576 calls per
  thread) are tabulated per workgroup too. Level 2: 956 → 252 ms; stored levels: 51 → 21 ms.
- **λmax reuse (`gpu_mg.rs`, `gpu_opt.rs`):** the power iteration is its own program, and
  `build(estimate)` runs it only on the first iteration, after a β change and every 5 iterations.
  λmax drifts a few % over 10 iterations (about 10% over a whole run), and the random-start estimate
  itself varies ±2%. The Chebyshev interval reaches 1.1 λmax, so that drift is covered. If a solve
  with an old estimate needs over 32 CG iterations, it re-estimates and continues.

**Result:** same iteration and CG counts, compliance within 0.01%:

| | before | after |
|---|---|---|
| bracket_domain (291k) | 12.0 s | 8.3 s |
| bracket_editor | 14.2 s | 11.2 s |
| book_stand1 | 11.0 s | 7.8 s |
| table (--ground grip) | 34.2 s | 31.7 s |
| 4.8M bracket, setup per iteration | 1.47 s | 0.38 s |

**Tried and not kept: earlier β-stage advances.** The max density change sits at the move limit
(0.2) in every iteration, so stages end through the 3-iteration stall check or the 25-iteration cap.
Advancing once compliance improves < 0.5% for 2 iterations saved 15–40% of the iterations, but
cost 0.5–2.5% compliance (book_stand1 +2.5%: its β = 2 stage keeps gaining ~0.1% per iteration for
24 iterations). Keeping β = 1 strict and loosening only the later stages still cost 2.4% on
book_stand1. The slow gains are real, so the rule stays as it was.

**Still open:**
- The level-2 assembly is still 2/3 of setup. P is 27/64 dense per child, so skipping its zeros
  would roughly halve the two products.
- The solve (Chebyshev smoothing on levels 0 and 1) is now most of each iteration.

**Verified:** topopt tests pass (`kernels_match_cpu` checks the assembly against the CPU,
`gpu_loop_matches_cpu_loop`), clippy clean, before/after runs above with the old binary.
`prims::tests::shapes_with_primitives_compile` still fails, as before.

## 2026-11-13: Showing the exported mesh

**What:**
- **App:** the Print step has a "show the mesh" option, plus "edges". The view then draws the exported
  STL/3MF instead of the raymarched shape, with the same camera, so switching back and forth compares
  them in place. The file is reloaded when an export rewrites it. Faces seen from behind are blue,
  which shows holes and inverted triangles.
- **CLI:** `gen3d render model.stl [--edges]` (also .3mf and .obj), and
  `gen3d snapshot --step print --mesh [--edges]`.

**How:**
- `meshview.rs` has two pipelines in one pass with a depth buffer. A fullscreen pass draws the
  preview's sky and ground grid, without shadows. Then the mesh is rasterized from storage buffers
  (non-indexed draw, three invocations per triangle), flat shaded with the preview's lights.
- The projection is the raymarcher's (uv = 1.8 (x, y) / z in view heights), so the two line up
  exactly.
- `Mesh::load` reads binary/ASCII STL, our 3MF and OBJ.
- `Mesh::save` now writes to `NAME.part` and renames it into place, so the app never reads half a
  file.

**Also fixed:** the raymarched preview of a result with `density_filtered` had wavy faces where the
solid meets the domain's edge, the same problem the mesher fixed in the previous entry. Samples on
either side of such a face now get ∓ half a voxel, which puts the surface on the cell face. Curved
domain boundaries (the bracket's sloped top) therefore show their voxel steps in the preview again.
They looked smooth before only because of that same artifact. The exported mesh smooths them, and
the mesh view shows that.

**Still open:**
- Section, bounds and clip don't apply to the mesh view.
- Meshes load on the UI thread: about instant for a few MB; a multi-million-triangle mesh stalls a
  frame.

**Verified:** `saved_meshes_load_back` (STL, 3MF and OBJ round trips, no temporary files left).
Snapshots of the bracket result: raymarched and mesh views line up, labels and triangle count are
right, and the edges overlay works. clippy clean.

## 2026-11-13: Smoother meshes of optimized results

**Why blocky:** the optimizer wrote only the projected density. At β = 32 that is nearly binary, so
marching cubes put every vertex at an edge midpoint: voxel terraces with 45° chamfers at any
resolution.

**What:**
- **Mesh the filtered density.** Full .vti writes gain a cell array `density_filtered`: x̃ before the
  Heaviside projection, NaN outside the design domain. The projection maps x̃ = 0.5 to 0.5, so its 0.5
  surface is the same design boundary, but x̃ varies smoothly across it. When `density` is shown,
  `Volume` takes the surface (preview and mesh) from x̃; the histogram stays on `density`. Live
  previews on large grids stay density-only. Older files without the array mesh as before.
- **Flat faces on the domain's edge.** Between a domain cell and a non-domain cell (or the padding
  beyond the grid), x̃ doesn't locate the surface: it depends on the material nearby, and meshing it
  gave wavy faces inset by up to 1/3 cell (the bracket lost 1 cm³ that way). Marching-cubes vertices
  on those edges now go to the edge midpoint, the cell face. Before this, the grid box cap sat 0–1/3
  cell short of the box.
- **Taubin smoothing** (`Mesh::smooth`, λ = 0.5, μ = −0.53, uniform weights): `export --smooth N` for
  .vti grids (default 10, 0 = off), and a "smoothing" slider in the app's Print step for grids.
- **Kept inside the domain.** μ steps overshoot at convex rims and pushed flat domain faces outward by
  ~0.1 cell, past the domain. After each pass, vertices outside the 0.5 level of the trilinear domain
  indicator are moved back onto it by Newton steps (`Volume::domain_constraint`). This is one-sided:
  vertices may still move inward.

**Result (bracket_editor, 0.71 mm, 79k triangles):** p90 dihedral angle between neighbouring faces
39° before, 21° with x̃, 10° with x̃ and 10 passes. Size and volume unchanged (76.04 × 24.14 × 39.76 mm,
26.7 cm³). Smoothing takes 0.01 s.

**Still open:**
- Free design surfaces (not on the domain's edge) can still overshoot ~0.1 cell at rims: the pin
  boss's end face moved out 0.07 mm.
- Kept-solid (passive) regions are smoothed like the rest. Their crisp edges should come from the
  design's own SDF.
- Faint dimples remain where x̃ itself has features a few cells wide. Projecting vertices back onto the
  x̃ surface would bound their deviation from it.

**Verified:** tests `smoothing_evens_out_steps_and_keeps_volume` (voxel-stepped sphere: radius spread
halved, volume within 1%, watertight) and `filtered_density_gives_a_smooth_surface_inside_the_domain`
(slope on its plane within 0.02 voxel, flat face exactly on the domain's edge, smoothing stays inside
it). `gpu_loop_matches_cpu_loop` passes. Renders before and after compared by eye. clippy clean.
`prims::tests::shapes_with_primitives_compile` fails, as it already did before this change.

## 2026-11-13: Default element count 300k on the GPU

**What:** with the GPU solver, the automatic element size now aims for ~300k elements in the bounds,
instead of 40k. The CPU solvers keep 40k.
- Bracket: 0.64 mm and 291k elements, 50 iterations in 2.7 s. Compliance 8.57 against 8.96 at 1.3 mm:
  a finer, stiffer design.
- `bracket_editor`: 211k elements, 3.6 s.

**How:** `topopt::auto_elements(gpu_solver)` sets the target.
- `optimize` (CLI) uses 300k when the solver resolves to `gpu`.
- The app's Optimize step uses it when its own adapter is a hardware GPU (`App::gpu_solver`).
- The preview's `g3d_reach` default uses the GPU value, the usual case.
- `auto_voxel` no longer stops at the shape's `//! voxel`. That cap dated from the Jacobi solver, and
  the shape's voxel is only its meshing and preview resolution. Without it, the bracket (`voxel 1`)
  would have stayed at 76k.

**Note:** fixed and loaded shells are 1.5 elements thick, so finer elements make them thinner. That
shifts results: `bracket_editor`'s compliance went from 13.3 to 17.0, probably mostly from the
thinner clamped bolt-hole shells (not verified separately).

**Verified:** CLI runs above; `gen3d snapshot --step optimize` shows "element size 0.640 mm, ≤ 299250
elements". Tests pass.

## 2026-11-13: Optimizer output at large grids: background writes, previews, full result on Stop

**What:**
- The `.vti` is written on a background thread, so the solver no longer waits for the disk.
- On grids over 4M elements the live file is a preview: density only, averaged over k³ blocks to
  ≤ 4M cells. At 44M elements a full file with every field is ~900 MB, and the viewer would reload it
  each time. The full fields are written at the end.
- The app's Stop sends SIGTERM. `optimize` now catches it, finishes the current iteration and writes
  that iteration's full result. A second SIGTERM kills it as before.

**How:** `src/topopt/output.rs` (`Output`, `Snapshot`) is shared by the CPU and GPU loops.
- Throttle: a live write starts only when the writer thread is idle and writing has stayed under ~10%
  of the time. Small grids still update every iteration.
- On the GPU path a preview reads back only the densities (4 B per element); the full fields are read
  back only for the final write.
- The signal handler only sets an atomic flag and restores the default handler. Using `libc`, which
  was already in the dependency tree, adds no new crate.

**Verified:**
- 9.5M elements (preview 2×): mid-run, the file is 201×121×101 with density only (4.8 MB). SIGTERM
  after iteration 5 prints "stopped: writing the result of iteration 5", exits 0, and leaves
  401×121×201 with density, energy and displacement.
- Decimation test. 33 tests pass; clippy is clean.

## 2026-11-13: GPU optimizer memory 218 → 164 B per element (44M elements on a 10 GB card)

**What:** the GPU path fits 44M elements (5.6 s per iteration) where 35M ran out of memory before.
`optimize` prints what it allocated: "GPU memory: 7344 MB (164 B per element)".

**How:**
- Loads are lists of (DOF, value), scattered into the residual. Before, they were a dense vector per
  load case; loads touch few nodes. −12 B/node per case.
- The level-0 smoother no longer needs its own temporary. Chebyshev's "r −= K d" is a mode of the
  mat-vec kernels (fine, stored levels, and level 1's restriction). Level 1, applied through the fine
  grid, borrows two fine vectors that are idle while the V-cycle is on coarser levels. −12 B/node.
- Coarse element matrices keep only their upper triangle (300 of 576 entries; `sym` in mg.wgsl). The
  coarsest readback expands them. −17 B/element. Level 2 even got slightly faster (0.55 ms).
- The filter's weight sums are computed in the filter and sensitivity kernels, which loop over the
  neighbours anyway (no `hs` array). The design mask and the clamped-node masks come from the element
  flags and node `free` flags (no `design`, no `emask`). −12 B/element.

**Where the 164 B go:** the fine level's 7 vectors (CG r, z, p, q, the smoother's direction, the
diagonal, u) and `free`, ~88 B; 10 per-element arrays, 40 B; level-2 matrices, 19 B; the rest
(levels ≥ 1 vectors) ~17 B. The plan's ~100 B budget would need f16 storage for some vectors or a
CG variant with fewer of them.

**Measured:** per iteration, 2.7 s at 22.5M elements, 4.0 s at 34.6M, 5.6 s at 43.9M. Profile at 9.6M:
CG iteration 49 ms.

**Verified:** all tests pass, including the GPU-vs-CPU kernel, solve and loop tests. `bracket_editor`
(2 load cases) is unchanged: 13.291, 4 flips against the accurate reference.

## 2026-11-13: GPU solver 2× faster (coarse-level kernels)

**What:** a CG iteration at 9.6M elements went from 111 ms to 50 ms. An optimizer iteration takes
1.02 s at 9.5M elements (was 2.1 s) and 2.66 s at 22.5M (was 5.6 s).

**How:** an ignored test, `profile_kernels` (`cargo test --release profile_kernels -- --ignored
--nocapture`), times each piece on a 9.6M-element box. The fine mat-vec was fine at 4 ms. Two coarse
levels cost far more than their size:
- **Level 2 (stored matrices), 8.7 ms for 1/64 of the elements.** Each thread read rows of 8 matrices
  laid out 576 floats apart, so the loads didn't coalesce. Storing them entry-major
  (`mats[entry · elements + e]`) brings it to 0.7 ms. Assembly writes that layout, and the coarsest
  readback converts it back.
- **Level 1 (on the fly), 12 ms.** Moving the Pᵀ ke P table into workgroup memory didn't help. What
  did: applying level 1 as Pᵀ (D K D) P u through the fine grid (prolong, fine mat-vec, restrict). It's
  the same operator, at 4.8 ms. Its two fine-size temporaries are level 0's smoother scratch, which
  is idle while the V-cycle works on coarser levels. The on-the-fly kernel is gone; the table still
  builds level 1's diagonal and level 2.

Tried and dropped: staging `ke` in workgroup memory for the fine mat-vec (no change), and a
27-neighbour form that loads each displacement once instead of 64 times (2.7× slower: a dynamically
indexed per-thread array plus branching costs more than the redundant, cached loads). The fine
mat-vec moves ~270 MB in 4 ms, about 60 GB/s, far below the 760 GB/s the card can do. So it's
compute- or latency-bound, and the next lever is probably fewer, fused passes (each Chebyshev step is
a separate `lin` pass over the vectors).

**Verified:** all GPU tests pass, including the loop-against-CPU test. Bench compliances are identical
before and after.

## 2026-11-13: The whole optimizer iteration on the GPU (scaling plan step 4)

**What:** with `--solver gpu` (the default on hardware GPUs) every per-element step runs on the GPU
too: stiffness from densities, sensitivities through the projection, the density filter (forward and
transposed), Heaviside projection, and the OC update with its λ search. Per iteration only scalars come
back: compliance, the volume at each λ trial, change, grey share and max displacement. The fields come
back only when the `.vti` is written (throttled; the densities the solve used are kept on the GPU so
the final iteration can still be written).

**How:**
- `gpu_engine.rs`: the shared machinery, split out of the solver. It compiles the kernels of
  `mg.wgsl` and `opt.wgsl` (each `//// kernel` section is its own module with the shared prelude). It
  records programs (dispatch lists with fixed bind groups), keeps a scalar buffer, and does reductions
  (dot, weighted dot, max |a − b|, grey count, max vector norm) and readback. Every submit runs in a
  validation error scope, so a bad binding fails the run instead of only being logged; that caught a
  buffer bound for reading and writing in one dispatch.
- `gpu_opt.rs`: the loop. The λ search is the CPU's (bracket around the previous λ, then bisect), with
  the volume of each trial read back: ~18 small readbacks per iteration, negligible next to the solve.
  Convergence, β continuation and the iteration line come from shared helpers (`Control`,
  `print_iteration`), so the CPU and GPU loops can't drift apart.
- Memory: level 0's smoother residual shares CG's `q`, and the unfiltered compliance sensitivity
  shares `xnew`. Allocation runs in an out-of-memory error scope. Buffers are filled with
  `write_buffer` rather than `create_buffer_init`, which panics when the allocation fails. A grid that
  doesn't fit now says so: "35.1M elements don't fit on this GPU; use a larger --voxel".

**Measured** (bracket; mean of 2–4 iterations; RTX 3080):

| voxel | domain elements | CG its/solve | setup s | solve s | update s | io s | s/iter |
|---|---|---|---|---|---|---|---|
| 1.3 | 34k | 9 | 0.012 | 0.015 | 0.002 | 0.001 | 0.030 |
| 0.5 | 607k | 9 | 0.037 | 0.063 | 0.007 | 0.008 | 0.115 |
| 0.4 | 1.19M | 12 | 0.043 | 0.150 | 0.012 | 0.019 | 0.224 |
| 0.25 | 4.9M | 11 | 0.21 | 0.68 | 0.045 | 0.065 | 0.99 |
| 0.2 | 9.5M | 12 | 0.39 | 1.49 | 0.091 | 0.11 | 2.08 |
| 0.15 | 22.5M | 13 | 0.99 | 4.0 | 0.20 | 0.43 | 5.6 |

- The update (filter + OC) went from 5.3 s on the CPU to 0.045 s at 4.9M elements.
- The default bracket run takes 1.4 s for 50 iterations, from 77 s this morning.
- 35M elements doesn't fit in 10 GB.

**Verified:**
- New test: the GPU loop follows the CPU loop on a two-load-case cantilever with a kept-solid strip,
  with and without projection. Compliance agrees within 2e-3 at every iteration (4 digits in
  practice), and the written densities agree.
- Bracket: 8.9645 (CPU 8.9635), 8 of 13k elements flip. `bracket_editor`: 13.291 (reference 13.289),
  4 flips. `--no-projection`: 10.356, as on the CPU.
- 32 tests pass; clippy and fmt are clean.

**Open / next:**
- Memory, ~230 B per element now:
  - fine vectors: 9 + 2 per load case, 12 B per node each;
  - 13 per-element arrays;
  - level-2 matrices: 36 B per element.

  Ideas, roughly by payoff:
  - f per load case as a sparse list (loads touch few nodes);
  - flags/emask/free packed into bits;
  - drop `design` (use the flags);
  - level 2 as symmetric halves or node stencils;
  - fp16 storage for some vectors.
- Speed: per CG iteration about 0.3 s at 22M elements, more than the fine mat-vecs alone should cost
  (~5 per iteration). Level 1 on the fly (compute-bound, dynamic T indexing) and the smoother's many
  small `lin` passes are the likely costs. Next is profiling with timestamp queries.
- Setup is ~1 s at 22M: power iterations (12 per level), assembly, and the coarsest inverse round
  trip. The λmax estimates could be reused across iterations.
- Output at this scale: decimated live preview, full fields at the end (plan step 5).

## 2026-11-13: MGCG on the GPU (scaling plan step 4, first part)

**What:** `--solver gpu` runs the whole multigrid-preconditioned CG on the GPU in f32: mat-vecs,
Chebyshev smoothing, restriction/prolongation, Galerkin coarsening, CG with its scalars, and element
energies. It is the default (`--solver auto`) unless the adapter is a software renderer. The density
filter and OC update still run on the CPU; per iteration, `ee` goes up and the energies and
displacements come back.

**How (`src/topopt/gpu_mg.rs`, `mg.wgsl`):**
- Level 0: matrix-free.
- Level 1 is never stored. Its element matrices are Σ_c ee_c T[mask_c][c] on the fly, where T holds
  Pᵀ (D ke D) P for every clamped-node mask (8 bits) and child position: 2048 matrices, 4.7 MB.
  Clamped nodes need no special cases this way, and level 1 costs about a fine mat-vec in arithmetic
  but no memory per element.
- Levels ≥ 2: f32 element matrices, assembled with one 64-thread workgroup per coarse element (the
  child matrix, K P and Pᵀ K P staged through workgroup memory).
- Coarsest level: the CPU reads back its few element matrices, makes a dense inverse with the banded
  Cholesky, and uploads it; the V-cycle applies it as a GPU mat-vec. One readback per optimizer
  iteration, none inside a V-cycle.
- CG's α, β and ρ live in a small GPU buffer. Dot products are two-pass tree reductions. The residual
  norm is read back every 4 iterations.
- Each phase (setup with power iterations for λmax, CG start, CG iteration, energies) is recorded once
  as a list of dispatches with fixed bind groups and replayed.
- **f32 and rigid-body modes:** at first the bracket came out 0.35% too soft. Converging further
  didn't change it, so the operator itself was off. Rounded to f32, `ke` no longer annihilates
  translations exactly, and the error scales with the absolute displacement (large on a cantilever,
  against small differences across one element). That acts like spurious springs. The fine mat-vec
  and the energy kernel now apply `ke` to displacements relative to each element's first node.
  That's the same in exact arithmetic (and still exact next to clamped nodes, whose u is 0), and the
  GPU then matches the CPU's compliance to 7 digits. The true residual of an f32 `u` stays near 1e-2
  (rounding in the vector itself); that is noise, not solution error.

**Measured** (bracket, 4 iterations; GPU RTX 3080, CPU 12 cores):

| voxel | domain elements | CG its/solve | GPU setup s | GPU solve s | CPU-MG solve s | update (CPU) s |
|---|---|---|---|---|---|---|
| 1.3 | 34k | 9 | 0.012 | 0.015 | 0.10 | 0.04 |
| 0.5 | 607k | 9 | 0.044 | 0.069 | 1.75 | 0.76 |
| 0.4 | 1.19M | 12 | 0.058 | 0.16 | 4.9 | 1.4 |
| 0.3 | 2.8M | 12 | 0.24 | 0.43 | – | 3.2 |
| 0.25 | 4.9M | 11 | 0.43 | 0.73 | – | 5.3 |

The default bracket run (34k elements, 50 iterations) takes 3.2 s, against 7.9 s on the CPU
multigrid and 77 s at the start of the day. On large grids the CPU filter + OC is now 80% of each
iteration.

**Verified:**
- Kernel tests against the CPU path: fine and level-1 mat-vecs and diagonals, level-2 GPU assembly,
  restriction and prolongation all agree to ≤ 4e-7 relative. λmax estimates are sane.
- A realistic-scale cantilever (E·h = 4550, a hollow section with 1e-9 core, light end load) gives the
  same compliance as the f64 CPU solve to 7 digits; that test fails the old way.
- Full runs: the bracket ends at 8.9642 against 8.9635 on the CPU (8 of 13k elements flip: tolerance
  noise). `bracket_editor` ends at 13.289, the accurate reference value, with 4 flips.
- 31 tests pass; clippy is clean.

**Open / next:**
- Port the density filter, projection, sensitivities and OC update (with its bisection) to the GPU,
  leaving only scalars to read back.
- Memory: f32 level-2 matrices are 36 B per fine element and the binding limit is 2 GB, so about 55M
  elements fit as is. Storing the symmetric half, or level 2 as node stencils, raises that.
- Optional: rigid-body-relative application on level 1 and the stored levels as well (they only
  precondition, so it affects speed, not accuracy).

## 2026-11-13: Multigrid CG for topology optimization (scaling plan steps 1–3)

**What:** `optimize` now solves K u = f with multigrid-preconditioned CG (MGCG) on the CPU, as the
reference for the GPU port in `docs/topopt-scaling-plan.md`. Each iteration line shows its time split
into preconditioner setup, solve, update (filter + OC) and file output. A hidden `bench-topopt` runs the
bracket at 4 element sizes and prints a table.

**How (`src/topopt/mg.rs`):**
- The grid coarsens 2× per level; odd sizes round up, so no padding is needed. With trilinear
  interpolation each fine element lies inside one coarse element, so the Galerkin operator Pᵀ K P is
  again a sum of 24×24 element matrices (K_E = Σ P_cᵀ K_c P_c over the 8 children).
- Clamped DOFs are masked out of the fine element matrices before coarsening (D K D), so the coarse
  levels are exact Galerkin projections of the constrained system. A test checks two levels against an
  explicit P, K, Pᵀ product.
- The fine level is matrix-free; coarse levels store element matrices (the CPU's ~1.2 kB/element is
  mostly level 1, which the GPU version will compute on the fly instead).
- Smoother: Chebyshev degree 2 on [0.1, 1.1]·λmax(D⁻¹K), with λmax from 12 power iterations per level.
  Damped Jacobi is the alternative (`--smoother jacobi`); it needs ~35% more CG iterations.
- Coarsest level: banded Cholesky. Coarsening continues until it costs under 2e7 flops.
- E_min is now 1e-9 (1e-6 was a concession to Jacobi). `--solver jacobi` keeps the old path.

Step 3 changes that carry over to the GPU:
- **OC:** `dc` is normalized by the compliance and λ is bracketed around the previous iteration's
  value. That's 17–19 volume evaluations instead of ~45, so the update is 3× faster.
- **Adaptive CG tolerance:** 0.05 × the last density change, clamped to [1e-4, 1e-2]; 1e-4 in the
  final β stage. This halves the CG iterations.
- One hierarchy per iteration, shared by the load cases. There's a warning when CG hits its limit.
- `.vti` writes are throttled to stay under ~10% of the run time, and the last iteration is always
  written.
- With multigrid the element cap goes up to 8M.

**Measured** (bracket, 12 cores; mean of the first 2 iterations, same settings both ways):

| voxel | domain elements | Jacobi CG its/solve | MG CG its/solve | Jacobi s/iter | MG s/iter |
|---|---|---|---|---|---|
| 1.3 | 34k | 1046 | 8.0 | 1.9 | 0.19 |
| 0.8 | 148k | 643 | 8.5 | 4.2 | 0.86 |
| 0.5 | 607k | 1179 | 9.0 | 34.5 | 3.4 |
| 0.4 | 1.19M | 1285 | 11.5 | 75.7 | 7.7 |

At 1.2M elements, MG's time goes 1.3 s to setup, 5.0 s to the solve and 1.4 s to the update. Nearly all
of the solve is fine-level mat-vecs (~5 per CG iteration). Following the plan, those aren't tuned on
the CPU.

**Verified:**
- Full bracket run at 1.3 mm: Jacobi 90.6 s; MG with the same E_min, 16.9 s; MG with E_min 1e-9, 14.9 s.
  All three take 48 iterations and end at compliance 8.965, with no solid/void flips between the
  designs. With the OC bracket and adaptive tolerance the run takes 7.9 s and 50 iterations, ends at
  8.9635, and 19 of 13k solid elements flip.
- `--no-projection`: 10.356, as before, in 3.8 s (was 52 s).
- `bracket_editor` (2 load cases): MG ends at 13.289 and Jacobi at 13.253. A Jacobi run at CG tolerance
  1e-7 also gives 13.289 in 48 iterations. So the difference was the old solver's error at its 1e-4
  tolerance steering the design. MG at 1e-4 matches the accurate Jacobi design with 0 solid/void
  flips; old Jacobi at 1e-4 differs from it by 60.
- Tests: restriction = prolongationᵀ, Galerkin = explicit product on two levels, the Cholesky solves
  its level, and MGCG reaches the Jacobi-CG solution (40 vs 419 iterations on a strut-in-void design
  at 1e-9 contrast). An unphysical per-element checkerboard contrast does badly (163 iterations); real
  designs don't look like that. 29 tests pass; clippy is clean.

**Open / next:** step 4, the wgpu port. Planned shape:
- fine level matrix-free;
- level 1 computed on the fly from the fine `ee` and 8 precomputed Pᵀ ke P matrices (per child
  position and clamped-node mask), so it stores nothing per element;
- levels ≥ 2 as assembled f32 element matrices;
- the coarsest level as a dense inverse applied on the GPU, so nothing round-trips inside a V-cycle;
- CG scalars kept on the GPU, with the residual read back every few iterations.

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
