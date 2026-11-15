# Plan: block-sparse optimizer grids

Goal: the GPU optimizer's memory and time scale with the material, not with the bounding box. Two
kinds of empty space, and the gains multiply:

- **Outside the domain** (static): the box around thin, spread-out parts (table frames, stands, legs).
  Known at setup.
- **Void inside the domain** (dynamic): at a 0.3 volume fraction most of the domain ends up void.
  Known once the layout has formed, which the coarse-start stage already does before the fine
  stage, where nearly all the time is spent (see "Void: the fine stage starts sparse").

## Where it stands

`setup` crops the grid to the domain's bounding box and keeps it dense. Elements outside the domain
get `ee = 0` and their nodes `free = 0`, but every buffer is sized by the box and every kernel is
dispatched over it (`apply_fine` early-outs on `free == 0`, after reading it).

- 164 B per box element on the GPU: ~11 per-element f32 arrays (`ee`, `ce`, `flags`, `x`, `xnew`,
  `xtilde`, `xphys`, `xused`, `dvs`, `dc`, `dv`) and ~7 per-node 3-vectors (`r`, `z`, `p`, `q`, `u`,
  `diag`, the smoother's `d`), plus coarse levels.
- `bracket_domain` fills 96% of its box (36,620 of 63×19×32), and the editor
  shapes in `shapes/` fill 61–100% (see "Void").

Rejected: a true adaptive octree (varying element size). Hanging nodes, a non-conforming multigrid,
filters across element sizes and remeshing as the design moves; a research project, and printable
parts are mostly boundary, which it would keep fine anyway.

## Layout

**Bricks.** The element grid is tiled by bricks of `B = 4` elements per axis (per axis: an axis one
element thick, as a 2D slab's, gets a brick extent of 1). Only *active* bricks are stored.

- **One brick set for elements and nodes.** Node (i, j, k) lives at the same coordinates as element
  (i, j, k), so the nodes on an element's +x/+y/+z faces can fall in the next brick. The active set
  is the bricks holding a domain element, plus the bricks holding an active node (that dilation is
  usually one brick layer on the + sides). Elements in a node-only brick are flagged void.
- **Slot-major arrays.** Element/node `(slot, local)` is at `slot · B³ + local`, local x-fastest.
  Vectors stay flat arrays, so `lin`, `dot_partial`, `scalar`, `fill_random`, `oc` and the CG loop
  don't change, only their lengths.
- **Lookups.** A page table (one u32 per box brick, `NONE` when empty: 4 B per 64 box elements) and
  a per-slot neighbour table (27 slots, the 3×3×3 bricks around it: ~1.7 B per element). Stencil
  kernels find their brick's neighbours once and index within them; a missing brick reads as void
  (0) and is never written.
- **Every level is sparse, with the same `B`.** Coarse element c covers fine elements 2c..2c+1, so
  coarse brick = fine brick / 2 per axis; the coarse active set is the fine set halved plus its node
  dilation. One indexing scheme for every level, and coarse levels stay ~1/8 of the one above.
- **The coarsest level** solves over its active DOFs only (compacted), so the dense inverse is sized
  by the part, not the box.

**Why B = 4.** At the 0.5–1 mm voxels used for printing, a 2–4 mm brick is about one wall thick, so
little of a brick is wasted. On the optimized bracket, 8³ bricks keep 74–84% of the domain live
against 62–74% for 4³ (see "Void"). A 64-element brick also fits a 64-thread workgroup.

**Memory.** ~164 B × (active slots · B³) plus ~2 B per element of tables. A shape filling 20% of its
box with ~60% brick occupancy would use ~1/3 of today's memory.

## Void: the fine stage starts sparse

With coarse start (the default), the β = 1 stage runs on a 2× coarser grid and the layout forms
there; the fine stage continues from it. Measured on `bracket_domain` (default settings, 0.5 mm:
606k domain elements, 99% of its box; 0.35 mm: 1.78M):

| | live elements (x > 1e-3, dilated 2 voxels) | live 4³ bricks | 8³ bricks |
|---|---|---|---|
| fine stage, first iteration (0.5 mm, it 67 of 126) | 65% of domain | 74% | 84% |
| fine stage, it 80 | 62% | 71% | 81% |
| final (0.5 mm) | 50% | 62% | 74% |
| final (0.35 mm) | 47% | 54% | 64% |

The shapes in `shapes/` (default voxel and settings, `--ground slide`; `wide_stand_leg` with
`--ground grip`, see "Still open"). Percentages of the bounding box, 4³ bricks, live = x > 1e-3
dilated by 2 voxels:

| shape | grid | domain bricks | live bricks, fine start | live bricks, final | coarse / fine stage time |
|---|---|---|---|---|---|
| `bracket_domain` | 160×48×80 @ 0.5 mm | 99% | 74% | 62% | 6.3 / 21.4 s |
| `table` | 52×52×107 @ 3.9 mm | 100% | 87% | 66% | 5.7 / 8.6 s |
| `monitor_table` (2D) | 1×440×210 @ 1.1 mm | 97% | 42% | 38% | 4.7 / 14.9 s |
| `book_stand1` | 67×44×79 @ 3 mm | 61% | 53% | 42% | 2.9 / 9.6 s |
| `stand_leg` | 26×158×73 @ 1.4 mm | 95% | 60% | 48% | 2.5 / 7.4 s |
| `wide_stand_leg` | 76×68×56 @ 2.7 mm | 80% | 62% | 49% | 2.9 / 9.2 s |

- **The fine stage never leaves its starting set.** On every shape, no element of the final design
  with x > 1e-3 lies outside the first fine iteration's live set dilated by 2 voxels. So the active
  set can be fixed once, when the fine stage starts: no reallocation during a stage.
- **The fine stage is most of the time**: 60–80% of each run.
- **The box is not where the space is.** Editor shapes start from a box design primitive, so the
  domain fills 61–100% of the box. The void is the bigger saving on every shape but `book_stand1`.
- **The gain is moderate on a compact part**: ~1.35× fewer bricks for the whole fine stage, 1.6–1.9×
  if the set were shrunk later. The floor is the solid fraction (38% here, kept-solid included) plus
  the margin; it improves with resolution (the margin is fixed in voxels) and at lower volume
  fractions. 8³ bricks lose most of it, which settles B = 4.

**Rule.** At the fine stage's start, a brick is active when it holds a node of an element that is
kept solid, supported, loaded, or within `margin` (2 voxels, at least the filter radius) of a
prolonged density above 1e-3. Everything else is fixed at x = 0.

**What changes in the method:**
- Dropped elements have x = 0 exactly instead of E_min stiffness. Nodes touching only dropped
  elements leave the system; that changes K by E_min/E0 = 1e-9 relative, below the f32 solve's
  tolerance.
- The filter must still **count dropped elements as domain elements with x = 0** (not exclude them),
  or the weight sums renormalize and densities near the set's edge shift. They are in the page
  table's "absent" bricks, so `filter`/`sens` need a per-box-brick "domain but dropped" bit, or
  (simpler) the weight sums are precomputed once on the dense grid at the stage start.
- The volume constraint counts them as 0, which they are.
- Output, checkpoints and resume scatter as before, with dropped elements written as 0.
- The coarse stage stays as it is (box-sparse only): it's the grey phase and 1/8 of the cost.

## What stays dense

- **The CPU path** (`Fem`, `mg.rs`, `Filter`) stays dense. It's the test oracle for small grids;
  GPU results are scattered to dense to compare (`gpu_loop_matches_cpu_loop` etc. keep working).
- **`Problem` on the host** (`domain`, `passive`, `fixed`, `loads`) stays dense at first. The GPU is
  the limit; host RAM is usually larger. `GpuMg::new`/`optimize_gpu` gather into slots, and loads'
  DOF lists are remapped to slot indices.
- **Output, checkpoints, coarse-to-fine continuation, symmetry and 2D extrusion** read back and
  scatter to dense, as they do now. Their formats don't change.

## Kernels

| kernel | change |
|---|---|
| `lin`, `dot_*`, `scalar`, `fill_random`, `oc`, `project`, `reduce_design`, `scatter_add` | lengths only |
| `apply_fine`, `diag_fine`, `energy`, `stiffness` | the 8 elements around a node / 8 nodes of an element via the neighbour table |
| `restrict`, `prolong` | 3×3×3 fine nodes around a coarse node, across two layouts |
| `diag_l1`, `assemble`, `apply_elem`, `diag_elem` | per active coarse element; fine children via lookup |
| `bottom` | compacted active DOFs |
| `filter`, `sens` | the rmin neighbourhood spans neighbouring bricks; radius > B needs a wider neighbour table or the page table (use the page table: rmin is often 2–3 voxels but not bounded) |
| `am_fwd`, `am_adj` | the layer sweep still dispatches over the box's lateral area per layer and looks up each element (box-sized thread count, but no box-sized memory); per-layer slot lists later if it shows in profiles |

Mirror planes in `filter`/`sens` keep working: mirror the coordinate, then look it up.

## Steps

0. **Measure first.** Done (2026-11-15), see the table in "Void": the fine stage would use 42–87% of
   today's bricks (median ~60%) with the set fixed at its start, 38–66% if it were shrunk to the
   final design. About **1.6× less memory and work for the fine stage** on these shapes, at their
   current resolutions; more at finer ones (the bracket: 62% at 0.5 mm, 54% at 0.35 mm final).
   That is about the same as the scaling plan's 164 → ~100 B per element. The two multiply, but the
   other one is less work and helps every shape, so it may be the one to do first. Sparsity also
   cuts time, which that doesn't.
1. **`src/topopt/bricks.rs`.** Done (2026-11-15). Page tables and per-slot coordinates, no
   neighbour table: kernels load the 27 slots around their brick into workgroup memory (`nbhd` in
   mg.wgsl). Shared WGSL goes in `//// lib` sections, pulled in by `#include`.
2. **Port the solver kernels.** Done (2026-11-15): every level is brick-sparse. `kernels_match_cpu`
   runs on full and shell-shaped domains, 3D and slabs: all agree with the CPU to f32 rounding.
3. **Port the optimizer kernels.** Done (2026-11-15). `filter`/`sens`/`am_*` look neighbours up
   through the page table; absent elements read as outside the domain.
4. **Host glue.** Done (2026-11-15): the GPU optimizer stores the bricks of the domain; the dense
   layout is gone. Slabs use 2×8×4-like bricks (kernels compiled per shape).
5. **Benchmark the box sparsity.**
   - `bench-topopt` on `bracket_domain` (96% fill) bounds the lookup overhead: target ≤ 15% slower
     per iteration than today.
   - The sparse shapes from step 0: memory per *domain* element, and iteration time, before/after.
   - The largest grid that fits on the card, against today's ~45M box elements.

   Partly done (2026-11-15, see the devlog): `bench-topopt` +8/+15/+17% at 1.3/0.8/0.5 mm, CG
   iteration on a 9.6M box +11% (accepted: tuning further would be tuning for one GPU).
   `book_stand1` 53 → 39 MB. **Coarse-level padding**: on small grids a coarse level rounds up
   to bricks (level 2 of `stand_leg`: 3,840 stored nodes for 2,100), and level-2+ element
   matrices cost 1200 B each, so `stand_leg` went 44 → 57 MB and the full bracket at 0.3 mm
   409 → 449 MB. Fix if it matters: store element matrices only for bricks that hold an active
   element (node-only bricks' matrices are zero). Largest grid: not measured yet.
6. **The fine stage starts sparse** (see "Void"): the active set from the prolonged coarse design,
   the filter's dropped-element handling, dropped elements written as 0. Verify: compliance within
   0.5% of the full-domain run on `bracket_domain` and the shapes from step 0, with the brick count
   and fine-stage time before/after (expect ~1.35× on the bracket). Without coarse start (`--no-coarse-start`, or
   grids too small for it) the whole run stays domain-sparse.
   Done (2026-11-15): `Stage::live`, set for the fine stage after a coarse start and for a
   resumed fine stage; `gpu_opt::live_elements` is the rule above (a 2D slab's hold across its
   thin axis and mirror rollers don't count as supports). The filter's weight sums are computed on
   the host over the whole domain (`wsum`), which also drops the kernels' per-iteration weight-sum
   loops. Compliance within 0.13% of the domain-brick run on all six shapes; live bricks 46–89%
   of the domain's; CG iteration 1.37× faster on the bracket, 3.4× on `monitor_table` (devlog).
7. **Host-side sparsity, if host RAM becomes the limit**: a sparse `Problem`, `setup` sampling only
   bricks near the domain (today `MAX_SAMPLED` caps the box at 200M cells), loads as DOF lists.

## Later: shrinking during a stage

The bracket's live set keeps shrinking through the fine stage (74% → 62% of bricks). Re-cutting the
set once or twice (say when β doubles) would take the remaining gain (Liu et al. 2018's narrow band),
at the cost of rebuilding the hierarchy and buffers mid-run. Bricks only leave, never come back: the
measurement above shows the design doesn't grow past a 2-voxel margin. Worth it only if step 6's
numbers show the late fine iterations dominate.

## Still open (found while measuring)

- `wide_stand_leg` with `--ground slide` is a mechanism: max displacement 36 m from the first
  iteration and 5 km at the end, one iteration with 20k CG iterations (64 s), 535 s in all. Mirrored in
  X and Y (1/4 solved), with 8 pinned nodes added. With `--ground grip` it is fine (12.8 s). Not
  related to this plan.

- `bracket_domain` with `--ground slide` breaks down the same way (found 2026-11-15, the original
  dense code too): from β = 32 (iteration 35 of the run) CG doesn't converge in 20,000
  iterations (~210 s each) and compliance reaches 1e12. Fine with its default ground.
- 2D slabs on bricks: `monitor_table`'s CG iteration was 14.7 ms dense, 19.4 ms on domain bricks
  (2×8×4 bricks, unprofiled); live bricks bring it to 5.7 ms.

## Risks and open questions

- **Lookup overhead in `apply_fine`**, the dominant kernel. Mitigation: one neighbour-table read per
  thread, and (if needed) staging the brick plus its halo of `u` in workgroup memory.
- **Node dilation** can nearly double the brick count for walls one brick thick. Step 0 measures it;
  a node-only brick could store just its −face nodes if it matters.
- **Symmetry groups** share `ee`, `ce` and CG scratch vectors across hierarchies: they share the
  brick set too, which holds since all groups use the same grid.
- **Workgroup size**: kernels use 128 threads; 2 bricks per workgroup, or 64-thread workgroups for
  the stencil kernels.

## References

- Liu, Hu, Zhu, Matusik & Sifakis (2018), *Narrow-band topology optimization on a sparsely populated
  grid* (SPGrid; >1B voxels on one machine).
- Setaluri, Aanjaneya, Bauer & Sifakis (2014), *SPGrid: a sparse paged grid structure applied to
  adaptive smoke simulation*.
- Museth (2013), *VDB: high-resolution sparse volumes with dynamic topology*.
