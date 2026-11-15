# Plan: block-sparse optimizer grids

Goal: the GPU optimizer's memory and time scale with the design domain, not with its bounding box.
For thin, spread-out parts (table frames, stands, legs) that is several times more elements on the
same card, at the same resolution everywhere.

## Where it stands

`setup` crops the grid to the domain's bounding box and keeps it dense. Elements outside the domain
get `ee = 0` and their nodes `free = 0`, but every buffer is sized by the box and every kernel is
dispatched over it (`apply_fine` early-outs on `free == 0`, after reading it).

- 164 B per box element on the GPU: ~11 per-element f32 arrays (`ee`, `ce`, `flags`, `x`, `xnew`,
  `xtilde`, `xphys`, `xused`, `dvs`, `dc`, `dv`) and ~7 per-node 3-vectors (`r`, `z`, `p`, `q`, `u`,
  `diag`, the smoother's `d`), plus coarse levels.
- `bracket_domain` fills 96% of its box (36,620 of 63×19×32), so it gains nothing. Thin-walled
  shapes are expected to fill much less. Unmeasured: see step 0.

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
little of a brick is wasted; 8³ would roughly double the waste on thin walls. A 64-element brick also
fits a 64-thread workgroup. Step 0 settles it with numbers.

**Memory.** ~164 B × (active slots · B³) plus ~2 B per element of tables. A shape filling 20% of its
box with ~60% brick occupancy would use ~1/3 of today's memory.

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

0. **Measure first.**
   - Log in `setup`: domain fill of the box, and for B = 4 and 8 the active brick count (with the
     node dilation) and brick occupancy.
   - Run it on the shapes in `shapes/` (those without supports with `--ground`).
   - **Go/no-go:** proceed if typical target shapes would use under ~60% of the box's memory.
     Otherwise do the scaling plan's 164 → ~100 B per element first.
1. **`src/topopt/bricks.rs`.** The brick set: build from the domain, coarsen, page and neighbour
   tables, gather/scatter dense ↔ slots, `index(i, j, k) -> Option<usize>`. Unit tests: round trip,
   an all-active set matches the dense count, the coarse set covers every fine brick's children and
   nodes. Plus the WGSL prelude helpers (`elem_slot`, `node_slot`, neighbour lookup).
2. **Port the solver kernels, one at a time**, each checked against the CPU on small grids (sparse
   GPU → scatter → compare), in the scaling plan's order: `apply_fine`, `diag_fine`,
   `restrict`/`prolong`, `diag_l1`, `assemble`/`apply_elem`/`diag_elem`, `bottom`, then a full
   MGCG solve.
3. **Port the optimizer kernels**: `stiffness`, `sens`, `filter`, `energy`, `am_fwd`/`am_adj`, then
   the whole iteration (`gpu_loop_matches_cpu_loop`, with and without symmetry, overhang, 2D).
4. **Host glue**: buffer sizes from slots in `gpu_mg.rs`/`gpu_opt.rs`, load remapping, readback
   scatter for output, checkpoints and coarse start. The dense GPU layout is removed, not kept
   beside it: an all-active brick set is the dense case.
5. **Benchmark.**
   - `bench-topopt` on `bracket_domain` (96% fill) bounds the lookup overhead: target ≤ 15% slower
     per iteration than today.
   - The sparse shapes from step 0: memory per *domain* element, and iteration time, before/after.
   - The largest grid that fits on the card, against today's ~45M box elements.
6. **Host-side sparsity, if host RAM becomes the limit**: a sparse `Problem`, `setup` sampling only
   bricks near the domain (today `MAX_SAMPLED` caps the box at 200M cells), loads as DOF lists.

## Later: narrow band

Once the layout is sparse, bricks that stay fully void for many iterations (x̄ at the minimum, with a
few voxels' margin) can be dropped from the active set, so late iterations cost what the structure
costs, not the domain (Liu et al. 2018). Needs rebuilding the hierarchy when the set changes and a
rule for letting bricks back in. Separate plan when step 5 is done.

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
