# 2D optimization plan

Goal: optimize a design domain as a 2D problem along a chosen axis (x, y or z) and extrude the result
back into 3D. For parts that are a profile extruded along an axis (a box or plate loaded in its own
plane, a bracket of constant thickness), this gives a much finer in-plane resolution for the same
time, and a part whose cross-section is the same all through, which is often what you'd print anyway.

The projection is knowingly wrong when the problem isn't a prism (a domain, support or load that
varies along the axis). That's accepted: it should do the obvious thing in the simple cases and
something reasonable otherwise.

## Approach

Project the set-up 3D problem into a **slab one element thick**, optimize that with the existing 3D
optimizer, and extrude the slab's fields back onto the 3D grid when the result is written.

A one-element slab of 8-node hexes with free faces is plane stress in the slab's plane. Reusing the
3D machinery (CPU and GPU multigrid, filter, AM filter, projection, symmetry, coarse start,
checkpoints) avoids a second optimizer with its own CPU/GPU paths to keep in sync. The cost is about
3× a dedicated 2D solver (two node layers and a third DOF per node), which is cheap next to 3D.

Rejected: a dedicated 2D (Q4, 2 DOFs per node) solver. Twice the code to keep in sync (CPU and GPU),
for a speedup the slab's cost doesn't need.

### 3D → 2D (`topopt/plane2d.rs`, at the end of `setup`)

The 3D problem is sampled as usual (same element size, same crop). For each column of elements or
nodes along the axis:

| | rule |
|---|---|
| domain | the column has any domain element (the silhouette) |
| kept solid | the column has any kept-solid element |
| supports | a node is held along the axes held anywhere in its node column |
| loads | the in-plane components summed over the column, half on each slab node layer; the out-of-plane component dropped |

**Thickness.** The slab is one element (h) thick and stands for a part `T` thick: the domain's
largest column (elements × h). The loads are scaled by h / T, so the slab's stresses, strains and
displacements are those of the T-thick part. Its compliance is h / T of the part's; the reported
compliance is multiplied back by T / h.

The 3D stabilizing pins and the rigid-body check run on the slab instead.

**Out of plane.** The slab's second node layer is held along the axis. A slab one element thin
bends out of its plane almost freely (stiffness ∝ h³); no load excites that, but it wrecked the
conditioning (GPU CG diverged on the beam's 667×223 slab). Holding one layer keeps plane stress
(the other can still move, so the slab gets thinner or thicker under load); holding both made the
beam 12% stiffer than in 3D (plane strain). It must be the second layer: multigrid interpolates the
first layer from one coarse node but the second from two, and with only the second free the coarse
matrix was singular (NaN in the coarsest Cholesky).

### 2D → 3D (`output.rs`)

The result `.vti` is on the 3D grid, with at most 24 cells along the axis (`write_grid` takes
per-axis spacing now; the beam at the slab's element size all the way through was 9M cells and
219 MB, now 2.4M and 59 MB). Every 3D element in the 3D domain gets its column's slab values (`density`, `density_filtered`, `energy`); outside the 3D domain density and energy are 0 and
`density_filtered` NaN, as usual. Displacements: the in-plane components of the slab's two node
layers averaged, out-of-plane 0. So a domain that isn't a prism is cut by its own shape again, and
voids stay empty.

### Interactions

- **Overhang control:** the AM filter sweeps the slab's layers, with the smooth max tuned for 3
  supports (Langelaar's 2D filter: Q = P + ln 3 / ln 0.5) instead of 5. With the print direction in the
  plane, the profile gets 45° overhangs, and so does its extrusion. With the print direction along
  the axis, the slab is one layer (on the plate): the extrusion is a prism standing up, always
  printable.
- **Mirror planes:** in-plane ones are used as usual. One across the axis is dropped.
- **Coarse start:** works as is (the coarse grid is projected too; the slab's origin along the axis
  is the bounds' minimum on both grids, so prolongation lines up).
- **Element size:** `auto_voxel` aims at a third of the 3D element count in the plane (100k on the
  GPU), but no finer than keeps the 3D sampling under 50M cells. A slab element costs about 3× a 3D
  one (twice the nodes per element, more levels), so this takes about the time of a 3D run.
- **Multigrid:** coarsening a 1-element axis keeps it at 1 element (`div_ceil`), so the levels stay
  slabs: 3–4 CG iterations on the solid beam, like 3D.
- **Coarsest level:** after a coarse start, the bracket's slab had a mechanism in the GPU's f32
  coarse matrix (a void link of 1e-9 lost in the sums next to solid). The Cholesky clamped its
  pivot to 1e-300, which put 1e150 into the dense inverse and inf into its f32 copy. A DOF whose
  pivot collapses (below 1e-6 of its diagonal for the GPU's matrices, 1e-12 for the CPU's) is now
  left out of the coarse solve, and a CG breakdown restarts once from zero after a careful λmax
  re-estimate.

### Where it's set

| | |
|---|---|
| CLI | `optimize --2d x|y|z` (`--3d` overrides a shape's 2D) |
| script shape | `//! 2d y` |
| editor shape | sidecar `"2d": "y"`, set from the app |
| app | Optimize: *Design space: 3D / 2D along X / Y / Z* (saved with the shape like *printed … up*) |
| result | FieldData `plane2d` (the axis), shown in the log |
| checkpoint | the key includes the axis |

## Steps

| Step | Status |
|---|---|
| `plane2d.rs`: project the problem, extrude the fields; unit tests | done |
| `setup`: project at the end, pins and rigid check on the slab; compliance scaled by T/h | done |
| Output: extrude cells and displacements when writing (coarser along the axis) | done |
| `Settings::plane2d`, `--2d` / `--3d`, checkpoint key, mirror plane across the axis dropped | done |
| `auto_voxel` for 2D | done |
| `//! 2d`, sidecar `"2d"`, app selector | done |
| AM filter with 3 supports on a slab (CPU and GPU) | done |
| Coarsest-level mechanisms and CG breakdowns (found on the bracket in 2D) | done |
| Test: the solid beam as a slab matches 3D; slab GPU = CPU | done |
| README, devlog | done |

## Still open

- Per-column thickness (a slab whose elements stand for different thicknesses) for domains that
  aren't prisms; for now one thickness for the whole slab.
- Axisymmetric (revolved) 2D.
