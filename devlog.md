# gen3d devlog

Newest entries first. Each entry covers what changed and why, the design decisions, how it was verified,
and what is still open.

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
