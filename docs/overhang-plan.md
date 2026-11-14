# Overhang control plan

Goal: optimized parts that print without support material, as the shape stands (+z up).

## Approach

Langelaar's additive-manufacturing filter (Langelaar 2017, "An additive manufacturing filter for
topology optimization of print-ready designs", SMO 55), between the density filter and the
Heaviside projection: x → x̃ → x̂ (AM) → x_phys. Going up from the plate, an element's printed density
is `smin(x̃_e, smax(x̂ of its 5 supports in the layer below))`. The filter is smooth and increases
with every input, so the volume sensitivities stay non-negative and the OC update works unchanged.
No constraints are added and there's no switch to MMA.

Rejected alternatives: overhang constraints on the density gradient (Qian 2017) or on surface
angles. They are extra constraints, which need MMA, and only penalise overhangs instead of ruling
them out.

## Steps

| Step | Status |
|---|---|
| CPU filter, adjoint, finite-difference and support tests (`overhang.rs`) | done |
| CPU loop, `--no-overhang`, checkpoint key, z mirror plane dropped | done |
| GPU kernels `am_fwd` / `am_adj` (one dispatch per layer), GPU-vs-CPU test | done |
| Unsupported-element report at the end of a run | done |
| App toggle (*Printable without supports*), on by default | done |
| Overhang report for any mesh in the Print step (downward area past 45°, highlighted) | open |
| Other angles: a multi-layer stencil (e.g. 2 up per 1 across for ~63°) | open, if needed |
| Print direction: any of ±x, ±y, ±z (`--up`), mesh turned to match in Print / export | done |
| Saving the print direction with the shape or result | open |
| Arbitrary print angles (resampling the grid) | open; rotate the shape for now |

## Decisions

- 45° only (5-element cross; diagonally about 55°). A 3×3 stencil would allow 35° along diagonals,
  too shallow for FDM.
- The plate is the lowest grid layer with domain elements (the slicer drops the part's lowest
  point onto the bed).
- Kept-solid elements are printed as given (density 1) and support what is above them. Overhanging
  kept-solid regions are reported separately.
- `smax` is scaled by the supports' largest density so ξ^40 doesn't underflow in f32 (the GPU
  diverged from the CPU by 0.8% after a few iterations before that).
