# Plan: topology optimization at very large grids

Goal: `gen3d optimize` on grids of tens of millions of elements (tens of millions to ~100M
elements on a 10 GB GPU), with the live preview still working.

## Where it stands

`src/topopt.rs` is top88/top3d-style SIMP (minimum compliance):

- One trilinear 8-node hex per voxel. A single `hex_ke` (24×24) is scaled by `E_e·h` per element.
- K·u is matrix-free, gathered per node, and parallel with rayon.
- Jacobi-preconditioned CG: tolerance 1e-4, up to 20k iterations, warm-started; one solve per load
  case.
- Cone density filter → Heaviside projection (β continuation 1 → 32) → OC update (move 0.2, bisection
  on λ). The `.vti` is rewritten every iteration.

Measured on `bracket_domain` at 1.3 mm (62×19×31, 34k domain elements): **~1000 CG iterations and
~1.9 s per optimizer iteration**, about 2 ms per mat-vec. Almost all of the time is the solve.

**Status (2026-11-13):** steps 1–3 are done (see the devlog). CPU MGCG needs 8–12 CG iterations per
solve from 34k to 1.2M elements, against ~1000–1300 for Jacobi, and gives the same designs. At 1.2M
elements an iteration takes 7.7 s instead of 76 s. Step 4's port is done: the whole iteration runs on the GPU,
0.14 s at 1.2M elements, 1.0 s at 9.5M, 2.7 s at 22M. Memory is 164 B per element (budget ~100), so
a 10 GB card holds ~45M elements (5.6 s per iteration at 44M).

## Why the solver comes before the hardware

- **Jacobi CG doesn't scale.** Its iteration count grows roughly with resolution (and with the 1e6
  stiffness contrast), so expect ~10⁴ CG iterations at tens of millions of elements. A
  multigrid-preconditioned CG (MGCG) stays at ~20–50 at any size: a 100–300× gain at large grids.
- **The GPU is a constant factor.** The mat-vec is limited by memory bandwidth: RTX 3080 ≈ 760 GB/s
  against ~60–80 GB/s for desktop RAM, so ~10×.
- **Both are needed, and they multiply.** But a fast GPU Jacobi solver is a dead end, and the
  multigrid design determines what the GPU kernels are. So the solver is chosen with the GPU in mind
  from day one, and CPU micro-tuning is skipped.

## Target architecture

**Everything on the GPU, f32.**
- Consumer GPUs run f64 at 1/64 rate, and wgpu's f64 support is patchy. MGCG in f32 is fine at a
  1e-3 to 1e-5 tolerance.
- Dot products are tree reductions in workgroup memory.

**The whole optimizer iteration stays resident**, not just the solve:
- The kernels: K·u, smoothers, restriction/prolongation, density filter, projection, sensitivities,
  element energy, and the OC update with its λ bisection.
- The bisection re-filters ~50 times per iteration, so a round-trip to the CPU would kill it.
- Only scalars (compliance, volume, max change, grey fraction) are read back each iteration.

**Multigrid:**
- Coarsen 2× per level on the voxel grid. Pad the grid dimensions to multiples of 2^levels.
- Smoother: damped Jacobi (ω ≈ 0.6) or Chebyshev. Not Gauss–Seidel: it doesn't parallelize.
- Fine level (and maybe level 1): matrix-free with the shared `ke`.
- Coarser levels: Galerkin `Pᵀ K P`, stored as 27-point stencils of 3×3 blocks per node.
  Re-discretized coarse levels lose robustness at the 10⁶ stiffness contrast.
- Coarsest level (a few thousand nodes): a Cholesky solve on the CPU, or many Jacobi-CG iterations on
  the GPU, whichever is simpler with good convergence.
- With MG, E_min can drop from 1e-6 to the usual 1e-9 (1e-6 was a concession to Jacobi).

**Memory budget (10 GB):** ~100 B per element.
- ~6 CG vectors × 3 components × f32 per node, for each load case.
- ~5 per-element arrays (x, x̃, x̄, dc, dv).
- Coarse levels add ~1/7.

That gives **~80–100M elements for one load case**; each extra case costs ~20 B per element (its
displacement for warm starts, and its CG vectors if solved concurrently).

**wgpu constraints:**
- Request the adapter's real limits: the default `max_storage_buffer_binding_size` is 128 MB.
- Keep each buffer under 4 GB so u32 indexing works; shard if needed.
- Check CG convergence only every k iterations: each readback is a sync point.
- Setup already samples fields on the GPU (`sample_field`). Its output should go straight into the
  solver's buffers instead of round-tripping through `Vec`s.

**Output:**
- A full-resolution `.vti` at 100M elements is over 1 GB.
- Live preview: density only, decimated, at most every few seconds, written off the solver thread.
- Full fields (energy, displacement): only at the end, or on Stop.

## Steps

1. **Benchmark harness.**
   - Per-iteration timing broken down into solve, filter + OC, and I/O, and CG iterations per solve.
   - A fixed set of 3–4 grid sizes of the bracket. Every later step is measured against it.
2. **MGCG on the CPU, as the reference.**
   - Plain Rust, small, written in the GPU's shape: structured grid, Jacobi/Chebyshev smoother,
     Galerkin coarse levels, f32-tolerant.
   - Settle the hardware-independent questions here:
     - smoother and its sweep count;
     - number of levels;
     - Galerkin vs re-discretized coarse levels at high contrast;
     - E_min;
     - the coarsest-level solve.
   - Success: CG iteration count roughly flat as the grid gets finer, and identical optimized
     designs (compliance within tolerance) against the current Jacobi solver.
   - This path stays as the test oracle and the fallback without a GPU.
3. **Algorithmic fixes that carry over to any backend** (do them alongside step 2):
   - Adaptive CG tolerance: loose (1e-2 to 1e-3) while the design is grey; tighten as β rises and the
     change shrinks.
   - OC: normalize `dc` by the compliance, and bracket λ around the previous iteration's value instead
     of 0 to 1e9.
   - Build the preconditioner and hierarchy once per optimizer iteration, shared by all load cases.
   - Warn when CG hits `maxit`.
   - Throttle the `.vti` writes.
4. **Port the full iteration to wgpu.**
   - One kernel at a time behind the same interface, each checked against the CPU path on small grids
     (mat-vec, then smoother, transfer operators, CG, filter, OC).
   - Then grow the grid until the memory budget is the limit.
5. **Scaling extras.**
   - Coarse-to-fine continuation: optimize at 2h, then prolongate the densities to h. Faster overall,
     and a quick first preview.
   - Helmholtz (PDE) filter solved with the same multigrid, so a large `rmin` costs O(n) instead of
     O(n·r³).
   - Block CG or concurrent solves for multiple load cases.
   - Decimated or streamed output for the viewer and mesher at this scale.

## Explicitly skipped

CPU mat-vec restructuring (27-point stencil, SIMD), f32 on the CPU, and parallelizing the CPU OC loops.
They are throwaway once the iteration runs on the GPU.

## References

- Amir, Aage & Lazarov (2014), *On multigrid-CG for efficient topology optimization*.
- Aage, Andreassen, Lazarov & Sigmund (2017), *Giga-voxel computational morphogenesis for structural
  design* (matrix-free fine level, assembled coarse levels).
- Wu, Dick & Westermann (2016), *A system for high-resolution topology optimization* (GPU multigrid).
- Lazarov & Sigmund (2011), *Filters in topology optimization based on Helmholtz-type differential
  equations*.
- Wang, Lazarov & Sigmund (2011), *On projection methods, convergence and robust formulations in
  topology optimization*.
