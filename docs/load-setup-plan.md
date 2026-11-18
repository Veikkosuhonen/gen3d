# Plan: easier supports and loads

Goal: everyday load situations ("stands on a table holding a book", "gets bumped/rocked") take a few
clicks, and set up physically sensible problems by default.

Two physics points drive it:
- Clamped feet tie the part to the ground: it can lean on the ground pulling down, or holding its feet
  together. A roller (only the normal direction held) removes the tie; removing the pull needs
  contact (see the follow-ups).
- Compliance is quadratic in the load, so +F and −F alone give the same design. Rocking only matters
  combined with the weight in the same case: weight ± sideways push.

## Todo

1. [x] **Weight loads**: a load can be given in kg (× 9.81 N, straight down) and has direction presets
   (down, ±X, ±Y, ±Z).
2. [x] **Cutting primitive as load/support**: say in the UI that a subtracted primitive can carry the
   role (its shell is the cut face), so a load on a face needs no region copy.
3. [x] **Per-axis supports**: a fixed primitive holds chosen axes (`fix: [x, y, z]`, default all).
   `Problem.fixed` becomes a per-node axis mask; CPU (Fem, MG) and GPU (mg.wgsl) solvers honour it.
   The CPU multigrid is exact for rollers; on the GPU, level 1 is exact but its diagonal and level 2
   come from the per-node table, which sees a roller as clamped. A ground of rollers needs ~3× the
   CG iterations of a clamped one (bookstand: 32 vs 12 per case).
4. [x] **Ground**: an Optimize option, "stands on the ground": the lowest layer of the domain is held
   in Z (rests on) or in all axes (glued). With "rests on", one node is pinned in X/Y and one more in
   X or Y so the part can't slide or spin. (`--ground none|slide|grip`; `stabilize` adds the pins
   for any support set that leaves rigid motions free.)
5. [x] **Rocking cases**: on a load, "also when rocked": generates cases weight ± s·(X, Y) with a
   sideways share s (default 20%); the Optimize panel lists the cases by name.
6. [x] **Checks and visuals**: arrows for loads (per case colour), markers for supports; before running,
   warn about loads/supports that reach no node and about supports that leave the part free to move.
7. [x] **Templates** (Optimize, "Quick start"): stands on a table, sticks out of a wall, hangs from
   above: add the role primitives sized to the shape's bounds and set the ground.

## Follow-ups

- **Lift-off (unilateral ground)**: release ground nodes whose reaction pulls on the part, re-solve,
  repeat (an active set per load case, every few optimizer iterations). Without it a stand can still
  hang from the ground.
  - [x] CPU (2026-11-18): `contact.rs`, `solve_contact`; `slide` and `grip` are contact, `glued` is
    the old bilateral ground; tipping check per domain piece.
  - [ ] GPU: per-case masks, the reactions on the GPU, the same rounds (it holds the ground both
    ways meanwhile).
  - [ ] Symmetry with contact: solve one case of each mirror pair and mirror its solution (now
    asymmetric cases are solved on the whole part).
- **GPU coarse levels with per-axis masks**, to get roller problems back to ~12 CG iterations.
- **Persist the ground** with the design (the sidecar, or a `//! ground` directive); it is an
  Optimize setting for now, like the material share.
- **Ground overlay at the real bottom** of the domain rather than the bounds' bottom.
