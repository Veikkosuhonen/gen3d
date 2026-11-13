# CLAUDE.md

gen3d is a single Rust binary (wgpu + egui) that turns WGSL signed distance functions into 3D prints:
live preview → optional topology optimization → GPU meshing → slicing to G-code. `README.md` is the
user and developer reference (shape format, directives, CLI, file-by-file role table under
"Development"); read it rather than re-deriving how things work.

## Git workflow

- **Commit directly to `master`.** No feature branches or PRs unless the user asks for one.
- **Commit every change as you go, without asking.** Each self-contained change (a feature, a fix, a
  perf step, a doc update) gets its own commit as soon as it builds and its tests pass. Don't batch
  unrelated work into one commit, and don't wait for the user to say "commit".
- Don't push unless asked.
- Commit message style (see `git log`): a short imperative subject with no prefix/tag
  ("GPU multigrid setup 3-4x faster", "Stop the final stage when compliance oscillates"), then a
  wrapped body explaining why, what changed, and measured results (timings, compliance, memory) when
  relevant.
- Don't commit generated output (`out/`, `target/`). Shapes in `shapes/` are user work: only commit
  ones you created or were asked to change.

## Build and check

```sh
cargo build --release
cargo test                      # some GPU tests are #[ignore]d; run them explicitly when touching GPU code
cargo clippy --all-targets      # keep clippy clean
cargo fmt                       # rustfmt.toml: max_width = 120
```

Headless checks of the app and optimizer:

```sh
gen3d snapshot shapes/bracket_domain.wgsl --step optimize -o out/ui.png   # render the UI to a PNG
gen3d bench-topopt [--voxels 1.3,0.8,0.5] [--iters 5] [--solver mg|jacobi]  # optimizer timing table
```

Use `cargo run --release -- <args>` when the installed `gen3d` may be stale. Write scratch output
under `out/`.

## Documentation conventions

- **`devlog.md`**: newest entry first, headed `## YYYY-MM-DD: <title>`. Each entry covers **Why**,
  **What**, **Result** (with numbers), **Verified** (tests, comparisons) and **Still open**. Add an
  entry for every substantive change, in the same commit.
- **`README.md`**: update user-facing behaviour (directives, CLI flags, app steps) and the file/role
  table in "Development" when files are added or change role.
- **`docs/*-plan.md`**: longer plans (e.g. `topopt-scaling-plan.md`); update their status sections
  when a step lands.

## Code notes

- Coordinates are millimetres, +Z up, build plate at z = 0.
- The topology optimizer has CPU (`src/topopt/mod.rs`, `mg.rs`) and GPU (`gpu_*.rs`, `*.wgsl`)
  paths that must agree; tests such as `gpu_loop_matches_cpu_loop` compare them. Keep both in sync
  when changing the algorithm.
- For performance work, measure before and after (`bench-topopt`, or `optimize` on a shape in
  `shapes/`) and report the numbers in the commit message and devlog.
