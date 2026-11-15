//! Block-sparse grids (PoC, `docs/sparse-grid-plan.md` step 1): the element/node grid tiled by
//! bricks of B³ = 4³, only the active ones stored. Element and node (i, j, k) share brick
//! coordinates, so one brick set serves both; a brick is active when it holds a node of an active
//! element. Values are slot-major: (slot, local) at `slot · 64 + local`, local x-fastest.
//!
//! Lookups: a page table (one u32 per box brick, `NONE` when absent) on the host, and per slot the
//! slots of its 3×3×3 neighbourhood (itself in the middle) for the GPU's stencil kernels.

#![allow(dead_code)]

pub const B: usize = 4;
pub const B3: usize = B * B * B;
pub const NONE: u32 = u32::MAX;

pub struct Bricks {
    /// elements per axis
    pub n: [usize; 3],
    /// bricks per axis over the box's nodes (⌈(n + 1) / B⌉)
    pub nb: [usize; 3],
    /// per box brick: its slot, or NONE
    pub page: Vec<u32>,
    /// per slot: its brick's coordinates
    pub coords: Vec<[u32; 3]>,
    /// per slot: the slots of the 27 bricks around it (dx + 3 dy + 9 dz, d ∈ {-1, 0, 1} + 1), NONE
    /// where absent
    pub nbr: Vec<u32>,
}

impl Bricks {
    /// The bricks holding a node of an `active` element (dense, x-fastest over `n`).
    pub fn from_elements(n: [usize; 3], active: &[bool]) -> Self {
        let nb = n.map(|v| (v + 1).div_ceil(B));
        let mut used = vec![false; nb.iter().product()];
        let bi = |b: [usize; 3]| (b[2] * nb[1] + b[1]) * nb[0] + b[0];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    if !active[(k * n[1] + j) * n[0] + i] {
                        continue;
                    }
                    // the element's 8 nodes; its own brick is node 0's
                    for c in 0..8 {
                        let p = [i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1)];
                        used[bi(p.map(|v| v / B))] = true;
                    }
                }
            }
        }
        Self::from_used(n, nb, &used)
    }

    /// Every brick of the box (the dense case).
    pub fn all(n: [usize; 3]) -> Self {
        let nb = n.map(|v| (v + 1).div_ceil(B));
        Self::from_used(n, nb, &vec![true; nb.iter().product()])
    }

    fn from_used(n: [usize; 3], nb: [usize; 3], used: &[bool]) -> Self {
        let mut page = vec![NONE; used.len()];
        let mut coords = vec![];
        for (b, &u) in used.iter().enumerate() {
            if u {
                page[b] = coords.len() as u32;
                coords.push([
                    (b % nb[0]) as u32,
                    ((b / nb[0]) % nb[1]) as u32,
                    (b / (nb[0] * nb[1])) as u32,
                ]);
            }
        }
        let mut nbr = Vec::with_capacity(27 * coords.len());
        for c in &coords {
            for d in 0..27 {
                let q = [d % 3, d / 3 % 3, d / 9].map(|v| v as i64 - 1);
                let b: [i64; 3] = std::array::from_fn(|a| c[a] as i64 + q[a]);
                let inside = (0..3).all(|a| b[a] >= 0 && b[a] < nb[a] as i64);
                nbr.push(if inside {
                    page[((b[2] as usize * nb[1]) + b[1] as usize) * nb[0] + b[0] as usize]
                } else {
                    NONE
                });
            }
        }
        Bricks {
            n,
            nb,
            page,
            coords,
            nbr,
        }
    }

    pub fn slots(&self) -> usize {
        self.coords.len()
    }

    /// Stored values per field (elements or nodes).
    pub fn len(&self) -> usize {
        self.slots() * B3
    }

    /// Sparse index of grid point (i, j, k), if its brick is stored.
    pub fn index(&self, p: [usize; 3]) -> Option<usize> {
        let b = p.map(|v| v / B);
        if (0..3).any(|a| b[a] >= self.nb[a]) {
            return None;
        }
        let s = self.page[(b[2] * self.nb[1] + b[1]) * self.nb[0] + b[0]];
        (s != NONE).then(|| s as usize * B3 + (p[2] % B * B + p[1] % B) * B + p[0] % B)
    }

    /// Dense → slots, `width` values per point, over a grid of `dims` points (n for elements, n + 1
    /// for nodes); brick points outside `dims` get `fill`.
    pub fn gather<T: Copy>(&self, dense: &[T], dims: [usize; 3], width: usize, fill: T) -> Vec<T> {
        let mut out = vec![fill; self.len() * width];
        for (s, c) in self.coords.iter().enumerate() {
            for l in 0..B3 {
                let p = [l % B, l / B % B, l / (B * B)].map(|v| v as u32);
                let p: [usize; 3] = std::array::from_fn(|a| (c[a] * B as u32 + p[a]) as usize);
                if (0..3).any(|a| p[a] >= dims[a]) {
                    continue;
                }
                let d = (p[2] * dims[1] + p[1]) * dims[0] + p[0];
                let o = s * B3 + l;
                out[o * width..(o + 1) * width].copy_from_slice(&dense[d * width..(d + 1) * width]);
            }
        }
        out
    }

    /// Slots → dense: the inverse of `gather`, `fill` where no brick is stored.
    pub fn scatter<T: Copy>(&self, sparse: &[T], dims: [usize; 3], width: usize, fill: T) -> Vec<T> {
        let mut out = vec![fill; dims.iter().product::<usize>() * width];
        for (s, c) in self.coords.iter().enumerate() {
            for l in 0..B3 {
                let p = [l % B, l / B % B, l / (B * B)].map(|v| v as u32);
                let p: [usize; 3] = std::array::from_fn(|a| (c[a] * B as u32 + p[a]) as usize);
                if (0..3).any(|a| p[a] >= dims[a]) {
                    continue;
                }
                let d = (p[2] * dims[1] + p[1]) * dims[0] + p[0];
                let o = s * B3 + l;
                out[d * width..(d + 1) * width].copy_from_slice(&sparse[o * width..(o + 1) * width]);
            }
        }
        out
    }

    /// The next coarser level (elements 2c, 2c + 1 → c): the parents of every element in a stored
    /// brick, so the coarse set covers each fine brick's children and their nodes.
    pub fn coarsen(&self) -> Bricks {
        let nc = self.n.map(|v| v.div_ceil(2));
        let mut active = vec![false; nc.iter().product()];
        for c in &self.coords {
            for l in 0..B3 {
                let p: [usize; 3] = std::array::from_fn(|a| c[a] as usize * B + [l % B, l / B % B, l / (B * B)][a]);
                if (0..3).all(|a| p[a] < self.n[a]) {
                    let q = p.map(|v| v / 2);
                    active[(q[2] * nc[1] + q[1]) * nc[0] + q[0]] = true;
                }
            }
        }
        Bricks::from_elements(nc, &active)
    }

    /// Bytes of the GPU-side tables (the neighbour table).
    pub fn table_bytes(&self) -> usize {
        self.nbr.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topopt::Problem;
    use crate::topopt::gpu_engine::{Engine, Params, dims4, groups, threads};

    fn ball(n: [usize; 3]) -> Vec<bool> {
        // a hollow ball: a shell, so many bricks are absent inside and outside it
        let c = n.map(|v| v as f64 / 2.0);
        let r = c.iter().cloned().fold(f64::MAX, f64::min);
        (0..n.iter().product::<usize>())
            .map(|e| {
                let p = [e % n[0], e / n[0] % n[1], e / (n[0] * n[1])];
                let d = (0..3).map(|a| (p[a] as f64 + 0.5 - c[a]).powi(2)).sum::<f64>().sqrt();
                d < r && d > 0.6 * r
            })
            .collect()
    }

    #[test]
    fn round_trip_and_index() {
        let n = [13, 9, 22];
        let dom = ball(n);
        let b = Bricks::from_elements(n, &dom);
        let nodes = n.map(|v| v + 1);
        assert!(b.slots() < b.page.len(), "{} of {}", b.slots(), b.page.len());
        // every node of an active element is stored
        for e in (0..dom.len()).filter(|&e| dom[e]) {
            let p = [e % n[0], e / n[0] % n[1], e / (n[0] * n[1])];
            for c in 0..8 {
                assert!(
                    b.index([p[0] + (c & 1), p[1] + (c >> 1 & 1), p[2] + (c >> 2 & 1)])
                        .is_some()
                );
            }
        }
        // scatter(gather(v)) keeps v where stored and fills 0 elsewhere
        let v: Vec<f32> = (0..3 * nodes.iter().product::<usize>())
            .map(|i| i as f32 + 1.0)
            .collect();
        let s = b.gather(&v, nodes, 3, 0.0);
        let back = b.scatter(&s, nodes, 3, 0.0);
        for m in 0..nodes.iter().product::<usize>() {
            let p = [m % nodes[0], m / nodes[0] % nodes[1], m / (nodes[0] * nodes[1])];
            match b.index(p) {
                Some(i) => {
                    assert_eq!(&back[3 * m..3 * m + 3], &v[3 * m..3 * m + 3]);
                    assert_eq!(&s[3 * i..3 * i + 3], &v[3 * m..3 * m + 3]);
                }
                None => assert_eq!(&back[3 * m..3 * m + 3], &[0.0; 3]),
            }
        }
    }

    #[test]
    fn all_active_is_dense() {
        let n = [7, 8, 9];
        let b = Bricks::all(n);
        assert_eq!(b.slots(), 2 * 3 * 3);
        let full = Bricks::from_elements(n, &vec![true; 7 * 8 * 9]);
        assert_eq!(full.slots(), b.slots());
        assert!(b.len() >= 8 * 9 * 10);
    }

    #[test]
    fn coarse_covers_children() {
        let n = [30, 17, 25];
        let b = Bricks::from_elements(n, &ball(n));
        let c = b.coarsen();
        assert!(c.slots() <= b.slots());
        for s in &b.coords {
            for l in 0..B3 {
                let p: [usize; 3] = std::array::from_fn(|a| s[a] as usize * B + [l % B, l / B % B, l / (B * B)][a]);
                if (0..3).all(|a| p[a] < n[a]) {
                    // the coarse element and its 8 nodes
                    let q = p.map(|v| v / 2);
                    for k in 0..8 {
                        assert!(
                            c.index([q[0] + (k & 1), q[1] + (k >> 1 & 1), q[2] + (k >> 2 & 1)])
                                .is_some()
                        );
                    }
                }
            }
        }
    }

    /// A problem on `dom` clamped at z = 0, nodes and loads as in `gpu_mg`'s tests.
    fn problem(n: [usize; 3], dom: Vec<bool>) -> Problem {
        let nn: usize = n.iter().map(|v| v + 1).product();
        let ne = dom.len();
        let mut p = Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            domain: dom,
            passive: vec![false; ne],
            fixed: vec![0; nn],
            loads: vec![vec![[0.0; 3]; nn]],
            cases: vec![1],
            extrude: None,
        };
        for j in 0..=n[1] {
            for i in 0..=n[0] {
                let m = p.node(i, j, 0);
                p.fixed[m] = 7;
            }
        }
        p
    }

    struct Case {
        dense: (Vec<f32>, Vec<u32>, Vec<f32>),
        sparse: (Vec<f32>, Vec<u32>, Vec<f32>),
        b: Bricks,
    }

    /// Dense and slot-major ee, free and u for a problem.
    fn case(p: &Problem) -> Case {
        let n = p.n;
        let nodes = n.map(|v| v + 1);
        let free: Vec<u32> = p.free_mask().iter().map(|&f| f as u32).collect();
        let ee: Vec<f32> = (0..p.domain.len())
            .map(|e| {
                if p.domain[e] {
                    if e % 3 == 0 { 1e-3 } else { 1.0 + (e % 5) as f32 }
                } else {
                    0.0
                }
            })
            .collect();
        let u: Vec<f32> = (0..3 * free.len())
            .map(|i| {
                if free[i / 3] >> (i % 3) & 1 == 1 {
                    ((i * 7919) % 23) as f32 / 11.0 - 1.0
                } else {
                    0.0
                }
            })
            .collect();
        let b = Bricks::from_elements(n, &p.domain);
        let sparse = (
            b.gather(&ee, n, 1, 0.0),
            b.gather(&free, nodes, 1, 0),
            b.gather(&u, nodes, 3, 0.0),
        );
        Case {
            dense: (ee, free, u),
            sparse,
            b,
        }
    }

    const SPARSE_KERNELS: [&str; 2] = ["apply_fine_sparse", "apply_fine_sparse_direct"];

    /// The sparse `apply_fine` (staged and direct) matches the dense kernel, scattered back.
    #[test]
    fn sparse_apply_matches_dense() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let eng = Engine::new(&gpu).unwrap();
        let ke: Vec<f32> = crate::topopt::hex_ke(0.3).iter().flatten().map(|&v| v as f32).collect();
        let ke = eng.init("ke", bytemuck::cast_slice(&ke));
        for n in [[13, 9, 22], [21, 18, 11], [8, 8, 8]] {
            let p = problem(n, ball(n));
            let c = case(&p);
            let nn: usize = n.iter().map(|v| v + 1).product();
            let (ee, free, u) = &c.dense;
            let (ee_d, free_d, u_d, y_d) = (
                eng.init("ee", bytemuck::cast_slice(ee)),
                eng.init("free", bytemuck::cast_slice(free)),
                eng.init("u", bytemuck::cast_slice(u)),
                eng.buf("y", 3 * nn),
            );
            let mut prog = vec![];
            let pd = Params {
                n: dims4(n),
                ..Default::default()
            };
            eng.step(
                &mut prog,
                "apply_fine",
                pd,
                &[&ee_d, &free_d, &ke, &u_d, &y_d],
                threads(nn),
            );
            eng.run(&[&prog]).unwrap();
            let want = eng.read_f32(&y_d, 3 * nn).unwrap();

            let (ee, free, u) = &c.sparse;
            let (ee_s, free_s, u_s, y_s, nbr) = (
                eng.init("ee", bytemuck::cast_slice(ee)),
                eng.init("free", bytemuck::cast_slice(free)),
                eng.init("u", bytemuck::cast_slice(u)),
                eng.buf("y", 3 * c.b.len()),
                eng.init("nbr", bytemuck::cast_slice(&c.b.nbr)),
            );
            for kernel in SPARSE_KERNELS {
                eng.write_f32(&y_s, &vec![f32::NAN; 3 * c.b.len()]);
                let mut prog = vec![];
                let ps = Params {
                    m: [0, c.b.slots() as u32, 0, 0],
                    ..Default::default()
                };
                eng.step(
                    &mut prog,
                    kernel,
                    ps,
                    &[&ee_s, &free_s, &ke, &u_s, &y_s, &nbr],
                    groups(c.b.slots()),
                );
                eng.run(&[&prog]).unwrap();
                let got =
                    c.b.scatter(&eng.read_f32(&y_s, 3 * c.b.len()).unwrap(), n.map(|v| v + 1), 3, 0.0);
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "{n:?} {kernel}: {} of {} bricks, max error {err:.2e} of {scale:.2e}",
                    c.b.slots(),
                    c.b.page.len()
                );
                assert!(err <= 1e-5 * scale, "{kernel} on {n:?}: max error {err} of {scale}");
            }
        }
    }

    /// Dense vs sparse `apply_fine` time and memory on a full box and on sparse shapes (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn profile_sparse_apply() {
        let gpu = crate::gpu::headless().unwrap();
        let eng = Engine::new(&gpu).unwrap();
        let ke: Vec<f32> = crate::topopt::hex_ke(0.3).iter().flatten().map(|&v| v as f32).collect();
        let ke = eng.init("ke", bytemuck::cast_slice(&ke));
        let table = |n: [usize; 3]| -> Vec<bool> {
            // a table: a top plate 6% of the height thick and four legs 12% of the width square
            let (t, w) = (n[2] * 6 / 100, n[0] * 12 / 100);
            (0..n.iter().product::<usize>())
                .map(|e| {
                    let p = [e % n[0], e / n[0] % n[1], e / (n[0] * n[1])];
                    let edge = |v: usize, m: usize| v < w || v >= m - w;
                    p[2] >= n[2] - t || (edge(p[0], n[0]) && edge(p[1], n[1]))
                })
                .collect()
        };
        let shapes: Vec<(&str, [usize; 3], Vec<bool>)> = vec![
            ("full box", [200, 120, 400], vec![true; 200 * 120 * 400]),
            ("hollow ball", [256, 256, 256], ball([256; 3])),
            ("table", [256, 256, 256], table([256; 3])),
        ];
        for (name, n, dom) in shapes {
            let p = problem(n, dom);
            let c = case(&p);
            let nn: usize = n.iter().map(|v| v + 1).product();
            let ne: usize = n.iter().product();
            let fill = p.domain.iter().filter(|&&d| d).count() as f64 / ne as f64;
            let time = |prog: &[crate::topopt::gpu_engine::Step]| {
                let reps = 20;
                let batch: Vec<&[_]> = vec![prog; reps];
                eng.run(&batch).unwrap();
                eng.read_scalars().unwrap();
                let t = std::time::Instant::now();
                eng.run(&batch).unwrap();
                eng.read_scalars().unwrap();
                t.elapsed().as_secs_f64() * 1e3 / reps as f64
            };
            let (ee, free, u) = &c.dense;
            let bufs = [
                eng.init("ee", bytemuck::cast_slice(ee)),
                eng.init("free", bytemuck::cast_slice(free)),
                eng.init("u", bytemuck::cast_slice(u)),
                eng.buf("y", 3 * nn),
            ];
            let mut prog = vec![];
            let pd = Params {
                n: dims4(n),
                ..Default::default()
            };
            eng.step(
                &mut prog,
                "apply_fine",
                pd,
                &[&bufs[0], &bufs[1], &ke, &bufs[2], &bufs[3]],
                threads(nn),
            );
            let td = time(&prog);
            let want = eng.read_f32(&bufs[3], 3 * nn).unwrap();
            drop(prog);
            drop(bufs);
            let (ee, free, u) = &c.sparse;
            let bufs = [
                eng.init("ee", bytemuck::cast_slice(ee)),
                eng.init("free", bytemuck::cast_slice(free)),
                eng.init("u", bytemuck::cast_slice(u)),
                eng.buf("y", 3 * c.b.len()),
                eng.init("nbr", bytemuck::cast_slice(&c.b.nbr)),
            ];
            eprintln!(
                "{name} {n:?}: domain {:.0}% of the box, {} of {} bricks ({:.0}%), {:.0}% of stored elements in the domain; tables {:.2} B per stored element",
                100.0 * fill,
                c.b.slots(),
                c.b.page.len(),
                100.0 * c.b.slots() as f64 / c.b.page.len() as f64,
                100.0 * fill * ne as f64 / c.b.len() as f64,
                (c.b.table_bytes() + 4 * c.b.page.len()) as f64 / c.b.len() as f64,
            );
            eprintln!("  {:>26}: {td:7.2} ms", "apply_fine (dense)");
            for kernel in SPARSE_KERNELS {
                let mut prog = vec![];
                let ps = Params {
                    m: [0, c.b.slots() as u32, 0, 0],
                    ..Default::default()
                };
                eng.step(
                    &mut prog,
                    kernel,
                    ps,
                    &[&bufs[0], &bufs[1], &ke, &bufs[2], &bufs[3], &bufs[4]],
                    groups(c.b.slots()),
                );
                let ts = time(&prog);
                let got = c.b.scatter(
                    &eng.read_f32(&bufs[3], 3 * c.b.len()).unwrap(),
                    n.map(|v| v + 1),
                    3,
                    0.0,
                );
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err <= 1e-5 * scale, "{kernel}: {err} of {scale}");
                eprintln!("  {kernel:>26}: {ts:7.2} ms ({:.2}× dense)", ts / td);
            }
        }
    }
}
