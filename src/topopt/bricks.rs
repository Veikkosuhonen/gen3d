//! Block-sparse grids (`docs/sparse-grid-plan.md`): a level's node grid tiled by bricks of 64
//! points, only the active bricks stored. Element and node (i, j, k) share brick coordinates, so one
//! brick set serves both; a brick is stored when it holds a node of an active element. Values are
//! slot-major: (slot, local) at `slot · 64 + local`, local x-fastest.
//!
//! Bricks are 4×4×4, or 2×8×4-like across an axis one element thick (a 2D slab, whose 2 node layers
//! would leave half of a 4-wide brick empty); a hierarchy keeps its fine level's shape on every
//! level (`gpu_engine::BRICK_SHAPES` compiles the kernels per shape).
//!
//! Lookups go through the page table (one u32 per box brick, `NONE` when absent); the GPU kernels
//! get it and each slot's brick coordinates.

use super::gpu_engine::BRICK_SHAPES;

pub const B3: usize = 64;
pub const NONE: u32 = u32::MAX;

#[derive(Clone)]
pub struct Bricks {
    /// elements per axis
    pub n: [usize; 3],
    /// points per brick along each axis (product 64)
    pub shape: [usize; 3],
    /// bricks per axis over the box's nodes (⌈(n + 1) / shape⌉)
    pub nb: [usize; 3],
    /// per box brick: its slot, or NONE
    pub page: Vec<u32>,
    /// per slot: its brick's coordinates
    pub coords: Vec<[u32; 3]>,
}

/// The brick shape for a grid of `n` elements: 4³, or 2 points across an axis one element thick.
pub fn shape_for(n: [usize; 3]) -> [usize; 3] {
    let i = match n.iter().position(|&v| v == 1) {
        Some(a) => a + 1,
        None => 0,
    };
    BRICK_SHAPES[i].1
}

impl Bricks {
    /// The bricks holding a node of an `active` element (dense, x-fastest over `n`), in `shape`.
    pub fn from_elements(n: [usize; 3], shape: [usize; 3], active: &[bool]) -> Self {
        let nb: [usize; 3] = std::array::from_fn(|a| (n[a] + 1).div_ceil(shape[a]));
        let mut used = vec![false; nb.iter().product()];
        let bi = |p: [usize; 3]| ((p[2] / shape[2]) * nb[1] + p[1] / shape[1]) * nb[0] + p[0] / shape[0];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    if !active[(k * n[1] + j) * n[0] + i] {
                        continue;
                    }
                    // the element's 8 nodes; its own brick is node 0's
                    for c in 0..8 {
                        used[bi([i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1)])] = true;
                    }
                }
            }
        }
        Self::from_used(n, shape, nb, &used)
    }

    /// Every brick of the box (the dense case).
    #[cfg(test)]
    pub fn all(n: [usize; 3], shape: [usize; 3]) -> Self {
        let nb: [usize; 3] = std::array::from_fn(|a| (n[a] + 1).div_ceil(shape[a]));
        Self::from_used(n, shape, nb, &vec![true; nb.iter().product()])
    }

    fn from_used(n: [usize; 3], shape: [usize; 3], nb: [usize; 3], used: &[bool]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), B3);
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
        Bricks {
            n,
            shape,
            nb,
            page,
            coords,
        }
    }

    pub fn slots(&self) -> usize {
        self.coords.len()
    }

    /// Stored values per field (elements or nodes).
    pub fn len(&self) -> usize {
        self.slots() * B3
    }

    /// The kernel name suffix of this shape (`gpu_engine::BRICK_SHAPES`).
    pub fn variant(&self) -> &'static str {
        BRICK_SHAPES.iter().find(|s| s.1 == self.shape).unwrap().0
    }

    /// Local index of a point within its brick.
    fn local(&self, p: [usize; 3]) -> usize {
        let [bx, by, bz] = self.shape;
        (p[2] % bz * by + p[1] % by) * bx + p[0] % bx
    }

    /// Point (i, j, k) of slot `s`'s local index `l`.
    pub fn point(&self, s: usize, l: usize) -> [usize; 3] {
        let [bx, by, _] = self.shape;
        let q = [l % bx, l / bx % by, l / (bx * by)];
        std::array::from_fn(|a| self.coords[s][a] as usize * self.shape[a] + q[a])
    }

    /// Sparse index of grid point (i, j, k), if its brick is stored.
    pub fn index(&self, p: [usize; 3]) -> Option<usize> {
        let b: [usize; 3] = std::array::from_fn(|a| p[a] / self.shape[a]);
        if (0..3).any(|a| b[a] >= self.nb[a]) {
            return None;
        }
        let s = self.page[(b[2] * self.nb[1] + b[1]) * self.nb[0] + b[0]];
        (s != NONE).then(|| s as usize * B3 + self.local(p))
    }

    /// Dense → slots, `width` values per point, over a grid of `dims` points (n for elements, n + 1
    /// for nodes); brick points outside `dims` get `fill`.
    pub fn gather<T: Copy + Send + Sync>(&self, dense: &[T], dims: [usize; 3], width: usize, fill: T) -> Vec<T> {
        use rayon::prelude::*;
        let mut out = vec![fill; self.len() * width];
        out.par_chunks_mut(B3 * width).enumerate().for_each(|(s, chunk)| {
            for l in 0..B3 {
                let p = self.point(s, l);
                if (0..3).any(|a| p[a] >= dims[a]) {
                    continue;
                }
                let d = (p[2] * dims[1] + p[1]) * dims[0] + p[0];
                chunk[l * width..(l + 1) * width].copy_from_slice(&dense[d * width..(d + 1) * width]);
            }
        });
        out
    }

    /// Slots → dense: the inverse of `gather`, `fill` where no brick is stored.
    pub fn scatter<T: Copy + Send + Sync>(&self, sparse: &[T], dims: [usize; 3], width: usize, fill: T) -> Vec<T> {
        use rayon::prelude::*;
        let mut out = vec![fill; dims.iter().product::<usize>() * width];
        out.par_chunks_mut(dims[0] * width)
            .enumerate()
            .for_each(|(row, chunk)| {
                let (j, k) = (row % dims[1], row / dims[1]);
                for i in 0..dims[0] {
                    if let Some(o) = self.index([i, j, k]) {
                        chunk[i * width..(i + 1) * width].copy_from_slice(&sparse[o * width..(o + 1) * width]);
                    }
                }
            });
        out
    }

    /// The next coarser level (elements 2c, 2c + 1 → c, in the same shape): the parents of every
    /// element in a stored brick, so the coarse set covers each fine brick's children and their
    /// nodes.
    pub fn coarsen(&self) -> Bricks {
        let nc = self.n.map(|v| v.div_ceil(2));
        let mut active = vec![false; nc.iter().product()];
        for s in 0..self.slots() {
            for l in 0..B3 {
                let p = self.point(s, l);
                if (0..3).all(|a| p[a] < self.n[a]) {
                    let q = p.map(|v| v / 2);
                    active[(q[2] * nc[1] + q[1]) * nc[0] + q[0]] = true;
                }
            }
        }
        Bricks::from_elements(nc, self.shape, &active)
    }

    /// Slot coordinates for the GPU (`coords`, one vec4 per slot).
    pub fn coords4(&self) -> Vec<[u32; 4]> {
        self.coords.iter().map(|c| [c[0], c[1], c[2], 0]).collect()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A hollow ball: a shell, so many bricks are absent inside and outside it.
    pub(in crate::topopt) fn ball(n: [usize; 3]) -> Vec<bool> {
        let c = n.map(|v| v as f64 / 2.0);
        let r = c.iter().cloned().fold(f64::MAX, f64::min).max(2.0);
        (0..n.iter().product::<usize>())
            .map(|e| {
                let p = [e % n[0], e / n[0] % n[1], e / (n[0] * n[1])];
                let d = (0..3)
                    .map(|a| {
                        if n[a] == 1 {
                            0.0
                        } else {
                            (p[a] as f64 + 0.5 - c[a]).powi(2)
                        }
                    })
                    .sum::<f64>()
                    .sqrt();
                d < r && d > 0.6 * r
            })
            .collect()
    }

    fn unflat(m: usize, d: [usize; 3]) -> [usize; 3] {
        [m % d[0], m / d[0] % d[1], m / (d[0] * d[1])]
    }

    #[test]
    fn round_trip_and_index() {
        for n in [[13, 9, 22], [1, 17, 30], [21, 1, 9], [12, 19, 1]] {
            let dom = ball(n);
            let b = Bricks::from_elements(n, shape_for(n), &dom);
            let nodes = n.map(|v| v + 1);
            assert!(b.slots() < b.page.len(), "{n:?}: {} of {}", b.slots(), b.page.len());
            // every node of an active element is stored
            for e in (0..dom.len()).filter(|&e| dom[e]) {
                let p = unflat(e, n);
                for c in 0..8 {
                    let q = [p[0] + (c & 1), p[1] + (c >> 1 & 1), p[2] + (c >> 2 & 1)];
                    assert!(b.index(q).is_some(), "{n:?}: node {q:?}");
                }
            }
            // index and point agree
            for s in 0..b.slots() {
                for l in 0..B3 {
                    assert_eq!(b.index(b.point(s, l)), Some(s * B3 + l));
                }
            }
            // scatter(gather(v)) keeps v where stored and fills 0 elsewhere
            let v: Vec<f32> = (0..3 * nodes.iter().product::<usize>())
                .map(|i| i as f32 + 1.0)
                .collect();
            let sp = b.gather(&v, nodes, 3, 0.0);
            let back = b.scatter(&sp, nodes, 3, 0.0);
            for m in 0..nodes.iter().product::<usize>() {
                match b.index(unflat(m, nodes)) {
                    Some(i) => {
                        assert_eq!(&back[3 * m..3 * m + 3], &v[3 * m..3 * m + 3]);
                        assert_eq!(&sp[3 * i..3 * i + 3], &v[3 * m..3 * m + 3]);
                    }
                    None => assert_eq!(&back[3 * m..3 * m + 3], &[0.0; 3]),
                }
            }
        }
    }

    #[test]
    fn all_active_is_dense() {
        let n = [7, 8, 9];
        let b = Bricks::all(n, shape_for(n));
        assert_eq!(b.slots(), 2 * 3 * 3);
        let full = Bricks::from_elements(n, shape_for(n), &vec![true; 7 * 8 * 9]);
        assert_eq!(full.slots(), b.slots());
        // a slab: 2 node layers in one brick layer
        let n = [1, 15, 7];
        assert_eq!(shape_for(n), [2, 8, 4]);
        assert_eq!(Bricks::all(n, shape_for(n)).slots(), 2 * 2);
    }

    #[test]
    fn coarse_covers_children() {
        for n in [[30, 17, 25], [1, 40, 22]] {
            let b = Bricks::from_elements(n, shape_for(n), &ball(n));
            let c = b.coarsen();
            assert_eq!(c.shape, b.shape);
            assert!(c.slots() <= b.slots());
            for s in 0..b.slots() {
                for l in 0..B3 {
                    let p = b.point(s, l);
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
    }
}
