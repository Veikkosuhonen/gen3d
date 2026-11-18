//! 2D optimization: a problem projected along one axis into a slab one element thick, optimized as
//! is, and its fields extruded back onto the 3D grid (see `docs/plane2d-plan.md`).
//!
//! A one-element slab of 8-node hexes with free faces is plane stress in its plane, so the whole 3D
//! optimizer (CPU and GPU, filter, AM filter, symmetry, coarse start) runs on it unchanged. Its
//! second node layer is held along the axis: a slab one element thin bends out of its plane almost
//! freely (stiffness ∝ h³), which no load here excites but which wrecks the solver's conditioning
//! (f32 CG diverged on a 667×223 slab). The first layer stays free, so the slab can still get
//! thinner or thicker under load (plane stress, not plane strain: holding both layers made the
//! beam 12% stiffer than in 3D, holding one 1% softer). It has to be the second layer: multigrid
//! interpolates the first from one coarse node, but the second from two, and with only the second
//! free those two coarse DOFs move it the same way (a singular coarse matrix, NaN in the
//! coarsest Cholesky).
//!
//! Per column along the axis: the slab's element is in the domain (kept solid) when any of the
//! column's is; a slab node is held along the axes held anywhere in its node column; the loads'
//! in-plane components are summed over the column (half on each of the slab's two node layers) and
//! the out-of-plane one is dropped.
//!
//! The slab stands for a part `layers` elements thick (the domain's deepest column). Its loads are
//! scaled by 1 / layers, so its stresses and displacements are those of the thick part, and its
//! compliance is 1 / layers of the part's (`Model::scale` reports the part's).
//!
//! Back in 3D, each element of the 3D domain gets its column's slab values; elements outside it
//! stay empty, so a domain that isn't a prism is cut by its own shape again. The 3D grid written
//! has at most `AXIS_CELLS` cells along the axis (a prism needs none, a cut one some): at the
//! slab's fine element size all the way through, the beam's result was 9M cells and 219 MB.

use super::Problem;

/// The most cells along the axis of the 3D grid a 2D result is written on.
const AXIS_CELLS: usize = 24;

/// How a slab problem maps back onto its 3D grid.
pub(crate) struct Extrude {
    /// the axis the problem was projected along
    pub axis: usize,
    /// the 3D grid written: elements per axis, first node, spacing per axis (the slab's element
    /// size, coarser along the axis), domain elements (x-fastest)
    pub n: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub domain: Vec<bool>,
    /// the thickness the slab stands for, in elements
    pub layers: usize,
}

/// The slab's grid for a 3D grid of `n` elements.
fn slab_dims(n: [usize; 3], axis: usize) -> [usize; 3] {
    let mut s = n;
    s[axis] = 1;
    s
}

/// The slab problem of `p` along `axis`. Its first node along the axis is at `origin_axis` (the
/// same for every element size, so a coarse and a fine slab line up).
pub(crate) fn project(p: &Problem, axis: usize, origin_axis: f64) -> Problem {
    let n = p.n;
    let sn = slab_dims(n, axis);
    let ns: usize = sn.iter().product();
    let cell = |ijk: [usize; 3], d: [usize; 3]| (ijk[2] * d[1] + ijk[1]) * d[0] + ijk[0];
    let ijk_of = |e: usize, d: [usize; 3]| [e % d[0], (e / d[0]) % d[1], e / (d[0] * d[1])];
    let mut domain = vec![false; ns];
    let mut passive = vec![false; ns];
    let mut depth = vec![0usize; ns];
    for e in 0..p.domain.len() {
        let mut ijk = ijk_of(e, n);
        ijk[axis] = 0;
        let s = cell(ijk, sn);
        if p.domain[e] {
            domain[s] = true;
            depth[s] += 1;
        }
        passive[s] |= p.passive[e];
    }
    let layers = depth.iter().copied().max().unwrap_or(0).max(1);
    // nodes: the slab's two layers share the column's supports and loads
    let (pn, spn) = (n.map(|v| v + 1), sn.map(|v| v + 1));
    let nsn: usize = spn.iter().product();
    let mut fixed = vec![0u8; nsn];
    let mut contact = vec![0u8; if p.contact.is_empty() { 0 } else { nsn }];
    let mut loads = vec![vec![[0.0; 3]; nsn]; p.loads.len()];
    let w = 0.5 / layers as f64;
    for i in 0..p.fixed.len() {
        let mut ijk = ijk_of(i, pn);
        for l in 0..2 {
            ijk[axis] = l;
            let s = cell(ijk, spn);
            fixed[s] |= p.fixed[i];
            if let Some(&c) = p.contact.get(i) {
                contact[s] |= c & !(1 << axis);
            }
            for (c, load) in p.loads.iter().enumerate() {
                for comp in (0..3).filter(|&comp| comp != axis) {
                    loads[c][s][comp] += w * load[i][comp];
                }
            }
        }
    }
    // the second node layer held along the axis (see the module docs)
    for (s, f) in fixed.iter_mut().enumerate() {
        if ijk_of(s, spn)[axis] == 1 {
            *f |= 1 << axis;
        }
    }
    // the written 3D grid: the same cells across the axis, at most AXIS_CELLS along it, each in the
    // domain when the fine cell at its centre is
    let m = n[axis].min(AXIS_CELLS);
    let mut wn = n;
    wn[axis] = m;
    let mut spacing = [p.h; 3];
    spacing[axis] = p.h * n[axis] as f64 / m as f64;
    let wdomain = (0..wn.iter().product())
        .map(|e| {
            let mut ijk = ijk_of(e, wn);
            ijk[axis] = ((ijk[axis] as f64 + 0.5) * n[axis] as f64 / m as f64) as usize;
            p.domain[cell(ijk, n)]
        })
        .collect();
    let mut origin = p.origin;
    origin[axis] = origin_axis;
    Problem {
        n: sn,
        h: p.h,
        origin,
        domain,
        passive,
        fixed,
        contact,
        loads,
        cases: p.cases.clone(),
        extrude: Some(std::sync::Arc::new(Extrude {
            axis,
            n: wn,
            origin: p.origin,
            spacing,
            domain: wdomain,
            layers,
        })),
    }
}

impl Extrude {
    /// Slab cell values on the 3D grid: each 3D domain element gets its column's, the others
    /// `outside`.
    pub fn cells(&self, v: &[f32], outside: f32) -> Vec<f32> {
        let n = self.n;
        let sn = slab_dims(n, self.axis);
        (0..self.domain.len())
            .map(|e| {
                if !self.domain[e] {
                    return outside;
                }
                let mut ijk = [e % n[0], (e / n[0]) % n[1], e / (n[0] * n[1])];
                ijk[self.axis] = 0;
                v[(ijk[2] * sn[1] + ijk[1]) * sn[0] + ijk[0]]
            })
            .collect()
    }

    /// Slab node vectors (3 components) on the 3D grid's nodes: the in-plane components of the
    /// column's two slab nodes averaged, the out-of-plane one 0.
    pub fn points(&self, v: &[f32]) -> Vec<f32> {
        let pn = self.n.map(|v| v + 1);
        let spn = slab_dims(self.n, self.axis).map(|v| v + 1);
        let count: usize = pn.iter().product();
        let mut out = vec![0f32; 3 * count];
        for (i, o) in out.chunks_mut(3).enumerate() {
            let mut ijk = [i % pn[0], (i / pn[0]) % pn[1], i / (pn[0] * pn[1])];
            for l in 0..2 {
                ijk[self.axis] = l;
                let s = (ijk[2] * spn[1] + ijk[1]) * spn[0] + ijk[0];
                for c in (0..3).filter(|&c| c != self.axis) {
                    o[c] += 0.5 * v[3 * s + c];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::grid;
    use super::*;

    /// A 4x3x5 grid projected along y: a domain that is a prism in y except one column, a kept
    /// element, supports and a load spread over a column.
    #[test]
    fn projects_and_extrudes() {
        let n = [4, 3, 5];
        let mut p = grid(n, 2.0);
        p.origin = [1.0, 2.0, 3.0];
        let cell = |i: usize, j: usize, k: usize| (k * 3 + j) * 4 + i;
        let node = |i: usize, j: usize, k: usize| (k * 4 + j) * 5 + i;
        // column (3, *, 4) only has its middle element
        p.domain[cell(3, 0, 4)] = false;
        p.domain[cell(3, 2, 4)] = false;
        // column (0, *, 0) is empty
        for j in 0..3 {
            p.domain[cell(0, j, 0)] = false;
        }
        p.passive[cell(1, 2, 1)] = true;
        p.fixed[node(0, 1, 5)] = 7;
        p.fixed[node(4, 3, 5)] = 4;
        // a load of 1 N per node along x, y, z on the column of node (2, *, 5)
        p.loads = vec![vec![[0.0; 3]; 5 * 4 * 6]];
        p.cases = vec![1];
        for j in 0..4 {
            p.loads[0][node(2, j, 5)] = [1.0, 1.0, -1.0];
        }
        let s = project(&p, 1, -7.0);
        assert_eq!(s.n, [4, 1, 5]);
        assert_eq!(s.origin, [1.0, -7.0, 3.0]);
        let sc = |i: usize, k: usize| k * 4 + i;
        let sn = |i: usize, l: usize, k: usize| (k * 2 + l) * 5 + i;
        assert!(s.domain[sc(3, 4)] && !s.domain[sc(0, 0)] && s.domain[sc(1, 1)]);
        assert!(s.passive[sc(1, 1)] && !s.passive[sc(2, 1)]);
        for l in 0..2 {
            // layer 1 is held along y (bit 1) everywhere
            let y = if l == 1 { 2 } else { 0 };
            assert_eq!(s.fixed[sn(0, l, 5)], 7);
            assert_eq!(s.fixed[sn(4, l, 5)], 4 | y);
            assert_eq!(s.fixed[sn(1, l, 5)], y);
            // 4 N in x over the column, 3 layers thick, half per node layer; y dropped
            assert_eq!(s.loads[0][sn(2, l, 5)], [4.0 / 3.0 / 2.0, 0.0, -4.0 / 3.0 / 2.0]);
        }
        let x = s.extrude.as_ref().unwrap();
        assert_eq!((x.axis, x.n, x.origin, x.layers), (1, n, p.origin, 3));
        assert_eq!(x.spacing, [2.0; 3]);
        // a deep grid is written with at most AXIS_CELLS cells along the axis, through the same
        // thickness; the domain sampled at their centres
        let mut deep = grid([2, 50, 2], 0.5);
        for e in 0..200 {
            deep.domain[e] = (e / 2) % 50 < 40;
        }
        let d = project(&deep, 1, 0.0);
        let dx = d.extrude.as_ref().unwrap();
        assert_eq!(
            (dx.n, dx.spacing, dx.layers),
            ([2, 24, 2], [0.5, 0.5 * 50.0 / 24.0, 0.5], 40)
        );
        // cell 19 is centred at fine 40.6 (out), 18 at 38.5 (in)
        assert!(dx.domain[18 * 2] && !dx.domain[19 * 2]);

        // back to 3D: the slab's values on the 3D domain only
        let v: Vec<f32> = (0..20).map(|e| e as f32).collect();
        let c = x.cells(&v, -1.0);
        assert_eq!(c.len(), 60);
        assert_eq!(c[cell(2, 0, 3)], sc(2, 3) as f32);
        assert_eq!(c[cell(2, 2, 3)], sc(2, 3) as f32);
        assert_eq!(c[cell(3, 1, 4)], sc(3, 4) as f32);
        assert_eq!(c[cell(3, 0, 4)], -1.0);
        assert_eq!(c[cell(0, 1, 0)], -1.0);
        // displacements: the two node layers averaged in plane, 0 out of plane
        let mut u = vec![0f32; 3 * 5 * 2 * 6];
        u[3 * sn(2, 0, 5)..3 * sn(2, 0, 5) + 3].copy_from_slice(&[1.0, 5.0, 2.0]);
        u[3 * sn(2, 1, 5)..3 * sn(2, 1, 5) + 3].copy_from_slice(&[3.0, -5.0, 4.0]);
        let d = x.points(&u);
        for j in 0..4 {
            let i = node(2, j, 5);
            assert_eq!(&d[3 * i..3 * i + 3], &[2.0, 0.0, 3.0]);
        }
    }

    /// A beam 24 x 3 x 8 on clamped pads at both ends of its bottom, pushed down (-z) in the
    /// middle of its top, the same through its thickness (y): a 2D problem along y.
    fn beam() -> Problem {
        let n = [24, 3, 8];
        let mut p = grid(n, 1.0);
        p.loads = vec![vec![[0.0; 3]; 25 * 4 * 9]];
        p.cases = vec![1];
        for j in 0..4 {
            for i in [0, 1, 23, 24] {
                let m = p.node(i, j, 0);
                p.fixed[m] = 7;
            }
            let m = p.node(12, j, 8);
            p.loads[0][m] = [0.0, 0.0, -2.5];
        }
        p
    }

    /// The solid beam as a 2D slab is about as stiff as in 3D, and multigrid converges on the slab
    /// as on any grid.
    #[test]
    fn slab_matches_the_solid_beam() {
        use super::super::{Fem, hex_ke, mg};
        let compliance = |p: &Problem| -> (f64, usize) {
            let ke = hex_ke(0.35);
            let fem = Fem {
                p,
                ke: &ke,
                ee: vec![1000.0 * p.h; p.domain.len()],
                free: p.free_mask(),
            };
            let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
            let mut m = mg::Multigrid::new(&fem, &mg::MgSettings::default());
            let mut u = vec![0.0; f.len()];
            let its = fem.solve(&f, &mut u, 1e-10, 500, &mut |r, z| m.precondition(r, z));
            (f.iter().zip(&u).map(|(a, b)| a * b).sum(), its)
        };
        let p = beam();
        let (c3, _) = compliance(&p);
        let slab = project(&p, 1, 0.0);
        let (c2, its) = compliance(&slab);
        // the slab carries 1/3 of the load (it stands for 3 elements): 1/3 of the compliance
        let c2 = c2 * slab.extrude.as_ref().unwrap().layers as f64;
        assert!(its < 30, "{its} MGCG iterations on the slab");
        // (within a few %: plane stress is softer, but in 3D the clamped pads also hold the beam
        // across its width, and here that wins: the slab is 3.4% stiffer)
        assert!((c2 - c3).abs() < 0.05 * c3, "slab {c2} vs 3D {c3}");
    }

    /// On the GPU too: the slab's run (overhang control across it, so the AM filter has 3
    /// supports per element) follows the CPU's, and the result is written on the 3D grid.
    #[test]
    fn slab_gpu_matches_cpu() {
        use super::super::mg::MgSettings;
        use super::super::tests::beam_settings;
        use super::super::{Settings, Solver, optimize};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        if gpu.is_software() {
            eprintln!("software GPU; skipped");
            return;
        }
        let slab = project(&beam(), 1, 0.0);
        let dir = std::env::temp_dir().join(format!("gen3d-slab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let settings = |solver| Settings {
            overhang: true,
            iters: 12,
            cg_tol: 1e-6,
            ..beam_settings(solver)
        };
        let ms = MgSettings::default();
        let cpu = optimize(&slab, &settings(Solver::Multigrid(ms)), &dir.join("cpu.vti"), None).unwrap();
        let g = optimize(&slab, &settings(Solver::Gpu(ms)), &dir.join("gpu.vti"), Some(&gpu)).unwrap();
        for (it, (a, b)) in cpu.stats.iter().zip(&g.stats).enumerate() {
            let rel = (a.compliance - b.compliance).abs() / a.compliance;
            assert!(
                rel < 2e-3,
                "iteration {}: {} (CPU) vs {} (GPU)",
                it + 1,
                a.compliance,
                b.compliance
            );
        }
        let v = crate::vti::read(&dir.join("gpu.vti")).unwrap();
        let density = v.arrays.iter().find(|a| a.name == "density").unwrap();
        assert_eq!(density.data.len(), 24 * 3 * 8, "written on the 3D grid");
        // the same through the thickness
        for e in 0..24 * 8 {
            let (i, k) = (e % 24, e / 24);
            let d: Vec<f32> = (0..3).map(|j| density.data[(k * 3 + j) * 24 + i]).collect();
            assert!(d[0] == d[1] && d[1] == d[2], "{d:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
