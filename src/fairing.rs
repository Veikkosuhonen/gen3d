//! Smooth surfaces from binary voxel data (topology-optimization results).
//!
//! An optimized density is near-binary on a voxel grid. Its filtered density x̃ is smooth only on the
//! scale of the filter (1.5 voxels), so the x̃ = 0.5 surface still has terraces on shallow slopes:
//! a voxel step every 1/slope voxels. Those terraces have a long wavelength, so mesh smoothing that
//! keeps real features (Taubin) hardly touches them; what tells them apart from features is their
//! amplitude, at most half a voxel.
//!
//! `fair` therefore looks for the smoothest field whose surface stays within about half a voxel of
//! the voxels' own surface (after Whitaker 2000, "Reducing aliasing artifacts in iso-surfaces of
//! binary volumes", and Lempitsky 2010, "Surface extraction from binary volumes with higher-order
//! smoothness"): it minimises the thin-plate energy Σ (Δφ)² with each sample's φ held within
//! `DELTA` of its signed distance to the voxel surface. The thin-plate energy (unlike Σ |∇φ|², whose
//! level sets shrink by mean curvature) leaves spheres, cylinders and planes where they are. Holding
//! φ near a distance (rather than only keeping each sample on its side) keeps a distance-like slope
//! inside thin members, which otherwise goes flat and lets the surface hug the inside samples.
//!
//! `grow` trades faithfulness for smoothness: the optimizer's own design can have ripples and
//! ridges a voxel or two high (thin shells that widen by a voxel at a time), which no surface within
//! half a voxel can smooth out. Letting the surface move further *outward* fills terraces, dents and
//! narrow pockets, and never thins a wall. It does thicken thin walls (by about half the distance on
//! each side): the thin-plate energy also flattens the field's V across a wall. Letting it move
//! further *inward* as well was tried and dropped: on optimized shells it mostly erodes (a 3-voxel
//! slab thins by the full distance while its ripple stays) and opens holes in them.

/// A scalar field on an x-fastest grid, in voxel units (sample i at position i).
#[derive(Clone)]
pub struct Field {
    pub dims: [usize; 3],
    pub v: Vec<f32>,
}

impl Field {
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// Trilinear value and gradient at `q` (clamped to the grid).
    pub fn sample(&self, q: [f32; 3]) -> (f32, [f32; 3]) {
        let d = self.dims;
        let mut i = [0usize; 3];
        let mut t = [0f32; 3];
        for k in 0..3 {
            let c = q[k].clamp(0.0, (d[k] - 1) as f32);
            i[k] = (c.floor() as usize).min(d[k].saturating_sub(2));
            t[k] = c - i[k] as f32;
        }
        let at = |a: usize, b: usize, c: usize| self.v[self.idx(i[0] + a, i[1] + b, i[2] + c)];
        let (mut val, mut g) = (0.0, [0.0; 3]);
        for c in 0..2 {
            for b in 0..2 {
                for a in 0..2 {
                    let f = at(a, b, c);
                    let w = |k: usize, s: usize| if s == 1 { t[k] } else { 1.0 - t[k] };
                    let dw = |s: usize| if s == 1 { 1.0 } else { -1.0 };
                    val += f * w(0, a) * w(1, b) * w(2, c);
                    g[0] += f * dw(a) * w(1, b) * w(2, c);
                    g[1] += f * w(0, a) * dw(b) * w(2, c);
                    g[2] += f * w(0, a) * w(1, b) * dw(c);
                }
            }
        }
        (val, g)
    }
}

/// How far (voxels) the faired surface may move from the voxels' surface. Just under half a voxel:
/// the surface may go almost anywhere between an inside and an outside sample (all the binary data
/// says). Wider bands don't help smooth shapes and let thin members shrink.
pub const DELTA: f32 = 0.45;

/// Iterations of `fair` with the default band (converged for the terraces on the test shapes).
/// Letting the surface grow further smooths longer wavelengths, which take longer to converge: the
/// iterations go up in proportion (measured on an optimized stand leg: growing by 4 voxels had
/// converged by 800 iterations, and changed the volume by 2% from 400 on), to at most 1000.
pub const ITERS: usize = 100;

/// Samples further than this (voxels) from the surface keep their initial value.
const BAND: f32 = 4.0;

/// Approximate signed distance (voxels, negative inside) to the surface of a binary mask:
/// ±(EDT to the other side − 0.5), so the surface sits halfway between an inside and an outside
/// sample. 1e3 where the other side doesn't exist.
pub fn mask_sdf(inside: &[bool], dims: [usize; 3]) -> Field {
    let outside: Vec<bool> = inside.iter().map(|b| !b).collect();
    let to_in = crate::volume::edt(inside, dims, [1.0; 3]);
    let to_out = crate::volume::edt(&outside, dims, [1.0; 3]);
    let v = (0..inside.len())
        .map(|i| {
            let d = |e: f32| if e.is_finite() { e - 0.5 } else { 1e3 };
            if inside[i] { -d(to_out[i]) } else { d(to_in[i]) }
        })
        .collect();
    Field { dims, v }
}

/// `inside` with each `free` sample given the state of a nearest non-free one (breadth-first over
/// face neighbours): the solid continues straight through the boundary of the free region.
fn extend(inside: &[bool], free: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let mut out = inside.to_vec();
    let mut done: Vec<bool> = free.iter().map(|f| !f).collect();
    let mut front: Vec<usize> = (0..inside.len()).filter(|&i| done[i]).collect();
    while !front.is_empty() {
        let mut next = vec![];
        for &i in &front {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let nbrs = [
                (x > 0).then(|| i - 1),
                (x + 1 < nx).then(|| i + 1),
                (y > 0).then(|| i - nx),
                (y + 1 < ny).then(|| i + nx),
                (z > 0).then(|| i - nx * ny),
                (z + 1 < nz).then(|| i + nx * ny),
            ];
            for j in nbrs.into_iter().flatten() {
                if !done[j] {
                    done[j] = true;
                    out[j] = out[i];
                    next.push(j);
                } else if free[j] && out[i] && !out[j] && next.contains(&j) {
                    // reached from both sides in the same layer: inside wins
                    out[j] = true;
                }
            }
        }
        front = next;
    }
    out
}

/// A smooth field (negative inside) whose zero level is within `DELTA` voxels inward and
/// `grow.max(DELTA)` outward of the surface of `inside` (see the module docs). `free` samples, e.g. outside the design domain, are
/// unconstrained, so the field continues smoothly through them (where the solid meets the domain's
/// boundary, the surface beyond it gives the outline on the boundary). They start as if the solid
/// continued straight through the free region's boundary (`extend`): starting them as outside
/// would, within the iterations, pull thin walls along that boundary outward on their other side.
pub fn fair(inside: &[bool], free: Option<&[bool]>, dims: [usize; 3], grow: f32) -> Field {
    let grow = grow.max(DELTA);
    let iters = ((ITERS as f32 * grow / DELTA).ceil() as usize).min(1000);
    let init = match free {
        Some(f) => mask_sdf(&extend(inside, f, dims), dims),
        None => mask_sdf(inside, dims),
    };
    let n = inside.len();
    // φ < 0 inside: a lower φ moves the surface outward
    let mut lo: Vec<f32> = init.v.iter().map(|v| v - grow).collect();
    let mut hi: Vec<f32> = init.v.iter().map(|v| v + DELTA).collect();
    if let Some(free) = free {
        for i in (0..n).filter(|&i| free[i]) {
            lo[i] = f32::NEG_INFINITY;
            hi[i] = f32::INFINITY;
        }
    }
    let active: Vec<bool> = init.v.iter().map(|v| v.abs() <= BAND + grow).collect();
    solve(init, &lo, &hi, &active, iters)
}

/// Minimises Σ (Lφ)² (L the 7-point Laplacian, mirrored at the grid's edges) subject to
/// lo ≤ φ ≤ hi, moving only `active` samples: projected gradient descent with Nesterov momentum
/// (FISTA). The gradient is L²φ; λmax(L²) = 144 sets the step.
fn solve(init: Field, lo: &[f32], hi: &[f32], active: &[bool], iters: usize) -> Field {
    use rayon::prelude::*;
    let dims = init.dims;
    let n = init.v.len();
    let [nx, ny, nz] = dims;
    let lap = |src: &[f32], dst: &mut [f32]| {
        dst.par_chunks_mut(nx).enumerate().for_each(|(row, out)| {
            let (y, z) = (row % ny, row / ny);
            let base = row * nx;
            for x in 0..nx {
                let c = src[base + x];
                let xm = if x > 0 { src[base + x - 1] } else { c };
                let xp = if x + 1 < nx { src[base + x + 1] } else { c };
                let ym = if y > 0 { src[base + x - nx] } else { c };
                let yp = if y + 1 < ny { src[base + x + nx] } else { c };
                let zm = if z > 0 { src[base + x - nx * ny] } else { c };
                let zp = if z + 1 < nz { src[base + x + nx * ny] } else { c };
                out[x] = xm + xp + ym + yp + zm + zp - 6.0 * c;
            }
        });
    };
    let step = 1.0 / 144.0;
    let mut phi = init.v;
    let mut prev = phi.clone();
    let mut y = phi.clone();
    let mut g = vec![0f32; n];
    let mut tmp = vec![0f32; n];
    let mut t = 1.0f32;
    for _ in 0..iters {
        lap(&y, &mut tmp);
        lap(&tmp, &mut g);
        std::mem::swap(&mut prev, &mut phi);
        phi.par_iter_mut().enumerate().for_each(|(i, p)| {
            *p = if active[i] {
                (y[i] - step * g[i]).clamp(lo[i], hi[i])
            } else {
                y[i]
            };
        });
        let tn = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let mom = (t - 1.0) / tn;
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, yv)| *yv = phi[i] + mom * (phi[i] - prev[i]));
        t = tn;
    }
    Field { dims, v: phi }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    /// Marching cubes of `f < 0`, padded with an outside layer; vertices in voxel units.
    fn mesh_field(f: &Field) -> Mesh {
        let [nx, ny, nz] = f.dims;
        let p = [nx + 2, ny + 2, nz + 2];
        let mut vol = vec![1f32; p[0] * p[1] * p[2]];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    vol[((x + 1) * p[1] + y + 1) * p[2] + z + 1] = f.v[f.idx(x, y, z)];
                }
            }
        }
        let grid = crate::mesher::Grid {
            vol,
            dims: p,
            origin: [-1.0; 3],
            step: 1.0,
        };
        crate::mesher::polygonize(grid).expect("surface")
    }

    type Sdf = Box<dyn Fn([f32; 3]) -> f32>;

    struct Case {
        name: &'static str,
        sdf: Sdf,
        volume: f64,
        /// where the surface error is measured (away from sharp edges)
        mask: Box<dyn Fn([f32; 3]) -> bool>,
    }

    /// Rotate about x by `ax`, then about y by `ay`.
    fn rot(p: [f32; 3], ax: f32, ay: f32) -> [f32; 3] {
        let (s, c) = ax.sin_cos();
        let p = [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]];
        let (s, c) = ay.sin_cos();
        [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]
    }

    /// Test shapes in a 48³ grid (voxel units), off the lattice and tilted so their voxel surfaces
    /// have terraces.
    fn cases() -> Vec<Case> {
        let o = 24.27f32;
        let pi = std::f64::consts::PI;
        let (bx, by, bz) = (15.0f32, 10.0f32, 6.0f32);
        let (rod_r, rod_l) = (1.9f32, 34.0f32);
        vec![
            Case {
                name: "sphere r11.3",
                sdf: Box::new(move |p| {
                    let d = [p[0] - o, p[1] - o, p[2] - o];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - 11.3
                }),
                volume: 4.0 / 3.0 * pi * 11.3f64.powi(3),
                mask: Box::new(|_| true),
            },
            Case {
                name: "box tilted 9°/5° (faces)",
                sdf: Box::new(move |p| {
                    let q = rot([p[0] - o, p[1] - o, p[2] - o], 0.09, 0.16);
                    let d = [q[0].abs() - bx, q[1].abs() - by, q[2].abs() - bz];
                    let out = (d[0].max(0.0).powi(2) + d[1].max(0.0).powi(2) + d[2].max(0.0).powi(2)).sqrt();
                    out + d[0].max(d[1]).max(d[2]).min(0.0)
                }),
                volume: 8.0 * (bx * by * bz) as f64,
                mask: Box::new(move |p| {
                    let q = rot([p[0] - o, p[1] - o, p[2] - o], 0.09, 0.16);
                    let mut d = [q[0].abs() - bx, q[1].abs() - by, q[2].abs() - bz];
                    d.sort_by(|a, b| b.total_cmp(a));
                    d[1] < -2.5
                }),
            },
            Case {
                name: "cylinder r7 tilted 6°",
                sdf: Box::new(move |p| {
                    let q = rot([p[0] - o, p[1] - o, p[2] - o], 0.0, 0.105);
                    ((q[1] * q[1] + q[2] * q[2]).sqrt() - 7.0).max(q[0].abs() - 18.0)
                }),
                volume: pi * 49.0 * 36.0,
                mask: Box::new(move |p| rot([p[0] - o, p[1] - o, p[2] - o], 0.0, 0.105)[0].abs() < 15.0),
            },
            Case {
                name: "rod r1.9 tilted",
                sdf: Box::new(move |p| {
                    let q = rot([p[0] - o, p[1] - o, p[2] - o], 0.21, 0.13);
                    let x = q[0].clamp(-rod_l / 2.0, rod_l / 2.0);
                    ((q[0] - x).powi(2) + q[1] * q[1] + q[2] * q[2]).sqrt() - rod_r
                }),
                volume: pi * (rod_r as f64).powi(2) * rod_l as f64 + 4.0 / 3.0 * pi * (rod_r as f64).powi(3),
                mask: Box::new(|_| true),
            },
        ]
    }

    const N: usize = 48;

    fn voxelise(c: &Case) -> Vec<bool> {
        let mut inside = vec![false; N * N * N];
        for z in 0..N {
            for y in 0..N {
                for x in 0..N {
                    inside[(z * N + y) * N + x] = (c.sdf)([x as f32, y as f32, z as f32]) < 0.0;
                }
            }
        }
        inside
    }

    /// 0.5 − x̃ of a converged binary design (< 0 inside): the optimizer's hat filter (rmin 1.5) of
    /// the voxels, the surface the mesher used before fairing.
    fn filtered(inside: &[bool]) -> Field {
        let at = |q: [i64; 3]| inside[((q[2] as usize) * N + q[1] as usize) * N + q[0] as usize] as u8 as f32;
        let mut v = vec![0f32; N * N * N];
        for z in 0..N {
            for y in 0..N {
                for x in 0..N {
                    let (mut s, mut ws) = (0.0, 0.0);
                    for dz in -1i64..=1 {
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                let w = 1.5 - ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                                let q = [x as i64 + dx, y as i64 + dy, z as i64 + dz];
                                if w <= 0.0 || q.iter().any(|&c| c < 0 || c >= N as i64) {
                                    continue;
                                }
                                s += w * at(q);
                                ws += w;
                            }
                        }
                    }
                    v[(z * N + y) * N + x] = 0.5 - s / ws;
                }
            }
        }
        Field { dims: [N; 3], v }
    }

    struct Stats {
        /// mean |distance| of the vertices from the true surface (voxels)
        dist: f64,
        /// area-weighted mean angle between face normals and the true normals (degrees)
        angle: f64,
        /// volume error (%)
        vol: f64,
    }

    fn measure(m: &Mesh, c: &Case) -> Stats {
        let d: Vec<f64> = m
            .verts
            .iter()
            .filter(|v| (c.mask)(**v))
            .map(|v| (c.sdf)(*v).abs() as f64)
            .collect();
        let (mut asum, mut wsum) = (0.0, 0.0);
        for t in &m.tris {
            let [a, b, cc] = t.map(|i| m.verts[i as usize]);
            let cen: [f32; 3] = std::array::from_fn(|k| (a[k] + b[k] + cc[k]) / 3.0);
            if !(c.mask)(cen) {
                continue;
            }
            let u: [f32; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let v: [f32; 3] = std::array::from_fn(|k| cc[k] - a[k]);
            let nrm = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let area = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            let g: [f32; 3] = std::array::from_fn(|k| {
                let (mut p, mut q) = (cen, cen);
                p[k] += 1e-3;
                q[k] -= 1e-3;
                (c.sdf)(p) - (c.sdf)(q)
            });
            let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            let cos = (nrm[0] * g[0] + nrm[1] * g[1] + nrm[2] * g[2]) / (area * gl).max(1e-12);
            asum += (cos.clamp(-1.0, 1.0) as f64).acos().to_degrees() * area as f64;
            wsum += area as f64;
        }
        Stats {
            dist: d.iter().sum::<f64>() / d.len() as f64,
            angle: asum / wsum,
            vol: (m.volume() / c.volume - 1.0) * 100.0,
        }
    }

    /// The faired surface's normals are closer to the true ones than the filtered density's (the
    /// mesher's surface before), on every test shape, without moving it noticeably further away or
    /// losing more volume (thin members included). Sharp convex edges (the box's) are rounded and
    /// bulge out by up to `DELTA`, which takes the box's faces near them a little further away.
    #[test]
    fn fairing_beats_the_filtered_density() {
        for c in cases() {
            let inside = voxelise(&c);
            let mut before = mesh_field(&filtered(&inside));
            before.smooth(10, None);
            let mut after = mesh_field(&fair(&inside, None, [N; 3], DELTA));
            after.smooth(crate::mesher::DEFAULT_SMOOTH, None);
            let (b, a) = (measure(&before, &c), measure(&after, &c));
            println!(
                "{:<26} before: {:.3} vox {:5.2}° {:+6.2}%   faired: {:.3} vox {:5.2}° {:+6.2}%",
                c.name, b.dist, b.angle, b.vol, a.dist, a.angle, a.vol
            );
            assert!(
                a.angle < 0.8 * b.angle,
                "{}: normals {} vs {}",
                c.name,
                a.angle,
                b.angle
            );
            assert!(a.dist < 1.1 * b.dist, "{}: distance {} vs {}", c.name, a.dist, b.dist);
            assert!(
                a.vol.abs() < b.vol.abs().max(3.0),
                "{}: volume {} vs {}",
                c.name,
                a.vol,
                b.vol
            );
        }
    }

    /// A slab with a rippled top (1 voxel amplitude, 8-voxel wavelength: what the optimizer's
    /// terraced shells look like) on a 3-voxel base, faired growing by up to `grow`: the top surface's
    /// height along a row, and the base's thinnest point (the lowest top minus the bottom surface's
    /// highest point).
    fn rippled_slab(grow: f32) -> (Vec<f32>, f32) {
        let d = [48usize, 12, 24];
        let n = d[0] * d[1] * d[2];
        let top = |x: f32| 8.4 + (x * std::f32::consts::TAU / 8.0).sin();
        let bottom = 4.4;
        let mut inside = vec![false; n];
        let mut free = vec![false; n];
        for z in 0..d[2] {
            for y in 0..d[1] {
                for x in 0..d[0] {
                    let i = (z * d[1] + y) * d[0] + x;
                    // free sides in x and y: the slab continues through them
                    free[i] = x == 0 || y == 0 || x == d[0] - 1 || y == d[1] - 1;
                    let zf = z as f32;
                    inside[i] = !free[i] && zf > bottom && zf < top(x as f32);
                }
            }
        }
        let f = fair(&inside, Some(&free), d, grow);
        let crossings = |x: usize| {
            let col: Vec<f32> = (0..d[2]).map(|z| f.v[f.idx(x, 6, z)]).collect();
            let at = |k: usize| k as f32 + col[k] / (col[k] - col[k + 1]);
            let up = (0..d[2] - 1).find(|&k| col[k] >= 0.0 && col[k + 1] < 0.0).map(at);
            let down = (0..d[2] - 1).rev().find(|&k| col[k] < 0.0 && col[k + 1] >= 0.0).map(at);
            (up.expect("bottom surface"), down.expect("top surface"))
        };
        let cols: Vec<(f32, f32)> = (4..44).map(crossings).collect();
        let tops: Vec<f32> = cols.iter().map(|c| c.1).collect();
        let lowest_top = tops.iter().fold(f32::MAX, |m, &t| m.min(t));
        let highest_bottom = cols.iter().fold(f32::MIN, |m, c| m.max(c.0));
        (tops, lowest_top - highest_bottom)
    }

    fn peak_to_peak(v: &[f32]) -> f32 {
        v.iter().fold(f32::MIN, |m, &x| m.max(x)) - v.iter().fold(f32::MAX, |m, &x| m.min(x))
    }

    /// The default band keeps the ripple (it is the design's, not the voxels'); growing by 2 voxels
    /// flattens it to a third, and the base only gets thicker (by about 2 voxels).
    #[test]
    fn growing_smooths_ripples_without_thinning() {
        let (tops, base) = rippled_slab(DELTA);
        let (g_tops, g_base) = rippled_slab(2.0);
        let (p, gp) = (peak_to_peak(&tops), peak_to_peak(&g_tops));
        println!("ripple {p:.2} -> {gp:.2}; base {base:.2} -> {g_base:.2} voxels");
        assert!(p > 1.4, "the default band keeps the ripple: {p}");
        assert!(gp < 0.35 * p, "growing smooths: {gp} vs {p}");
        assert!(g_base >= base, "growing doesn't thin: {g_base} vs {base}");
        assert!(
            g_base < base + 2.5,
            "growing thickens by at most ~the distance: {g_base} vs {base}"
        );
    }

    /// A tilted plane (slope 0.3, voxel steps every 3⅓ voxels) filling the grid, with free samples
    /// around it: the faired surface is the plane to within a tenth of a voxel (the voxels' own
    /// surface is off by up to half a voxel).
    #[test]
    fn fairing_flattens_terraces() {
        let d = [42usize, 18, 26];
        let n = d[0] * d[1] * d[2];
        let plane = |x: f32| 0.3 * x + 8.27;
        let mut inside = vec![false; n];
        let mut free = vec![false; n];
        for z in 0..d[2] {
            for y in 0..d[1] {
                for x in 0..d[0] {
                    let i = (z * d[1] + y) * d[0] + x;
                    let pad = x == 0 || y == 0 || z == 0 || x == d[0] - 1 || y == d[1] - 1 || z == d[2] - 1;
                    free[i] = pad;
                    inside[i] = !pad && (z as f32) < plane(x as f32);
                }
            }
        }
        let f = fair(&inside, Some(&free), d, DELTA);
        let err: Vec<f32> = (6..36)
            .map(|x| {
                // the zero crossing up the column at y = 9
                let col: Vec<f32> = (0..d[2]).map(|z| f.v[f.idx(x, 9, z)]).collect();
                let k = (1..d[2] - 1).find(|&k| col[k] < 0.0 && col[k + 1] >= 0.0).unwrap();
                k as f32 + col[k] / (col[k] - col[k + 1]) - plane(x as f32)
            })
            .collect();
        let max = err.iter().fold(0f32, |m, e| m.max(e.abs()));
        let mean = err.iter().sum::<f32>() / err.len() as f32;
        assert!(max < 0.12 && mean.abs() < 0.02, "max {max}, mean {mean}: {err:?}");
    }
}
