//! Overhang control: Langelaar's additive-manufacturing filter (Langelaar 2017, "An additive
//! manufacturing filter for topology optimization of print-ready designs", SMO 55).
//!
//! The part is printed layer by layer along the build direction (`Up`: ±x, ±y or ±z, +z by default),
//! starting from the first grid layer that has domain elements (the build plate). An element can
//! only be as dense as the material under it allows:
//!
//! ```text
//! Σ_e = smax(ξ_s for the supports s of e: the element below it and the 4 beside that one)
//! ξ_e = smin(x_e, Σ_e)
//! ```
//!
//! With cubic elements the 5-element cross gives the usual 45° limit (along the layer's two axes;
//! diagonally the limit is steeper, about 54.7°). The plate layer is supported. Kept-solid elements
//! are printed regardless (1) and support what is above them; elements outside the domain are empty
//! and support nothing. Mirror planes on the grid's low faces across the build direction reflect
//! the stencil like the density filter (one normal to the build direction is not allowed: printing
//! breaks that symmetry).
//!
//! The filter is smooth (`smin`, `smax` below) and increases with every input, so the volume
//! sensitivities stay non-negative and the OC update keeps working. Its chain rule is an adjoint
//! sweep from the top layer down (`Am::sensitivities`).
//!
//! In the optimizer the filter sits between the density filter and the Heaviside projection:
//! x → x̃ (density filter) → x̂ (this) → x_phys (projection). The projection increases too, so x_phys
//! inherits the support condition, and x̂ = 0.5 is still the boundary of x_phys = 0.5 (what the
//! mesher reads from `density_filtered`).

use super::Problem;
use crate::mesh::Up;

/// `smax`'s exponent (Langelaar: P = 40).
pub(super) const P: f64 = 40.0;
/// `smin`'s smoothing (Langelaar: ε = 1e-4).
pub(super) const EPS: f64 = 1e-4;
/// The number of supports per element.
pub(super) const SUPPORTS: usize = 5;
/// The supports' offsets in the layer below, along the layer's two axes (`Am::lateral`).
pub(super) const STENCIL: [[isize; 2]; SUPPORTS] = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]];

/// `smax`'s root: Q = P + ln N / ln ξ0 with ξ0 = 0.5, so that N supports of density 0.5 give 0.5.
pub(super) fn q() -> f64 {
    P + (SUPPORTS as f64).ln() / 0.5f64.ln()
}

/// Smooth minimum of a and b (≈ min, exact when a = b = 0). Never below 0 for a, b ≥ 0, but
/// rounding can take it a hair under (in f32 about -4e-8), so it is clamped at 0.
fn smin(a: f64, b: f64) -> f64 {
    (0.5 * (a + b - ((a - b) * (a - b) + EPS).sqrt() + EPS.sqrt())).max(0.0)
}

/// ∂smin/∂a and ∂smin/∂b.
fn smin_deriv(a: f64, b: f64) -> (f64, f64) {
    let t = (a - b) / ((a - b) * (a - b) + EPS).sqrt();
    (0.5 * (1.0 - t), 0.5 * (1.0 + t))
}

/// The AM filter on a problem's grid. Layers are numbered in build order (layer 0 is the grid's low
/// face for a + direction, its high face for a - one).
pub(super) struct Am {
    n: [usize; 3],
    up: Up,
    /// the two axes across the build direction, in order
    lateral: [usize; 2],
    /// the plate layer: the first one with domain elements
    pub base: usize,
    /// the lateral axes have a mirror plane on their low face
    mirror: [bool; 2],
}

/// The filter's forward pass: printed densities, and per element its supports' largest density m
/// and r = Σ_s (ξ_s / m)^P (the adjoint needs them). `smax` is m^(P/Q) r^(1/Q) = (Σ_s ξ_s^P)^(1/Q),
/// scaled by m so that ξ^P doesn't underflow (in f32 on the GPU it would below ξ ≈ 0.1, where smax
/// is still about 0.07).
pub(super) struct Printed {
    pub xi: Vec<f64>,
    pub max: Vec<f64>,
    pub r: Vec<f64>,
}

/// smax of supports with largest density m and r = Σ (ξ / m)^P.
fn smax(m: f64, r: f64) -> f64 {
    if m <= 0.0 {
        0.0
    } else {
        m.powf(P / q()) * r.powf(1.0 / q())
    }
}

impl Am {
    pub fn new(p: &Problem, mirror: [bool; 3], up: Up) -> Am {
        let b = up.axis();
        assert!(
            !mirror[b],
            "the AM filter can't have a mirror plane across the build direction"
        );
        let lateral = up.lateral();
        let mut am = Am {
            n: p.n,
            up,
            lateral,
            base: 0,
            mirror: lateral.map(|a| mirror[a]),
        };
        am.base = (0..p.n[b])
            .find(|&l| am.layer(l).any(|(_, _, e)| p.domain[e]))
            .unwrap_or(0);
        am
    }

    /// Layers in build order.
    fn layers(&self) -> usize {
        self.n[self.up.axis()]
    }

    /// The element at lateral position (u, v) in layer l.
    fn elem(&self, u: usize, v: usize, l: usize) -> usize {
        let b = self.up.axis();
        let mut c = [0; 3];
        c[self.lateral[0]] = u;
        c[self.lateral[1]] = v;
        c[b] = if self.up.negative() { self.n[b] - 1 - l } else { l };
        (c[2] * self.n[1] + c[1]) * self.n[0] + c[0]
    }

    /// The elements of layer l: (u, v, element).
    fn layer(&self, l: usize) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let [n0, n1] = self.lateral.map(|a| self.n[a]);
        (0..n1).flat_map(move |v| (0..n0).map(move |u| (u, v, self.elem(u, v, l))))
    }

    /// The supports of the element at (u, v) in layer l + 1: element indices in layer l.
    fn supports(&self, u: usize, v: usize, l: usize) -> impl Iterator<Item = usize> + '_ {
        let [n0, n1] = self.lateral.map(|a| self.n[a] as isize);
        STENCIL.iter().filter_map(move |[du, dv]| {
            let fold = |w: isize, ax: usize| if w < 0 && self.mirror[ax] { -w - 1 } else { w };
            let (a, b) = (fold(u as isize + du, 0), fold(v as isize + dv, 1));
            (a >= 0 && b >= 0 && a < n0 && b < n1).then(|| self.elem(a as usize, b as usize, l))
        })
    }

    /// Printed densities ξ from blueprint densities x (x̃): design elements filtered, kept-solid 1,
    /// outside the domain 0.
    pub fn forward(&self, p: &Problem, x: &[f64]) -> Printed {
        let mut xi = vec![0.0; x.len()];
        let mut max = vec![0.0; x.len()];
        let mut r = vec![0.0; x.len()];
        for l in 0..self.layers() {
            for (u, v, e) in self.layer(l) {
                if !p.domain[e] {
                    continue;
                }
                if p.passive[e] {
                    xi[e] = 1.0;
                    continue;
                }
                if l <= self.base {
                    xi[e] = x[e];
                    continue;
                }
                let m = self.supports(u, v, l - 1).map(|n| xi[n]).fold(0.0, f64::max);
                if m > 0.0 {
                    max[e] = m;
                    r[e] = self.supports(u, v, l - 1).map(|n| (xi[n] / m).powf(P)).sum();
                }
                xi[e] = smin(x[e], smax(max[e], r[e]));
            }
        }
        Printed { xi, max, r }
    }

    /// Chain rule through the filter: from df/dξ (`d`), df/dx for the design elements (0 elsewhere).
    /// `x` and `fw` are the forward pass's input and result.
    pub fn sensitivities(&self, p: &Problem, x: &[f64], fw: &Printed, d: &[f64]) -> Vec<f64> {
        let q = q();
        // λ = df/dξ including the layers above, complete for layer l when it is reached
        let mut lam = d.to_vec();
        let mut out = vec![0.0; x.len()];
        for l in (0..self.layers()).rev() {
            for (u, v, e) in self.layer(l) {
                if !p.domain[e] || p.passive[e] {
                    continue;
                }
                if l <= self.base {
                    out[e] = lam[e];
                    continue;
                }
                let (m, r) = (fw.max[e], fw.r[e]);
                let sm = smax(m, r);
                let (da, db) = smin_deriv(x[e], sm);
                out[e] = lam[e] * da;
                if m <= 0.0 {
                    continue;
                }
                // ∂smax/∂ξ_n = (P/Q) smax ξ_n^(P-1) / Σ ξ^P = (P/Q) smax / (m r) (ξ_n / m)^(P-1)
                let c = lam[e] * db * P / q * sm / (m * r);
                for n in self.supports(u, v, l - 1) {
                    lam[n] += c * (fw.xi[n] / m).powf(P - 1.0);
                }
            }
        }
        out
    }

    /// Print how much of the densities `x` overhangs (see `unsupported`), the optimized material and
    /// the kept-solid regions (printed as given) separately.
    pub fn report(&self, p: &Problem, x: &[f64]) {
        let solid = (0..x.len())
            .filter(|&e| p.domain[e] && !p.passive[e] && x[e] > 0.5)
            .count();
        let [design, kept] = self.unsupported(p, x);
        let mut line = format!(
            "overhang: {design} of {solid} solid design elements unsupported ({:.2}%)",
            100.0 * design as f64 / solid.max(1) as f64
        );
        if kept > 0 {
            line += &format!(", {kept} kept-solid ones (as given)");
        }
        println!("{line}");
    }

    /// Elements (x > 0.5) above the plate with no support x > 0.5 under them, unprintable without
    /// support material: [design elements, kept-solid ones].
    pub fn unsupported(&self, p: &Problem, x: &[f64]) -> [usize; 2] {
        let mut count = [0; 2];
        for l in self.base + 1..self.layers() {
            for (u, v, e) in self.layer(l) {
                if p.domain[e] && x[e] > 0.5 && !self.supports(u, v, l - 1).any(|n| x[n] > 0.5) {
                    count[p.passive[e] as usize] += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topopt::tests::grid;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64).clamp(0.0, 1.0)
            })
            .collect()
    }

    #[test]
    fn supported_material_passes_and_overhangs_are_removed() {
        let p = grid([5, 1, 4], 1.0);
        let at = |i: usize, k: usize| k * 5 + i;
        let mut x = vec![0.0; p.domain.len()];
        // a 45° staircase from (0, 0), and one element at the top resting on its corner
        for k in 0..4 {
            x[at(k, k)] = 1.0;
        }
        x[at(2, 3)] = 1.0;
        let am = Am::new(&p, [false; 3], Up::PlusZ);
        let xi = am.forward(&p, &x).xi;
        for k in 0..4 {
            assert!(xi[at(k, k)] > 0.95, "stair {k}: {}", xi[at(k, k)]);
        }
        assert!(xi[at(2, 3)] > 0.95, "supported by (2, 2) below it");
        // a block of material over nothing
        let mut y = vec![0.0; p.domain.len()];
        y[at(4, 3)] = 1.0;
        let yi = am.forward(&p, &y).xi;
        assert!(yi[at(4, 3)] < 0.01, "{}", yi[at(4, 3)]);
        assert_eq!(am.unsupported(&p, &x), [0, 0]);
        assert_eq!(am.unsupported(&p, &y), [1, 0]);
    }

    /// A horizontal bar floats when printed +z up, and stands when printed from either end.
    #[test]
    fn build_direction_sets_what_is_supported() {
        let p = grid([4, 1, 5], 1.0);
        let mut x = vec![0.0; p.domain.len()];
        for i in 0..4 {
            x[3 * 4 + i] = 1.0;
        }
        for (up, unsupported) in [(Up::PlusZ, 4), (Up::MinusZ, 4), (Up::PlusX, 0), (Up::MinusX, 0)] {
            let am = Am::new(&p, [false; 3], up);
            assert_eq!(am.unsupported(&p, &x), [unsupported, 0], "{up:?}");
            let xi = am.forward(&p, &x).xi;
            let kept = (0..4).filter(|i| xi[12 + i] > 0.95).count();
            assert_eq!(kept, 4 - unsupported, "{up:?}: {:?}", &xi[12..16]);
        }
        // -z: the plate is the grid's top layer
        let mut p = grid([2, 2, 3], 1.0);
        for e in 8..12 {
            p.domain[e] = false;
        }
        assert_eq!(Am::new(&p, [false; 3], Up::MinusZ).base, 1);
        assert_eq!(Am::new(&p, [false; 3], Up::PlusZ).base, 0);
    }

    #[test]
    fn plate_is_the_lowest_domain_layer() {
        let mut p = grid([3, 3, 4], 1.0);
        for e in 0..9 {
            p.domain[e] = false;
        }
        let am = Am::new(&p, [false; 3], Up::PlusZ);
        assert_eq!(am.base, 1);
        let x = vec![0.7; p.domain.len()];
        let xi = am.forward(&p, &x).xi;
        assert!((xi[9 + 4] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn mirror_plane_reflects_the_supports() {
        let p = grid([3, 1, 2], 1.0);
        let mut x = vec![0.0; 6];
        // only element 0 of the bottom layer: on the mirror side its image supports (0, 1) twice
        x[0] = 1.0;
        x[3] = 1.0;
        let plain = Am::new(&p, [false; 3], Up::PlusZ).forward(&p, &x);
        let mirrored = Am::new(&p, [true, false, false], Up::PlusZ).forward(&p, &x);
        assert!((plain.r[3] - 1.0).abs() < 1e-12);
        assert!((mirrored.r[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn sensitivities_match_finite_differences() {
        // scale 0.1: supports so thin that ξ^P underflows in f32
        let cases = [
            ([false; 3], 1.0, Up::PlusZ),
            ([true, true, false], 1.0, Up::PlusZ),
            ([false; 3], 0.1, Up::PlusZ),
            ([true, false, true], 1.0, Up::MinusY),
            ([false, true, false], 1.0, Up::PlusX),
        ];
        for (mirror, scale, up) in cases {
            let mut p = grid([4, 3, 5], 1.0);
            p.passive[(2 * 3 + 1) * 4 + 2] = true;
            p.domain[(4 * 3 + 2) * 4 + 3] = false;
            let am = Am::new(&p, mirror, up);
            let x: Vec<f64> = pseudo_random(p.domain.len(), 7).iter().map(|v| v * scale).collect();
            let w = pseudo_random(p.domain.len(), 11);
            let f = |x: &[f64]| -> f64 { am.forward(&p, x).xi.iter().zip(&w).map(|(a, b)| a * b).sum() };
            let fw = am.forward(&p, &x);
            let g = am.sensitivities(&p, &x, &fw, &w);
            for e in 0..x.len() {
                let h = 1e-6;
                let (mut a, mut b) = (x.clone(), x.clone());
                a[e] += h;
                b[e] -= h;
                let fd = if p.domain[e] && !p.passive[e] {
                    (f(&a) - f(&b)) / (2.0 * h)
                } else {
                    0.0
                };
                assert!(
                    (fd - g[e]).abs() < 1e-5 * (1e-3 + fd.abs()),
                    "{up:?} mirror {mirror:?} scale {scale} element {e}: fd {fd}, adjoint {}",
                    g[e]
                );
            }
        }
    }
}
