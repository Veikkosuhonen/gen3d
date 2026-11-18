//! Mirror symmetry: solving one side of a symmetric problem.
//!
//! With mirror planes (the editor's, see `prims::Mirror`) the grid is aligned so each plane is a
//! node layer in the middle of the grid. When the domain, the kept regions and the supports are
//! symmetric about the planes, the optimizer keeps only the elements on the upper side of each
//! plane (2^k fewer for k planes) and makes the design symmetric.
//!
//! The loads need not be symmetric. Each load is split into parts that are symmetric or
//! antisymmetric about each plane (2^k patterns): for pattern s,
//!
//!   f_s(x) = 2^-k Σ_g χ_s(g) M_g f(g x)
//!
//! over the mirror combinations g, with M_g flipping the mirrored components and χ_s(g) = -1 to the
//! number of planes in g that s is antisymmetric about. A symmetric (antisymmetric) part has a
//! symmetric (antisymmetric) displacement, which on the reduced grid is the solution with the
//! plane's normal (tangential) displacements held. So each pattern is its own problem on the
//! reduced grid (a "group": its own supports, hence its own multigrid hierarchy), solved for every
//! load case that has a part in it. Nodes on a plane carry half their load (the other half belongs
//! to the mirrored side).
//!
//! With a symmetric design the parts don't interact: the full compliance is 2^k times the sum of
//! the groups' compliances, and per element the sum of the groups' energies is the full energy
//! averaged over the element's mirror images (the cross terms cancel in pairs), which is what the
//! symmetric design's sensitivity needs. The full displacement of a case at a mirrored point is
//! M_g Σ_s χ_s(g) u_s at the reduced point.

use super::Problem;

/// How a model was reduced (see the module docs).
#[derive(Clone)]
pub(crate) struct Sym {
    /// the mirrored axes; each plane is the reduced grid's low face along its axis
    pub axes: [bool; 3],
    /// elements per axis of the reduced grid, and of the full grid with its origin
    pub n: [usize; 3],
    pub full_n: [usize; 3],
    pub full_origin: [f64; 3],
    /// per group: its pattern, bit a set = antisymmetric about axis a's plane
    pub patterns: Vec<u8>,
}

/// The problem the optimizer solves: one group per load pattern (without symmetry, one group: the
/// problem itself). All groups share the grid, domain and kept regions; they differ in their
/// supports and loads. Every group has a load vector for every case (zero where a case has no
/// part in that pattern).
#[derive(Clone)]
pub(crate) struct Model {
    pub groups: Vec<Problem>,
    pub sym: Option<Sym>,
}

impl Model {
    pub fn single(p: Problem) -> Model {
        Model {
            groups: vec![p],
            sym: None,
        }
    }

    /// The reduced grid's geometry (domain, kept regions), shared by the groups.
    pub fn p(&self) -> &Problem {
        &self.groups[0]
    }

    /// How many copies of the model's grid make the whole grid (2^planes).
    pub fn copies(&self) -> f64 {
        self.sym.as_ref().map_or(1.0, |s| (1u32 << s.k()) as f64)
    }

    /// The whole part's compliance per the model's: 2^planes, and for a 2D slab the thickness it
    /// stands for, in elements (its loads are scaled down by that, see `plane2d.rs`).
    pub fn scale(&self) -> f64 {
        self.copies() * self.p().extrude.as_ref().map_or(1.0, |x| x.layers as f64)
    }

    /// The axes the filter reflects at the grid's low face.
    pub fn mirror_axes(&self) -> [bool; 3] {
        self.sym.as_ref().map_or([false; 3], |s| s.axes)
    }

    pub fn cases(&self) -> usize {
        self.p().loads.len()
    }

    /// Per group, the sign of its parts in the full displacement on mirror side `region` (a set
    /// of axes, as bits); without symmetry, 1.
    pub fn signs(&self, region: u8) -> Vec<f64> {
        match &self.sym {
            Some(s) => s.patterns.iter().map(|&p| sign(p, region)).collect(),
            None => vec![1.0],
        }
    }

    /// The mirror sides: every subset of the mirrored axes (just 0 without symmetry).
    pub fn regions(&self) -> Vec<u8> {
        let mask = self.sym.as_ref().map_or(0, |s| axes_mask(s.axes));
        (0..=mask).filter(|r| r & !mask == 0).collect()
    }
}

/// χ: -1 to the number of planes in `region` that `pattern` is antisymmetric about.
pub(crate) fn sign(pattern: u8, region: u8) -> f64 {
    if (pattern & region).count_ones() % 2 == 1 {
        -1.0
    } else {
        1.0
    }
}

fn axes_mask(axes: [bool; 3]) -> u8 {
    (0..3).filter(|&a| axes[a]).fold(0, |m, a| m | 1 << a)
}

impl Sym {
    /// number of planes
    pub fn k(&self) -> u32 {
        axes_mask(self.axes).count_ones()
    }

    /// A full-grid coordinate along axis `a` (elements: `nodes` false) → (reduced coordinate,
    /// mirrored?).
    fn fold(&self, a: usize, t: usize, nodes: bool) -> (usize, bool) {
        if !self.axes[a] {
            return (t, false);
        }
        let m = self.n[a];
        if t >= m {
            (t - m, false)
        } else if nodes {
            (m - t, true)
        } else {
            (m - 1 - t, true)
        }
    }

    /// A cell array on the reduced grid → the full grid (mirrored).
    pub fn expand_cells(&self, v: &[f32]) -> Vec<f32> {
        let [fx, fy, fz] = self.full_n;
        let [rx, ry, _] = self.n;
        let mut out = Vec::with_capacity(fx * fy * fz);
        for k in 0..fz {
            let kk = self.fold(2, k, false).0;
            for j in 0..fy {
                let jj = self.fold(1, j, false).0;
                for i in 0..fx {
                    let ii = self.fold(0, i, false).0;
                    out.push(v[(kk * ry + jj) * rx + ii]);
                }
            }
        }
        out
    }

    /// A load case's displacement on the full grid, from its part in each group (reduced node
    /// arrays, 3 per node).
    pub fn expand_disp(&self, parts: &[&[f32]]) -> Vec<f32> {
        let [fx, fy, fz] = self.full_n.map(|v| v + 1);
        let [rx, ry, _] = self.n.map(|v| v + 1);
        let mut out = Vec::with_capacity(3 * fx * fy * fz);
        for k in 0..fz {
            let (kk, mk) = self.fold(2, k, true);
            for j in 0..fy {
                let (jj, mj) = self.fold(1, j, true);
                for i in 0..fx {
                    let (ii, mi) = self.fold(0, i, true);
                    let region = mi as u8 | (mj as u8) << 1 | (mk as u8) << 2;
                    let r = (kk * ry + jj) * rx + ii;
                    let mut u = [0f32; 3];
                    for (&pat, part) in self.patterns.iter().zip(parts) {
                        let s = sign(pat, region) as f32;
                        for c in 0..3 {
                            u[c] += s * part[3 * r + c];
                        }
                    }
                    for c in 0..3 {
                        if region >> c & 1 == 1 {
                            u[c] = -u[c];
                        }
                    }
                    out.extend(u);
                }
            }
        }
        out
    }
}

/// Reduce the full problem `full` (its grid aligned: each mirrored axis has an even number of
/// elements, the plane in the middle) to the upper side of the planes on `axes`. Fails with the
/// reason when the problem isn't symmetric.
pub(crate) fn reduce(full: &Problem, axes: [bool; 3]) -> Result<Model, String> {
    let names = ["X", "Y", "Z"];
    let n = full.n;
    for a in 0..3 {
        if axes[a] && !n[a].is_multiple_of(2) {
            return Err(format!("the grid isn't aligned with the {} plane", names[a]));
        }
    }
    let rn: [usize; 3] = std::array::from_fn(|a| if axes[a] { n[a] / 2 } else { n[a] });
    let sym = Sym {
        axes,
        n: rn,
        full_n: n,
        full_origin: full.origin,
        patterns: vec![],
    };
    let [ex, ey, ez] = n;
    let [nx, ny, nz] = n.map(|v| v + 1);
    let el = |i: usize, j: usize, k: usize| (k * ey + j) * ex + i;
    let nd = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;

    // symmetric geometry and supports
    for (a, name) in names.iter().enumerate().filter(|(a, _)| axes[*a]) {
        let flip = |v: [usize; 3], len: usize| -> [usize; 3] {
            let mut w = v;
            w[a] = len - 1 - v[a];
            w
        };
        for k in 0..ez {
            for j in 0..ey {
                for i in 0..ex {
                    let [mi, mj, mk] = flip([i, j, k], n[a]);
                    let (e, m) = (el(i, j, k), el(mi, mj, mk));
                    if full.domain[e] != full.domain[m] || full.passive[e] != full.passive[m] {
                        return Err(format!("the domain isn't symmetric about the {name} plane"));
                    }
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let [mi, mj, mk] = flip([i, j, k], n[a] + 1);
                    let (q, r) = (nd(i, j, k), nd(mi, mj, mk));
                    if full.fixed[q] != full.fixed[r] || full.contact.get(q) != full.contact.get(r) {
                        return Err(format!("the supports aren't symmetric about the {name} plane"));
                    }
                }
            }
        }
    }

    // the reduced geometry
    let off: [usize; 3] = std::array::from_fn(|a| n[a] - rn[a]);
    let origin: [f64; 3] = std::array::from_fn(|a| full.origin[a] + off[a] as f64 * full.h);
    let rne: usize = rn.iter().product();
    let mut domain = Vec::with_capacity(rne);
    let mut passive = Vec::with_capacity(rne);
    for k in 0..rn[2] {
        for j in 0..rn[1] {
            for i in 0..rn[0] {
                let e = el(i + off[0], j + off[1], k + off[2]);
                domain.push(full.domain[e]);
                passive.push(full.passive[e]);
            }
        }
    }
    let [rx, ry, rz] = rn.map(|v| v + 1);
    let rnn = rx * ry * rz;
    // per reduced node: its full node under each mirror combination g (a set of axes), and the
    // share of its load it carries (1/2 per plane it lies on)
    let mask = axes_mask(axes);
    let combos: Vec<u8> = (0..=mask).filter(|g| g & !mask == 0).collect();
    let k = mask.count_ones();
    let image = |r: [usize; 3], g: u8| -> usize {
        let t: [usize; 3] = std::array::from_fn(|a| if g >> a & 1 == 1 { off[a] - r[a] } else { off[a] + r[a] });
        nd(t[0], t[1], t[2])
    };
    let mut groups = vec![];
    let mut patterns = vec![];
    for &pat in &combos {
        let mut fixed = vec![0u8; rnn];
        let mut contact = vec![0u8; if full.contact.is_empty() { 0 } else { rnn }];
        let mut loads = vec![vec![[0.0f64; 3]; rnn]; full.loads.len()];
        let mut any = false;
        for kk in 0..rz {
            for jj in 0..ry {
                for ii in 0..rx {
                    let r = [ii, jj, kk];
                    let ri = (kk * ry + jj) * rx + ii;
                    let mut f = full.fixed[image(r, 0)];
                    if let Some(c) = contact.get_mut(ri) {
                        *c = full.contact[image(r, 0)];
                    }
                    let mut w = 1.0;
                    for a in (0..3).filter(|&a| axes[a] && r[a] == 0) {
                        // on the plane: a symmetric part can't cross it, an antisymmetric one
                        // can't slide along it
                        f |= if pat >> a & 1 == 1 { 7 & !(1 << a) } else { 1 << a };
                        w *= 0.5;
                    }
                    fixed[ri] = f;
                    for (c, load) in full.loads.iter().enumerate() {
                        let mut v = [0.0; 3];
                        for &g in &combos {
                            let s = sign(pat, g);
                            let src = load[image(r, g)];
                            for (comp, vc) in v.iter_mut().enumerate() {
                                let m = if g >> comp & 1 == 1 { -1.0 } else { 1.0 };
                                *vc += s * m * src[comp];
                            }
                        }
                        let scale = w / (1u32 << k) as f64;
                        let v = v.map(|x| x * scale);
                        if v.iter().any(|x| x.abs() > 0.0) {
                            any = true;
                        }
                        loads[c][ri] = v;
                    }
                }
            }
        }
        if any {
            patterns.push(pat);
            groups.push(Problem {
                n: rn,
                h: full.h,
                origin,
                domain: domain.clone(),
                passive: passive.clone(),
                fixed,
                contact: contact.clone(),
                loads,
                cases: full.cases.clone(),
                extrude: full.extrude.clone(),
            });
        }
    }
    Ok(Model {
        groups,
        sym: Some(Sym { patterns, ..sym }),
    })
}

impl Model {
    /// A line describing the reduction: the planes and the load patterns.
    pub fn describe(&self) -> Option<String> {
        let s = self.sym.as_ref()?;
        let names = ["X", "Y", "Z"];
        let planes: Vec<&str> = (0..3).filter(|&a| s.axes[a]).map(|a| names[a]).collect();
        let pats: Vec<String> = s
            .patterns
            .iter()
            .map(|&p| {
                (0..3)
                    .filter(|&a| s.axes[a])
                    .map(|a| if p >> a & 1 == 1 { 'A' } else { 'S' })
                    .collect()
            })
            .collect();
        Some(format!(
            "symmetry: mirrored in {}, solving 1/{} of the grid; load parts {} ({} per plane: S symmetric, A antisymmetric)",
            planes.join(", "),
            1 << s.k(),
            pats.join(", "),
            planes.join("")
        ))
    }
}
