//! Viewing a scalar array from a .vti file as a solid: the region where the array is above (or
//! below) a threshold, drawn smooth or as voxel cubes, optionally coloured by a second array.
//!
//! The viewer's raymarcher needs a distance-like field, so the array is converted on the CPU:
//! - smooth: a signed distance to the threshold surface, sampled at the grid points. Points next to
//!   the surface get `(value - iso) / |gradient|` (sub-voxel accurate); the rest get a lower bound
//!   from an exact Euclidean distance transform (EDT) to the other side, so sphere tracing is safe.
//! - blocky: each sample is a cube centred on it. The shader computes exact distances to the cubes in
//!   the 3x3x3 neighbourhood; for farther cubes it uses an EDT stored per sample as a lower bound.
//!
//! The arrays go to the GPU as storage buffers, read by the WGSL in `VOLUME_WGSL`, which supplies
//! `map()` (and the colour tint) for the normal preview harness.

use std::path::Path;

use anyhow::{Result, bail};

use crate::shape::Shape;
use crate::vti::{VtiArray, VtiFile};

/// How to show a .vti file (from the command line); `None` picks a default per array.
#[derive(Clone, Debug, Default)]
pub struct VolumeOpts {
    pub array: Option<String>,
    pub iso: Option<f32>,
    /// inside = values below the threshold (SDF-like) instead of above (density-like)
    pub below: Option<bool>,
    pub blocky: bool,
    pub color: Option<String>,
    /// fixed colour range instead of the automatic one (viewer only)
    pub color_range: Option<(f32, f32)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolUniforms {
    origin: [f32; 4],
    spacing: [f32; 4],
    dims: [u32; 4], // w: blocky
    info: [f32; 4], // colour range min, max, has colour
}

pub struct Volume {
    pub file: VtiFile,
    pub array: usize,
    pub color: Option<usize>,
    pub iso: f32,
    pub below: bool,
    pub blocky: bool,
    origin: [f64; 3],
    spacing: [f64; 3],
    dims: [usize; 3],
    /// the shown array on its lattice (vector arrays as magnitudes)
    values: Vec<f32>,
    /// the field the surface is taken from when it is not `values` (`FILTERED` for `density`, with
    /// 0 outside the domain)
    surf: Option<Vec<f32>>,
    /// samples in the design domain (from `FILTERED`'s NaNs); `None` = all
    domain: Option<Vec<bool>>,
    pub range: (f32, f32),
    /// smooth signed distance per sample
    pub sd: Vec<f32>,
    /// blocky: per sample, -EDT to the outside set when inside, +EDT to the inside set when outside
    pub blk: Vec<f32>,
    /// colour array resampled onto the lattice (all zeros when off)
    pub col: Vec<f32>,
    crange: (f32, f32),
    /// fixed colour range set in the viewer; `None` = automatic (min..max over inside samples)
    pub color_range: Option<(f32, f32)>,
    pub inside_count: usize,
    /// histogram of the shown array over `range`
    pub hist: Vec<u32>,
}

/// The optimizer's filtered densities (x̃ before the Heaviside projection; with overhang control the
/// AM filter's x̂, which is the projection's input then). Where a file has it, the
/// surface of `density` is taken from it: x̃ = 0.5 is the same boundary as density = 0.5 (the
/// projection maps 0.5 to 0.5), but x̃ varies smoothly across it, so the surface is placed between
/// samples instead of following the voxels in steps. It is NaN outside the design domain, which
/// keeps mesh smoothing inside the domain (`Volume::domain_constraint`).
pub const FILTERED: &str = "density_filtered";

/// An optimizer result's print orientation (FieldData): the up direction it was made printable for,
/// a unit vector along an axis (`mesh::Up::vector`).
pub const PRINT_UP: &str = "print_up";

/// Histogram bins for the Volume panel.
pub const HIST_BINS: usize = 64;

fn scalars(a: &VtiArray) -> Vec<f32> {
    if a.comps == 1 {
        return a.data.clone();
    }
    a.data
        .chunks_exact(a.comps)
        .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

fn finite_range(v: &[f32]) -> (f32, f32) {
    let r = v
        .iter()
        .filter(|x| x.is_finite())
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    if r.0 > r.1 { (0.0, 0.0) } else { r }
}

impl Volume {
    pub fn load(path: &Path, opts: &VolumeOpts) -> Result<Volume> {
        let file = crate::vti::read(path)?;
        let find = |name: &str| -> Result<usize> {
            match file.arrays.iter().position(|a| a.name == name) {
                Some(i) => Ok(i),
                None => {
                    let names: Vec<_> = file.arrays.iter().map(|a| a.name.as_str()).collect();
                    bail!("no array `{name}` in {} (arrays: {})", path.display(), names.join(", "))
                }
            }
        };
        let array = match &opts.array {
            Some(n) => find(n)?,
            None => 0,
        };
        let color = opts.color.as_deref().map(find).transpose()?;
        let mut v = Volume {
            file,
            array,
            color,
            iso: 0.0,
            below: false,
            blocky: opts.blocky,
            origin: [0.0; 3],
            spacing: [1.0; 3],
            dims: [1; 3],
            values: vec![],
            surf: None,
            domain: None,
            range: (0.0, 0.0),
            sd: vec![],
            blk: vec![],
            col: vec![],
            crange: (0.0, 1.0),
            color_range: opts.color_range,
            inside_count: 0,
            hist: vec![],
        };
        v.select(array);
        if let Some(b) = opts.below {
            v.below = b;
        }
        if let Some(iso) = opts.iso {
            v.iso = iso;
        }
        v.recompute();
        Ok(v)
    }

    /// Show array `i` with its default threshold: 0 with inside below for SDF-like arrays (named
    /// *sdf*, or with values on both sides of 0 and not named *density*: an optimizer's density can
    /// round a hair below 0), the middle of the range with inside above otherwise (densities,
    /// masks).
    pub fn select(&mut self, i: usize) {
        self.array = i;
        let a = &self.file.arrays[i];
        (self.origin, self.dims) = self.file.lattice(a.cell);
        self.spacing = self.file.spacing.map(f64::abs);
        self.values = scalars(a);
        self.surf = (a.name == "density")
            .then(|| {
                self.file
                    .arrays
                    .iter()
                    .find(|f| f.name == FILTERED && f.cell == a.cell && f.comps == 1 && f.data.len() == a.data.len())
            })
            .flatten()
            .map(scalars);
        self.domain = self
            .surf
            .as_ref()
            .filter(|f| f.iter().any(|v| v.is_nan()))
            .map(|f| f.iter().map(|v| !v.is_nan()).collect());
        if let Some(f) = &mut self.surf {
            for v in f.iter_mut().filter(|v| v.is_nan()) {
                *v = 0.0;
            }
        }
        self.range = finite_range(&self.values);
        let (lo, hi) = self.range;
        self.hist = vec![0; HIST_BINS];
        for &v in self.values.iter().filter(|v| v.is_finite()) {
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.0 };
            self.hist[((t * HIST_BINS as f32) as usize).min(HIST_BINS - 1)] += 1;
        }
        let name = a.name.to_lowercase();
        let sdf_like = a.comps == 1 && (name.contains("sdf") || (!name.contains("density") && lo < 0.0 && hi > 0.0));
        self.below = sdf_like;
        self.iso = if sdf_like { 0.0 } else { 0.5 * (lo + hi) };
    }

    pub fn array_name(&self) -> &str {
        &self.file.arrays[self.array].name
    }

    pub fn color_name(&self) -> Option<&str> {
        self.color.map(|c| self.file.arrays[c].name.as_str())
    }

    /// The values the surface is taken from (see `FILTERED`).
    fn field(&self) -> &[f32] {
        self.surf.as_deref().unwrap_or(&self.values)
    }

    fn inside(&self, v: f32) -> bool {
        if self.below { v < self.iso } else { v > self.iso }
    }

    pub fn count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Recompute the distance fields after a change of array, threshold or direction.
    pub fn recompute(&mut self) {
        let [nx, ny, nz] = self.dims;
        let s = self.spacing;
        let n = self.count();
        let inside: Vec<bool> = self.field().iter().map(|&v| self.inside(v)).collect();
        self.inside_count = inside.iter().filter(|b| **b).count();
        let outside: Vec<bool> = inside.iter().map(|b| !b).collect();
        let to_in = edt(&inside, self.dims, s);
        let to_out = edt(&outside, self.dims, s);
        const FAR: f32 = 1e6;
        let fin = |d: f32| if d.is_finite() { d } else { FAR };

        self.blk = (0..n)
            .map(|i| if inside[i] { -fin(to_out[i]) } else { fin(to_in[i]) })
            .collect();

        let smin = s.iter().cloned().fold(f64::MAX, f64::min) as f32;
        let diag = s.iter().map(|v| v * v).sum::<f64>().sqrt() as f32;
        let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
        let vals = self.field();
        let sign_out = if self.below { 1.0 } else { -1.0 }; // (v - iso) * sign_out > 0 outside
        let mut sd = vec![0f32; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = idx(x, y, z);
                    let sign = if inside[i] { -1.0 } else { 1.0 };
                    let mut band = false;
                    'nb: for dz in z.saturating_sub(1)..(z + 2).min(nz) {
                        for dy in y.saturating_sub(1)..(y + 2).min(ny) {
                            for dx in x.saturating_sub(1)..(x + 2).min(nx) {
                                if inside[idx(dx, dy, dz)] != inside[i] {
                                    band = true;
                                    break 'nb;
                                }
                            }
                        }
                    }
                    sd[i] = if band {
                        // central (or one-sided) differences
                        let g = |k: usize, c: usize, len: usize, at: &dyn Fn(usize) -> usize| -> f32 {
                            if len < 2 {
                                return 0.0;
                            }
                            let (a, b) = (c.saturating_sub(1), (c + 1).min(len - 1));
                            (vals[at(b)] - vals[at(a)]) / ((b - a) as f64 * s[k]) as f32
                        };
                        let gx = g(0, x, nx, &|c| idx(c, y, z));
                        let gy = g(1, y, ny, &|c| idx(x, c, z));
                        let gz = g(2, z, nz, &|c| idx(x, y, c));
                        let gl = (gx * gx + gy * gy + gz * gz).sqrt();
                        let est = (vals[i] - self.iso) * sign_out / gl.max(1e-30);
                        sign * est.abs().min(smin).max(1e-4 * smin)
                    } else {
                        let other = if inside[i] { to_out[i] } else { to_in[i] };
                        sign * (fin(other) - diag).max(smin)
                    };
                }
            }
        }
        // where the solid meets the domain's edge, x̃ doesn't locate the surface (see `mesh`): it is
        // on the cell face, halfway between the two samples
        if let Some(dom) = &self.domain {
            let half = 0.5 * smin;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let i = idx(x, y, z);
                        if !dom[i] {
                            continue;
                        }
                        let nbrs = [
                            (x > 0).then(|| idx(x - 1, y, z)),
                            (x + 1 < nx).then(|| idx(x + 1, y, z)),
                            (y > 0).then(|| idx(x, y - 1, z)),
                            (y + 1 < ny).then(|| idx(x, y + 1, z)),
                            (z > 0).then(|| idx(x, y, z - 1)),
                            (z + 1 < nz).then(|| idx(x, y, z + 1)),
                        ];
                        for j in nbrs.into_iter().flatten() {
                            if !dom[j] && inside[i] {
                                sd[i] = -half;
                                sd[j] = half;
                            }
                        }
                    }
                }
            }
        }
        self.sd = sd;
        self.update_color();
    }

    /// Resample the colour array onto the lattice and pick its display range (over inside samples).
    pub fn update_color(&mut self) {
        let n = self.count();
        let Some(c) = self.color else {
            self.col = vec![0.0; n];
            return;
        };
        let a = &self.file.arrays[c];
        let src = scalars(a);
        let (co, cd) = self.file.lattice(a.cell);
        self.col = if cd == self.dims && co == self.origin {
            src
        } else {
            let [nx, ny, _] = self.dims;
            (0..n)
                .map(|i| {
                    let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                    let p = [x, y, z];
                    let q: [f64; 3] = std::array::from_fn(|k| {
                        let pos = self.origin[k] + p[k] as f64 * self.spacing[k];
                        ((pos - co[k]) / self.spacing[k]).clamp(0.0, (cd[k] - 1) as f64)
                    });
                    trilinear(&src, cd, q)
                })
                .collect()
        };
        let mut vals: Vec<f32> = (0..n)
            .filter(|&i| self.blk[i] < 0.0)
            .map(|i| self.col[i])
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            vals = self.col.iter().copied().filter(|v| v.is_finite()).collect();
        }
        self.crange = percentile_range(&mut vals, 0.01);
    }

    /// Colour range in use: the fixed one if set, else the automatic one.
    pub fn crange(&self) -> (f32, f32) {
        self.color_range.unwrap_or(self.crange)
    }

    /// Automatic colour range (1st..99th percentile of the colour array over the inside samples).
    pub fn auto_crange(&self) -> (f32, f32) {
        self.crange
    }

    /// Grid of the shown array: first sample position, spacing, sample counts.
    pub fn grid(&self) -> ([f64; 3], [f64; 3], [usize; 3]) {
        (self.origin, self.spacing, self.dims)
    }

    /// Volume of the inside samples, each counted as one cell (mm³).
    pub fn inside_volume(&self) -> f64 {
        self.inside_count as f64 * self.spacing.iter().product::<f64>()
    }

    pub fn uniforms(&self) -> VolUniforms {
        let f = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        VolUniforms {
            origin: f(self.origin),
            spacing: f(self.spacing),
            dims: [
                self.dims[0] as u32,
                self.dims[1] as u32,
                self.dims[2] as u32,
                self.blocky as u32,
            ],
            info: [
                self.crange().0,
                self.crange().1,
                self.color.is_some() as u32 as f32,
                0.0,
            ],
        }
    }

    /// A shape whose `map()` reads this volume, for the preview harness.
    pub fn shape(&self, path: &Path) -> Shape {
        let lo: [f32; 3] = std::array::from_fn(|k| (self.origin[k] - 0.5 * self.spacing[k]) as f32);
        let hi: [f32; 3] =
            std::array::from_fn(|k| (self.origin[k] + ((self.dims[k] - 1) as f64 + 0.5) * self.spacing[k]) as f32);
        let voxel = self.spacing.iter().cloned().fold(f64::MAX, f64::min) as f32;
        let mut shape = Shape::generated(path, VOLUME_WGSL, (lo, hi), voxel);
        // the smooth field is only approximately a distance away from the surface
        shape.directives.insert("step".into(), "0.8".into());
        shape.up = self.up();
        shape
    }

    /// The side printed up that an optimizer result was made for (its `print_up`), if recorded.
    pub fn up(&self) -> Option<crate::mesh::Up> {
        let (_, v) = self.file.field.iter().find(|(n, _)| n == PRINT_UP)?;
        crate::mesh::Up::from_vector(v.get(..3)?.try_into().ok()?)
    }

    /// Threshold step for the +/- keys: 1/50 of the value range.
    pub fn nudge_iso(&mut self, dir: f32) {
        let step = (self.range.1 - self.range.0).max(1e-6) / 50.0;
        self.iso += dir * step;
        self.recompute();
    }

    pub fn label(&self) -> String {
        let op = if self.below { "<" } else { ">" };
        let mode = if self.blocky { "blocky" } else { "smooth" };
        let mut s = format!("{} {op} {:.4} ({mode})", self.array_name(), self.iso);
        if self.surf.is_some() && !self.blocky {
            s += &format!(" surface from {FILTERED}");
        }
        if let Some(c) = self.color_name() {
            let (a, b) = self.crange();
            s += &format!(" colour: {c} {a:.3}..{b:.3}");
        }
        s
    }
}

/// How `Volume::mesh` makes the surface. The default is the raw interpolated surface.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshOpts {
    /// fair the voxels' surface (`fairing::fair`) instead of interpolating the array
    pub fair: bool,
    /// Taubin smoothing passes (`Mesh::smooth`)
    pub smooth: u32,
    /// with `fair`: how far (mm) the surface may move outward from the voxels' to be smoother
    /// (`fairing::fair`); at least `fairing::DELTA` voxels whatever this says
    pub grow: f32,
}

impl Volume {
    /// Whether `mesh` fairs by default: for optimizer densities, whose surface is the voxels'.
    pub fn fair_by_default(&self) -> bool {
        self.array_name() == "density"
    }

    /// Mesh of the solid: marching cubes, smoothing, then scaled to the grid's position and spacing.
    /// The grid is padded with an outside layer so the mesh is always closed.
    ///
    /// The surface is either the array's threshold surface (linear interpolation, like the smooth
    /// view) or, with `opts.fair`, the faired surface of the samples past the threshold (smooth,
    /// without the voxel terraces of a near-binary array; `fairing`).
    ///
    /// For cell arrays the samples are cell centres and each is a cell of the design domain or not
    /// (`domain`; the padding is not). Where the solid ends at the domain's edge, the field doesn't
    /// say where between the two samples the surface is (x̃ there depends on the material nearby),
    /// but the domain does: on the cell face. So vertices on such edges go to the edge's midpoint,
    /// and the part ends in flat faces on the domain boundary and the grid box. Smoothing keeps
    /// vertices inside the domain (`domain_constraint`). With fairing, the faired field continues
    /// through the domain's boundary, and the vertices on the outline of each such face are put on
    /// it (`outline_to_field`), so the outline is a smooth curve rather than a staircase.
    pub fn mesh(&self, opts: &MeshOpts) -> Result<crate::mesh::Mesh> {
        if self.inside_count == 0 {
            bail!(
                "nothing inside: no samples of `{}` are past the threshold {}",
                self.array_name(),
                self.iso
            );
        }
        let cell = self.file.arrays[self.array].cell;
        let in_domain = |i: [i64; 3]| self.in_domain(i);
        let Padded { p, mut f, free, faired } = self.padded(opts);
        let n = f.len();
        let pidx = |x: usize, y: usize, z: usize| (z * p[1] + y) * p[0] + x;
        if let Some(fv) = &faired {
            for i in 0..n {
                f[i] = if free[i] { 1.0 } else { fv.v[i] };
            }
        }
        // mc wants z-fastest order; mesh in grid units
        let mut vol = vec![1f32; n];
        for z in 0..p[2] {
            for y in 0..p[1] {
                for x in 0..p[0] {
                    vol[(x * p[1] + y) * p[2] + z] = f[pidx(x, y, z)];
                }
            }
        }
        let grid = crate::mesher::Grid {
            vol,
            dims: p,
            origin: [-1.0; 3],
            step: 1.0,
        };
        let mut mesh = crate::mesher::polygonize(grid)?;
        // per vertex: the axis of the domain face it was put on
        let mut face = vec![None; mesh.verts.len()];
        if cell {
            for (vi, v) in mesh.verts.iter_mut().enumerate() {
                // a marching-cubes vertex lies on a lattice edge: two whole coordinates
                let frac: Vec<usize> = (0..3).filter(|&k| (v[k] - v[k].round()).abs() > 1e-4).collect();
                let &[k] = frac.as_slice() else { continue };
                let mut a: [i64; 3] = std::array::from_fn(|j| v[j].round() as i64);
                a[k] = v[k].floor() as i64;
                let mut b = a;
                b[k] += 1;
                if in_domain(a) != in_domain(b) {
                    v[k] = a[k] as f32 + 0.5;
                    face[vi] = Some(k);
                }
            }
        }
        let constraint = cell.then(|| domain_constraint(&free, p));
        // outlines onto the faired field before smoothing (the whole way: smoothing then evens out
        // the strip beside them) and after it (only as far as flips no triangle)
        if let Some(fv) = &faired {
            outline_to_field(&mut mesh, &face, fv, false);
        }
        mesh.smooth(
            opts.smooth,
            constraint.as_ref().map(|c| c as &crate::mesh::VertexConstraint),
        );
        if let Some(fv) = &faired {
            outline_to_field(&mut mesh, &face, fv, true);
            // an outline vertex near a corner of the domain may have slid past the other face
            if let Some(c) = &constraint {
                mesh.verts.iter_mut().for_each(c);
            }
        }
        for v in &mut mesh.verts {
            for (k, c) in v.iter_mut().enumerate() {
                *c = (self.origin[k] + *c as f64 * self.spacing[k]) as f32;
            }
        }
        Ok(mesh)
    }
}

/// A volume's surface field on its lattice padded by one sample on each side (x-fastest).
struct Padded {
    p: [usize; 3],
    /// the shown array less the threshold, signed `< 0` inside (1 at `free` samples)
    f: Vec<f32>,
    /// samples outside the design domain or in the padding
    free: Vec<bool>,
    /// with `MeshOpts::fair`: the faired field (voxels)
    faired: Option<crate::fairing::Field>,
}

/// A signed distance field on a regular lattice (mm, negative inside, x-fastest): a volume's solid
/// for use inside a shape (`Volume::sdf`).
#[derive(Clone, Debug)]
pub struct SdfGrid {
    /// position of the first sample
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub dims: [usize; 3],
    pub sd: Vec<f32>,
}

impl SdfGrid {
    /// The centre of the lattice's box.
    pub fn centre(&self) -> [f32; 3] {
        std::array::from_fn(|k| (self.origin[k] + 0.5 * (self.dims[k] - 1) as f64 * self.spacing[k]) as f32)
    }

    /// Half the lattice box's size.
    pub fn half(&self) -> [f32; 3] {
        std::array::from_fn(|k| (0.5 * (self.dims[k] - 1) as f64 * self.spacing[k]) as f32)
    }

    /// The field at `p` as `import::wgsl`'s `g3d_imp_sample` computes it (for picking on the CPU).
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let q: [f64; 3] = std::array::from_fn(|k| (p[k] as f64 - self.origin[k]) / self.spacing[k]);
        let qc: [f64; 3] = std::array::from_fn(|k| q[k].clamp(0.0, (self.dims[k] - 1) as f64));
        let v = trilinear(&self.sd, self.dims, qc);
        let out = (0..3)
            .map(|k| ((q[k] - qc[k]) * self.spacing[k]).powi(2))
            .sum::<f64>()
            .sqrt() as f32;
        if out > 0.0 {
            (out * out + v.max(0.0).powi(2)).sqrt()
        } else {
            v
        }
    }
}

impl Volume {
    /// Whether padded-lattice index `i` (lattice index; may be -1 or `dims`) is in the design domain.
    fn in_domain(&self, i: [i64; 3]) -> bool {
        let [nx, ny, _] = self.dims;
        (0..3).all(|k| i[k] >= 0 && i[k] < self.dims[k] as i64)
            && (!self.file.arrays[self.array].cell
                || self
                    .domain
                    .as_ref()
                    .is_none_or(|d| d[((i[2] as usize) * ny + i[1] as usize) * nx + i[0] as usize]))
    }

    fn padded(&self, opts: &MeshOpts) -> Padded {
        let [nx, ny, nz] = self.dims;
        let p = [nx + 2, ny + 2, nz + 2];
        let field = self.field();
        let pidx = |x: usize, y: usize, z: usize| (z * p[1] + y) * p[0] + x;
        let n = p[0] * p[1] * p[2];
        let mut free = vec![false; n];
        let mut f = vec![1f32; n];
        for z in 0..p[2] {
            for y in 0..p[1] {
                for x in 0..p[0] {
                    let c = [x as i64 - 1, y as i64 - 1, z as i64 - 1];
                    let i = pidx(x, y, z);
                    if !self.in_domain(c) {
                        free[i] = true;
                        continue;
                    }
                    let v = field[((c[2] as usize) * ny + c[1] as usize) * nx + c[0] as usize];
                    let d = if self.below { v - self.iso } else { self.iso - v };
                    f[i] = if d.is_finite() { d } else { 1.0 };
                }
            }
        }
        let faired = opts.fair.then(|| {
            let inside: Vec<bool> = f.iter().zip(&free).map(|(&v, &fr)| !fr && v < 0.0).collect();
            crate::fairing::fair(&inside, Some(&free), p, opts.grow / self.mean_spacing())
        });
        Padded { p, f, free, faired }
    }

    /// The (geometric) mean spacing: the fairing works in voxels of this size.
    fn mean_spacing(&self) -> f32 {
        (self.spacing.iter().product::<f64>()).cbrt() as f32
    }

    /// The solid as a signed distance field, for combining it with other shapes (`import.rs`).
    ///
    /// With `opts.fair` (as `mesh` would), the faired surface, cut off where the design domain and
    /// the grid end: halfway between the last sample in it and the first outside, as the mesh's flat
    /// faces there are. That is on the lattice padded by one sample, so the solid is closed.
    /// Without, the smooth view's field (`sd`), on the lattice itself. `opts.smooth` is unused.
    pub fn sdf(&self, opts: &MeshOpts) -> SdfGrid {
        if !opts.fair {
            return SdfGrid {
                origin: self.origin,
                spacing: self.spacing,
                dims: self.dims,
                sd: self.sd.clone(),
            };
        }
        let Padded { p, free, faired, .. } = self.padded(opts);
        let faired = faired.expect("fair");
        let dom = crate::fairing::mask_sdf(&free.iter().map(|f| !f).collect::<Vec<_>>(), p);
        let mut sd: Vec<f32> = faired.v.iter().zip(&dom.v).map(|(a, b)| a.max(*b)).collect();
        // a solid sample next to the domain's edge: at least half a voxel inside, so the face is on
        // the cell face (as the mesh puts it) even where the faired field is shallow (thin plates)
        let [nx, ny, nz] = p;
        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = (z * ny + y) * nx + x;
                    let edge = [1, nx, nx * ny].iter().any(|&s| free[i - s] || free[i + s]);
                    if !free[i] && sd[i] < 0.0 && edge {
                        sd[i] = sd[i].min(-0.5);
                    }
                }
            }
        }
        let h = self.mean_spacing();
        SdfGrid {
            origin: std::array::from_fn(|k| self.origin[k] - self.spacing[k]),
            spacing: self.spacing,
            dims: p,
            sd: sd.into_iter().map(|v| v * h).collect(),
        }
    }
}

/// For `Mesh::smooth` of a cell array's mesh, in grid units: moves a vertex that left the region it
/// may occupy (the design domain, else the grid box; `outside` marks the padded lattice's samples
/// outside it) back onto its boundary. The region is the 0.5 level of the trilinear domain
/// indicator, which runs along the cell faces: the same place `Volume::mesh` puts the surface where
/// the solid meets the domain's edge. A few Newton steps along the indicator's gradient; vertices
/// inside are left alone.
fn domain_constraint(outside: &[bool], p: [usize; 3]) -> impl Fn(&mut [f32; 3]) + Sync + use<> {
    let g: Vec<f32> = outside.iter().map(|&o| if o { 0.0 } else { 1.0 }).collect();
    move |v: &mut [f32; 3]| {
        // padded lattice coordinates
        let mut q: [f64; 3] = std::array::from_fn(|k| v[k] as f64 + 1.0);
        let mut moved = false;
        for _ in 0..4 {
            let (val, grad) = trilinear_grad(&g, p, q);
            if val >= 0.5 {
                break;
            }
            let gg: f64 = grad.iter().map(|c| c * c).sum();
            if gg < 1e-12 {
                break;
            }
            // aim slightly inside, so the vertex isn't left a rounding error outside
            let s = (0.5 + 1e-4 - val) / gg;
            for k in 0..3 {
                q[k] += s * grad[k];
            }
            moved = true;
        }
        if moved {
            for k in 0..3 {
                v[k] = (q[k] - 1.0) as f32;
            }
        }
    }
}

/// Slides the vertices on the outline of each domain face (face vertices with a neighbour on the
/// free surface, so not where two faces of the domain meet) within the face's plane onto the zero
/// level of the faired field `f` (padded lattice: grid coordinates + 1), which continues through
/// the face. Marching cubes can only put face vertices at lattice points, so without this the
/// outline is a staircase. Moves are limited to 1.5 voxels; with `safe`, each vertex goes the
/// largest of 1, ½, ¼ or ⅛ of the way that flips none of its triangles (a vertex that would overrun
/// its neighbours on the surface goes less far).
fn outline_to_field(mesh: &mut crate::mesh::Mesh, face: &[Option<usize>], f: &crate::fairing::Field, safe: bool) {
    let nv = mesh.verts.len();
    let mut on_outline = vec![false; nv];
    let mut vtris: Vec<Vec<u32>> = vec![vec![]; nv];
    for (ti, t) in mesh.tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t[e] as usize, t[(e + 1) % 3] as usize);
            // next to the free surface; a neighbour on another face is a corner of the domain
            if face[a].is_some() && face[b].is_none() {
                on_outline[a] = true;
            }
            if face[b].is_some() && face[a].is_none() {
                on_outline[b] = true;
            }
            vtris[a].push(ti as u32);
        }
    }
    let normal = |v: &[[f32; 3]], t: [u32; 3]| {
        let [a, b, c] = t.map(|i| v[i as usize]);
        let u: [f32; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let w: [f32; 3] = std::array::from_fn(|k| c[k] - a[k]);
        [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ]
    };
    for i in 0..nv {
        let (true, Some(k)) = (on_outline[i], face[i]) else {
            continue;
        };
        let start = mesh.verts[i];
        let mut q = start.map(|c| c + 1.0);
        for _ in 0..4 {
            let (val, mut g) = f.sample(q);
            g[k] = 0.0;
            let gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            if gg < 1e-8 || val.abs() < 1e-4 {
                break;
            }
            for j in 0..3 {
                q[j] -= val * g[j] / gg;
            }
        }
        if f.sample(q).0.abs() > 0.05 {
            continue; // no zero level nearby in the plane
        }
        let mut d: [f32; 3] = std::array::from_fn(|j| q[j] - 1.0 - start[j]);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len > 1.5 {
            d = d.map(|c| c * 1.5 / len);
        }
        if !safe {
            mesh.verts[i] = std::array::from_fn(|j| start[j] + d[j]);
            continue;
        }
        let before: Vec<[f32; 3]> = vtris[i]
            .iter()
            .map(|&t| normal(&mesh.verts, mesh.tris[t as usize]))
            .collect();
        for s in [1.0, 0.5, 0.25, 0.125] {
            mesh.verts[i] = std::array::from_fn(|j| start[j] + s * d[j]);
            let ok = vtris[i].iter().zip(&before).all(|(&t, n0)| {
                let n = normal(&mesh.verts, mesh.tris[t as usize]);
                n[0] * n0[0] + n[1] * n0[1] + n[2] * n0[2] > 0.0
            });
            if ok {
                break;
            }
            mesh.verts[i] = start;
        }
    }
}

/// The `q` and `1 - q` quantiles: a colour range that a few extreme values (stress or energy
/// hotspots) don't flatten. Falls back to min..max when that range would be empty.
fn percentile_range(v: &mut [f32], q: f64) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let n = v.len();
    let lo_i = ((n - 1) as f64 * q).round() as usize;
    let hi_i = ((n - 1) as f64 * (1.0 - q)).round() as usize;
    let cmp = |a: &f32, b: &f32| a.total_cmp(b);
    let lo = *v.select_nth_unstable_by(lo_i, cmp).1;
    let hi = *v.select_nth_unstable_by(hi_i, cmp).1;
    if hi > lo { (lo, hi) } else { finite_range(v) }
}

fn trilinear(v: &[f32], d: [usize; 3], q: [f64; 3]) -> f32 {
    let i0: [usize; 3] = std::array::from_fn(|k| (q[k].floor() as usize).min(d[k].saturating_sub(2)));
    let f: [f32; 3] = std::array::from_fn(|k| (q[k] - i0[k] as f64).clamp(0.0, 1.0) as f32);
    let at = |x: usize, y: usize, z: usize| {
        let (x, y, z) = (x.min(d[0] - 1), y.min(d[1] - 1), z.min(d[2] - 1));
        v[(z * d[1] + y) * d[0] + x]
    };
    let [x, y, z] = i0;
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c = |dy: usize, dz: usize| lerp(at(x, y + dy, z + dz), at(x + 1, y + dy, z + dz), f[0]);
    lerp(lerp(c(0, 0), c(1, 0), f[1]), lerp(c(0, 1), c(1, 1), f[1]), f[2])
}

/// Trilinear value and gradient (per lattice unit) of an x-fastest grid at lattice point `q`;
/// 0 outside the grid.
fn trilinear_grad(v: &[f32], d: [usize; 3], q: [f64; 3]) -> (f64, [f64; 3]) {
    if (0..3).any(|k| q[k] < 0.0 || q[k] > (d[k] - 1) as f64) {
        return (0.0, [0.0; 3]);
    }
    let i: [usize; 3] = std::array::from_fn(|k| (q[k].floor() as usize).min(d[k] - 2));
    let t: [f64; 3] = std::array::from_fn(|k| q[k] - i[k] as f64);
    let at = |a: usize, b: usize, c: usize| v[((i[2] + c) * d[1] + i[1] + b) * d[0] + i[0] + a] as f64;
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

/// Exact Euclidean distance from every sample to the nearest sample with `seed` set (separable
/// algorithm of Felzenszwalb & Huttenlocher), on an x-fastest grid with per-axis spacing.
/// Infinite where there are no seeds.
pub fn edt(seed: &[bool], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut f: Vec<f64> = seed.iter().map(|&s| if s { 0.0 } else { f64::INFINITY }).collect();
    let mut line = vec![];
    let mut out = vec![];
    let mut v = vec![];
    let mut zb = vec![];
    for axis in 0..3 {
        let n = dims[axis];
        let stride = [1, nx, nx * ny][axis];
        let starts: Vec<usize> = match axis {
            0 => (0..ny * nz).map(|i| i * nx).collect(),
            1 => (0..nz).flat_map(|z| (0..nx).map(move |x| z * nx * ny + x)).collect(),
            _ => (0..nx * ny).collect(),
        };
        line.resize(n, 0.0);
        out.resize(n, 0.0);
        for s in starts {
            for k in 0..n {
                line[k] = f[s + k * stride];
            }
            dt1d(&line, spacing[axis], &mut out, &mut v, &mut zb);
            for k in 0..n {
                f[s + k * stride] = out[k];
            }
        }
    }
    f.into_iter().map(|d| d.sqrt() as f32).collect()
}

/// 1D squared distance transform: `out[q] = min_p (w (q - p))^2 + f[p]` over finite `f[p]`.
fn dt1d(f: &[f64], w: f64, out: &mut [f64], v: &mut Vec<usize>, z: &mut Vec<f64>) {
    let n = f.len();
    v.clear();
    z.clear();
    for q in 0..n {
        if !f[q].is_finite() {
            continue;
        }
        let xq = q as f64 * w;
        while let Some(&p) = v.last() {
            let xp = p as f64 * w;
            let s = ((f[q] + xq * xq) - (f[p] + xp * xp)) / (2.0 * (xq - xp));
            if s <= *z.last().unwrap() {
                v.pop();
                z.pop();
            } else {
                v.push(q);
                z.push(s);
                break;
            }
        }
        if v.is_empty() {
            v.push(q);
            z.push(f64::NEG_INFINITY);
        }
    }
    if v.is_empty() {
        out.fill(f64::INFINITY);
        return;
    }
    let mut j = 0;
    for (q, o) in out.iter_mut().enumerate() {
        let x = q as f64 * w;
        while j + 1 < v.len() && z[j + 1] < x {
            j += 1;
        }
        let d = x - v[j] as f64 * w;
        *o = d * d + f[v[j]];
    }
}

/// `map()` and the colour tint for a volume, read from storage buffers. Appended to by the
/// preview harness (which supplies binding 0, the view uniforms).
pub const VOLUME_WGSL: &str = r#"
// ---- gen3d volume (.vti) shape ----
struct G3dVol { origin: vec4f, spacing: vec4f, dims: vec4u, info: vec4f }  // dims.w: blocky; info: colour min, max, on
@group(0) @binding(1) var<uniform> g3d_vol: G3dVol;
@group(0) @binding(2) var<storage, read> g3d_sd: array<f32>;
@group(0) @binding(3) var<storage, read> g3d_blk: array<f32>;
@group(0) @binding(4) var<storage, read> g3d_col: array<f32>;

fn g3d_idx(i: vec3i) -> u32 {
    let d = vec3i(g3d_vol.dims.xyz);
    let c = clamp(i, vec3i(0), d - 1);
    return u32((c.z * d.y + c.y) * d.x + c.x);
}
fn g3d_inrange(i: vec3i) -> bool {
    return all(i >= vec3i(0)) && all(i < vec3i(g3d_vol.dims.xyz));
}
// lattice coordinates of a point
fn g3d_lat(p: vec3f) -> vec3f { return (p - g3d_vol.origin.xyz) / g3d_vol.spacing.xyz; }

fn g3d_tri_sd(q: vec3f) -> f32 {
    let i = clamp(vec3i(floor(q)), vec3i(0), max(vec3i(g3d_vol.dims.xyz) - 2, vec3i(0)));
    let f = clamp(q - vec3f(i), vec3f(0.0), vec3f(1.0));
    let a = mix(mix(g3d_sd[g3d_idx(i)], g3d_sd[g3d_idx(i + vec3i(1, 0, 0))], f.x),
                mix(g3d_sd[g3d_idx(i + vec3i(0, 1, 0))], g3d_sd[g3d_idx(i + vec3i(1, 1, 0))], f.x), f.y);
    let b = mix(mix(g3d_sd[g3d_idx(i + vec3i(0, 0, 1))], g3d_sd[g3d_idx(i + vec3i(1, 0, 1))], f.x),
                mix(g3d_sd[g3d_idx(i + vec3i(0, 1, 1))], g3d_sd[g3d_idx(i + vec3i(1, 1, 1))], f.x), f.y);
    return mix(a, b, f.z);
}
fn g3d_tri_col(q: vec3f) -> f32 {
    let i = clamp(vec3i(floor(q)), vec3i(0), max(vec3i(g3d_vol.dims.xyz) - 2, vec3i(0)));
    let f = clamp(q - vec3f(i), vec3f(0.0), vec3f(1.0));
    let a = mix(mix(g3d_col[g3d_idx(i)], g3d_col[g3d_idx(i + vec3i(1, 0, 0))], f.x),
                mix(g3d_col[g3d_idx(i + vec3i(0, 1, 0))], g3d_col[g3d_idx(i + vec3i(1, 1, 0))], f.x), f.y);
    let b = mix(mix(g3d_col[g3d_idx(i + vec3i(0, 0, 1))], g3d_col[g3d_idx(i + vec3i(1, 0, 1))], f.x),
                mix(g3d_col[g3d_idx(i + vec3i(0, 1, 1))], g3d_col[g3d_idx(i + vec3i(1, 1, 1))], f.x), f.y);
    return mix(a, b, f.z);
}

fn g3d_vbox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn g3d_smooth(p: vec3f) -> f32 {
    let q = g3d_lat(p);
    let qc = clamp(q, vec3f(0.0), vec3f(g3d_vol.dims.xyz - 1u));
    let v = g3d_tri_sd(qc);
    let out = length((q - qc) * g3d_vol.spacing.xyz);
    // outside the lattice the surface is at least as far as from the nearest lattice point
    if (out > 0.0) { return sqrt(out * out + max(v, 0.0) * max(v, 0.0)); }
    return v;
}

fn g3d_blocky(p: vec3f) -> f32 {
    let s = g3d_vol.spacing.xyz;
    let o = g3d_vol.origin.xyz;
    let qi = vec3i(round(g3d_lat(p)));
    let c0 = o + vec3f(qi) * s;
    let h = 0.5 * s;
    var dIn = 1e9;
    var dOut = 1e9;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let n = qi + vec3i(dx, dy, dz);
                let bd = g3d_vbox(p - (o + vec3f(n) * s), h);
                if (g3d_inrange(n) && g3d_blk[g3d_idx(n)] < 0.0) { dIn = min(dIn, bd); } else { dOut = min(dOut, bd); }
            }
        }
    }
    // cubes beyond the 3x3x3 block are at least this far away
    let a = 1.5 * s - abs(p - c0);
    let dBlock = min(a.x, min(a.y, a.z));
    if (g3d_inrange(qi) && g3d_blk[g3d_idx(qi)] < 0.0) { return -min(dOut, dBlock); }
    let qc = clamp(qi, vec3i(0), vec3i(g3d_vol.dims.xyz) - 1);
    let far = g3d_blk[g3d_idx(qc)] - length(p - (o + vec3f(qc) * s)) - 0.5 * length(s);
    return min(dIn, max(far, dBlock));
}

fn map(p: vec3f) -> f32 {
    if (g3d_vol.dims.w == 1u) { return g3d_blocky(p); }
    return g3d_smooth(p);
}

fn g3d_viridis(t: f32) -> vec3f {
    let c0 = vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    let c1 = vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    let c2 = vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    let c3 = vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    let c4 = vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105);
    let c5 = vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234);
    let c6 = vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

fn g3d_field_tint(p: vec3f, ps: vec3f, scale: f32) -> vec4f {
    if (g3d_vol.info.z == 0.0) { return vec4f(0.0); }
    let q = g3d_lat(p);
    var v: f32;
    if (g3d_vol.dims.w == 1u) {
        v = g3d_col[g3d_idx(vec3i(round(q)))];
    } else {
        v = g3d_tri_col(clamp(q, vec3f(0.0), vec3f(g3d_vol.dims.xyz - 1u)));
    }
    let t = clamp((v - g3d_vol.info.x) / max(g3d_vol.info.y - g3d_vol.info.x, 1e-20), 0.0, 1.0);
    return vec4f(pow(clamp(g3d_viridis(t), vec3f(0.0), vec3f(1.0)), vec3f(2.2)), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_matches_brute_force() {
        let dims = [7, 5, 4];
        let sp = [0.5, 1.0, 2.0];
        let n: usize = dims.iter().product();
        let seed: Vec<bool> = (0..n).map(|i| i % 11 == 3).collect();
        let d = edt(&seed, dims, sp);
        let pos = |i: usize| {
            let (x, y, z) = (i % 7, (i / 7) % 5, i / 35);
            [x as f64 * sp[0], y as f64 * sp[1], z as f64 * sp[2]]
        };
        for (i, &got) in d.iter().enumerate() {
            let want = (0..n)
                .filter(|&j| seed[j])
                .map(|j| {
                    let (a, b) = (pos(i), pos(j));
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
                })
                .fold(f64::MAX, f64::min);
            assert!((got as f64 - want).abs() < 1e-4, "{i}: {got} vs {want}");
        }
        assert!(edt(&vec![false; n], dims, sp).iter().all(|v| v.is_infinite()));
    }

    /// An optimizer-style result: a solid under the slope z = 0.3 x + 3, binary `density` and a
    /// linear `density_filtered`, with cells x >= 20 outside the domain (NaN). One empty cell's
    /// density is a hair below 0 (as the AM filter's rounding gave), and it was made for -x up.
    fn slope_vti(path: &Path) {
        let n = [24usize, 16, 16];
        let plane = |x: f32| 0.3 * x + 3.0;
        let (mut dens, mut filt) = (vec![], vec![]);
        for k in 0..n[2] {
            for _j in 0..n[1] {
                for i in 0..n[0] {
                    let (x, z) = (i as f32 + 0.5, k as f32 + 0.5);
                    let t = (0.5 + (plane(x) - z) / 3.0).clamp(0.0, 1.0);
                    let inside_domain = i < 20;
                    dens.push(if inside_domain && t > 0.5 { 1.0 } else { 0.0 });
                    filt.push(if inside_domain { t } else { f32::NAN });
                }
            }
        }
        *dens.last_mut().unwrap() = -3e-8;
        let arr = |name, data| crate::vti::GridArray {
            name,
            cell: true,
            comps: 1,
            data,
        };
        crate::vti::write_grid(
            path,
            [0.0; 3],
            [1.0; 3],
            n.map(|v| v + 1),
            &[arr("density", &dens), arr(FILTERED, &filt)],
            &[(PRINT_UP, &crate::mesh::Up::MinusX.vector())],
        )
        .unwrap();
    }

    /// A density a hair below 0 is still a density (inside above the middle of its range), and the
    /// result's print orientation comes back as its shape's.
    #[test]
    fn optimizer_result_threshold_and_orientation() {
        let dir = std::env::temp_dir().join(format!("gen3d-resup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slope.vti");
        slope_vti(&path);
        let vol = Volume::load(&path, &VolumeOpts::default()).unwrap();
        assert!(
            !vol.below && (vol.iso - 0.5).abs() < 0.01,
            "iso {} below {}",
            vol.iso,
            vol.below
        );
        assert_eq!(vol.up(), Some(crate::mesh::Up::MinusX));
        assert_eq!(vol.shape(&path).up, Some(crate::mesh::Up::MinusX));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn filtered_density_gives_a_smooth_surface_inside_the_domain() {
        let dir = std::env::temp_dir().join(format!("gen3d-slope-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slope.vti");
        slope_vti(&path);
        let vol = Volume::load(&path, &VolumeOpts::default()).unwrap();
        assert!(vol.label().contains(FILTERED), "{}", vol.label());
        let raw = MeshOpts::default();
        let m = vol.mesh(&raw).unwrap();
        assert!(m.is_watertight());

        // the slope away from the edges: on the plane (binary density would leave 0.5-voxel steps)
        let off_plane = |m: &crate::mesh::Mesh| {
            m.verts
                .iter()
                .filter(|v| v[0] > 3.0 && v[0] < 17.0 && v[1] > 3.0 && v[1] < 13.0 && v[2] > 1.0)
                .map(|v| (v[2] - (0.3 * v[0] + 3.0)).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(off_plane(&m) < 0.02, "{}", off_plane(&m));
        // where it reaches the domain's edge: a flat face on the cell faces at x = 20
        let x_max = m.verts.iter().map(|v| v[0]).fold(f32::MIN, f32::max);
        assert!((x_max - 20.0).abs() < 1e-4, "{x_max}");
        assert!(
            m.verts
                .iter()
                .filter(|v| v[0] > 19.5)
                .all(|v| (v[0] - 20.0).abs() < 1e-4)
        );

        let v0 = m.volume();
        for fair in [false, true] {
            let m = vol
                .mesh(&MeshOpts {
                    fair,
                    smooth: 10,
                    ..Default::default()
                })
                .unwrap();
            assert!(m.is_watertight());
            let x_max = m.verts.iter().map(|v| v[0]).fold(f32::MIN, f32::max);
            assert!(
                x_max < 20.0 + 1e-3,
                "fair {fair}: smoothing left the domain: x = {x_max}"
            );
            // this x̃ is exactly linear, so its surface is the plane; fairing uses only the binary
            // density (0.5-voxel steps) and gets within a quarter voxel of it on this small slab
            let tol = if fair { 0.25 } else { 0.05 };
            assert!(off_plane(&m) < tol, "fair {fair}: {}", off_plane(&m));
            assert!(
                (m.volume() / v0 - 1.0).abs() < 0.01,
                "fair {fair}: volume {v0} -> {}",
                m.volume()
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
