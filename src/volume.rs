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
    pub range: (f32, f32),
    /// smooth signed distance per sample
    pub sd: Vec<f32>,
    /// blocky: per sample, -EDT to the outside set when inside, +EDT to the inside set when outside
    pub blk: Vec<f32>,
    /// colour array resampled onto the lattice (all zeros when off)
    pub col: Vec<f32>,
    crange: (f32, f32),
    pub inside_count: usize,
}

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
            range: (0.0, 0.0),
            sd: vec![],
            blk: vec![],
            col: vec![],
            crange: (0.0, 1.0),
            inside_count: 0,
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
    /// *sdf* or with values on both sides of 0), the middle of the range with inside above otherwise
    /// (densities, masks).
    pub fn select(&mut self, i: usize) {
        self.array = i;
        let a = &self.file.arrays[i];
        (self.origin, self.dims) = self.file.lattice(a.cell);
        self.spacing = self.file.spacing.map(f64::abs);
        self.values = scalars(a);
        self.range = finite_range(&self.values);
        let (lo, hi) = self.range;
        let sdf_like = a.comps == 1 && (a.name.to_lowercase().contains("sdf") || (lo < 0.0 && hi > 0.0));
        self.below = sdf_like;
        self.iso = if sdf_like { 0.0 } else { 0.5 * (lo + hi) };
    }

    pub fn array_name(&self) -> &str {
        &self.file.arrays[self.array].name
    }

    pub fn color_name(&self) -> Option<&str> {
        self.color.map(|c| self.file.arrays[c].name.as_str())
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
        let inside: Vec<bool> = self.values.iter().map(|&v| self.inside(v)).collect();
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
        let vals = &self.values;
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
        let inside: Vec<f32> = (0..n).filter(|&i| self.blk[i] < 0.0).map(|i| self.col[i]).collect();
        self.crange = finite_range(if inside.is_empty() { &self.col } else { &inside });
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
            info: [self.crange.0, self.crange.1, self.color.is_some() as u32 as f32, 0.0],
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
        shape
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
        if let Some(c) = self.color_name() {
            s += &format!(" colour: {c} {:.3}..{:.3}", self.crange.0, self.crange.1);
        }
        s
    }
}

impl Volume {
    /// Marching-cubes mesh of the threshold surface (linear interpolation of the array, like the
    /// smooth view). The grid is padded with an outside layer so the mesh is always closed.
    pub fn mesh(&self) -> Result<crate::mesh::Mesh> {
        if self.inside_count == 0 {
            bail!(
                "nothing inside: no samples of `{}` are past the threshold {}",
                self.array_name(),
                self.iso
            );
        }
        let [nx, ny, nz] = self.dims;
        let p = [nx + 2, ny + 2, nz + 2];
        // f < 0 inside; mc wants z-fastest order
        let mut vol = vec![1f32; p[0] * p[1] * p[2]];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let v = self.values[(z * ny + y) * nx + x];
                    let f = if self.below { v - self.iso } else { self.iso - v };
                    vol[((x + 1) * p[1] + y + 1) * p[2] + z + 1] = if f.is_finite() { f } else { 1.0 };
                }
            }
        }
        // mesh in grid units, then scale each axis (spacing may differ per axis)
        let grid = crate::mesher::Grid {
            vol,
            dims: p,
            origin: [-1.0; 3],
            step: 1.0,
        };
        let mut mesh = crate::mesher::polygonize(grid)?;
        for v in &mut mesh.verts {
            for (k, c) in v.iter_mut().enumerate() {
                *c = (self.origin[k] + *c as f64 * self.spacing[k]) as f32;
            }
        }
        Ok(mesh)
    }
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
}
