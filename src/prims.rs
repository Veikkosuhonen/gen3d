//! Primitives placed in the Design step's editor. They live in a sidecar file next to the shape
//! (`knob.wgsl` → `knob.prims.json`) and are combined with the script's `map()` in list order, each
//! by its [`Op`] (union, subtract or intersect, optionally smooth): `load_shape` renames
//! the script's `fn map` to `g3d_script_map` and appends a generated `map` (see [`wgsl`]), so every
//! consumer (preview, mesher, optimizer, VTI export, the CLI) sees the combined shape.
//!
//! The primitives' numbers live in one `array<vec4f>` named `g3d_ed`: a uniform buffer in the preview,
//! so moving or resizing one is a buffer write (no shader compile), and a `const` everywhere else.
//! Adding, removing, hiding or changing a kind regenerates the source.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    #[default]
    Sphere,
    Box,
    Cylinder,
    Capsule,
    Torus,
    Cone,
}

impl Kind {
    pub const ALL: [Kind; 6] = [
        Kind::Sphere,
        Kind::Box,
        Kind::Cylinder,
        Kind::Capsule,
        Kind::Torus,
        Kind::Cone,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Sphere => "sphere",
            Kind::Box => "box",
            Kind::Cylinder => "cylinder",
            Kind::Capsule => "capsule",
            Kind::Torus => "torus",
            Kind::Cone => "cone",
        }
    }

    /// What `dims[0..n]` mean for this kind (all in mm). Cylinders, capsules, tori and cones are
    /// along the primitive's local Z.
    pub fn dims(self) -> &'static [&'static str] {
        match self {
            Kind::Sphere => &["radius"],
            Kind::Box => &["size x", "size y", "size z"],
            Kind::Cylinder => &["radius", "height"],
            Kind::Capsule => &["radius", "height"],
            Kind::Torus => &["radius", "tube radius"],
            Kind::Cone => &["bottom radius", "top radius", "height"],
        }
    }

    /// Whether `round` (edge rounding) applies.
    pub fn rounds(self) -> bool {
        matches!(self, Kind::Box | Kind::Cylinder)
    }

    /// Reasonable dimensions for a new primitive about `s` mm across.
    pub fn default_dims(self, s: f32) -> [f32; 3] {
        let h = s / 2.0;
        match self {
            Kind::Sphere => [h, 0.0, 0.0],
            Kind::Box => [s, s, s],
            Kind::Cylinder => [h, s, 0.0],
            Kind::Capsule => [s / 4.0, s * 1.5, 0.0],
            Kind::Torus => [h, s / 6.0, 0.0],
            Kind::Cone => [h, 0.0, s],
        }
    }
}

/// How a primitive combines with everything before it (the script, then earlier primitives).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    #[default]
    Union,
    Subtract,
    Intersect,
    /// Doesn't change the shape; only marks a region (for an optimization role).
    Region,
}

impl Op {
    pub const ALL: [Op; 4] = [Op::Union, Op::Subtract, Op::Intersect, Op::Region];

    pub fn name(self) -> &'static str {
        match self {
            Op::Union => "add",
            Op::Subtract => "subtract",
            Op::Intersect => "intersect",
            Op::Region => "region",
        }
    }

    /// The mark in the primitive list.
    pub fn sign(self) -> &'static str {
        match self {
            Op::Union => "+",
            Op::Subtract => "-",
            Op::Intersect => "&",
            Op::Region => "o",
        }
    }

    /// The value `g3d_ed_combine` branches on.
    fn code(self) -> f32 {
        match self {
            Op::Union => 0.0,
            Op::Subtract => 1.0,
            Op::Intersect => 2.0,
            Op::Region => 3.0,
        }
    }

    /// `a` combined with `b` with smoothing radius `k`, as `g3d_ed_combine` does (the tests' reference).
    #[cfg(test)]
    pub fn apply(self, a: f32, b: f32, k: f32) -> f32 {
        let mix = |x: f32, y: f32, h: f32| x + (y - x) * h;
        match self {
            Op::Region => a,
            Op::Union if k <= 0.0 => a.min(b),
            Op::Union => {
                let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
                mix(b, a, h) - k * h * (1.0 - h)
            }
            Op::Subtract if k <= 0.0 => a.max(-b),
            Op::Subtract => {
                let h = (0.5 - 0.5 * (a + b) / k).clamp(0.0, 1.0);
                mix(a, -b, h) + k * h * (1.0 - h)
            }
            Op::Intersect if k <= 0.0 => a.max(b),
            Op::Intersect => {
                let h = (0.5 - 0.5 * (b - a) / k).clamp(0.0, 1.0);
                mix(b, a, h) + k * h * (1.0 - h)
            }
        }
    }
}

/// What a primitive means to topology optimization (`gen3d optimize`). The role's region is the
/// primitive's inside plus a shell `g3d_reach` thick around it (about 1.5 elements; the optimizer sets
/// it), so a fixed *subtracted* cylinder clamps the wall of its hole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Optimized like the rest of the design domain.
    #[default]
    Design,
    /// Kept solid (non-design): its inside, where it's in the domain.
    Keep,
    /// Clamped: nodes in its region don't move. Kept solid there.
    Fixed,
    /// Loaded: its region carries `force` N along `load`. Kept solid there.
    Load,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::Design, Role::Keep, Role::Fixed, Role::Load];

    pub fn name(self) -> &'static str {
        match self {
            Role::Design => "design",
            Role::Keep => "keep solid",
            Role::Fixed => "fixed",
            Role::Load => "load",
        }
    }

    /// A boundary condition (the region includes the shell around the primitive).
    pub fn is_bc(self) -> bool {
        matches!(self, Role::Fixed | Role::Load)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prim {
    pub kind: Kind,
    /// How it combines with what comes before it.
    pub op: Op,
    pub name: String,
    /// Centre, mm.
    pub pos: [f32; 3],
    /// Rotation in degrees about X, then Y, then Z (world axes).
    pub rot: [f32; 3],
    /// Size; meaning per kind, see [`Kind::dims`].
    pub dims: [f32; 3],
    /// Edge rounding radius (boxes and cylinders).
    pub round: f32,
    /// Smoothing radius of the operation with everything before it (0 = sharp).
    pub blend: f32,
    pub hidden: bool,
    /// Meaning for topology optimization.
    pub role: Role,
    /// Load direction (`Role::Load`); normalized when used.
    pub load: [f32; 3],
    /// Load magnitude in N, spread over the region's nodes (`Role::Load`).
    pub force: f32,
    /// Load case (`Role::Load`): loads in one case act together; separate cases are separate
    /// situations the part must survive (the optimizer minimizes their summed compliance).
    pub case: u32,
    /// `force` is a mass in kg (`Role::Load`): its weight, `force` × 9.81 N.
    pub kg: bool,
    /// Rocking (`Role::Load`): its case also comes in variants with a sideways push of this share of
    /// the load added, both ways along each axis across the load (0 = off).
    pub rock: f32,
    /// The axes a fixed primitive holds (`Role::Fixed`): all three clamp; only the one across a
    /// surface makes a roller that lets the part slide along it.
    pub fix: [bool; 3],
}

/// Standard gravity, N per kg.
pub const G: f64 = 9.81;

impl Default for Prim {
    fn default() -> Prim {
        Prim {
            kind: Kind::default(),
            op: Op::default(),
            name: String::new(),
            pos: [0.0; 3],
            rot: [0.0; 3],
            dims: [0.0; 3],
            round: 0.0,
            blend: 0.0,
            hidden: false,
            role: Role::Design,
            load: [0.0, 0.0, -1.0],
            force: 100.0,
            case: 1,
            kg: false,
            rock: 0.0,
            fix: [true; 3],
        }
    }
}

impl Prim {
    /// The load in N (`force`, or its weight when given in kg).
    pub fn newtons(&self) -> f64 {
        if self.kg {
            self.force as f64 * G
        } else {
            self.force as f64
        }
    }

    /// The fixed axes as bits (bit c for axis c).
    pub fn fix_mask(&self) -> u8 {
        (0..3).filter(|&c| self.fix[c]).fold(0, |m, c| m | 1 << c)
    }
}

/// A load for the optimizer: the script's `field_load` (vector field, scaled to `--force`) or a
/// primitive with the load role (region < 0, with its own direction and force).
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub name: String,
    /// `fn(p) -> vec3f` if `dir` is None, else `fn(p) -> f32` (< 0 inside the region)
    pub func: String,
    pub dir: Option<[f32; 3]>,
    /// N; None = the optimizer's `--force`
    pub force: Option<f64>,
    pub case: u32,
    /// for a generated rocking variant: its case's name, e.g. "1 rocked +X"
    pub label: Option<String>,
}

/// A support for the optimizer: the script's `field_fixed` (all axes) or a primitive with the fixed
/// role.
#[derive(Clone, Debug)]
pub struct SupportSpec {
    pub name: String,
    /// `fn(p) -> f32`, < 0 inside the region
    pub func: String,
    /// the held axes, bit c for axis c
    pub axes: u8,
}

/// The supports of the visible primitives with the fixed role.
pub fn support_specs(prims: &[Prim]) -> Vec<SupportSpec> {
    prims
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.role == Role::Fixed && p.fix_mask() != 0)
        .map(|(i, p)| SupportSpec {
            name: p.name.clone(),
            func: format!("g3d_bc{i}"),
            axes: p.fix_mask(),
        })
        .collect()
}

/// The two world axes most across direction `d`, as indices.
fn across(d: [f32; 3]) -> [usize; 2] {
    let main = (0..3).max_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs())).unwrap_or(2);
    let mut out = [0; 2];
    for (k, c) in (0..3).filter(|&c| c != main).enumerate() {
        out[k] = c;
    }
    out
}

/// Case number of a generated rocking variant: case `c`'s variant `k` (1..=4).
pub fn rock_case(c: u32, k: u32) -> u32 {
    1000 * c + k
}

/// The sidecar file.
#[derive(Default, Serialize, Deserialize)]
struct PrimFile {
    #[serde(default)]
    note: String,
    #[serde(default)]
    prims: Vec<Prim>,
}

const NOTE: &str = "gen3d primitives, edited in the Design step; unioned with the shape's map()";

/// The sidecar that holds the primitives of `shape` (`x.wgsl` → `x.prims.json`).
pub fn sidecar(shape: &Path) -> PathBuf {
    let stem = shape.file_stem().unwrap_or_default().to_string_lossy();
    shape.with_file_name(format!("{stem}.prims.json"))
}

/// Read a sidecar. Non-finite numbers are zeroed so they can't break the generated WGSL.
pub fn load(path: &Path) -> Result<Vec<Prim>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let file: PrimFile =
        serde_json::from_str(&text).with_context(|| format!("{}: not a primitives file", path.display()))?;
    let mut prims = file.prims;
    for p in &mut prims {
        for v in p.pos.iter_mut().chain(&mut p.rot).chain(&mut p.dims) {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        for v in p.load.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        if !p.force.is_finite() {
            p.force = 0.0;
        }
        p.case = p.case.max(1);
        if !p.rock.is_finite() {
            p.rock = 0.0;
        }
        p.rock = p.rock.clamp(0.0, 1.0);
        for v in [&mut p.round, &mut p.blend] {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    Ok(prims)
}

pub fn save(path: &Path, prims: &[Prim]) -> Result<()> {
    let file = PrimFile {
        note: NOTE.into(),
        prims: prims.to_vec(),
    };
    let mut text = serde_json::to_string_pretty(&file)?;
    text.push('\n');
    // write-then-rename so the file watcher (and a running `gen3d` job) never sees half a file
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

// ------------------------------------------------------------------------------------- geometry

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn len(a: V3) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn len2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}
/// Rotate (x, y) counter-clockwise by `a` radians.
fn rot(x: f32, y: f32, a: f32) -> (f32, f32) {
    let (s, c) = a.sin_cos();
    (c * x - s * y, s * x + c * y)
}

impl Prim {
    pub fn new(kind: Kind, name: String, pos: V3, size: f32) -> Prim {
        Prim {
            kind,
            name,
            pos,
            dims: kind.default_dims(size),
            ..Default::default()
        }
    }

    /// World point → the primitive's local frame (same steps as `g3d_ed_local` in WGSL).
    fn local(&self, p: V3) -> V3 {
        let [a, b, c] = self.rot.map(f32::to_radians);
        let [mut x, mut y, mut z] = sub(p, self.pos);
        (x, y) = rot(x, y, -c);
        (z, x) = rot(z, x, -b);
        (y, z) = rot(y, z, -a);
        [x, y, z]
    }

    /// Signed distance, mirroring the WGSL in [`LIBRARY`] (for picking on the CPU).
    pub fn sdf(&self, p: V3) -> f32 {
        let q = self.local(p);
        let [d0, d1, d2] = self.dims;
        match self.kind {
            Kind::Sphere => len(q) - d0.max(0.0),
            Kind::Box => {
                let b = [d0, d1, d2].map(|v| (0.5 * v).max(0.0));
                let r = self.round.clamp(0.0, b[0].min(b[1]).min(b[2]));
                let d = [0, 1, 2].map(|k| q[k].abs() - (b[k] - r));
                len(d.map(|v| v.max(0.0))) + d[0].max(d[1]).max(d[2]).min(0.0) - r
            }
            Kind::Cylinder => {
                let (r, h) = (d0.max(0.0), (0.5 * d1).max(0.0));
                let rr = self.round.clamp(0.0, r.min(h));
                let dx = len2(q[0], q[1]) - (r - rr);
                let dy = q[2].abs() - (h - rr);
                dx.max(dy).min(0.0) + len2(dx.max(0.0), dy.max(0.0)) - rr
            }
            Kind::Capsule => {
                let r = d0.max(0.0);
                let h = (0.5 * d1 - r).max(0.0);
                len([q[0], q[1], q[2] - q[2].clamp(-h, h)]) - r
            }
            Kind::Torus => len2(len2(q[0], q[1]) - d0, q[2]) - d1.max(0.0),
            Kind::Cone => {
                let (r1, r2, h) = (d0.max(0.0), d1.max(0.0), (0.5 * d2).max(1e-4));
                let (qx, qy) = (len2(q[0], q[1]), q[2]);
                let (k1x, k1y) = (r2, h);
                let (k2x, k2y) = (r2 - r1, 2.0 * h);
                let rr = if qy < 0.0 { r1 } else { r2 };
                let (cax, cay) = (qx - qx.min(rr), qy.abs() - h);
                let t = (((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y)).clamp(0.0, 1.0);
                let (cbx, cby) = (qx - k1x + k2x * t, qy - k1y + k2y * t);
                let s = if cbx < 0.0 && cay < 0.0 { -1.0 } else { 1.0 };
                s * (cax * cax + cay * cay).min(cbx * cbx + cby * cby).sqrt()
            }
        }
    }

    /// Local direction → world (the inverse of `local`, without the translation).
    fn to_world(&self, v: V3) -> V3 {
        let [a, b, c] = self.rot.map(f32::to_radians);
        let [mut x, mut y, mut z] = v;
        (y, z) = rot(y, z, a);
        (z, x) = rot(z, x, b);
        (x, y) = rot(x, y, c);
        [x, y, z]
    }

    /// Axis-aligned box around the primitive (as rotated), including the bulge a smooth union can add.
    pub fn aabb(&self) -> (V3, V3) {
        let [d0, d1, d2] = self.dims.map(|v| v.max(0.0));
        let half = match self.kind {
            Kind::Sphere => [d0; 3],
            Kind::Box => [0.5 * d0, 0.5 * d1, 0.5 * d2],
            Kind::Cylinder => [d0, d0, 0.5 * d1],
            Kind::Capsule => [d0, d0, (0.5 * d1).max(d0)],
            Kind::Torus => [d0.abs() + d1, d0.abs() + d1, d1],
            Kind::Cone => [d0.max(d1), d0.max(d1), 0.5 * d2],
        };
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|e| self.to_world(e));
        let pad = 0.25 * self.blend.max(0.0);
        let ext = [0, 1, 2].map(|i| (0..3).map(|j| axes[j][i].abs() * half[j]).sum::<f32>() + pad);
        (
            [0, 1, 2].map(|i| self.pos[i] - ext[i]),
            [0, 1, 2].map(|i| self.pos[i] + ext[i]),
        )
    }

    /// Radius of a sphere around `pos` that contains the primitive (any rotation), including the
    /// bulge a smooth union can add.
    pub fn bound_radius(&self) -> f32 {
        let [d0, d1, d2] = self.dims.map(|v| v.max(0.0));
        let r = match self.kind {
            Kind::Sphere => d0,
            Kind::Box => 0.5 * len([d0, d1, d2]),
            Kind::Cylinder => len2(d0, 0.5 * d1),
            Kind::Capsule => (0.5 * d1).max(d0),
            Kind::Torus => d0.abs() + d1,
            Kind::Cone => len2(d0.max(d1), 0.5 * d2),
        };
        r + 0.25 * self.blend.max(0.0)
    }
}

/// Grow `bounds` to contain the visible added primitives (plus `margin`); subtracting and
/// intersecting ones never add material.
pub fn grow_bounds(bounds: Option<(V3, V3)>, prims: &[Prim], margin: f32) -> Option<(V3, V3)> {
    let mut out = bounds;
    for p in prims.iter().filter(|p| !p.hidden && p.op == Op::Union) {
        let (lo, hi) = p.aabb();
        let lo = lo.map(|c| c - margin);
        let hi = hi.map(|c| c + margin);
        out = Some(match out {
            None => (lo, hi),
            Some((a, b)) => ([0, 1, 2].map(|k| a[k].min(lo[k])), [0, 1, 2].map(|k| b[k].max(hi[k]))),
        });
    }
    out
}

/// The visible primitive a ray hits first (sphere tracing their plain union), if any.
pub fn pick(prims: &[Prim], ro: V3, rd: V3, far: f32) -> Option<usize> {
    let scene = |p: V3| {
        prims
            .iter()
            .enumerate()
            .filter(|(_, q)| !q.hidden)
            .map(|(i, q)| (q.sdf(p), i))
            .min_by(|a, b| a.0.total_cmp(&b.0))
    };
    let eps = far * 1e-5;
    let mut t = 0.0;
    for _ in 0..256 {
        let p = [0, 1, 2].map(|k| ro[k] + rd[k] * t);
        let (d, i) = scene(p)?;
        if d < eps.max(1e-3) {
            return Some(i);
        }
        t += d;
        if t > far {
            break;
        }
    }
    None
}

// ----------------------------------------------------------------------------------------- WGSL

/// The editor's own SDF library, prefixed so it can't clash with the script (which may or may not
/// include sdf.wgsl, or define its own `sdBox`).
pub const LIBRARY: &str = "
// ---- gen3d editor primitives (generated from the .prims.json file) ----
fn g3d_ed_rot(v: vec2f, a: f32) -> vec2f {
    let c = cos(a);
    let s = sin(a);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}
// world -> local; the primitive is rotated about X, then Y, then Z (degrees)
fn g3d_ed_local(p: vec3f, c: vec3f, r: vec3f) -> vec3f {
    let a = radians(r);
    var q = p - c;
    q = vec3f(g3d_ed_rot(q.xy, -a.z), q.z);
    let zx = g3d_ed_rot(q.zx, -a.y);
    q = vec3f(zx.y, q.y, zx.x);
    q = vec3f(q.x, g3d_ed_rot(q.yz, -a.x));
    return q;
}
fn g3d_ed_sphere(q: vec3f, d: vec3f, r: f32) -> f32 { return length(q) - max(d.x, 0.0); }
fn g3d_ed_box(q: vec3f, d: vec3f, r: f32) -> f32 {
    let b = max(0.5 * d, vec3f(0.0));
    let rr = clamp(r, 0.0, min(b.x, min(b.y, b.z)));
    let e = abs(q) - (b - rr);
    return length(max(e, vec3f(0.0))) + min(max(e.x, max(e.y, e.z)), 0.0) - rr;
}
fn g3d_ed_cylinder(q: vec3f, d: vec3f, r: f32) -> f32 {
    let rad = max(d.x, 0.0);
    let h = max(0.5 * d.y, 0.0);
    let rr = clamp(r, 0.0, min(rad, h));
    let e = vec2f(length(q.xy) - (rad - rr), abs(q.z) - (h - rr));
    return min(max(e.x, e.y), 0.0) + length(max(e, vec2f(0.0))) - rr;
}
fn g3d_ed_capsule(q: vec3f, d: vec3f, r: f32) -> f32 {
    let rad = max(d.x, 0.0);
    let h = max(0.5 * d.y - rad, 0.0);
    return length(vec3f(q.xy, q.z - clamp(q.z, -h, h))) - rad;
}
fn g3d_ed_torus(q: vec3f, d: vec3f, r: f32) -> f32 {
    return length(vec2f(length(q.xy) - d.x, q.z)) - max(d.y, 0.0);
}
fn g3d_ed_cone(p: vec3f, d: vec3f, r: f32) -> f32 {
    let r1 = max(d.x, 0.0);
    let r2 = max(d.y, 0.0);
    let h = max(0.5 * d.z, 1e-4);
    let q = vec2f(length(p.xy), p.z);
    let k1 = vec2f(r2, h);
    let k2 = vec2f(r2 - r1, 2.0 * h);
    let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
// a combined with b: op 0 = union, 1 = subtract (a minus b), 2 = intersect, 3 = none (a region
// marker); k = smoothing radius
fn g3d_ed_combine(a: f32, b: f32, k: f32, op: f32) -> f32 {
    if (op > 2.5) { return a; } // region only
    if (op > 1.5) {
        if (k <= 0.0) { return max(a, b); }
        let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) + k * h * (1.0 - h);
    }
    if (op > 0.5) {
        if (k <= 0.0) { return max(a, -b); }
        let h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0);
        return mix(a, -b, h) + k * h * (1.0 - h);
    }
    if (k <= 0.0) { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
";

/// Draws the selected primitive as a translucent ghost wherever it isn't part of the visible surface
/// (a subtracting one, one inside the script's shape, or one past an intersection), so it can be
/// seen and moved. `t_hit` is the distance to the shown surface (1e9 for none).
pub const OVERLAY: &str = "
fn g3d_overlay(ro: vec3f, rd: vec3f, t_hit: f32, col: vec3f, scale: f32) -> vec3f {
    if (g3d_ed[0].x < 0.0) { return col; }
    let eps = 1e-4 * scale;
    var t = 0.0;
    for (var i = 0; i < 160; i++) {
        let p = ro + rd * t;
        let d = g3d_ed_selected(p);
        if (d < eps * max(1.0, t / scale)) {
            if (t > t_hit - scale * 0.003) { return col; } // it is the surface there: tinted already
            let k = vec2f(1.0, -1.0) * eps * 2.0;
            let n = normalize(k.xyy * g3d_ed_selected(p + k.xyy) + k.yyx * g3d_ed_selected(p + k.yyx) +
                              k.yxy * g3d_ed_selected(p + k.yxy) + k.xxx * g3d_ed_selected(p + k.xxx));
            let facing = abs(dot(n, rd));
            let shade = 0.55 + 0.45 * abs(dot(n, normalize(vec3f(0.6, -0.4, 0.9))));
            let edge = pow(1.0 - facing, 2.0);
            return mix(col, vec3f(0.25, 0.65, 1.0) * shade, 0.3 + 0.45 * edge);
        }
        t += max(d, eps);
        if (t > min(t_hit, scale * 20.0)) { break; }
    }
    return col;
}
";

/// `g3d_overlay` for previews without editor primitives.
pub const NO_OVERLAY: &str =
    "fn g3d_overlay(ro: vec3f, rd: vec3f, t_hit: f32, col: vec3f, scale: f32) -> vec3f { return col; }\n";

/// Index of primitive `i`'s first vec4 in `g3d_ed` (element 0 holds the selection).
fn base(i: usize) -> usize {
    1 + 4 * i
}

/// Length of `g3d_ed` for `n` primitives.
pub fn data_len(n: usize) -> usize {
    1 + 4 * n
}

/// The values `wgsl` reads from `g3d_ed`: element 0 = (selected index or -1, 0, 0, 0), then four
/// vec4s per primitive (hidden ones too, so indices stay put): (pos, blend), (rot, round), (dims, op),
/// (load direction, force).
pub fn data(prims: &[Prim], selected: Option<usize>) -> Vec<[f32; 4]> {
    let sel = selected.filter(|&i| prims.get(i).is_some_and(|p| !p.hidden));
    let mut out = vec![[sel.map_or(-1.0, |i| i as f32), 0.0, 0.0, 0.0]];
    for p in prims {
        let [x, y, z] = p.pos;
        let [rx, ry, rz] = p.rot;
        let [a, b, c] = p.dims;
        let [lx, ly, lz] = p.load;
        out.extend([
            [x, y, z, p.blend],
            [rx, ry, rz, p.round],
            [a, b, c, p.op.code()],
            [lx, ly, lz, p.force],
        ]);
    }
    out
}

/// `g3d_ed` as a uniform buffer (the preview: moving a primitive is a buffer write, no recompile).
pub fn decl_uniform(len: usize) -> String {
    format!("@group(1) @binding(0) var<uniform> g3d_ed: array<vec4f, {len}>;\n")
}

/// `g3d_ed` as a constant (meshing, export, optimization: compiled once with the values baked in).
pub fn decl_const(data: &[[f32; 4]]) -> String {
    let items: Vec<String> = data
        .iter()
        .map(|v| format!("vec4f({:?}, {:?}, {:?}, {:?})", v[0], v[1], v[2], v[3]))
        .collect();
    format!("const g3d_ed = array<vec4f, {}>({});\n", data.len(), items.join(", "))
}

/// Which of its own functions the script defines that the generated code wraps (each renamed to
/// `g3d_script_<name>`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Script {
    pub map: bool,
    pub fixed: bool,
    pub load: bool,
    pub keep: bool,
}

/// Which optimization fields the visible primitives' roles add to: (field_fixed, field_load,
/// field_keep). Fixed and loaded regions are kept solid too.
pub fn role_fields(prims: &[Prim]) -> (bool, bool, bool) {
    let has = |r: Role| prims.iter().any(|p| !p.hidden && p.role == r);
    let (f, l, k) = (has(Role::Fixed), has(Role::Load), has(Role::Keep));
    (f, l, f || l || k)
}

/// Generated WGSL: the primitives as functions reading their numbers from `g3d_ed` (declared by the
/// caller with `decl_uniform` or `decl_const`), `map` = the script's map (if it has one) combined
/// with them, `g3d_ed_selected` (the distance to the selected primitive, for the preview's
/// highlight), and for primitives with optimization roles `field_fixed` / `field_load` /
/// `field_keep` (the script's own, if any, plus the primitives' regions; `reach` = default shell
/// thickness around boundary-condition primitives, `g3d_reach`).
pub fn wgsl(prims: &[Prim], script: Script, reach: f32) -> String {
    let script_map = script.map;
    let mut s = String::from(LIBRARY);
    let visible: Vec<_> = prims.iter().enumerate().filter(|(_, p)| !p.hidden).collect();
    for &(i, p) in &visible {
        let b = base(i);
        let _ = writeln!(s, "// {} ({})", p.name.replace(['\n', '\r'], " "), p.kind.name());
        let _ = writeln!(
            s,
            "fn g3d_e{i}(p: vec3f) -> f32 {{\n    \
             let q = g3d_ed_local(p, g3d_ed[{b}].xyz, g3d_ed[{}].xyz);\n    \
             return g3d_ed_{kind}(q, g3d_ed[{}].xyz, g3d_ed[{}].w);\n}}",
            b + 1,
            b + 2,
            b + 1,
            kind = p.kind.name()
        );
    }
    s += "fn map(p: vec3f) -> f32 {\n";
    s += if script_map {
        "    var d = g3d_script_map(p);\n"
    } else {
        "    var d = 1e10;\n"
    };
    for &(i, _) in &visible {
        let b = base(i);
        let _ = writeln!(
            s,
            "    d = g3d_ed_combine(d, g3d_e{i}(p), g3d_ed[{b}].w, g3d_ed[{}].w);",
            b + 2
        );
    }
    s += "    return d;\n}\n";
    s += "fn g3d_ed_selected(p: vec3f) -> f32 {\n    switch (i32(g3d_ed[0].x)) {\n";
    for &(i, _) in &visible {
        let _ = writeln!(s, "        case {i}: {{ return g3d_e{i}(p); }}");
    }
    s += "        default: { return 1e10; }\n    }\n}\n";

    // optimization roles
    let (gen_fixed, gen_load, gen_keep) = role_fields(prims);
    if gen_fixed || gen_load || gen_keep {
        let _ = writeln!(s, "override g3d_reach: f32 = {reach:?};");
    }
    let with = |r: Role| visible.iter().filter(move |(_, p)| p.role == r).map(|(i, _)| *i);
    let bc: Vec<usize> = visible
        .iter()
        .filter(|(_, p)| p.role.is_bc())
        .map(|(i, _)| *i)
        .collect();
    for &i in &bc {
        let _ = writeln!(s, "fn g3d_bc{i}(p: vec3f) -> f32 {{ return g3d_e{i}(p) - g3d_reach; }}");
    }
    if gen_fixed {
        s += "fn field_fixed(p: vec3f) -> f32 {\n";
        s += if script.fixed {
            "    var d = g3d_script_field_fixed(p);\n"
        } else {
            "    var d = 1e10;\n"
        };
        for i in with(Role::Fixed) {
            let _ = writeln!(s, "    d = min(d, g3d_bc{i}(p));");
        }
        s += "    return d;\n}\n";
    }
    if gen_load {
        s += "fn field_load(p: vec3f) -> vec3f {\n";
        s += if script.load {
            "    var v = g3d_script_field_load(p);\n"
        } else {
            "    var v = vec3f(0.0);\n"
        };
        for i in with(Role::Load) {
            let _ = writeln!(
                s,
                "    if (g3d_bc{i}(p) < 0.0) {{\n        let dir = g3d_ed[{}].xyz;\n        \
                 if (dot(dir, dir) > 0.0) {{ v += normalize(dir); }}\n    }}",
                base(i) + 3
            );
        }
        s += "    return v;\n}\n";
    }
    if gen_keep {
        s += "fn field_keep(p: vec3f) -> f32 {\n";
        s += if script.keep {
            "    var d = g3d_script_field_keep(p);\n"
        } else {
            "    var d = 1e10;\n"
        };
        for i in with(Role::Keep) {
            let _ = writeln!(s, "    d = min(d, g3d_e{i}(p));");
        }
        for &i in &bc {
            let _ = writeln!(s, "    d = min(d, g3d_bc{i}(p));");
        }
        s += "    return d;\n}\n";
    }
    s
}

/// The loads of the visible primitives with the load role.
pub fn load_specs(prims: &[Prim]) -> Vec<LoadSpec> {
    prims
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.role == Role::Load)
        .map(|(i, p)| LoadSpec {
            name: p.name.clone(),
            func: format!("g3d_bc{i}"),
            dir: Some(p.load),
            force: Some(p.newtons()),
            case: p.case.max(1),
            label: None,
        })
        .chain(rock_variants(prims))
        .collect()
}

/// The rocking variants: for each case with a rocking load, four more cases with every load of the
/// case, the rocking ones tilted by ± their share sideways along each axis across them.
fn rock_variants(prims: &[Prim]) -> Vec<LoadSpec> {
    let loads: Vec<(usize, &Prim)> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.role == Role::Load)
        .collect();
    let mut cases: Vec<u32> = loads
        .iter()
        .filter(|(_, p)| p.rock > 0.0)
        .map(|(_, p)| p.case.max(1))
        .collect();
    cases.sort();
    cases.dedup();
    let mut out = vec![];
    for c in cases {
        // sideways is across the case's first rocking load
        let lead = loads
            .iter()
            .find(|(_, p)| p.case.max(1) == c && p.rock > 0.0)
            .unwrap()
            .1;
        let mut k = 0;
        for axis in across(lead.load) {
            for sign in [1.0f32, -1.0] {
                k += 1;
                let label = format!(
                    "{c} rocked {}{}",
                    if sign > 0.0 { "+" } else { "-" },
                    ["X", "Y", "Z"][axis]
                );
                for &(i, p) in loads.iter().filter(|(_, p)| p.case.max(1) == c) {
                    let l = p.load;
                    let len = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
                    let (dir, force) = if p.rock > 0.0 && len > 0.0 {
                        let mut d = l.map(|v| v / len);
                        d[axis] += sign * p.rock;
                        let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        (d, p.newtons() * dl as f64)
                    } else {
                        (l, p.newtons())
                    };
                    out.push(LoadSpec {
                        name: p.name.clone(),
                        func: format!("g3d_bc{i}"),
                        dir: Some(dir),
                        force: Some(force),
                        case: rock_case(c, k),
                        label: Some(label.clone()),
                    });
                }
            }
        }
    }
    out
}

/// Byte offset of the name in the script's `fn <name>(` definition (ignoring `//` comments), if any.
pub fn find_fn(source: &str, name: &str) -> Result<Option<usize>> {
    let mut found = None;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let code = line.split("//").next().unwrap_or("");
        let mut rest = code;
        let mut base = 0;
        while let Some(i) = rest.find("fn") {
            let before_ok = i == 0 || !is_ident(rest.as_bytes()[i - 1]);
            let after = &rest[i + 2..];
            let trimmed = after.trim_start();
            let ws = after.len() - trimmed.len();
            if before_ok
                && ws > 0
                && let Some(tail) = trimmed.strip_prefix(name)
                && !tail.bytes().next().is_some_and(is_ident)
            {
                if found.is_some() {
                    bail!("more than one `fn {name}` in the shape");
                }
                found = Some(offset + base + i + 2 + ws);
            }
            base += i + 2;
            rest = &rest[i + 2..];
        }
        offset += line.len();
    }
    Ok(found)
}

/// Rename the script's `fn <name>` definition to `g3d_script_<name>` in place, keeping the line
/// structure so error locations still map back (calls elsewhere then reach the generated wrapper).
/// Returns whether there was one.
pub fn rename_script_fn(source: &mut String, name: &str) -> Result<bool> {
    let at = find_fn(source, name)?;
    if let Some(at) = at {
        source.replace_range(at..at + name.len(), &format!("g3d_script_{name}"));
    }
    Ok(at.is_some())
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The script's own optimization fields are kept and extended by the primitives' roles.
    #[test]
    fn roles_extend_script_fields() {
        use crate::shape::load_shape;
        let dir = std::env::temp_dir().join(format!("gen3d-roles-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wgsl");
        std::fs::write(
            &p,
            "fn map(p: vec3f) -> f32 { return length(p) - 10.0; }\n\
             fn field_fixed(p: vec3f) -> f32 { return p.z + 9.0; }\n\
             fn field_load(p: vec3f) -> vec3f { return select(vec3f(0.0), vec3f(1.0, 0.0, 0.0), p.x > 9.0); }\n",
        )
        .unwrap();
        // only a load primitive: the script's field_fixed stays as is, field_load is wrapped
        let push = Prim {
            role: Role::Load,
            case: 2,
            ..Prim::new(Kind::Sphere, "push".into(), [0.0, 0.0, 9.0], 4.0)
        };
        save(&sidecar(&p), std::slice::from_ref(&push)).unwrap();
        let s = load_shape(&p, &[]).unwrap();
        assert!(
            s.source
                .contains("fn field_fixed(p: vec3f) -> f32 { return p.z + 9.0; }")
        );
        assert!(
            s.source.contains("fn g3d_script_field_load(") && s.source.contains("var v = g3d_script_field_load(p);")
        );
        let specs = s.load_specs();
        let got: Vec<_> = specs.iter().map(|l| (l.func.as_str(), l.case, l.force)).collect();
        assert_eq!(got, [("g3d_script_field_load", 1, None), ("g3d_bc0", 2, Some(100.0))]);
        let names: Vec<_> = s.fields().unwrap().into_iter().map(|f| f.name).collect();
        assert_eq!(names, ["fixed", "load", "keep"]);
        let h = crate::viewer::preview_harness(&s.fields().unwrap(), true);
        s.compile_preview(&h).unwrap();
        s.compile(&crate::mesher::compute_harness("field_load", 3)).unwrap();
        // hidden: back to the script's fields only
        let mut hidden = push;
        hidden.hidden = true;
        save(&sidecar(&p), &[hidden]).unwrap();
        let s = load_shape(&p, &[]).unwrap();
        assert!(!s.bc && s.source.contains("fn field_load(") && s.load_specs().len() == 1);
        // old sidecars (no role fields) load with sensible defaults
        std::fs::write(sidecar(&p), r#"{"prims":[{"kind":"box","dims":[1,1,1]}]}"#).unwrap();
        let old = load(&sidecar(&p)).unwrap();
        assert_eq!(
            (old[0].role, old[0].op, old[0].case, old[0].force),
            (Role::Design, Op::Union, 1, 100.0)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rename_finds_the_definition_only() {
        let mut s = "// fn map(p) old\nfn helper() -> f32 { return map(vec3f(0)); }\nfn  map (p: vec3f) -> f32 { return 1.0; }\nfn mapper() {}\n".to_string();
        assert!(rename_script_fn(&mut s, "map").unwrap());
        assert!(s.contains("fn  g3d_script_map (p: vec3f)"), "{s}");
        assert!(s.contains("return map(vec3f(0))"));
        assert!(s.contains("// fn map(p) old"));
        assert!(s.contains("fn mapper()"));
        let mut none = "fn mapper() {}\n".to_string();
        assert!(!rename_script_fn(&mut none, "map").unwrap());
    }

    #[test]
    fn shapes_with_primitives_compile() {
        use crate::shape::load_shape;
        let dir = std::env::temp_dir().join(format!("gen3d-prims-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wgsl");
        std::fs::write(
            &p,
            "//! bounds -10 -10 0 10 10 20\noverride R: f32 = 5.0;\n\
             fn field_keep(p: vec3f) -> f32 { return map(p); }\n\
             fn map(p: vec3f) -> f32 { return length(p - vec3f(0, 0, 10)) - R; }\n",
        )
        .unwrap();
        // no sidecar: unchanged
        let s = load_shape(&p, &[]).unwrap();
        assert!(!s.editor && s.source.contains("fn map("));

        let mut prims: Vec<Prim> = Kind::ALL
            .iter()
            .enumerate()
            .map(|(i, &k)| Prim::new(k, format!("{} {i}", k.name()), [i as f32 * 4.0, 0.0, 5.0], 4.0))
            .collect();
        prims[1].round = 0.5;
        prims[2].blend = 2.0;
        prims[3].hidden = true;
        prims[5].pos = [40.0, 0.0, 5.0]; // outside the declared bounds
        save(&sidecar(&p), &prims).unwrap();
        assert_eq!(load(&sidecar(&p)).unwrap(), prims);

        let s = load_shape(&p, &[("R".into(), 6.0)]).unwrap();
        assert!(s.editor && s.script_map);
        assert!(s.files.contains(&sidecar(&p)));
        assert!(s.bounds.1[0] > 42.0 && s.base_bounds.1[0] == 10.0, "{:?}", s.bounds);
        let names: Vec<_> = s.params().unwrap().into_iter().map(|p| p.name).collect();
        assert_eq!(names, ["R"]);
        let d = data(&s.prims, Some(2));
        assert_eq!((d.len(), d[0][0], d[base(1) + 1][3]), (data_len(6), 2.0, 0.5));
        assert_eq!(
            data(&s.prims, Some(3))[0][0],
            -1.0,
            "hidden primitives can't be selected"
        );
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        s.compile(&crate::mesher::compute_harness("field_keep", 1)).unwrap();
        let preview = crate::viewer::preview_harness(&s.fields().unwrap(), true);
        s.compile(&preview).unwrap();
        let c = s.compile_preview(&preview).unwrap();
        assert!(c.source.contains("var<uniform> g3d_ed"));
        assert!(
            c.constants.is_empty() && c.live == ["R"],
            "R is an f32: live in the preview"
        );

        // errors in the script still point at the script's lines
        std::fs::write(&p, "fn map(p: vec3f) -> f32 {\n  return oops;\n}\n").unwrap();
        let err = load_shape(&p, &[])
            .unwrap()
            .compile(&crate::mesher::compute_harness("map", 1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("t.wgsl:2:"), "{err}");

        // no map in the script: the primitives are the shape, bounds fit them
        std::fs::write(&p, "//! voxel 0.5\n").unwrap();
        let s = load_shape(&p, &[]).unwrap();
        assert!(!s.script_map);
        assert!(s.bounds.0[0] < -1.9 && s.bounds.1[0] > 42.0, "{:?}", s.bounds);
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        save(&sidecar(&p), &[]).unwrap();
        load_shape(&p, &[])
            .unwrap()
            .compile_preview(&crate::viewer::preview_harness(&[], true))
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The CPU copies (used for picking) must agree with the generated WGSL, rotations included.
    /// Skipped without a GPU adapter.
    #[test]
    fn cpu_sdf_matches_gpu() {
        use crate::mesher::{Grid, sample_field};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-prims-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wgsl");
        std::fs::write(&p, "//! bounds -12 -12 -12 12 12 12\n").unwrap();
        for (k, kind) in Kind::ALL.into_iter().enumerate() {
            let mut prim = Prim::new(kind, "x".into(), [1.0, -2.0, 0.5], 9.0);
            prim.rot = [25.0 + k as f32, -40.0, 70.0];
            prim.round = 1.0;
            save(&sidecar(&p), std::slice::from_ref(&prim)).unwrap();
            let shape = crate::shape::load_shape(&p, &[]).unwrap();
            let grid = Grid {
                vol: vec![],
                dims: [9, 9, 9],
                origin: [-12.0; 3],
                step: 3.0,
            };
            let gpu_vals = sample_field(&shape, &grid, "map", 1, &gpu).unwrap();
            let mut i = 0;
            for x in 0..9 {
                for y in 0..9 {
                    for z in 0..9 {
                        let q = [x, y, z].map(|c| -12.0 + 3.0 * c as f32);
                        let cpu = prim.sdf(q);
                        assert!(
                            (cpu - gpu_vals[i]).abs() < 1e-3,
                            "{kind:?} at {q:?}: cpu {cpu} gpu {}",
                            gpu_vals[i]
                        );
                        i += 1;
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Add, smooth subtract, intersect and smooth add in list order, on the GPU, against the CPU fold.
    #[test]
    fn ops_match_gpu() {
        use crate::mesher::{Grid, sample_field};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-ops-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wgsl");
        std::fs::write(&p, "//! bounds -12 -12 -12 12 12 12\n").unwrap();
        let mut prims = vec![
            Prim::new(Kind::Box, "body".into(), [0.0; 3], 14.0),
            Prim::new(Kind::Sphere, "hole".into(), [5.0, 0.0, 3.0], 8.0),
            Prim::new(Kind::Cylinder, "trim".into(), [0.0, 0.0, 0.0], 18.0),
            Prim::new(Kind::Torus, "ring".into(), [0.0, 0.0, -6.0], 12.0),
        ];
        prims[1].op = Op::Subtract;
        prims[1].blend = 1.5;
        prims[2].op = Op::Intersect;
        prims[2].rot = [30.0, 0.0, 0.0];
        prims[3].blend = 1.0;
        save(&sidecar(&p), &prims).unwrap();
        let shape = crate::shape::load_shape(&p, &[]).unwrap();
        let grid = Grid {
            vol: vec![],
            dims: [9, 9, 9],
            origin: [-12.0; 3],
            step: 3.0,
        };
        let got = sample_field(&shape, &grid, "map", 1, &gpu).unwrap();
        let mut i = 0;
        let mut inside = 0;
        for x in 0..9 {
            for y in 0..9 {
                for z in 0..9 {
                    let q = [x, y, z].map(|c| -12.0 + 3.0 * c as f32);
                    let want = prims.iter().fold(1e10, |d, p| p.op.apply(d, p.sdf(q), p.blend));
                    assert!((want - got[i]).abs() < 1e-3, "{q:?}: cpu {want} gpu {}", got[i]);
                    inside += (want < 0.0) as usize;
                    i += 1;
                }
            }
        }
        assert!(inside > 20, "the test shape isn't empty");
        // the subtracted sphere's centre is carved out, outside the intersection is empty
        let at = |q: V3| prims.iter().fold(1e10, |d, p| p.op.apply(d, p.sdf(q), p.blend));
        assert!(at([5.0, 0.0, 3.0]) > 0.0 && at([-4.0, -4.0, -4.0]) < 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cpu_sdf_and_bounds() {
        let mut b = Prim::new(Kind::Box, "b".into(), [10.0, 0.0, 5.0], 10.0);
        assert!((b.sdf([10.0, 0.0, 5.0]) + 5.0).abs() < 1e-5);
        assert!((b.sdf([20.0, 0.0, 5.0]) - 5.0).abs() < 1e-5);
        // rotated 90° about Z: a 10 x 2 x 2 bar along X becomes one along Y
        b.dims = [10.0, 2.0, 2.0];
        b.rot = [0.0, 0.0, 90.0];
        assert!((b.sdf([10.0, 4.0, 5.0]) + 1.0).abs() < 1e-4);
        assert!(b.sdf([14.0, 0.0, 5.0]) > 2.0);
        // 90° about Y: the bar lies along Z
        b.rot = [0.0, 90.0, 0.0];
        assert!(b.sdf([10.0, 0.0, 9.0]) < 0.0);
        let c = Prim::new(Kind::Cone, "c".into(), [0.0; 3], 10.0);
        assert!(c.sdf([0.0, 0.0, -4.9]) < 0.0 && c.sdf([0.0, 0.0, 5.1]) > 0.0);
        let (lo, hi) = grow_bounds(Some(([0.0; 3], [1.0; 3])), &[c], 0.0).unwrap();
        assert!(lo[2] <= -5.0 && hi[0] >= 5.0 && hi[2] >= 5.0);
        // the box holds every inside point, for any rotation
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for kind in Kind::ALL {
            let mut p = Prim::new(kind, "x".into(), [1.0, 2.0, 3.0], 8.0);
            p.rot = [rnd() * 360.0, rnd() * 360.0, rnd() * 360.0];
            let (lo, hi) = p.aabb();
            for _ in 0..20000 {
                let q = [0, 1, 2].map(|k| p.pos[k] + (rnd() - 0.5) * 30.0);
                if p.sdf(q) < 0.0 {
                    assert!(
                        (0..3).all(|k| q[k] >= lo[k] && q[k] <= hi[k]),
                        "{kind:?} {q:?} {lo:?} {hi:?}"
                    );
                }
            }
        }
        let s = Prim::new(Kind::Sphere, "s".into(), [0.0, 0.0, 0.0], 10.0);
        assert_eq!(
            pick(std::slice::from_ref(&s), [0.0, -50.0, 0.0], [0.0, 1.0, 0.0], 200.0),
            Some(0)
        );
        assert_eq!(pick(&[s], [0.0, -50.0, 20.0], [0.0, 1.0, 0.0], 200.0), None);
    }
}
