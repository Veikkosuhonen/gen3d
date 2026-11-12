//! Primitives placed in the Design step's editor. They live in a sidecar file next to the shape
//! (`knob.wgsl` → `knob.prims.json`) and are unioned with the script's `map()`: `load_shape` renames
//! the script's `fn map` to `g3d_script_map` and appends a generated `map` (see [`wgsl`]), so every
//! consumer (preview, mesher, optimizer, VTI export, the CLI) sees the combined shape.
//!
//! Each primitive's numbers are WGSL `override` constants, like shape parameters: moving or resizing
//! one only rebuilds the pipeline, while adding, removing, hiding or changing its kind regenerates
//! the source.

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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prim {
    pub kind: Kind,
    pub name: String,
    /// Centre, mm.
    pub pos: [f32; 3],
    /// Rotation in degrees about X, then Y, then Z (world axes).
    pub rot: [f32; 3],
    /// Size; meaning per kind, see [`Kind::dims`].
    pub dims: [f32; 3],
    /// Edge rounding radius (boxes and cylinders).
    pub round: f32,
    /// Smooth-union radius with everything before it (0 = sharp).
    pub blend: f32,
    pub hidden: bool,
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

/// Grow `bounds` to contain the visible primitives (plus `margin`).
pub fn grow_bounds(bounds: Option<(V3, V3)>, prims: &[Prim], margin: f32) -> Option<(V3, V3)> {
    let mut out = bounds;
    for p in prims.iter().filter(|p| !p.hidden) {
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
fn g3d_ed_union(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
";

/// The override constants of primitive `i`, in the order `wgsl` declares them.
fn values(p: &Prim) -> [(&'static str, f32); 11] {
    [
        ("px", p.pos[0]),
        ("py", p.pos[1]),
        ("pz", p.pos[2]),
        ("rx", p.rot[0]),
        ("ry", p.rot[1]),
        ("rz", p.rot[2]),
        ("a", p.dims[0]),
        ("b", p.dims[1]),
        ("c", p.dims[2]),
        ("round", p.round),
        ("blend", p.blend),
    ]
}

/// Name of the override that selects which primitive the preview highlights (-1 = none).
pub const SELECT: &str = "g3d_ed_sel";

/// Values for every override `wgsl` declares (hidden primitives have none).
pub fn constants(prims: &[Prim]) -> Vec<(String, f64)> {
    let mut out = vec![];
    for (i, p) in prims.iter().enumerate().filter(|(_, p)| !p.hidden) {
        for (k, v) in values(p) {
            out.push((format!("g3d_e{i}_{k}"), v as f64));
        }
    }
    out
}

/// Generated WGSL: the primitives as functions of `override` constants (defaults = current values),
/// `map` = the script's map (if `script_map`) unioned with them, and `g3d_ed_selected`, the distance
/// to the primitive picked by `g3d_ed_sel`, for the preview's highlight.
pub fn wgsl(prims: &[Prim], script_map: bool) -> String {
    let mut s = String::from(LIBRARY);
    let _ = writeln!(s, "override {SELECT}: i32 = -1;");
    let visible: Vec<_> = prims.iter().enumerate().filter(|(_, p)| !p.hidden).collect();
    for &(i, p) in &visible {
        let _ = writeln!(s, "// {} ({})", p.name.replace(['\n', '\r'], " "), p.kind.name());
        for (k, v) in values(p) {
            let _ = writeln!(s, "override g3d_e{i}_{k}: f32 = {v:?};");
        }
        let _ = writeln!(
            s,
            "fn g3d_e{i}(p: vec3f) -> f32 {{\n    \
             let q = g3d_ed_local(p, vec3f(g3d_e{i}_px, g3d_e{i}_py, g3d_e{i}_pz), vec3f(g3d_e{i}_rx, g3d_e{i}_ry, g3d_e{i}_rz));\n    \
             return g3d_ed_{kind}(q, vec3f(g3d_e{i}_a, g3d_e{i}_b, g3d_e{i}_c), g3d_e{i}_round);\n}}",
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
        let _ = writeln!(s, "    d = g3d_ed_union(d, g3d_e{i}(p), g3d_e{i}_blend);");
    }
    s += "    return d;\n}\n";
    s += &format!("fn g3d_ed_selected(p: vec3f) -> f32 {{\n    switch ({SELECT}) {{\n");
    for &(i, _) in &visible {
        let _ = writeln!(s, "        case {i}: {{ return g3d_e{i}(p); }}");
    }
    s += "        default: { return 1e10; }\n    }\n}\n";
    s
}

/// Find the script's `fn map(` definition (ignoring `//` comments) and rename it in place, keeping
/// the line structure so error locations still map back. Returns whether there was one.
pub fn rename_script_map(source: &mut String) -> Result<bool> {
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
                && let Some(tail) = trimmed.strip_prefix("map")
                && !tail.bytes().next().is_some_and(is_ident)
            {
                if found.is_some() {
                    bail!("more than one `fn map` in the shape");
                }
                found = Some(offset + base + i + 2 + ws);
            }
            base += i + 2;
            rest = &rest[i + 2..];
        }
        offset += line.len();
    }
    if let Some(at) = found {
        source.replace_range(at..at + 3, "g3d_script_map");
    }
    Ok(found.is_some())
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_finds_the_definition_only() {
        let mut s = "// fn map(p) old\nfn helper() -> f32 { return map(vec3f(0)); }\nfn  map (p: vec3f) -> f32 { return 1.0; }\nfn mapper() {}\n".to_string();
        assert!(rename_script_map(&mut s).unwrap());
        assert!(s.contains("fn  g3d_script_map (p: vec3f)"), "{s}");
        assert!(s.contains("return map(vec3f(0))"));
        assert!(s.contains("// fn map(p) old"));
        assert!(s.contains("fn mapper()"));
        let mut none = "fn mapper() {}\n".to_string();
        assert!(!rename_script_map(&mut none).unwrap());
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
        let c = s.preview_constants(Some(2));
        assert!(c.contains(&("R".into(), 6.0)) && c.contains(&(SELECT.into(), 2.0)));
        assert!(
            !c.iter().any(|(k, _)| k.starts_with("g3d_e3_")),
            "hidden primitive has no constants"
        );
        assert_eq!(s.preview_constants(Some(3)).last().unwrap().1, -1.0);
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        s.compile(&crate::mesher::compute_harness("field_keep", 1)).unwrap();
        s.compile(&crate::viewer::preview_harness(&s.fields().unwrap(), true))
            .unwrap();

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
            .compile(&crate::viewer::preview_harness(&[], true))
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
