//! Load WGSL shape files: resolve `#include`, parse `//!` directives, map naga errors back to files.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

use crate::prims::{self, Prim};

/// Built-in library files, available to `#include <name>` without being on disk.
pub const LIBRARY: &[(&str, &str)] = &[("sdf.wgsl", include_str!("wgsl/sdf.wgsl"))];

/// Where a line of the expanded source came from.
#[derive(Clone, Debug)]
struct Origin {
    file: usize,
    line: usize,
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub path: PathBuf,
    /// Fully expanded WGSL (shape + includes), without any gen3d harness code.
    pub source: String,
    /// Files that make up `source`; on-disk ones are watched for hot reload.
    pub files: Vec<PathBuf>,
    origins: Vec<Origin>,
    pub bounds: ([f32; 3], [f32; 3]),
    pub voxel: f32,
    pub directives: BTreeMap<String, String>,
    /// `-D NAME=VALUE` overrides, applied as WGSL pipeline-overridable constants.
    pub defines: Vec<(String, f64)>,
    /// Primitives from the editor's sidecar file (see `prims.rs`), unioned with the script's map.
    pub prims: Vec<Prim>,
    /// Whether the editor's code is part of `source` (the sidecar file exists).
    pub editor: bool,
    /// Whether the script itself defines `fn map` (without one, the primitives are the shape).
    pub script_map: bool,
    /// The bounds the file declares (`bounds` grows from these to fit the primitives).
    pub base_bounds: ([f32; 3], [f32; 3]),
}

impl Shape {
    pub fn size(&self) -> [f32; 3] {
        let (lo, hi) = self.bounds;
        [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    }

    pub fn step(&self) -> f32 {
        self.directives.get("step").and_then(|s| s.parse().ok()).unwrap_or(0.9)
    }

    pub fn decimate(&self) -> Option<f32> {
        self.directives
            .get("decimate")
            .and_then(|s| s.parse().ok())
            .filter(|&f: &f32| f > 0.0)
    }

    /// A shape from generated WGSL (e.g. a .vti volume), not loaded from a .wgsl file. `path` is
    /// watched for hot reload.
    pub fn generated(path: &Path, source: &str, bounds: ([f32; 3], [f32; 3]), voxel: f32) -> Shape {
        Shape {
            source: source.to_string(),
            bounds,
            base_bounds: bounds,
            voxel,
            ..Shape::placeholder(path)
        }
    }

    /// A placeholder used by the viewer when the very first load fails.
    pub fn placeholder(path: &Path) -> Shape {
        Shape {
            path: path.to_path_buf(),
            source: String::new(),
            files: vec![path.to_path_buf()],
            origins: vec![],
            bounds: ([-50.0, -50.0, 0.0], [50.0, 50.0, 100.0]),
            voxel: 0.25,
            directives: BTreeMap::new(),
            defines: vec![],
            prims: vec![],
            editor: false,
            script_map: true,
            base_bounds: ([-50.0, -50.0, 0.0], [50.0, 50.0, 100.0]),
        }
    }

    /// The editor's sidecar file for this shape (it may not exist yet).
    pub fn prims_path(&self) -> PathBuf {
        prims::sidecar(&self.path)
    }

    /// Replace the primitives' values without regenerating the source: valid while the kinds and
    /// visibility match what `source` was generated for (only the override constants change).
    pub fn set_prims(&mut self, prims: Vec<Prim>) {
        self.prims = prims;
        self.update_bounds();
    }

    fn update_bounds(&mut self) {
        let declared = self.script_map || self.directives.contains_key("bounds");
        let base = declared.then_some(self.base_bounds);
        let margin = 2.0 * self.voxel;
        let prims = if self.editor { &self.prims[..] } else { &[] };
        self.bounds = prims::grow_bounds(base, prims, margin).unwrap_or(self.base_bounds);
    }

    /// The primitives' values for the preview's `g3d_ed` uniform buffer (None without the editor).
    pub fn editor_data(&self, selected: Option<usize>) -> Option<Vec<[f32; 4]>> {
        self.editor.then(|| prims::data(&self.prims, selected))
    }

    /// Append gen3d's harness (entry points, uniforms) to the shape and validate it with naga.
    /// Errors point at `file:line:col` in the user's files. The editor's primitives are baked in.
    pub fn compile(&self, harness: &str) -> Result<Compiled> {
        let decl = match self.editor {
            true => prims::decl_const(&prims::data(&self.prims, None)),
            false => String::new(),
        };
        self.compile_with(harness, &decl)
    }

    /// Like `compile`, but the primitives' values come from a uniform buffer at group 1, binding 0
    /// (`editor_data`), so the preview can move them without compiling.
    pub fn compile_preview(&self, harness: &str) -> Result<Compiled> {
        let decl = match self.editor {
            true => prims::decl_uniform(1 + 3 * self.prims.len()),
            false => String::new(),
        };
        self.compile_with(harness, &decl)
    }

    /// The source with `g3d_ed` declared (as a constant), for parsing it on its own.
    fn with_editor_decl(&self) -> String {
        match self.editor {
            true => format!(
                "{}\n{}",
                self.source,
                prims::decl_const(&prims::data(&self.prims, None))
            ),
            false => self.source.clone(),
        }
    }

    fn compile_with(&self, harness: &str, decl: &str) -> Result<Compiled> {
        let source = format!("{}\n{}{}", self.source, decl, harness);
        let module = self.parse(&source)?;
        let mut validator =
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        validator.validate(&module).map_err(|e| {
            let labels: Vec<_> = e.spans().map(|(span, label)| (*span, label.clone())).collect();
            let mut msg = e.as_inner().to_string();
            let mut src: &dyn std::error::Error = e.as_inner();
            while let Some(next) = src.source() {
                let _ = write!(msg, ": {next}");
                src = next;
            }
            anyhow!(self.format_error(&source, &msg, &labels, &[]))
        })?;
        if !module.functions.iter().any(|(_, f)| f.name.as_deref() == Some("map")) {
            bail!("{}: no `fn map(p: vec3f) -> f32` found", self.path.display());
        }
        let declared: Vec<String> = module.overrides.iter().filter_map(|(_, o)| o.name.clone()).collect();
        for (name, _) in &self.defines {
            if !declared.contains(name) {
                let hint = if declared.is_empty() {
                    "the shape declares no parameters".to_string()
                } else {
                    format!("declared parameters: {}", declared.join(", "))
                };
                bail!(
                    "-D {name}: no `override {name}: f32 = ...;` in {} ({hint})",
                    self.path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
        }
        Ok(Compiled {
            source,
            constants: self.defines.clone(),
        })
    }

    fn parse(&self, source: &str) -> Result<naga::Module> {
        naga::front::wgsl::parse_str(source).map_err(|e| {
            let labels: Vec<_> = e.labels().map(|(span, label)| (span, label.to_string())).collect();
            let notes: Vec<_> = e.notes().map(str::to_string).collect();
            anyhow!(self.format_error(source, e.message(), &labels, &notes))
        })
    }

    /// The shape's extra fields: every `fn field_<name>(p: vec3f) -> f32` (or `-> vec3f`), in
    /// declaration order. They are sampled alongside the SDF by `export --format vti`.
    pub fn fields(&self) -> Result<Vec<Field>> {
        let module = self.parse(&self.with_editor_decl())?;
        let mut out = vec![];
        for (_, f) in module.functions.iter() {
            let Some(func) = f.name.as_deref() else { continue };
            let Some(name) = func.strip_prefix("field_") else {
                continue;
            };
            let kind = |ty| match module.types[ty].inner {
                naga::TypeInner::Scalar(naga::Scalar::F32) => Some(1),
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar::F32,
                } => Some(3),
                _ => None,
            };
            let arg_ok = f.arguments.len() == 1 && kind(f.arguments[0].ty) == Some(3);
            let comps = f.result.as_ref().and_then(|r| kind(r.ty));
            let (true, Some(comps)) = (arg_ok, comps) else {
                bail!("fn {func}: fields must be `fn {func}(p: vec3f) -> f32` or `-> vec3f`");
            };
            if name.is_empty() || name == "sdf" {
                bail!("fn {func}: field name `{name}` is reserved; pick another `field_<name>`");
            }
            out.push(Field {
                name: name.to_string(),
                func: func.to_string(),
                comps,
            });
        }
        Ok(out)
    }

    /// The shape's parameters: every `override NAME: T = default;`, in declaration order. A trailing
    /// `// comment` on the declaration becomes its note, and `//! range NAME min max` sets the range
    /// the viewer's slider covers.
    pub fn params(&self) -> Result<Vec<Param>> {
        let module = self.parse(&self.with_editor_decl())?;
        let mut out = vec![];
        for (_, o) in module.overrides.iter() {
            let Some(name) = o.name.clone() else { continue };
            if name.starts_with("g3d_") {
                continue; // gen3d's own (the editor's primitives)
            }
            let kind = match module.types[o.ty].inner {
                naga::TypeInner::Scalar(s) => match s {
                    naga::Scalar::BOOL => ParamKind::Bool,
                    naga::Scalar::I32 => ParamKind::Int,
                    naga::Scalar::U32 => ParamKind::Uint,
                    _ => ParamKind::Float,
                },
                _ => ParamKind::Float,
            };
            let default = o.init.and_then(|h| const_value(&module, h)).unwrap_or(0.0);
            let decl = format!("override {name}");
            let note = self
                .source
                .lines()
                .find(|l| {
                    l.trim_start()
                        .strip_prefix(&decl)
                        .is_some_and(|r| r.starts_with([':', ' ', '=', ';']))
                })
                .and_then(|l| l.split_once("//"))
                .map(|(_, c)| c.trim().to_string())
                .unwrap_or_default();
            let range = self.source.lines().find_map(|l| {
                let mut it = l.trim_start().strip_prefix("//!")?.split_whitespace();
                (it.next()? == "range" && it.next()? == name).then_some(())?;
                let lo: f64 = it.next()?.parse().ok()?;
                let hi: f64 = it.next()?.parse().ok()?;
                (hi > lo).then_some((lo, hi))
            });
            out.push(Param {
                name,
                kind,
                default,
                note,
                range,
            });
        }
        Ok(out)
    }

    fn locate(&self, source: &str, offset: usize) -> (String, usize, usize, String) {
        let offset = offset.min(source.len());
        let line_idx = source[..offset].matches('\n').count();
        let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
        let col = source[line_start..offset].chars().count() + 1;
        let text = source[line_start..].lines().next().unwrap_or("").to_string();
        match self.origins.get(line_idx) {
            Some(o) => (display_path(&self.files[o.file]), o.line, col, text),
            None => (
                "<gen3d harness>".to_string(),
                line_idx + 1 - self.origins.len(),
                col,
                text,
            ),
        }
    }

    fn format_error(&self, source: &str, msg: &str, labels: &[(naga::Span, String)], notes: &[String]) -> String {
        let mut out = String::new();
        let mut first = true;
        for (span, label) in labels {
            let Some(range) = span.to_range() else { continue };
            let (file, line, col, text) = self.locate(source, range.start);
            if first {
                let _ = writeln!(out, "{file}:{line}:{col}: {msg}");
                first = false;
            } else {
                let _ = writeln!(out, "{file}:{line}:{col}:");
            }
            let width = source[range.clone()]
                .lines()
                .next()
                .map_or(1, |s| s.chars().count().max(1));
            let _ = writeln!(out, "    {text}");
            let _ = writeln!(out, "    {}{} {label}", " ".repeat(col - 1), "^".repeat(width));
        }
        if first {
            let _ = writeln!(out, "{}: {msg}", display_path(&self.path));
        }
        for n in notes {
            let _ = writeln!(out, "  = note: {n}");
        }
        out.trim_end().to_string()
    }
}

/// A user-defined field function, exported as an extra point-data array named `name`.
#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub func: String,
    /// 1 for `-> f32`, 3 for `-> vec3f`
    pub comps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamKind {
    Float,
    Int,
    Uint,
    Bool,
}

/// A shape parameter: a WGSL `override` constant, settable with `-D NAME=VALUE`.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub default: f64,
    /// Trailing comment on the declaration, if any.
    pub note: String,
    /// From `//! range NAME min max`.
    pub range: Option<(f64, f64)>,
}

impl Param {
    /// Slider range: the declared one, or a guess around the default.
    pub fn slider_range(&self) -> (f64, f64) {
        if let Some(r) = self.range {
            return r;
        }
        let d = self.default;
        match self.kind {
            ParamKind::Bool => (0.0, 1.0),
            _ if d == 0.0 => (-10.0, 10.0),
            ParamKind::Float if d > 0.0 => (0.0, 2.0 * d),
            ParamKind::Float => (2.0 * d, -2.0 * d),
            _ if d > 0.0 => (0.0, (2.0 * d).max(10.0)),
            _ => (2.0 * d, -2.0 * d),
        }
    }
}

/// Evaluate a simple constant initializer (a literal, possibly negated).
fn const_value(module: &naga::Module, h: naga::Handle<naga::Expression>) -> Option<f64> {
    use naga::{Expression, Literal};
    Some(match module.global_expressions[h] {
        Expression::Literal(l) => match l {
            Literal::F64(v) | Literal::AbstractFloat(v) => v,
            Literal::F32(v) => v as f64,
            Literal::F16(v) => f32::from(v) as f64,
            Literal::I32(v) => v as f64,
            Literal::U32(v) => v as f64,
            Literal::I16(v) => v as f64,
            Literal::U16(v) => v as f64,
            Literal::I64(v) | Literal::AbstractInt(v) => v as f64,
            Literal::U64(v) => v as f64,
            Literal::Bool(b) => b as u8 as f64,
        },
        Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr,
        } => -const_value(module, expr)?,
        Expression::Constant(c) => const_value(module, module.constants[c].init)?,
        _ => return None,
    })
}

/// A validated module ready for `create_shader_module`, plus the override constants to apply.
#[derive(Debug)]
pub struct Compiled {
    pub source: String,
    pub constants: Vec<(String, f64)>,
}

impl Compiled {
    pub fn constants(&self) -> Vec<(&str, f64)> {
        self.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    }
}

fn display_path(p: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir()
        && let Ok(rel) = p.strip_prefix(&cwd)
    {
        return rel.display().to_string();
    }
    p.display().to_string()
}

fn library_path(name: &str) -> PathBuf {
    PathBuf::from(format!("<gen3d>/{name}"))
}

fn read_file(path: &Path) -> Result<String> {
    if let Ok(rest) = path.strip_prefix("<gen3d>")
        && let Some((_, text)) = LIBRARY.iter().find(|(n, _)| Path::new(n) == rest)
    {
        return Ok(text.to_string());
    }
    std::fs::read_to_string(path).map_err(|e| anyhow!("cannot read {}: {e}", path.display()))
}

fn parse_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim();
    let (open, close) = match rest.chars().next()? {
        '<' => ('<', '>'),
        '"' => ('"', '"'),
        _ => return None,
    };
    let inner = rest.strip_prefix(open)?;
    inner.find(close).map(|end| &inner[..end])
}

struct Expander {
    files: Vec<PathBuf>,
    origins: Vec<Origin>,
    out: String,
    seen: HashSet<PathBuf>,
}

impl Expander {
    fn expand(&mut self, path: &Path, stack: &mut Vec<PathBuf>) -> Result<()> {
        let path = if path.starts_with("<gen3d>") {
            path.to_path_buf()
        } else {
            path.canonicalize()
                .map_err(|e| anyhow!("cannot read {}: {e}", path.display()))?
        };
        if stack.contains(&path) {
            bail!("circular #include: {}", path.display());
        }
        if !self.seen.insert(path.clone()) {
            return Ok(()); // include guard: each file only once
        }
        let text = read_file(&path)?;
        let idx = self.files.len();
        self.files.push(path.clone());
        stack.push(path.clone());
        for (i, line) in text.lines().enumerate() {
            if let Some(name) = parse_include(line) {
                let local = path.parent().map(|d| d.join(name)).filter(|p| p.exists());
                let target = match local {
                    Some(p) => p,
                    None if LIBRARY.iter().any(|(n, _)| *n == name) => library_path(name),
                    None => bail!("{}:{}: include not found: {name}", display_path(&path), i + 1),
                };
                self.expand(&target, stack)?;
                // keep a blank line so the include line itself still maps somewhere sensible
                self.push_line("", idx, i + 1);
            } else {
                self.push_line(line, idx, i + 1);
            }
        }
        stack.pop();
        Ok(())
    }

    fn push_line(&mut self, line: &str, file: usize, lineno: usize) {
        self.out.push_str(line);
        self.out.push('\n');
        self.origins.push(Origin { file, line: lineno });
    }
}

/// Parse `-D NAME=VALUE` items. Values are numbers (or true/false) for WGSL `override` constants.
pub fn parse_defines(items: &[String]) -> Result<Vec<(String, f64)>> {
    items
        .iter()
        .map(|it| {
            let (k, v) = it.split_once('=').unwrap_or((it, "1"));
            let v = v.trim();
            let val = match v {
                "true" => 1.0,
                "false" => 0.0,
                _ => v
                    .parse::<f64>()
                    .map_err(|_| anyhow!("-D {it}: value must be a number"))?,
            };
            Ok((k.trim().to_string(), val))
        })
        .collect()
}

pub fn load_shape(path: &Path, defines: &[(String, f64)]) -> Result<Shape> {
    let mut ex = Expander {
        files: vec![],
        origins: vec![],
        out: String::new(),
        seen: HashSet::new(),
    };
    ex.expand(path, &mut vec![])?;
    let path = ex.files[0].clone();
    let mut shape = Shape {
        path: path.clone(),
        source: ex.out,
        files: ex.files,
        origins: ex.origins,
        defines: defines.to_vec(),
        ..Shape::placeholder(&path)
    };

    for line in read_file(&path)?.lines() {
        let Some(rest) = line.trim_start().strip_prefix("//!") else {
            continue;
        };
        let rest = rest.trim();
        let key_end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let key = rest[..key_end].to_lowercase();
        if key.is_empty() {
            continue;
        }
        let mut val = rest[key_end..].trim_start().trim_start_matches(':').trim();
        // drop trailing `// note` or `(note)`
        for sep in [" //", "\t//", " (", "\t("] {
            if let Some(i) = val.find(sep) {
                val = val[..i].trim();
            }
        }
        shape.directives.insert(key.clone(), val.to_string());
        let bad = |e: &str| anyhow!("{}: bad //! {key}: {e}", display_path(&path));
        match key.as_str() {
            "bounds" => {
                let v: Vec<f32> = val
                    .replace(',', " ")
                    .split_whitespace()
                    .map(|x| x.parse::<f32>().map_err(|e| bad(&e.to_string())))
                    .collect::<Result<_>>()?;
                if v.len() != 6 {
                    return Err(bad("expected 6 numbers: xmin ymin zmin xmax ymax zmax"));
                }
                shape.bounds = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
            }
            "voxel" => {
                shape.voxel = val
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .parse()
                    .map_err(|_| bad("expected a number"))?;
            }
            _ => {}
        }
    }
    shape.base_bounds = shape.bounds;

    // the editor's primitives: rename the script's map and append the generated union
    let sidecar = prims::sidecar(&path);
    if sidecar.exists() {
        shape.prims = prims::load(&sidecar)?;
        shape.editor = true;
        shape.script_map = prims::rename_script_map(&mut shape.source)?;
        shape.files.push(sidecar);
        let idx = shape.files.len();
        shape.files.push(library_path("editor.wgsl"));
        for (i, line) in prims::wgsl(&shape.prims, shape.script_map).lines().enumerate() {
            shape.source.push_str(line);
            shape.source.push('\n');
            shape.origins.push(Origin { file: idx, line: i + 1 });
        }
        shape.update_bounds();
    }
    Ok(shape)
}

const TEMPLATE: &str = "// {name}.wgsl — gen3d shape. Units: mm, +Z up, build plate at z = 0.
// Export region (xmin ymin zmin  xmax ymax zmax) and mesh resolution in mm:
//! bounds -30 -30 0  30 30 50
//! voxel 0.2
//! range R 4 20
#include <sdf.wgsl>

override R: f32 = 12.0;  // ball radius (mm)

fn map(p: vec3f) -> f32 {
    let body = sdRoundBox(p - vec3f(0, 0, 15), vec3f(20, 20, 15), 3.0);
    let hole = sdCylinder(rotX(p - vec3f(0, 0, 13), deg(90.0)), 7.0, 25.0);  // through-hole along Y
    let ball = sdSphere(p - vec3f(0, 0, 30), R);
    return opSmoothUnion(opSmoothSubtract(body, hole, 2.0), ball, 4.0);
}
";

/// Write a new shape from the template. `name` is a file name (`.wgsl` added if missing) in `dir`.
pub fn create_from_template(dir: &Path, name: &str) -> Result<PathBuf> {
    let name = name.trim();
    if name.is_empty() || name.contains(['/', '\\']) {
        bail!("give the new shape a file name (no folders)");
    }
    let mut path = dir.join(name);
    if path.extension().is_none_or(|e| e != "wgsl") {
        path = dir.join(format!("{name}.wgsl"));
    }
    if path.exists() {
        bail!("{} already exists", path.display());
    }
    std::fs::create_dir_all(dir)?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    std::fs::write(&path, TEMPLATE.replace("{name}", &stem))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_compiles() {
        let dir = std::env::temp_dir().join(format!("gen3d-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wgsl");
        std::fs::write(&p, "//! bounds -1 -1 0 1 1 2\n#include <sdf.wgsl>\noverride R: f32 = 1.0;\nfn map(p: vec3f) -> f32 { return sdSphere(p, R) + fbm(p) * 0.0; }\n").unwrap();
        let s = load_shape(&p, &[("R".into(), 2.0)]).unwrap();
        assert_eq!(s.bounds, ([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]));
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        s.compile(&crate::viewer::preview_harness(&[], false)).unwrap();
        let ps = s.params().unwrap();
        assert_eq!(
            (ps[0].name.as_str(), ps[0].default, ps[0].kind),
            ("R", 1.0, ParamKind::Float)
        );
        let bad = load_shape(&p, &[("NOPE".into(), 2.0)]).unwrap();
        assert!(bad.compile(&crate::mesher::compute_harness("map", 1)).is_err());

        std::fs::write(
            &p,
            "#include <sdf.wgsl>\n\nfn map(p: vec3f) -> f32 {\n  return sdSphere(p, 1.0) + oops;\n}\n",
        )
        .unwrap();
        let err = load_shape(&p, &[])
            .unwrap()
            .compile(&crate::mesher::compute_harness("map", 1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("t.wgsl:4:"), "{err}");

        std::fs::write(
            &p,
            "fn map(p: vec3f) -> f32 { return length(p) - 1.0; }\n\
             fn field_load(p: vec3f) -> vec3f { return p; }\n\
             fn field_keep(p: vec3f) -> f32 { return p.x; }\n\
             fn helper(p: vec3f) -> f32 { return 0.0; }\n",
        )
        .unwrap();
        let s = load_shape(&p, &[]).unwrap();
        let f = s.fields().unwrap();
        let got: Vec<_> = f.iter().map(|f| (f.name.as_str(), f.comps)).collect();
        assert_eq!(got, [("load", 3), ("keep", 1)]);
        for f in &f {
            s.compile(&crate::mesher::compute_harness(&f.func, f.comps)).unwrap();
        }
        s.compile(&crate::viewer::preview_harness(&f, false)).unwrap();
        std::fs::write(
            &p,
            "//! range W 0.5 4\n\
             override W: f32 = -1.5;   // wall, mm\n\
             override N: i32 = 6;\n\
             override HOLLOW: bool = true;\n\
             fn map(p: vec3f) -> f32 { return length(p) - W + f32(N) + select(0.0, 1.0, HOLLOW); }\n",
        )
        .unwrap();
        let ps = load_shape(&p, &[]).unwrap().params().unwrap();
        let got: Vec<_> = ps.iter().map(|p| (p.name.as_str(), p.kind, p.default)).collect();
        assert_eq!(
            got,
            [
                ("W", ParamKind::Float, -1.5),
                ("N", ParamKind::Int, 6.0),
                ("HOLLOW", ParamKind::Bool, 1.0)
            ]
        );
        assert_eq!((ps[0].note.as_str(), ps[0].range), ("wall, mm", Some((0.5, 4.0))));
        assert_eq!(ps[1].slider_range(), (0.0, 12.0));
        std::fs::write(
            &p,
            "fn map(p: vec3f) -> f32 { return 0.0; }\nfn field_x(p: vec3f) -> i32 { return 0; }\n",
        )
        .unwrap();
        assert!(load_shape(&p, &[]).unwrap().fields().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
