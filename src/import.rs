//! `//! import NAME FILE.vti [options]`: a .vti grid's solid (e.g. an optimization result) as an SDF
//! function `fn NAME(p: vec3f) -> f32` that the shape can combine with exact primitives.
//!
//! The grid is turned into a signed distance field on the CPU when the shape loads (`Volume::sdf`:
//! for an optimizer's `density`, the same faired surface Print meshes), and all imported grids go to
//! the GPU in one storage buffer, `g3d_imp` at group 0, binding `BINDING`, which every harness that
//! compiles a shape binds (`buffer`). The functions sample it trilinearly; outside a grid they return
//! a lower bound on the distance. The fields are cached by file and options, so a hot reload of the
//! script doesn't redo the fairing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::{Result, anyhow, bail};
use wgpu::util::DeviceExt;

use crate::volume::{MeshOpts, SdfGrid, Volume, VolumeOpts};

/// The storage buffer's binding in group 0 (the preview uses 0-5, the sampler 0-1).
pub const BINDING: u32 = 8;

/// An imported grid, as the shape refers to it.
#[derive(Clone, Debug)]
pub struct Import {
    /// the WGSL function's name
    pub name: String,
    pub path: PathBuf,
    pub grid: Arc<SdfGrid>,
    /// the print orientation recorded in the file (an optimizer result's `print_up`)
    pub up: Option<crate::mesh::Up>,
}

/// Options after the file name: `array=NAME`, `iso=X`, `below` / `above`, `fair` / `no-fair`,
/// `grow=MM`.
#[derive(Clone, Debug, Default, PartialEq)]
struct Opts {
    vol: VolumeOptsKey,
    fair: Option<bool>,
    grow: f32,
}

/// The `VolumeOpts` that matter here, comparable for the cache.
#[derive(Clone, Debug, Default, PartialEq)]
struct VolumeOptsKey {
    array: Option<String>,
    iso: Option<f32>,
    below: Option<bool>,
}

/// Parse the directive's value: `NAME FILE [options]`. `dir` is the shape's directory; the file is
/// looked up there first, then relative to the working directory (where `out/` usually is).
pub fn parse(val: &str, dir: &Path) -> Result<Import> {
    let mut it = val.split_whitespace();
    let (Some(name), Some(file)) = (it.next(), it.next()) else {
        bail!("expected `NAME FILE.vti [array=NAME] [iso=X] [below|above] [fair|no-fair] [grow=MM]`");
    };
    let ident = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ident || name == "map" || name.starts_with("g3d_") || name.starts_with("field_") {
        bail!("`{name}` can't be a function name (use letters, digits and _; not map, field_*, g3d_*)");
    }
    let mut o = Opts::default();
    for opt in it {
        let (k, v) = opt.split_once('=').unwrap_or((opt, ""));
        let num = |v: &str| {
            v.parse::<f32>()
                .map_err(|_| anyhow!("{k}: expected a number, got `{v}`"))
        };
        match k {
            "array" if !v.is_empty() => o.vol.array = Some(v.to_string()),
            "iso" => o.vol.iso = Some(num(v)?),
            "below" => o.vol.below = Some(true),
            "above" => o.vol.below = Some(false),
            "fair" => o.fair = Some(true),
            "no-fair" => o.fair = Some(false),
            "grow" => o.grow = num(v)?,
            _ => bail!("unknown option `{opt}` (array=NAME, iso=X, below, above, fair, no-fair, grow=MM)"),
        }
    }
    let path = resolve(file, dir)?;
    let (grid, up) = load(&path, &o)?;
    Ok(Import {
        name: name.to_string(),
        path,
        grid,
        up,
    })
}

/// Where `file` (as written in a shape or its sidecar) is: next to the shape (`dir`), else
/// relative to the working directory.
pub fn resolve(file: &str, dir: &Path) -> Result<PathBuf> {
    let local = dir.join(file);
    if local.exists() || Path::new(file).is_absolute() {
        Ok(local)
    } else if Path::new(file).exists() {
        Ok(PathBuf::from(file))
    } else {
        bail!("{file}: not found (next to the shape or in the working directory)");
    }
}

/// `path`'s solid with the default options (as Print would mesh it), for an import primitive.
pub fn load_default(path: &Path) -> Result<Arc<SdfGrid>> {
    Ok(load(path, &Opts::default())?.0)
}

/// `path` as a shape in `dir` should refer to it: relative to `dir` when they share a root.
pub fn relative(path: &Path, dir: &Path) -> String {
    let (Ok(p), Ok(d)) = (path.canonicalize(), dir.canonicalize()) else {
        return path.display().to_string();
    };
    let (pc, dc): (Vec<_>, Vec<_>) = (p.components().collect(), d.components().collect());
    let common = pc.iter().zip(&dc).take_while(|(a, b)| a == b).count();
    if common <= 1 {
        return p.display().to_string();
    }
    let mut out = PathBuf::new();
    for _ in common..dc.len() {
        out.push("..");
    }
    for c in &pc[common..] {
        out.push(c);
    }
    out.display().to_string()
}

type Cached = (SystemTime, u64, Opts, Arc<SdfGrid>, Option<crate::mesh::Up>);

/// The newest field per file (one per file, so a result rewritten while optimizing doesn't pile up).
static CACHE: Mutex<Option<HashMap<PathBuf, Cached>>> = Mutex::new(None);

fn load(path: &Path, o: &Opts) -> Result<(Arc<SdfGrid>, Option<crate::mesh::Up>)> {
    let meta = std::fs::metadata(path)?;
    let stamp = (meta.modified()?, meta.len());
    let key = path.canonicalize()?;
    if let Some((t, len, opts, grid, up)) = CACHE.lock().unwrap().get_or_insert_default().get(&key)
        && (*t, *len) == stamp
        && opts == o
    {
        return Ok((grid.clone(), *up));
    }
    let vol = Volume::load(
        path,
        &VolumeOpts {
            array: o.vol.array.clone(),
            iso: o.vol.iso,
            below: o.vol.below,
            ..Default::default()
        },
    )?;
    if vol.inside_count == 0 {
        bail!(
            "{}: nothing inside: no samples of `{}` are past the threshold {}",
            path.display(),
            vol.array_name(),
            vol.iso
        );
    }
    let opts = MeshOpts {
        fair: o.fair.unwrap_or(vol.fair_by_default()),
        smooth: 0,
        grow: o.grow,
    };
    let grid = Arc::new(vol.sdf(&opts));
    let up = vol.up();
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(key, (stamp.0, stamp.1, o.clone(), grid.clone(), up));
    Ok((grid, up))
}

/// The box the imported solids can occupy: each grid's lattice (outside it the field is positive).
pub fn bounds(imports: &[Import]) -> Option<([f32; 3], [f32; 3])> {
    imports
        .iter()
        .map(|i| {
            let g = &i.grid;
            let lo: [f32; 3] = std::array::from_fn(|k| g.origin[k] as f32);
            let hi: [f32; 3] = std::array::from_fn(|k| (g.origin[k] + (g.dims[k] - 1) as f64 * g.spacing[k]) as f32);
            (lo, hi)
        })
        .reduce(|(a, b), (c, d)| {
            (
                std::array::from_fn(|k| a[k].min(c[k])),
                std::array::from_fn(|k| b[k].max(d[k])),
            )
        })
}

/// The WGSL declaring `g3d_imp` and each import's function.
pub fn wgsl(imports: &[Import]) -> String {
    let mut s = format!(
        "// ---- gen3d imported grids (//! import) ----
@group(0) @binding({BINDING}) var<storage, read> g3d_imp: array<f32>;
fn g3d_imp_at(o: u32, d: vec3i, i: vec3i) -> f32 {{
    let c = clamp(i, vec3i(0), d - 1);
    return g3d_imp[o + u32((c.z * d.y + c.y) * d.x + c.x)];
}}
// trilinear inside the lattice; outside, at least the distance to it (and to the surface beyond)
fn g3d_imp_sample(p: vec3f, o: u32, d: vec3i, org: vec3f, sp: vec3f) -> f32 {{
    let q = (p - org) / sp;
    let qc = clamp(q, vec3f(0.0), vec3f(d - 1));
    let i = clamp(vec3i(floor(qc)), vec3i(0), max(d - 2, vec3i(0)));
    let f = clamp(qc - vec3f(i), vec3f(0.0), vec3f(1.0));
    let a = mix(mix(g3d_imp_at(o, d, i), g3d_imp_at(o, d, i + vec3i(1, 0, 0)), f.x),
                mix(g3d_imp_at(o, d, i + vec3i(0, 1, 0)), g3d_imp_at(o, d, i + vec3i(1, 1, 0)), f.x), f.y);
    let b = mix(mix(g3d_imp_at(o, d, i + vec3i(0, 0, 1)), g3d_imp_at(o, d, i + vec3i(1, 0, 1)), f.x),
                mix(g3d_imp_at(o, d, i + vec3i(0, 1, 1)), g3d_imp_at(o, d, i + vec3i(1, 1, 1)), f.x), f.y);
    let v = mix(a, b, f.z);
    let out = length((q - qc) * sp);
    if (out > 0.0) {{ return sqrt(out * out + max(v, 0.0) * max(v, 0.0)); }}
    return v;
}}
"
    );
    let mut offset = 0usize;
    for i in imports {
        let g = &i.grid;
        let v = |a: [f64; 3]| format!("vec3f({:?}, {:?}, {:?})", a[0] as f32, a[1] as f32, a[2] as f32);
        s += &format!(
            "fn {}(p: vec3f) -> f32 {{ return g3d_imp_sample(p, {offset}u, vec3i({}, {}, {}), {}, {}); }}  // {}\n",
            i.name,
            g.dims[0],
            g.dims[1],
            g.dims[2],
            v(g.origin),
            v(g.spacing),
            i.path.display()
        );
        offset += g.sd.len();
    }
    s
}

/// `fn map` for a shape that has none: the union of the imported solids.
pub fn map(imports: &[Import]) -> String {
    let d = imports
        .iter()
        .map(|i| format!("{}(p)", i.name))
        .reduce(|a, b| format!("min({a}, {b})"))
        .unwrap_or_else(|| "1e9".into());
    format!("fn map(p: vec3f) -> f32 {{ return {d}; }}\n")
}

/// The storage buffer `wgsl` declares: every import's field, in order.
pub fn buffer(gpu: &crate::gpu::Gpu, imports: &[Import]) -> Result<Option<wgpu::Buffer>> {
    if imports.is_empty() {
        return Ok(None);
    }
    let data: Vec<f32> = imports.iter().flat_map(|i| i.grid.sd.iter().copied()).collect();
    let bytes = (data.len() * 4) as u64;
    let limits = gpu.device.limits();
    let max = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    if bytes > max {
        bail!(
            "imported grids too large for the GPU ({} samples; this GPU allows {})",
            data.len(),
            max / 4
        );
    }
    Ok(Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("imported grids"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    })))
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::mesher::{Grid, sample_field};
    use crate::shape::load_shape;
    use crate::vti::{GridArray, write_grid};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gen3d-import-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A binary `density` (cell data) of the box |x|, |y| < 4, 2 < z < 10 on 1 mm cells over
    /// [-8, 8] x [-8, 8] x [0, 12], like an optimizer result.
    pub(crate) fn write_box(path: &std::path::Path) {
        let (n, m) = ([16, 16, 12], [17, 17, 13]);
        let mut d = vec![0f32; n.iter().product()];
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let c = [x as f64 - 7.5, y as f64 - 7.5, z as f64 + 0.5];
                    if c[0].abs() < 4.0 && c[1].abs() < 4.0 && c[2] > 2.0 && c[2] < 10.0 {
                        d[(z * n[1] + y) * n[0] + x] = 1.0;
                    }
                }
            }
        }
        let a = GridArray {
            name: "density",
            cell: true,
            comps: 1,
            data: &d,
        };
        write_grid(path, [-8.0, -8.0, 0.0], [1.0; 3], m, &[a], &[]).unwrap();
    }

    #[test]
    fn directive_parses_and_compiles() {
        let dir = tmp("parse");
        write_box(&dir.join("r.vti"));
        let p = dir.join("s.wgsl");
        let shape = |text: &str| {
            std::fs::write(&p, text).unwrap();
            load_shape(&p, &[])
        };
        let s = shape(
            "//! import a r.vti iso=0.4 grow=1\nfn map(p: vec3f) -> f32 { return min(a(p), length(p) - 1.0); }\n",
        )
        .unwrap();
        assert_eq!(s.imports.len(), 1);
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        s.compile_preview(&crate::viewer::preview_harness(&[], false)).unwrap();
        // the grids give the bounds when the file has none; the file is watched
        let (lo, hi) = s.bounds;
        assert_eq!((lo, hi), ([-8.5, -8.5, -0.5], [8.5, 8.5, 12.5]));
        assert!(s.files.iter().any(|f| f.ends_with("r.vti") && f.exists()));
        // without a map: the shape is the imports (and works as an editor shape)
        let s = shape("//! import a r.vti\n//! import b r.vti no-fair\n").unwrap();
        s.compile(&crate::mesher::compute_harness("map", 1)).unwrap();
        assert!(s.editor && s.script_map);
        for bad in [
            "//! import a\n",
            "//! import map r.vti\n",
            "//! import a nope.vti\n",
            "//! import a r.vti iso=x\n",
            "//! import a r.vti sideways\n",
            "//! import a r.vti array=nope\n",
        ] {
            assert!(shape(bad).is_err(), "{bad}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The imported field on the GPU: negative inside the box, positive outside, about a distance
    /// near the surface (faired: within half a voxel of the cell faces) and beyond the grid.
    /// Skipped without a GPU adapter.
    #[test]
    fn imported_box_samples_on_gpu() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = tmp("gpu");
        write_box(&dir.join("r.vti"));
        let p = dir.join("s.wgsl");
        for opt in ["", "no-fair"] {
            std::fs::write(&p, format!("//! import a r.vti {opt}\n")).unwrap();
            let shape = load_shape(&p, &[]).unwrap();
            let grid = Grid {
                vol: vec![],
                dims: [1, 1, 15],
                origin: [0.3, -0.2, -1.0],
                step: 1.0,
            };
            // along z through the box: the surface at z = 2 and 10
            let v = sample_field(&shape, &grid, "map", 1, &gpu).unwrap();
            for (k, d) in v.iter().enumerate() {
                let z = -1.0 + k as f32;
                let exact = (2.0 - z).max(z - 10.0);
                assert!(
                    (d - exact).abs() < 0.6 || (exact.abs() > 1.5 && d.signum() == exact.signum()),
                    "{opt} z={z}: {d} vs {exact}"
                );
            }
            // far outside the grid: about the distance to it
            let far = Grid {
                vol: vec![],
                dims: [1, 1, 1],
                origin: [30.0, 0.0, 6.0],
                step: 1.0,
            };
            let d = sample_field(&shape, &far, "map", 1, &gpu).unwrap()[0];
            assert!(d > 21.0 && d < 27.0, "{opt}: {d}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An import primitive, moved and rotated: the GPU's field matches the CPU copy used for
    /// picking, and the box is where `pos` puts its centre. Skipped without a GPU adapter.
    #[test]
    fn import_primitive_moves_and_matches_cpu() {
        use crate::prims::{Kind, Prim, save, sidecar};
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = tmp("prim");
        write_box(&dir.join("r.vti"));
        let p = dir.join("s.wgsl");
        std::fs::write(&p, "").unwrap();
        for rot in [[0.0, 0.0, 0.0], [25.0, -40.0, 70.0]] {
            let prim = Prim {
                file: "r.vti".into(),
                rot,
                ..Prim::new(Kind::Import, "r".into(), [5.0, 5.0, 20.0], 0.0)
            };
            save(&sidecar(&p), &[prim], &[None; 3], None).unwrap();
            let shape = load_shape(&p, &[]).unwrap();
            let prim = &shape.prims[0];
            assert!(prim.grid.is_some());
            // the grid's box (17 x 17 x 13 mm around the centre) moved to pos, rotated
            let (lo, hi) = shape.bounds;
            let pos = [5.0, 5.0, 20.0];
            assert!(
                (0..3).all(|k| lo[k] < pos[k] - 6.0 && hi[k] > pos[k] + 6.0),
                "{rot:?}: {lo:?} {hi:?}"
            );
            let grid = Grid {
                vol: vec![],
                dims: [9, 9, 9],
                origin: [-7.0, -7.0, 8.0],
                step: 3.0,
            };
            let gpu_vals = sample_field(&shape, &grid, "map", 1, &gpu).unwrap();
            let mut i = 0;
            for x in 0..9 {
                for y in 0..9 {
                    for z in 0..9 {
                        let q = [-7.0 + 3.0 * x as f32, -7.0 + 3.0 * y as f32, 8.0 + 3.0 * z as f32];
                        let cpu = prim.sdf(q);
                        assert!(
                            (cpu - gpu_vals[i]).abs() < 1e-3,
                            "{rot:?} at {q:?}: cpu {cpu} gpu {}",
                            gpu_vals[i]
                        );
                        i += 1;
                    }
                }
            }
            // the box's centre (z = 6 in the file) is at pos, well inside (4 mm from the faces; the
            // faired field is flatter than a distance that deep)
            let c = prim.sdf([5.0, 5.0, 20.0]);
            assert!(c < -2.5 && c > -4.5, "{rot:?}: centre {c}");
            assert!(prim.sdf([0.0, 0.0, 6.0]) > 2.0, "not at the file's position any more");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
