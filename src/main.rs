//! gen3d: procedural SDF modelling workbench. WGSL shapes -> live preview -> mesh -> G-code.

mod gpu;
mod mc;
mod mesh;
mod mesher;
mod shape;
mod slicer;
mod viewer;
mod vti;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use shape::{Shape, load_shape, parse_defines};

const TEMPLATE: &str = "// {name}.wgsl — gen3d shape. Units: mm, +Z up, build plate at z = 0.
// Export region (xmin ymin zmin  xmax ymax zmax) and mesh resolution in mm:
//! bounds -30 -30 0  30 30 50
//! voxel 0.2
#include <sdf.wgsl>

// parameters: override with -D R=15
override R: f32 = 12.0;

fn map(p: vec3f) -> f32 {
    let body = sdRoundBox(p - vec3f(0, 0, 15), vec3f(20, 20, 15), 3.0);
    let hole = sdCylinder(rotX(p - vec3f(0, 0, 13), deg(90.0)), 7.0, 25.0);  // through-hole along Y
    let ball = sdSphere(p - vec3f(0, 0, 30), R);
    return opSmoothUnion(opSmoothSubtract(body, hole, 2.0), ball, 4.0);
}
";

#[derive(Parser)]
#[command(name = "gen3d", about = "Procedural SDF modelling workbench", version)]
#[command(after_help = "slice: arguments after `--` are passed to the slicer, e.g. -- --layer-height 0.2")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct ShapeArgs {
    /// path to .wgsl shape file
    shape: PathBuf,
    /// set a shape parameter (an `override` constant), repeatable, e.g. -D RADIUS=12
    #[arg(short = 'D', long = "define", value_name = "NAME=VALUE")]
    define: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Format {
    Stl,
    #[value(name = "3mf")]
    ThreeMf,
    Obj,
    Ply,
    /// sampled SDF grid as VTK ImageData (no meshing), for simulation
    Vti,
}

impl Format {
    fn ext(self) -> &'static str {
        match self {
            Format::Stl => ".stl",
            Format::ThreeMf => ".3mf",
            Format::Obj => ".obj",
            Format::Ply => ".ply",
            Format::Vti => ".vti",
        }
    }
}

#[derive(Args)]
struct MeshArgs {
    /// voxel size in mm (overrides //! voxel)
    #[arg(long)]
    voxel: Option<f32>,
    #[arg(long, num_args = 6, value_names = ["X0", "Y0", "Z0", "X1", "Y1", "Z1"], allow_negative_numbers = true)]
    bounds: Option<Vec<f32>>,
    /// keep this fraction of triangles (0-1)
    #[arg(long)]
    decimate: Option<f32>,
    #[arg(long, value_enum, default_value = "stl")]
    format: Format,
}

#[derive(Subcommand)]
enum Cmd {
    /// live preview window with hot reload
    View {
        #[command(flatten)]
        s: ShapeArgs,
    },
    /// mesh the SDF and write STL/3MF/OBJ/PLY, or the raw SDF grid as VTI
    Export {
        #[command(flatten)]
        s: ShapeArgs,
        #[command(flatten)]
        m: MeshArgs,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// export and slice to G-code (PrusaSlicer/OrcaSlicer CLI)
    Slice {
        #[command(flatten)]
        s: ShapeArgs,
        #[command(flatten)]
        m: MeshArgs,
        /// G-code output path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// slicer config (PrusaSlicer .ini, or Orca .json presets); repeatable
        #[arg(short, long)]
        config: Vec<PathBuf>,
        /// prefer this slicer
        #[arg(long, value_enum)]
        slicer: Option<slicer::Kind>,
        /// extra slicer arguments (after `--`)
        #[arg(last = true)]
        slicer_args: Vec<String>,
    },
    /// export and open in the slicer GUI
    Open {
        #[command(flatten)]
        s: ShapeArgs,
        #[command(flatten)]
        m: MeshArgs,
        #[arg(long, value_enum)]
        slicer: Option<slicer::Kind>,
    },
    /// create a new shape from a template
    New { name: String },
    /// render a preview image (PNG) headlessly
    Render {
        #[command(flatten)]
        s: ShapeArgs,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, num_args = 2, value_names = ["W", "H"], default_values_t = [1024u32, 768])]
        size: Vec<u32>,
        /// show cross-section
        #[arg(long)]
        section: bool,
        #[arg(long, default_value_t = -60.0, allow_negative_numbers = true)]
        yaw: f32,
        #[arg(long, default_value_t = 28.0, allow_negative_numbers = true)]
        pitch: f32,
    },
    /// compile a shape and report errors
    Check {
        #[command(flatten)]
        s: ShapeArgs,
    },
}

fn default_out(shape_path: &Path, ext: &str, defines: &[(String, f64)]) -> PathBuf {
    let stem = shape_path.file_stem().unwrap_or_default().to_string_lossy();
    let suffix: String = defines.iter().map(|(k, v)| format!("_{k}-{v}")).collect();
    std::env::current_dir()
        .unwrap_or_default()
        .join("out")
        .join(format!("{stem}{suffix}{ext}"))
}

fn resolve_configs(configs: &[PathBuf], shape: &Shape) -> Vec<PathBuf> {
    if !configs.is_empty() {
        return configs.to_vec();
    }
    if let Some(d) = shape.directives.get("slicer") {
        let dir = shape.path.parent().unwrap_or(Path::new("."));
        return d
            .split(',')
            .map(|c| dir.join(c.trim()))
            .map(|p| p.canonicalize().unwrap_or(p))
            .collect();
    }
    let default = PathBuf::from("printers/default.ini");
    if default.exists() {
        vec![std::env::current_dir().unwrap_or_default().join(default)]
    } else {
        vec![]
    }
}

/// Mesh the shape. `ext = None` (the `export` command) means `-o` is the mesh path; otherwise the
/// mesh goes to out/ and `-o` means something else.
fn export(s: &ShapeArgs, m: &MeshArgs, output: Option<&Path>, ext: Option<&str>) -> Result<(Shape, PathBuf)> {
    let defines = parse_defines(&s.define)?;
    let mut shape = load_shape(&s.shape, &defines)?;
    if let Some(b) = &m.bounds {
        shape.bounds = ([b[0], b[1], b[2]], [b[3], b[4], b[5]]);
    }
    let vti = ext.is_none()
        && (m.format == Format::Vti
            || output.is_some_and(|o| o.extension().is_some_and(|e| e.eq_ignore_ascii_case("vti"))));
    if vti {
        let out = output.map_or_else(|| default_out(&shape.path, ".vti", &defines), Path::to_path_buf);
        let gpu = gpu::headless()?;
        let grid = mesher::sample_grid(&shape, m.voxel, &gpu)?;
        let fields = shape.fields()?;
        let sampled = fields
            .iter()
            .map(|f| mesher::sample_field(&shape, &grid, &f.func, f.comps, &gpu))
            .collect::<Result<Vec<_>>>()?;
        let mut arrays = vec![vti::Array {
            name: "sdf",
            comps: 1,
            data: &grid.vol,
        }];
        arrays.extend(fields.iter().zip(&sampled).map(|(f, data)| vti::Array {
            name: &f.name,
            comps: f.comps,
            data,
        }));
        vti::save(&grid, &arrays, &out)?;
        return Ok((shape, out));
    }
    if ext.is_some() && m.format == Format::Vti {
        bail!("--format vti only applies to `export`");
    }
    let decimate = m.decimate.or(shape.decimate());
    let mesh = mesher::build_mesh(&shape, m.voxel, decimate)?;
    mesher::report(&mesh);
    let out = match (ext, output) {
        (None, Some(o)) => o.to_path_buf(),
        (None, None) => default_out(&shape.path, m.format.ext(), &defines),
        (Some(ext), _) => default_out(&shape.path, ext, &defines),
    };
    mesher::save(&mesh, &out)?;
    Ok((shape, out))
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::View { s } => viewer::view(&s.shape, parse_defines(&s.define)?),
        Cmd::Export { s, m, output } => export(&s, &m, output.as_deref(), None).map(|_| ()),
        Cmd::Slice {
            s,
            m,
            output,
            config,
            slicer: prefer,
            slicer_args,
        } => {
            let (shape, model) = export(&s, &m, None, Some(".stl"))?;
            let gcode = output.unwrap_or_else(|| model.with_extension("gcode"));
            let configs = resolve_configs(&config, &shape);
            if configs.is_empty() {
                println!(
                    "note: no slicer config given (--config, //! slicer, or printers/default.ini); \
                     using the slicer's built-in defaults"
                );
            }
            slicer::slice_to_gcode(&model, &gcode, &configs, &slicer_args, prefer)
        }
        Cmd::Open { s, m, slicer: prefer } => {
            let ext = if m.format == Format::ThreeMf { ".3mf" } else { ".stl" };
            let (_, model) = export(&s, &m, None, Some(ext))?;
            let exe = slicer::open_in_gui(&model, prefer)?;
            println!("opened in {}", exe.display());
            Ok(())
        }
        Cmd::New { name } => {
            let mut path = PathBuf::from(&name);
            if path.extension().is_none_or(|e| e != "wgsl") {
                path = Path::new("shapes").join(format!("{name}.wgsl"));
            }
            if path.exists() {
                bail!("{} already exists", path.display());
            }
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            std::fs::write(&path, TEMPLATE.replace("{name}", &stem))?;
            println!("created {}\nrun: gen3d view {}", path.display(), path.display());
            Ok(())
        }
        Cmd::Render {
            s,
            output,
            size,
            section,
            yaw,
            pitch,
        } => {
            let defines = parse_defines(&s.define)?;
            let out = output.unwrap_or_else(|| default_out(&s.shape, ".png", &defines));
            viewer::render_png(&s.shape, &defines, &out, (size[0], size[1]), section, yaw, pitch)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Check { s } => {
            let shape = load_shape(&s.shape, &parse_defines(&s.define)?)?;
            shape.compile(&mesher::compute_harness("map", 1))?;
            shape.compile(viewer::PREVIEW_HARNESS)?;
            let fields = shape.fields()?;
            for f in &fields {
                shape.compile(&mesher::compute_harness(&f.func, f.comps))?;
            }
            let name = shape.path.file_name().unwrap_or_default().to_string_lossy();
            println!(
                "ok: {name} compiles; bounds {:?}, voxel {} mm",
                shape.bounds, shape.voxel
            );
            if !fields.is_empty() {
                let list: Vec<_> = fields.iter().map(|f| format!("{} ({}f)", f.name, f.comps)).collect();
                println!("fields: {}", list.join(", "));
            }
            Ok(())
        }
    }
}

fn main() {
    // Mesa's EGL loader is chatty when probing GPUs; only show real failures.
    for (k, v) in [("EGL_LOG_LEVEL", "fatal"), ("MESA_LOG_LEVEL", "error")] {
        if std::env::var_os(k).is_none() {
            // SAFETY: single-threaded at this point, before any GPU or windowing initialisation.
            unsafe { std::env::set_var(k, v) };
        }
    }
    if let Err(e) = run(Cli::parse()) {
        eprintln!("\x1b[31merror:\x1b[0m {e:#}");
        std::process::exit(1);
    }
}
