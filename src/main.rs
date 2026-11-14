//! gen3d: procedural SDF modelling workbench. WGSL shapes -> live preview -> mesh -> G-code.

mod app;
mod browser;
mod gpu;
mod jobs;
mod mc;
mod mesh;
mod mesher;
mod meshview;
mod prims;
mod shape;
mod slicer;
mod topopt;
mod viewer;
mod volume;
mod vti;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use shape::{Shape, load_shape, parse_defines};

#[derive(Parser)]
#[command(name = "gen3d", about = "Procedural SDF modelling workbench", version)]
#[command(after_help = "Run `gen3d` without a command to open the app.\n\
                        slice: arguments after `--` are passed to the slicer, e.g. -- --layer-height 0.2")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Args)]
struct ShapeArgs {
    /// path to .wgsl shape file
    shape: PathBuf,
    /// set a shape parameter (an `override` constant), repeatable, e.g. -D RADIUS=12
    #[arg(short = 'D', long = "define", value_name = "NAME=VALUE")]
    define: Vec<String>,
    /// .vti: array that forms the solid (default: the first)
    #[arg(long, help_heading = ".vti files")]
    array: Option<String>,
    /// .vti: threshold (default: 0 for SDF-like arrays, else the middle of the value range)
    #[arg(long, allow_negative_numbers = true, help_heading = ".vti files")]
    iso: Option<f32>,
    /// .vti: inside is where the array is below the threshold (default for SDF-like arrays)
    #[arg(long, conflicts_with = "above", help_heading = ".vti files")]
    below: bool,
    /// .vti: inside is where the array is above the threshold (default for densities)
    #[arg(long, help_heading = ".vti files")]
    above: bool,
    /// .vti: draw voxels as cubes instead of a smooth surface (view/render)
    #[arg(long, help_heading = ".vti files")]
    blocky: bool,
    /// .vti: colour the surface by this array (view/render)
    #[arg(long, help_heading = ".vti files")]
    color: Option<String>,
}

impl ShapeArgs {
    fn volume_opts(&self) -> volume::VolumeOpts {
        volume::VolumeOpts {
            array: self.array.clone(),
            iso: self.iso,
            below: if self.below {
                Some(true)
            } else if self.above {
                Some(false)
            } else {
                None
            },
            blocky: self.blocky,
            color: self.color.clone(),
            color_range: None,
        }
    }
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
struct SolverArgs {
    /// linear solver: gpu (multigrid-preconditioned CG on the GPU, f32), mg (the same on the CPU),
    /// jacobi (Jacobi CG on the CPU: slow on large grids), or auto (gpu unless the GPU is a software
    /// renderer, then mg)
    #[arg(long, value_enum, default_value = "auto")]
    solver: SolverKind,
    /// multigrid smoother
    #[arg(long, value_enum, default_value = "chebyshev", hide = true)]
    smoother: SmootherKind,
    /// Chebyshev degree, or Jacobi sweeps
    #[arg(long, default_value_t = 2, hide = true)]
    sweeps: usize,
    /// Jacobi smoother damping, as a multiple of 1 / λmax(D⁻¹K)
    #[arg(long, default_value_t = 1.2, hide = true)]
    omega: f64,
    /// stiffness of empty elements relative to the modulus (default 1e-9; 1e-6 with jacobi)
    #[arg(long, hide = true)]
    emin: Option<f64>,
    /// relative residual at which each linear solve stops
    #[arg(long, default_value_t = 1e-4, hide = true)]
    cg_tol: f64,
    /// always solve to --cg-tol (by default the solves are looser while the design moves a lot)
    #[arg(long, hide = true)]
    fixed_cg_tol: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum SolverKind {
    Auto,
    Mg,
    Gpu,
    Jacobi,
}

#[derive(Clone, Copy, ValueEnum)]
enum SmootherKind {
    Chebyshev,
    Jacobi,
}

impl SolverArgs {
    fn solver(&self, gpu: &gpu::Gpu) -> topopt::Solver {
        let kind = match self.solver {
            SolverKind::Auto if gpu.is_software() => SolverKind::Mg,
            SolverKind::Auto => SolverKind::Gpu,
            k => k,
        };
        match kind {
            SolverKind::Jacobi => topopt::Solver::Jacobi,
            _ => {
                let ms = topopt::mg::MgSettings {
                    smoother: match self.smoother {
                        SmootherKind::Chebyshev => topopt::mg::Smoother::Chebyshev { degree: self.sweeps },
                        SmootherKind::Jacobi => topopt::mg::Smoother::Jacobi {
                            sweeps: self.sweeps,
                            omega: self.omega,
                        },
                    },
                    ..Default::default()
                };
                if matches!(kind, SolverKind::Gpu) {
                    topopt::Solver::Gpu(ms)
                } else {
                    topopt::Solver::Multigrid(ms)
                }
            }
        }
    }

    fn emin(&self, gpu: &gpu::Gpu) -> f64 {
        self.emin
            .unwrap_or_else(|| topopt::Settings::default_emin(&self.solver(gpu)))
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
    /// .vti: Taubin smoothing passes on the mesh (0 = off; default 10)
    #[arg(long)]
    smooth: Option<u32>,
    #[arg(long, value_enum, default_value = "stl")]
    format: Format,
}

#[derive(Subcommand)]
enum Cmd {
    /// open the app (design -> optimize -> print), optionally with a .wgsl shape or .vti grid
    View {
        /// .wgsl shape or .vti grid to open
        file: Option<PathBuf>,
        /// set a shape parameter, e.g. -D RADIUS=12
        #[arg(short = 'D', long = "define", value_name = "NAME=VALUE")]
        define: Vec<String>,
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
    /// render a preview image (PNG) headlessly, of a shape, a .vti grid or a mesh (.stl/.3mf/.obj)
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
        /// hide the `field_*` region overlays
        #[arg(long)]
        no_fields: bool,
        /// mesh files (.stl, .3mf, .obj): draw the triangle edges
        #[arg(long)]
        edges: bool,
    },
    /// topology optimization (SIMP, minimum compliance) of the shape's design domain; see README
    Optimize {
        #[command(flatten)]
        s: ShapeArgs,
        /// element size in mm (default: about 300k elements in the bounds with the GPU solver, 40k
        /// with the CPU ones)
        #[arg(long)]
        voxel: Option<f32>,
        /// fraction of the design domain to fill
        #[arg(long, default_value_t = 0.3)]
        volfrac: f64,
        /// maximum iterations (stops early when converged)
        #[arg(long, default_value_t = 150)]
        iters: usize,
        /// final sharpness of the Heaviside projection (crisp 0/1 designs); beta starts at 1 and
        /// doubles each time the design settles
        #[arg(long, default_value_t = 32.0)]
        beta_max: f64,
        /// no Heaviside projection (faster, but leaves grey, partly dense regions)
        #[arg(long)]
        no_projection: bool,
        /// do the first projection stage on the fine grid too, instead of on a grid of twice the
        /// element size first (slower; the coarse start applies from 80k domain elements)
        #[arg(long)]
        no_coarse_start: bool,
        /// solve the whole part even when the shape has mirror planes (by default a symmetric
        /// problem is solved on one side of them)
        #[arg(long)]
        no_symmetry: bool,
        /// density filter radius in voxels (larger = thicker, simpler members)
        #[arg(long, default_value_t = 1.5)]
        rmin: f64,
        /// SIMP penalty
        #[arg(long, default_value_t = 3.0)]
        penal: f64,
        /// total load in N, spread over the field_load region
        #[arg(long, default_value_t = 100.0)]
        force: f64,
        /// the part stands on the ground: the bottom of the domain is supported, held vertically
        /// only (slide) or in all directions (grip)
        #[arg(long, value_enum, default_value_t = topopt::Ground::None)]
        ground: topopt::Ground,
        /// Young's modulus in MPa (PLA ~3500)
        #[arg(long, default_value_t = 3500.0)]
        modulus: f64,
        /// Poisson's ratio
        #[arg(long, default_value_t = 0.35)]
        nu: f64,
        /// output .vti (default out/NAME_topopt.vti), rewritten as it runs
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// continue a stopped run from the checkpoint it left next to the output (same settings;
        /// --iters still counts from the run's start)
        #[arg(long)]
        resume: bool,
        #[command(flatten)]
        solver: SolverArgs,
    },
    /// time `optimize` on a shape at several element sizes (default: the bracket at 4 sizes)
    #[command(hide = true)]
    BenchTopopt {
        #[arg(default_value = "shapes/bracket_domain.wgsl")]
        shape: PathBuf,
        /// element sizes (mm)
        #[arg(long, value_delimiter = ',', default_values_t = [1.3, 0.8, 0.5, 0.4])]
        voxels: Vec<f32>,
        /// optimizer iterations per size
        #[arg(long, default_value_t = 5)]
        iters: usize,
        #[command(flatten)]
        solver: SolverArgs,
    },
    /// draw the app's window offscreen to a PNG (for checking the UI without a display)
    #[command(hide = true)]
    Snapshot {
        file: Option<PathBuf>,
        #[arg(short = 'D', long = "define", value_name = "NAME=VALUE")]
        define: Vec<String>,
        /// design, optimize or print
        #[arg(long, default_value = "design")]
        step: String,
        /// in the Print step, print the optimization result
        #[arg(long)]
        result: bool,
        /// show the Open dialog
        #[arg(long)]
        browser: bool,
        #[arg(long)]
        section: bool,
        /// select primitive N of the shape's editor
        #[arg(long)]
        select: Option<usize>,
        /// in the Print step, show the exported mesh
        #[arg(long)]
        mesh: bool,
        /// with --mesh: draw the triangle edges
        #[arg(long)]
        edges: bool,
        #[arg(long, num_args = 2, value_names = ["W", "H"], default_values_t = [1440u32, 900])]
        size: Vec<u32>,
        #[arg(short, long, default_value = "out/snapshot.png")]
        output: PathBuf,
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

/// Where `optimize` writes its result for a design and parameter values: out/NAME[_K-V]_topopt.vti.
fn topopt_out(shape_path: &Path, defines: &[(String, f64)]) -> PathBuf {
    let p = default_out(shape_path, ".vti", defines);
    p.with_file_name(format!(
        "{}_topopt.vti",
        p.file_stem().unwrap_or_default().to_string_lossy()
    ))
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
    if viewer::is_vti(&s.shape) {
        return export_volume(s, m, output, ext);
    }
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
    if m.smooth.is_some() {
        bail!("--smooth applies to .vti grids (shapes are meshed from their exact distance field)");
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

/// Mesh the threshold surface of a .vti array (e.g. a topology-optimization density).
fn export_volume(s: &ShapeArgs, m: &MeshArgs, output: Option<&Path>, ext: Option<&str>) -> Result<(Shape, PathBuf)> {
    if !s.define.is_empty() {
        bail!("-D parameters don't apply to .vti files");
    }
    if m.format == Format::Vti {
        bail!(
            "{} is already a .vti file; export it as stl, 3mf, obj or ply",
            s.shape.display()
        );
    }
    if m.voxel.is_some() || m.bounds.is_some() {
        bail!("--voxel and --bounds don't apply to .vti files (the grid is fixed)");
    }
    if s.blocky {
        bail!("--blocky export is not supported yet; export meshes the smooth threshold surface");
    }
    let t0 = std::time::Instant::now();
    let vol = volume::Volume::load(&s.shape, &s.volume_opts())?;
    println!(
        "{}: {} of {} samples inside",
        vol.label(),
        vol.inside_count,
        vol.count()
    );
    let mut mesh = vol.mesh()?;
    println!(
        "marching cubes in {:.2}s: {} triangles",
        t0.elapsed().as_secs_f64(),
        mesh.tris.len()
    );
    let passes = m.smooth.unwrap_or(mesher::DEFAULT_SMOOTH);
    if passes > 0 {
        let t1 = std::time::Instant::now();
        let constraint = vol.domain_constraint();
        mesh.smooth(passes, constraint.as_ref().map(|c| c as &mesh::VertexConstraint));
        println!("smoothed ({passes} passes) in {:.2}s", t1.elapsed().as_secs_f64());
    }
    let mesh = mesher::decimate_checked(mesh, m.decimate);
    mesher::report(&mesh);
    let shape = vol.shape(&s.shape);
    let out = match (ext, output) {
        (None, Some(o)) => o.to_path_buf(),
        (None, None) => default_out(&s.shape, m.format.ext(), &[]),
        (Some(ext), _) => default_out(&s.shape, ext, &[]),
    };
    mesher::save(&mesh, &out)?;
    Ok((shape, out))
}

fn run(cli: Cli) -> Result<()> {
    let Some(cmd) = cli.cmd else {
        return app::run(None, vec![]);
    };
    match cmd {
        Cmd::View { file, define } => app::run(file.as_deref(), parse_defines(&define)?),
        Cmd::Snapshot {
            file,
            define,
            step,
            result,
            browser,
            section,
            select,
            mesh,
            edges,
            size,
            output,
        } => {
            let setup = app::SnapshotSetup {
                step,
                print_result: result,
                browser,
                section,
                select,
                mesh,
                edges,
            };
            app::snapshot(
                file.as_deref(),
                parse_defines(&define)?,
                &setup,
                (size[0], size[1]),
                &output,
            )?;
            println!("wrote {}", output.display());
            Ok(())
        }
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
            let given = PathBuf::from(&name);
            let path = if given.extension().is_some_and(|e| e == "wgsl") {
                let dir = given
                    .parent()
                    .filter(|d| !d.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let file = given.file_name().unwrap_or_default().to_string_lossy();
                shape::create_from_template(dir, &file)?
            } else {
                shape::create_from_template(Path::new("shapes"), &name)?
            };
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
            no_fields,
            edges,
        } => {
            let defines = parse_defines(&s.define)?;
            let out = output.unwrap_or_else(|| default_out(&s.shape, ".png", &defines));
            if meshview::is_mesh(&s.shape) {
                meshview::render_png(&s.shape, &out, (size[0], size[1]), yaw, pitch, edges)?;
                println!("wrote {}", out.display());
                return Ok(());
            }
            let view = viewer::RenderView {
                size: (size[0], size[1]),
                section,
                yaw,
                pitch,
                show_fields: !no_fields,
            };
            viewer::render_png(&s.shape, &defines, &s.volume_opts(), &out, &view)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Optimize {
            s,
            voxel,
            volfrac,
            iters,
            beta_max,
            no_projection,
            no_coarse_start,
            no_symmetry,
            rmin,
            penal,
            force,
            ground,
            modulus,
            nu,
            output,
            resume,
            solver,
        } => {
            if viewer::is_vti(&s.shape) {
                bail!("optimize takes a .wgsl problem (design domain + field_fixed/field_load), not a .vti");
            }
            if !(0.0..1.0).contains(&volfrac) || volfrac == 0.0 {
                bail!("--volfrac must be between 0 and 1");
            }
            let defines = parse_defines(&s.define)?;
            let shape = load_shape(&s.shape, &defines)?;
            let gpu = gpu::headless()?;
            let on_gpu = matches!(solver.solver(&gpu), topopt::Solver::Gpu(_));
            let settings = topopt::Settings {
                voxel: voxel.unwrap_or_else(|| topopt::auto_voxel(&shape, topopt::auto_elements(on_gpu))) as f64,
                volfrac,
                penal,
                rmin,
                iters,
                tol: 0.01,
                e0: modulus,
                nu,
                force,
                beta_max: if no_projection { 0.0 } else { beta_max.max(1.0) },
                solver: solver.solver(&gpu),
                emin: solver.emin(&gpu),
                cg_tol: solver.cg_tol,
                adaptive_tol: !solver.fixed_cg_tol,
                ground,
                coarse_start: !no_coarse_start,
                symmetry: !no_symmetry,
            };
            let out = output.unwrap_or_else(|| topopt_out(&shape.path, &defines));
            topopt::run(&shape, &settings, &out, &gpu, resume)?;
            Ok(())
        }
        Cmd::BenchTopopt {
            shape,
            voxels,
            iters,
            solver,
        } => {
            let shape = load_shape(&shape, &[])?;
            let gpu = gpu::headless()?;
            let dir = std::env::temp_dir().join(format!("gen3d-bench-{}", std::process::id()));
            std::fs::create_dir_all(&dir)?;
            let mut rows = vec![];
            for &voxel in &voxels {
                println!("\n=== voxel {voxel} mm ===");
                let settings = topopt::Settings {
                    voxel: voxel as f64,
                    volfrac: 0.3,
                    penal: 3.0,
                    rmin: 1.5,
                    iters,
                    tol: 0.0,
                    e0: 3500.0,
                    nu: 0.35,
                    force: 100.0,
                    beta_max: 32.0,
                    solver: solver.solver(&gpu),
                    emin: solver.emin(&gpu),
                    cg_tol: solver.cg_tol,
                    adaptive_tol: !solver.fixed_cg_tol,
                    ground: topopt::Ground::None,
                    coarse_start: false,
                    symmetry: true,
                };
                let problem = topopt::setup(&shape, &settings, &gpu)?;
                let ne = problem.domain.iter().filter(|d| **d).count();
                let r = topopt::optimize(&problem, &settings, &dir.join("bench.vti"), Some(&gpu))?;
                rows.push((voxel, problem.n, ne, r));
            }
            std::fs::remove_dir_all(&dir).ok();
            println!(
                "\n{:>6} {:>13} {:>9} {:>7} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>11}",
                "voxel",
                "grid",
                "domain",
                "CG/sol",
                "OC/it",
                "setup s",
                "solve s",
                "update s",
                "io s",
                "iter s",
                "compliance"
            );
            for (voxel, n, ne, r) in rows {
                let k = r.stats.len() as f64;
                let mean = |f: fn(&topopt::IterStats) -> f64| r.stats.iter().map(f).sum::<f64>() / k;
                let solves: usize = r.stats.iter().map(|s| s.cg.len()).sum();
                let cg: usize = r.stats.iter().flat_map(|s| &s.cg).sum();
                println!(
                    "{voxel:>6} {:>13} {ne:>9} {:>7.1} {:>6.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>11.4e}",
                    format!("{}x{}x{}", n[0], n[1], n[2]),
                    cg as f64 / solves as f64,
                    mean(|s| s.oc_steps as f64),
                    mean(|s| s.setup),
                    mean(|s| s.solve),
                    mean(|s| s.update),
                    mean(|s| s.io),
                    mean(|s| s.setup + s.solve + s.update + s.io),
                    r.compliance,
                );
            }
            Ok(())
        }
        Cmd::Check { s } if viewer::is_vti(&s.shape) => {
            let f = vti::read(&s.shape)?;
            let (o, d) = f.lattice(false);
            println!(
                "ok: {} points from ({}, {}, {}) mm, spacing {:?} mm\narrays:\n{}",
                d.iter().map(|n| n.to_string()).collect::<Vec<_>>().join("x"),
                o[0],
                o[1],
                o[2],
                f.spacing,
                f.describe().trim_end()
            );
            Ok(())
        }
        Cmd::Check { s } => {
            let shape = load_shape(&s.shape, &parse_defines(&s.define)?)?;
            shape.compile(&mesher::compute_harness("map", 1))?;
            let fields = shape.fields()?;
            shape.compile(&viewer::preview_harness(&fields, shape.editor))?;
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
