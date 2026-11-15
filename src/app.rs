//! The gen3d app: one window that takes a shape from design to G-code in three steps.
//!
//! 1. **Design**: open (or create) a .wgsl shape, edit it in your editor (the preview reloads on
//!    save), tune its parameters with sliders, and add primitives (spheres, boxes, ...) that are
//!    unioned with the script (`prims.rs`).
//! 2. **Optimize** (optional): topology optimization of the shape as a design domain. The result
//!    grows live in the same viewport; pick its threshold.
//! 3. **Print**: mesh the design or the optimized result, pick a printer profile, slice to G-code
//!    or open it in the slicer.
//!
//! Every output has a fixed place in `out/` derived from the design and its parameter values, so the
//! app finds earlier results again when you reopen a shape. Long steps run as `gen3d` subprocesses
//! (`jobs.rs`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::browser::{Browser, Pick, ago};
use crate::gpu::Gpu;
use crate::jobs::{Job, Slot, mesh_report, opt_progress};
use crate::mesh::Up;
use crate::prims::{self, Kind, Op, Prim, Role};
use crate::shape::{Field, Param, ParamKind, Shape};
use crate::slicer;
use crate::topopt::Ground;
use crate::viewer::{
    Bindings, Camera, Options, Pipeline, Source, VolGpu, all_fields, build_pipeline, compiled_defines, draw,
    field_color, is_vti, load_source, preview_module, uniform_buffer, uniforms, volume_source,
};
use crate::volume::{Volume, VolumeOpts};

const KEYS: &[(&str, &str)] = &[
    ("Ctrl+O", "open a shape"),
    ("left-drag", "orbit"),
    ("right/middle-drag", "pan"),
    ("wheel", "zoom"),
    ("double-click", "reset camera"),
    ("click", "select a primitive"),
    (
        "drag arrows",
        "move it along X/Y/Z (centre: in the view plane; Ctrl snaps to 1 mm)",
    ),
    ("Del  Esc", "delete / deselect the primitive"),
    ("Ctrl+D", "duplicate the primitive"),
    ("Ctrl+Z  Ctrl+Shift+Z", "undo / redo primitive edits"),
    ("1 2 3 4", "front / side / top / iso"),
    ("R", "reset camera"),
    ("C", "section view"),
    ("[  ]", "move section plane"),
    ("B", "clip to bounds"),
    ("V", "bounds box"),
    ("F", "cycle field overlays (grids: colour by array)"),
    ("-  =", "grids: lower / raise threshold"),
    ("M", "grids: smooth / blocky"),
    ("I", "grids: flip inside"),
    ("A", "grids: next array"),
    ("F5", "force reload"),
];

/// Format of the offscreen texture the preview is raymarched into. egui samples textures as
/// gamma-encoded, so the shader applies gamma itself (as for `render_png`).
const VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const OK_GREEN: egui::Color32 = egui::Color32::from_rgb(90, 190, 100);
/// The preview's surface orange, for "inside" in the threshold histogram.
const SURFACE_ORANGE: egui::Color32 = egui::Color32::from_rgb(242, 165, 80);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Step {
    Design,
    Optimize,
    Print,
}

/// The primitive editor of the Design step.
#[derive(Default)]
struct EditUi {
    sel: Option<usize>,
    undo: Vec<Vec<Prim>>,
    redo: Vec<Vec<Prim>>,
    /// Edits within a short time of each other (a slider or gizmo drag) are one undo step.
    last_edit: Option<Instant>,
    drag: Option<GizmoDrag>,
    /// a mirror plane's position while it is being dragged or typed (applied when that ends)
    mirror_draft: [Option<f32>; 3],
}

/// A gizmo handle being dragged.
struct GizmoDrag {
    /// 0..3: along X/Y/Z; 3: in the view plane
    handle: usize,
    start: [f32; 3],
    /// pointer movement since the drag started, in points
    moved: egui::Vec2,
}

/// Screen-space layout of the gizmo of the selected primitive.
struct Gizmo {
    centre: egui::Pos2,
    /// tips of the X, Y, Z arrows
    tips: [egui::Pos2; 3],
    /// world length of an arrow
    len: f32,
    /// world units per point in the view plane at the primitive's depth
    per_point: f32,
}

const AXIS_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(230, 80, 80),
    egui::Color32::from_rgb(90, 200, 100),
    egui::Color32::from_rgb(90, 140, 255),
];

/// A "nice" size near `v` (1, 2 or 5 times a power of ten).
fn nice(v: f32) -> f32 {
    let e = 10f32.powf(v.max(1e-3).log10().floor());
    let m = v / e;
    e * if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    }
}

/// Three drag values side by side; true if any changed.
fn drag3(ui: &mut egui::Ui, v: &mut [f32; 3], speed: f64, suffix: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for (k, c) in v.iter_mut().enumerate() {
            let r = ui.add(
                egui::DragValue::new(c)
                    .speed(speed)
                    .max_decimals(2)
                    .suffix(suffix)
                    .prefix(["x ", "y ", "z "][k]),
            );
            changed |= r.changed();
        }
    });
    changed
}

/// Settings of the Optimize step (`gen3d optimize` flags).
struct OptUi {
    volfrac: f64,
    iters: usize,
    /// None = `topopt::auto_voxel`
    voxel: Option<f32>,
    force: f64,
    rmin: f64,
    /// Heaviside projection (crisp 0/1 designs)
    projection: bool,
    ground: Ground,
    /// overhang control: printable along +z without support material
    overhang: bool,
}

impl Default for OptUi {
    fn default() -> Self {
        OptUi {
            volfrac: 0.3,
            iters: 150,
            voxel: None,
            force: 100.0,
            rmin: 1.5,
            projection: true,
            ground: Ground::None,
            overhang: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Supports {
    /// whatever the profile says
    Profile,
    Off,
    Everywhere,
    BuildPlate,
}

/// Settings of the Print step.
struct PrintUi {
    /// mesh voxel for a .wgsl design; None = the shape's `//! voxel`
    voxel: Option<f32>,
    /// keep this fraction of triangles; None = the shape's `//! decimate` (or all)
    decimate: Option<f32>,
    /// Taubin smoothing passes for a grid's mesh; None = the default (`mesher::DEFAULT_SMOOTH`)
    smooth: Option<u32>,
    /// mm a grid's faired surface may grow outward to be smoother (0 = only the voxels' half voxel)
    grow: f32,
    /// drop pieces and voids under this % of the largest body (0 = keep all)
    min_body: f64,
    /// show the exported mesh in the viewport instead of the raymarched shape
    show_mesh: bool,
    /// with `show_mesh`: draw the triangle edges
    edges: bool,
    three_mf: bool,
    /// None = automatic (the shape's `//! slicer`, else printers/default.ini, else slicer defaults)
    profile: Option<PathBuf>,
    profiles: Vec<PathBuf>,
    /// PrusaSlicer overrides on top of the profile; None = keep the profile's value
    layer: Option<f32>,
    infill: Option<u32>,
    supports: Supports,
}

impl Default for PrintUi {
    fn default() -> Self {
        PrintUi {
            voxel: None,
            decimate: None,
            smooth: None,
            grow: 0.0,
            min_body: 0.0,
            show_mesh: false,
            edges: false,
            three_mf: false,
            profile: None,
            profiles: vec![],
            layer: None,
            infill: None,
            supports: Supports::Profile,
        }
    }
}

/// Offscreen target for the preview, shown as an image in the central panel.
struct Viewport {
    view: wgpu::TextureView,
    size: (u32, u32),
    id: egui::TextureId,
}

impl Viewport {
    fn target(gpu: &Gpu, size: (u32, u32)) -> wgpu::TextureView {
        gpu.device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("viewport"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: VIEW_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }
}

/// The OS window and its swapchain; absent when the app draws offscreen (`snapshot`).
struct OsWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui_state: egui_winit::State,
}

/// The exported mesh shown in the Print step (`PrintUi::show_mesh`).
struct ShownMesh {
    path: PathBuf,
    mtime: Option<SystemTime>,
    /// the mesh on the GPU, or why it couldn't be loaded
    gpu: Result<crate::meshview::MeshGpu, String>,
    tris: usize,
}

struct Win {
    os: Option<OsWindow>,
    gpu: Arc<Gpu>,
    ubuf: wgpu::Buffer,
    /// The shown shape's preview shader; kept so parameter changes only rebuild the pipeline.
    module: Option<wgpu::ShaderModule>,
    pipeline: Option<Pipeline>,
    vol: Option<VolGpu>,
    /// the editor primitives' `g3d_ed` buffer of the current pipeline
    ed: Option<wgpu::Buffer>,
    /// the live (`f32`) parameters' buffer, and their names in buffer order
    par: Option<wgpu::Buffer>,
    live: Vec<String>,
    egui: egui::Context,
    renderer: egui_wgpu::Renderer,
    viewport: Viewport,
    title: String,
    /// mesh drawing, made when a mesh is first shown
    meshr: Option<crate::meshview::MeshRenderer>,
    mesh: Option<ShownMesh>,
    /// The viewport image is out of date: something it shows changed on the GPU (pipeline,
    /// parameters, primitives, volume, mesh, target size). Changes to the view uniforms (camera,
    /// options) are found by comparing them with `drawn`.
    dirty: std::cell::Cell<bool>,
    /// the view uniforms of the image in the viewport target
    drawn: Vec<u8>,
    /// the same without the animation time: when these change, the view itself changed
    drawn_still: Vec<u8>,
}

impl Win {
    /// GPU state for the preview and egui, drawing into targets of `format`.
    fn new(gpu: Arc<Gpu>, format: wgpu::TextureFormat, os: Option<OsWindow>) -> Win {
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<crate::viewer::ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut renderer = egui_wgpu::Renderer::new(&gpu.device, format, Default::default());
        let view = Viewport::target(&gpu, (1, 1));
        let id = renderer.register_native_texture(&gpu.device, &view, wgpu::FilterMode::Linear);
        let egui = match &os {
            Some(o) => o.egui_state.egui_ctx().clone(),
            None => egui::Context::default(),
        };
        Win {
            os,
            gpu,
            ubuf,
            module: None,
            pipeline: None,
            vol: None,
            ed: None,
            par: None,
            live: vec![],
            egui,
            renderer,
            viewport: Viewport { view, size: (1, 1), id },
            title: String::new(),
            meshr: None,
            mesh: None,
            dirty: std::cell::Cell::new(true),
            drawn: vec![],
            drawn_still: vec![],
        }
    }

    /// The viewport image must be redrawn (see `dirty`).
    fn touch(&self) {
        self.dirty.set(true);
    }
}

type Rebuild = std::sync::mpsc::Receiver<(u64, Vec<(String, f64)>, Result<Pipeline>)>;
/// A G-code file's summary as (label, value) rows.
type Summary = Vec<(String, String)>;

struct App {
    /// The opened file: a .wgsl design, or a .vti grid opened directly. None until one is opened.
    design: Option<PathBuf>,
    /// The file the viewport shows: the design, or its optimization result.
    shown: PathBuf,
    step: Step,
    /// In the Print step: print the optimization result instead of the design.
    print_result: bool,
    /// Parameter values that differ from the design's defaults.
    defines: Vec<(String, f64)>,
    /// How to show a grid the next time one loads (kept while the design is shown instead).
    vopts: VolumeOpts,
    volume: Option<Volume>,
    /// Threshold being dragged; applied on release (recomputing is CPU work).
    iso_drag: Option<f32>,
    /// The .wgsl design, its fields and parameters; kept while the result is shown.
    design_shape: Option<Shape>,
    design_fields: Vec<Field>,
    params: Vec<Param>,
    edit: EditUi,
    opt: OptUi,
    print: PrintUi,
    /// the side of the part that faces up on the printer: the build direction of Optimize's overhang
    /// control, and how Print turns the mesh (`--up`)
    up: Up,
    /// Optimize in 2D, projected along this axis (`--2d`); None: 3D
    plane2d: Option<usize>,
    slicer: Option<(slicer::Kind, PathBuf)>,
    browser: Browser,
    /// What the viewport shows (the design's shape, or a grid's generated shape).
    shape: Shape,
    fields: Vec<Field>,
    /// overlay mode: 0 = all fields, 1..=n = only field n-1, n+1 = none
    field_mode: usize,
    start: Instant,
    loaded: bool,
    loaded_at: String,
    mtimes: HashMap<PathBuf, Option<SystemTime>>,
    error: Option<String>,
    /// Pipeline build failure for the current parameter values (the source itself is fine).
    param_error: Option<String>,
    notice: Option<String>,
    params_dirty: bool,
    /// In-flight background pipeline build (see `request_rebuild`).
    rebuild: Option<Rebuild>,
    rebuild_again: bool,
    /// Bumped on every reload so stale background builds are dropped.
    rebuild_gen: u64,
    /// Background pipeline builds started (for tests).
    rebuilds: u32,
    /// The `override` parameter values the current (or pending) pipeline is built with.
    built_defines: Vec<(String, f64)>,
    cam: Camera,
    opts: Options,
    /// Fraction of the viewport's pixels that are raymarched (lower = faster).
    render_scale: f32,
    /// Viewport size wanted by this frame's layout, in physical pixels.
    view_px: (u32, u32),
    /// Where the viewport is, in points.
    view_rect: egui::Rect,
    /// `gen3d optimize` runs
    opt_job: Slot,
    /// export / slice / open runs
    out_job: Slot,
    gcode_cache: HashMap<PathBuf, (Option<SystemTime>, Summary)>,
    gpu_name: String,
    /// `gen3d optimize` will use the GPU solver (a hardware GPU): sets the automatic element size
    gpu_solver: bool,
    win: Option<Win>,
    last_poll: Instant,
    /// when the next frame is due (None: only on input or a change)
    next_frame: Option<Instant>,
    /// the preview animates until then (see `animating`)
    anim_until: Option<Instant>,
    /// viewport renders since `fps_t`, and their rate (shown while the viewport redraws)
    preview_frames: u32,
    fps: f32,
    fps_t: Instant,
}

/// Frame rate of the preview's animation (the load bands move 1.5 bands per second, so this looks as
/// smooth as the display's refresh rate at a fraction of the GPU time).
const ANIMATION_FPS: f32 = 30.0;

/// How long the preview animates after the view changes (camera, options, overlay, reload); then it
/// freezes, so an idle window doesn't keep the GPU busy.
const ANIMATION_SECS: f32 = 3.0;

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

fn stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn fmt_num(v: f64) -> String {
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn file_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().to_string()
}

fn define_list(defines: &[(String, f64)]) -> String {
    defines
        .iter()
        .map(|(k, v)| format!("{k} = {}", fmt_num(*v)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Viridis colour map (same polynomial as the shader), as sRGB.
fn viridis(t: f32) -> egui::Color32 {
    const C: [[f32; 3]; 7] = [
        [0.277_727_3, 0.005_407_345, 0.334_099_8],
        [0.105_093_04, 1.404_613_5, 1.384_590_2],
        [-0.330_861_83, 0.214_847_56, 0.095_095_16],
        [-4.634_230_5, -5.799_101, -19.332_441],
        [6.228_27, 14.179_933, 56.690_55],
        [4.776_385, -13.745_145, -65.353_035],
        [-5.435_456, 4.645_852_6, 26.312_435],
    ];
    let t = t.clamp(0.0, 1.0);
    let ch = |k: usize| {
        let v = C.iter().rev().fold(0.0, |acc, c| acc * t + c[k]);
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    egui::Color32::from_rgb(ch(0), ch(1), ch(2))
}

/// Histogram of a volume's array (log-scaled counts) with the threshold marked; bars on the inside
/// side are drawn in the surface colour. Returns the response and the value under the pointer.
fn histogram(ui: &mut egui::Ui, hist: &[u32], range: (f32, f32), iso: f32, below: bool) -> (egui::Response, f32) {
    let (lo, hi) = range;
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 64.0), egui::Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    let span = (hi - lo).max(f32::MIN_POSITIVE);
    let x_of = |v: f32| rect.left() + ((v - lo) / span).clamp(0.0, 1.0) * rect.width();
    let max = hist.iter().copied().max().unwrap_or(0) as f32;
    let n = hist.len().max(1) as f32;
    let bw = rect.width() / n;
    let weak = ui.visuals().weak_text_color().gamma_multiply(0.6);
    for (i, &c) in hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let h = (1.0 + c as f32).ln() / (1.0 + max).ln() * (rect.height() - 6.0);
        let centre = lo + (i as f32 + 0.5) / n * span;
        let inside = if below { centre < iso } else { centre > iso };
        let x0 = rect.left() + i as f32 * bw;
        let bar = egui::Rect::from_min_max(
            egui::pos2(x0 + 0.5, rect.bottom() - 2.0 - h),
            egui::pos2((x0 + bw - 0.5).max(x0 + 1.0), rect.bottom() - 2.0),
        );
        p.rect_filled(bar, 0.0, if inside { SURFACE_ORANGE } else { weak });
    }
    let xi = x_of(iso);
    p.line_segment(
        [egui::pos2(xi, rect.top()), egui::pos2(xi, rect.bottom())],
        egui::Stroke::new(2.0, ui.visuals().strong_text_color()),
    );
    let value_at = |x: f32| lo + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) * span;
    let hover = resp.hover_pos().map(|p| value_at(p.x));
    let resp = resp.on_hover_text(match hover {
        Some(v) => format!("{}  ·  click or drag to set the threshold", fmt_num(v as f64)),
        None => "click or drag to set the threshold".into(),
    });
    let picked = resp.interact_pointer_pos().map_or(f32::NAN, |p| value_at(p.x));
    ui.horizontal(|ui| {
        ui.weak(fmt_num(lo as f64));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(fmt_num(hi as f64));
        });
    });
    (resp, picked)
}

/// A horizontal viridis bar with the range's ends written under it.
fn colour_bar(ui: &mut egui::Ui, range: (f32, f32)) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 12.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    let n = 48;
    for i in 0..n {
        let x0 = rect.left() + rect.width() * i as f32 / n as f32;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / n as f32;
        let r = egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1 + 0.5, rect.bottom()));
        p.rect_filled(r, 0.0, viridis((i as f32 + 0.5) / n as f32));
    }
    ui.horizontal(|ui| {
        ui.weak(fmt_num(range.0 as f64));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(fmt_num(range.1 as f64));
        });
    });
}

fn status_dot(ui: &mut egui::Ui, color: egui::Color32) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(r.center(), 4.0, color);
}

fn rgb_gamma(c: [f32; 3]) -> egui::Color32 {
    let [r, g, b] = c.map(|c| (c.powf(1.0 / 2.2) * 255.0) as u8);
    egui::Color32::from_rgb(r, g, b)
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(title).strong().size(15.0));
    ui.add_space(2.0);
}

/// A rough duration: "40 s", "3 min 20 s", "1 h 05 min".
fn fmt_duration(secs: f32) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        // seconds are noise beyond a few minutes
        if s < 600 {
            format!("{} min {} s", s / 60, s % 60 / 10 * 10)
        } else {
            format!("{} min", (s + 30) / 60)
        }
    } else {
        format!("{} h {:02} min", s / 3600, s % 3600 / 60)
    }
}

/// Starting setups for common situations (`App::apply_template`).
#[derive(Clone, Copy)]
enum Template {
    Stand,
    Wall,
    Hang,
}

impl Template {
    const ALL: [Template; 3] = [Template::Stand, Template::Wall, Template::Hang];

    fn label(self) -> &'static str {
        match self {
            Template::Stand => "stands on a table",
            Template::Wall => "sticks out of a wall",
            Template::Hang => "hangs from above",
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Template::Stand => {
                "the ground holds the bottom; a 1 kg item rests on top and the part gets rocked 20% \
                 sideways"
            }
            Template::Wall => "the low-X side is screwed to a wall; 2 kg hangs on the top of the far end",
            Template::Hang => "the top is held; 2 kg hangs from the bottom",
        }
    }

    fn added(self) -> &'static str {
        match self {
            Template::Stand => "a load region on top (and set the ground)",
            Template::Wall => "a fixed wall region and a load region",
            Template::Hang => "a fixed region on top and a load region at the bottom",
        }
    }
}

/// A colour per load case.
fn case_color(case: u32) -> egui::Color32 {
    const COLORS: [egui::Color32; 4] = [
        egui::Color32::from_rgb(255, 120, 90),
        egui::Color32::from_rgb(240, 200, 80),
        egui::Color32::from_rgb(200, 120, 240),
        egui::Color32::from_rgb(90, 210, 210),
    ];
    COLORS[(case.max(1) as usize - 1) % COLORS.len()]
}

/// The Optimize step's summary of the editor primitives' supports and loads, by load case.
/// Which side of the part faces up on the printer (shared by Optimize and Print).
fn up_ui(ui: &mut egui::Ui, up: &mut Up, id: &str) {
    let label = |u: Up| format!("{} up", u.name());
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(*up))
        .show_ui(ui, |ui| {
            for u in Up::ALL {
                ui.selectable_value(up, u, label(u));
            }
        })
        .response
        .on_hover_text(
            "the side of the part that faces up on the printer. Optimize keeps the part printable that \
             way up; Print turns the mesh so it is (and sets it on the plate).",
        );
}

fn supports_and_loads_ui(ui: &mut egui::Ui, prims: &[Prim], ground: Ground, bounds: ([f32; 3], [f32; 3])) {
    let fixed: Vec<String> = prims
        .iter()
        .filter(|p| !p.hidden && p.role == Role::Fixed)
        .map(|p| match p.fix_mask() {
            7 => p.name.clone(),
            m => {
                let axes: String = (0..3).filter(|c| m >> c & 1 == 1).map(|c| ['X', 'Y', 'Z'][c]).collect();
                format!("{} (holds {})", p.name, if axes.is_empty() { "nothing" } else { &axes })
            }
        })
        .collect();
    let loads = prims::load_specs(prims);
    if fixed.is_empty() && loads.is_empty() && ground == Ground::None {
        return;
    }
    section(ui, "Supports & loads");
    egui::Grid::new("bcs").num_columns(2).show(ui, |ui| {
        if ground != Ground::None {
            ui.label("ground");
            ui.label(match ground {
                Ground::Slide => "the bottom rests on it, free to slide",
                _ => "the bottom rests on it, gripped",
            });
            ui.end_row();
        }
        if !fixed.is_empty() {
            ui.label("fixed");
            ui.label(fixed.join(", "));
            ui.end_row();
        }
        let mut last = None;
        for l in &loads {
            let case = l.label.clone().unwrap_or_else(|| l.case.to_string());
            ui.label(if last.as_ref() != Some(&case) {
                format!("case {case}")
            } else {
                String::new()
            });
            let d = l.dir.unwrap_or_default();
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
            let f = l.force.unwrap_or(0.0);
            let along = [0, 1, 2].map(|c| fmt_num((d[c] / len * 100.0).round() as f64 / 100.0));
            ui.label(format!(
                "{}: {} N along ({}, {}, {})",
                l.name,
                fmt_num((f * 10.0).round() / 10.0),
                along[0],
                along[1],
                along[2]
            ));
            ui.end_row();
            last = Some(case);
        }
    });
    let ncases = {
        let mut c: Vec<u32> = loads.iter().map(|l| l.case).collect();
        c.dedup();
        c.len()
    };
    if ncases > 1 {
        ui.weak(format!(
            "{ncases} load cases: the part is made stiff for each (one solve per case)."
        ));
    }
    for w in bc_warnings(prims, bounds) {
        ui.colored_label(ui.visuals().warn_fg_color, w);
    }
    ui.weak("Set in Design (select a primitive: Optimize as).");
}

/// Mistakes in the supports and loads that show before running.
fn bc_warnings(prims: &[Prim], (lo, hi): ([f32; 3], [f32; 3])) -> Vec<String> {
    let mut out = vec![];
    for p in prims.iter().filter(|p| !p.hidden && p.role.is_bc()) {
        let (a, b) = p.aabb();
        if (0..3).any(|k| b[k] < lo[k] || a[k] > hi[k]) {
            out.push(format!(
                "`{}` is outside the shape, so it holds or carries nothing.",
                p.name
            ));
        }
        match p.role {
            Role::Fixed if p.fix_mask() == 0 => out.push(format!(
                "`{}` holds no direction: tick at least one of X, Y, Z.",
                p.name
            )),
            Role::Load if p.force <= 0.0 => out.push(format!("`{}` has no force.", p.name)),
            Role::Load if p.load.iter().all(|v| *v == 0.0) => out.push(format!("`{}` has no direction.", p.name)),
            _ => {}
        }
    }
    out
}

/// Plain-language steps for defining an optimization problem (Optimize step).
fn optimize_howto(ui: &mut egui::Ui) {
    let steps = [
        (
            "The shape is the space to fill.",
            "Design the shape as the largest part that could fit. The optimizer keeps only some of \
             that space as material and carves the rest away.",
        ),
        (
            "Say what holds the part.",
            "Standing on a table? Pick \"Stands on the ground\" above. Screwed or clamped? In Design, \
             select a primitive at that spot (e.g. a bolt hole) and set its \"optimize as\" to fixed.",
        ),
        (
            "Say what it carries.",
            "Select the primitive where the load sits and set \"optimize as\" to load. For something \
             resting on it, give its weight in kg; otherwise a force in newtons and a direction. A \
             subtracted primitive set to load puts the load on the face it cuts.",
        ),
        (
            "Optional: rocking and other situations.",
            "Tick rocking on a load if the part also gets bumped or tipped sideways. Loads with \
             different case numbers are separate situations; the part is made stiff for each.",
        ),
        (
            "Optional: keep spots solid.",
            "Set \"optimize as\" to keep solid where material must stay (e.g. a mounting face). To \
             mark a spot without changing the shape, set the primitive's operation to region.",
        ),
        (
            "Pick the amount of material and run.",
            "Set material below (e.g. 30% of the shape) and click Optimize. When it finishes, move \
             the threshold until the part looks right, then go to Print.",
        ),
    ];
    for (i, (title, text)) in steps.iter().enumerate() {
        ui.horizontal_top(|ui| {
            ui.strong(format!("{}.", i + 1));
            ui.vertical(|ui| {
                ui.strong(*title);
                ui.label(*text);
            });
        });
        ui.add_space(3.0);
    }
    ui.weak(
        "Script shapes can define `fn field_fixed(p: vec3f) -> f32` (< 0 where held) and \
         `fn field_load(p: vec3f) -> vec3f` instead; see shapes/bracket_domain.wgsl.",
    );
}

/// A primary action button (filled with the selection colour).
fn primary(ui: &mut egui::Ui, enabled: bool, text: &str) -> egui::Response {
    let fill = ui.visuals().selection.bg_fill;
    ui.add_enabled(
        enabled,
        egui::Button::new(egui::RichText::new(text).strong().color(egui::Color32::WHITE))
            .fill(fill)
            .min_size(egui::vec2(0.0, 26.0)),
    )
}

/// Readable names for the G-code comments `slicer::gcode_summary` picks out.
fn pretty_summary(lines: &[String]) -> Vec<(String, String)> {
    lines
        .iter()
        .filter_map(|l| {
            let (k, v) = l.split_once(" = ")?;
            let v = v.trim();
            Some(match k.trim() {
                "estimated printing time (normal mode)" => ("print time".into(), v.into()),
                "filament used [g]" => ("filament".into(), format!("{v} g")),
                "filament used [mm]" => (
                    "filament length".into(),
                    v.parse::<f64>().map_or(v.into(), |mm| format!("{:.2} m", mm / 1000.0)),
                ),
                "layer_height" => ("layer height".into(), format!("{v} mm")),
                "fill_density" => ("infill".into(), v.into()),
                "support_material" => ("supports".into(), if v == "1" { "on" } else { "off" }.into()),
                "nozzle_diameter" => ("nozzle".into(), format!("{v} mm")),
                "printer_model" if !v.is_empty() => ("printer".into(), v.into()),
                _ => return None,
            })
        })
        .collect()
}

/// Show a job's status line (with Stop while it runs) and its log.
fn job_ui(ui: &mut egui::Ui, slot: &Slot, j: &Job) {
    ui.horizontal_wrapped(|ui| {
        if j.running {
            ui.spinner();
            ui.label(format!("{} running… {:.1} s", j.cmd, j.elapsed()));
            if ui
                .small_button("Stop")
                .on_hover_text("stop the job (keeps what it wrote)")
                .clicked()
            {
                slot.stop();
            }
        } else if j.stopped {
            status_dot(ui, ui.visuals().warn_fg_color);
            ui.label(format!("{} stopped after {:.1} s", j.cmd, j.elapsed()));
        } else if j.ok == Some(true) {
            status_dot(ui, OK_GREEN);
            ui.label(format!("{} finished in {:.1} s", j.cmd, j.elapsed()));
        } else {
            status_dot(ui, ui.visuals().error_fg_color);
            ui.colored_label(ui.visuals().error_fg_color, format!("{} failed", j.cmd));
        }
    });
    if j.log.is_empty() {
        return;
    }
    egui::CollapsingHeader::new("Log")
        .id_salt(("log", &j.cmd))
        .default_open(j.ok == Some(false) && !j.stopped)
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("gen3d {} {}", j.cmd, j.args))
                        .weak()
                        .small(),
                )
                .wrap(),
            );
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .stick_to_bottom(true)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(egui::RichText::new(j.log.trim_end()).monospace().small()).selectable(true),
                    );
                });
        });
}

/// One line about an output file: name, size, age, whether it's older than its inputs, and a
/// button to show it in the file manager. Returns whether the file exists.
fn file_line(ui: &mut egui::Ui, path: &Path, inputs: &[PathBuf], notice: &mut Option<String>) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        ui.weak(format!("{} not made yet", file_name(path)));
        return false;
    };
    let t = meta.modified().ok();
    let stale = inputs.iter().any(|i| mtime(i) > t);
    ui.horizontal_wrapped(|ui| {
        if ui
            .small_button("Show")
            .on_hover_text("open the folder in your file manager")
            .clicked()
            && let Err(e) = slicer::open_with_desktop(path.parent().unwrap_or(Path::new(".")))
        {
            *notice = Some(format!("{e:#}"));
        }
        ui.monospace(file_name(path)).on_hover_text(path.display().to_string());
        ui.weak(format!("{:.1} MB · {}", meta.len() as f64 / 1e6, ago(t)));
    });
    if stale {
        ui.colored_label(ui.visuals().warn_fg_color, "older than its source: make it again");
    }
    true
}

impl App {
    fn new(cwd: PathBuf) -> App {
        App {
            design: None,
            shown: cwd.clone(),
            step: Step::Design,
            print_result: false,
            defines: vec![],
            vopts: VolumeOpts::default(),
            volume: None,
            iso_drag: None,
            design_shape: None,
            design_fields: vec![],
            params: vec![],
            edit: EditUi::default(),
            opt: OptUi::default(),
            print: PrintUi::default(),
            up: Up::PlusZ,
            plane2d: None,
            slicer: slicer::find_slicer(None),
            browser: Browser::new(cwd.clone()),
            shape: Shape::placeholder(&cwd),
            fields: vec![],
            field_mode: 0,
            start: Instant::now(),
            loaded: false,
            loaded_at: String::new(),
            mtimes: HashMap::new(),
            error: None,
            param_error: None,
            notice: None,
            params_dirty: false,
            rebuild: None,
            rebuild_again: false,
            rebuild_gen: 0,
            rebuilds: 0,
            built_defines: vec![],
            cam: Camera {
                yaw: 0.0,
                pitch: 0.0,
                dist: 1.0,
                target: [0.0; 3],
            },
            opts: Options {
                clip: true,
                section: false,
                cut: 0.0,
                show_bounds: true,
                fields: 0,
                time: 0.0,
            },
            render_scale: 1.0,
            view_px: (1, 1),
            view_rect: egui::Rect::NOTHING,
            opt_job: Slot::default(),
            out_job: Slot::default(),
            gcode_cache: HashMap::new(),
            gpu_name: String::new(),
            gpu_solver: true,
            win: None,
            last_poll: Instant::now(),
            next_frame: None,
            anim_until: None,
            preview_frames: 0,
            fps: 0.0,
            fps_t: Instant::now(),
        }
    }

    // ------------------------------------------------------------------------------ files & state

    /// Switch to another design file; everything tied to the old one is reset.
    fn open_design(&mut self, path: &Path, defines: Vec<(String, f64)>) {
        let path = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                self.notice = Some(format!("cannot open {}: {e}", path.display()));
                return;
            }
        };
        if let Some(dir) = path.parent() {
            self.browser.dir = dir.to_path_buf();
        }
        self.design = Some(path.clone());
        self.shown = path;
        self.step = Step::Design;
        self.print_result = false;
        self.up = Up::PlusZ;
        self.plane2d = None;
        self.defines = defines;
        self.vopts = VolumeOpts::default();
        self.volume = None;
        self.design_shape = None;
        self.design_fields.clear();
        self.params.clear();
        self.edit = EditUi::default();
        self.fields.clear();
        self.field_mode = 0;
        self.loaded = false;
        self.error = None;
        self.param_error = None;
        self.notice = None;
        self.opt.voxel = None;
        self.print.voxel = None;
        self.print.decimate = None;
        self.print.profile = None;
        self.mtimes.clear();
        if let Some(win) = &mut self.win {
            win.pipeline = None;
            win.module = None;
            win.touch();
            self.reload(true);
        }
    }

    fn design_is_grid(&self) -> bool {
        self.design.as_deref().is_some_and(is_vti)
    }

    /// Where the optimization result for the current design and parameter values goes.
    fn result_path(&self) -> Option<PathBuf> {
        let d = self.design.as_ref().filter(|d| !is_vti(d))?;
        Some(crate::topopt_out(d, &self.defines))
    }

    fn existing_result(&self) -> Option<PathBuf> {
        self.result_path().filter(|r| r.exists())
    }

    /// Whether the design defines the fields `gen3d optimize` needs.
    fn optimizable(&self) -> bool {
        let has = |n: &str, c: usize| self.design_fields.iter().any(|f| f.name == n && f.comps == c);
        (has("fixed", 1) || self.opt.ground != Ground::None) && has("load", 3)
    }

    /// The design's files on disk (the .wgsl and its includes).
    fn design_files(&self) -> Vec<PathBuf> {
        match &self.design_shape {
            Some(s) => s.files.iter().filter(|f| f.exists()).cloned().collect(),
            None => self.design.iter().cloned().collect(),
        }
    }

    /// The file the viewport should show for the current step.
    fn wanted_shown(&self) -> Option<PathBuf> {
        let design = self.design.clone()?;
        Some(match (self.step, self.existing_result()) {
            (Step::Optimize, Some(r)) => r,
            (Step::Print, Some(r)) if self.print_result => r,
            _ => design,
        })
    }

    fn sync_shown(&mut self) {
        let Some(want) = self.wanted_shown() else { return };
        if want != self.shown {
            if self.volume.is_some() {
                self.vopts = self.current_vopts();
            }
            self.shown = want;
            self.reload(false);
        }
    }

    fn set_step(&mut self, step: Step) {
        if self.step == Step::Optimize && step == Step::Print && self.existing_result().is_some() {
            self.print_result = true;
        }
        self.step = step;
        if step == Step::Print {
            self.print.profiles = slicer::profiles();
            self.slicer = slicer::find_slicer(None);
        }
        self.sync_shown();
    }

    /// Options that reproduce the current grid view, so a reloaded .vti keeps them.
    fn current_vopts(&self) -> VolumeOpts {
        match &self.volume {
            Some(v) => VolumeOpts {
                array: Some(v.array_name().to_string()),
                iso: Some(v.iso),
                below: Some(v.below),
                blocky: v.blocky,
                color: v.color_name().map(str::to_string),
                color_range: v.color_range,
            },
            None => self.vopts.clone(),
        }
    }

    fn reload(&mut self, first: bool) {
        if self.design.is_none() {
            return;
        }
        let path = self.shown.clone();
        let grid = is_vti(&path);
        let defines = if grid { vec![] } else { self.defines.clone() };
        // a grid caught mid-write, or whose remembered array is gone: fall back to the defaults
        let src = load_source(&path, &defines, &self.current_vopts()).or_else(|e| {
            if grid {
                load_source(&path, &[], &VolumeOpts::default())
            } else {
                Err(e)
            }
        });
        let src = match src {
            Ok(s) => s,
            Err(e) => {
                self.mtimes = HashMap::from([(path.clone(), mtime(&path))]);
                if !grid {
                    self.volume = None;
                }
                return self.fail(&format!("{e:#}"));
            }
        };
        self.mtimes = src
            .shape
            .files
            .iter()
            .filter(|f| f.exists())
            .map(|f| (f.clone(), mtime(f)))
            .collect();
        self.install(src, first);
    }

    /// Build the GPU pipeline for `src` and switch to it.
    fn install(&mut self, mut src: Source, first: bool) {
        let grid = src.volume.is_some();
        let mut params = vec![];
        if !grid {
            params = match src.shape.params() {
                Ok(p) => p,
                Err(e) => return self.fail(&e.to_string()),
            };
            // values for parameters the shape doesn't (or no longer) declare are dropped
            let unknown: Vec<String> = self
                .defines
                .iter()
                .filter(|(k, _)| !params.iter().any(|p| &p.name == k))
                .map(|(k, _)| k.clone())
                .collect();
            if !unknown.is_empty() {
                self.defines.retain(|(k, _)| !unknown.contains(k));
                let names: Vec<_> = params.iter().map(|p| p.name.as_str()).collect();
                let msg = format!(
                    "ignored {}: not a parameter of this shape (parameters: {})",
                    unknown.join(", "),
                    if names.is_empty() {
                        "none".into()
                    } else {
                        names.join(", ")
                    }
                );
                println!("[{}] {msg}", stamp());
                self.notice = Some(msg);
            }
            src.shape.defines = self.defines.clone();
            if self.edit.sel.is_some_and(|i| i >= src.shape.prims.len()) {
                self.edit.sel = None;
            }
        }
        let ed_data = src.shape.editor_data(self.edit.sel);
        let defines = self.defines.clone();
        let Some(win) = &mut self.win else { return };
        let vol = match src.volume.as_ref().map(|v| VolGpu::new(&win.gpu, v)).transpose() {
            Ok(v) => v,
            Err(e) => return self.fail(&format!("{e:#}")),
        };
        let built = preview_module(&win.gpu, &src).and_then(|(m, live)| {
            let bindings = Bindings {
                ubuf: win.ubuf.clone(),
                vol,
                par: (!live.is_empty()).then(|| {
                    let data = crate::shape::param_data(&live, &params, &defines);
                    uniform_buffer(&win.gpu, "parameters", &data)
                }),
                ed: ed_data.map(|d| uniform_buffer(&win.gpu, "editor primitives", &d)),
            };
            let constants = compiled_defines(&src.shape.defines, &live);
            let p = build_pipeline(&win.gpu, &m, &constants, VIEW_FORMAT, &bindings)?;
            Ok((m, p, bindings, live, constants))
        });
        match built {
            Ok((m, p, b, live, constants)) => {
                win.module = Some(m);
                win.pipeline = Some(p);
                win.touch();
                win.vol = b.vol;
                win.ed = b.ed;
                win.par = b.par;
                win.live = live;
                self.built_defines = constants;
                self.rebuild_gen += 1;
                self.param_error = None;
            }
            Err(e) => return self.fail(&e.to_string()),
        }
        let Source {
            shape, fields, volume, ..
        } = src;
        // the camera stays put when switching between the design and its result, and when an
        // optimizer rewrites the result; it reframes when a file's own bounds change (not when
        // primitives grow them)
        let reframe =
            first || (self.loaded && self.shape.path == shape.path && self.shape.base_bounds != shape.base_bounds);
        if !grid {
            self.params = params;
            // the shape's saved print orientation (a script's without one keeps the session's)
            if let Some(u) = shape.up {
                self.up = u;
            }
            if shape.plane2d.is_some() {
                self.plane2d = shape.plane2d;
            }
            self.design_shape = Some(shape.clone());
            self.design_fields = fields.clone();
        }
        self.volume = volume;
        self.shape = shape;
        let names = |f: &[Field]| f.iter().map(|f| f.name.clone()).collect::<Vec<_>>();
        if names(&fields) != names(&self.fields) {
            self.field_mode = 0;
        }
        self.fields = fields;
        self.set_field_mask();
        self.loaded = true;
        self.error = None;
        self.loaded_at = stamp();
        if reframe {
            self.reset_camera();
        }
        println!("[{}] loaded {}", self.loaded_at, file_name(&self.shown));
    }

    fn fail(&mut self, msg: &str) {
        println!("\n[{}] \x1b[31merror\x1b[0m\n{}\n", stamp(), msg.trim());
        self.error = Some(msg.trim().to_string());
    }

    /// Apply a change to the grid view: `rebuild` when the array (and so the grid size) changed.
    fn edit_volume(&mut self, rebuild: bool, f: impl FnOnce(&mut Volume)) {
        let Some(mut v) = self.volume.take() else { return };
        f(&mut v);
        if rebuild {
            let src = volume_source(&self.shown, v);
            return self.install(src, false);
        }
        if let Some(win) = &self.win
            && let Some(g) = &win.vol
        {
            g.upload(&win.gpu, &v);
            win.touch();
        }
        self.volume = Some(v);
    }

    /// Apply a change that only touches the grid's uniforms (smooth/blocky, colour range).
    fn edit_volume_uniforms(&mut self, f: impl FnOnce(&mut Volume)) {
        let Some(v) = &mut self.volume else { return };
        f(v);
        if let Some(win) = &self.win
            && let Some(g) = &win.vol
        {
            g.upload_uniforms(&win.gpu, v);
            win.touch();
        }
    }

    /// Rebuild the pipeline with the current parameter values. The driver's shader compile takes
    /// ~100 ms, so it runs on a worker thread; changes made meanwhile are coalesced into one more
    /// build, so dragging a slider keeps the UI smooth and the preview follows as fast as it can.
    fn request_rebuild(&mut self) {
        self.shape.defines = self.defines.clone();
        if self.volume.is_some() {
            return; // the result is shown; the design rebuilds when it's shown again
        }
        if self.rebuild.is_some() {
            self.rebuild_again = true;
            return;
        }
        let Some(win) = &self.win else { return };
        let Some(module) = win.module.clone() else { return };
        let (gpu, ubuf, defines, generation) = (
            win.gpu.clone(),
            win.ubuf.clone(),
            self.defines.clone(),
            self.rebuild_gen,
        );
        let bindings = Bindings {
            ubuf,
            vol: None,
            par: win.par.clone(),
            ed: win.ed.clone(),
        };
        let constants = compiled_defines(&defines, &win.live);
        self.built_defines = constants.clone();
        self.rebuilds += 1;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = build_pipeline(&gpu, &module, &constants, VIEW_FORMAT, &bindings);
            let _ = tx.send((generation, defines, result));
        });
        self.rebuild = Some(rx);
    }

    /// Show the current parameter values: live (`f32`) ones are written to their uniform buffer;
    /// the pipeline is rebuilt only if an `override` one (`i32`/`u32`/`bool`) changed.
    fn sync_params(&mut self) {
        self.shape.defines = self.defines.clone();
        if self.volume.is_some() {
            return; // the result is shown; the design rebuilds when it's shown again
        }
        let Some(win) = &self.win else { return };
        if let Some(buf) = &win.par {
            let data = crate::shape::param_data(&win.live, &self.params, &self.defines);
            win.gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
            win.touch();
        }
        if compiled_defines(&self.defines, &win.live) != self.built_defines {
            self.request_rebuild();
        }
    }

    fn poll_rebuild(&mut self) {
        let Some(rx) = &self.rebuild else { return };
        let (generation, defines, result) = match rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.rebuild = None;
                return;
            }
        };
        self.rebuild = None;
        // a reload since the request already built the pipeline from the new source
        if generation == self.rebuild_gen {
            match result {
                Ok(p) => {
                    if let Some(win) = &mut self.win {
                        win.pipeline = Some(p);
                        win.touch();
                    }
                    self.param_error = None;
                }
                Err(e) => {
                    let msg = format!("with {}: {e}", define_list(&defines));
                    println!("\n[{}] \x1b[31mpipeline error\x1b[0m\n{msg}\n", stamp());
                    self.param_error = Some(msg);
                }
            }
        }
        if std::mem::take(&mut self.rebuild_again) {
            self.request_rebuild();
        }
    }

    // ------------------------------------------------------------------------- primitive editor

    /// A design with nothing in it yet: no script `map` and no visible added primitive.
    fn is_empty_design(&self) -> bool {
        self.design_shape
            .as_ref()
            .is_some_and(|s| s.editor && !s.script_map && !s.prims.iter().any(|p| !p.hidden && p.op == Op::Union))
    }

    /// Whether the viewport shows the .wgsl design, so its primitives can be edited.
    fn can_edit(&self) -> bool {
        self.step == Step::Design
            && self.loaded
            && self.volume.is_none()
            && self.design_shape.is_some()
            && Some(&self.shown) == self.design.as_ref()
    }

    fn prims(&self) -> Vec<Prim> {
        match &self.design_shape {
            Some(s) if s.editor => s.prims.clone(),
            _ => vec![],
        }
    }

    fn select(&mut self, sel: Option<usize>) {
        if sel != self.edit.sel {
            self.edit.sel = sel;
            self.upload_prims();
        }
    }

    /// Write the primitives' values and the selection into the preview's `g3d_ed` buffer.
    fn upload_prims(&self) {
        if let (Some(win), Some(data)) = (&self.win, self.shape.editor_data(self.edit.sel))
            && let Some(buf) = &win.ed
            && buf.size() == (data.len() * 16) as u64
        {
            win.gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
            win.touch();
        }
    }

    /// Make `new` the primitives, as one undo step (or part of the current one, while a drag goes on).
    fn set_prims(&mut self, new: Vec<Prim>) {
        let old = self.prims();
        if new == old {
            return;
        }
        let structural = new.len() != old.len();
        let recent = self
            .edit
            .last_edit
            .is_some_and(|t| t.elapsed() < Duration::from_millis(750));
        if structural || !recent {
            self.edit.undo.push(old);
            self.edit.redo.clear();
        }
        self.edit.last_edit = (!structural).then(Instant::now);
        self.apply_prims(new);
    }

    /// Save `new` to the sidecar and show it: by changing override constants when only numbers
    /// changed, else by regenerating the source.
    fn apply_prims(&mut self, new: Vec<Prim>) {
        let Some(shape) = &self.design_shape else { return };
        let path = shape.prims_path();
        let old = if shape.editor { shape.prims.clone() } else { vec![] };
        let structural = !shape.editor
            || new.len() != old.len()
            || new
                .iter()
                .zip(&old)
                .any(|(a, b)| a.kind != b.kind || a.hidden != b.hidden || a.role != b.role);
        if let Err(e) = prims::save(&path, &new, &shape.mirror, shape.sidecar_meta()) {
            self.notice = Some(format!("{e:#}"));
            return;
        }
        // our own write: don't hot-reload for it
        self.mtimes.insert(path.clone(), mtime(&path));
        if self.edit.sel.is_some_and(|i| i >= new.len()) {
            self.edit.sel = None;
        }
        if structural {
            self.reload(false);
        } else {
            self.shape.set_prims(new.clone());
            if let Some(s) = &mut self.design_shape {
                s.set_prims(new);
            }
            self.upload_prims();
        }
    }

    /// Keep the chosen print orientation with the shape: in its sidecar (an editor shape), else a
    /// note on the `//! up` directive that would keep it.
    fn save_up(&mut self) {
        let Some(shape) = &self.design_shape else { return };
        if !shape.editor {
            if shape.up != Some(self.up) {
                self.notice = Some(format!(
                    "To keep this orientation, add `//! up {}` to {}.",
                    self.up.name(),
                    shape.path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
            return;
        }
        let path = shape.prims_path();
        let meta = prims::Meta {
            up: Some(self.up),
            ..shape.sidecar_meta()
        };
        if let Err(e) = prims::save(&path, &shape.prims, &shape.mirror, meta) {
            self.notice = Some(format!("{e:#}"));
            return;
        }
        self.mtimes.insert(path.clone(), mtime(&path));
        if let Some(s) = &mut self.design_shape {
            s.up = Some(self.up);
            s.sidecar_up = Some(self.up);
        }
    }

    /// Keep the chosen 2D / 3D design space with the shape, like `save_up`.
    fn save_plane2d(&mut self) {
        let Some(shape) = &self.design_shape else { return };
        if !shape.editor {
            if shape.plane2d != self.plane2d {
                let file = shape.path.file_name().unwrap_or_default().to_string_lossy();
                self.notice = Some(match self.plane2d {
                    Some(a) => format!("To keep 2D, add `//! 2d {}` to {file}.", ["x", "y", "z"][a]),
                    None => format!("To keep 3D, remove the `//! 2d` line from {file}."),
                });
            }
            return;
        }
        let path = shape.prims_path();
        let meta = prims::Meta {
            plane2d: self.plane2d,
            ..shape.sidecar_meta()
        };
        if let Err(e) = prims::save(&path, &shape.prims, &shape.mirror, meta) {
            self.notice = Some(format!("{e:#}"));
            return;
        }
        self.mtimes.insert(path.clone(), mtime(&path));
        if let Some(s) = &mut self.design_shape {
            s.plane2d = self.plane2d;
            s.sidecar_plane2d = self.plane2d;
        }
    }

    /// Save new mirror planes to the sidecar and show them (the generated code changes: a reload).
    fn set_mirror(&mut self, mirror: prims::Mirror) {
        let Some(shape) = &self.design_shape else { return };
        let path = shape.prims_path();
        let prims = if shape.editor { shape.prims.clone() } else { vec![] };
        if let Err(e) = prims::save(&path, &prims, &mirror, shape.sidecar_meta()) {
            self.notice = Some(format!("{e:#}"));
            return;
        }
        self.mtimes.insert(path.clone(), mtime(&path));
        self.reload(false);
    }

    /// The mirror planes: a toggle per axis, and the plane's position when on.
    fn mirror_ui(&mut self, ui: &mut egui::Ui) {
        let Some(shape) = &self.design_shape else { return };
        let before = shape.mirror;
        let (lo, hi) = shape.bounds;
        let mut m = before;
        ui.horizontal(|ui| {
            ui.label("Mirror").on_hover_text(
                "mirror every primitive across a plane: each becomes itself plus its mirror image (holes \
                 too), and the optimizer then solves only one side of the part",
            );
            for (a, name) in ["X", "Y", "Z"].iter().enumerate() {
                let mut on = m[a].is_some();
                if ui
                    .checkbox(&mut on, *name)
                    .on_hover_text(format!("mirror across a plane {name} = const"))
                    .changed()
                {
                    // a new plane goes through the middle of the bounds, rounded to 0.1 mm
                    m[a] = on.then(|| (5.0 * (lo[a] + hi[a])).round() / 10.0);
                }
                if let Some(c) = m[a] {
                    // edited as a draft, applied when the drag or typing ends (each change compiles)
                    let mut v = self.edit.mirror_draft[a].unwrap_or(c);
                    let r = ui
                        .add(egui::DragValue::new(&mut v).speed(0.1).suffix(" mm"))
                        .on_hover_text(format!("position of the {name} mirror plane"));
                    if r.changed() {
                        self.edit.mirror_draft[a] = Some(v);
                    }
                    if !r.dragged()
                        && !r.has_focus()
                        && let Some(d) = self.edit.mirror_draft[a].take()
                    {
                        m[a] = Some(d);
                    }
                }
            }
        });
        if m != before {
            self.set_mirror(m);
        }
    }

    fn undo(&mut self, redo: bool) {
        let current = self.prims();
        let (from, to) = if redo {
            (&mut self.edit.redo, &mut self.edit.undo)
        } else {
            (&mut self.edit.undo, &mut self.edit.redo)
        };
        let Some(prev) = from.pop() else { return };
        to.push(current);
        self.edit.last_edit = None;
        self.apply_prims(prev);
    }

    /// Add a primitive at the centre of the view, sized to the shape.
    fn add_prim(&mut self, kind: Kind) {
        let mut prims = self.prims();
        let size = nice(self.shape.size().iter().cloned().fold(0.0, f32::max) / 4.0);
        let pos = self.cam.target.map(|c| (c * 2.0).round() / 2.0);
        let n = prims.iter().filter(|p| p.kind == kind).count() + 1;
        prims.push(Prim::new(kind, format!("{} {n}", kind.name()), pos, size));
        let i = prims.len() - 1;
        self.set_prims(prims);
        self.select(Some(i));
    }

    /// Adds a starting setup's supports and loads (regions sized to the design's bounds) and sets
    /// the ground to match.
    fn apply_template(&mut self, t: Template) {
        let Some(shape) = &self.design_shape else { return };
        let margin = 2.0 * shape.voxel;
        let (lo, hi) = shape.bounds;
        let lo = lo.map(|v| v + margin);
        let hi = hi.map(|v| v - margin);
        let size: [f32; 3] = std::array::from_fn(|k| (hi[k] - lo[k]).max(1e-3));
        let mid: [f32; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        // a region slab on a face of the bounds: `axis`, at its low (false) or high (true) end
        let slab = |name: &str, axis: usize, high: bool, share: f32, role: Role| {
            let t = (share * size[axis]).max(margin);
            let mut pos = mid;
            pos[axis] = if high { hi[axis] - 0.5 * t } else { lo[axis] + 0.5 * t };
            let mut dims = size;
            dims[axis] = t;
            Prim {
                op: Op::Region,
                role,
                dims,
                ..Prim::new(Kind::Box, name.into(), pos, 1.0)
            }
        };
        let weight = |mut p: Prim, kg: f32, rock: f32| {
            p.kg = true;
            p.force = kg;
            p.load = [0.0, 0.0, -1.0];
            p.rock = rock;
            p
        };
        let (add, ground) = match t {
            Template::Stand => (
                vec![weight(slab("item on top", 2, true, 0.1, Role::Load), 1.0, 0.2)],
                Ground::Grip,
            ),
            Template::Wall => {
                // the wall at the low X side; the load on top at the far end
                let mut load = slab("load on the end", 2, true, 0.1, Role::Load);
                load.dims[0] = 0.25 * size[0];
                load.pos[0] = hi[0] - 0.125 * size[0];
                (
                    vec![slab("wall", 0, false, 0.05, Role::Fixed), weight(load, 2.0, 0.0)],
                    Ground::None,
                )
            }
            Template::Hang => (
                vec![
                    slab("hangs from", 2, true, 0.05, Role::Fixed),
                    weight(slab("hanging weight", 2, false, 0.1, Role::Load), 2.0, 0.0),
                ],
                Ground::None,
            ),
        };
        let mut prims = self.prims();
        let first = prims.len();
        prims.extend(add);
        self.set_prims(prims);
        self.opt.ground = ground;
        self.select(Some(first));
        self.notice = Some(format!(
            "Added {}: move and resize them in Design to match your part.",
            t.added()
        ));
    }

    fn duplicate_prim(&mut self, i: usize) {
        let mut prims = self.prims();
        let Some(p) = prims.get(i).cloned() else { return };
        let mut copy = p.clone();
        copy.name = format!("{} copy", p.name);
        copy.pos[0] += nice(p.bound_radius());
        prims.insert(i + 1, copy);
        self.set_prims(prims);
        self.select(Some(i + 1));
    }

    fn delete_prim(&mut self, i: usize) {
        let mut prims = self.prims();
        if i >= prims.len() {
            return;
        }
        prims.remove(i);
        self.edit.sel = None;
        self.set_prims(prims);
    }

    /// Editor shortcuts (only while no text field has focus).
    fn edit_keys(&mut self, ui: &mut egui::Ui) {
        if !self.can_edit() || ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let key = |ui: &mut egui::Ui, m: egui::Modifiers, k: egui::Key| ui.input_mut(|i| i.consume_key(m, k));
        let shift_cmd = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
        if key(ui, shift_cmd, egui::Key::Z) || key(ui, egui::Modifiers::COMMAND, egui::Key::Y) {
            self.undo(true);
        } else if key(ui, egui::Modifiers::COMMAND, egui::Key::Z) {
            self.undo(false);
        }
        if let Some(i) = self.edit.sel {
            if key(ui, egui::Modifiers::NONE, egui::Key::Delete) {
                self.delete_prim(i);
            } else if key(ui, egui::Modifiers::COMMAND, egui::Key::D) {
                self.duplicate_prim(i);
            } else if key(ui, egui::Modifiers::NONE, egui::Key::Escape) {
                self.select(None);
            }
        }
    }

    /// World to screen for the current camera (None behind the camera).
    fn projector(&self, rect: egui::Rect) -> impl Fn([f32; 3]) -> Option<egui::Pos2> {
        let eye = self.cam.eye();
        let (fw, rt, up) = self.cam.basis();
        let h = rect.height();
        move |q: [f32; 3]| {
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let v = [0, 1, 2].map(|k| q[k] - eye[k]);
            let z = dot(v, fw);
            (z > 1e-3).then(|| rect.center() + egui::vec2(dot(v, rt), -dot(v, up)) * (1.8 * h / z))
        }
    }

    /// Supports, loads and the ground drawn over the view (Design and Optimize steps): an arrow per
    /// load (coloured by load case, with its rocking pushes), a marker per support, the ground plane,
    /// and in Optimize the build plate when the part is printed another side than +z up.
    fn bc_overlay(&self, ui: &egui::Ui, rect: egui::Rect) {
        if self.step == Step::Print || self.volume.is_some() && self.step == Step::Design {
            return;
        }
        let Some(shape) = &self.design_shape else { return };
        let plate = self.step == Step::Optimize && self.opt.overhang && self.up != Up::PlusZ;
        if !shape.editor && self.opt.ground == Ground::None && !plate {
            return;
        }
        let prims = self.prims();
        let project = self.projector(rect);
        let painter = ui.painter_at(rect);
        let (lo, hi) = shape.bounds;
        let size = [0, 1, 2].map(|k| hi[k] - lo[k]);
        let scale = size.iter().fold(0.0f32, |m, v| m.max(*v));
        let font = egui::FontId::proportional(12.0);
        let text = egui::Color32::from_white_alpha(220);

        // the ground: the bottom of the bounds, a little wider
        if self.step == Step::Optimize && self.opt.ground != Ground::None {
            let z = lo[2];
            let m = 0.08 * scale;
            let c = [
                [lo[0] - m, lo[1] - m],
                [hi[0] + m, lo[1] - m],
                [hi[0] + m, hi[1] + m],
                [lo[0] - m, hi[1] + m],
            ];
            let pts: Vec<_> = c.iter().filter_map(|p| project([p[0], p[1], z])).collect();
            if pts.len() == 4 {
                let col = egui::Color32::from_rgba_unmultiplied(140, 170, 120, 40);
                painter.add(egui::Shape::convex_polygon(
                    pts.clone(),
                    col,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(140, 170, 120)),
                ));
                let name = if self.opt.ground == Ground::Slide {
                    "ground (slides)"
                } else {
                    "ground (grips)"
                };
                painter.text(pts[0], egui::Align2::LEFT_TOP, name, font.clone(), text);
            }
        }
        // the build plate: the bounds' face the part is printed from, a little wider
        if plate {
            let b = self.up.axis();
            let [l0, l1] = self.up.lateral();
            let w = if self.up.negative() { hi[b] } else { lo[b] };
            let m = 0.08 * scale;
            let corner = |a: f32, c: f32| {
                let mut p = [0.0; 3];
                p[b] = w;
                p[l0] = a;
                p[l1] = c;
                p
            };
            let c = [
                corner(lo[l0] - m, lo[l1] - m),
                corner(hi[l0] + m, lo[l1] - m),
                corner(hi[l0] + m, hi[l1] + m),
                corner(lo[l0] - m, hi[l1] + m),
            ];
            let pts: Vec<_> = c.iter().filter_map(|p| project(*p)).collect();
            if pts.len() == 4 {
                let col = egui::Color32::from_rgb(120, 160, 220);
                painter.add(egui::Shape::convex_polygon(
                    pts.clone(),
                    egui::Color32::from_rgba_unmultiplied(120, 160, 220, 35),
                    egui::Stroke::new(1.5, col),
                ));
                let name = format!("build plate ({} up)", self.up.name());
                // in its middle (a corner may be off-screen)
                let at = pts.iter().fold(egui::Pos2::ZERO, |a, p| a + p.to_vec2() / 4.0);
                painter.text(at, egui::Align2::CENTER_CENTER, name, font.clone(), text);
            }
        }
        if !shape.editor {
            return;
        }
        // where a primitive acts: its box clipped to the shape's bounds
        let anchor = |p: &Prim| {
            let (a, b) = p.aabb();
            [0, 1, 2].map(|k| {
                let (l, h) = (a[k].max(lo[k]), b[k].min(hi[k]));
                if l <= h {
                    0.5 * (l + h)
                } else {
                    p.pos[k].clamp(lo[k], hi[k])
                }
            })
        };
        let arrow = |from: [f32; 3], to: [f32; 3], col: egui::Color32, width: f32| {
            let (Some(a), Some(b)) = (project(from), project(to)) else {
                return None;
            };
            painter.line_segment([a, b], egui::Stroke::new(width, col));
            let d = b - a;
            if d.length() > 1.0 {
                let u = d.normalized();
                let n = egui::vec2(-u.y, u.x);
                let head = 4.0 + 2.0 * width;
                painter.add(egui::Shape::convex_polygon(
                    vec![b, b - u * head * 1.6 + n * head, b - u * head * 1.6 - n * head],
                    col,
                    egui::Stroke::NONE,
                ));
            }
            Some(a)
        };
        let len = 0.18 * scale;
        for p in prims.iter().filter(|p| !p.hidden) {
            let at = anchor(p);
            match p.role {
                Role::Load => {
                    let l = p.load;
                    let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
                    if n == 0.0 {
                        continue;
                    }
                    let d = l.map(|v| v / n);
                    let col = case_color(p.case);
                    let tail = [0, 1, 2].map(|k| at[k] - d[k] * len);
                    if let Some(a) = arrow(tail, at, col, 2.5) {
                        let amount = if p.kg {
                            format!("{} kg", fmt_num(p.force as f64))
                        } else {
                            format!("{} N", fmt_num(p.force as f64))
                        };
                        let case = if p.case > 1 {
                            format!(" · case {}", p.case)
                        } else {
                            String::new()
                        };
                        painter.text(
                            a,
                            egui::Align2::CENTER_BOTTOM,
                            format!("{amount}{case}"),
                            font.clone(),
                            col,
                        );
                    }
                    if p.rock > 0.0 {
                        let side = (0..3).max_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs())).unwrap_or(2);
                        for ax in (0..3).filter(|&a| a != side) {
                            for sign in [1.0f32, -1.0] {
                                let mut tip = at;
                                tip[ax] += sign * len * p.rock.max(0.15);
                                arrow(at, tip, col.gamma_multiply(0.7), 1.5);
                            }
                        }
                    }
                }
                Role::Fixed => {
                    let Some(c) = project(at) else { continue };
                    let col = egui::Color32::from_rgb(110, 170, 255);
                    let m = p.fix_mask();
                    // a triangle standing on a base line: clamped; with wheels: a roller
                    let r = 8.0;
                    painter.add(egui::Shape::convex_polygon(
                        vec![c, c + egui::vec2(-r, 1.6 * r), c + egui::vec2(r, 1.6 * r)],
                        col.gamma_multiply(0.5),
                        egui::Stroke::new(1.5, col),
                    ));
                    let base = c.y + 1.6 * r + if m == 7 { 2.0 } else { 7.0 };
                    if m != 7 {
                        for x in [-0.6 * r, 0.6 * r] {
                            painter.circle_stroke(egui::pos2(c.x + x, base - 3.0), 2.5, egui::Stroke::new(1.5, col));
                        }
                    }
                    painter.line_segment(
                        [egui::pos2(c.x - 1.4 * r, base), egui::pos2(c.x + 1.4 * r, base)],
                        egui::Stroke::new(2.0, col),
                    );
                    let label = match m {
                        7 => p.name.clone(),
                        _ => {
                            let axes: String = (0..3).filter(|c| m >> c & 1 == 1).map(|c| ['X', 'Y', 'Z'][c]).collect();
                            format!("{} (holds {axes})", p.name)
                        }
                    };
                    painter.text(
                        egui::pos2(c.x + 1.6 * r, c.y + r),
                        egui::Align2::LEFT_CENTER,
                        label,
                        font.clone(),
                        text,
                    );
                }
                _ => {}
            }
        }
    }

    /// Where the gizmo of the selected primitive is on screen (None if it isn't in front of the camera).
    fn gizmo(&self, rect: egui::Rect) -> Option<Gizmo> {
        let p = self.prims().get(self.edit.sel?).filter(|p| !p.hidden)?.pos;
        let eye = self.cam.eye();
        let (fw, rt, up) = self.cam.basis();
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let h = rect.height();
        let project = |q: [f32; 3]| {
            let v = [0, 1, 2].map(|k| q[k] - eye[k]);
            let z = dot(v, fw);
            (z > 1e-3).then(|| rect.center() + egui::vec2(dot(v, rt), -dot(v, up)) * (1.8 * h / z))
        };
        let depth = dot([0, 1, 2].map(|k| p[k] - eye[k]), fw);
        let centre = project(p)?;
        let len = depth * 0.07;
        let tips = [0, 1, 2].map(|k| {
            let mut q = p;
            q[k] += len;
            project(q)
        });
        Some(Gizmo {
            centre,
            tips: [tips[0]?, tips[1]?, tips[2]?],
            len,
            per_point: depth / (1.8 * h),
        })
    }

    /// The gizmo handle under `pos`: 0..3 the arrow tips, 3 the centre.
    fn gizmo_handle(g: &Gizmo, pos: egui::Pos2) -> Option<usize> {
        (0..3)
            .find(|&k| g.tips[k].distance(pos) < 10.0)
            .or((g.centre.distance(pos) < 9.0).then_some(3))
    }

    /// Selection, the gizmo and its dragging. Returns true while a handle is being dragged (the
    /// camera then stays put).
    fn edit_viewport(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: egui::Rect) -> bool {
        if !self.can_edit() {
            self.edit.drag = None;
            return false;
        }
        let gizmo = self.gizmo(rect);
        if resp.drag_started_by(egui::PointerButton::Primary)
            && let (Some(g), Some(origin)) = (&gizmo, ui.input(|i| i.pointer.press_origin()))
            && let Some(handle) = Self::gizmo_handle(g, origin)
            && let Some(p) = self.prims().get(self.edit.sel.unwrap_or(usize::MAX))
        {
            self.edit.drag = Some(GizmoDrag {
                handle,
                start: p.pos,
                moved: egui::Vec2::ZERO,
            });
            self.edit.last_edit = None; // a new undo step
        }
        let mut dragging = false;
        if let (Some(drag), Some(g), Some(i)) = (&mut self.edit.drag, &gizmo, self.edit.sel) {
            if resp.dragged_by(egui::PointerButton::Primary) {
                dragging = true;
                drag.moved += resp.drag_delta();
                let snap = ui.input(|i| i.modifiers.command);
                let mut pos = drag.start;
                if drag.handle < 3 {
                    let k = drag.handle;
                    let u = g.tips[k] - g.centre;
                    if u.length_sq() > 1.0 {
                        pos[k] += drag.moved.dot(u) / u.length_sq() * g.len;
                    }
                } else {
                    let (_, rt, up) = self.cam.basis();
                    for k in 0..3 {
                        pos[k] += (rt[k] * drag.moved.x - up[k] * drag.moved.y) * g.per_point;
                    }
                }
                for (c, start) in pos.iter_mut().zip(drag.start) {
                    if (*c - start).abs() > 1e-6 {
                        *c = if snap { c.round() } else { (*c * 100.0).round() / 100.0 };
                    }
                }
                let mut prims = self.prims();
                if let Some(p) = prims.get_mut(i) {
                    p.pos = pos;
                    self.set_prims(prims);
                }
            } else {
                self.edit.drag = None;
            }
        }

        // click: select what's under the pointer (primitives only; the script's shape isn't
        // known on the CPU)
        if resp.clicked()
            && let Some(pos) = resp.interact_pointer_pos()
        {
            let on_handle = gizmo.as_ref().is_some_and(|g| Self::gizmo_handle(g, pos).is_some());
            if !on_handle {
                let (fw, rt, up) = self.cam.basis();
                let uv = (pos - rect.center()) / rect.height();
                let rd = [0, 1, 2].map(|k| fw[k] * 1.8 + uv.x * rt[k] - uv.y * up[k]);
                let l = (rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]).sqrt();
                let far = self.cam.dist * 4.0 + self.shape.size().iter().sum::<f32>();
                let hit = prims::pick(
                    &self.prims(),
                    &self.shape.mirror,
                    self.cam.eye(),
                    rd.map(|c| c / l),
                    far,
                );
                self.select(hit);
            }
        }

        // draw the gizmo
        if let Some(g) = &gizmo {
            let painter = ui.painter_at(rect);
            let hover = resp.hover_pos().and_then(|p| Self::gizmo_handle(g, p));
            let active = self.edit.drag.as_ref().map(|d| d.handle);
            for (k, &col) in AXIS_COLORS.iter().enumerate() {
                let hot = hover == Some(k) || active == Some(k);
                painter.line_segment(
                    [g.centre, g.tips[k]],
                    egui::Stroke::new(if hot { 3.5 } else { 2.5 }, col),
                );
                painter.circle_filled(g.tips[k], if hot { 7.0 } else { 5.5 }, col);
            }
            let hot = hover == Some(3) || active == Some(3);
            painter.circle(
                g.centre,
                if hot { 7.0 } else { 5.5 },
                egui::Color32::from_white_alpha(if hot { 120 } else { 60 }),
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            );
            if let Some(p) = self.prims().get(self.edit.sel.unwrap_or(usize::MAX)) {
                painter.text(
                    g.centre + egui::vec2(10.0, 10.0),
                    egui::Align2::LEFT_TOP,
                    &p.name,
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_white_alpha(200),
                );
            }
            if hover.is_some() || active.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
            }
        }
        dragging
    }

    fn prims_ui(&mut self, ui: &mut egui::Ui) {
        if !self.loaded || self.design_shape.is_none() {
            ui.weak("shown once the shape compiles");
            return;
        }
        let before = self.prims();
        let mut prims = before.clone();
        let mut add = None;
        let mut sel = self.edit.sel;
        let mut action = None; // (duplicate?, index)
        ui.horizontal_wrapped(|ui| {
            ui.label("Add");
            for k in Kind::ALL {
                if ui.small_button(k.name()).clicked() {
                    add = Some(k);
                }
            }
        });
        let file = self
            .design_shape
            .as_ref()
            .map(|s| file_name(&s.prims_path()))
            .unwrap_or_default();
        if prims.is_empty() {
            ui.weak(format!(
                "Combined with the script's map() in list order: added, subtracted or intersected. \
                 Saved to {file}. Click one in the view to select it; drag its arrows to move it."
            ));
        } else {
            ui.weak(format!("applied to map() top to bottom; saved to {file}"));
        }
        let mut new_step = false;
        for (i, p) in prims.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let mut shown = !p.hidden;
                if ui.checkbox(&mut shown, "").on_hover_text("show / hide").changed() {
                    p.hidden = !shown;
                }
                let role = match p.role {
                    Role::Design => String::new(),
                    r => format!("  [{}]", r.name()),
                };
                let text = egui::RichText::new(format!("{}  {}  ·  {}{role}", p.op.sign(), p.name, p.kind.name()));
                let text = if p.hidden { text.weak() } else { text };
                if ui.selectable_label(sel == Some(i), text).clicked() {
                    sel = if sel == Some(i) { None } else { Some(i) };
                }
            });
        }
        if let Some(i) = sel.filter(|&i| i < prims.len()) {
            ui.add_space(4.0);
            let p = &mut prims[i];
            egui::Grid::new("prim").num_columns(2).show(ui, |ui| {
                ui.label("name");
                ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(180.0));
                ui.end_row();
                ui.label("operation")
                    .on_hover_text("how it combines with everything above it in the list (and the script)");
                ui.horizontal(|ui| {
                    for op in Op::ALL {
                        if ui.selectable_label(p.op == op, op.name()).clicked() && p.op != op {
                            p.op = op;
                            new_step = true;
                        }
                    }
                });
                ui.end_row();
                ui.label("optimize as").on_hover_text(
                    "its meaning for Optimize: fixed and load apply to its inside plus a thin shell around \
                     it, and keep those spots solid. A subtracted primitive works too: a fixed subtracted \
                     cylinder clamps its hole's wall, and a subtracted box set to load loads the face it \
                     cuts (no extra region primitive needed)",
                );
                ui.horizontal_wrapped(|ui| {
                    for r in Role::ALL {
                        if ui.selectable_label(p.role == r, r.name()).clicked() && p.role != r {
                            p.set_role(r);
                            new_step = true;
                        }
                    }
                });
                ui.end_row();
                if p.op == Op::Subtract && p.role.is_bc() {
                    ui.label("");
                    ui.weak(format!(
                        "acts on the surface this primitive cuts ({} there)",
                        p.role.name()
                    ));
                    ui.end_row();
                }
                if p.op == Op::Union && p.role.is_bc() {
                    ui.label("");
                    ui.weak("adds its own volume to the part, kept solid (\"region\" only marks the spot)");
                    ui.end_row();
                }
                if p.role == Role::Fixed {
                    ui.label("holds").on_hover_text(
                        "the directions it holds the part in. All three clamp it; only the one across a \
                         surface makes a roller: the part can't move into or off the surface, but slides \
                         along it",
                    );
                    ui.horizontal(|ui| {
                        for (c, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                            if ui.checkbox(&mut p.fix[c], name).changed() {
                                new_step = true;
                            }
                        }
                        let kind = match p.fix_mask() {
                            7 => "clamped",
                            0 => "holds nothing",
                            1 | 2 | 4 => "roller",
                            _ => "slides along one axis",
                        };
                        ui.weak(kind);
                    });
                    ui.end_row();
                }
                if p.role == Role::Load {
                    ui.label("load");
                    ui.horizontal(|ui| {
                        let suffix = if p.kg { " kg" } else { " N" };
                        ui.add(
                            egui::DragValue::new(&mut p.force)
                                .speed(if p.kg { 0.05 } else { 1.0 })
                                .range(0.0..=1e6)
                                .max_decimals(2)
                                .suffix(suffix),
                        );
                        let was = p.kg;
                        ui.selectable_value(&mut p.kg, false, "N")
                            .on_hover_text("a force in newtons, along the direction below");
                        ui.selectable_value(&mut p.kg, true, "kg")
                            .on_hover_text("the weight of something resting on it (mass × 9.81 N, straight down)");
                        if p.kg != was {
                            // keep the force the same when switching units
                            p.force = if p.kg {
                                p.force / prims::G as f32
                            } else {
                                p.force * prims::G as f32
                            };
                            if p.kg {
                                p.load = [0.0, 0.0, -1.0];
                            }
                            new_step = true;
                        }
                        if p.kg {
                            ui.weak(format!("= {} N", fmt_num(p.newtons())));
                        }
                    });
                    ui.end_row();
                    ui.label("direction").on_hover_text("normalized when used");
                    ui.vertical(|ui| {
                        drag3(ui, &mut p.load, 0.02, "");
                        ui.horizontal(|ui| {
                            for (label, d) in [
                                ("-Z", [0.0, 0.0, -1.0]),
                                ("+Z", [0.0, 0.0, 1.0]),
                                ("-X", [-1.0, 0.0, 0.0]),
                                ("+X", [1.0, 0.0, 0.0]),
                                ("-Y", [0.0, -1.0, 0.0]),
                                ("+Y", [0.0, 1.0, 0.0]),
                            ] {
                                if ui.small_button(label).clicked() {
                                    p.load = d;
                                }
                            }
                        });
                    });
                    ui.end_row();
                    ui.label("load case").on_hover_text(
                        "loads in the same case act together; different cases are separate situations \
                         the part must handle (optimized for all of them)",
                    );
                    ui.add(egui::DragValue::new(&mut p.case).range(1..=16).speed(0.05));
                    ui.end_row();
                    ui.label("rocking").on_hover_text(
                        "the part also gets pushed sideways while carrying this load (bumped, rocked, \
                         tipped). Adds four load cases: this load tilted by the given share of itself, \
                         both ways along each sideways axis. Each extra case adds solve time",
                    );
                    ui.horizontal(|ui| {
                        let mut on = p.rock > 0.0;
                        if ui.checkbox(&mut on, "").changed() {
                            p.rock = if on { 0.2 } else { 0.0 };
                            new_step = true;
                        }
                        if on {
                            ui.add(
                                egui::Slider::new(&mut p.rock, 0.05..=1.0)
                                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                    .custom_parser(|s| {
                                        s.trim_end_matches('%').trim().parse::<f64>().ok().map(|v| v / 100.0)
                                    }),
                            )
                            .on_hover_text("sideways push as a share of the load");
                        }
                    });
                    ui.end_row();
                }
                ui.label("kind");
                let mut kind = p.kind;
                egui::ComboBox::from_id_salt("prim kind")
                    .selected_text(kind.name())
                    .show_ui(ui, |ui| {
                        for k in Kind::ALL {
                            ui.selectable_value(&mut kind, k, k.name());
                        }
                    });
                if kind != p.kind {
                    p.kind = kind;
                    p.dims = kind.default_dims(nice(2.0 * p.bound_radius() / 3f32.sqrt()));
                }
                ui.end_row();
                ui.label("position");
                drag3(ui, &mut p.pos, 0.1, "");
                ui.end_row();
                ui.label("rotation").on_hover_text("degrees about X, then Y, then Z");
                drag3(ui, &mut p.rot, 1.0, "°");
                ui.end_row();
                for (k, label) in p.kind.dims().iter().enumerate() {
                    ui.label(*label);
                    ui.add(
                        egui::DragValue::new(&mut p.dims[k])
                            .speed(0.1)
                            .range(0.0..=f32::MAX)
                            .max_decimals(2)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                }
                if p.kind.rounds() {
                    ui.label("rounding");
                    ui.add(
                        egui::DragValue::new(&mut p.round)
                            .speed(0.05)
                            .range(0.0..=f32::MAX)
                            .max_decimals(2)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                }
                ui.label("blend")
                    .on_hover_text("smoothing radius of the operation with the shape so far (0 = sharp)");
                ui.add(
                    egui::DragValue::new(&mut p.blend)
                        .speed(0.05)
                        .range(0.0..=f32::MAX)
                        .max_decimals(2)
                        .suffix(" mm"),
                );
                ui.end_row();
            });
            let mut moved = None;
            ui.horizontal(|ui| {
                if ui.button("Duplicate").on_hover_text("Ctrl+D").clicked() {
                    action = Some((true, i));
                }
                if ui.button("Delete").on_hover_text("Del").clicked() {
                    action = Some((false, i));
                }
                ui.separator();
                let n = prims.len();
                if ui
                    .add_enabled(i > 0, egui::Button::new("Up"))
                    .on_hover_text("apply it earlier")
                    .clicked()
                {
                    moved = Some(i - 1);
                }
                if ui
                    .add_enabled(i + 1 < n, egui::Button::new("Down"))
                    .on_hover_text("apply it later")
                    .clicked()
                {
                    moved = Some(i + 1);
                }
            });
            if let Some(j) = moved {
                prims.swap(i, j);
                sel = Some(j);
                new_step = true;
            }
        }
        if new_step {
            self.edit.last_edit = None; // its own undo step
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.edit.undo.is_empty(), egui::Button::new("Undo").small())
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                action = Some((false, usize::MAX));
            }
            if ui
                .add_enabled(!self.edit.redo.is_empty(), egui::Button::new("Redo").small())
                .on_hover_text("Ctrl+Shift+Z")
                .clicked()
            {
                action = Some((true, usize::MAX));
            }
        });
        if prims != before {
            self.set_prims(prims);
        }
        self.select(sel);
        match action {
            Some((redo, usize::MAX)) => self.undo(redo),
            Some((true, i)) => self.duplicate_prim(i),
            Some((false, i)) => self.delete_prim(i),
            None => {}
        }
        if let Some(k) = add {
            self.add_prim(k);
        }
    }

    fn param_value(&self, p: &Param) -> f64 {
        self.defines
            .iter()
            .find(|(k, _)| *k == p.name)
            .map_or(p.default, |(_, v)| *v)
    }

    fn set_param(&mut self, p: &Param, v: f64) {
        self.defines.retain(|(k, _)| *k != p.name);
        if v != p.default {
            self.defines.push((p.name.clone(), v));
        }
        self.params_dirty = true;
    }

    fn set_field_mask(&mut self) {
        let n = self.fields.len();
        self.opts.fields = match self.field_mode {
            0 => all_fields(&self.fields),
            m if m <= n.min(32) => 1 << (m - 1),
            _ => 0,
        };
    }

    fn reset_camera(&mut self) {
        self.cam = Camera::framing(&self.shape);
        self.opts.cut = self.cam.target[1];
    }

    fn set_view(&mut self, yaw: f32, pitch: f32) {
        (self.cam.yaw, self.cam.pitch) = (yaw.to_radians(), pitch.to_radians());
    }

    // ------------------------------------------------------------------------------------ jobs

    /// The file the Print step meshes: the design, or the optimization result.
    /// The mesh file the viewport should show instead of the shape: the Print step's exported mesh
    /// when "show the mesh" is on and the file exists.
    fn wanted_mesh(&self) -> Option<PathBuf> {
        if self.step != Step::Print || !self.print.show_mesh {
            return None;
        }
        Some(self.mesh_path(&self.print_source()?)).filter(|p| p.exists())
    }

    /// Load the wanted mesh onto the GPU when it changed on disk (not while an export is writing
    /// it), or drop it when none is wanted.
    fn sync_mesh(&mut self) {
        let want = self.wanted_mesh();
        let job = self.out_job.get();
        let Some(win) = &mut self.win else { return };
        let Some(path) = want else {
            if win.mesh.take().is_some() {
                win.touch();
            }
            return;
        };
        if job.running && job.target == path {
            return;
        }
        let m = mtime(&path);
        if win.mesh.as_ref().is_some_and(|s| s.path == path && s.mtime == m) {
            return;
        }
        win.touch();
        if win.meshr.is_none() {
            match crate::meshview::MeshRenderer::new(&win.gpu, VIEW_FORMAT, &win.ubuf) {
                Ok(r) => win.meshr = Some(r),
                Err(e) => {
                    win.mesh = Some(ShownMesh {
                        path,
                        mtime: m,
                        gpu: Err(format!("{e:#}")),
                        tris: 0,
                    });
                    return;
                }
            }
        }
        let r = win.meshr.as_ref().unwrap();
        let loaded = crate::mesh::Mesh::load(&path).and_then(|mesh| Ok((r.upload(&win.gpu, &mesh)?, mesh.tris.len())));
        win.mesh = Some(match loaded {
            Ok((g, tris)) => ShownMesh {
                path,
                mtime: m,
                gpu: Ok(g),
                tris,
            },
            Err(e) => ShownMesh {
                path,
                mtime: m,
                gpu: Err(format!("{e:#}")),
                tris: 0,
            },
        });
    }

    fn print_source(&self) -> Option<PathBuf> {
        if self.print_result
            && let Some(r) = self.existing_result()
        {
            return Some(r);
        }
        self.design.clone()
    }

    fn out_path(&self, src: &Path, ext: &str) -> PathBuf {
        let defines: &[(String, f64)] = if is_vti(src) { &[] } else { &self.defines };
        crate::default_out(src, ext, defines)
    }

    fn mesh_path(&self, src: &Path) -> PathBuf {
        self.out_path(src, if self.print.three_mf { ".3mf" } else { ".stl" })
    }

    /// `gen3d export|slice|open` arguments that mesh `src` the way the app shows it.
    fn mesh_args(&self, src: &Path) -> Vec<String> {
        let mut a = vec![src.display().to_string()];
        if is_vti(src) {
            let o = if self.shown == src {
                self.current_vopts()
            } else {
                self.vopts.clone()
            };
            if let Some(n) = o.array {
                a.extend(["--array".into(), n]);
            }
            if let Some(iso) = o.iso {
                a.extend(["--iso".into(), iso.to_string()]);
            }
            match o.below {
                Some(true) => a.push("--below".into()),
                Some(false) => a.push("--above".into()),
                None => {}
            }
            if let Some(n) = self.print.smooth {
                a.extend(["--smooth".into(), n.to_string()]);
            }
            if self.print.grow > 0.0 {
                a.extend(["--grow".into(), fmt_num(self.print.grow as f64)]);
            }
        } else {
            for (k, v) in &self.defines {
                a.extend(["-D".into(), format!("{k}={v}")]);
            }
            if let Some(v) = self.print.voxel {
                a.extend(["--voxel".into(), fmt_num(v as f64)]);
            }
        }
        if let Some(d) = self.print.decimate {
            a.extend(["--decimate".into(), fmt_num(d as f64)]);
        }
        if self.print.min_body > 0.0 {
            a.extend(["--min-body".into(), fmt_num(self.print.min_body)]);
        }
        // always: without it the file's own orientation would apply
        a.push(format!("--up={}", self.up.name()));
        a
    }

    /// PrusaSlicer options for the overrides chosen in the Print step.
    fn slicer_overrides(&self) -> Vec<String> {
        if self.slicer.as_ref().is_none_or(|(k, _)| *k != slicer::Kind::Prusa) {
            return vec![];
        }
        let mut a = vec![];
        if let Some(l) = self.print.layer {
            a.extend(["--layer-height".into(), fmt_num(l as f64)]);
        }
        if let Some(i) = self.print.infill {
            a.extend(["--fill-density".into(), format!("{i}%")]);
        }
        match self.print.supports {
            Supports::Profile => {}
            Supports::Off => a.push("--no-support-material".into()),
            Supports::Everywhere => a.extend(["--support-material", "--support-material-auto"].map(String::from)),
            Supports::BuildPlate => a.extend(
                [
                    "--support-material",
                    "--support-material-auto",
                    "--support-material-buildplate-only",
                ]
                .map(String::from),
            ),
        }
        a
    }

    fn start_export(&mut self) {
        let Some(src) = self.print_source() else { return };
        let out = self.mesh_path(&src);
        let mut a = self.mesh_args(&src);
        a.extend([
            "--format".into(),
            if self.print.three_mf { "3mf" } else { "stl" }.into(),
        ]);
        a.extend(["-o".into(), out.display().to_string()]);
        self.out_job.start("export", a, out);
    }

    fn start_slice(&mut self) {
        let Some(src) = self.print_source() else { return };
        let gcode = self.out_path(&src, ".gcode");
        let mut a = self.mesh_args(&src);
        a.extend(["-o".into(), gcode.display().to_string()]);
        if let Some(p) = &self.print.profile {
            a.extend(["-c".into(), p.display().to_string()]);
        }
        let extra = self.slicer_overrides();
        if !extra.is_empty() {
            a.push("--".into());
            a.extend(extra);
        }
        self.out_job.start("slice", a, gcode);
    }

    fn start_open(&mut self) {
        let Some(src) = self.print_source() else { return };
        let mut a = self.mesh_args(&src);
        a.extend([
            "--format".into(),
            if self.print.three_mf { "3mf" } else { "stl" }.into(),
        ]);
        let target = self.mesh_path(&src);
        self.out_job.start("open", a, target);
    }

    /// Start an optimization; `resume` continues the stopped run that wrote the result.
    fn start_optimize(&mut self, resume: bool) {
        let (Some(design), Some(result), Some(shape)) = (self.design.clone(), self.result_path(), &self.design_shape)
        else {
            return;
        };
        let o = &self.opt;
        let vox = o.voxel.unwrap_or_else(|| {
            crate::topopt::auto_voxel(shape, crate::topopt::auto_elements(self.gpu_solver), self.plane2d)
        });
        let mut a = vec![design.display().to_string()];
        for (k, v) in &self.defines {
            a.extend(["-D".into(), format!("{k}={v}")]);
        }
        for (k, v) in [
            ("--volfrac", o.volfrac),
            ("--iters", o.iters as f64),
            ("--voxel", vox as f64),
            ("--force", o.force),
            ("--rmin", o.rmin),
        ] {
            a.extend([k.to_string(), fmt_num(v)]);
        }
        if !o.projection {
            a.push("--no-projection".into());
        }
        if !o.overhang {
            a.push("--no-overhang".into());
        }
        // always: without it the file's own orientation would apply
        a.push(format!("--up={}", self.up.name()));
        // always: without it the file's own 2D setting would apply
        match self.plane2d {
            Some(ax) => a.extend(["--2d".into(), ["x", "y", "z"][ax].into()]),
            None => a.push("--3d".into()),
        }
        if o.ground != Ground::None {
            a.extend(["--ground".into(), o.ground.name().into()]);
        }
        a.extend(["-o".into(), result.display().to_string()]);
        if resume {
            a.push("--resume".into());
        }
        self.opt_job.start("optimize", a, result);
    }

    fn gcode_summary(&mut self, p: &Path) -> Vec<(String, String)> {
        let m = mtime(p);
        if let Some((t, s)) = self.gcode_cache.get(p)
            && *t == m
        {
            return s.clone();
        }
        let s = pretty_summary(&slicer::gcode_summary(p).unwrap_or_default());
        self.gcode_cache.insert(p.to_path_buf(), (m, s.clone()));
        s
    }

    fn edit_design(&mut self) {
        if let Some(d) = &self.design
            && let Err(e) = slicer::open_in_editor(d)
        {
            self.notice = Some(format!("{e:#}"));
        }
    }

    fn on_key(&mut self, key: KeyCode) {
        let step = self.shape.size()[1] / 50.0;
        match key {
            KeyCode::KeyR => self.reset_camera(),
            KeyCode::KeyC => self.opts.section = !self.opts.section,
            KeyCode::BracketLeft => self.opts.cut -= step,
            KeyCode::BracketRight => self.opts.cut += step,
            KeyCode::KeyB => self.opts.clip = !self.opts.clip,
            KeyCode::KeyV => self.opts.show_bounds = !self.opts.show_bounds,
            KeyCode::Minus | KeyCode::Equal if self.volume.is_some() => {
                let dir = if key == KeyCode::Minus { -1.0 } else { 1.0 };
                self.edit_volume(false, |v| v.nudge_iso(dir));
            }
            KeyCode::KeyM if self.volume.is_some() => self.edit_volume_uniforms(|v| v.blocky = !v.blocky),
            KeyCode::KeyI if self.volume.is_some() => self.edit_volume(false, |v| {
                v.below = !v.below;
                v.recompute();
            }),
            KeyCode::KeyA if self.volume.is_some() => self.edit_volume(true, |v| {
                let next = (v.array + 1) % v.file.arrays.len();
                v.select(next);
                v.recompute();
            }),
            KeyCode::KeyF if self.volume.is_some() => self.edit_volume(false, |v| {
                let n = v.file.arrays.len();
                v.color = match v.color {
                    None => Some(0),
                    Some(c) if c + 1 < n => Some(c + 1),
                    Some(_) => None,
                };
                v.update_color();
            }),
            KeyCode::KeyF if !self.fields.is_empty() => {
                self.field_mode = (self.field_mode + 1) % (self.fields.len() + 2);
                self.set_field_mask();
            }
            KeyCode::F5 => self.reload(false),
            KeyCode::Digit1 => self.set_view(-90.0, 0.0),
            KeyCode::Digit2 => self.set_view(0.0, 0.0),
            KeyCode::Digit3 => self.set_view(-90.0, 89.9),
            KeyCode::Digit4 => self.set_view(-60.0, 28.0),
            _ => {}
        }
    }

    // -------------------------------------------------------------------------------------- ui

    fn ui(&mut self, ui: &mut egui::Ui, tex: egui::TextureId) {
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::O)) {
            let dir = self.browser.dir.clone();
            self.browser.show_at(&dir);
        }
        if !self.browser.open {
            self.edit_keys(ui);
        }
        let ctx = ui.ctx().clone();
        match self.browser.ui(&ctx, self.design.is_some()) {
            Some(Pick::Open(p)) => self.open_design(&p, vec![]),
            Some(Pick::Created(p)) => self.open_design(&p, vec![]),
            None => {}
        }
        egui::Panel::top("steps").show(ui, |ui| self.top_bar(ui));
        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(340.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(4.0);
                    if let Some(n) = self.notice.clone() {
                        ui.horizontal_wrapped(|ui| {
                            ui.colored_label(ui.visuals().warn_fg_color, n);
                            if ui.small_button("Dismiss").clicked() {
                                self.notice = None;
                            }
                        });
                        ui.separator();
                    }
                    if self.design.is_none() {
                        ui.weak("Open a shape to start: Ctrl+O, or the Open button above.");
                        return;
                    }
                    match self.step {
                        Step::Design => self.design_ui(ui),
                        Step::Optimize => self.optimize_ui(ui),
                        Step::Print => self.print_ui(ui),
                    }
                    ui.add_space(12.0);
                    egui::CollapsingHeader::new("Keyboard & mouse").show(ui, |ui| {
                        egui::Grid::new("keys").num_columns(2).striped(true).show(ui, |ui| {
                            for (k, v) in KEYS {
                                ui.monospace(*k);
                                ui.label(*v);
                                ui.end_row();
                            }
                        });
                    });
                });
            });
        if let Some(err) = self.error.as_ref().or(self.param_error.as_ref()) {
            egui::Panel::bottom("error")
                .resizable(true)
                .default_size(170.0)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.colored_label(ui.visuals().error_fg_color, egui::RichText::new("Error").strong());
                        ui.weak(if self.loaded {
                            "— showing the last version that worked; fix the file and save"
                        } else {
                            "— fix the file and save"
                        });
                    });
                    egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(err).monospace()).selectable(true));
                    });
                });
        }
        egui::CentralPanel::no_frame().show(ui, |ui| {
            egui::Frame::new()
                .fill(ui.visuals().panel_fill)
                .inner_margin(egui::Margin::symmetric(8, 4))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    self.toolbar(ui)
                });
            self.viewport_ui(ui, tex);
        });
    }

    /// File, the three steps, and running jobs.
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .button("Open…")
                .on_hover_text("open or create a shape (Ctrl+O)")
                .clicked()
            {
                let dir = self.browser.dir.clone();
                self.browser.show_at(&dir);
            }
            if let Some(d) = self.design.clone() {
                if self.error.is_some() {
                    status_dot(ui, ui.visuals().error_fg_color);
                } else if self.loaded {
                    status_dot(ui, OK_GREEN);
                }
                ui.label(egui::RichText::new(file_name(&d)).strong())
                    .on_hover_text(format!(
                        "{}\nreloads when the file (or an include) is saved; last loaded {}",
                        d.display(),
                        self.loaded_at
                    ));
                if !is_vti(&d)
                    && ui
                        .button("Edit")
                        .on_hover_text("open the .wgsl in your editor ($GEN3D_EDITOR, else the desktop default)")
                        .clicked()
                {
                    self.edit_design();
                }
            }
            ui.separator();
            let steps = [
                (Step::Design, "1  Design", "shape and parameters"),
                (
                    Step::Optimize,
                    "2  Optimize",
                    "optional: topology optimization of the design domain",
                ),
                (Step::Print, "3  Print", "mesh and slice to G-code"),
            ];
            for (i, (step, label, tip)) in steps.into_iter().enumerate() {
                if i > 0 {
                    ui.weak(">");
                }
                let enabled = self.design.is_some() && !(step == Step::Optimize && self.design_is_grid());
                let r = ui
                    .add_enabled_ui(enabled, |ui| ui.selectable_label(self.step == step, label))
                    .inner
                    .on_hover_text(tip);
                if r.clicked() && self.step != step {
                    self.set_step(step);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(&self.gpu_name);
                for slot in [&self.out_job, &self.opt_job] {
                    let j = slot.get();
                    if j.running {
                        let what = if j.cmd == "optimize" {
                            format!("optimizing: iteration {}", opt_progress(&j.log).iteration)
                        } else {
                            format!("{}: {}", j.cmd, file_name(&j.target))
                        };
                        ui.label(what);
                        ui.spinner();
                        ui.separator();
                    }
                }
            });
        });
        ui.add_space(2.0);
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (label, key, yaw, pitch) in [
                ("Front", "1", -90.0, 0.0),
                ("Side", "2", 0.0, 0.0),
                ("Top", "3", -90.0, 89.9),
                ("Iso", "4", -60.0, 28.0),
            ] {
                if ui.small_button(label).on_hover_text(key).clicked() {
                    self.set_view(yaw, pitch);
                }
            }
            if ui
                .small_button("Reset")
                .on_hover_text("R, or double-click the view")
                .clicked()
            {
                self.reset_camera();
            }
            ui.separator();
            ui.toggle_value(&mut self.opts.section, "Section")
                .on_hover_text("cut the part open (C); the cut face shows internal distance bands, [ and ] move it");
            if self.opts.section {
                let (lo, hi) = self.shape.bounds;
                ui.spacing_mut().slider_width = 120.0;
                ui.add(
                    egui::Slider::new(&mut self.opts.cut, lo[1]..=hi[1])
                        .suffix(" mm")
                        .clamping(egui::SliderClamping::Never),
                )
                .on_hover_text("y of the cut plane");
            }
            ui.toggle_value(&mut self.opts.show_bounds, "Bounds")
                .on_hover_text("show the export region (V)");
            ui.toggle_value(&mut self.opts.clip, "Clip")
                .on_hover_text("cut everything outside the export region (B)");
            ui.separator();
            egui::ComboBox::from_id_salt("res")
                .width(64.0)
                .selected_text(format!("{:.0}%", self.render_scale * 100.0))
                .show_ui(ui, |ui| {
                    for s in [1.0, 0.75, 0.5, 0.25] {
                        ui.selectable_value(&mut self.render_scale, s, format!("{:.0}%", s * 100.0));
                    }
                })
                .response
                .on_hover_text("resolution: raymarch fewer pixels for heavy shapes");
        });
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui, tex: egui::TextureId) {
        let rect = ui.available_rect_before_wrap();
        let ppp = ui.ctx().pixels_per_point();
        let px = |v: f32| ((v * ppp * self.render_scale).round() as u32).max(1);
        self.view_px = (px(rect.width()), px(rect.height()));
        self.view_rect = rect;
        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        if self.design.is_none() {
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(28, 30, 34));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open a shape to start (Ctrl+O)",
                egui::FontId::proportional(18.0),
                egui::Color32::from_white_alpha(200),
            );
            return;
        }
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        painter.image(tex, rect, uv, egui::Color32::WHITE);

        let editing = self.edit_viewport(ui, &resp, rect);
        self.bc_overlay(ui, rect);

        // camera (rates per physical pixel)
        let d = resp.drag_delta();
        if editing {
        } else if resp.dragged_by(egui::PointerButton::Primary) {
            self.cam.yaw -= d.x * ppp * 0.008;
            self.cam.pitch = (self.cam.pitch + d.y * ppp * 0.008).clamp(-1.55, 1.55);
        } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let s = self.cam.dist / rect.height().max(1.0) * 1.1;
            let right = (-self.cam.yaw.sin(), self.cam.yaw.cos());
            self.cam.target[0] -= right.0 * d.x * s;
            self.cam.target[1] -= right.1 * d.x * s;
            self.cam.target[2] += d.y * s;
        }
        if resp.double_clicked() {
            self.reset_camera();
        }
        if resp.hovered() {
            let dy: f32 = ui.input(|i| {
                i.events
                    .iter()
                    .map(|e| match e {
                        egui::Event::MouseWheel { unit, delta, .. } => match unit {
                            egui::MouseWheelUnit::Point => delta.y / 40.0,
                            egui::MouseWheelUnit::Line => delta.y,
                            egui::MouseWheelUnit::Page => delta.y * 10.0,
                        },
                        _ => 0.0,
                    })
                    .sum()
            });
            self.cam.dist *= 0.9f32.powf(dy);
        }
        if resp.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }

        // what is shown, top left; frame rate and modes, top right
        let shown_mesh = self
            .win
            .as_ref()
            .and_then(|w| w.mesh.as_ref())
            .filter(|m| m.gpu.is_ok())
            .map(|m| m.path.clone());
        let what = if let Some(m) = shown_mesh {
            format!("exported mesh · {}", file_name(&m))
        } else if self.volume.is_some() && Some(&self.shown) != self.design.as_ref() {
            format!("optimization result · {}", file_name(&self.shown))
        } else {
            file_name(&self.shown)
        };
        painter.text(
            rect.left_top() + egui::vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            what,
            egui::FontId::proportional(13.0),
            egui::Color32::from_white_alpha(170),
        );
        // the viewport's own redraw rate: shown while it redraws (it is drawn only on changes)
        let mut info = vec![];
        if self.rebuild.is_some() {
            info.push("compiling…".to_string());
        }
        if self.fps > 0.0 {
            info.push(format!("{:.0} fps", self.fps));
        }
        painter.text(
            rect.right_top() + egui::vec2(-10.0, 8.0),
            egui::Align2::RIGHT_TOP,
            info.join("  ·  "),
            egui::FontId::proportional(13.0),
            egui::Color32::from_white_alpha(170),
        );
        if self.loaded && self.step == Step::Design && self.is_empty_design() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Empty shape: add a primitive (Primitives > Add), or click Edit to script it",
                egui::FontId::proportional(16.0),
                egui::Color32::from_white_alpha(190),
            );
        }
        if !self.loaded {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "nothing to show until the shape compiles",
                egui::FontId::proportional(18.0),
                egui::Color32::from_white_alpha(200),
            );
        }
    }

    // ---------------------------------------------------------------------------- 1. Design

    fn design_ui(&mut self, ui: &mut egui::Ui) {
        if self.design_is_grid() {
            ui.weak(
                "A .vti grid (e.g. a simulation result): pick the array and threshold that form the \
                 solid, then print it.",
            );
            section(ui, "Solid");
            self.volume_ui(ui);
            egui::CollapsingHeader::new("Grid details").show(ui, |ui| self.grid_ui(ui));
            return;
        }
        if self.is_empty_design() {
            ui.weak(
                "An empty shape: add primitives below, or click Edit to script it (`fn map`); the \
                 preview reloads when you save.",
            );
        } else {
            ui.weak("Edit the .wgsl in your editor; the preview reloads when you save.");
        }
        // a shape made of primitives only has no use for the parameters' how-to
        let script = self.design_shape.as_ref().is_none_or(|s| s.script_map);
        if script || !self.params.is_empty() {
            section(ui, "Parameters");
            self.params_ui(ui);
        }
        section(ui, "Primitives");
        self.prims_ui(ui);
        if self.loaded && self.design_shape.is_some() {
            self.mirror_ui(ui);
        }
        if !self.fields.is_empty() {
            section(ui, "Field overlays");
            self.fields_ui(ui);
        }
        ui.add_space(6.0);
        egui::CollapsingHeader::new("Shape details").show(ui, |ui| self.shape_ui(ui));
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) {
        if !self.loaded {
            ui.weak("shown once the shape compiles");
            return;
        }
        if self.params.is_empty() {
            ui.weak(
                "This shape has no parameters. Declare one with\n`override SIZE: f32 = 10.0;` to get a slider here.",
            );
            return;
        }
        ui.spacing_mut().slider_width = 150.0;
        let params = self.params.clone();
        let live = self.win.as_ref().map(|w| w.live.clone()).unwrap_or_default();
        egui::Grid::new("params").num_columns(3).show(ui, |ui| {
            for p in &params {
                let cur = self.param_value(p);
                let mut tip = p.note.clone();
                if !tip.is_empty() {
                    tip.push('\n');
                }
                tip += &format!("default {}", fmt_num(p.default));
                let compiled = !live.contains(&p.name);
                if compiled {
                    tip += "\nchanging it compiles the shader (only f32 parameters update live)";
                }
                let name = egui::RichText::new(&p.name);
                ui.label(if compiled { name.italics() } else { name })
                    .on_hover_text(tip);
                let mut v = cur;
                let changed = match p.kind {
                    ParamKind::Bool => {
                        let mut b = v != 0.0;
                        let r = ui.checkbox(&mut b, "");
                        v = b as u8 as f64;
                        r.changed()
                    }
                    kind => {
                        let (lo, hi) = p.slider_range();
                        let mut s = egui::Slider::new(&mut v, lo..=hi).clamping(egui::SliderClamping::Never);
                        if kind != ParamKind::Float {
                            s = s.integer();
                        }
                        let changed = ui.add(s).changed();
                        if kind == ParamKind::Uint {
                            v = v.max(0.0);
                        }
                        changed
                    }
                };
                if changed && v != cur {
                    self.set_param(p, v);
                }
                if v != p.default {
                    if ui
                        .small_button("reset")
                        .on_hover_text(format!("reset to {}", fmt_num(p.default)))
                        .clicked()
                    {
                        self.set_param(p, p.default);
                    }
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });
        if !self.defines.is_empty() && ui.button("Reset all").clicked() {
            self.defines.clear();
            self.params_dirty = true;
        }
    }

    fn fields_ui(&mut self, ui: &mut egui::Ui) {
        let before = self.field_mode;
        let n = self.fields.len();
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.field_mode, 0, "All").on_hover_text("F cycles");
            for (i, f) in self.fields.iter().enumerate() {
                ui.radio_value(&mut self.field_mode, i + 1, &f.name)
                    .on_hover_text(if f.comps == 1 {
                        "scalar field: tints the surface where it is < 0"
                    } else {
                        "vector field: tints where it is non-zero; bands move along it"
                    });
                let (r, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 3.0, rgb_gamma(field_color(i)));
            }
            ui.radio_value(&mut self.field_mode, n + 1, "Off");
        });
        if self.field_mode != before {
            self.set_field_mask();
        }
    }

    fn shape_ui(&mut self, ui: &mut egui::Ui) {
        let Some(s) = self.design_shape.clone() else {
            ui.weak("shown once the shape compiles");
            return;
        };
        let (lo, hi) = s.bounds;
        let v3 = |v: [f32; 3]| {
            format!(
                "{} {} {}",
                fmt_num(v[0] as f64),
                fmt_num(v[1] as f64),
                fmt_num(v[2] as f64)
            )
        };
        egui::Grid::new("shape").num_columns(2).show(ui, |ui| {
            ui.label("bounds");
            ui.monospace(format!("{}  to  {}", v3(lo), v3(hi)));
            ui.end_row();
            ui.label("size");
            ui.monospace(format!("{} mm", v3(s.size())));
            ui.end_row();
            ui.label("voxel");
            ui.monospace(format!("{} mm", fmt_num(s.voxel as f64)));
            ui.end_row();
            ui.label("step");
            ui.monospace(fmt_num(s.step() as f64));
            ui.end_row();
            if let Some(d) = s.decimate() {
                ui.label("decimate");
                ui.monospace(fmt_num(d as f64));
                ui.end_row();
            }
            ui.label("files");
            let files: Vec<_> = s.files.iter().map(|f| f.display().to_string()).collect();
            ui.label(files.len().to_string()).on_hover_text(files.join("\n"));
            ui.end_row();
            ui.label("GPU");
            ui.label(&self.gpu_name);
            ui.end_row();
        });
    }

    /// Threshold, inside side, smooth/blocky, material volume and colour of the shown grid.
    fn volume_ui(&mut self, ui: &mut egui::Ui) {
        let Some(v) = &self.volume else {
            ui.weak("shown once the file loads");
            return;
        };
        let names: Vec<String> = v.file.arrays.iter().map(|a| a.name.clone()).collect();
        let (cur_array, cur_below, cur_blocky, cur_color) = (v.array, v.below, v.blocky, v.color);
        let (lo, hi) = v.range;
        let hist = v.hist.clone();
        let (inside, count, inside_mm3) = (v.inside_count, v.count(), v.inside_volume());
        let (crange, auto_crange, fixed_crange) = (v.crange(), v.auto_crange(), v.color_range);
        let (mut array, mut below, mut blocky, mut color) = (cur_array, cur_below, cur_blocky, cur_color);
        let cur_iso = v.iso;
        let mut iso = self.iso_drag.unwrap_or(cur_iso);

        if names.len() > 1 {
            egui::ComboBox::from_label("array")
                .selected_text(&names[array])
                .show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut array, i, n);
                    }
                })
                .response
                .on_hover_text("the array that forms the solid (A cycles)");
        }

        // threshold: histogram (click/drag) and slider; recomputing is CPU work, so both apply on release
        let (h, picked) = histogram(ui, &hist, (lo, hi), iso, below);
        if picked.is_finite() && (h.dragged() || h.clicked()) {
            iso = picked;
        }
        let r = ui
            .add(
                egui::Slider::new(&mut iso, lo..=hi)
                    .text("threshold")
                    .clamping(egui::SliderClamping::Never),
            )
            .on_hover_text("- and = step it; applied when you let go");
        let dragging = h.dragged() || r.dragged();
        let commit = h.clicked() || h.drag_stopped() || r.drag_stopped() || (r.changed() && !r.dragged());
        self.iso_drag = if dragging { Some(iso) } else { None };

        ui.horizontal(|ui| {
            ui.label("solid where");
            ui.radio_value(&mut below, true, "below");
            ui.radio_value(&mut below, false, "above");
        });
        ui.horizontal(|ui| {
            ui.label("draw");
            ui.radio_value(&mut blocky, false, "smooth");
            ui.radio_value(&mut blocky, true, "blocky")
                .on_hover_text("M · every inside sample as a cube (for cell data: the elements)");
        });

        let pct = 100.0 * inside as f64 / count.max(1) as f64;
        ui.label(format!(
            "≈ {} cm³ · {} g PLA · {pct:.1}% of the grid",
            fmt_num((inside_mm3 / 1000.0 * 100.0).round() / 100.0),
            fmt_num((inside_mm3 / 1000.0 * 1.24 * 10.0).round() / 10.0)
        ))
        .on_hover_text(format!(
            "{inside} of {count} samples inside × cell volume; solid, so an upper bound for the print \
             (PLA at 1.24 g/cm³, 100% infill)"
        ));

        ui.add_space(4.0);
        egui::ComboBox::from_label("colour by")
            .selected_text(color.map_or("none", |c| names[c].as_str()))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut color, None, "none");
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(&mut color, Some(i), n);
                }
            })
            .response
            .on_hover_text("F cycles");
        let mut new_crange = fixed_crange;
        if color.is_some() && color == cur_color {
            colour_bar(ui, crange);
            let mut fixed = fixed_crange.is_some();
            ui.horizontal(|ui| {
                ui.checkbox(&mut fixed, "fixed range").on_hover_text(
                    "by default the range is the 1st..99th percentile over the inside samples; fix it to \
                     compare iterations or to ignore a few hotspots",
                );
                if fixed {
                    let (mut a, mut b) = fixed_crange.unwrap_or(auto_crange);
                    let speed = ((b - a).abs() / 200.0).max(1e-6);
                    ui.add(egui::DragValue::new(&mut a).speed(speed));
                    ui.add(egui::DragValue::new(&mut b).speed(speed));
                    new_crange = Some((a, b));
                } else {
                    new_crange = None;
                }
            });
        }

        if array != cur_array {
            return self.edit_volume(true, |v| {
                v.select(array);
                v.recompute();
            });
        }
        if commit && iso != cur_iso {
            self.edit_volume(false, |v| {
                v.iso = iso;
                v.recompute();
            });
        }
        if below != cur_below {
            self.edit_volume(false, |v| {
                v.below = below;
                v.recompute();
            });
        }
        if blocky != cur_blocky {
            self.edit_volume_uniforms(|v| v.blocky = blocky);
        }
        if color != cur_color {
            self.edit_volume(false, |v| {
                v.color = color;
                v.update_color();
            });
        } else if new_crange != fixed_crange {
            self.edit_volume_uniforms(|v| v.color_range = new_crange);
        }
    }

    /// Lattice and arrays of the shown grid.
    fn grid_ui(&mut self, ui: &mut egui::Ui) {
        let Some(v) = &self.volume else { return };
        let (o, sp, d) = v.grid();
        let f = &v.file;
        let v3 = |a: [f64; 3]| format!("{} {} {}", fmt_num(a[0]), fmt_num(a[1]), fmt_num(a[2]));
        let (_, pd) = f.lattice(false);
        egui::Grid::new("grid").num_columns(2).show(ui, |ui| {
            ui.label("points");
            ui.monospace(format!("{} × {} × {}", pd[0], pd[1], pd[2]));
            ui.end_row();
            ui.label("spacing");
            ui.monospace(format!("{} mm", v3(sp)));
            ui.end_row();
            ui.label("first sample");
            ui.monospace(format!("{} mm", v3(o)))
                .on_hover_text("of the shown array (cell data sits half a cell inside the points)");
            ui.end_row();
            ui.label("shown samples");
            ui.monospace(format!("{} × {} × {}", d[0], d[1], d[2]));
            ui.end_row();
        });
        ui.add_space(4.0);
        egui::Grid::new("arrays").num_columns(3).striped(true).show(ui, |ui| {
            ui.strong("array");
            ui.strong("data");
            ui.strong("range");
            ui.end_row();
            for (i, a) in f.arrays.iter().enumerate() {
                let name = if a.comps == 1 {
                    a.name.clone()
                } else {
                    format!("{} [{}]", a.name, a.comps)
                };
                if i == v.array {
                    ui.strong(name).on_hover_text("shown");
                } else {
                    ui.label(name);
                }
                ui.label(if a.cell { "cell" } else { "point" });
                let (lo, hi) = a
                    .data
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
                ui.monospace(format!("{} .. {}", fmt_num(lo as f64), fmt_num(hi as f64)));
                ui.end_row();
            }
        });
    }

    // -------------------------------------------------------------------------- 2. Optimize

    fn optimize_ui(&mut self, ui: &mut egui::Ui) {
        ui.weak(
            "Optional. Finds the stiffest part that uses a given share of the design domain (the \
             shape), held at its supports and carrying its loads.",
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Stands on the ground").on_hover_text(
                "the bottom of the shape rests on a table or floor. The optimizer picks where the part \
                 touches down. Note: the ground can still pull on the part (a linear model can't let it \
                 lift off), so check the result for feet that would hang in tension",
            );
            let running = self.opt_job.running();
            ui.add_enabled_ui(!running, |ui| {
                let g = &mut self.opt.ground;
                ui.selectable_value(g, Ground::None, "no");
                ui.selectable_value(g, Ground::Slide, "slides").on_hover_text(
                    "held up only: free to slide and spread sideways (a slippery surface), so the part \
                     can't use the ground to tie its feet together",
                );
                ui.selectable_value(g, Ground::Grip, "grips")
                    .on_hover_text("held in all directions: a grippy surface, or glued down");
            });
        });
        ui.add_enabled_ui(!self.opt_job.running(), |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.opt.overhang, "Printable without supports,")
                    .on_hover_text(
                        "no overhang shallower than 45° when printed with the chosen side up, from the \
                         design domain's face on the plate. Material is only placed where the layer below \
                         can carry it, so the part may be a little less stiff. Kept-solid regions are \
                         printed as they are: one that overhangs still needs supports.",
                    );
                let before = self.up;
                up_ui(ui, &mut self.up, "opt_up");
                if self.up != before {
                    self.save_up();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Design space").on_hover_text(
                    "2D: the problem is flattened along an axis (domain, supports and loads seen \
                     through the part), the profile is optimized, and the result is extruded back \
                     through the design domain. About 4x finer than 3D in the same time, and right for parts \
                     of constant thickness loaded in their own plane (a plate or a block pushed from \
                     the side, top or front). Wrong for anything else: forces along the axis are \
                     ignored.",
                );
                let before = self.plane2d;
                let name = |p: Option<usize>| match p {
                    None => "3D".to_string(),
                    Some(a) => format!("2D along {}", ["X", "Y", "Z"][a]),
                };
                egui::ComboBox::from_id_salt("opt_plane2d")
                    .selected_text(name(self.plane2d))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.plane2d, None, "3D");
                        for ax in 0..3 {
                            ui.selectable_value(&mut self.plane2d, Some(ax), name(Some(ax)))
                                .on_hover_text(format!(
                                    "a profile in the {} plane, extruded along {}",
                                    ["YZ", "XZ", "XY"][ax],
                                    ["X", "Y", "Z"][ax]
                                ));
                        }
                    });
                if self.plane2d != before {
                    self.save_plane2d();
                }
            });
        });
        if self.design_shape.as_ref().is_some_and(|s| s.editor) && !self.opt_job.running() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Quick start: the part")
                    .on_hover_text("adds supports and loads sized to the shape, for you to adjust");
                for t in Template::ALL {
                    if ui.small_button(t.label()).on_hover_text(t.tip()).clicked() {
                        self.apply_template(t);
                    }
                }
            });
        }
        let ready = self.optimizable();
        if !ready {
            ui.add_space(6.0);
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Not ready yet: the shape needs something holding it (a fixed spot, or the ground) and \
                 one load.",
            );
        }
        egui::CollapsingHeader::new("How to set up an optimization")
            .id_salt("opt_howto")
            .default_open(!ready)
            .show(ui, optimize_howto);
        if !ready {
            return;
        }
        let Some(shape) = self.design_shape.clone() else { return };
        if !self.defines.is_empty() {
            ui.weak(format!("for {}", define_list(&self.defines)));
        }
        if shape.editor {
            supports_and_loads_ui(ui, &shape.prims, self.opt.ground, shape.bounds);
        }
        let result = self.result_path();
        let job = self.opt_job.get();
        let mine = result.as_ref() == Some(&job.target);
        let running_here = job.running && mine;

        section(ui, "Settings");
        let auto = crate::topopt::auto_voxel(&shape, crate::topopt::auto_elements(self.gpu_solver), self.plane2d);
        ui.add_enabled_ui(!running_here, |ui| {
            let o = &mut self.opt;
            egui::Grid::new("opt").num_columns(2).show(ui, |ui| {
                ui.label("material");
                ui.add(
                    egui::Slider::new(&mut o.volfrac, 0.05..=0.8)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                        .custom_parser(|s| s.trim_end_matches('%').trim().parse::<f64>().ok().map(|v| v / 100.0)),
                )
                .on_hover_text("share of the design domain to fill with material");
                ui.end_row();
                if shape.script_load.is_some() {
                    ui.label("script load");
                    ui.add(
                        egui::DragValue::new(&mut o.force)
                            .speed(1.0)
                            .range(0.0..=1e6)
                            .suffix(" N"),
                    )
                    .on_hover_text(
                        "total force over the script's field_load region (load case 1); only scales the \
                         displacements",
                    );
                    ui.end_row();
                }
                ui.label("element size");
                ui.horizontal(|ui| {
                    let mut v = o.voxel.unwrap_or(auto);
                    let speed = v * 0.01;
                    if ui
                        .add(
                            egui::DragValue::new(&mut v)
                                .speed(speed)
                                .range(1e-3..=1e3)
                                .suffix(" mm"),
                        )
                        .on_hover_text("voxel size of the simulation; time grows quickly as it shrinks")
                        .changed()
                    {
                        o.voxel = Some(v);
                    }
                    if o.voxel.is_some() && ui.small_button("auto").clicked() {
                        o.voxel = None;
                    }
                });
                ui.end_row();
                ui.label("min. member");
                ui.add(
                    egui::Slider::new(&mut o.rmin, 1.0..=4.0)
                        .fixed_decimals(1)
                        .suffix(" elements"),
                )
                .on_hover_text("filter radius: larger gives thicker, simpler members");
                ui.end_row();
                ui.label("iterations");
                ui.add(egui::Slider::new(&mut o.iters, 5..=300))
                    .on_hover_text("at most; stops early when converged");
                ui.end_row();
                ui.label("");
                ui.checkbox(&mut o.projection, "crisp 0/1 design").on_hover_text(
                    "Heaviside projection: sharpens step by step (beta 1 to 32) so no grey, partly dense \
                     material is left. Takes more iterations; off is faster but leaves grey regions.",
                );
                ui.end_row();
            });
        });
        let vox = self.opt.voxel.unwrap_or(auto);
        let n: usize = shape
            .size()
            .iter()
            .enumerate()
            .filter(|(a, _)| Some(*a) != self.plane2d)
            .map(|(_, s)| (*s / vox).ceil().max(1.0) as usize)
            .product();
        let plane = if self.plane2d.is_some() {
            " in the 2D profile"
        } else {
            ""
        };
        ui.weak(format!(
            "≤ {n} elements{plane} (only those inside the domain are solved)"
        ));
        ui.add_space(6.0);

        if running_here {
            let p = opt_progress(&job.log);
            ui.add(
                egui::ProgressBar::new(p.iteration as f32 / self.opt.iters.max(1) as f32)
                    .text(format!("iteration {} of at most {}", p.iteration, self.opt.iters)),
            );
            let beta_max = if self.opt.projection { 32.0 } else { 0.0 };
            match p.eta(self.opt.iters, beta_max) {
                Some(e) => {
                    let text = match e.likely {
                        Some(l) if e.max - l > 1.0 => {
                            format!("about {} left (at most {})", fmt_duration(l), fmt_duration(e.max))
                        }
                        Some(_) => format!("about {} left", fmt_duration(e.max)),
                        None => format!("at most {} left", fmt_duration(e.max)),
                    };
                    ui.label(text).on_hover_text(
                        "estimated from the recent iterations' times. It may stop sooner when the \
                         design converges; \"at most\" is if it runs to the iteration limit",
                    );
                }
                None => {
                    ui.weak("estimating time left…");
                }
            }
            let mut parts = vec![];
            if let Some(c) = p.compliance {
                parts.push(format!("compliance {c}"));
            }
            if let Some(g) = p.grey {
                parts.push(format!("grey {g}"));
            }
            if let Some(b) = p.beta {
                parts.push(format!("beta {b}"));
            }
            ui.weak(parts.join(" · ")).on_hover_text(
                "compliance: lower is stiffer. grey: share of partly dense elements. beta: \
                 sharpness of the projection (the run ends at 32)",
            );
        } else {
            let busy = self.opt_job.running();
            let label = if self.existing_result().is_some() {
                "Optimize again"
            } else {
                "Optimize"
            };
            let resumable = result.as_ref().is_some_and(|r| crate::topopt::resume_path(r).exists());
            ui.horizontal(|ui| {
                if resumable
                    && primary(ui, !busy, "Resume")
                        .on_hover_text(
                            "continue the stopped run where it left off (needs the same settings; \
                             the iteration limit counts from the run's start)",
                        )
                        .clicked()
                {
                    self.start_optimize(true);
                }
                let fresh = if resumable {
                    ui.add_enabled(!busy, egui::Button::new("Start over").min_size(egui::vec2(0.0, 26.0)))
                        .on_hover_text("discard the stopped run and optimize from the beginning")
                } else {
                    primary(ui, !busy, label).on_hover_text("runs in the background; the design forms live in the view")
                };
                if fresh.clicked() {
                    self.start_optimize(false);
                }
            });
            if busy {
                ui.weak(format!("another optimization is running ({})", file_name(&job.target)));
            }
        }
        if mine {
            job_ui(ui, &self.opt_job, &job);
            if let Some(f) = opt_progress(&job.log).finished {
                ui.weak(f);
            }
        }

        section(ui, "Result");
        let Some(result) = result else { return };
        let mut notice = self.notice.take();
        let exists = file_line(ui, &result, &self.design_files(), &mut notice);
        self.notice = notice;
        if !exists {
            ui.weak("Run the optimization to get a result for these parameter values.");
            return;
        }
        if self.volume.is_some() && self.shown == result {
            ui.weak("Pick the threshold that forms the part; the mesh follows it.");
            self.volume_ui(ui);
        }
    }

    // ------------------------------------------------------------------------------ 3. Print

    fn print_ui(&mut self, ui: &mut egui::Ui) {
        let result = self.existing_result();
        if result.is_some() {
            ui.horizontal(|ui| {
                ui.label("Print");
                let before = self.print_result;
                ui.radio_value(&mut self.print_result, false, "the design");
                ui.radio_value(&mut self.print_result, true, "the optimized result");
                if before != self.print_result {
                    self.sync_shown();
                }
            });
        } else {
            self.print_result = false;
        }
        let Some(src) = self.print_source() else { return };
        let grid = is_vti(&src);
        let job = self.out_job.get();
        let busy = job.running;
        if !grid && !self.defines.is_empty() {
            ui.weak(format!("with {}", define_list(&self.defines)));
        }

        section(ui, "Mesh");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.label("Printed");
                let before = self.up;
                up_ui(ui, &mut self.up, "print_up");
                if self.up != before {
                    self.save_up();
                }
            });
            // a result is printable the way up it was optimized for
            let made_for = self
                .print_result
                .then(|| self.volume.as_ref().and_then(|v| v.up()))
                .flatten();
            if let Some(u) = made_for.filter(|u| *u != self.up) {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("The result was optimized to print {} up.", u.name()),
                    );
                    if ui.small_button(format!("Print it {} up", u.name())).clicked() {
                        self.up = u;
                        self.save_up();
                    }
                });
            }
        });
        if grid {
            match &self.volume {
                Some(v) if self.shown == src => {
                    let op = if v.below { "<" } else { ">" };
                    ui.weak(format!(
                        "surface where {} {op} {}",
                        v.array_name(),
                        fmt_num(v.iso as f64)
                    ));
                }
                _ => {}
            }
            if self.design_is_grid() {
                ui.weak("(change the threshold in Design)");
            } else {
                ui.weak("(change the threshold in Optimize)");
            }
            ui.horizontal(|ui| {
                ui.label("smoothing");
                let mut n = self.print.smooth.unwrap_or(crate::mesher::DEFAULT_SMOOTH);
                if ui
                    .add(egui::Slider::new(&mut n, 0..=50).suffix(" passes"))
                    .on_hover_text(
                        "Taubin smoothing of the mesh, after the optimized density's voxel terraces have been \
                         faired away: evens out the triangles without shrinking the part; faces on the design \
                         domain's edge stay in place (0 = off)",
                    )
                    .changed()
                {
                    self.print.smooth = Some(n);
                }
                if self.print.smooth.is_some() && ui.small_button("default").clicked() {
                    self.print.smooth = None;
                }
            });
            ui.horizontal(|ui| {
                ui.label("grow to smooth");
                ui.add(
                    egui::DragValue::new(&mut self.print.grow)
                        .speed(0.05)
                        .range(0.0..=10.0)
                        .suffix(" mm"),
                )
                .on_hover_text(
                    "let the faired surface move up to this far outward to be smoother: fills the \
                     ripples, terraces and narrow pockets the optimizer leaves (try 1-3 voxels). Walls \
                     only get thicker, so the part gets heavier and stiffer; it no longer matches the \
                     optimized design exactly (0 = off)",
                );
            });
        } else if let Some(shape) = self.design_shape.clone() {
            egui::Grid::new("mesh").num_columns(2).show(ui, |ui| {
                ui.label("voxel");
                ui.horizontal(|ui| {
                    let mut v = self.print.voxel.unwrap_or(shape.voxel);
                    let speed = v * 0.01;
                    if ui
                        .add(
                            egui::DragValue::new(&mut v)
                                .speed(speed)
                                .range(0.01..=10.0)
                                .suffix(" mm"),
                        )
                        .on_hover_text("mesh resolution: smaller is finer and slower (default: //! voxel)")
                        .changed()
                    {
                        self.print.voxel = Some(v);
                    }
                    if self.print.voxel.is_some() && ui.small_button("default").clicked() {
                        self.print.voxel = None;
                    }
                    let n: f64 = shape.size().iter().map(|s| (*s / v) as f64 + 1.0).product();
                    ui.weak(format!("{:.0}M samples", (n / 1e6).max(0.1)));
                });
                ui.end_row();
                ui.label("triangles");
                ui.horizontal(|ui| {
                    let mut on = self.print.decimate.is_some();
                    let def = shape.decimate();
                    if ui
                        .checkbox(&mut on, "reduce")
                        .on_hover_text("decimate the mesh to a fraction of its triangles")
                        .changed()
                    {
                        self.print.decimate = on.then(|| def.unwrap_or(0.5));
                    }
                    if let Some(d) = &mut self.print.decimate {
                        ui.add(
                            egui::Slider::new(d, 0.05..=1.0)
                                .custom_formatter(|v, _| format!("keep {:.0}%", v * 100.0))
                                .custom_parser(|s| {
                                    s.trim_start_matches("keep")
                                        .trim_end_matches('%')
                                        .trim()
                                        .parse::<f64>()
                                        .ok()
                                        .map(|v| v / 100.0)
                                }),
                        );
                    } else if let Some(d) = def {
                        ui.weak(format!("shape keeps {:.0}%", d * 100.0));
                    }
                });
                ui.end_row();
            });
        }
        ui.horizontal(|ui| {
            let mut on = self.print.min_body > 0.0;
            if ui
                .checkbox(&mut on, "drop small pieces")
                .on_hover_text(
                    "remove separate crumbs of material and fill small voids inside the part, smaller than \
                     this share of the largest body's volume (slicers print crumbs as blobs)",
                )
                .changed()
            {
                self.print.min_body = if on { 1.0 } else { 0.0 };
            }
            if on {
                ui.add(
                    egui::DragValue::new(&mut self.print.min_body)
                        .speed(0.1)
                        .range(0.01..=50.0)
                        .prefix("under ")
                        .suffix(" %"),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.label("format");
            ui.radio_value(&mut self.print.three_mf, false, "STL");
            ui.radio_value(&mut self.print.three_mf, true, "3MF");
            ui.add_space(8.0);
            if ui
                .add_enabled(!busy && self.loaded, egui::Button::new("Export mesh"))
                .on_hover_text("write the mesh only (slicing below also does this)")
                .clicked()
            {
                self.start_export();
            }
        });
        let inputs = if grid { vec![src.clone()] } else { self.design_files() };
        let mesh = self.mesh_path(&src);
        let mut notice = self.notice.take();
        file_line(ui, &mesh, &inputs, &mut notice);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.print.show_mesh, "show the mesh")
                .on_hover_text("draw the exported mesh in the view instead of the shape");
            if self.print.show_mesh {
                ui.checkbox(&mut self.print.edges, "edges")
                    .on_hover_text("outline every triangle");
                let shown = self
                    .win
                    .as_ref()
                    .and_then(|w| w.mesh.as_ref())
                    .filter(|m| m.path == mesh);
                match shown.map(|m| (&m.gpu, m.tris)) {
                    _ if !mesh.exists() => {
                        ui.weak("export it first");
                    }
                    Some((Ok(_), n)) => {
                        ui.weak(format!("{} triangles", crate::mesher::fmt_count(n)));
                    }
                    Some((Err(e), _)) => {
                        ui.colored_label(ui.visuals().warn_fg_color, e.as_str());
                    }
                    None => {}
                }
            }
        });
        if job.cmd != "open"
            && job.target == mesh
            && let Some(r) = mesh_report(&job.log)
        {
            ui.weak(r);
        }

        section(ui, "Slice");
        match &self.slicer {
            Some((_, exe)) => {
                ui.weak(format!("slicer: {}", exe.display()));
            }
            None => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "No slicer found. Install PrusaSlicer (sudo pacman -S prusa-slicer), or set \
                     GEN3D_SLICER to a PrusaSlicer/OrcaSlicer executable.",
                );
            }
        }
        let auto_label = match self.design_shape.as_ref() {
            Some(s) if !grid || self.print_result => {
                let c = crate::resolve_configs(&[], s);
                if c.is_empty() {
                    "slicer defaults".to_string()
                } else {
                    c.iter().map(|p| file_name(p)).collect::<Vec<_>>().join(", ")
                }
            }
            _ => {
                let d = Path::new("printers/default.ini");
                if d.exists() {
                    "default.ini".into()
                } else {
                    "slicer defaults".into()
                }
            }
        };
        let profiles = self.print.profiles.clone();
        egui::ComboBox::from_label("printer profile")
            .selected_text(match &self.print.profile {
                Some(p) => file_name(p),
                None => format!("auto: {auto_label}"),
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.print.profile, None, format!("auto: {auto_label}"))
                    .on_hover_text("the shape's //! slicer directive, else printers/default.ini");
                for p in &profiles {
                    ui.selectable_value(&mut self.print.profile, Some(p.clone()), file_name(p));
                }
            })
            .response
            .on_hover_text(
                "profiles are read from printers/ (PrusaSlicer: File > Export > Export Config, see \
                 printers/README.md)",
            );
        if auto_label == "slicer defaults" && self.print.profile.is_none() {
            ui.weak("no profile: the slicer's generic printer; export yours to printers/default.ini");
        }
        if matches!(self.slicer, Some((slicer::Kind::Prusa, _))) {
            egui::Grid::new("overrides").num_columns(2).show(ui, |ui| {
                ui.label("layer height");
                egui::ComboBox::from_id_salt("layer")
                    .selected_text(self.print.layer.map_or("profile".into(), |l| format!("{l} mm")))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.print.layer, None, "profile");
                        for l in [0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3] {
                            ui.selectable_value(&mut self.print.layer, Some(l), format!("{l} mm"));
                        }
                    });
                ui.end_row();
                ui.label("infill");
                egui::ComboBox::from_id_salt("infill")
                    .selected_text(self.print.infill.map_or("profile".into(), |i| format!("{i}%")))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.print.infill, None, "profile");
                        for i in [0, 5, 10, 15, 20, 30, 40, 60, 100] {
                            ui.selectable_value(&mut self.print.infill, Some(i), format!("{i}%"));
                        }
                    });
                ui.end_row();
                ui.label("supports");
                let name = |s: Supports| match s {
                    Supports::Profile => "profile",
                    Supports::Off => "off",
                    Supports::Everywhere => "everywhere",
                    Supports::BuildPlate => "from the plate only",
                };
                egui::ComboBox::from_id_salt("supports")
                    .selected_text(name(self.print.supports))
                    .show_ui(ui, |ui| {
                        for s in [
                            Supports::Profile,
                            Supports::Off,
                            Supports::Everywhere,
                            Supports::BuildPlate,
                        ] {
                            ui.selectable_value(&mut self.print.supports, s, name(s));
                        }
                    });
                ui.end_row();
            });
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let ok = !busy && self.loaded && self.slicer.is_some();
            if primary(ui, ok, "Slice to G-code")
                .on_hover_text("mesh and slice in the background")
                .clicked()
            {
                self.start_slice();
            }
            if ui
                .add_enabled(ok, egui::Button::new("Open in slicer"))
                .on_hover_text("mesh and open it in the slicer's window, to arrange or tweak it there")
                .clicked()
            {
                self.start_open();
            }
        });
        let gcode = self.out_path(&src, ".gcode");
        let mut gcode_inputs = inputs.clone();
        gcode_inputs.extend(self.print.profile.clone());
        if file_line(ui, &gcode, &gcode_inputs, &mut notice) && !(busy && job.target == gcode) {
            let summary = self.gcode_summary(&gcode);
            if !summary.is_empty() {
                egui::Grid::new("gcode").num_columns(2).show(ui, |ui| {
                    for (k, v) in summary {
                        ui.label(k);
                        ui.strong(v);
                        ui.end_row();
                    }
                });
            }
        }
        self.notice = notice;
        if !job.cmd.is_empty() {
            ui.add_space(6.0);
            job_ui(ui, &self.out_job, &job);
        }
    }

    // ---------------------------------------------------------------------------------- frame

    /// Run the UI for one frame and the work that follows it (pipeline rebuilds).
    fn run_ui(&mut self, raw: egui::RawInput) -> Option<egui::FullOutput> {
        let win = self.win.as_ref()?;
        let ctx = win.egui.clone();
        let tex = win.viewport.id;
        let out = ctx.run_ui(raw, |ui| self.ui(ui, tex));
        if std::mem::take(&mut self.params_dirty) {
            self.sync_params();
        }
        self.poll_rebuild();
        // time only moves while something animates, so a still view keeps identical uniforms
        if self.animating() {
            self.opts.time = self.start.elapsed().as_secs_f32();
        }
        Some(out)
    }

    /// Draw the preview and the UI into `target` (`None`: only apply egui's texture changes).
    fn paint(&mut self, mut out: egui::FullOutput, target: Option<(&wgpu::TextureView, [u32; 2])>) {
        if target.is_some() {
            self.sync_mesh();
        }
        let Some(win) = &mut self.win else { return };
        // egui's texture uploads must be applied (and frees honoured) even if this frame isn't drawn
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in &deltas {
                win.renderer.update_texture(&win.gpu.device, &win.gpu.queue, id, delta);
            }
        }
        let free: Vec<_> = out.textures_delta.free.drain().collect();
        let Some((view, size_in_pixels)) = target else {
            free.iter().for_each(|id| win.renderer.free_texture(id));
            return;
        };
        if win.viewport.size != self.view_px {
            win.touch();
            win.viewport.size = self.view_px;
            win.viewport.view = Viewport::target(&win.gpu, self.view_px);
            win.renderer.update_egui_texture_from_wgpu_texture(
                &win.gpu.device,
                &win.viewport.view,
                wgpu::FilterMode::Linear,
                win.viewport.id,
            );
        }
        // the viewport is drawn into its own texture, which egui shows: redrawn only when what it
        // shows changed, so UI-only frames (hover, the job spinner) don't raymarch the preview
        let mesh = win.mesh.as_ref().and_then(|m| m.gpu.as_ref().ok());
        let shown_mesh = mesh.is_some() && win.meshr.is_some();
        let u = if shown_mesh {
            crate::viewer::ViewUniforms::for_mesh(
                self.shape.bounds,
                &self.cam,
                win.viewport.size,
                true,
                self.print.edges,
            )
        } else {
            uniforms(&self.shape, &self.cam, &self.opts, win.viewport.size, true)
        };
        let mut key = bytemuck::bytes_of(&u).to_vec();
        key.push(shown_mesh as u8);
        // a change other than the animation's time restarts the animation
        let still = if shown_mesh {
            key.clone()
        } else {
            let t = std::mem::replace(&mut self.opts.time, 0.0);
            let u = uniforms(&self.shape, &self.cam, &self.opts, win.viewport.size, true);
            self.opts.time = t;
            bytemuck::bytes_of(&u).to_vec()
        };
        if win.dirty.get() || still != win.drawn_still {
            win.drawn_still = still;
            self.anim_until = Some(Instant::now() + Duration::from_secs_f32(ANIMATION_SECS));
        }
        let preview = (win.dirty.get() || key != win.drawn).then(|| {
            win.gpu.queue.write_buffer(&win.ubuf, 0, bytemuck::bytes_of(&u));
            win.dirty.set(false);
            win.drawn = key;
            self.preview_frames += 1;
            match (mesh, &mut win.meshr) {
                (Some(m), Some(r)) => r.draw(&win.gpu, m, &win.viewport.view, win.viewport.size),
                _ => draw(&win.gpu, win.pipeline.as_ref(), &win.viewport.view),
            }
        });

        let (device, queue) = (&win.gpu.device, &win.gpu.queue);
        let jobs = win
            .egui
            .tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels,
            pixels_per_point: out.pixels_per_point,
        };
        let mut enc = device.create_command_encoder(&Default::default());
        let extra = win.renderer.update_buffers(device, queue, &mut enc, &jobs, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            win.renderer.render(&mut pass, &jobs, &screen);
        }
        queue.submit(preview.into_iter().chain(extra).chain([enc.finish()]));
        free.iter().for_each(|id| win.renderer.free_texture(id));
    }

    /// Whether the preview animates: the moving bands of a shown vector field overlay (the load
    /// directions), for `ANIMATION_SECS` after the view last changed. Only then does its time
    /// uniform advance and the viewport redraw (at `ANIMATION_FPS`). The bands hold still while an
    /// optimization runs, which needs the GPU.
    fn animating(&self) -> bool {
        if self.opt_job.running() || self.anim_until.is_none_or(|t| Instant::now() >= t) {
            return false;
        }
        let mesh_shown = self
            .win
            .as_ref()
            .is_some_and(|w| w.meshr.is_some() && w.mesh.as_ref().is_some_and(|m| m.gpu.is_ok()));
        !mesh_shown
            && self
                .fields
                .iter()
                .enumerate()
                .any(|(i, f)| f.comps == 3 && i < 32 && self.opts.fields >> i & 1 == 1)
    }

    /// Decide when the next frame is due after one was drawn: egui's request, continuously while
    /// the preview animates, soon while a pipeline compiles (to pick it up), and once more after
    /// the viewport stops redrawing (so the fps readout clears). Otherwise frames come only from
    /// input and from the file/job polling in `about_to_wait`.
    fn schedule(&mut self, egui_delay: Duration) {
        let now = Instant::now();
        let mut next = (egui_delay < Duration::from_secs(3600)).then(|| now + egui_delay);
        let mut at = |t: Instant| next = Some(next.map_or(t, |n| n.min(t)));
        if self.animating() {
            at(now + Duration::from_secs_f32(1.0 / ANIMATION_FPS));
        }
        if self.rebuild.is_some() {
            at(now + Duration::from_millis(30));
        }
        if self.fps > 0.0 {
            at(self.fps_t + Duration::from_millis(500));
        }
        self.next_frame = next;
    }

    /// What the 250 ms poll watches besides the shown files: jobs starting or ending, and the
    /// mesh to show (an export finishing, or the file changing on disk).
    fn watch_state(&self) -> (bool, bool, Option<(PathBuf, Option<SystemTime>)>) {
        (
            self.out_job.running(),
            self.opt_job.running(),
            self.wanted_mesh().map(|p| {
                let m = mtime(&p);
                (p, m)
            }),
        )
    }

    fn request_redraw(&self) {
        if let Some(os) = self.win.as_ref().and_then(|w| w.os.as_ref()) {
            os.window.request_redraw();
        }
    }

    fn frame(&mut self) {
        let Some(os) = self.win.as_mut().and_then(|w| w.os.as_mut()) else {
            return;
        };
        let raw = os.egui_state.take_egui_input(&os.window);
        let Some(mut out) = self.run_ui(raw) else { return };
        // egui's next repaint: now for animations and the job spinner, later for tooltips, never
        // when idle
        let repaint_delay = out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        let title = match &self.design {
            Some(d) => format!("gen3d — {}", file_name(d)),
            None => "gen3d".into(),
        };
        let Some(win) = &mut self.win else { return };
        let Some(os) = &mut win.os else { return };
        if win.title != title {
            os.window.set_title(&title);
            win.title = title;
        }
        os.egui_state
            .handle_platform_output(&os.window, std::mem::take(&mut out.platform_output));
        let frame = match os.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                os.surface.configure(&win.gpu.device, &os.config);
                return self.paint(out, None);
            }
            _ => return self.paint(out, None),
        };
        let size = [os.config.width, os.config.height];
        let (window, gpu) = (os.window.clone(), win.gpu.clone());
        let view = frame.texture.create_view(&Default::default());
        self.paint(out, Some((&view, size)));
        window.pre_present_notify();
        gpu.queue.present(frame);

        let dt = self.fps_t.elapsed().as_secs_f32();
        if dt >= 0.5 {
            self.fps = self.preview_frames as f32 / dt;
            self.preview_frames = 0;
            self.fps_t = Instant::now();
        }
        self.schedule(repaint_delay);
    }

    fn open_window(&mut self, el: &ActiveEventLoop) -> Result<Win> {
        let attrs = Window::default_attributes()
            .with_title("gen3d")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(el.create_window(attrs)?);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone())?;
        let gpu = Arc::new(crate::gpu::create(instance, Some(&surface), true)?);
        let info = gpu.adapter.get_info();
        self.gpu_name = format!("{} ({:?})", info.name, info.backend);
        self.gpu_solver = !gpu.is_software();
        let caps = surface.get_capabilities(&gpu.adapter);
        // egui blends in gamma space: prefer a non-sRGB swapchain format
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("surface not supported by the GPU adapter"))?;
        config.format = format;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&gpu.device, &config);
        let egui_state = egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let os = OsWindow {
            window,
            surface,
            config,
            egui_state,
        };
        Ok(Win::new(gpu, format, Some(os)))
    }

    /// Draw the app offscreen at `size` (logical pixels = physical) and save it as a PNG.
    fn snapshot_png(&mut self, size: (u32, u32), out: &Path) -> Result<()> {
        let gpu = self.win.as_ref().map(|w| w.gpu.clone()).context("no GPU")?;
        let (w, h) = size;
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("snapshot"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VIEW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        // a few frames: egui sizes some widgets from the previous frame's layout
        // (time advances a second per frame so fade-ins finish)
        for i in 0..4 {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w as f32, h as f32),
                )),
                time: Some(i as f64),
                predicted_dt: 1.0,
                ..Default::default()
            };
            let Some(o) = self.run_ui(raw) else { break };
            self.paint(o, Some((&view, [w, h])));
            // let a parameter rebuild started by this frame land before the next one
            while self.rebuild.is_some() {
                std::thread::sleep(Duration::from_millis(10));
                self.poll_rebuild();
            }
        }
        crate::viewer::save_texture_png(&gpu, &tex, size, out)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.win.is_some() {
            return;
        }
        match self.open_window(el) {
            Ok(w) => {
                self.win = Some(w);
                self.reload(true);
                if !self.loaded {
                    self.reset_camera();
                }
                self.request_redraw();
            }
            Err(e) => {
                eprintln!("\x1b[31merror:\x1b[0m could not open the window: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = &mut self.win else { return };
        let Some(os) = &mut win.os else { return };
        let response = os.egui_state.on_window_event(&os.window, &event);
        // (egui asks to repaint after any event, the redraw itself included: that would loop)
        if response.repaint && !matches!(event, WindowEvent::RedrawRequested) {
            os.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                os.config.width = size.width.max(1);
                os.config.height = size.height.max(1);
                os.surface.configure(&win.gpu.device, &os.config);
                os.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !win.egui.egui_wants_keyboard_input()
                    && !self.browser.open
                    && self.design.is_some() =>
            {
                let ctrl = win.egui.input(|i| i.modifiers.command);
                if let PhysicalKey::Code(code) = event.physical_key
                    && !ctrl
                {
                    self.on_key(code);
                    self.request_redraw();
                }
            }
            // drawn on demand: `frame` schedules the next one (see `schedule`)
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        let poll = Duration::from_millis(250);
        if now - self.last_poll > poll {
            self.last_poll = now;
            let before = (self.shown.clone(), self.watch_state());
            // a result appearing (first iteration of an optimization) or vanishing switches the view
            self.sync_shown();
            let mut changed = false;
            if self.mtimes.iter().any(|(f, m)| mtime(f) != *m) {
                std::thread::sleep(Duration::from_millis(50)); // let editors finish writing
                self.reload(false);
                changed = true;
            }
            if changed || before != (self.shown.clone(), self.watch_state()) {
                self.request_redraw();
            }
        }
        if self.next_frame.is_some_and(|t| t <= now) {
            self.next_frame = None;
            self.request_redraw();
        }
        let wake = self.last_poll + poll;
        let wake = self.next_frame.map_or(wake, |t| t.min(wake));
        el.set_control_flow(ControlFlow::WaitUntil(wake));
    }
}

/// What `snapshot` sets up before drawing.
pub struct SnapshotSetup {
    pub step: String,
    pub print_result: bool,
    pub browser: bool,
    pub section: bool,
    /// select this primitive (Design step)
    pub select: Option<usize>,
    /// Print step: show the exported mesh (with its edges)
    pub mesh: bool,
    pub edges: bool,
    pub up: Option<Up>,
}

/// Draw the app's window offscreen, without opening one, and save it as a PNG: for checking the
/// UI headlessly (`gen3d snapshot`). Same GPU code paths as the window.
pub fn snapshot(
    file: Option<&Path>,
    defines: Vec<(String, f64)>,
    setup: &SnapshotSetup,
    size: (u32, u32),
    out: &Path,
) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut app = App::new(cwd.clone());
    let gpu = Arc::new(crate::gpu::headless()?);
    let info = gpu.adapter.get_info();
    app.gpu_name = format!("{} ({:?})", info.name, info.backend);
    app.gpu_solver = !gpu.is_software();
    app.win = Some(Win::new(gpu, VIEW_FORMAT, None));
    match file {
        Some(f) => app.open_design(f, defines),
        None => app.browser.show_at(&cwd.join("shapes")),
    }
    if !app.loaded {
        app.reset_camera();
    }
    app.print_result = setup.print_result;
    app.opts.section = setup.section;
    app.print.show_mesh = setup.mesh;
    app.print.edges = setup.edges;
    if let Some(u) = setup.up {
        app.up = u;
    }
    match setup.step.as_str() {
        "design" => {}
        "optimize" => app.set_step(Step::Optimize),
        "print" => app.set_step(Step::Print),
        s => anyhow::bail!("unknown step `{s}` (design, optimize, print)"),
    }
    if setup.select.is_some() {
        app.select(setup.select);
    }
    if setup.browser {
        let dir = app.browser.dir.clone();
        app.browser.show_at(&dir);
    }
    app.snapshot_png(size, out)
}

/// Open the app, optionally with a file.
pub fn run(file: Option<&Path>, defines: Vec<(String, f64)>) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut app = App::new(cwd.clone());
    match file {
        Some(f) => {
            let p = f
                .canonicalize()
                .with_context(|| format!("cannot open {}", f.display()))?;
            if !crate::browser::openable(&p) {
                anyhow::bail!("{} is not a .wgsl shape or .vti grid", p.display());
            }
            app.open_design(&p, defines);
        }
        None => {
            let shapes = cwd.join("shapes");
            app.browser.show_at(if shapes.is_dir() { &shapes } else { &cwd });
        }
    }
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_gcode_summary() {
        let lines: Vec<String> = [
            "estimated printing time (normal mode) = 1h 2m 3s",
            "filament used [g] = 12.34",
            "filament used [mm] = 4123.4",
            "fill_density = 15%",
            "support_material = 0",
            "printer_model = ",
        ]
        .map(String::from)
        .to_vec();
        let p = pretty_summary(&lines);
        assert_eq!(p[0], ("print time".into(), "1h 2m 3s".into()));
        assert_eq!(p[1], ("filament".into(), "12.34 g".into()));
        assert_eq!(p[2], ("filament length".into(), "4.12 m".into()));
        assert_eq!(p[3], ("infill".into(), "15%".into()));
        assert_eq!(p[4], ("supports".into(), "off".into()));
        assert_eq!(p.len(), 5, "an empty printer model is dropped");
    }

    /// Add, click-select, drag along X with the gizmo, undo; through egui input like a user would.
    /// Skipped without a GPU adapter.
    #[test]
    fn primitive_editor() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-editor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wgsl");
        std::fs::write(
            &path,
            "//! bounds -20 -20 0 20 20 20\nfn map(p: vec3f) -> f32 { return length(p - vec3f(0, 0, 10)) - 3.0; }\n",
        )
        .unwrap();
        let mut app = App::new(dir.clone());
        app.win = Some(Win::new(Arc::new(gpu), VIEW_FORMAT, None));
        app.open_design(&path, vec![]);
        assert!(app.loaded, "{:?}", app.error);
        let mut t = 0.0;
        let mut frame = |app: &mut App, events: Vec<egui::Event>| {
            t += 0.05;
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))),
                time: Some(t),
                predicted_dt: 0.05,
                events,
                ..Default::default()
            };
            if let Some(out) = app.run_ui(raw) {
                app.paint(out, None);
            }
            while app.rebuild.is_some() {
                std::thread::sleep(Duration::from_millis(5));
                app.poll_rebuild();
            }
            assert!(
                app.param_error.is_none() && app.error.is_none(),
                "{:?} {:?}",
                app.param_error,
                app.error
            );
        };
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };

        app.add_prim(Kind::Box);
        let sidecar = prims::sidecar(&app.design.clone().unwrap());
        assert!(sidecar.exists());
        assert_eq!(app.edit.sel, Some(0));
        let mut ps = app.prims();
        ps[0].pos = [12.0, 0.0, 10.0];
        app.set_prims(ps);
        assert!(app.shape.bounds.1[0] > 14.0, "bounds grow: {:?}", app.shape.bounds);
        app.select(None);
        frame(&mut app, vec![]);
        frame(&mut app, vec![]);

        // click on the box selects it
        app.select(Some(0));
        let centre = app.gizmo(app.view_rect).unwrap().centre;
        app.select(None);
        frame(&mut app, vec![egui::Event::PointerMoved(centre)]);
        frame(&mut app, vec![button(centre, true)]);
        frame(&mut app, vec![button(centre, false)]);
        assert_eq!(app.edit.sel, Some(0), "click picks the box");

        // drag the X arrow: only x changes, the camera stays put
        let yaw = app.cam.yaw;
        let rebuilds = app.rebuilds;
        let g = app.gizmo(app.view_rect).unwrap();
        let tip = g.tips[0];
        let along = (g.tips[0] - g.centre).normalized();
        frame(&mut app, vec![egui::Event::PointerMoved(tip)]);
        frame(&mut app, vec![button(tip, true)]);
        let mut at = tip;
        for _ in 0..6 {
            at += along * 10.0;
            frame(&mut app, vec![egui::Event::PointerMoved(at)]);
        }
        frame(&mut app, vec![button(at, false)]);
        let p = app.prims()[0].clone();
        assert!(p.pos[0] > 13.0 && p.pos[1] == 0.0 && p.pos[2] == 10.0, "{:?}", p.pos);
        assert_eq!(app.cam.yaw, yaw);
        assert_eq!(
            app.rebuilds, rebuilds,
            "moving and selecting write a buffer, no pipeline build"
        );
        assert_eq!(prims::load(&sidecar).unwrap()[0].pos, p.pos, "saved");
        assert_eq!(app.design_shape.as_ref().unwrap().prims[0].pos, p.pos);

        // switching to subtract is a value change: saved, no pipeline build, no bounds growth
        let rebuilds = app.rebuilds;
        let mut ps = app.prims();
        ps[0].op = Op::Subtract;
        app.edit.last_edit = None; // as the sidebar does: its own undo step
        app.set_prims(ps);
        frame(&mut app, vec![]);
        assert_eq!(app.rebuilds, rebuilds);
        assert_eq!(prims::load(&sidecar).unwrap()[0].op, Op::Subtract);
        assert_eq!(app.shape.bounds, app.shape.base_bounds);
        app.undo(false);
        assert_eq!(app.prims()[0].op, Op::Union);

        // the whole drag is one undo step
        app.undo(false);
        assert_eq!(app.prims()[0].pos, [12.0, 0.0, 10.0]);
        app.undo(true);
        assert_eq!(app.prims()[0].pos, p.pos);
        app.delete_prim(0);
        assert!(app.prims().is_empty() && app.loaded);
        app.undo(false);
        assert_eq!(app.prims().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// f32 parameters update through a uniform buffer; i32 ones rebuild the pipeline.
    #[test]
    fn float_params_are_live() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("gen3d-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wgsl");
        std::fs::write(
            &path,
            "//! bounds -20 -20 0 20 20 20\noverride R: f32 = 3.0;\noverride N: i32 = 2;\n\
             fn map(p: vec3f) -> f32 {\n  var d = 1e9;\n  for (var i = 0; i < N; i++) {\n    \
             d = min(d, length(p - vec3f(f32(i) * 8.0, 0, 10)) - R);\n  }\n  return d;\n}\n",
        )
        .unwrap();
        let mut app = App::new(dir.clone());
        app.win = Some(Win::new(Arc::new(gpu), VIEW_FORMAT, None));
        app.open_design(&path, vec![]);
        assert!(app.loaded, "{:?}", app.error);
        assert_eq!(app.win.as_ref().unwrap().live, ["R"]);
        let settle = |app: &mut App| {
            app.sync_params();
            while app.rebuild.is_some() {
                std::thread::sleep(Duration::from_millis(5));
                app.poll_rebuild();
            }
            assert!(app.param_error.is_none(), "{:?}", app.param_error);
        };
        let (r, n) = (app.params[0].clone(), app.params[1].clone());
        let before = app.rebuilds;
        for v in [3.5, 4.0, 5.0] {
            app.set_param(&r, v);
            settle(&mut app);
        }
        assert_eq!(app.rebuilds, before, "f32 parameters don't rebuild");
        app.set_param(&n, 3.0);
        settle(&mut app);
        assert_eq!(app.rebuilds, before + 1, "i32 parameters do");
        app.set_param(&r, 2.0);
        settle(&mut app);
        assert_eq!(app.rebuilds, before + 1);
        // the window title / jobs still see every value
        assert_eq!(app.shape.defines.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn result_and_output_paths_follow_parameters() {
        let mut app = App::new(PathBuf::from("/tmp"));
        app.design = Some(PathBuf::from("/x/shapes/bracket.wgsl"));
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(app.result_path(), Some(cwd.join("out/bracket_topopt.vti")));
        app.defines = vec![("T".into(), 4.0)];
        assert_eq!(app.result_path(), Some(cwd.join("out/bracket_T-4_topopt.vti")));
        let r = app.result_path().unwrap();
        assert_eq!(app.mesh_path(&r), cwd.join("out/bracket_T-4_topopt.stl"));
        assert_eq!(
            app.mesh_path(Path::new("/x/shapes/bracket.wgsl")),
            cwd.join("out/bracket_T-4.stl")
        );
        // a grid opened directly has no optimization step
        app.design = Some(PathBuf::from("/x/out/sim.vti"));
        assert_eq!(app.result_path(), None);
    }
}
