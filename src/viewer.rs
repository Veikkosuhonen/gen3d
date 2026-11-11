//! Live raymarching preview window with hot reload, and headless PNG rendering.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::gpu::Gpu;
use crate::shape::{Field, Param, ParamKind, Shape, load_shape};
use crate::volume::{Volume, VolumeOpts};

pub const PREVIEW_HARNESS: &str = r#"
// ---- gen3d preview harness ----
struct G3dView {
    res: vec2f,
    relax: f32,       // raymarch step factor for non-exact SDFs
    cut: f32,         // section plane y
    eye: vec4f,
    look: vec4f,
    bmin: vec4f,
    bmax: vec4f,
    flags: vec4u,     // clip to bounds, section, show bounds, apply gamma
    fmask: vec4u,     // x: bitmask of field overlays to show
    misc: vec4f,      // x: time in seconds, y: inward offset for field lookups (half a voxel)
}
@group(0) @binding(0) var<uniform> g3d_v: G3dView;

@vertex
fn g3d_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn g3d_boxSd(p: vec3f, lo: vec3f, hi: vec3f) -> f32 {
    let c = 0.5 * (lo + hi);
    let b = 0.5 * (hi - lo);
    let q = abs(p - c) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// returns (distance, id); id 0 = shape surface, 1 = section cut face
fn g3d_scene(p: vec3f) -> vec2f {
    var d = map(p);
    if (g3d_v.flags.x == 1u) { d = max(d, g3d_boxSd(p, g3d_v.bmin.xyz, g3d_v.bmax.xyz)); }
    if (g3d_v.flags.y == 1u) {
        let c = g3d_v.cut - p.y;    // remove y < cut (the half facing the default camera)
        if (c > d) { return vec2f(c, 1.0); }
    }
    return vec2f(d, 0.0);
}
fn g3d_sceneD(p: vec3f) -> f32 { return g3d_scene(p).x; }

fn g3d_normal(p: vec3f, h: f32) -> vec3f {
    let k = vec2f(1.0, -1.0);
    return normalize(k.xyy * g3d_sceneD(p + k.xyy * h) + k.yyx * g3d_sceneD(p + k.yyx * h) +
                     k.yxy * g3d_sceneD(p + k.yxy * h) + k.xxx * g3d_sceneD(p + k.xxx * h));
}

fn g3d_boxHit(ro: vec3f, rd: vec3f, lo: vec3f, hi: vec3f) -> vec2f {
    let inv = 1.0 / rd;
    let t0 = (lo - ro) * inv;
    let t1 = (hi - ro) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

fn g3d_softShadow(ro: vec3f, rd: vec3f, tmax: f32) -> f32 {
    var res = 1.0;
    var t = 0.05;
    for (var i = 0; i < 64; i++) {
        if (t >= tmax) { break; }
        let h = g3d_sceneD(ro + rd * t);
        res = min(res, 10.0 * h / t);
        if (res < 0.01) { break; }
        t += clamp(h * g3d_v.relax, 0.05, 5.0);
    }
    return clamp(res, 0.0, 1.0);
}

fn g3d_ao(p: vec3f, n: vec3f, scale: f32) -> f32 {
    var o = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 5; i++) {
        let h = scale * f32(i);
        o += w * (h - g3d_sceneD(p + n * h));
        w *= 0.6;
    }
    return clamp(1.0 - o / scale * 0.35, 0.0, 1.0);
}

fn g3d_gridLine(p: vec2f, fw: vec2f, spacing: f32, width: f32) -> f32 {
    let g = abs(fract(p / spacing - 0.5) - 0.5) * spacing;
    let l = 1.0 - smoothstep(vec2f(0.0), fw * width, g);
    return max(l.x, l.y);
}

fn g3d_edgeLines(p: vec3f, lo: vec3f, hi: vec3f, w: f32) -> f32 {
    // distance to the 12 edges of the bounds box, measured in the plane of the face
    let a = min(abs(p - lo), abs(p - hi));
    let s = step(a, vec3f(w));
    return clamp(s.x * s.y + s.y * s.z + s.x * s.z, 0.0, 1.0);
}

@fragment
fn g3d_fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let res = g3d_v.res;
    let fc = vec2f(frag.x, res.y - frag.y);
    let uv = (fc - 0.5 * res) / res.y;
    let eye = g3d_v.eye.xyz;
    let bmin = g3d_v.bmin.xyz;
    let bmax = g3d_v.bmax.xyz;
    let clip = g3d_v.flags.x == 1u;
    let fw = normalize(g3d_v.look.xyz - eye);
    let side = cross(fw, vec3f(0, 0, 1));
    let rt = select(normalize(side), vec3f(1, 0, 0), length(side) < 1e-4);
    let up = cross(rt, fw);
    let ro = eye;
    let rd = normalize(fw * 1.8 + uv.x * rt + uv.y * up);

    let scale = length(bmax - bmin);
    var col = mix(vec3f(0.10, 0.11, 0.13), vec3f(0.20, 0.22, 0.26), 0.5 + 0.5 * rd.z);

    // ground plane z = 0; screen-space derivatives must be taken in uniform control flow
    let tPlane = select(1e9, -ro.z / rd.z, rd.z < 0.0);
    let pPlane = ro + rd * min(tPlane, scale * 1000.0);
    let dPlane = fwidth(pPlane.xy);

    // march only inside an expanded bounds box (or a large region if not clipping)
    let pad = vec3f(scale * select(0.5, 0.01, clip));
    let tb = g3d_boxHit(ro, rd, bmin - pad, bmax + pad);
    var t = max(tb.x, 0.0);
    let tEnd = min(tb.y, scale * 20.0);
    var hit = false;
    var id = 0.0;
    let eps = 1e-4 * scale;
    for (var i = 0; i < 400; i++) {
        if (t >= tEnd) { break; }
        let s = g3d_scene(ro + rd * t);
        id = s.y;
        if (abs(s.x) < eps * max(1.0, t / scale)) { hit = true; break; }
        t += s.x * g3d_v.relax;
    }

    let light = normalize(vec3f(0.6, -0.4, 0.9));
    if (hit && t < tPlane + eps) {
        let p = ro + rd * t;
        let n = g3d_normal(p, eps * 2.0);
        var base = select(vec3f(0.95, 0.62, 0.22), vec3f(0.85, 0.25, 0.22), id == 1.0);
        if (id == 1.0) { // section face: show internal distance field as bands
            let di = map(p);
            base *= 0.75 + 0.25 * smoothstep(-0.1, 0.1, sin(di * 6.2831));
        }
        // look fields up just inside the surface, where the exported grid samples them; regions
        // written as shells around a hole (max(d - t, -d)) have the hole wall on their boundary
        let ft = g3d_field_tint(p - n * g3d_v.misc.y, p, scale);
        base = mix(base, ft.rgb, ft.a);
        let dif = clamp(dot(n, light), 0.0, 1.0);
        let sh = g3d_softShadow(p + n * eps * 4.0, light, scale * 2.0);
        let occ = g3d_ao(p, n, scale * 0.01);
        let back = clamp(dot(n, -light), 0.0, 1.0);
        let rim = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 3.0);
        let h = normalize(light - rd);
        let spe = pow(clamp(dot(n, h), 0.0, 1.0), 48.0) * dif * sh;
        let sky = (0.20 + 0.10 * n.z) * mix(0.35, 1.0, occ);   // hemisphere ambient
        let bounce = 0.10 * clamp(-n.z, 0.0, 1.0);              // light bounced off the bed
        col = base * (dif * sh + sky + bounce + 0.15 * back * occ) + 0.25 * rim * occ + 0.3 * spe;
    } else if (tPlane < 1e8) {
        let p = pPlane;
        let sh = g3d_softShadow(p + vec3f(0, 0, eps * 4.0), light, scale * 2.0);
        let minor = g3d_gridLine(p.xy, dPlane, 1.0, 1.0) * 0.25;
        let major = g3d_gridLine(p.xy, dPlane, 10.0, 1.5);
        let axisX = 1.0 - smoothstep(0.0, dPlane.x * 2.0, abs(p.x));
        let axisY = 1.0 - smoothstep(0.0, dPlane.y * 2.0, abs(p.y));
        var g = vec3f(0.17, 0.18, 0.20);
        g += vec3f(0.05) * minor + vec3f(0.12) * major;
        g = mix(g, vec3f(0.8, 0.25, 0.25), axisY * 0.8);   // x axis (y = 0) red
        g = mix(g, vec3f(0.25, 0.75, 0.3), axisX * 0.8);   // y axis (x = 0) green
        let fade = exp(-tPlane / (scale * 6.0));
        col = mix(col, g * (0.45 + 0.55 * sh), fade);
    }

    if (g3d_v.flags.z == 1u) {
        let bh = g3d_boxHit(ro, rd, bmin, bmax);
        if (bh.x < bh.y && bh.y > 0.0) {
            let w = scale * 0.002;
            // shapes often fill their bounds exactly; a tolerance keeps box edges that lie on the
            // surface consistently visible instead of flickering against it
            let tFar = select(1e9, t + w * 2.0 * max(1.0, t / scale), hit);
            for (var k = 0; k < 2; k++) {
                let tb2 = select(bh.y, bh.x, k == 0);
                if (tb2 > 0.0 && tb2 < tFar) {
                    let e = g3d_edgeLines(ro + rd * tb2, bmin, bmax, w * max(1.0, tb2 / scale));
                    col = mix(col, vec3f(0.4, 0.7, 1.0), e * 0.6);
                }
            }
        }
    }

    if (g3d_v.flags.w == 1u) { col = pow(max(col, vec3f(0.0)), vec3f(0.4545)); }
    return vec4f(col, 1.0);
}
"#;

/// Overlay colours for `field_*` regions (linear RGB), cycled by declaration order. Chosen to stand
/// apart from the orange surface and red section face.
const FIELD_COLORS: &[[f32; 3]] = &[
    [0.10, 0.30, 0.95],
    [0.15, 0.75, 0.20],
    [0.75, 0.15, 0.80],
    [0.05, 0.70, 0.75],
    [0.90, 0.85, 0.10],
    [0.95, 0.95, 0.95],
];

fn field_color(i: usize) -> [f32; 3] {
    FIELD_COLORS[i % FIELD_COLORS.len()]
}

/// The preview harness plus `g3d_field_tint`, which colours surface points inside the shape's
/// `field_*` regions: scalar fields where they are < 0, vector fields where they are non-zero (with
/// bands moving along the vector). Overlapping regions alternate in diagonal stripes fixed to the surface (in world space, so
/// they don't swim when the camera moves).
pub fn preview_harness(fields: &[Field]) -> String {
    let mut f = String::from("\nfn g3d_field_tint(p: vec3f, ps: vec3f, scale: f32) -> vec4f {\n");
    if !fields.is_empty() {
        f += &format!(
            "    let mask = g3d_v.fmask.x;\n    var cols: array<vec3f, {}>;\n    var n = 0u;\n",
            fields.len()
        );
    }
    for (i, field) in fields.iter().enumerate() {
        let [r, g, b] = field_color(i);
        let col = format!("vec3f({r:?}, {g:?}, {b:?})");
        let func = &field.func;
        if field.comps == 1 {
            f += &format!(
                "    if ((mask & {bit}u) != 0u && {func}(p) < 0.0) {{ cols[n] = {col}; n++; }}\n",
                bit = 1u32 << i
            );
        } else {
            f += &format!(
                "    if ((mask & {bit}u) != 0u) {{\n        let v = {func}(p);\n        \
                 if (dot(v, v) > 0.0) {{\n            \
                 let band = fract(dot(p, normalize(v)) / (scale * 0.03) - g3d_v.misc.x * 1.5);\n            \
                 cols[n] = {col} * mix(0.45, 1.0, smoothstep(0.0, 0.15, band) * (1.0 - smoothstep(0.5, 0.65, band)));\n            \
                 n++;\n        }}\n    }}\n",
                bit = 1u32 << i
            );
        }
    }
    if fields.is_empty() {
        f += "    return vec4f(0.0);\n}\n";
    } else {
        f += "    if (n == 0u) { return vec4f(0.0); }\n    \
              let j = i32(floor(dot(ps, vec3f(1.0)) / (scale * 0.015)));\n    \
              let k = u32(((j % i32(n)) + i32(n)) % i32(n));\n    \
              return vec4f(cols[k], 0.85);\n}\n";
    }
    format!("{PREVIEW_HARNESS}{f}")
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniforms {
    res: [f32; 2],
    relax: f32,
    cut: f32,
    eye: [f32; 4],
    target: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
    flags: [u32; 4],
    fmask: [u32; 4],
    misc: [f32; 4],
}

#[derive(Clone, Copy)]
struct Camera {
    yaw: f32,
    pitch: f32,
    dist: f32,
    target: [f32; 3],
}

impl Camera {
    fn framing(shape: &Shape) -> Camera {
        let (lo, hi) = shape.bounds;
        let target = [0, 1, 2].map(|k| (lo[k] + hi[k]) / 2.0);
        let diag = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt();
        Camera {
            yaw: (-60f32).to_radians(),
            pitch: 28f32.to_radians(),
            dist: 1.6 * diag,
            target,
        }
    }

    fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.dist * cp * self.yaw.cos(),
            self.target[1] + self.dist * cp * self.yaw.sin(),
            self.target[2] + self.dist * self.pitch.sin(),
        ]
    }
}

struct Options {
    clip: bool,
    section: bool,
    cut: f32,
    show_bounds: bool,
    /// bitmask of field overlays to draw
    fields: u32,
    time: f32,
}

fn uniforms(shape: &Shape, cam: &Camera, o: &Options, size: (u32, u32), gamma: bool) -> ViewUniforms {
    let (lo, hi) = shape.bounds;
    let e = cam.eye();
    ViewUniforms {
        res: [size.0 as f32, size.1 as f32],
        relax: shape.step(),
        cut: o.cut,
        eye: [e[0], e[1], e[2], 0.0],
        target: [cam.target[0], cam.target[1], cam.target[2], 0.0],
        bmin: [lo[0], lo[1], lo[2], 0.0],
        bmax: [hi[0], hi[1], hi[2], 0.0],
        flags: [o.clip as u32, o.section as u32, o.show_bounds as u32, gamma as u32],
        fmask: [o.fields, 0, 0, 0],
        misc: [o.time, 0.5 * shape.voxel, 0.0, 0.0],
    }
}

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
}

/// What the viewer shows: a .wgsl shape, or a .vti grid turned into a shape by `volume.rs`.
struct Source {
    shape: Shape,
    fields: Vec<Field>,
    harness: String,
    volume: Option<Volume>,
}

pub fn is_vti(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("vti"))
}

fn load_source(path: &Path, defines: &[(String, f64)], vopts: &VolumeOpts) -> Result<Source> {
    if is_vti(path) {
        if !defines.is_empty() {
            anyhow::bail!("-D parameters don't apply to .vti files");
        }
        return Ok(volume_source(path, Volume::load(path, vopts)?));
    }
    let shape = load_shape(path, defines)?;
    let fields = shape.fields()?;
    Ok(Source {
        harness: preview_harness(&fields),
        shape,
        fields,
        volume: None,
    })
}

fn volume_source(path: &Path, v: Volume) -> Source {
    println!(
        "{}: {} samples inside of {} ({})",
        path.file_name().unwrap_or_default().to_string_lossy(),
        v.inside_count,
        v.count(),
        v.label()
    );
    Source {
        shape: v.shape(path),
        fields: vec![],
        harness: PREVIEW_HARNESS.to_string(),
        volume: Some(v),
    }
}

/// GPU copies of a volume's fields (bindings 1-4 of the preview pipeline).
#[derive(Clone)]
struct VolGpu {
    uni: wgpu::Buffer,
    sd: wgpu::Buffer,
    blk: wgpu::Buffer,
    col: wgpu::Buffer,
}

impl VolGpu {
    fn new(gpu: &Gpu, v: &Volume) -> Result<VolGpu> {
        let bytes = (v.count() * 4) as u64;
        let limits = gpu.device.limits();
        let max = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        if bytes > max {
            anyhow::bail!(
                "grid too large for the GPU ({} samples; this GPU allows {} per array)",
                v.count(),
                max / 4
            );
        }
        let storage = |label, data: &[f32]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(VolGpu {
            uni: gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("volume"),
                contents: bytemuck::bytes_of(&v.uniforms()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            sd: storage("volume sd", &v.sd),
            blk: storage("volume blocky", &v.blk),
            col: storage("volume colour", &v.col),
        })
    }

    /// Upload only the uniforms (mode and colour range): cheap, for edits that keep the fields.
    fn upload_uniforms(&self, gpu: &Gpu, v: &Volume) {
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&v.uniforms()));
    }

    fn upload(&self, gpu: &Gpu, v: &Volume) {
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&v.uniforms()));
        gpu.queue.write_buffer(&self.sd, 0, bytemuck::cast_slice(&v.sd));
        gpu.queue.write_buffer(&self.blk, 0, bytemuck::cast_slice(&v.blk));
        gpu.queue.write_buffer(&self.col, 0, bytemuck::cast_slice(&v.col));
    }
}

/// Validate the source with its preview harness and create the shader module.
fn preview_module(gpu: &Gpu, src: &Source) -> Result<wgpu::ShaderModule> {
    let compiled = src.shape.compile(&src.harness)?;
    gpu.shader(&compiled, "preview")
}

/// Build the preview pipeline with the given parameter values (`override` constants). Cheap
/// compared to `preview_module`, so the viewer redoes only this when a parameter changes.
fn build_pipeline(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    constants: &[(String, f64)],
    format: wgpu::TextureFormat,
    ubuf: &wgpu::Buffer,
    vol: Option<&VolGpu>,
) -> Result<Pipeline> {
    let constants: Vec<(&str, f64)> = constants.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let options = || wgpu::PipelineCompilationOptions {
        constants: &constants,
        ..Default::default()
    };
    let pipeline = gpu.scoped(|| {
        gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview"),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("g3d_vs"),
                compilation_options: options(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("g3d_fs"),
                compilation_options: options(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    })?;
    let mut buffers = vec![ubuf];
    if let Some(v) = vol {
        buffers.extend([&v.uni, &v.sd, &v.blk, &v.col]);
    }
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = gpu.scoped(|| {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    })?;
    Ok(Pipeline { pipeline, bind })
}

/// Raymarch the preview into `view`; without a pipeline (nothing compiled yet) fill it dark red.
fn draw(gpu: &Gpu, p: Option<&Pipeline>, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
    let clear = match p {
        Some(_) => wgpu::Color::BLACK,
        None => wgpu::Color {
            r: 0.3,
            g: 0.05,
            b: 0.05,
            a: 1.0,
        },
    };
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        if let Some(p) = p {
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
    enc.finish()
}

/// Camera and display settings for `render_png`.
pub struct RenderView {
    pub size: (u32, u32),
    pub section: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub show_fields: bool,
}

/// Render the preview offscreen (same shader as the viewer) to a PNG.
pub fn render_png(
    path: &Path,
    defines: &[(String, f64)],
    vopts: &VolumeOpts,
    out: &Path,
    view: &RenderView,
) -> Result<()> {
    let &RenderView {
        size,
        section,
        yaw,
        pitch,
        show_fields,
    } = view;
    let src = load_source(path, defines, vopts)?;
    let (shape, fields) = (&src.shape, &src.fields);
    if show_fields {
        print_legend(fields);
    }
    let gpu = crate::gpu::headless()?;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut cam = Camera::framing(shape);
    cam.yaw = yaw.to_radians();
    cam.pitch = pitch.to_radians();
    let opts = Options {
        clip: true,
        section,
        cut: cam.target[1],
        show_bounds: true,
        fields: if show_fields { all_fields(fields) } else { 0 },
        time: 0.0,
    };
    let (w, h) = size;
    let ubuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("view"),
        contents: bytemuck::bytes_of(&uniforms(shape, &cam, &opts, size, true)),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let vol = src.volume.as_ref().map(|v| VolGpu::new(&gpu, v)).transpose()?;
    let module = preview_module(&gpu, &src)?;
    let pipeline = build_pipeline(&gpu, &module, &src.shape.defines, format, &ubuf, vol.as_ref())?;
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixels"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cmd = draw(&gpu, Some(&pipeline), &view);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([cmd, enc.finish()]);
    let data = gpu.read_buffer(&buf, (row * h) as u64)?;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        for px in data[y * row as usize..][..w as usize * 4].chunks(4) {
            rgb.extend_from_slice(&px[..3]);
        }
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::fs::File::create(out).with_context(|| format!("cannot write {}", out.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    Ok(())
}

fn all_fields(fields: &[Field]) -> u32 {
    (0..fields.len().min(32)).fold(0, |m, i| m | 1 << i)
}

/// Print each field's overlay colour as a terminal swatch.
fn print_legend(fields: &[Field]) {
    if fields.is_empty() {
        return;
    }
    let items: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let [r, g, b] = field_color(i).map(|c| (c.powf(1.0 / 2.2) * 255.0) as u8);
            let kind = if f.comps == 1 {
                "< 0"
            } else {
                "≠ 0, bands move along it"
            };
            format!("\x1b[38;2;{r};{g};{b}m■\x1b[0m {} ({kind})", f.name)
        })
        .collect();
    println!("fields: {}", items.join("   "));
}

// ------------------------------------------------------------------------------------------ viewer

const KEYS: &[(&str, &str)] = &[
    ("left-drag", "orbit"),
    ("right/middle-drag", "pan"),
    ("wheel", "zoom"),
    ("double-click", "reset camera"),
    ("1 2 3 4", "front / side / top / iso"),
    ("R", "reset camera"),
    ("C", "section view"),
    ("[  ]", "move section plane"),
    ("B", "clip to bounds"),
    ("V", "bounds box"),
    ("F", "cycle field overlays (.vti: colour by array)"),
    ("-  =", ".vti: lower / raise threshold"),
    ("M", ".vti: smooth / blocky"),
    ("I", ".vti: flip inside"),
    ("A", ".vti: next array"),
    ("F5", "force reload"),
    ("E  G  O", "export / slice / open in slicer"),
    ("Q  Esc", "quit"),
];

/// Format of the offscreen texture the preview is raymarched into. egui samples textures as
/// gamma-encoded, so the shader applies gamma itself (as for `render_png`).
const VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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

struct Window_ {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Arc<Gpu>,
    ubuf: wgpu::Buffer,
    /// The current shape's preview shader; kept so parameter changes only rebuild the pipeline.
    module: Option<wgpu::ShaderModule>,
    pipeline: Option<Pipeline>,
    vol: Option<VolGpu>,
    egui: egui::Context,
    egui_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    viewport: Viewport,
}

/// A background `gen3d export|slice|open` run started from the viewer.
#[derive(Default)]
struct Job {
    cmd: String,
    running: bool,
    ok: Option<bool>,
    log: String,
    started: Option<Instant>,
    took: Duration,
    /// process id while running, for Stop
    pid: Option<u32>,
    stopped: bool,
}

/// Settings of the Topology optimization section (`gen3d optimize` flags).
struct OptUi {
    volfrac: f64,
    iters: usize,
    /// None = `topopt::auto_voxel`
    voxel: Option<f32>,
    force: f64,
    rmin: f64,
}

impl Default for OptUi {
    fn default() -> Self {
        OptUi {
            volfrac: 0.3,
            iters: 50,
            voxel: None,
            force: 100.0,
            rmin: 1.5,
        }
    }
}

struct Viewer {
    path: PathBuf,
    /// Parameter values that differ from the shape's defaults (as `-D NAME=VALUE`).
    defines: Vec<(String, f64)>,
    /// .vti display options from the command line
    vopts: VolumeOpts,
    volume: Option<Volume>,
    /// Threshold being dragged in the Volume panel; applied on release (recomputing is CPU work).
    iso_drag: Option<f32>,
    /// Grid voxel for "Export VTI" (None = `default_vti_voxel`).
    vti_voxel: Option<f32>,
    opt: OptUi,
    shape: Shape,
    fields: Vec<Field>,
    params: Vec<Param>,
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
    cam: Camera,
    opts: Options,
    /// Fraction of the viewport's pixels that are raymarched (lower = faster).
    render_scale: f32,
    /// Viewport size wanted by this frame's layout, in physical pixels.
    view_px: (u32, u32),
    job: Arc<Mutex<Job>>,
    gpu_name: String,
    quit: bool,
    win: Option<Window_>,
    last_poll: Instant,
    frames: u32,
    fps: f32,
    fps_t: Instant,
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

fn stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

/// Drop ANSI escape sequences (colours) from a line of job output.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            if it.peek() == Some(&'[') {
                it.next();
                for c in it.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn fmt_num(v: f64) -> String {
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

const OK_GREEN: egui::Color32 = egui::Color32::from_rgb(90, 190, 100);
/// The preview's surface orange, for "inside" in the Volume panel's histogram.
const SURFACE_ORANGE: egui::Color32 = egui::Color32::from_rgb(242, 165, 80);

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
/// side are drawn in the surface colour. Returns the response and a function from x to value.
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

/// Default grid voxel for `export --format vti`: the shape's voxel, coarsened so the grid has at
/// most about a million samples (simulation grids are usually much coarser than print meshes).
fn default_vti_voxel(shape: &Shape) -> f32 {
    let vol: f32 = shape.size().iter().map(|s| s.max(1e-3)).product();
    let v = shape.voxel.max((vol / 1e6).cbrt());
    // round up to 2 significant digits
    let mag = 10f32.powf(v.log10().floor() - 1.0);
    (v / mag).ceil() * mag
}

/// Points of the grid `export --format vti` writes for `shape` at `voxel`.
fn vti_dims(shape: &Shape, voxel: f32) -> [usize; 3] {
    shape.size().map(|s| (s / voxel).ceil().max(0.0) as usize + 1)
}

/// The last `wrote PATH (...)` line of a job's log.
fn written_path(log: &str) -> Option<PathBuf> {
    log.lines().rev().find_map(|l| {
        let rest = l.trim().strip_prefix("wrote ")?;
        let end = rest.rfind(" (").unwrap_or(rest.len());
        Some(PathBuf::from(&rest[..end]))
    })
}

fn status_dot(ui: &mut egui::Ui, color: egui::Color32) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(r.center(), 4.0, color);
}

fn define_flags(defines: &[(String, f64)]) -> String {
    defines
        .iter()
        .map(|(k, v)| format!("-D {k}={}", fmt_num(*v)))
        .collect::<Vec<_>>()
        .join(" ")
}

type Rebuild = std::sync::mpsc::Receiver<(u64, Vec<(String, f64)>, Result<Pipeline>)>;

fn rgb_gamma(c: [f32; 3]) -> egui::Color32 {
    let [r, g, b] = c.map(|c| (c.powf(1.0 / 2.2) * 255.0) as u8);
    egui::Color32::from_rgb(r, g, b)
}

impl Viewer {
    /// Options that reproduce the current volume view, so a reloaded .vti keeps them.
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

    fn is_volume(&self) -> bool {
        self.volume.is_some() || is_vti(&self.path)
    }

    fn reload(&mut self, first: bool) {
        // a .vti being rewritten (e.g. by an optimizer) may be caught half-written: keep the old
        // view and retry on the next change
        let src = load_source(&self.path, &self.defines, &self.current_vopts()).or_else(|e| {
            if self.volume.is_some() {
                load_source(&self.path, &[], &self.vopts)
            } else {
                Err(e)
            }
        });
        let src = match src {
            Ok(s) => s,
            Err(e) => {
                self.mtimes = HashMap::from([(self.path.clone(), mtime(&self.path))]);
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
        let params = match src.shape.params() {
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
                "ignored -D {}: not a parameter of this shape (parameters: {})",
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
        let Some(win) = &mut self.win else { return };
        let vol = match src.volume.as_ref().map(|v| VolGpu::new(&win.gpu, v)).transpose() {
            Ok(v) => v,
            Err(e) => return self.fail(&format!("{e:#}")),
        };
        let built = preview_module(&win.gpu, &src).and_then(|m| {
            let p = build_pipeline(&win.gpu, &m, &src.shape.defines, VIEW_FORMAT, &win.ubuf, vol.as_ref())?;
            Ok((m, p))
        });
        match built {
            Ok((m, p)) => {
                win.module = Some(m);
                win.pipeline = Some(p);
                win.vol = vol;
                self.rebuild_gen += 1;
                self.param_error = None;
            }
            Err(e) => return self.fail(&e.to_string()),
        }
        let Source {
            shape, fields, volume, ..
        } = src;
        self.volume = volume;
        let bounds_changed = !self.loaded || self.shape.bounds != shape.bounds;
        self.shape = shape;
        let names = |f: &[Field]| f.iter().map(|f| f.name.clone()).collect::<Vec<_>>();
        if names(&fields) != names(&self.fields) {
            self.field_mode = 0;
        }
        self.fields = fields;
        self.params = params;
        self.set_field_mask();
        self.loaded = true;
        self.error = None;
        self.loaded_at = stamp();
        if first || bounds_changed {
            self.reset_camera();
        }
        println!(
            "[{}] loaded {}",
            self.loaded_at,
            self.path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    fn fail(&mut self, msg: &str) {
        let what = if self.is_volume() { "error" } else { "shader error" };
        println!("\n[{}] \x1b[31m{what}\x1b[0m\n{}\n", stamp(), msg.trim());
        self.error = Some(msg.trim().to_string());
    }

    /// Apply a change to the volume view: `rebuild` when the array (and so the grid size) changed.
    fn edit_volume(&mut self, rebuild: bool, f: impl FnOnce(&mut Volume)) {
        let Some(mut v) = self.volume.take() else { return };
        f(&mut v);
        if rebuild {
            let src = volume_source(&self.path, v);
            return self.install(src, false);
        }
        if let Some(win) = &self.win
            && let Some(g) = &win.vol
        {
            g.upload(&win.gpu, &v);
        }
        self.volume = Some(v);
    }

    /// Apply a change that only touches the volume's uniforms (smooth/blocky, colour range).
    fn edit_volume_uniforms(&mut self, f: impl FnOnce(&mut Volume)) {
        let Some(v) = &mut self.volume else { return };
        f(v);
        if let Some(win) = &self.win
            && let Some(g) = &win.vol
        {
            g.upload_uniforms(&win.gpu, v);
        }
    }

    /// Rebuild the pipeline with the current parameter values. The driver's shader compile takes
    /// ~100 ms, so it runs on a worker thread; changes made meanwhile are coalesced into one more
    /// build, so dragging a slider keeps the UI smooth and the preview follows as fast as it can.
    fn request_rebuild(&mut self) {
        self.shape.defines = self.defines.clone();
        if self.rebuild.is_some() {
            self.rebuild_again = true;
            return;
        }
        let Some(win) = &self.win else { return };
        let Some(module) = win.module.clone() else { return };
        let (gpu, ubuf, vol, defines, generation) = (
            win.gpu.clone(),
            win.ubuf.clone(),
            win.vol.clone(),
            self.defines.clone(),
            self.rebuild_gen,
        );
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = build_pipeline(&gpu, &module, &defines, VIEW_FORMAT, &ubuf, vol.as_ref());
            let _ = tx.send((generation, defines, result));
        });
        self.rebuild = Some(rx);
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
                    }
                    self.param_error = None;
                }
                Err(e) => {
                    let msg = format!("with {}: {e}", define_flags(&defines));
                    println!("\n[{}] \x1b[31mpipeline error\x1b[0m\n{msg}\n", stamp());
                    self.param_error = Some(msg);
                }
            }
        }
        if std::mem::take(&mut self.rebuild_again) {
            self.request_rebuild();
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

    fn run_bg(&self, cmd: &str) {
        self.run_bg_with(cmd, &[]);
    }

    /// Run `gen3d <cmd> <file> [extra...]` in the background, with the current parameters (or
    /// volume settings), capturing its output for the Output panel.
    fn run_bg_with(&self, cmd: &str, extra: &[String]) {
        {
            let mut job = self.job.lock().unwrap();
            if job.running {
                return;
            }
            *job = Job {
                cmd: [cmd.to_string()]
                    .into_iter()
                    .chain(extra.iter().cloned())
                    .collect::<Vec<_>>()
                    .join(" "),
                running: true,
                started: Some(Instant::now()),
                ..Default::default()
            };
        }
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("gen3d"));
        let mut c = std::process::Command::new(exe);
        c.arg(cmd).arg(&self.path).args(extra);
        if let Some(v) = &self.volume {
            c.args(["--array", v.array_name(), "--iso", &v.iso.to_string()]);
            c.arg(if v.below { "--below" } else { "--above" });
        }
        for (k, v) in &self.defines {
            c.arg("-D").arg(format!("{k}={v}"));
        }
        c.stdout(Stdio::piped()).stderr(Stdio::piped());
        let job = self.job.clone();
        std::thread::spawn(move || {
            let push = |job: &Mutex<Job>, line: &str| {
                let line = strip_ansi(line.rsplit('\r').next().unwrap_or(line));
                let mut j = job.lock().unwrap();
                j.log.push_str(&line);
                j.log.push('\n');
            };
            let result = (|| -> std::io::Result<std::process::ExitStatus> {
                let mut child = c.spawn()?;
                job.lock().unwrap().pid = Some(child.id());
                let (out, err) = (child.stdout.take(), child.stderr.take());
                let j = job.clone();
                let reader = std::thread::spawn(move || {
                    for l in out
                        .into_iter()
                        .flat_map(|o| BufReader::new(o).lines().map_while(|l| l.ok()))
                    {
                        println!("{l}");
                        push(&j, &l);
                    }
                });
                for l in err
                    .into_iter()
                    .flat_map(|e| BufReader::new(e).lines().map_while(|l| l.ok()))
                {
                    eprintln!("{l}");
                    push(&job, &l);
                }
                let _ = reader.join();
                child.wait()
            })();
            let mut j = job.lock().unwrap();
            j.running = false;
            j.pid = None;
            j.took = j.started.map_or_else(Duration::default, |s| s.elapsed());
            j.ok = Some(match result {
                Ok(s) => s.success(),
                Err(e) => {
                    let msg = format!("could not run gen3d {}: {e}\n", j.cmd);
                    j.log.push_str(&msg);
                    false
                }
            });
        });
    }

    fn on_key(&mut self, key: KeyCode) {
        let step = self.shape.size()[1] / 50.0;
        match key {
            KeyCode::Escape | KeyCode::KeyQ => self.quit = true,
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
            KeyCode::KeyE => self.run_bg("export"),
            KeyCode::KeyG => self.run_bg("slice"),
            KeyCode::KeyO => self.run_bg("open"),
            KeyCode::Digit1 => self.set_view(-90.0, 0.0),
            KeyCode::Digit2 => self.set_view(0.0, 0.0),
            KeyCode::Digit3 => self.set_view(-90.0, 89.9),
            KeyCode::Digit4 => self.set_view(-60.0, 28.0),
            _ => {}
        }
    }

    // -------------------------------------------------------------------------------------- ui

    fn ui(&mut self, ui: &mut egui::Ui, tex: egui::TextureId) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(320.0)
            .size_range(240.0..=600.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        if let Some(err) = self.error.as_ref().or(self.param_error.as_ref()) {
            egui::Panel::bottom("error")
                .resizable(true)
                .default_size(170.0)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let title = if self.is_volume() { "Error" } else { "Shader error" };
                        ui.colored_label(ui.visuals().error_fg_color, egui::RichText::new(title).strong());
                        ui.weak(if self.loaded {
                            "— showing the last version that compiled; fix the file and save"
                        } else {
                            "— fix the file and save"
                        });
                    });
                    egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(err).monospace()).selectable(true));
                    });
                });
        }
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport_ui(ui, tex));
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui, tex: egui::TextureId) {
        let rect = ui.available_rect_before_wrap();
        let ppp = ui.ctx().pixels_per_point();
        let px = |v: f32| ((v * ppp * self.render_scale).round() as u32).max(1);
        self.view_px = (px(rect.width()), px(rect.height()));
        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter().image(tex, rect, uv, egui::Color32::WHITE);

        // camera: same feel as before (rates per physical pixel)
        let d = resp.drag_delta();
        if resp.dragged_by(egui::PointerButton::Primary) {
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

        // overlay: frame rate and active modes
        let mut info = vec![format!("{:.0} fps", self.fps)];
        if self.rebuild.is_some() {
            info.insert(0, "compiling…".into());
        }
        if self.render_scale < 1.0 {
            info.push(format!("{:.0}% res", self.render_scale * 100.0));
        }
        if self.opts.section {
            info.push(format!("section y = {:.1}", self.opts.cut));
        }
        if !self.opts.clip {
            info.push("unclipped".into());
        }
        let painter = ui.painter_at(rect);
        painter.text(
            rect.right_top() + egui::vec2(-10.0, 8.0),
            egui::Align2::RIGHT_TOP,
            info.join("  ·  "),
            egui::FontId::proportional(13.0),
            egui::Color32::from_white_alpha(170),
        );
        if !self.loaded {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "no shape loaded yet",
                egui::FontId::proportional(18.0),
                egui::Color32::from_white_alpha(200),
            );
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let name = self.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        ui.heading(&name).on_hover_text(self.path.display().to_string());
        ui.horizontal(|ui| {
            if self.error.is_some() {
                status_dot(ui, ui.visuals().error_fg_color);
                ui.colored_label(ui.visuals().error_fg_color, "error");
            } else if self.loaded {
                status_dot(ui, OK_GREEN);
                ui.label(format!("loaded {}", self.loaded_at))
                    .on_hover_text("reloads automatically when the file (or an include) is saved");
            }
            if ui.small_button("Reload").on_hover_text("F5").clicked() {
                self.reload(false);
            }
        });
        ui.weak(&self.gpu_name);
        if let Some(n) = self.notice.clone() {
            ui.colored_label(ui.visuals().warn_fg_color, n);
            if ui.small_button("Dismiss").clicked() {
                self.notice = None;
            }
        }
        ui.separator();

        if is_vti(&self.path) {
            egui::CollapsingHeader::new("Volume")
                .default_open(true)
                .show(ui, |ui| self.volume_ui(ui));
        } else {
            egui::CollapsingHeader::new("Parameters")
                .default_open(true)
                .show(ui, |ui| self.params_ui(ui));
        }
        egui::CollapsingHeader::new("View")
            .default_open(true)
            .show(ui, |ui| self.view_ui(ui));
        if !self.fields.is_empty() {
            egui::CollapsingHeader::new("Field overlays")
                .default_open(true)
                .show(ui, |ui| self.fields_ui(ui));
        }
        if self.volume.is_none() && self.loaded {
            egui::CollapsingHeader::new("Topology optimization")
                .default_open(self.optimizable())
                .show(ui, |ui| self.optimize_ui(ui));
        }
        egui::CollapsingHeader::new("Output")
            .default_open(true)
            .show(ui, |ui| self.output_ui(ui));
        if self.volume.is_some() {
            egui::CollapsingHeader::new("Grid").show(ui, |ui| self.grid_ui(ui));
        } else {
            egui::CollapsingHeader::new("Shape").show(ui, |ui| self.shape_ui(ui));
        }
        egui::CollapsingHeader::new("Keyboard & mouse").show(ui, |ui| {
            egui::Grid::new("keys").num_columns(2).striped(true).show(ui, |ui| {
                for (k, v) in KEYS {
                    ui.monospace(*k);
                    ui.label(*v);
                    ui.end_row();
                }
            });
        });
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
        egui::Grid::new("params").num_columns(3).show(ui, |ui| {
            for p in &params {
                let cur = self.param_value(p);
                let mut tip = p.note.clone();
                if !tip.is_empty() {
                    tip.push('\n');
                }
                tip += &format!("default {}", fmt_num(p.default));
                ui.label(&p.name).on_hover_text(tip);
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
        ui.horizontal(|ui| {
            let any = !self.defines.is_empty();
            if ui.add_enabled(any, egui::Button::new("Reset all")).clicked() {
                self.defines.clear();
                self.params_dirty = true;
            }
            if ui
                .add_enabled(any, egui::Button::new("Copy -D flags"))
                .on_hover_text("copy the changed values as command-line flags")
                .clicked()
            {
                ui.ctx().copy_text(define_flags(&self.defines));
            }
        });
    }

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

        egui::ComboBox::from_label("array")
            .selected_text(&names[array])
            .show_ui(ui, |ui| {
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(&mut array, i, n);
                }
            })
            .response
            .on_hover_text("A cycles");

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
            ui.label("inside is");
            ui.radio_value(&mut below, true, "below");
            ui.radio_value(&mut below, false, "above");
        })
        .response
        .on_hover_text("I");
        ui.horizontal(|ui| {
            ui.radio_value(&mut blocky, false, "smooth");
            ui.radio_value(&mut blocky, true, "blocky");
        })
        .response
        .on_hover_text("M · blocky draws every inside sample as a cube (for cell data: the elements)");

        // how much material that is
        let pct = 100.0 * inside as f64 / count.max(1) as f64;
        egui::Grid::new("vol_stats").num_columns(2).show(ui, |ui| {
            ui.label("inside");
            ui.label(format!("{inside} of {count} samples ({pct:.1}%)"))
                .on_hover_text("the volume fraction, for a density from topology optimization");
            ui.end_row();
            ui.label("volume");
            ui.label(format!(
                "≈ {} cm³  ·  {} g PLA",
                fmt_num((inside_mm3 / 1000.0 * 100.0).round() / 100.0),
                fmt_num((inside_mm3 / 1000.0 * 1.24 * 10.0).round() / 10.0)
            ))
            .on_hover_text(
                "inside samples × cell volume; solid, so an upper bound for the print \
                 (PLA at 1.24 g/cm³, 100% infill)",
            );
            ui.end_row();
        });

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

    /// Whether the shape defines the fields `gen3d optimize` needs.
    fn optimizable(&self) -> bool {
        let has = |n: &str, c: usize| self.fields.iter().any(|f| f.name == n && f.comps == c);
        has("fixed", 1) && has("load", 3)
    }

    fn optimize_ui(&mut self, ui: &mut egui::Ui) {
        if !self.optimizable() {
            ui.weak(
                "The shape is the design domain. Add `fn field_fixed(p: vec3f) -> f32` (< 0 where it is \
                 clamped) and `fn field_load(p: vec3f) -> vec3f` (load direction) to optimize it; \
                 optional field_keep / field_void force material / empty space. See \
                 shapes/bracket_domain.wgsl.",
            );
            return;
        }
        let running = self.job.lock().unwrap().running;
        let o = &mut self.opt;
        let auto = crate::topopt::auto_voxel(&self.shape, 40_000.0);
        egui::Grid::new("opt").num_columns(2).show(ui, |ui| {
            ui.label("volume fraction");
            ui.add(egui::Slider::new(&mut o.volfrac, 0.05..=0.8).fixed_decimals(2))
                .on_hover_text("share of the design domain to fill with material");
            ui.end_row();
            ui.label("load");
            ui.add(
                egui::DragValue::new(&mut o.force)
                    .speed(1.0)
                    .range(0.0..=1e6)
                    .suffix(" N"),
            )
            .on_hover_text("total force over the field_load region; only scales the displacements");
            ui.end_row();
            ui.label("voxel");
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
                    .on_hover_text("element size; time grows quickly as it shrinks")
                    .changed()
                {
                    o.voxel = Some(v);
                }
                if o.voxel.is_some() && ui.small_button("auto").clicked() {
                    o.voxel = None;
                }
            });
            ui.end_row();
            ui.label("filter radius");
            ui.add(
                egui::Slider::new(&mut o.rmin, 1.0..=4.0)
                    .fixed_decimals(1)
                    .suffix(" voxels"),
            )
            .on_hover_text("minimum member size: larger gives thicker, simpler members");
            ui.end_row();
            ui.label("iterations");
            ui.add(egui::Slider::new(&mut o.iters, 5..=200))
                .on_hover_text("at most; stops early when converged");
            ui.end_row();
        });
        let vox = o.voxel.unwrap_or(auto);
        let n: usize = self
            .shape
            .size()
            .iter()
            .map(|s| (*s / vox).ceil().max(1.0) as usize)
            .product();
        ui.weak(format!("≤ {n} elements (only those inside the domain are solved)"));
        let args: Vec<String> = [
            ("--volfrac", o.volfrac),
            ("--iters", o.iters as f64),
            ("--voxel", vox as f64),
            ("--force", o.force),
            ("--rmin", o.rmin),
        ]
        .iter()
        .flat_map(|(k, v)| [k.to_string(), fmt_num(*v)])
        .collect();
        if ui
            .add_enabled(!running, egui::Button::new("Optimize"))
            .on_hover_text("run gen3d optimize in the background; View shows the design live")
            .clicked()
        {
            self.run_bg_with("optimize", &args);
        }
    }

    /// The Grid section for .vti files: lattice and every array in the file.
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

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for (label, key, yaw, pitch) in [
                ("Front", "1", -90.0, 0.0),
                ("Side", "2", 0.0, 0.0),
                ("Top", "3", -90.0, 89.9),
                ("Iso", "4", -60.0, 28.0),
            ] {
                if ui.button(label).on_hover_text(key).clicked() {
                    self.set_view(yaw, pitch);
                }
            }
            if ui
                .button("Reset")
                .on_hover_text("R, or double-click the view")
                .clicked()
            {
                self.reset_camera();
            }
        });
        ui.checkbox(&mut self.opts.section, "Section view").on_hover_text("C");
        if self.opts.section {
            let (lo, hi) = self.shape.bounds;
            ui.add(
                egui::Slider::new(&mut self.opts.cut, lo[1]..=hi[1])
                    .text("cut y (mm)")
                    .clamping(egui::SliderClamping::Never),
            )
            .on_hover_text("[ and ] step it");
        }
        ui.checkbox(&mut self.opts.clip, "Clip to bounds").on_hover_text("B");
        ui.checkbox(&mut self.opts.show_bounds, "Show bounds box")
            .on_hover_text("V");
        ui.add(
            egui::Slider::new(&mut self.render_scale, 0.25..=1.0)
                .text("resolution")
                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
        )
        .on_hover_text("raymarch fewer pixels for heavy shapes");
    }

    fn fields_ui(&mut self, ui: &mut egui::Ui) {
        let before = self.field_mode;
        let n = self.fields.len();
        ui.radio_value(&mut self.field_mode, 0, "All").on_hover_text("F cycles");
        for (i, f) in self.fields.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.field_mode, i + 1, &f.name)
                    .on_hover_text(if f.comps == 1 {
                        "scalar field: tints the surface where it is < 0"
                    } else {
                        "vector field: tints where it is non-zero; bands move along it"
                    });
                let (r, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 3.0, rgb_gamma(field_color(i)));
            });
        }
        ui.radio_value(&mut self.field_mode, n + 1, "Off");
        if self.field_mode != before {
            self.set_field_mask();
        }
    }

    fn output_ui(&mut self, ui: &mut egui::Ui) {
        let (running, ok, cmd, log, took, started) = {
            let j = self.job.lock().unwrap();
            (j.running, j.ok, j.cmd.clone(), j.log.clone(), j.took, j.started)
        };
        ui.horizontal_wrapped(|ui| {
            let enabled = !running && self.loaded;
            for (label, cmd, key, tip) in [
                ("Export mesh", "export", "E", "mesh to STL/3MF in out/"),
                ("Slice", "slice", "G", "export and slice to G-code"),
                ("Open in slicer", "open", "O", "export and open in the slicer GUI"),
            ] {
                if ui
                    .add_enabled(enabled, egui::Button::new(label))
                    .on_hover_text(format!("{tip} ({key})"))
                    .clicked()
                {
                    self.run_bg(cmd);
                }
            }
        });
        if let Some(v) = &self.volume {
            let op = if v.below { "<" } else { ">" };
            ui.weak(format!("meshes {} {op} {}", v.array_name(), fmt_num(v.iso as f64)));
        } else if self.loaded {
            // the simulation grid: SDF plus field_* arrays
            ui.horizontal(|ui| {
                let def = default_vti_voxel(&self.shape);
                let mut vox = self.vti_voxel.unwrap_or(def);
                let enabled = !running;
                if ui
                    .add_enabled(enabled, egui::Button::new("Export VTI"))
                    .on_hover_text(
                        "write the SDF and every field_* function as a VTK grid (out/*.vti), \
                         for simulation or topology optimization",
                    )
                    .clicked()
                {
                    let args = ["--format", "vti", "--voxel"].map(String::from);
                    let mut args = args.to_vec();
                    args.push(fmt_num(vox as f64));
                    self.run_bg_with("export", &args);
                }
                ui.label("voxel");
                let speed = vox * 0.01;
                let r = ui.add(
                    egui::DragValue::new(&mut vox)
                        .speed(speed)
                        .range(1e-3..=1e3)
                        .suffix(" mm"),
                );
                if r.changed() {
                    self.vti_voxel = Some(vox);
                }
                if self.vti_voxel.is_some() && ui.small_button("auto").clicked() {
                    self.vti_voxel = None;
                }
            });
            let vox = self.vti_voxel.unwrap_or_else(|| default_vti_voxel(&self.shape));
            let d = vti_dims(&self.shape, vox);
            let n: usize = d.iter().product();
            let floats = 1 + self.fields.iter().map(|f| f.comps).sum::<usize>();
            let mut arrays = vec!["sdf".to_string()];
            arrays.extend(self.fields.iter().map(|f| f.name.clone()));
            ui.weak(format!(
                "{} × {} × {} = {} samples, {:.1} MB · {}",
                d[0],
                d[1],
                d[2],
                n,
                (n * floats * 4) as f64 / 1e6,
                arrays.join(", ")
            ));
        }
        if !self.defines.is_empty() {
            ui.weak(format!("uses {}", define_flags(&self.defines)));
        }
        if cmd.is_empty() {
            return;
        }
        let (pid, stopped) = {
            let j = self.job.lock().unwrap();
            (j.pid, j.stopped)
        };
        ui.horizontal_wrapped(|ui| {
            if running {
                ui.spinner();
                let t = started.map_or(0.0, |s| s.elapsed().as_secs_f32());
                ui.label(format!("{} running… {t:.1} s", cmd.split(' ').next().unwrap_or("")));
                if let Some(pid) = pid
                    && ui
                        .small_button("Stop")
                        .on_hover_text("stop the job (keeps what it wrote)")
                        .clicked()
                {
                    self.job.lock().unwrap().stopped = true;
                    let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                }
            } else if stopped {
                status_dot(ui, ui.visuals().warn_fg_color);
                ui.label(format!(
                    "{} stopped after {:.1} s",
                    cmd.split(' ').next().unwrap_or(""),
                    took.as_secs_f32()
                ));
            } else if ok == Some(true) {
                status_dot(ui, OK_GREEN);
                ui.label(format!(
                    "{} finished in {:.1} s",
                    cmd.split(' ').next().unwrap_or(""),
                    took.as_secs_f32()
                ));
            } else {
                status_dot(ui, ui.visuals().error_fg_color);
                ui.colored_label(ui.visuals().error_fg_color, format!("{cmd} failed"));
            }
            // a .vti the job wrote (an optimization rewrites it every iteration: watch it live)
            if (running || stopped || ok == Some(true))
                && let Some(out) = written_path(&log).filter(|p| is_vti(p))
                && ui
                    .small_button("View")
                    .on_hover_text(format!("open {} in a new viewer window", out.display()))
                    .clicked()
            {
                let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("gen3d"));
                if let Err(e) = std::process::Command::new(exe).arg("view").arg(&out).spawn() {
                    self.notice = Some(format!("could not open a viewer: {e}"));
                }
            }
        });
        ui.weak(&cmd);
        if !log.is_empty() {
            egui::ScrollArea::vertical()
                .max_height(160.0)
                .stick_to_bottom(true)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(log.trim_end()).monospace().small()).selectable(true));
                });
        }
    }

    fn shape_ui(&mut self, ui: &mut egui::Ui) {
        let (lo, hi) = self.shape.bounds;
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
            ui.monospace(format!("{}  →  {}", v3(lo), v3(hi)));
            ui.end_row();
            ui.label("size");
            ui.monospace(format!("{} mm", v3(self.shape.size())));
            ui.end_row();
            ui.label("voxel");
            ui.monospace(format!("{} mm", fmt_num(self.shape.voxel as f64)));
            ui.end_row();
            ui.label("step");
            ui.monospace(fmt_num(self.shape.step() as f64));
            ui.end_row();
            if let Some(d) = self.shape.decimate() {
                ui.label("decimate");
                ui.monospace(fmt_num(d as f64));
                ui.end_row();
            }
            ui.label("files");
            let files: Vec<_> = self.shape.files.iter().map(|f| f.display().to_string()).collect();
            ui.label(files.len().to_string()).on_hover_text(files.join("\n"));
            ui.end_row();
        });
    }

    // ---------------------------------------------------------------------------------- frame

    fn frame(&mut self) {
        let Some(win) = &mut self.win else { return };
        let ctx = win.egui.clone();
        let raw = win.egui_state.take_egui_input(&win.window);
        let tex = win.viewport.id;
        let mut out = ctx.run_ui(raw, |ui| self.ui(ui, tex));
        if std::mem::take(&mut self.params_dirty) {
            self.request_rebuild();
        }
        self.poll_rebuild();
        self.opts.time = self.start.elapsed().as_secs_f32();
        let Some(win) = &mut self.win else {
            out.textures_delta.clear();
            return;
        };
        win.egui_state
            .handle_platform_output(&win.window, std::mem::take(&mut out.platform_output));
        // egui's texture uploads must be applied (and frees honoured) even if this frame isn't drawn
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in &deltas {
                win.renderer.update_texture(&win.gpu.device, &win.gpu.queue, id, delta);
            }
        }
        let free: Vec<_> = out.textures_delta.free.drain().collect();
        let free_all = |r: &mut egui_wgpu::Renderer| free.iter().for_each(|id| r.free_texture(id));

        if win.viewport.size != self.view_px {
            win.viewport.size = self.view_px;
            win.viewport.view = Viewport::target(&win.gpu, self.view_px);
            win.renderer.update_egui_texture_from_wgpu_texture(
                &win.gpu.device,
                &win.viewport.view,
                wgpu::FilterMode::Linear,
                win.viewport.id,
            );
        }
        let frame = match win.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                win.surface.configure(&win.gpu.device, &win.config);
                return free_all(&mut win.renderer);
            }
            _ => return free_all(&mut win.renderer),
        };
        let u = uniforms(&self.shape, &self.cam, &self.opts, win.viewport.size, true);
        win.gpu.queue.write_buffer(&win.ubuf, 0, bytemuck::bytes_of(&u));
        let preview = draw(&win.gpu, win.pipeline.as_ref(), &win.viewport.view);

        let (device, queue) = (&win.gpu.device, &win.gpu.queue);
        let jobs = ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [win.config.width, win.config.height],
            pixels_per_point: out.pixels_per_point,
        };
        let mut enc = device.create_command_encoder(&Default::default());
        let extra = win.renderer.update_buffers(device, queue, &mut enc, &jobs, &screen);
        let view = frame.texture.create_view(&Default::default());
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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
        queue.submit(std::iter::once(preview).chain(extra).chain([enc.finish()]));
        win.window.pre_present_notify();
        queue.present(frame);
        free_all(&mut win.renderer);

        self.frames += 1;
        let dt = self.fps_t.elapsed().as_secs_f32();
        if dt >= 0.5 {
            self.fps = self.frames as f32 / dt;
            self.frames = 0;
            self.fps_t = Instant::now();
        }
    }

    fn open_window(&mut self, el: &ActiveEventLoop) -> Result<Window_> {
        let name = self.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let attrs = Window::default_attributes()
            .with_title(format!("gen3d — {name}"))
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 880.0));
        let window = Arc::new(el.create_window(attrs)?);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone())?;
        let gpu = Arc::new(crate::gpu::create(instance, Some(&surface), true)?);
        let info = gpu.adapter.get_info();
        self.gpu_name = format!("{} ({:?})", info.name, info.backend);
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
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let egui = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let mut renderer = egui_wgpu::Renderer::new(&gpu.device, format, Default::default());
        let view = Viewport::target(&gpu, (1, 1));
        let id = renderer.register_native_texture(&gpu.device, &view, wgpu::FilterMode::Linear);
        Ok(Window_ {
            window,
            surface,
            config,
            gpu,
            ubuf,
            module: None,
            pipeline: None,
            vol: None,
            egui,
            egui_state,
            renderer,
            viewport: Viewport { view, size: (1, 1), id },
        })
    }
}

impl ApplicationHandler for Viewer {
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
                if let Some(w) = &self.win {
                    w.window.request_redraw();
                }
            }
            Err(e) => {
                eprintln!("\x1b[31merror:\x1b[0m could not open the viewer: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = &mut self.win else { return };
        let _ = win.egui_state.on_window_event(&win.window, &event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                win.config.width = size.width.max(1);
                win.config.height = size.height.max(1);
                win.surface.configure(&win.gpu.device, &win.config);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !win.egui.egui_wants_keyboard_input() =>
            {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.on_key(code);
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                if let Some(w) = &self.win {
                    w.window.request_redraw();
                }
            }
            _ => {}
        }
        if self.quit {
            el.exit();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        if now - self.last_poll > Duration::from_millis(250) {
            self.last_poll = now;
            if self.mtimes.iter().any(|(f, m)| mtime(f) != *m) {
                std::thread::sleep(Duration::from_millis(50)); // let editors finish writing
                self.reload(false);
            }
        }
        el.set_control_flow(ControlFlow::WaitUntil(now + Duration::from_millis(250)));
    }
}

pub fn view(path: &Path, defines: Vec<(String, f64)>, vopts: VolumeOpts) -> Result<()> {
    let path = path
        .canonicalize()
        .with_context(|| format!("cannot open {}", path.display()))?;
    let event_loop = EventLoop::new()?;
    let mut app = Viewer {
        shape: Shape::placeholder(&path),
        fields: vec![],
        params: vec![],
        field_mode: 0,
        start: Instant::now(),
        path,
        defines,
        vopts,
        volume: None,
        iso_drag: None,
        vti_voxel: None,
        opt: OptUi::default(),
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
        job: Arc::new(Mutex::new(Job::default())),
        gpu_name: String::new(),
        quit: false,
        win: None,
        last_poll: Instant::now(),
        frames: 0,
        fps: 0.0,
        fps_t: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_written_path() {
        let log =
            "grid 83x27x43\nwrote /a b/out/x.vti (81x25x41 points, origin -40 -12 0, arrays: sdf, load[3], 2.0 MB)\n";
        assert_eq!(written_path(log), Some(PathBuf::from("/a b/out/x.vti")));
        assert_eq!(
            written_path("wrote out/k.stl (3.3 MB)"),
            Some(PathBuf::from("out/k.stl"))
        );
        assert_eq!(written_path("nothing here"), None);
    }

    #[test]
    fn vti_voxel_caps_samples() {
        let mut s = Shape::placeholder(Path::new("x.wgsl"));
        s.bounds = ([-20.0, -20.0, 0.0], [20.0, 20.0, 22.0]);
        s.voxel = 0.12;
        let v = default_vti_voxel(&s);
        let n: usize = vti_dims(&s, v).iter().product();
        assert!((0.3..0.4).contains(&v) && n <= 1_100_000, "{v} -> {n}");
        s.voxel = 1.0; // already coarse: keep it
        assert_eq!(default_vti_voxel(&s), 1.0);
    }
}
