//! Live raymarching preview window with hot reload, and headless PNG rendering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::gpu::Gpu;
use crate::shape::{Field, Shape, load_shape};
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

    fn upload(&self, gpu: &Gpu, v: &Volume) {
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&v.uniforms()));
        gpu.queue.write_buffer(&self.sd, 0, bytemuck::cast_slice(&v.sd));
        gpu.queue.write_buffer(&self.blk, 0, bytemuck::cast_slice(&v.blk));
        gpu.queue.write_buffer(&self.col, 0, bytemuck::cast_slice(&v.col));
    }
}

fn build_pipeline(
    gpu: &Gpu,
    src: &Source,
    format: wgpu::TextureFormat,
    ubuf: &wgpu::Buffer,
    vol: Option<&VolGpu>,
) -> Result<Pipeline> {
    let compiled = src.shape.compile(&src.harness)?;
    let module = gpu.shader(&compiled, "preview")?;
    let constants = compiled.constants();
    let options = || wgpu::PipelineCompilationOptions {
        constants: &constants,
        ..Default::default()
    };
    let pipeline = gpu.scoped(|| {
        gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("g3d_vs"),
                compilation_options: options(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
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

fn draw(gpu: &Gpu, p: &Pipeline, view: &wgpu::TextureView, clear: Option<wgpu::Color>) -> wgpu::CommandBuffer {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear.unwrap_or(wgpu::Color::BLACK)),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        if clear.is_none() {
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
    let pipeline = build_pipeline(&gpu, &src, format, &ubuf, vol.as_ref())?;
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
    let cmd = draw(&gpu, &pipeline, &view, None);
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

const HELP: &str = "gen3d viewer — edit the .wgsl file in any editor; it reloads on save (.vti files too).
  mouse: left-drag orbit · right/middle-drag pan · wheel zoom
  1/2/3/4 front/side/top/iso   R reset camera   F5 force reload
  C section view   [ ] move section plane   B toggle clip-to-bounds   V toggle bounds box
  F cycle field overlays (all, each field alone, off)
  .vti: - = threshold   M smooth/blocky   I flip inside   A next array   F colour by array
  E export mesh    G slice to G-code        O open in slicer GUI      Q/Esc quit";

struct Window_ {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gamma: bool,
    gpu: Gpu,
    ubuf: wgpu::Buffer,
    pipeline: Option<Pipeline>,
    vol: Option<VolGpu>,
}

struct Viewer {
    path: PathBuf,
    defines: Vec<(String, f64)>,
    /// .vti display options from the command line
    vopts: VolumeOpts,
    volume: Option<Volume>,
    shape: Shape,
    fields: Vec<Field>,
    /// overlay mode: 0 = all fields, 1..=n = only field n-1, n+1 = none
    field_mode: usize,
    start: Instant,
    loaded: bool,
    mtimes: HashMap<PathBuf, Option<SystemTime>>,
    error: bool,
    cam: Camera,
    opts: Options,
    drag: Option<MouseButton>,
    cursor: (f64, f64),
    busy: Arc<AtomicBool>,
    win: Option<Window_>,
    last_poll: Instant,
    frames: u32,
    fps_t: Instant,
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

fn stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
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
            },
            None => self.vopts.clone(),
        }
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
    fn install(&mut self, src: Source, first: bool) {
        let Some(win) = &mut self.win else { return };
        let vol = match src.volume.as_ref().map(|v| VolGpu::new(&win.gpu, v)).transpose() {
            Ok(v) => v,
            Err(e) => return self.fail(&format!("{e:#}")),
        };
        match build_pipeline(&win.gpu, &src, win.config.format, &win.ubuf, vol.as_ref()) {
            Ok(p) => {
                win.pipeline = Some(p);
                win.vol = vol;
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
            print_legend(&fields);
        }
        self.fields = fields;
        self.set_field_mask();
        self.loaded = true;
        self.error = false;
        if first || bounds_changed {
            self.reset_camera();
        }
        println!(
            "[{}] loaded {}",
            stamp(),
            self.path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    fn fail(&mut self, msg: &str) {
        self.error = true;
        let what = if self.volume.is_some() || is_vti(&self.path) {
            "error"
        } else {
            "shader error"
        };
        println!("\n[{}] \x1b[31m{what}\x1b[0m\n{}\n", stamp(), msg.trim());
    }

    fn set_field_mask(&mut self) {
        let n = self.fields.len();
        self.opts.fields = match self.field_mode {
            0 => all_fields(&self.fields),
            m if m <= n.min(32) => 1 << (m - 1),
            _ => 0,
        };
    }

    fn field_label(&self) -> Option<String> {
        let n = self.fields.len();
        match self.field_mode {
            _ if n == 0 => None,
            0 => Some("fields: all".into()),
            m if m <= n => Some(format!("field: {}", self.fields[m - 1].name)),
            _ => Some("fields: off".into()),
        }
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

    fn reset_camera(&mut self) {
        self.cam = Camera::framing(&self.shape);
        self.opts.cut = self.cam.target[1];
    }

    fn run_bg(&self, cmd: &str) {
        if self.busy.swap(true, Ordering::SeqCst) {
            println!("busy, wait for the previous job");
            return;
        }
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("gen3d"));
        let mut c = std::process::Command::new(exe);
        c.arg(cmd).arg(&self.path);
        if let Some(v) = &self.volume {
            c.args(["--array", v.array_name(), "--iso", &v.iso.to_string()]);
            c.arg(if v.below { "--below" } else { "--above" });
        }
        for (k, v) in &self.defines {
            c.arg("-D").arg(format!("{k}={v}"));
        }
        let busy = self.busy.clone();
        std::thread::spawn(move || {
            if let Err(e) = c.status() {
                eprintln!("could not run job: {e}");
            }
            busy.store(false, Ordering::SeqCst);
        });
    }

    fn on_key(&mut self, key: KeyCode, el: &ActiveEventLoop) {
        let step = self.shape.size()[1] / 50.0;
        let view = |y: f32, p: f32| (y.to_radians(), p.to_radians());
        match key {
            KeyCode::Escape | KeyCode::KeyQ => el.exit(),
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
            KeyCode::KeyM if self.volume.is_some() => self.edit_volume(false, |v| v.blocky = !v.blocky),
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
            KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4 => {
                (self.cam.yaw, self.cam.pitch) = match key {
                    KeyCode::Digit1 => view(-90.0, 0.0),
                    KeyCode::Digit2 => view(0.0, 0.0),
                    KeyCode::Digit3 => view(-90.0, 89.9),
                    _ => view(-60.0, 28.0),
                };
            }
            _ => {}
        }
    }

    fn on_move(&mut self, x: f64, y: f64) {
        let (dx, dy) = ((x - self.cursor.0) as f32, (y - self.cursor.1) as f32);
        self.cursor = (x, y);
        match self.drag {
            None => {}
            Some(MouseButton::Left) => {
                self.cam.yaw -= dx * 0.008;
                self.cam.pitch = (self.cam.pitch + dy * 0.008).clamp(-1.55, 1.55);
            }
            Some(_) => {
                let h = self.win.as_ref().map_or(800, |w| w.config.height.max(1));
                let s = self.cam.dist / h as f32 * 1.1;
                let right = (-self.cam.yaw.sin(), self.cam.yaw.cos());
                self.cam.target[0] -= right.0 * dx * s;
                self.cam.target[1] -= right.1 * dx * s;
                self.cam.target[2] += dy * s;
            }
        }
    }

    fn render(&mut self) {
        let Some(win) = &mut self.win else { return };
        let frame = match win.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                win.surface.configure(&win.gpu.device, &win.config);
                return;
            }
            _ => return,
        };
        let size = (win.config.width, win.config.height);
        self.opts.time = self.start.elapsed().as_secs_f32();
        let u = uniforms(&self.shape, &self.cam, &self.opts, size, win.gamma);
        win.gpu.queue.write_buffer(&win.ubuf, 0, bytemuck::bytes_of(&u));
        let view = frame.texture.create_view(&Default::default());
        let cmd = match &win.pipeline {
            Some(p) => draw(&win.gpu, p, &view, None),
            None => {
                // nothing compiled yet: dark red screen until the shape loads
                let mut enc = win.gpu.device.create_command_encoder(&Default::default());
                enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.3,
                                g: 0.05,
                                b: 0.05,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                enc.finish()
            }
        };
        win.gpu.queue.submit([cmd]);
        win.window.pre_present_notify();
        win.gpu.queue.present(frame);
        self.frames += 1;
    }

    fn update_title(&mut self) {
        let now = Instant::now();
        let dt = (now - self.fps_t).as_secs_f32();
        if dt < 0.5 {
            return;
        }
        let Some(win) = &self.win else { return };
        let status = if self.error {
            "ERROR (see terminal)".to_string()
        } else {
            format!("{:.0} fps", self.frames as f32 / dt)
        };
        let mut flags = vec![];
        if self.opts.section {
            flags.push(format!("section y={:.1}", self.opts.cut));
        }
        if !self.opts.clip {
            flags.push("unclipped".into());
        }
        if let Some(l) = self.field_label() {
            flags.push(l);
        }
        if let Some(v) = &self.volume {
            flags.push(v.label());
        }
        if self.busy.load(Ordering::SeqCst) {
            flags.push("working…".into());
        }
        let name = self.path.file_name().unwrap_or_default().to_string_lossy();
        win.window
            .set_title(&format!("gen3d — {name} — {status} {}", flags.join(" ")));
        self.frames = 0;
        self.fps_t = now;
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.win.is_some() {
            return;
        }
        let name = self.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let attrs = Window::default_attributes()
            .with_title(format!("gen3d — {name}"))
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let result = (|| -> Result<Window_> {
            let window = Arc::new(el.create_window(attrs)?);
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
                el.owned_display_handle(),
            )));
            let surface = instance.create_surface(window.clone())?;
            let gpu = crate::gpu::create(instance, Some(&surface), true)?;
            let caps = surface.get_capabilities(&gpu.adapter);
            // the shader applies its own gamma; prefer a linear (non-sRGB) swapchain format
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
            Ok(Window_ {
                window,
                surface,
                config,
                gamma: !format.is_srgb(),
                gpu,
                ubuf,
                pipeline: None,
                vol: None,
            })
        })();
        match result {
            Ok(w) => {
                self.win = Some(w);
                println!("{HELP}");
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
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(win) = &mut self.win {
                    win.config.width = size.width.max(1);
                    win.config.height = size.height.max(1);
                    win.surface.configure(&win.gpu.device, &win.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.on_key(code, el);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = (state == ElementState::Pressed).then_some(button);
            }
            WindowEvent::CursorMoved { position, .. } => self.on_move(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.cam.dist *= 0.9f32.powf(dy);
            }
            WindowEvent::RedrawRequested => {
                self.render();
                self.update_title();
                if let Some(w) = &self.win {
                    w.window.request_redraw();
                }
            }
            _ => {}
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
        field_mode: 0,
        start: Instant::now(),
        path,
        defines,
        vopts,
        volume: None,
        loaded: false,
        mtimes: HashMap::new(),
        error: false,
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
        drag: None,
        cursor: (0.0, 0.0),
        busy: Arc::new(AtomicBool::new(false)),
        win: None,
        last_poll: Instant::now(),
        frames: 0,
        fps_t: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
