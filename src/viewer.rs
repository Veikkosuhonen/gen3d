//! The raymarching preview: the WGSL harness, pipelines, and headless PNG rendering. The
//! interactive window that uses them is in `app.rs`.

use std::path::Path;

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;
use crate::shape::{Field, Shape, load_shape, param_data};
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
    g3d_load_params();
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

    // the selected editor primitive where it isn't the visible surface (no-op otherwise)
    col = g3d_overlay(ro, rd, select(1e9, t, hit), col, scale);

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

pub fn field_color(i: usize) -> [f32; 3] {
    FIELD_COLORS[i % FIELD_COLORS.len()]
}

/// The preview harness plus `g3d_field_tint`, which colours surface points inside the shape's
/// `field_*` regions: scalar fields where they are < 0, vector fields where they are non-zero (with
/// bands moving along the vector). Overlapping regions alternate in diagonal stripes fixed to the surface (in world space, so
/// they don't swim when the camera moves). With the editor's primitives (`editor`), the selected
/// one is tinted first.
pub fn preview_harness(fields: &[Field], editor: bool) -> String {
    let mut f = String::from("\nfn g3d_field_tint(p: vec3f, ps: vec3f, scale: f32) -> vec4f {\n");
    if editor {
        f += "    if (abs(g3d_ed_selected(ps)) < scale * 0.003) {\n        \
              let band = step(0.5, fract(dot(ps, vec3f(1.0)) / (scale * 0.02)));\n        \
              return vec4f(mix(vec3f(0.20, 0.65, 1.0), vec3f(0.45, 0.80, 1.0), band), 0.7);\n    }\n";
    }
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
pub struct ViewUniforms {
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
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub target: [f32; 3],
}

impl Camera {
    pub fn framing(shape: &Shape) -> Camera {
        Camera::framing_bounds(shape.bounds)
    }

    /// The default view of a box: from the front left, above, the whole box in view.
    pub fn framing_bounds((lo, hi): ([f32; 3], [f32; 3])) -> Camera {
        let target = [0, 1, 2].map(|k| (lo[k] + hi[k]) / 2.0);
        let diag = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt();
        Camera {
            yaw: (-60f32).to_radians(),
            pitch: 28f32.to_radians(),
            dist: 1.6 * diag,
            target,
        }
    }

    /// (forward, right, up) unit vectors, as `g3d_fs` builds them. A point at camera-space (x, y, z)
    /// (along right, up, forward) lands at uv = 1.8 (x, y) / z, with uv in units of the view height.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let e = self.eye();
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            v.map(|c| c / l)
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let fw = norm([0, 1, 2].map(|k| self.target[k] - e[k]));
        let side = cross(fw, [0.0, 0.0, 1.0]);
        let len = (side[0] * side[0] + side[1] * side[1]).sqrt();
        let rt = if len < 1e-4 { [1.0, 0.0, 0.0] } else { norm(side) };
        (fw, rt, cross(rt, fw))
    }

    pub fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.dist * cp * self.yaw.cos(),
            self.target[1] + self.dist * cp * self.yaw.sin(),
            self.target[2] + self.dist * self.pitch.sin(),
        ]
    }
}

pub struct Options {
    pub clip: bool,
    pub section: bool,
    pub cut: f32,
    pub show_bounds: bool,
    /// bitmask of field overlays to draw
    pub fields: u32,
    pub time: f32,
}

pub fn uniforms(shape: &Shape, cam: &Camera, o: &Options, size: (u32, u32), gamma: bool) -> ViewUniforms {
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

impl ViewUniforms {
    /// Uniforms for drawing a mesh (`meshview`) with bounds `(lo, hi)`; `edges` overlays the
    /// triangle edges.
    pub fn for_mesh(
        (lo, hi): ([f32; 3], [f32; 3]),
        cam: &Camera,
        size: (u32, u32),
        gamma: bool,
        edges: bool,
    ) -> ViewUniforms {
        let e = cam.eye();
        ViewUniforms {
            res: [size.0 as f32, size.1 as f32],
            relax: 1.0,
            cut: 0.0,
            eye: [e[0], e[1], e[2], 0.0],
            target: [cam.target[0], cam.target[1], cam.target[2], 0.0],
            bmin: [lo[0], lo[1], lo[2], 0.0],
            bmax: [hi[0], hi[1], hi[2], 0.0],
            flags: [0, 0, 0, gamma as u32],
            fmask: [0; 4],
            misc: [0.0, 0.0, edges as u32 as f32, 0.0],
        }
    }
}

pub struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    /// the editor's primitives (`g3d_ed`), if the shape has them
    bind_ed: Option<wgpu::BindGroup>,
}

/// A uniform buffer the preview reads values from that change without compiling: the editor's
/// primitives (`Shape::editor_data`) or the live parameters (`shape::param_data`).
pub fn uniform_buffer(gpu: &Gpu, label: &str, data: &[[f32; 4]]) -> wgpu::Buffer {
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// The buffers a preview pipeline binds.
#[derive(Clone)]
pub struct Bindings {
    /// view uniforms (group 0, binding 0)
    pub ubuf: wgpu::Buffer,
    /// a .vti grid's fields (group 0, bindings 1-4)
    pub vol: Option<VolGpu>,
    /// live parameters (group 0, binding 5)
    pub par: Option<wgpu::Buffer>,
    /// editor primitives (group 1, binding 0)
    pub ed: Option<wgpu::Buffer>,
    /// imported grids (group 0, `import::BINDING`)
    pub imp: Option<wgpu::Buffer>,
}

/// What the viewer shows: a .wgsl shape, or a .vti grid turned into a shape by `volume.rs`.
pub struct Source {
    pub shape: Shape,
    pub fields: Vec<Field>,
    pub harness: String,
    pub volume: Option<Volume>,
}

pub fn is_vti(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("vti"))
}

pub fn load_source(path: &Path, defines: &[(String, f64)], vopts: &VolumeOpts) -> Result<Source> {
    if is_vti(path) {
        if !defines.is_empty() {
            anyhow::bail!("-D parameters don't apply to .vti files");
        }
        return Ok(volume_source(path, Volume::load(path, vopts)?));
    }
    let shape = load_shape(path, defines)?;
    let fields = shape.fields()?;
    Ok(Source {
        harness: preview_harness(&fields, shape.editor),
        shape,
        fields,
        volume: None,
    })
}

pub fn volume_source(path: &Path, v: Volume) -> Source {
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
pub struct VolGpu {
    uni: wgpu::Buffer,
    sd: wgpu::Buffer,
    blk: wgpu::Buffer,
    col: wgpu::Buffer,
}

impl VolGpu {
    pub fn new(gpu: &Gpu, v: &Volume) -> Result<VolGpu> {
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
    pub fn upload_uniforms(&self, gpu: &Gpu, v: &Volume) {
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&v.uniforms()));
    }

    pub fn upload(&self, gpu: &Gpu, v: &Volume) {
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&v.uniforms()));
        gpu.queue.write_buffer(&self.sd, 0, bytemuck::cast_slice(&v.sd));
        gpu.queue.write_buffer(&self.blk, 0, bytemuck::cast_slice(&v.blk));
        gpu.queue.write_buffer(&self.col, 0, bytemuck::cast_slice(&v.col));
    }
}

/// Validate the source with its preview harness and create the shader module. Also returns the
/// live parameters (read from a uniform buffer; see `Shape::compile_preview`).
pub fn preview_module(gpu: &Gpu, src: &Source) -> Result<(wgpu::ShaderModule, Vec<String>)> {
    let compiled = src.shape.compile_preview(&src.harness)?;
    Ok((gpu.shader(&compiled, "preview")?, compiled.live))
}

/// The parameter values to apply as `override` constants: all but the live ones.
pub fn compiled_defines(defines: &[(String, f64)], live: &[String]) -> Vec<(String, f64)> {
    defines.iter().filter(|(k, _)| !live.contains(k)).cloned().collect()
}

/// Build the preview pipeline with the given parameter values (`override` constants). Cheap
/// compared to `preview_module`, so the viewer redoes only this when a parameter changes.
pub fn build_pipeline(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    constants: &[(String, f64)],
    format: wgpu::TextureFormat,
    b: &Bindings,
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
    let mut buffers = vec![(0, &b.ubuf)];
    if let Some(v) = &b.vol {
        buffers.extend([(1, &v.uni), (2, &v.sd), (3, &v.blk), (4, &v.col)]);
    }
    if let Some(p) = &b.par {
        buffers.push((5, p));
    }
    if let Some(p) = &b.imp {
        buffers.push((crate::import::BINDING, p));
    }
    let entries: Vec<_> = buffers
        .iter()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: *i,
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
    let bind_ed =
        b.ed.as_ref()
            .map(|b| {
                gpu.scoped(|| {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("editor primitives"),
                        layout: &pipeline.get_bind_group_layout(1),
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: b.as_entire_binding(),
                        }],
                    })
                })
            })
            .transpose()?;
    Ok(Pipeline {
        pipeline,
        bind,
        bind_ed,
    })
}

/// Raymarch the preview into `view`; without a pipeline (nothing compiled yet) fill it dark red.
pub fn draw(gpu: &Gpu, p: Option<&Pipeline>, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
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
            if let Some(b) = &p.bind_ed {
                pass.set_bind_group(1, b, &[]);
            }
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
    let (module, live) = preview_module(&gpu, &src)?;
    let params = if live.is_empty() { vec![] } else { src.shape.params()? };
    let bindings = Bindings {
        ubuf,
        vol,
        par: (!live.is_empty())
            .then(|| uniform_buffer(&gpu, "parameters", &param_data(&live, &params, &src.shape.defines))),
        ed: src
            .shape
            .editor_data(None)
            .map(|d| uniform_buffer(&gpu, "editor primitives", &d)),
        imp: crate::import::buffer(&gpu, &src.shape.imports)?,
    };
    let constants = compiled_defines(&src.shape.defines, &live);
    let pipeline = build_pipeline(&gpu, &module, &constants, format, &bindings)?;
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
    gpu.queue.submit([draw(&gpu, Some(&pipeline), &view)]);
    save_texture_png(&gpu, &tex, size, out)
}

/// Read back an Rgba8 texture (`COPY_SRC`) and write it as an RGB PNG.
pub fn save_texture_png(gpu: &Gpu, tex: &wgpu::Texture, size: (u32, u32), out: &Path) -> Result<()> {
    let (w, h) = size;
    let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixels"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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
    gpu.queue.submit([enc.finish()]);
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

pub fn all_fields(fields: &[Field]) -> u32 {
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
