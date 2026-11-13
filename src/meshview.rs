//! Drawing a triangle mesh (an exported STL/3MF) in the viewport, in place of the raymarched shape:
//! the same camera, background, ground grid and lights as the preview (`viewer.rs`), so switching
//! between the two shows exactly what changed. Flat shading per face, so the facets the mesher made
//! are visible, with an optional overlay of the triangle edges. Faces seen from behind are drawn in
//! a different colour: they show holes and inverted triangles.

use std::path::Path;

use anyhow::{Result, bail};
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;
use crate::mesh::Mesh;
use crate::viewer::{Camera, ViewUniforms};

const WGSL: &str = r#"
struct G3dView {
    res: vec2f,
    relax: f32,
    cut: f32,
    eye: vec4f,
    look: vec4f,
    bmin: vec4f,
    bmax: vec4f,
    flags: vec4u,     // w: apply gamma
    fmask: vec4u,
    misc: vec4f,      // z: draw triangle edges
}
@group(0) @binding(0) var<uniform> V: G3dView;

// the camera basis `g3d_fs` uses: forward, right, up
fn basis() -> mat3x3f {
    let fw = normalize(V.look.xyz - V.eye.xyz);
    let side = cross(fw, vec3f(0, 0, 1));
    let rt = select(normalize(side), vec3f(1, 0, 0), length(side) < 1e-4);
    return mat3x3f(fw, rt, cross(rt, fw));
}

fn finish(c: vec3f) -> vec4f {
    var col = c;
    if (V.flags.w == 1u) { col = pow(max(col, vec3f(0.0)), vec3f(0.4545)); }
    return vec4f(col, 1.0);
}

// ---- background: sky and ground grid, as in the preview (without shadows)

@vertex
fn bg_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 1.0, 1.0);
}

fn grid_line(p: vec2f, fw: vec2f, spacing: f32, width: f32) -> f32 {
    let g = abs(fract(p / spacing - 0.5) - 0.5) * spacing;
    let l = 1.0 - smoothstep(vec2f(0.0), fw * width, g);
    return max(l.x, l.y);
}

@fragment
fn bg_fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let res = V.res;
    let fc = vec2f(frag.x, res.y - frag.y);
    let uv = (fc - 0.5 * res) / res.y;
    let b = basis();
    let ro = V.eye.xyz;
    let rd = normalize(b[0] * 1.8 + uv.x * b[1] + uv.y * b[2]);
    let scale = length(V.bmax.xyz - V.bmin.xyz);
    var col = mix(vec3f(0.10, 0.11, 0.13), vec3f(0.20, 0.22, 0.26), 0.5 + 0.5 * rd.z);
    let tPlane = select(1e9, -ro.z / rd.z, rd.z < 0.0);
    let p = ro + rd * min(tPlane, scale * 1000.0);
    let dp = fwidth(p.xy);
    if (tPlane < 1e8) {
        let minor = grid_line(p.xy, dp, 1.0, 1.0) * 0.25;
        let major = grid_line(p.xy, dp, 10.0, 1.5);
        let axisX = 1.0 - smoothstep(0.0, dp.x * 2.0, abs(p.x));
        let axisY = 1.0 - smoothstep(0.0, dp.y * 2.0, abs(p.y));
        var g = vec3f(0.17, 0.18, 0.20);
        g += vec3f(0.05) * minor + vec3f(0.12) * major;
        g = mix(g, vec3f(0.8, 0.25, 0.25), axisY * 0.8);
        g = mix(g, vec3f(0.25, 0.75, 0.3), axisX * 0.8);
        let fade = exp(-tPlane / (scale * 6.0));
        col = mix(col, g, fade);
    }
    return finish(col);
}

// ---- the mesh

@group(0) @binding(1) var<storage, read> verts: array<f32>;   // xyz per vertex
@group(0) @binding(2) var<storage, read> tris: array<u32>;    // 3 indices per triangle

struct Out {
    @builtin(position) pos: vec4f,
    @location(0) wp: vec3f,
    @location(1) bary: vec3f,
    @location(2) @interpolate(flat) n: vec3f,
}

fn vert(i: u32) -> vec3f {
    return vec3f(verts[3u * i], verts[3u * i + 1u], verts[3u * i + 2u]);
}

@vertex
fn mesh_vs(@builtin(vertex_index) vi: u32) -> Out {
    let t = vi / 3u;
    let c = vi % 3u;
    let a = vert(tris[3u * t]);
    let b = vert(tris[3u * t + 1u]);
    let d = vert(tris[3u * t + 2u]);
    let p = array<vec3f, 3>(a, b, d)[c];
    // camera space, then the preview's projection: uv = 1.8 (x, y) / z in units of the view height
    let m = basis();
    let q = p - V.eye.xyz;
    let z = dot(q, m[0]);
    let x = dot(q, m[1]);
    let y = dot(q, m[2]);
    let scale = length(V.bmax.xyz - V.bmin.xyz);
    let near = scale * 1e-3;
    let far = scale * 100.0;
    var o: Out;
    o.pos = vec4f(1.8 * x * 2.0 * V.res.y / V.res.x, 1.8 * y * 2.0, far * (z - near) / (far - near), z);
    o.wp = p;
    o.bary = array<vec3f, 3>(vec3f(1, 0, 0), vec3f(0, 1, 0), vec3f(0, 0, 1))[c];
    let n = cross(b - a, d - a);
    o.n = select(vec3f(0, 0, 1), normalize(n), dot(n, n) > 0.0);
    return o;
}

@fragment
fn mesh_fs(i: Out) -> @location(0) vec4f {
    // edge distance in pixels (derivatives in uniform control flow)
    let w = fwidth(i.bary);
    let e3 = smoothstep(vec3f(0.0), w * 1.2, i.bary);
    let edge = 1.0 - min(e3.x, min(e3.y, e3.z));

    let rd = normalize(i.wp - V.eye.xyz);
    var n = i.n;
    var base = vec3f(0.95, 0.62, 0.22);
    if (dot(n, rd) > 0.0) {
        // the back of a face: a hole or an inverted triangle
        n = -n;
        base = vec3f(0.35, 0.45, 0.95);
    }
    let light = normalize(vec3f(0.6, -0.4, 0.9));
    let dif = clamp(dot(n, light), 0.0, 1.0);
    let back = clamp(dot(n, -light), 0.0, 1.0);
    let rim = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 3.0);
    let h = normalize(light - rd);
    let spe = pow(clamp(dot(n, h), 0.0, 1.0), 48.0) * dif;
    let sky = 0.20 + 0.10 * n.z;
    let bounce = 0.10 * clamp(-n.z, 0.0, 1.0);
    var col = base * (dif + sky + bounce + 0.15 * back) + 0.25 * rim + 0.3 * spe;
    if (V.misc.z > 0.5) {
        col = mix(col, col * 0.45, edge);
    }
    return finish(col);
}
"#;

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The pipelines, the background's bindings and a depth buffer for the viewport's size.
pub struct MeshRenderer {
    bg: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    bg_bind: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    depth: Option<((u32, u32), wgpu::TextureView)>,
}

/// A mesh on the GPU, ready to draw with the `MeshRenderer` that uploaded it.
pub struct MeshGpu {
    bind: wgpu::BindGroup,
    tris: u32,
}

impl MeshRenderer {
    /// Pipelines drawing into `format` targets, reading the view uniforms from `ubuf`.
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat, ubuf: &wgpu::Buffer) -> Result<MeshRenderer> {
        let module = gpu.scoped(|| {
            gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh view"),
                source: wgpu::ShaderSource::Wgsl(WGSL.into()),
            })
        })?;
        let pipeline = |vs: &str, fs: &str, depth_write: bool, compare: wgpu::CompareFunction| {
            gpu.scoped(|| {
                gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(fs),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH,
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(compare),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(fs),
                        compilation_options: Default::default(),
                        targets: &[Some(format.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            })
        };
        let bg = pipeline("bg_vs", "bg_fs", false, wgpu::CompareFunction::Always)?;
        let mesh = pipeline("mesh_vs", "mesh_fs", true, wgpu::CompareFunction::Less)?;
        let bg_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh view background"),
            layout: &bg.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        Ok(MeshRenderer {
            bg,
            mesh,
            bg_bind,
            ubuf: ubuf.clone(),
            depth: None,
        })
    }

    /// Copy `m` to the GPU.
    pub fn upload(&self, gpu: &Gpu, m: &Mesh) -> Result<MeshGpu> {
        if m.tris.is_empty() {
            bail!("the mesh has no triangles");
        }
        let limits = gpu.device.limits();
        let max = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let bytes = (m.verts.len().max(m.tris.len()) * 12) as u64;
        if bytes > max {
            bail!(
                "mesh too large to show on this GPU ({} triangles; at most {} MB per array)",
                m.tris.len(),
                max / 1_000_000
            );
        }
        let buf = |label, data: &[u8]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let verts = buf("mesh vertices", bytemuck::cast_slice(&m.verts));
        let tris = buf("mesh triangles", bytemuck::cast_slice(&m.tris));
        let bind = gpu.scoped(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh"),
                layout: &self.mesh.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ubuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: verts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tris.as_entire_binding(),
                    },
                ],
            })
        })?;
        Ok(MeshGpu {
            bind,
            tris: m.tris.len() as u32,
        })
    }

    /// Draw the background and `mesh` into `view` (of `size` pixels).
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        mesh: &MeshGpu,
        view: &wgpu::TextureView,
        size: (u32, u32),
    ) -> wgpu::CommandBuffer {
        if self.depth.as_ref().is_none_or(|(s, _)| *s != size) {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mesh view depth"),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((size, tex.create_view(&Default::default())));
        }
        let depth = &self.depth.as_ref().unwrap().1;
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.bg);
            pass.set_bind_group(0, &self.bg_bind, &[]);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.mesh);
            pass.set_bind_group(0, &mesh.bind, &[]);
            pass.draw(0..mesh.tris * 3, 0..1);
        }
        enc.finish()
    }
}

/// Render a mesh file headlessly to a PNG (`gen3d render model.stl`), framed like a shape with its
/// bounds.
pub fn render_png(path: &Path, out: &Path, size: (u32, u32), yaw: f32, pitch: f32, edges: bool) -> Result<()> {
    let m = Mesh::load(path)?;
    let bounds = m.bounds();
    let gpu = crate::gpu::headless()?;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut cam = Camera::framing_bounds(bounds);
    cam.yaw = yaw.to_radians();
    cam.pitch = pitch.to_radians();
    let ubuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("view"),
        contents: bytemuck::bytes_of(&ViewUniforms::for_mesh(bounds, &cam, size, true, edges)),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut r = MeshRenderer::new(&gpu, format, &ubuf)?;
    let mg = r.upload(&gpu, &m)?;
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
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
    gpu.queue.submit([r.draw(&gpu, &mg, &view, size)]);
    println!("{}: {} triangles", path.display(), m.tris.len());
    crate::viewer::save_texture_png(&gpu, &tex, size, out)
}

pub fn is_mesh(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["stl", "3mf", "obj"].contains(&e.to_lowercase().as_str()))
}
