//! Evaluate a WGSL SDF on the GPU (compute shader) and polygonize it with marching cubes.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;
use crate::mesh::Mesh;
use crate::shape::Shape;

/// Compute entry point `g3d_sample` that evaluates `func(p)` (returning f32, or vec3f when
/// `comps == 3`) over a slab of the grid and writes `comps` floats per sample.
pub fn compute_harness(func: &str, comps: usize) -> String {
    let store = match comps {
        1 => "g3d_out[i] = v;".to_string(),
        _ => (0..comps)
            .map(|k| format!("g3d_out[i * {comps}u + {k}u] = v[{k}];"))
            .collect::<Vec<_>>()
            .join(" "),
    };
    format!(
        r#"
// ---- gen3d sampling harness ----
struct G3dSample {{ origin: vec4f, dims: vec4u }}   // origin.w = voxel size, dims.w = slab x offset
@group(0) @binding(0) var<uniform> g3d_s: G3dSample;
@group(0) @binding(1) var<storage, read_write> g3d_out: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn g3d_sample(@builtin(global_invocation_id) gid: vec3u) {{
    let d = g3d_s.dims.xyz;
    if (any(gid >= d)) {{ return; }}
    let p = g3d_s.origin.xyz + vec3f(gid + vec3u(g3d_s.dims.w, 0u, 0u)) * g3d_s.origin.w;
    let i = (gid.x * d.y + gid.y) * d.z + gid.z;
    let v = {func}(p);
    {store}
}}
"#
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleUniforms {
    origin: [f32; 4],
    dims: [u32; 4],
}

pub struct Grid {
    pub vol: Vec<f32>,
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub step: f64,
}

/// Sample the SDF on a grid padded by one voxel so the mesh always closes.
pub fn sample_grid(shape: &Shape, voxel: Option<f32>, gpu: &Gpu) -> Result<Grid> {
    let voxel = voxel.unwrap_or(shape.voxel);
    let step = voxel as f64;
    if step <= 0.0 {
        bail!("voxel size must be positive");
    }
    let (lo, hi) = shape.bounds;
    let origin = [0, 1, 2].map(|k| lo[k] as f64 - step);
    let dims = [0, 1, 2].map(|k| ((hi[k] - lo[k]) as f64 / step).ceil().max(0.0) as usize + 3);
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    println!("grid {nx}x{ny}x{nz} = {:.1}M samples @ {voxel} mm", n as f64 / 1e6);
    if n > 1_500_000_000 {
        bail!("grid too large; increase voxel size or shrink bounds");
    }
    let mut grid = Grid {
        vol: vec![],
        dims,
        origin,
        step,
    };
    grid.vol = sample_field(shape, &grid, "map", 1, gpu)?;
    let vol = &mut grid.vol;

    // force the padding shell to be outside so the surface is always closed
    let s = step as f32;
    let idx = |x: usize, y: usize, z: usize| (x * ny + y) * nz + z;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1 {
                    vol[idx(x, y, z)] = s;
                }
            }
        }
    }
    let mut bad = false;
    for v in vol.iter_mut() {
        if !v.is_finite() {
            bad = true;
            *v = if *v == f32::NEG_INFINITY { -s } else { s };
        }
    }
    if bad {
        eprintln!("warning: SDF produced NaN/Inf values; treating them as outside");
    }
    Ok(grid)
}

/// Evaluate the shape's function `func` (f32 or vec3f, `comps` floats) at every point of `grid`'s
/// lattice (`grid.vol` is ignored). Returns `comps` floats per point, in the same order as `Grid::vol`.
pub fn sample_field(shape: &Shape, grid: &Grid, func: &str, comps: usize, gpu: &Gpu) -> Result<Vec<f32>> {
    let [nx, ny, nz] = grid.dims;
    let (origin, step) = (grid.origin, grid.step);
    let compiled = shape.compile(&compute_harness(func, comps))?;
    let module = gpu.shader(&compiled, "sample")?;
    let constants = compiled.constants();
    let pipeline = gpu.scoped(|| {
        gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sample"),
            layout: None,
            module: &module,
            entry_point: Some("g3d_sample"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
    })?;

    let limits = gpu.device.limits();
    let max_bytes = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
        .min(256 << 20);
    let layer_bytes = (ny * nz * comps * 4) as u64;
    if layer_bytes > max_bytes {
        bail!("a single grid layer ({ny}x{nz}) exceeds the GPU buffer limit; increase voxel size");
    }
    let max_groups = limits.max_compute_workgroups_per_dimension as usize;
    if ny.div_ceil(4) > max_groups || nz.div_ceil(4) > max_groups {
        bail!("grid exceeds the GPU dispatch limit; increase voxel size");
    }
    let slab = ((max_bytes / layer_bytes) as usize).clamp(1, nx).min(max_groups * 4);
    let buf_size = slab as u64 * layer_bytes;
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("samples"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniforms = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sample uniforms"),
        contents: bytemuck::bytes_of(&SampleUniforms {
            origin: [0.0; 4],
            dims: [0; 4],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let imports = crate::import::buffer(gpu, &shape.imports)?;
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: out.as_entire_binding(),
        },
    ];
    if let Some(b) = &imports {
        entries.push(wgpu::BindGroupEntry {
            binding: crate::import::BINDING,
            resource: b.as_entire_binding(),
        });
    }
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let layer = ny * nz * comps;
    let mut vol = vec![0f32; nx * layer];
    for x0 in (0..nx).step_by(slab) {
        let sx = slab.min(nx - x0);
        let u = SampleUniforms {
            origin: [origin[0] as f32, origin[1] as f32, origin[2] as f32, step as f32],
            dims: [sx as u32, ny as u32, nz as u32, x0 as u32],
        };
        gpu.queue.write_buffer(&uniforms, 0, bytemuck::bytes_of(&u));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(sx.div_ceil(4) as u32, ny.div_ceil(4) as u32, nz.div_ceil(4) as u32);
        }
        gpu.queue.submit([enc.finish()]);
        let bytes = gpu.read_buffer(&out, sx as u64 * layer_bytes)?;
        vol[x0 * layer..(x0 + sx) * layer].copy_from_slice(bytemuck::cast_slice(&bytes));
    }
    Ok(vol)
}

pub fn polygonize(mut grid: Grid) -> Result<Mesh> {
    let (min, max) = grid
        .vol
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    if min >= 0.0 || max <= 0.0 {
        bail!("no surface inside bounds (SDF never crosses zero)");
    }
    // Samples on or extremely near the iso-level (common when flat faces like z=0 coincide with grid
    // planes) would put vertices at identical positions, and slicers that merge STL vertices by
    // position would pinch those spots into non-manifold edges. Keeping every sample at least 1% of a
    // voxel away from 0 keeps all vertices distinct (surface shift is at most ~0.01 voxel).
    let delta = (grid.step * 0.01) as f32;
    for v in &mut grid.vol {
        if v.abs() < delta {
            *v = if *v < 0.0 { -delta } else { delta };
        }
    }
    let mut mesh = crate::mc::polygonize(&grid.vol, grid.dims, grid.origin, grid.step);
    if mesh.volume() < 0.0 {
        mesh.invert();
    }
    Ok(mesh)
}

pub fn build_mesh(shape: &Shape, voxel: Option<f32>, decimate: Option<f32>, min_body: Option<f64>) -> Result<Mesh> {
    let t0 = Instant::now();
    let gpu = crate::gpu::headless()?;
    let grid = sample_grid(shape, voxel, &gpu)?;
    let t1 = Instant::now();
    let mut mesh = polygonize(grid)?;
    let t2 = Instant::now();
    println!(
        "sampled in {:.2}s, marching cubes in {:.2}s: {} triangles",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        fmt_count(mesh.tris.len())
    );
    drop_small_pieces(&mut mesh, min_body);
    Ok(decimate_checked(mesh, decimate))
}

/// `Mesh::drop_small_pieces` below `min_body` % of the largest body, saying what went.
pub fn drop_small_pieces(mesh: &mut Mesh, min_body: Option<f64>) {
    let Some(pct) = min_body.filter(|p| *p > 0.0) else {
        return;
    };
    let (n, vol) = mesh.drop_small_pieces(pct / 100.0);
    if n > 0 {
        println!(
            "removed {n} piece{} under {pct}% of the largest body (volume {:+.3} cm³)",
            if n == 1 { "" } else { "s" },
            -vol / 1000.0
        );
    }
}

/// Taubin smoothing passes for meshes of .vti grids (`Mesh::smooth`).
pub const DEFAULT_SMOOTH: u32 = 10;

/// Decimate to fraction `decimate` of the triangles, unless that breaks watertightness.
pub fn decimate_checked(mut mesh: Mesh, decimate: Option<f32>) -> Mesh {
    if let Some(f) = decimate.filter(|f| *f > 0.0 && *f < 1.0) {
        let (dec, defects) = mesh.decimated(f);
        if defects.is_empty() && dec.is_watertight() {
            mesh = dec;
            println!("decimated to {} triangles", fmt_count(mesh.tris.len()));
        } else {
            let open = defects.iter().filter(|d| d.2 == 1).count();
            eprintln!(
                "warning: decimation broke watertightness ({open} open, {} non-manifold edges; thin features?); \
                 keeping the full mesh",
                defects.len() - open
            );
        }
    }
    mesh
}

pub fn report(mesh: &Mesh) {
    let (lo, hi) = mesh.bounds();
    println!(
        "size {:.2} x {:.2} x {:.2} mm, volume {:.2} cm³, watertight={}, bodies={}",
        hi[0] - lo[0],
        hi[1] - lo[1],
        hi[2] - lo[2],
        mesh.volume() / 1000.0,
        mesh.is_watertight(),
        mesh.body_count()
    );
    if lo[2] < -0.05 {
        println!(
            "note: mesh extends below z=0 (min z {:.2}); slicers drop it to the bed",
            lo[2]
        );
    }
}

pub fn save(mesh: &Mesh, out: &Path) -> Result<()> {
    mesh.save(out)?;
    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!("wrote {} ({:.1} MB)", out.display(), size as f64 / 1e6);
    Ok(())
}

pub fn fmt_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
