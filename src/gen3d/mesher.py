"""Evaluate a GLSL SDF on the GPU (compute shader) and polygonize with marching cubes."""

from __future__ import annotations

import time
from pathlib import Path

import os

import numpy as np

os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
os.environ.setdefault("MESA_LOG_LEVEL", "error")

from .shader import Shape, ShaderError, map_errors

COMPUTE_HEAD = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
"""

COMPUTE_TAIL = """
#line 1 999
layout(std430, binding = 0) buffer OutBuf { float outD[]; };
uniform vec3 u_origin;
uniform float u_step;
uniform ivec3 u_dims;   // dims of this slab (nx_slab, ny, nz)
uniform int u_x0;       // slab x offset
void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(g, u_dims))) return;
    vec3 p = u_origin + vec3(g + ivec3(u_x0, 0, 0)) * u_step;
    outD[(g.x * u_dims.y + g.y) * u_dims.z + g.z] = map(p);
}
"""


def create_context(log=print):
    """Headless GL 4.3 context. Prefers a hardware GPU; falls back to software (llvmpipe)."""
    import contextlib
    import io
    import moderngl

    attempts = [{"backend": "egl"}, {}] + [{"backend": "egl", "device_index": i} for i in range(1, 4)]
    software = None
    err = None
    for kw in attempts:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ctx = moderngl.create_standalone_context(require=430, **kw)
        except Exception as e:  # noqa: BLE001
            err = e
            continue
        renderer = ctx.info.get("GL_RENDERER", "")
        if "llvmpipe" in renderer.lower() or "softpipe" in renderer.lower():
            if software is None:
                software = ctx
            else:
                ctx.release()
            continue
        if software:
            software.release()
        return ctx
    if software:
        log("warning: no hardware GL context available, using software rendering (slow)")
        return software
    raise RuntimeError(f"could not create an OpenGL 4.3 context: {err}")


def compile_compute(ctx, shape: Shape):
    src = COMPUTE_HEAD + shape.source + COMPUTE_TAIL
    try:
        return ctx.compute_shader(src)
    except Exception as e:  # moderngl raises plain Error
        raise ShaderError(map_errors(str(e), shape)) from None


def sample_grid(shape: Shape, voxel: float | None = None, ctx=None, log=print):
    """Return (volume[nx,ny,nz] float32, origin, step). Padded by one voxel so the mesh closes."""
    step = voxel or shape.voxel
    lo, hi = np.array(shape.bounds[0]), np.array(shape.bounds[1])
    origin = lo - step  # one voxel of padding on each side
    dims = np.ceil((hi - lo) / step).astype(int) + 3
    nx, ny, nz = (int(d) for d in dims)
    n = nx * ny * nz
    log(f"grid {nx}x{ny}x{nz} = {n / 1e6:.1f}M samples @ {step} mm")
    if n > 1_500_000_000:
        raise ValueError("grid too large; increase voxel size or shrink bounds")

    own = ctx is None
    ctx = ctx or create_context()
    try:
        prog = compile_compute(ctx, shape)
        vol = np.empty((nx, ny, nz), dtype=np.float32)
        slab = max(1, min(nx, (64 * 1024 * 1024) // (ny * nz)))  # ~256 MB per slab
        buf = ctx.buffer(reserve=slab * ny * nz * 4)
        buf.bind_to_storage_buffer(0)
        prog["u_origin"].value = tuple(float(v) for v in origin)
        prog["u_step"].value = float(step)
        for x0 in range(0, nx, slab):
            sx = min(slab, nx - x0)
            prog["u_dims"].value = (sx, ny, nz)
            prog["u_x0"].value = x0
            prog.run((sx + 3) // 4, (ny + 3) // 4, (nz + 3) // 4)
            data = np.frombuffer(buf.read(size=sx * ny * nz * 4), dtype=np.float32)
            vol[x0 : x0 + sx] = data.reshape(sx, ny, nz)
        buf.release()
        prog.release()
    finally:
        if own:
            ctx.release()

    # force the padding shell to be outside so the surface is always closed
    vol[0, :, :] = vol[-1, :, :] = step
    vol[:, 0, :] = vol[:, -1, :] = step
    vol[:, :, 0] = vol[:, :, -1] = step
    if not np.isfinite(vol).all():
        log("warning: SDF produced NaN/Inf values; treating them as outside")
        np.nan_to_num(vol, copy=False, nan=step, posinf=step, neginf=-step)
    return vol, origin, step


def polygonize(vol, origin, step):
    import trimesh
    from skimage.measure import marching_cubes

    if vol.min() >= 0 or vol.max() <= 0:
        raise ValueError("no surface inside bounds (SDF never crosses zero)")
    # Samples on or extremely near the iso-level (common when flat faces like z=0 coincide with
    # grid planes) make marching cubes emit vertices at identical positions. Slicers merge STL
    # vertices by position, which pinches those spots into non-manifold edges. Keeping every
    # sample at least 1% of a voxel away from 0 keeps all vertices distinct (surface shift is
    # at most ~0.01 voxel).
    delta = np.float32(step * 0.01)
    near = np.abs(vol) < delta
    vol[near] = np.where(vol[near] < 0, -delta, delta)
    verts, faces, _normals, _ = marching_cubes(
        vol, level=0.0, spacing=(step, step, step), gradient_direction="ascent"
    )  # keep degenerate faces: dropping them breaks manifoldness, and slicers ignore them
    verts += origin
    # skimage already shares vertices between faces, so the mesh is closed as-is; don't re-merge.
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def build_mesh(shape: Shape, voxel=None, decimate: float | None = None, log=print):
    t0 = time.perf_counter()
    vol, origin, step = sample_grid(shape, voxel, log=log)
    t1 = time.perf_counter()
    mesh = polygonize(vol, origin, step)
    t2 = time.perf_counter()
    log(f"sampled in {t1 - t0:.2f}s, marching cubes in {t2 - t1:.2f}s: {len(mesh.faces):,} triangles")
    if decimate and 0 < decimate < 1:
        target = int(len(mesh.faces) * decimate)
        dec = mesh.simplify_quadric_decimation(face_count=target)
        dec.merge_vertices()
        if dec.is_watertight:
            mesh = dec
            log(f"decimated to {len(mesh.faces):,} triangles")
        else:
            log("warning: decimation broke watertightness (thin features?); keeping the full mesh")
    return mesh


def report(mesh, log=print):
    ext = mesh.extents
    log(
        f"size {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} mm, volume {mesh.volume / 1000:.2f} cm³, "
        f"watertight={mesh.is_watertight}, bodies={mesh.body_count}"
    )
    if mesh.bounds[0][2] < -0.05:
        log(f"note: mesh extends below z=0 (min z {mesh.bounds[0][2]:.2f}); slicers drop it to the bed")


def save_mesh(mesh, out: Path, log=print):
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
