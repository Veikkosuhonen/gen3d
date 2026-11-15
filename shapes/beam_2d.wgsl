// A simply supported beam (the classic MBB problem), optimized in 2D: a 120 x 20 x 40 mm block
// resting on two pads at the ends of its bottom, pushed down in the middle of its top. The problem
// is the same through the beam's 20 mm thickness (y), so `//! 2d y` optimizes its XZ profile and
// extrudes the result through it.
//! bounds -60 -10 0  60 10 40
//! voxel 0.5
//! 2d y
#include <sdf.wgsl>

fn map(p: vec3f) -> f32 {
    return sdBox(p - vec3f(0, 0, 20), vec3f(60, 10, 20));
}

// pads 6 mm wide under both ends (one clamped, the other too: a fixed-fixed beam on its pads)
fn field_fixed(p: vec3f) -> f32 {
    return sdBox(vec3f(abs(p.x) - 57.0, p.y, p.z), vec3f(3, 11, 1.5));
}

// the load: a 6 mm strip across the middle of the top
fn field_load(p: vec3f) -> vec3f {
    if (sdBox(p - vec3f(0, 0, 40), vec3f(3, 11, 1.5)) < 0.0) { return vec3f(0, 0, -1); }
    return vec3f(0);
}
