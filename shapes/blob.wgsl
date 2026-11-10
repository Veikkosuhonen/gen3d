// Organic sculpture: smooth-blended spheres with noise displacement.
//! bounds -35 -35 0  35 35 70
//! voxel 0.25
//! step 0.6
#include <sdf.wgsl>

fn map(p: vec3f) -> f32 {
    var d = sdSphere(p - vec3f(0, 0, 22), 20.0);
    for (var i = 0; i < 6; i++) {
        let a = f32(i) * TAU / 6.0;
        let c = vec3f(cos(a) * 18.0, sin(a) * 18.0, 22.0 + 18.0 * sin(a * 2.0 + 1.0));
        d = opSmoothUnion(d, sdSphere(p - c, 9.0 + 3.0 * sin(a * 3.0)), 9.0);
    }
    d += (fbm(p * 0.12) - 0.5) * 3.0;                                    // lumpy surface
    d = opSmoothUnion(d, sdCylinder(p - vec3f(0, 0, 2), 18.0, 2.0), 6.0); // flat base
    return max(d, -p.z);                                                  // flat bottom at z = 0
}
