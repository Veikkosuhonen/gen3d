// Organic sculpture: smooth-blended spheres with noise displacement.
//! bounds -35 -35 0  35 35 70
//! voxel 0.25
//! step 0.6
#include <sdf.glsl>

float map(vec3 p) {
    float d = sdSphere(p - vec3(0, 0, 22), 20.0);
    for (int i = 0; i < 6; i++) {
        float a = float(i) * TAU / 6.0;
        vec3 c = vec3(cos(a) * 18.0, sin(a) * 18.0, 22.0 + 18.0 * sin(a * 2.0 + 1.0));
        d = opSmoothUnion(d, sdSphere(p - c, 9.0 + 3.0 * sin(a * 3.0)), 9.0);
    }
    d += (fbm(p * 0.12) - 0.5) * 3.0;               // lumpy surface
    d = opSmoothUnion(d, sdCylinder(p - vec3(0, 0, 2), 18.0, 2.0), 6.0); // flat base
    return max(d, -p.z);                             // flat bottom at z = 0
}
