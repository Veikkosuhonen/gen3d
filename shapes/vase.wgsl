// Twisted polygonal vase. Hollow with a closed floor, so it prints normally
// (or set WALL large and use your slicer's spiral/vase mode on the solid version).
//! bounds -45 -45 0  45 45 120
//! voxel 0.3
#include <sdf.wgsl>

override SIDES: f32 = 7.0;
override WALL: f32 = 1.6;

fn profile(z: f32) -> f32 {            // radius as a function of height
    let t = z / 120.0;
    return 26.0 + 14.0 * sin(t * 3.4 + 0.3) - 6.0 * t;
}

fn map(p: vec3f) -> f32 {
    let q = opTwist(p, deg(120.0) / 120.0);         // 120 degrees over the height
    let a = atan2(q.y, q.x);
    let r = length(q.xy);
    let flutes = 1.2 * cos(a * SIDES);              // fluted cross-section
    var solid = r - (profile(q.z) + flutes);
    solid *= 0.7;                                    // twist makes the field non-exact
    solid = max(solid, max(-p.z, p.z - 118.0));      // cap top and bottom
    let inner = max(r - (profile(q.z) + flutes - WALL), -(p.z - 2.0)); // cavity above a 2 mm floor
    return opSubtract(solid, inner * 0.7);
}
