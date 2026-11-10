// Twisted polygonal vase. Hollow with a closed floor, so it prints normally
// (or set WALL large and use your slicer's spiral/vase mode on the solid version).
//! bounds -45 -45 0  45 45 120
//! voxel 0.3
#include <sdf.glsl>

#ifndef SIDES
#define SIDES 7.0
#endif
#ifndef WALL
#define WALL 1.6
#endif

float profile(float z) {            // radius as a function of height
    float t = z / 120.0;
    return 26.0 + 14.0 * sin(t * 3.4 + 0.3) - 6.0 * t;
}

float map(vec3 p) {
    vec3 q = opTwist(p, deg(120.0) / 120.0);       // 120 degrees over the height
    float a = atan(q.y, q.x);
    float r = length(q.xy);
    float flutes = 1.2 * cos(a * SIDES);           // fluted cross-section
    float solid = r - (profile(q.z) + flutes);
    solid *= 0.7;                                   // twist makes the field non-exact
    solid = max(solid, max(-p.z, p.z - 118.0));     // cap top and bottom
    float inner = max(r - (profile(q.z) + flutes - WALL), -(p.z - 2.0)); // cavity above a 2 mm floor
    return opSubtract(solid, inner * 0.7);
}
