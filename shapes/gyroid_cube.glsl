// Rounded cube: solid skin with a gyroid lattice inside. Use C in the viewer to see the section.
//! bounds -21 -21 0  21 21 42
//! voxel 0.2
#include <sdf.glsl>

#ifndef CELL
#define CELL 8.0      // gyroid period, mm
#endif
#ifndef SKIN
#define SKIN 1.2      // outer wall, mm
#endif

float map(vec3 p) {
    vec3 c = p - vec3(0, 0, 20);
    float outer = sdRoundBox(c, vec3(20), 4.0);
    float skin = opShell(outer + SKIN * 0.5, SKIN);           // wall inside the outer surface
    float lattice = opShell(sdGyroid(p, CELL), 0.9);          // 0.9 mm thick gyroid sheet
    float inside = opIntersect(lattice, outer);
    float d = opUnion(skin, inside);
    // a window in the front face so the lattice is visible
    d = opSubtract(d, sdRoundBox(c - vec3(0, -20, 0), vec3(12, 6, 12), 2.0));
    return d;
}
