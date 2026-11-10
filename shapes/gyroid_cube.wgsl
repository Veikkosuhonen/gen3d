// Rounded cube: solid skin with a gyroid lattice inside. Use C in the viewer to see the section.
//! bounds -21 -21 0  21 21 42
//! voxel 0.2
#include <sdf.wgsl>

override CELL: f32 = 8.0;   // gyroid period, mm
override SKIN: f32 = 1.2;   // outer wall, mm

fn map(p: vec3f) -> f32 {
    let c = p - vec3f(0, 0, 20);
    let outer = sdRoundBox(c, vec3f(20.0), 4.0);
    let skin = opShell(outer + SKIN * 0.5, SKIN);            // wall inside the outer surface
    let lattice = opShell(sdGyroid(p, CELL), 0.9);           // 0.9 mm thick gyroid sheet
    let inside = opIntersect(lattice, outer);
    var d = opUnion(skin, inside);
    // a window in the front face so the lattice is visible
    d = opSubtract(d, sdRoundBox(c - vec3f(0, -20, 0), vec3f(12, 6, 12), 2.0));
    return d;
}
