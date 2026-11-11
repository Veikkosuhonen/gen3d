// Topology-optimization problem for a wall bracket: the design domain plus boundary conditions.
// A plate at x = -40 bolts to a wall with two M5 bolts; a pin through the hole at the free end
// carries a downward load. Export with: gen3d export shapes/bracket_domain.wgsl --format vti
//
// `map` is the design domain (where material may go). Each `fn field_<name>` becomes an extra
// array in the .vti; gen3d does not interpret them, they are conventions for your solver:
//   fixed  (f32)   < 0 where nodes are clamped (here: the bolt hole walls)
//   load   (vec3f) force direction × weight where load is applied, zero elsewhere (pin hole wall);
//                  scale it in the solver so the total matches the force you want
//   keep   (f32)   < 0 where material is forced solid (non-design: mounting plate, pin boss)
//! bounds -40 -12 0  40 12 40
//! voxel 1
#include <sdf.wgsl>

override BOLT_R: f32 = 2.75;   // M5 clearance
override PIN_R: f32 = 3.0;
override WALL: f32 = 1.5;      // thickness of the fixed / loaded shells around the holes, mm

fn bolts(p: vec3f) -> f32 {
    let q = vec3f(p.x + 37.0, p.y, abs(p.z - 20.0) - 12.0);   // two holes, at z = 8 and z = 32
    return sdCylinder(rotY(q, deg(90.0)), BOLT_R, 3.01);
}

fn pin(p: vec3f) -> f32 {
    return sdCylinder(rotX(p - vec3f(32.0, 0, 20), deg(90.0)), PIN_R, 12.01);
}

fn map(p: vec3f) -> f32 {
    let domain = sdBox(p - vec3f(0, 0, 20), vec3f(40, 12, 20));
    return opSubtract(opSubtract(domain, bolts(p)), pin(p));
}

fn field_fixed(p: vec3f) -> f32 {
    return max(max(bolts(p) - WALL, -bolts(p)), p.x + 34.0);   // hole wall only, not past its end
}

fn field_load(p: vec3f) -> vec3f {
    let shell = max(pin(p) - WALL, -pin(p));
    if (shell < 0.0) { return vec3f(0, 0, -1); }
    return vec3f(0);
}

fn field_keep(p: vec3f) -> f32 {
    let plate = sdBox(p - vec3f(-37, 0, 20), vec3f(3, 12, 20));
    let boss = sdCylinder(rotX(p - vec3f(32.0, 0, 20), deg(90.0)), PIN_R + 4.0, 12.0);
    return min(plate, boss);
}
