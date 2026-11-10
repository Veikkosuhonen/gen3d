// Knurled control knob for a 6 mm D-shaft. Tweak with e.g.: gen3d export shapes/knob.wgsl -D DIAM=30
//! bounds -20 -20 0  20 20 22
//! voxel 0.12
#include <sdf.wgsl>

override DIAM: f32 = 26.0;
override HEIGHT: f32 = 16.0;
override SHAFT: f32 = 6.15;   // shaft diameter + clearance

fn map(p: vec3f) -> f32 {
    let R = DIAM * 0.5;
    // body: cylinder with a 1.5 mm rounded top edge; bottom kept sharp & flat for the bed
    let q = p - vec3f(0, 0, HEIGHT * 0.5);
    var body = sdRoundCylinder(q, R, HEIGHT * 0.5, 1.5);
    body = min(body, sdCylinder(q + vec3f(0, 0, HEIGHT * 0.25), R, HEIGHT * 0.25));

    // knurls: 30 grooves around the side, stopping below the rounded top
    let pr = opPolar(p, 30.0);
    let groove = sdCylinder(pr - vec3f(R + 0.35, 0, 0), 0.9, HEIGHT - 2.5);
    body = opSmoothSubtract(body, groove, 0.3);

    // pointer dimple on top
    body = opSmoothSubtract(body, sdCapsule(p, vec3f(0, R * 0.35, HEIGHT), vec3f(0, R * 0.8, HEIGHT), 1.2), 0.5);

    // D-shaft hole, 12 mm deep
    var hole = sdCylinder(p - vec3f(0, 0, 6), SHAFT * 0.5, 6.0 + 0.01);
    hole = opIntersect(hole, -(p.y - (SHAFT * 0.5 - 0.5)));   // the flat of the D
    hole = min(hole, sdCappedCone(p - vec3f(0, 0, 0.5), 0.6, SHAFT * 0.5 + 0.6, SHAFT * 0.5)); // entry chamfer
    return opSubtract(body, hole);
}
