//! Marching cubes with a generated, crack-free case table.
//!
//! Instead of the classic hand-written 256-entry table (whose ambiguous faces can disagree between
//! neighbouring cubes and leave holes), the table is generated from one rule applied per cube face:
//! walking a face counter-clockwise, every outside→inside crossing is joined to the next
//! inside→outside crossing. Neighbouring cubes see the same face and therefore always produce the
//! same segments, so the surface is closed. Segments chain into loops on the cube surface that are
//! triangulated either as a fan or, when a fan diagonal could coincide with a neighbour's edge, around
//! an extra centre vertex. Shared vertices are keyed by grid edge, so the output is an indexed,
//! watertight, consistently oriented (outward, counter-clockwise) mesh.

use std::sync::OnceLock;

use crate::mesh::Mesh;

struct Loop {
    edges: Vec<u8>,
    /// Index into `edges` to fan-triangulate from, or `None` to triangulate around a centre vertex.
    fan_start: Option<usize>,
}

fn corner(x: usize, y: usize, z: usize) -> usize {
    x | (y << 1) | (z << 2)
}

/// Corners of cube edge `e`. Edges 0-3 run along x, 4-7 along y, 8-11 along z.
fn edge_corners(e: usize) -> (usize, usize) {
    let (k, axis) = (e % 4, e / 4);
    let (a, b) = (k & 1, k >> 1);
    let c0 = match axis {
        0 => corner(0, a, b),
        1 => corner(a, 0, b),
        _ => corner(a, b, 0),
    };
    (c0, c0 | (1 << axis))
}

fn edge_between(c0: usize, c1: usize) -> usize {
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    (0..12)
        .find(|&e| edge_corners(e) == (lo, hi))
        .expect("corners are not adjacent")
}

/// The 6 faces as corner cycles, counter-clockwise when viewed from outside the cube.
fn faces() -> Vec<[usize; 4]> {
    let mut out = vec![];
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in 0..2 {
            let mut cyc = [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(cu, cv)| (side << axis) | (cu << u) | (cv << v));
            if side == 0 {
                cyc.reverse();
            }
            out.push(cyc);
        }
    }
    out
}

fn table() -> &'static Vec<Vec<Loop>> {
    static TABLE: OnceLock<Vec<Vec<Loop>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let faces = faces();
        let face_edges: Vec<[usize; 4]> = faces
            .iter()
            .map(|f| [0, 1, 2, 3].map(|k| edge_between(f[k], f[(k + 1) % 4])))
            .collect();
        let share_face = |a: usize, b: usize| face_edges.iter().any(|f| f.contains(&a) && f.contains(&b));

        (0..256usize)
            .map(|case| {
                let inside = |c: usize| case >> c & 1 == 1;
                let mut next = [usize::MAX; 12];
                for f in &faces {
                    // (edge, entering) for each sign change, in counter-clockwise order
                    let crossings: Vec<(usize, bool)> = (0..4)
                        .filter(|&k| inside(f[k]) != inside(f[(k + 1) % 4]))
                        .map(|k| (edge_between(f[k], f[(k + 1) % 4]), inside(f[(k + 1) % 4])))
                        .collect();
                    for (i, &(e, entering)) in crossings.iter().enumerate() {
                        if entering {
                            let exit = (1..crossings.len())
                                .map(|j| crossings[(i + j) % crossings.len()])
                                .find(|&(_, ent)| !ent)
                                .expect("unbalanced face crossings");
                            next[e] = exit.0;
                        }
                    }
                }
                let mut seen = [false; 12];
                let mut loops = vec![];
                for start in 0..12 {
                    if next[start] == usize::MAX || seen[start] {
                        continue;
                    }
                    let mut edges = vec![];
                    let mut e = start;
                    while !seen[e] {
                        seen[e] = true;
                        edges.push(e as u8);
                        e = next[e];
                    }
                    assert_eq!(e, start, "segments do not close into a loop");
                    let n = edges.len();
                    let fan_start = (0..n)
                        .find(|&s| (2..n - 1).all(|i| !share_face(edges[s] as usize, edges[(s + i) % n] as usize)));
                    loops.push(Loop { edges, fan_start });
                }
                loops
            })
            .collect()
    })
}

/// Polygonize the zero level set of `vol` (layout `[x][y][z]`, z fastest). Samples must not be
/// exactly zero (the caller nudges them), and the outer shell must be positive for a closed mesh.
pub fn polygonize(vol: &[f32], dims: [usize; 3], origin: [f64; 3], step: f64) -> Mesh {
    let [nx, ny, nz] = dims;
    let table = table();
    let idx = |x: usize, y: usize, z: usize| (x * ny + y) * nz + z;
    let pos = |x: usize, y: usize, z: usize| {
        [
            origin[0] + x as f64 * step,
            origin[1] + y as f64 * step,
            origin[2] + z as f64 * step,
        ]
    };

    let mut mesh = Mesh::default();
    let layer = ny * nz;
    // vertex ids per grid edge: y- and z-edges for the two layers of the current slab, x-edges between
    let mut ye = [vec![u32::MAX; layer], vec![u32::MAX; layer]];
    let mut ze = [vec![u32::MAX; layer], vec![u32::MAX; layer]];
    let mut xe = vec![u32::MAX; layer];
    let mut ids = [0u32; 12];

    let vertex = |mesh: &mut Mesh, slot: &mut u32, a: (usize, usize, usize), axis: usize| -> u32 {
        if *slot == u32::MAX {
            let (x, y, z) = a;
            let b = match axis {
                0 => (x + 1, y, z),
                1 => (x, y + 1, z),
                _ => (x, y, z + 1),
            };
            let (va, vb) = (vol[idx(x, y, z)] as f64, vol[idx(b.0, b.1, b.2)] as f64);
            let t = va / (va - vb);
            let mut p = pos(x, y, z);
            p[axis] += t * step;
            *slot = mesh.verts.len() as u32;
            mesh.verts.push(p.map(|c| c as f32));
        }
        *slot
    };

    for x in 0..nx - 1 {
        for y in 0..ny - 1 {
            for z in 0..nz - 1 {
                let mut case = 0;
                for c in 0..8 {
                    if vol[idx(x + (c & 1), y + (c >> 1 & 1), z + (c >> 2))] < 0.0 {
                        case |= 1 << c;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                let loops = &table[case];
                for l in loops {
                    for &e in &l.edges {
                        let e = e as usize;
                        let (k, axis) = (e % 4, e / 4);
                        let (a, b) = (k & 1, k >> 1);
                        ids[e] = match axis {
                            0 => vertex(&mut mesh, &mut xe[(y + a) * nz + z + b], (x, y + a, z + b), 0),
                            1 => vertex(&mut mesh, &mut ye[a][y * nz + z + b], (x + a, y, z + b), 1),
                            _ => vertex(&mut mesh, &mut ze[a][(y + b) * nz + z], (x + a, y + b, z), 2),
                        };
                    }
                    let v: Vec<u32> = l.edges.iter().map(|&e| ids[e as usize]).collect();
                    let n = v.len();
                    match l.fan_start {
                        Some(s) => {
                            for i in 1..n - 1 {
                                mesh.tris.push([v[s], v[(s + i) % n], v[(s + i + 1) % n]]);
                            }
                        }
                        None => {
                            let mut c = [0f32; 3];
                            for &i in &v {
                                for (ck, vk) in c.iter_mut().zip(mesh.verts[i as usize]) {
                                    *ck += vk / n as f32;
                                }
                            }
                            let ci = mesh.verts.len() as u32;
                            mesh.verts.push(c);
                            for i in 0..n {
                                mesh.tris.push([ci, v[i], v[(i + 1) % n]]);
                            }
                        }
                    }
                }
            }
        }
        // slide the window: layer x+1 becomes layer x
        ye.swap(0, 1);
        ze.swap(0, 1);
        ye[1].fill(u32::MAX);
        ze[1].fill(u32::MAX);
        xe.fill(u32::MAX);
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_case_is_closed() {
        // each crossing edge appears in exactly one loop, and loops cover all crossings
        for (case, loops) in table().iter().enumerate() {
            let mut count = [0; 12];
            for l in loops {
                for &e in &l.edges {
                    count[e as usize] += 1;
                }
            }
            for (e, &n) in count.iter().enumerate() {
                let (a, b) = edge_corners(e);
                let crosses = (case >> a & 1) != (case >> b & 1);
                assert_eq!(n, crosses as i32, "case {case} edge {e}");
            }
        }
    }

    #[test]
    fn random_fields_are_watertight() {
        // a noisy field with lots of ambiguous configurations
        let n = 24;
        let mut vol = vec![0f32; n * n * n];
        let mut seed = 12345u32;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let r = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
                    let border = x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1;
                    vol[(x * n + y) * n + z] = if border {
                        1.0
                    } else if r.abs() < 0.01 {
                        0.01
                    } else {
                        r
                    };
                }
            }
        }
        let m = polygonize(&vol, [n, n, n], [0.0; 3], 1.0);
        assert!(m.tris.len() > 1000);
        assert!(m.is_watertight());
        assert!(m.volume() > 0.0);
    }

    #[test]
    fn sphere_volume() {
        let n = 40;
        let mut vol = vec![0f32; n * n * n];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let p = [x, y, z].map(|c| c as f32 - 19.5);
                    vol[(x * n + y) * n + z] = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 15.0;
                }
            }
        }
        let m = polygonize(&vol, [n, n, n], [0.0; 3], 1.0);
        assert!(m.is_watertight());
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 15f64.powi(3);
        assert!((m.volume() / exact - 1.0).abs() < 0.01, "{} vs {exact}", m.volume());
        assert_eq!(m.body_count(), 1);
    }
}
