//! Indexed triangle mesh: analysis (volume, watertightness, bodies), decimation and file export.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Applied to each vertex after a smoothing pass (`Mesh::smooth`).
pub type VertexConstraint = dyn Fn(&mut [f32; 3]) + Sync;

#[derive(Default, Clone)]
pub struct Mesh {
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
}

impl Mesh {
    /// Signed volume (positive when faces point outward).
    pub fn volume(&self) -> f64 {
        self.tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.verts[i as usize].map(f64::from));
                a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
            })
            .sum::<f64>()
            / 6.0
    }

    pub fn invert(&mut self) {
        for t in &mut self.tris {
            t.swap(1, 2);
        }
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in &self.verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        (lo, hi)
    }

    /// Every edge is used by exactly two faces, once in each direction.
    pub fn is_watertight(&self) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let mut edges: Vec<u64> = Vec::with_capacity(self.tris.len() * 3);
        for t in &self.tris {
            for k in 0..3 {
                let (a, b) = (t[k] as u64, t[(k + 1) % 3] as u64);
                edges.push(a << 32 | b);
            }
        }
        edges.sort_unstable();
        if edges.windows(2).any(|w| w[0] == w[1]) {
            return false;
        }
        edges
            .iter()
            .all(|&e| edges.binary_search(&((e & 0xffff_ffff) << 32 | e >> 32)).is_ok())
    }

    /// Edges that are open (one face) or non-manifold (more than two faces), as (a, b, face count).
    pub fn defective_edges(&self) -> Vec<(u32, u32, usize)> {
        let mut edges: Vec<u64> = Vec::with_capacity(self.tris.len() * 3);
        for t in &self.tris {
            for k in 0..3 {
                let (a, b) = (t[k] as u64, t[(k + 1) % 3] as u64);
                edges.push(a.min(b) << 32 | a.max(b));
            }
        }
        edges.sort_unstable();
        edges
            .chunk_by(|a, b| a == b)
            .filter(|run| run.len() != 2)
            .map(|run| ((run[0] >> 32) as u32, run[0] as u32, run.len()))
            .collect()
    }

    /// Number of connected components.
    pub fn body_count(&self) -> usize {
        let mut parent: Vec<u32> = (0..self.verts.len() as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        let mut used = vec![false; self.verts.len()];
        for t in &self.tris {
            for &i in t {
                used[i as usize] = true;
            }
            let r = find(&mut parent, t[0]);
            for &i in &t[1..] {
                let ri = find(&mut parent, i);
                parent[ri as usize] = r;
            }
        }
        (0..self.verts.len() as u32)
            .filter(|&i| used[i as usize] && find(&mut parent, i) == i)
            .count()
    }

    pub fn remove_unreferenced_vertices(&mut self) {
        let mut remap = vec![u32::MAX; self.verts.len()];
        let mut verts = Vec::with_capacity(self.verts.len());
        for t in &mut self.tris {
            for i in t.iter_mut() {
                if remap[*i as usize] == u32::MAX {
                    remap[*i as usize] = verts.len() as u32;
                    verts.push(self.verts[*i as usize]);
                }
                *i = remap[*i as usize];
            }
        }
        self.verts = verts;
    }

    /// Quadric-error simplification (meshoptimizer) down to `fraction` of the triangles.
    ///
    /// Edge collapses occasionally pinch thin regions into non-manifold edges. When that happens the
    /// vertices around the defect are locked and the simplification is redone, so the result stays
    /// watertight whenever the input was. Returns the mesh and its remaining defective edges.
    pub fn decimated(&self, fraction: f32) -> (Mesh, Vec<(u32, u32, usize)>) {
        let indices: &[u32] = bytemuck::cast_slice(&self.tris);
        let adapter = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&self.verts), 12, 0).expect("vertex layout");
        let target = ((indices.len() as f32 * fraction) as usize / 3) * 3;
        let mut lock = vec![false; self.verts.len()];
        let mut attempt = 0;
        loop {
            let out = meshopt::simplify_with_locks(
                indices,
                &adapter,
                &lock,
                target,
                1.0,
                meshopt::SimplifyOptions::None,
                None,
            );
            let mut m = Mesh {
                verts: self.verts.clone(),
                tris: bytemuck::cast_slice(&out).to_vec(),
            };
            m.remove_folds();
            let defects = m.defective_edges();
            attempt += 1;
            if defects.is_empty() || attempt == 8 {
                m.remove_unreferenced_vertices();
                return (m, defects);
            }
            // lock the whole neighbourhood of each defect, so nothing collapses into it again
            let mut bad = vec![false; self.verts.len()];
            for (a, b, _) in defects {
                bad[a as usize] = true;
                bad[b as usize] = true;
            }
            for t in m.tris.iter().chain(&self.tris) {
                if t.iter().any(|&i| bad[i as usize]) {
                    for &i in t {
                        lock[i as usize] = true;
                    }
                }
            }
        }
    }

    /// Taubin λ|μ smoothing (Taubin 1995): `passes` pairs of a shrinking Laplacian step (λ) and an
    /// inflating one (μ < -λ). Together they remove the high-frequency ripple of marching cubes on a
    /// voxel grid while keeping the shape's size, unlike plain Laplacian smoothing, which shrinks
    /// it. Uniform weights, so the steps also even out the triangles' sizes along the surface.
    ///
    /// `constrain` is applied to every vertex after each pass, e.g. to keep it inside a region: the
    /// inflating step overshoots at convex rims, pushing flat faces next to them outward by a
    /// fraction of an edge. Only vertices move: the connectivity, and with it watertightness, is
    /// unchanged.
    pub fn smooth(&mut self, passes: u32, constrain: Option<&VertexConstraint>) {
        use rayon::prelude::*;
        const LAMBDA: f32 = 0.5;
        const MU: f32 = -0.53;
        if passes == 0 || self.tris.is_empty() {
            return;
        }
        let nv = self.verts.len();
        // neighbours of each vertex (CSR), from the edges of the triangles
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.tris.len() * 6);
        for t in &self.tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edges.push((a, b));
                edges.push((b, a));
            }
        }
        edges.par_sort_unstable();
        edges.dedup();
        let mut start = vec![0usize; nv + 1];
        for &(a, _) in &edges {
            start[a as usize + 1] += 1;
        }
        for i in 0..nv {
            start[i + 1] += start[i];
        }
        let nbrs: Vec<u32> = edges.iter().map(|e| e.1).collect();
        drop(edges);
        let mut next = self.verts.clone();
        for _ in 0..passes {
            for f in [LAMBDA, MU] {
                let cur = &self.verts;
                next.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let ns = &nbrs[start[i]..start[i + 1]];
                    let v = cur[i];
                    if ns.is_empty() {
                        *out = v;
                        return;
                    }
                    let mut c = [0f32; 3];
                    for &n in ns {
                        let q = cur[n as usize];
                        for k in 0..3 {
                            c[k] += q[k];
                        }
                    }
                    let inv = 1.0 / ns.len() as f32;
                    for k in 0..3 {
                        out[k] = v[k] + f * (c[k] * inv - v[k]);
                    }
                });
                std::mem::swap(&mut self.verts, &mut next);
            }
            if let Some(c) = constrain {
                self.verts.par_iter_mut().for_each(c);
            }
        }
    }

    /// Edge collapses can squash a thin region into two back-to-back triangles on the same three
    /// vertices, leaving non-manifold edges around them. Deleting both triangles of each such pair
    /// removes the zero-volume fold and usually restores a closed 2-manifold.
    fn remove_folds(&mut self) {
        let key = |t: &[u32; 3]| {
            let mut k = *t;
            k.sort_unstable();
            k
        };
        let mut order: Vec<usize> = (0..self.tris.len()).collect();
        order.sort_unstable_by_key(|&i| key(&self.tris[i]));
        let mut drop = vec![false; self.tris.len()];
        for run in order.chunk_by(|&a, &b| key(&self.tris[a]) == key(&self.tris[b])) {
            if run.len() >= 2 {
                for &i in run {
                    drop[i] = true;
                }
            }
        }
        let mut i = 0;
        self.tris.retain(|_| {
            i += 1;
            !drop[i - 1]
        });
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let file = File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        match ext.as_str() {
            "stl" => self.write_stl(&mut w)?,
            "obj" => self.write_obj(&mut w)?,
            "ply" => self.write_ply(&mut w)?,
            "3mf" => {
                drop(w);
                self.write_3mf(path)?;
                return Ok(());
            }
            _ => bail!("unsupported mesh format: .{ext} (use stl, 3mf, obj or ply)"),
        }
        w.flush()?;
        Ok(())
    }

    fn write_stl(&self, w: &mut impl Write) -> Result<()> {
        let mut header = [0u8; 80];
        header[..20].copy_from_slice(b"gen3d binary STL    ");
        w.write_all(&header)?;
        w.write_all(&(self.tris.len() as u32).to_le_bytes())?;
        for t in &self.tris {
            let [a, b, c] = t.map(|i| self.verts[i as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let mut n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                n = n.map(|x| x / len);
            }
            for p in [n, a, b, c] {
                for x in p {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            w.write_all(&[0, 0])?;
        }
        Ok(())
    }

    fn write_obj(&self, w: &mut impl Write) -> Result<()> {
        writeln!(w, "# gen3d")?;
        for v in &self.verts {
            writeln!(w, "v {} {} {}", v[0], v[1], v[2])?;
        }
        for t in &self.tris {
            writeln!(w, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)?;
        }
        Ok(())
    }

    fn write_ply(&self, w: &mut impl Write) -> Result<()> {
        write!(
            w,
            "ply\nformat binary_little_endian 1.0\ncomment gen3d\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\nelement face {}\n\
             property list uchar int vertex_indices\nend_header\n",
            self.verts.len(),
            self.tris.len()
        )?;
        w.write_all(bytemuck::cast_slice(&self.verts))?;
        for t in &self.tris {
            w.write_all(&[3])?;
            for i in t {
                w.write_all(&i.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn write_3mf(&self, path: &Path) -> Result<()> {
        use zip::write::SimpleFileOptions;
        let mut zip = zip::ZipWriter::new(BufWriter::new(File::create(path)?));
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("[Content_Types].xml", opts)?;
        zip.write_all(
            br#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#,
        )?;
        zip.start_file("_rels/.rels", opts)?;
        zip.write_all(
            br#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#,
        )?;
        zip.start_file("3D/3dmodel.model", opts)?;
        let mut w = BufWriter::with_capacity(1 << 20, &mut zip);
        writeln!(w, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(
            w,
            r#"<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">"#
        )?;
        writeln!(w, r#" <resources><object id="1" type="model"><mesh><vertices>"#)?;
        for v in &self.verts {
            writeln!(w, r#"<vertex x="{}" y="{}" z="{}"/>"#, v[0], v[1], v[2])?;
        }
        writeln!(w, "</vertices><triangles>")?;
        for t in &self.tris {
            writeln!(w, r#"<triangle v1="{}" v2="{}" v3="{}"/>"#, t[0], t[1], t[2])?;
        }
        writeln!(w, "</triangles></mesh></object></resources>")?;
        writeln!(w, r#" <build><item objectid="1"/></build>"#)?;
        writeln!(w, "</model>")?;
        w.flush()?;
        drop(w);
        zip.finish()?.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// Marching cubes of a sphere from a voxel-quantized field (each sample rounded to whole
    /// voxels, so the surface has voxel steps): smoothing removes the steps without shrinking it.
    #[test]
    fn smoothing_evens_out_steps_and_keeps_volume() {
        let n = 40;
        let c = 19.5;
        let r = 14.0f32;
        let mut vol = vec![0f32; n * n * n];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let p = [x, y, z].map(|v| v as f32 - c);
                    let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r;
                    vol[(x * n + y) * n + z] = if d < 0.0 { -0.5 } else { 0.5 };
                }
            }
        }
        let mut m = crate::mc::polygonize(&vol, [n, n, n], [0.0; 3], 1.0);
        let spread = |m: &super::Mesh| {
            let rs: Vec<f32> = m
                .verts
                .iter()
                .map(|v| v.iter().map(|q| (q - c) * (q - c)).sum::<f32>().sqrt())
                .collect();
            let mean = rs.iter().sum::<f32>() / rs.len() as f32;
            (rs.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rs.len() as f32).sqrt()
        };
        let (s0, v0) = (spread(&m), m.volume());
        m.smooth(10, None);
        let (s1, v1) = (spread(&m), m.volume());
        assert!(m.is_watertight());
        assert!(s1 < 0.5 * s0, "radius spread {s0} -> {s1}");
        assert!((v1 / v0 - 1.0).abs() < 0.01, "volume {v0} -> {v1}");
    }
}
