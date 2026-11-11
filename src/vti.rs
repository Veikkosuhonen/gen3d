//! VTK ImageData (.vti) export of the sampled SDF grid, for simulation and topology optimization.
//!
//! Writes the point-data array `sdf` (Float32, mm, negative inside) plus one array per
//! `fn field_<name>` in the shape (Float32, 1 or 3 components) on a regular grid covering the shape's
//! bounds, as XML VTK with raw appended binary data. Opens in ParaView and reads with
//! `pyvista.read(path)` / `vtk.vtkXMLImageDataReader`. Values are stored x-fastest (VTK order), so
//! `mesh["sdf"].reshape(nz, ny, nx)` gives a `[z, y, x]` NumPy array.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::mesher::Grid;

/// One point-data array: `comps` floats per point, in `Grid::vol` order (z fastest, padded grid).
pub struct Array<'a> {
    pub name: &'a str,
    pub comps: usize,
    pub data: &'a [f32],
}

pub fn save(grid: &Grid, arrays: &[Array], out: &Path) -> Result<()> {
    // `sample_grid` pads by one voxel on each side and forces that shell to "outside" for marching
    // cubes; drop it so the field is the true SDF and the grid starts exactly at the bounds' minimum.
    let [nx, ny, nz] = grid.dims;
    let [cx, cy, cz] = [nx - 2, ny - 2, nz - 2];
    let origin = grid.origin.map(|o| (o + grid.step) as f32);
    let s = grid.step as f32;

    let mut blobs = vec![];
    let mut xml = String::new();
    let mut offset = 0usize;
    for a in arrays {
        let c = a.comps;
        let mut data = vec![0f32; cx * cy * cz * c];
        for x in 0..cx {
            for y in 0..cy {
                let src = (((x + 1) * ny + y + 1) * nz + 1) * c;
                for (z, v) in a.data[src..src + cz * c].chunks_exact(c).enumerate() {
                    let dst = ((z * cy + y) * cx + x) * c;
                    data[dst..dst + c].copy_from_slice(v);
                }
            }
        }
        let nonfinite = data.iter().filter(|v| !v.is_finite()).count();
        if nonfinite > 0 {
            eprintln!(
                "warning: `{}` has {nonfinite} NaN/Inf values; writing them as 0",
                a.name
            );
            data.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = 0.0);
        }
        let range = if c == 1 {
            let (min, max) = data
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            format!(r#" RangeMin="{min}" RangeMax="{max}""#)
        } else {
            String::new()
        };
        let _ = writeln!(
            xml,
            r#"        <DataArray type="Float32" Name="{}" NumberOfComponents="{c}" format="appended" offset="{offset}"{range}/>"#,
            a.name
        );
        offset += 8 + data.len() * 4;
        blobs.push(data);
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = File::create(out).with_context(|| format!("cannot write {}", out.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let extent = format!("0 {} 0 {} 0 {}", cx - 1, cy - 1, cz - 1);
    let scalars = arrays.first().map_or("", |a| a.name);
    write!(
        w,
        r#"<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="{extent}" Origin="{} {} {}" Spacing="{s} {s} {s}" Direction="1 0 0 0 1 0 0 0 1">
    <Piece Extent="{extent}">
      <PointData Scalars="{scalars}">
{xml}      </PointData>
      <CellData/>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
   _"#,
        origin[0], origin[1], origin[2]
    )?;
    for data in &blobs {
        w.write_all(&(data.len() as u64 * 4).to_le_bytes())?;
        if cfg!(target_endian = "little") {
            w.write_all(bytemuck::cast_slice(data))?;
        } else {
            for v in data {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.write_all(b"\n  </AppendedData>\n</VTKFile>\n")?;
    w.flush()?;

    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    let names: Vec<_> = arrays
        .iter()
        .map(|a| {
            if a.comps == 1 {
                a.name.to_string()
            } else {
                format!("{}[{}]", a.name, a.comps)
            }
        })
        .collect();
    println!(
        "wrote {} ({cx}x{cy}x{cz} points, origin {:.3} {:.3} {:.3}, spacing {s} mm, arrays: {}, {:.1} MB)",
        out.display(),
        origin[0],
        origin[1],
        origin[2],
        names.join(", "),
        size as f64 / 1e6
    );
    Ok(())
}
