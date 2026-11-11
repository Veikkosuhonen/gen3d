//! VTK ImageData (.vti) files: export of the sampled SDF grid (for simulation and topology
//! optimization), and a reader for viewing grids written by other tools.
//!
//! The writer writes the point-data array `sdf` (Float32, mm, negative inside) plus one array per
//! `fn field_<name>` in the shape (Float32, 1 or 3 components) on a regular grid covering the shape's
//! bounds, as XML VTK with raw appended binary data. Opens in ParaView and reads with
//! `pyvista.read(path)` / `vtk.vtkXMLImageDataReader`. Values are stored x-fastest (VTK order), so
//! `mesh["sdf"].reshape(nz, ny, nx)` gives a `[z, y, x]` NumPy array.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;

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

/// One array for `write_grid`: `comps` floats per point (or per cell), x-fastest.
pub struct GridArray<'a> {
    pub name: &'a str,
    pub cell: bool,
    pub comps: usize,
    pub data: &'a [f32],
}

/// Write a regular grid with point and cell arrays. `points` is the number of grid points per axis
/// (cells = points - 1), `origin` the first point. Cell arrays are written first, so a viewer that
/// defaults to the first array shows them. The file is written to a temporary name and renamed into
/// place, so a viewer reloading it never sees a half-written file. Returns the file size.
pub fn write_grid(
    path: &Path,
    origin: [f64; 3],
    spacing: f64,
    points: [usize; 3],
    arrays: &[GridArray],
) -> Result<u64> {
    let [px, py, pz] = points;
    let extent = format!("0 {} 0 {} 0 {}", px - 1, py - 1, pz - 1);
    let mut xml = [String::new(), String::new()]; // cell, point
    let mut offset = 0usize;
    let mut order = vec![];
    for cell in [true, false] {
        for a in arrays.iter().filter(|a| a.cell == cell) {
            let n = if cell {
                (px - 1) * (py - 1) * (pz - 1)
            } else {
                px * py * pz
            };
            anyhow::ensure!(a.data.len() == n * a.comps, "{}: wrong length", a.name);
            let _ = writeln!(
                xml[!cell as usize],
                r#"        <DataArray type="Float32" Name="{}" NumberOfComponents="{}" format="appended" offset="{offset}"/>"#,
                a.name, a.comps
            );
            offset += 8 + a.data.len() * 4;
            order.push(a);
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("vti.part");
    {
        let file = File::create(&tmp).with_context(|| format!("cannot write {}", tmp.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        write!(
            w,
            r#"<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="{extent}" Origin="{} {} {}" Spacing="{spacing} {spacing} {spacing}" Direction="1 0 0 0 1 0 0 0 1">
    <Piece Extent="{extent}">
      <CellData>
{}      </CellData>
      <PointData>
{}      </PointData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
   _"#,
            origin[0], origin[1], origin[2], xml[0], xml[1]
        )?;
        for a in order {
            w.write_all(&(a.data.len() as u64 * 4).to_le_bytes())?;
            for v in a.data {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.write_all(b"\n  </AppendedData>\n</VTKFile>\n")?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
}

// ------------------------------------------------------------------------------------------ reader

/// A VTK ImageData file: a regular grid plus named arrays. Arrays are converted to f32 and stored
/// x-fastest (VTK order).
pub struct VtiFile {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    /// WholeExtent: index range of the grid points, `[x0, x1, y0, y1, z0, z1]`
    pub extent: [i64; 6],
    pub arrays: Vec<VtiArray>,
}

pub struct VtiArray {
    pub name: String,
    /// cell data (one value per voxel) rather than point data (one per grid point)
    pub cell: bool,
    pub comps: usize,
    pub data: Vec<f32>,
}

impl VtiFile {
    /// Position of the first sample and the sample counts of point or cell data.
    pub fn lattice(&self, cell: bool) -> ([f64; 3], [usize; 3]) {
        let e = self.extent;
        let mut origin = [0.0; 3];
        let mut dims = [0; 3];
        for k in 0..3 {
            let (a, b) = (e[2 * k], e[2 * k + 1]);
            // cells sit between points; on a flat axis (2D grids) they sit on the points
            let half = if cell && b > a { 0.5 } else { 0.0 };
            origin[k] = self.origin[k] + (a as f64 + half) * self.spacing[k];
            dims[k] = ((b - a) as usize + if cell { 0 } else { 1 }).max(1);
        }
        (origin, dims)
    }

    /// One line per array, for `gen3d check`.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        for a in &self.arrays {
            let (min, max) = range(&a.data);
            let kind = if a.cell { "cell" } else { "point" };
            let comps = if a.comps == 1 {
                String::new()
            } else {
                format!("[{}]", a.comps)
            };
            let _ = writeln!(out, "  {}{comps}: {kind} data, range {min} .. {max}", a.name);
        }
        out
    }
}

fn range(v: &[f32]) -> (f32, f32) {
    v.iter()
        .filter(|x| x.is_finite())
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)))
}

struct Tag<'a> {
    name: &'a str,
    close: bool,
    self_closing: bool,
    attrs: Vec<(&'a str, &'a str)>,
    /// byte index just past the closing `>`
    end: usize,
}

impl Tag<'_> {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
}

fn parse_attrs(mut s: &str) -> Vec<(&str, &str)> {
    let mut out = vec![];
    while let Some(eq) = s.find('=') {
        let key = s[..eq].trim();
        let rest = s[eq + 1..].trim_start();
        let Some(q) = rest.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let Some(close) = rest[1..].find(q) else { break };
        out.push((key, &rest[1..1 + close]));
        s = &rest[close + 2..];
    }
    out
}

fn scan_tags(xml: &str) -> Vec<Tag<'_>> {
    let mut tags = vec![];
    let mut i = 0;
    while let Some(lt) = xml[i..].find('<') {
        let s = i + lt;
        let skip_to = if xml[s..].starts_with("<?") {
            Some("?>")
        } else if xml[s..].starts_with("<!--") {
            Some("-->")
        } else {
            None
        };
        if let Some(end) = skip_to {
            match xml[s..].find(end) {
                Some(e) => {
                    i = s + e + end.len();
                    continue;
                }
                None => break,
            }
        }
        let Some(gt) = xml[s..].find('>') else { break };
        let inner = &xml[s + 1..s + gt];
        let close = inner.starts_with('/');
        let self_closing = inner.ends_with('/');
        let inner = inner.trim_start_matches('/').trim_end_matches('/');
        let name_end = inner.find(char::is_whitespace).unwrap_or(inner.len());
        tags.push(Tag {
            name: &inner[..name_end],
            close,
            self_closing,
            attrs: parse_attrs(&inner[name_end..]),
            end: s + gt + 1,
        });
        i = s + gt + 1;
    }
    tags
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn nums<const N: usize, T: std::str::FromStr>(s: Option<&str>, what: &str) -> Result<[T; N]> {
    let s = s.ok_or_else(|| anyhow!("missing {what}"))?;
    let v: Vec<T> = s
        .split_whitespace()
        .map(|x| x.parse().map_err(|_| anyhow!("bad {what}: {s}")))
        .collect::<Result<_>>()?;
    v.try_into().map_err(|_| anyhow!("bad {what}: {s}"))
}

#[derive(Clone, Copy)]
struct Binary {
    header_bytes: usize,
    zlib: bool,
}

impl Binary {
    fn uint(&self, b: &[u8], i: usize) -> Result<usize> {
        let h = self.header_bytes;
        let s = b
            .get(i * h..(i + 1) * h)
            .ok_or_else(|| anyhow!("truncated data block header"))?;
        Ok(if h == 8 {
            u64::from_le_bytes(s.try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(s.try_into().unwrap()) as usize
        })
    }

    fn inflate(&self, blocks: &[u8], sizes: &[usize]) -> Result<Vec<u8>> {
        use std::io::Read;
        let mut out = vec![];
        let mut pos = 0;
        for &n in sizes {
            let block = blocks
                .get(pos..pos + n)
                .ok_or_else(|| anyhow!("truncated compressed data"))?;
            flate2::read::ZlibDecoder::new(block)
                .read_to_end(&mut out)
                .context("zlib")?;
            pos += n;
        }
        Ok(out)
    }

    /// A data block in raw bytes (appended raw encoding).
    fn raw(&self, b: &[u8]) -> Result<Vec<u8>> {
        if !self.zlib {
            let n = self.uint(b, 0)?;
            let h = self.header_bytes;
            return Ok(b.get(h..h + n).ok_or_else(|| anyhow!("truncated data"))?.to_vec());
        }
        let nb = self.uint(b, 0)?;
        let sizes: Vec<usize> = (0..nb).map(|i| self.uint(b, 3 + i)).collect::<Result<_>>()?;
        self.inflate(&b[(3 + nb) * self.header_bytes..], &sizes)
    }

    /// A data block in base64 (inline `binary` or appended base64). Uncompressed blocks encode
    /// header and data together; compressed blocks encode the header separately.
    fn base64(&self, text: &[u8]) -> Result<Vec<u8>> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let dec =
            |t: &[u8]| -> Result<Vec<u8>> { b64.decode(t.get(..).unwrap_or_default()).context("bad base64 data") };
        let chars = |bytes: usize| 4 * bytes.div_ceil(3);
        let get = |n: usize| text.get(..n).ok_or_else(|| anyhow!("truncated base64 data"));
        let h = self.header_bytes;
        if !self.zlib {
            let n = self.uint(&dec(get(chars(h).next_multiple_of(4))?)?, 0)?;
            let all = dec(get(chars(h + n))?)?;
            return Ok(all.get(h..h + n).ok_or_else(|| anyhow!("truncated data"))?.to_vec());
        }
        let nb = self.uint(&dec(get(chars(3 * h))?)?, 0)?;
        let hlen = chars((3 + nb) * h);
        let header = dec(get(hlen)?)?;
        let sizes: Vec<usize> = (0..nb).map(|i| self.uint(&header, 3 + i)).collect::<Result<_>>()?;
        let total: usize = sizes.iter().sum();
        let rest = text.get(hlen..).unwrap_or_default();
        let blocks = dec(rest
            .get(..chars(total))
            .ok_or_else(|| anyhow!("truncated base64 data"))?)?;
        self.inflate(&blocks, &sizes)
    }
}

fn to_f32(ty: &str, b: &[u8]) -> Result<Vec<f32>> {
    macro_rules! conv {
        ($t:ty) => {
            b.chunks_exact(size_of::<$t>())
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        };
    }
    Ok(match ty {
        "Float32" => conv!(f32),
        "Float64" => conv!(f64),
        "Int8" => conv!(i8),
        "UInt8" => conv!(u8),
        "Int16" => conv!(i16),
        "UInt16" => conv!(u16),
        "Int32" => conv!(i32),
        "UInt32" => conv!(u32),
        "Int64" => conv!(i64),
        "UInt64" => conv!(u64),
        _ => bail!("unsupported data type {ty}"),
    })
}

/// Read a .vti file: raw, base64 or ascii data, optionally zlib-compressed (what gen3d, pyvista,
/// ParaView and VTK's own writers produce). Only single-piece, axis-aligned, little-endian files.
pub fn read(path: &Path) -> Result<VtiFile> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    read_bytes(&bytes).with_context(|| format!("{}: not a readable VTK ImageData file", path.display()))
}

fn read_bytes(bytes: &[u8]) -> Result<VtiFile> {
    let app = find(bytes, b"<AppendedData");
    let head = std::str::from_utf8(&bytes[..app.unwrap_or(bytes.len())]).context("header is not UTF-8")?;
    let (app_raw, app_start) = match app {
        Some(a) => {
            let gt = a + find(&bytes[a..], b">").ok_or_else(|| anyhow!("bad AppendedData"))?;
            let tag = std::str::from_utf8(&bytes[a..gt]).context("bad AppendedData")?;
            let raw = parse_attrs(tag.trim_start_matches("<AppendedData")).contains(&("encoding", "raw"));
            let us = gt + find(&bytes[gt..], b"_").ok_or_else(|| anyhow!("AppendedData without `_`"))?;
            (raw, us + 1)
        }
        None => (false, 0),
    };

    let tags = scan_tags(head);
    let root = tags
        .iter()
        .find(|t| t.name == "VTKFile")
        .ok_or_else(|| anyhow!("no <VTKFile>"))?;
    if root.attr("type") != Some("ImageData") {
        bail!("type is {:?}, not ImageData", root.attr("type").unwrap_or("?"));
    }
    if root.attr("byte_order") == Some("BigEndian") {
        bail!("big-endian files are not supported");
    }
    let bin = Binary {
        header_bytes: if root.attr("header_type") == Some("UInt64") {
            8
        } else {
            4
        },
        zlib: match root.attr("compressor") {
            None | Some("") => false,
            Some("vtkZLibDataCompressor") => true,
            Some(c) => bail!("unsupported compressor {c} (use zlib or none)"),
        },
    };
    let img = tags
        .iter()
        .find(|t| t.name == "ImageData" && !t.close)
        .ok_or_else(|| anyhow!("no <ImageData>"))?;
    let extent: [i64; 6] = nums(img.attr("WholeExtent"), "WholeExtent")?;
    let origin: [f64; 3] = nums(img.attr("Origin").or(Some("0 0 0")), "Origin")?;
    let spacing: [f64; 3] = nums(img.attr("Spacing").or(Some("1 1 1")), "Spacing")?;
    if let Some(d) = img.attr("Direction") {
        let d: [f64; 9] = nums(Some(d), "Direction")?;
        if d != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
            eprintln!("warning: ignoring the grid's rotation (Direction), showing it axis-aligned");
        }
    }
    if tags.iter().filter(|t| t.name == "Piece" && !t.close).count() > 1 {
        bail!("multi-piece files are not supported");
    }
    let mut file = VtiFile {
        origin,
        spacing,
        extent,
        arrays: vec![],
    };
    let count = |cell| {
        let (_, d) = file.lattice(cell);
        d[0] * d[1] * d[2]
    };
    let (npoints, ncells) = (count(false), count(true));

    let mut section = None;
    for t in &tags {
        match (t.name, t.close) {
            ("PointData" | "CellData", false) if !t.self_closing => section = Some(t.name == "CellData"),
            ("PointData" | "CellData", true) => section = None,
            ("DataArray", false) => {
                let Some(cell) = section else { continue };
                let name = t.attr("Name").unwrap_or("unnamed").to_string();
                let ty = t.attr("type").ok_or_else(|| anyhow!("{name}: no type"))?;
                let comps: usize = t.attr("NumberOfComponents").unwrap_or("1").parse()?;
                let content = || {
                    let rest = &head[t.end..];
                    // data runs up to the next tag (`</DataArray>`, or the `<InformationKey>`
                    // elements VTK adds to vector arrays)
                    let end = rest.find('<').unwrap_or(0);
                    &rest[..end]
                };
                let data = match t.attr("format").unwrap_or("ascii") {
                    "ascii" => content()
                        .split_whitespace()
                        .map(|x| x.parse::<f64>().map(|v| v as f32))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| anyhow!("{name}: bad ascii data"))?,
                    "binary" => {
                        let text: Vec<u8> = content().bytes().filter(|b| !b.is_ascii_whitespace()).collect();
                        to_f32(ty, &bin.base64(&text)?)?
                    }
                    "appended" => {
                        let off: usize = t.attr("offset").unwrap_or("0").trim().parse()?;
                        let src = bytes
                            .get(app_start + off..)
                            .ok_or_else(|| anyhow!("{name}: bad offset"))?;
                        let raw = if app_raw { bin.raw(src)? } else { bin.base64(src)? };
                        to_f32(ty, &raw)?
                    }
                    f => bail!("{name}: unknown format {f}"),
                };
                let n = if cell { ncells } else { npoints };
                if data.len() != n * comps {
                    bail!("{name}: expected {} values, found {}", n * comps, data.len());
                }
                file.arrays.push(VtiArray {
                    name,
                    cell,
                    comps,
                    data,
                });
            }
            _ => {}
        }
    }
    if file.arrays.is_empty() {
        bail!("no point or cell data arrays");
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_inline_and_compressed() {
        // 2x1x1 points, 1 cell; ascii point data and zlib+base64 cell data with a UInt32 header
        let cell: Vec<u8> = 7.5f32.to_le_bytes().to_vec();
        let mut z = flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
        std::io::Write::write_all(&mut z, &cell).unwrap();
        let comp = z.finish().unwrap();
        let header: Vec<u8> = [1u32, 4, 4, comp.len() as u32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b64 = base64::engine::general_purpose::STANDARD;
        let data = format!("{}{}", b64.encode(&header), b64.encode(&comp));
        let xml = format!(
            r#"<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
<ImageData WholeExtent="0 1 0 0 0 0" Origin="1 2 3" Spacing="0.5 0.5 0.5">
<Piece Extent="0 1 0 0 0 0">
<PointData><DataArray type="Float64" Name="d" format="ascii">1 2.5</DataArray></PointData>
<CellData><DataArray type="Float32" Name="c" format="binary">{data}</DataArray></CellData>
</Piece></ImageData></VTKFile>"#
        );
        let f = read_bytes(xml.as_bytes()).unwrap();
        assert_eq!(f.arrays.len(), 2);
        assert_eq!(f.arrays[0].data, [1.0, 2.5]);
        assert!(f.arrays[1].cell);
        assert_eq!(f.arrays[1].data, [7.5]);
        assert_eq!(f.lattice(true).0, [1.25, 2.0, 3.0]);
    }

    #[test]
    fn write_grid_round_trips() {
        let dir = std::env::temp_dir().join(format!("gen3d-wg-{}", std::process::id()));
        let p = dir.join("g.vti");
        let cells: Vec<f32> = (0..2).map(|i| i as f32 + 0.5).collect();
        let pts: Vec<f32> = (0..12 * 3).map(|i| i as f32).collect();
        let arrays = [
            GridArray {
                name: "u",
                cell: false,
                comps: 3,
                data: &pts,
            },
            GridArray {
                name: "rho",
                cell: true,
                comps: 1,
                data: &cells,
            },
        ];
        write_grid(&p, [1.0, 2.0, 3.0], 0.5, [3, 2, 2], &arrays).unwrap();
        let f = read(&p).unwrap();
        assert_eq!(f.arrays[0].name, "rho"); // cell data first
        assert_eq!(f.arrays[0].data, cells);
        assert_eq!(f.arrays[1].data, pts);
        assert_eq!(f.lattice(true).0, [1.25, 2.25, 3.25]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn round_trips_own_export() {
        let grid = Grid {
            vol: (0..27).map(|i| i as f32).collect(),
            dims: [3, 3, 3],
            origin: [0.0; 3],
            step: 1.0,
        };
        let dir = std::env::temp_dir().join(format!("gen3d-vti-{}", std::process::id()));
        let p = dir.join("t.vti");
        let vec3: Vec<f32> = (0..81).map(|i| i as f32).collect();
        let arrays = [
            Array {
                name: "sdf",
                comps: 1,
                data: &grid.vol,
            },
            Array {
                name: "v",
                comps: 3,
                data: &vec3,
            },
        ];
        save(&grid, &arrays, &p).unwrap();
        let f = read(&p).unwrap();
        // the padded 3x3x3 grid crops to its single centre point
        assert_eq!(f.arrays[0].data, [13.0]);
        assert_eq!(f.arrays[1].data, [39.0, 40.0, 41.0]);
        assert_eq!(f.lattice(false), ([1.0; 3], [1, 1, 1]));
        std::fs::remove_dir_all(&dir).ok();
    }
}
