//! Shared machinery for the GPU optimizer: the compute kernels of `mg.wgsl` and `opt.wgsl`, small
//! "programs" (lists of dispatches with fixed bind groups, recorded once and replayed), a scalar
//! buffer for reductions and CG coefficients, and readback.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;

const SOURCES: [&str; 2] = [include_str!("mg.wgsl"), include_str!("opt.wgsl")];

/// Workgroup size of one-thread-per-item kernels (`WG` in the WGSL prelude).
pub const WG: usize = 128;
/// Workgroups of the first reduction pass.
const PARTIALS: usize = 1024;
/// Scalar slots.
pub const SLOTS: usize = 544;
/// "no scalar slot" in `lin`
pub const NONE: u32 = u32::MAX;

/// Kernel parameters (the WGSL prelude's `Params`).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub n: [u32; 4],
    pub m: [u32; 4],
    pub f: [f32; 4],
    pub g: [f32; 4],
}

/// Reductions (`dot_partial` modes).
#[derive(Clone, Copy)]
pub enum Reduce {
    /// Σ a b
    Dot,
    /// Σ a b w
    WeightedDot,
    /// max |a - b| over w > 0
    MaxDiff,
    /// count of 0.1 < a < 0.9 over w > 0
    Grey,
    /// max |a_node| of a 3-vector field (len = nodes)
    MaxNorm,
}

pub struct Step {
    pipe: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
}

/// Dispatch shape for `items` workgroups (2D past the 65535 limit; kernels flatten it).
pub fn groups(items: usize) -> [u32; 3] {
    if items <= 65535 {
        [items.max(1) as u32, 1, 1]
    } else {
        [65535, items.div_ceil(65535) as u32, 1]
    }
}

/// Dispatch shape for one thread per item.
pub fn threads(n: usize) -> [u32; 3] {
    groups(n.div_ceil(WG))
}

pub fn dims4(n: [usize; 3]) -> [u32; 4] {
    [n[0] as u32, n[1] as u32, n[2] as u32, 0]
}

/// Brick shapes (bricks.rs): kernels tagged `bricks` (`//// kernel NAME bricks`) are compiled once
/// per shape, under their name plus the suffix, with the prelude's `BX`, `BY`, `BZ` set to it (and
/// the tile sizes `HN`, `EN`, `CN` that follow from it).
/// Constant strides are measurably faster than ones passed in `Params`. (Not WGSL `override`
/// constants: naga 30 fails to specialize some kernels with them, `ExpressionAlreadyInScope`.)
pub const BRICK_SHAPES: [(&str, [usize; 3]); 4] =
    [("", [4, 4, 4]), ("@x", [2, 8, 4]), ("@y", [8, 2, 4]), ("@z", [8, 4, 2])];

/// `src` with `const NAME: u32 = ...;` set to each of `constants`.
fn with_constants(src: &str, constants: &[(&str, u32)]) -> Result<String> {
    let mut out = src.to_string();
    for (name, value) in constants {
        let decl = format!("const {name}: u32 = ");
        let at = out.find(&decl).ok_or_else(|| anyhow!("no `{decl}...;` to set"))? + decl.len();
        let end = at + out[at..].find(';').ok_or_else(|| anyhow!("unterminated `{decl}`"))?;
        out.replace_range(at..end, &format!("{value}u"));
    }
    Ok(out)
}

/// The sections of the WGSL sources: lines `//// prelude`, `//// lib NAME` and `//// kernel NAME
/// [bricks]` start one (text before the first is a comment). Returns the prelude, the libs and the
/// kernels (name, tagged `bricks`, body).
#[allow(clippy::type_complexity)]
fn sections() -> Result<(String, HashMap<String, String>, Vec<(String, bool, String)>)> {
    let mut prelude = None;
    let mut libs = HashMap::new();
    let mut kernels = vec![];
    for src in SOURCES {
        let mut head: Option<&str> = None;
        let mut body = String::new();
        let mut flush = |head: Option<&str>, body: &mut String| -> Result<()> {
            let text = std::mem::take(body);
            let Some(h) = head else { return Ok(()) };
            let mut words = h.split_whitespace();
            match (words.next(), words.next()) {
                (Some("prelude"), _) => prelude = Some(text),
                (Some("lib"), Some(name)) => {
                    libs.insert(name.to_string(), text);
                }
                (Some("kernel"), Some(name)) => {
                    let bricks = words.next() == Some("bricks");
                    kernels.push((name.to_string(), bricks, text));
                }
                _ => return Err(anyhow!("bad section header `//// {h}`")),
            }
            Ok(())
        };
        for line in src.lines() {
            if let Some(h) = line.strip_prefix("//// ") {
                flush(head, &mut body)?;
                head = Some(h);
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }
        flush(head, &mut body)?;
    }
    Ok((prelude.ok_or_else(|| anyhow!("no prelude"))?, libs, kernels))
}

/// `body` with each line `#include LIB KEY=VALUE ...` replaced by that lib, `$KEY` set to VALUE.
fn expand(body: &str, libs: &HashMap<String, String>) -> Result<String> {
    let mut out = String::new();
    for line in body.lines() {
        match line.trim().strip_prefix("#include ") {
            Some(rest) => {
                let mut words = rest.split_whitespace();
                let name = words.next().unwrap_or("");
                let mut text = libs.get(name).ok_or_else(|| anyhow!("no lib `{name}`"))?.clone();
                for kv in words {
                    let (k, v) = kv
                        .split_once('=')
                        .ok_or_else(|| anyhow!("bad include argument `{kv}`"))?;
                    text = text.replace(&format!("${k}"), v);
                }
                // libs may include others
                out.push_str(&expand(&text, libs)?);
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

pub struct Engine<'g> {
    pub gpu: &'g Gpu,
    pipes: HashMap<String, wgpu::ComputePipeline>,
    pub scalars: wgpu::Buffer,
    partials: wgpu::Buffer,
    /// bytes of storage buffers allocated through `buf`/`init`
    allocated: std::cell::Cell<u64>,
}

impl<'g> Engine<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        let dev = &gpu.device;
        let (prelude, libs, kernels) = sections()?;
        let mut pipes = HashMap::new();
        for (name, bricks, body) in kernels {
            let src = expand(&format!("{prelude}\n{body}"), &libs).map_err(|e| anyhow!("kernel {name}: {e}"))?;
            let shapes = if bricks { &BRICK_SHAPES[..] } else { &BRICK_SHAPES[..1] };
            for (suffix, b) in shapes {
                let label = format!("{name}{suffix}");
                let size = |d: usize| b.iter().map(|&v| (v + d) as u32).product::<u32>();
                let constants = [
                    ("BX", b[0] as u32),
                    ("BY", b[1] as u32),
                    ("BZ", b[2] as u32),
                    ("HN", size(2)),
                    ("EN", size(1)),
                    ("CN", b.iter().map(|&v| (v / 2 + 2) as u32).product()),
                ];
                let src = with_constants(&src, &constants).map_err(|e| anyhow!("kernel {label}: {e}"))?;
                let pipe = gpu
                    .scoped(|| {
                        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&label),
                            source: wgpu::ShaderSource::Wgsl(src.into()),
                        });
                        dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(&label),
                            layout: None,
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        })
                    })
                    .map_err(|e| anyhow!("kernel {label}: {e}"))?;
                pipes.insert(label, pipe);
            }
        }
        let mut e = Engine {
            gpu,
            pipes,
            scalars: dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scalars"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            partials: dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            allocated: Default::default(),
        };
        e.scalars = e.buf("scalars", SLOTS);
        e.partials = e.buf("partials", PARTIALS);
        Ok(e)
    }

    /// Bytes allocated so far.
    pub fn allocated(&self) -> u64 {
        self.allocated.get()
    }

    /// The largest buffer one binding can hold (bytes).
    pub fn max_binding(&self) -> u64 {
        let l = self.gpu.device.limits();
        l.max_storage_buffer_binding_size.min(l.max_buffer_size)
    }

    /// A zeroed storage buffer of `len` 32-bit values.
    pub fn buf(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.allocated.set(self.allocated.get() + (len.max(1) * 4) as u64);
        self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A storage buffer holding `data`. (Not `create_buffer_init`: that maps the buffer at creation
    /// and panics when the allocation fails, instead of reporting out of memory.)
    pub fn init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let b = self.buf(label, data.len().div_ceil(4));
        self.gpu.queue.write_buffer(&b, 0, data);
        b
    }

    pub fn write_f32(&self, b: &wgpu::Buffer, data: &[f32]) {
        self.gpu.queue.write_buffer(b, 0, bytemuck::cast_slice(data));
    }

    pub fn set_scalar(&self, slot: u32, v: f32) {
        self.gpu
            .queue
            .write_buffer(&self.scalars, slot as u64 * 4, bytemuck::bytes_of(&v));
    }

    /// Append a dispatch of `kernel` with parameters `p` and storage buffers bound from 1.
    pub fn step(&self, out: &mut Vec<Step>, kernel: &str, p: Params, bufs: &[&wgpu::Buffer], groups: [u32; 3]) {
        let dev = &self.gpu.device;
        let pipe = self.pipes.get(kernel).unwrap_or_else(|| panic!("no kernel {kernel}"));
        let uniform = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(kernel),
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            });
        }
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        out.push(Step {
            pipe: pipe.clone(),
            bind,
            groups,
        });
    }

    /// y = a y + c x (x / w where w > 0 when `w` is given); a and c optionally times a scalar slot
    #[allow(clippy::too_many_arguments)]
    pub fn lin(
        &self,
        out: &mut Vec<Step>,
        y: &wgpu::Buffer,
        x: &wgpu::Buffer,
        w: Option<&wgpu::Buffer>,
        a: (f32, u32),
        c: (f32, u32),
        len: usize,
    ) {
        self.step(
            out,
            "lin",
            Params {
                m: [len as u32, a.1, c.1, w.is_some() as u32],
                f: [a.0, c.0, 0.0, 0.0],
                ..Default::default()
            },
            &[&self.scalars, x, w.unwrap_or(x), y],
            threads(len),
        );
    }

    /// S[slot] = Σ a b (w)
    pub fn dot(
        &self,
        out: &mut Vec<Step>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        w: Option<&wgpu::Buffer>,
        len: usize,
        slot: u32,
    ) {
        let mode = if w.is_some() { Reduce::WeightedDot } else { Reduce::Dot };
        self.reduce(out, mode, a, b, w.unwrap_or(a), len, slot);
    }

    /// S[slot] = a reduction over `len` items
    #[allow(clippy::too_many_arguments)]
    pub fn reduce(
        &self,
        out: &mut Vec<Step>,
        mode: Reduce,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        w: &wgpu::Buffer,
        len: usize,
        slot: u32,
    ) {
        let max = matches!(mode, Reduce::MaxDiff | Reduce::MaxNorm) as u32;
        self.step(
            out,
            "dot_partial",
            Params {
                m: [len as u32, mode as u32, 0, 0],
                ..Default::default()
            },
            &[a, b, w, &self.partials],
            [PARTIALS as u32, 1, 1],
        );
        self.step(
            out,
            "dot_final",
            Params {
                m: [PARTIALS as u32, slot, max, 0],
                ..Default::default()
            },
            &[&self.partials, &self.scalars],
            [1, 1, 1],
        );
    }

    /// S[slot] = a reduction over the design elements (flags bit 2): `Dot` gives Σ a, `MaxDiff`
    /// max |a - b|, `Grey` the grey count.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_design(
        &self,
        out: &mut Vec<Step>,
        mode: Reduce,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        flags: &wgpu::Buffer,
        len: usize,
        slot: u32,
    ) {
        let max = matches!(mode, Reduce::MaxDiff) as u32;
        self.step(
            out,
            "reduce_design",
            Params {
                m: [len as u32, mode as u32, 0, 0],
                ..Default::default()
            },
            &[a, b, flags, &self.partials],
            [PARTIALS as u32, 1, 1],
        );
        self.step(
            out,
            "dot_final",
            Params {
                m: [PARTIALS as u32, slot, max, 0],
                ..Default::default()
            },
            &[&self.partials, &self.scalars],
            [1, 1, 1],
        );
    }

    pub fn scalar(&self, out: &mut Vec<Step>, op: u32, a: u32, b: u32, dst: u32) {
        self.step(
            out,
            "scalar",
            Params {
                m: [op, a, b, dst],
                ..Default::default()
            },
            &[&self.scalars],
            [1, 1, 1],
        );
    }

    /// Submit programs, in order, in one compute pass. Validation errors (a bad binding, say) come
    /// back as errors instead of only being logged.
    pub fn run(&self, programs: &[&[Step]]) -> Result<()> {
        self.gpu.scoped(|| self.submit(programs))
    }

    fn submit(&self, programs: &[&[Step]]) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for prog in programs {
                for s in prog.iter() {
                    pass.set_pipeline(&s.pipe);
                    pass.set_bind_group(0, &s.bind, &[]);
                    pass.dispatch_workgroups(s.groups[0], s.groups[1], s.groups[2]);
                }
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    pub fn read_f32(&self, b: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        let bytes = self.gpu.read_buffer(b, (len * 4) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn read_scalars(&self) -> Result<Vec<f32>> {
        self.read_f32(&self.scalars, SLOTS)
    }

    /// Scalar slot `slot` alone (CG's convergence checks read only the residual).
    pub fn read_scalar(&self, slot: u32) -> Result<f32> {
        Ok(self.read_f32(&self.scalars, slot as usize + 1)?[slot as usize])
    }
}
