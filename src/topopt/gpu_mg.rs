//! Multigrid-preconditioned CG on the GPU, in f32: the same method as `mg.rs`, with every vector
//! resident on the GPU.
//!
//! - Level 0 (fine) is matrix-free.
//! - Level 1 is also never stored: it is computed on the fly from the fine `ee` and a table of
//!   Pᵀ (D ke D) P per clamped-node mask and child position (`mg::masked_ptkp_table`). That costs
//!   about a fine mat-vec in arithmetic, but no memory per element.
//! - Levels ≥ 2 are assembled on the GPU as 24×24 element matrices (Galerkin, one workgroup per
//!   coarse element).
//! - The coarsest level is solved with a dense inverse, computed on the CPU once per optimizer
//!   iteration and applied on the GPU, so nothing round-trips inside a V-cycle.
//! - CG's scalars (α, β, ρ) live in a small GPU buffer; the residual norm is read back every
//!   few iterations to test convergence.
//!
//! Each phase (setup, CG start, CG iteration, energies) is recorded once as a list of dispatches
//! with fixed bind groups and replayed.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use wgpu::util::DeviceExt;

use super::Problem;
use super::mg::{self, MgSettings, Smoother};
use crate::gpu::Gpu;

const SRC: &str = include_str!("mg.wgsl");

/// Kernel workgroup size for one-thread-per-item kernels (matches `WG` in mg.wgsl).
const WG: usize = 128;
/// Workgroups of the first reduction pass.
const PARTIALS: usize = 1024;
/// CG iterations between convergence checks.
const CHECK_EVERY: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: [u32; 4],
    m: [u32; 4],
    f: [f32; 4],
}

const NONE: u32 = u32::MAX;

// scalar slots
const RR: u32 = 0;
const FF: u32 = 1;
const RZ: u32 = 2;
const RZN: u32 = 3;
const PQ: u32 = 4;
const ALPHA: u32 = 5;
const BETA: u32 = 6;
const NUM: u32 = 7;
const DEN: u32 = 8;
const NRM: u32 = 9;
const SC: u32 = 10;
/// λmax of level l at LAM + l, its inverse at INVL + l
const LAM: u32 = 16;
const INVL: u32 = 48;
const SLOTS: usize = 80;

// scalar kernel ops
const OP_DIV: u32 = 0;
const OP_COPY: u32 = 1;
const OP_RSQRT: u32 = 2;
const OP_RECIP: u32 = 3;

const KERNELS: [&str; 16] = [
    "apply_fine",
    "diag_fine",
    "apply_l1",
    "diag_l1",
    "apply_elem",
    "diag_elem",
    "assemble",
    "restrict",
    "prolong",
    "lin",
    "dot_partial",
    "dot_final",
    "scalar",
    "fill_random",
    "bottom",
    "energy",
];

struct Step {
    kernel: &'static str,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
}

fn nodes(n: [usize; 3]) -> usize {
    n.iter().map(|v| v + 1).product()
}

fn coarse(n: [usize; 3]) -> [usize; 3] {
    n.map(|v| v.div_ceil(2))
}

fn groups(items: usize) -> [u32; 3] {
    if items <= 65535 {
        [items.max(1) as u32, 1, 1]
    } else {
        [65535, items.div_ceil(65535) as u32, 1]
    }
}

fn threads(n: usize) -> [u32; 3] {
    groups(n.div_ceil(WG))
}

fn dims4(n: [usize; 3]) -> [u32; 4] {
    [n[0] as u32, n[1] as u32, n[2] as u32, 0]
}

pub(super) struct GpuMg<'g> {
    gpu: &'g Gpu,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    /// elements per axis, per level (0 = fine)
    dims: Vec<[usize; 3]>,
    smoother: Smoother,
    ee: wgpu::Buffer,
    emask: wgpu::Buffer,
    free: wgpu::Buffer,
    ke: wgpu::Buffer,
    table: wgpu::Buffer,
    scalars: wgpu::Buffer,
    partials: wgpu::Buffer,
    ce: wgpu::Buffer,
    diag: Vec<wgpu::Buffer>,
    /// element matrices, levels ≥ 2 (index = level)
    mats: Vec<Option<wgpu::Buffer>>,
    /// right-hand side and solution, levels ≥ 1
    lb: Vec<Option<wgpu::Buffer>>,
    lx: Vec<Option<wgpu::Buffer>>,
    /// smoother scratch, smoothed levels
    lr: Vec<wgpu::Buffer>,
    lt: Vec<wgpu::Buffer>,
    ld: Vec<wgpu::Buffer>,
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    p: wgpu::Buffer,
    q: wgpu::Buffer,
    us: Vec<wgpu::Buffer>,
    fs: Vec<wgpu::Buffer>,
    bottom: wgpu::Buffer,
    setup: Vec<Step>,
    init: Vec<Vec<Step>>,
    iter: Vec<Vec<Step>>,
    energy: Vec<Step>,
}

impl<'g> GpuMg<'g> {
    /// Buffers and programs for a problem. `free` per node, as in `Fem`.
    pub(super) fn new(gpu: &'g Gpu, p: &Problem, ke: &[[f64; 24]; 24], free: &[bool], s: &MgSettings) -> Result<Self> {
        let dev = &gpu.device;
        // levels: at least fine, level 1 (virtual) and one stored level
        let mut dims = vec![p.n, coarse(p.n), coarse(coarse(p.n))];
        loop {
            let last = *dims.last().unwrap();
            if mg::direct_cost(last) <= s.max_direct_flops || last.iter().all(|&v| v <= 1) {
                break;
            }
            dims.push(coarse(last));
        }
        let last = dims.len() - 1;
        let ne: usize = p.n.iter().product();
        let nf = 3 * nodes(p.n);
        let limit = dev
            .limits()
            .max_storage_buffer_binding_size
            .min(dev.limits().max_buffer_size);
        let bytes2 = 576 * 4 * dims[2].iter().product::<usize>() as u64;
        if (nf * 4) as u64 > limit || bytes2 > limit {
            bail!(
                "grid too large for the GPU solver ({} MB per vector, {} MB for level 2; limit {} MB)",
                (nf * 4) >> 20,
                bytes2 >> 20,
                limit >> 20
            );
        }

        let mut pipes = HashMap::new();
        let prelude_at = SRC.find("//// prelude").ok_or_else(|| anyhow!("mg.wgsl: no prelude"))?;
        let parts: Vec<&str> = SRC[prelude_at..].split("//// kernel ").collect();
        let prelude = parts[0].trim_start_matches("//// prelude");
        for part in &parts[1..] {
            let (name, body) = part.split_once('\n').unwrap_or((part, ""));
            let name = KERNELS
                .iter()
                .find(|k| **k == name.trim())
                .ok_or_else(|| anyhow!("mg.wgsl: unknown kernel `{name}`"))?;
            let src = format!("{prelude}\n{body}");
            let pipe = gpu
                .scoped(|| {
                    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                    dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .map_err(|e| anyhow!("kernel {name}: {e}"))?;
            pipes.insert(*name, pipe);
        }

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buf = |label: &str, len: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let init = |label: &str, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
        };
        let ke32: Vec<f32> = ke.iter().flatten().map(|&v| v as f32).collect();
        let table: Vec<f32> = mg::masked_ptkp_table(ke).iter().map(|&v| v as f32).collect();
        let free32: Vec<u32> = free.iter().map(|&f| f as u32).collect();
        let [ex, ey, _] = p.n;
        let (nx, ny) = (ex + 1, ey + 1);
        let emask: Vec<u32> = (0..ne)
            .map(|e| {
                let (i, j, k) = (e % ex, (e / ex) % ey, e / (ex * ey));
                (0..8)
                    .filter(|b| !free[((k + (b >> 2 & 1)) * ny + j + (b >> 1 & 1)) * nx + i + (b & 1)])
                    .map(|b| 1u32 << b)
                    .sum()
            })
            .collect();
        let len = |l: usize| 3 * nodes(dims[l]);
        let fs = p
            .loads
            .iter()
            .map(|l| {
                let f: Vec<f32> = l
                    .iter()
                    .enumerate()
                    .flat_map(|(m, v)| v.map(|c| if free[m] { c as f32 } else { 0.0 }))
                    .collect();
                init("f", bytemuck::cast_slice(&f))
            })
            .collect();
        let bdofs = len(last);
        let mut g = GpuMg {
            gpu,
            pipes,
            smoother: s.smoother,
            ee: buf("ee", ne),
            emask: init("emask", bytemuck::cast_slice(&emask)),
            free: init("free", bytemuck::cast_slice(&free32)),
            ke: init("ke", bytemuck::cast_slice(&ke32)),
            table: init("ptkp table", bytemuck::cast_slice(&table)),
            scalars: buf("scalars", SLOTS),
            partials: buf("partials", PARTIALS),
            ce: buf("ce", ne),
            diag: (0..=last).map(|l| buf("diag", len(l))).collect(),
            mats: (0..=last)
                .map(|l| (l >= 2).then(|| buf("mats", 576 * dims[l].iter().product::<usize>())))
                .collect(),
            lb: (0..=last).map(|l| (l >= 1).then(|| buf("b", len(l)))).collect(),
            lx: (0..=last).map(|l| (l >= 1).then(|| buf("x", len(l)))).collect(),
            lr: (0..last).map(|l| buf("r", len(l))).collect(),
            lt: (0..last).map(|l| buf("t", len(l))).collect(),
            ld: (0..last).map(|l| buf("d", len(l))).collect(),
            r: buf("cg r", nf),
            z: buf("cg z", nf),
            p: buf("cg p", nf),
            q: buf("cg q", nf),
            us: p.loads.iter().map(|_| buf("u", nf)).collect(),
            fs,
            bottom: buf("bottom inverse", bdofs * bdofs),
            dims,
            setup: vec![],
            init: vec![],
            iter: vec![],
            energy: vec![],
        };
        g.setup = g.record_setup();
        for c in 0..g.us.len() {
            let i = g.record_init(c);
            g.init.push(i);
            let i = g.record_iter(c);
            g.iter.push(i);
        }
        for c in 0..g.us.len() {
            let mut e = vec![];
            g.step(
                &mut e,
                "energy",
                Params {
                    n: dims4(g.dims[0]),
                    m: [(c > 0) as u32, 0, 0, 0],
                    ..Default::default()
                },
                &[&g.ee, &g.ke, &g.us[c], &g.ce],
                threads(ne),
            );
            g.energy.extend(e);
        }
        Ok(g)
    }

    pub(super) fn levels(&self) -> usize {
        self.dims.len()
    }

    pub(super) fn coarsest(&self) -> [usize; 3] {
        *self.dims.last().unwrap()
    }

    fn last(&self) -> usize {
        self.dims.len() - 1
    }

    fn len(&self, l: usize) -> usize {
        3 * nodes(self.dims[l])
    }

    fn step(&self, out: &mut Vec<Step>, kernel: &'static str, p: Params, bufs: &[&wgpu::Buffer], groups: [u32; 3]) {
        let dev = &self.gpu.device;
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
            layout: &self.pipes[kernel].get_bind_group_layout(0),
            entries: &entries,
        });
        out.push(Step { kernel, bind, groups });
    }

    /// y = K_l u
    fn apply(&self, out: &mut Vec<Step>, l: usize, u: &wgpu::Buffer, y: &wgpu::Buffer) {
        let t = threads(nodes(self.dims[l]));
        match l {
            0 => self.step(
                out,
                "apply_fine",
                Params {
                    n: dims4(self.dims[0]),
                    ..Default::default()
                },
                &[&self.ee, &self.free, &self.ke, u, y],
                t,
            ),
            1 => self.step(
                out,
                "apply_l1",
                Params {
                    n: dims4(self.dims[0]),
                    ..Default::default()
                },
                &[&self.ee, &self.emask, &self.table, u, y],
                t,
            ),
            _ => self.step(
                out,
                "apply_elem",
                Params {
                    n: dims4(self.dims[l]),
                    ..Default::default()
                },
                &[self.mats[l].as_ref().unwrap(), u, y],
                t,
            ),
        }
    }

    /// y = a y + c x (x / w where w > 0 when `w` is given); a and c optionally times a scalar slot
    #[allow(clippy::too_many_arguments)]
    fn lin(
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
    fn dot(
        &self,
        out: &mut Vec<Step>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        w: Option<&wgpu::Buffer>,
        len: usize,
        slot: u32,
    ) {
        self.step(
            out,
            "dot_partial",
            Params {
                m: [len as u32, w.is_some() as u32, 0, 0],
                ..Default::default()
            },
            &[a, b, w.unwrap_or(a), &self.partials],
            [PARTIALS as u32, 1, 1],
        );
        self.step(
            out,
            "dot_final",
            Params {
                m: [PARTIALS as u32, slot, 0, 0],
                ..Default::default()
            },
            &[&self.partials, &self.scalars],
            [1, 1, 1],
        );
    }

    fn scalar(&self, out: &mut Vec<Step>, op: u32, a: u32, b: u32, dst: u32) {
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

    fn record_setup(&self) -> Vec<Step> {
        let mut out = vec![];
        let last = self.last();
        self.step(
            &mut out,
            "diag_fine",
            Params {
                n: dims4(self.dims[0]),
                ..Default::default()
            },
            &[&self.ee, &self.free, &self.ke, &self.diag[0]],
            threads(nodes(self.dims[0])),
        );
        self.step(
            &mut out,
            "diag_l1",
            Params {
                n: dims4(self.dims[0]),
                ..Default::default()
            },
            &[&self.ee, &self.emask, &self.table, &self.diag[1]],
            threads(nodes(self.dims[1])),
        );
        for l in 2..=last {
            let mats = self.mats[l].as_ref().unwrap();
            let (n, virt, child) = if l == 2 {
                (self.dims[0], 1, &self.table)
            } else {
                (self.dims[l - 1], 0, self.mats[l - 1].as_ref().unwrap())
            };
            self.step(
                &mut out,
                "assemble",
                Params {
                    n: dims4(n),
                    m: [virt, 0, 0, 0],
                    ..Default::default()
                },
                &[&self.ee, &self.emask, &self.table, child, mats],
                groups(self.dims[l].iter().product()),
            );
            self.step(
                &mut out,
                "diag_elem",
                Params {
                    n: dims4(self.dims[l]),
                    ..Default::default()
                },
                &[mats, &self.diag[l]],
                threads(nodes(self.dims[l])),
            );
        }
        // λmax(D⁻¹K) per smoothed level, by power iteration
        for l in 0..last {
            let (v, kv, d, n) = (&self.lr[l], &self.lt[l], &self.diag[l], self.len(l));
            self.step(
                &mut out,
                "fill_random",
                Params {
                    m: [n as u32, 0, 0, 0],
                    ..Default::default()
                },
                &[d, v],
                threads(n),
            );
            for _ in 0..12 {
                self.apply(&mut out, l, v, kv);
                self.dot(&mut out, v, kv, None, n, NUM);
                self.dot(&mut out, v, v, Some(d), n, DEN);
                self.scalar(&mut out, OP_DIV, NUM, DEN, LAM + l as u32);
                self.lin(&mut out, v, kv, Some(d), (0.0, NONE), (1.0, NONE), n);
                self.dot(&mut out, v, v, Some(d), n, NRM);
                self.scalar(&mut out, OP_RSQRT, NRM, NRM, SC);
                self.lin(&mut out, v, kv, None, (1.0, SC), (0.0, NONE), n);
            }
            self.scalar(&mut out, OP_RECIP, LAM + l as u32, 0, INVL + l as u32);
        }
        out
    }

    fn smooth(&self, out: &mut Vec<Step>, l: usize, b: &wgpu::Buffer, x: &wgpu::Buffer, zero_start: bool) {
        let (r, t, d, diag, n) = (&self.lr[l], &self.lt[l], &self.ld[l], &self.diag[l], self.len(l));
        let invl = INVL + l as u32;
        // r = b - K x
        let residual = |out: &mut Vec<Step>, zero: bool| {
            if zero {
                self.lin(out, r, b, None, (0.0, NONE), (1.0, NONE), n);
            } else {
                self.apply(out, l, x, r);
                self.lin(out, r, b, None, (-1.0, NONE), (1.0, NONE), n);
            }
        };
        match self.smoother {
            Smoother::Jacobi { sweeps, omega } => {
                for k in 0..sweeps {
                    let zero = zero_start && k == 0;
                    residual(out, zero);
                    let a = if zero { 0.0 } else { 1.0 };
                    self.lin(out, x, r, Some(diag), (a, NONE), (omega as f32, invl), n);
                }
            }
            Smoother::Chebyshev { degree } => {
                // on [0.1, 1.1]·λmax: θ = 0.6 λ, δ = 0.5 λ, σ = θ / δ
                let sigma = 1.2f64;
                let mut rho = 1.0 / sigma;
                residual(out, zero_start);
                self.lin(out, d, r, Some(diag), (0.0, NONE), ((1.0 / 0.6) as f32, invl), n);
                for k in 1..=degree {
                    let a = if zero_start && k == 1 { 0.0 } else { 1.0 };
                    self.lin(out, x, d, None, (a, NONE), (1.0, NONE), n);
                    if k == degree {
                        break;
                    }
                    self.apply(out, l, d, t);
                    self.lin(out, r, t, None, (1.0, NONE), (-1.0, NONE), n);
                    let rho_new = 1.0 / (2.0 * sigma - rho);
                    self.lin(
                        out,
                        d,
                        r,
                        Some(diag),
                        ((rho_new * rho) as f32, NONE),
                        ((2.0 * rho_new / 0.5) as f32, invl),
                        n,
                    );
                    rho = rho_new;
                }
            }
        }
    }

    fn vcycle(&self, out: &mut Vec<Step>, l: usize, b: &wgpu::Buffer, x: &wgpu::Buffer) {
        if l == self.last() {
            let n = self.len(l);
            self.step(
                out,
                "bottom",
                Params {
                    m: [n as u32, 0, 0, 0],
                    ..Default::default()
                },
                &[&self.bottom, b, x],
                threads(n),
            );
            return;
        }
        let n = self.len(l);
        self.smooth(out, l, b, x, true);
        let r = &self.lr[l];
        self.apply(out, l, x, r);
        self.lin(out, r, b, None, (-1.0, NONE), (1.0, NONE), n);
        let (cb, cx) = (self.lb[l + 1].as_ref().unwrap(), self.lx[l + 1].as_ref().unwrap());
        self.step(
            out,
            "restrict",
            Params {
                n: dims4(self.dims[l]),
                ..Default::default()
            },
            &[r, cb],
            threads(nodes(self.dims[l + 1])),
        );
        self.vcycle(out, l + 1, cb, cx);
        self.step(
            out,
            "prolong",
            Params {
                n: dims4(self.dims[l]),
                ..Default::default()
            },
            &[cx, &self.diag[l], x],
            threads(nodes(self.dims[l])),
        );
        self.smooth(out, l, b, x, false);
    }

    fn record_init(&self, c: usize) -> Vec<Step> {
        let mut out = vec![];
        let n = self.len(0);
        let (u, f) = (&self.us[c], &self.fs[c]);
        self.apply(&mut out, 0, u, &self.q);
        self.lin(&mut out, &self.r, f, None, (0.0, NONE), (1.0, NONE), n);
        self.lin(&mut out, &self.r, &self.q, None, (1.0, NONE), (-1.0, NONE), n);
        self.dot(&mut out, &self.r, &self.r, None, n, RR);
        self.dot(&mut out, f, f, None, n, FF);
        self.vcycle(&mut out, 0, &self.r, &self.z);
        self.lin(&mut out, &self.p, &self.z, None, (0.0, NONE), (1.0, NONE), n);
        self.dot(&mut out, &self.r, &self.z, None, n, RZ);
        out
    }

    fn record_iter(&self, c: usize) -> Vec<Step> {
        let mut out = vec![];
        let n = self.len(0);
        let u = &self.us[c];
        self.apply(&mut out, 0, &self.p, &self.q);
        self.dot(&mut out, &self.p, &self.q, None, n, PQ);
        self.scalar(&mut out, OP_DIV, RZ, PQ, ALPHA);
        self.lin(&mut out, u, &self.p, None, (1.0, NONE), (1.0, ALPHA), n);
        self.lin(&mut out, &self.r, &self.q, None, (1.0, NONE), (-1.0, ALPHA), n);
        self.vcycle(&mut out, 0, &self.r, &self.z);
        self.dot(&mut out, &self.r, &self.z, None, n, RZN);
        self.scalar(&mut out, OP_DIV, RZN, RZ, BETA);
        self.scalar(&mut out, OP_COPY, RZN, RZN, RZ);
        self.lin(&mut out, &self.p, &self.z, None, (1.0, BETA), (1.0, NONE), n);
        self.dot(&mut out, &self.r, &self.r, None, n, RR);
        out
    }

    fn run(&self, programs: &[&[Step]]) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for prog in programs {
                for s in prog.iter() {
                    pass.set_pipeline(&self.pipes[s.kernel]);
                    pass.set_bind_group(0, &s.bind, &[]);
                    pass.dispatch_workgroups(s.groups[0], s.groups[1], s.groups[2]);
                }
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn read_f32(&self, b: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        let bytes = self.gpu.read_buffer(b, (len * 4) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Build the hierarchy for element stiffnesses `ee` (E_e h per fine element).
    pub(super) fn set_stiffness(&self, ee: &[f64]) -> Result<()> {
        let ee32: Vec<f32> = ee.iter().map(|&v| v as f32).collect();
        self.gpu.queue.write_buffer(&self.ee, 0, bytemuck::cast_slice(&ee32));
        self.run(&[&self.setup]);
        // exact solve on the coarsest level
        let last = self.last();
        let n = self.dims[last];
        let mats = self.read_f32(self.mats[last].as_ref().unwrap(), 576 * n.iter().product::<usize>())?;
        let inv = mg::dense_inverse(n, mats.iter().map(|&v| v as f64).collect());
        let inv32: Vec<f32> = inv.iter().map(|&v| v as f32).collect();
        self.gpu
            .queue
            .write_buffer(&self.bottom, 0, bytemuck::cast_slice(&inv32));
        Ok(())
    }

    /// Solve load case `c` from its previous solution to relative residual `tol`. Returns the CG
    /// iterations (a multiple of the check interval).
    pub(super) fn solve(&self, c: usize, tol: f64, maxit: usize) -> Result<usize> {
        self.run(&[&self.init[c]]);
        let s = self.read_f32(&self.scalars, SLOTS)?;
        let target = (tol * tol) as f32 * s[FF as usize];
        if s[RR as usize] <= target {
            return Ok(0);
        }
        let mut its = 0;
        let batch: Vec<&[Step]> = vec![&self.iter[c]; CHECK_EVERY];
        while its < maxit {
            self.run(&batch);
            its += CHECK_EVERY;
            let rr = self.read_f32(&self.scalars, SLOTS)?[RR as usize];
            if !rr.is_finite() {
                bail!("GPU CG diverged (residual {rr})");
            }
            if rr <= target {
                break;
            }
        }
        Ok(its)
    }

    /// Per fine element: Σ over load cases of u_eᵀ ke u_e.
    pub(super) fn energies(&self) -> Result<Vec<f32>> {
        self.run(&[&self.energy]);
        self.read_f32(&self.ce, self.dims[0].iter().product())
    }

    pub(super) fn displacement(&self, c: usize) -> Result<Vec<f32>> {
        self.read_f32(&self.us[c], self.len(0))
    }

    /// Scalar slot (tests): λmax estimate of level l.
    #[cfg(test)]
    fn lambda(&self, l: usize) -> Result<f32> {
        Ok(self.read_f32(&self.scalars, SLOTS)?[(LAM + l as u32) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topopt::Fem;

    fn problem(n: [usize; 3]) -> Problem {
        let ne: usize = n.iter().product();
        let nn = nodes(n);
        let mut p = Problem {
            n,
            h: 1.0,
            origin: [0.0; 3],
            domain: (0..ne).map(|e| e % 11 != 4).collect(),
            passive: vec![false; ne],
            fixed: vec![false; nn],
            loads: vec![vec![[0.0; 3]; nn]],
            cases: vec![1],
        };
        let [nx, ny, _] = n.map(|v| v + 1);
        for j in 0..ny {
            for i in 0..nx / 2 {
                let m = p.node(i, j, 0);
                p.fixed[m] = true;
            }
            let m = p.node(nx - 1, j, n[2]);
            p.loads[0][m] = [0.3, -1.0, 0.5];
        }
        p
    }

    fn pseudo(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104729) % 23) as f64 / 11.0 - 1.0)
            .collect()
    }

    fn close(got: &[f32], want: &[f64], what: &str) {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max);
        assert!(err <= 2e-5 * scale, "{what}: max error {err} of {scale}");
        eprintln!("{what}: max error {:.2e} relative", err / scale);
    }

    /// Fine and level-1 operators, diagonals and transfers agree with the CPU path.
    #[test]
    fn kernels_match_cpu() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let p = problem([7, 5, 9]);
        let ke = super::super::hex_ke(0.3);
        let active = p.active_nodes();
        let free: Vec<bool> = (0..active.len()).map(|i| active[i] && !p.fixed[i]).collect();
        let fem = Fem {
            p: &p,
            ke: &ke,
            ee: (0..p.domain.len())
                .map(|e| {
                    if p.domain[e] {
                        if e % 3 == 0 { 1e-3 } else { 1.0 + (e % 5) as f64 }
                    } else {
                        0.0
                    }
                })
                .collect(),
            free: free.clone(),
        };
        let g = GpuMg::new(&gpu, &p, &ke, &free, &MgSettings::default()).unwrap();
        g.set_stiffness(&fem.ee).unwrap();
        let write = |b: &wgpu::Buffer, v: &[f64]| {
            let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            gpu.queue.write_buffer(b, 0, bytemuck::cast_slice(&v32));
        };
        let masked = |v: Vec<f64>| -> Vec<f64> {
            v.iter()
                .enumerate()
                .map(|(i, x)| if free[i / 3] { *x } else { 0.0 })
                .collect()
        };
        let n0 = g.len(0);
        let u = masked(pseudo(n0, 1));

        // fine mat-vec and diagonal
        write(&g.p, &u);
        let mut prog = vec![];
        g.apply(&mut prog, 0, &g.p, &g.q);
        g.run(&[&prog]);
        let mut want = vec![0.0; n0];
        fem.matvec(&u, &mut want);
        close(&g.read_f32(&g.q, n0).unwrap(), &want, "fine K u");
        let inv = fem.inv_diag();
        let d: Vec<f64> = inv.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();
        close(&g.read_f32(&g.diag[0], n0).unwrap(), &d, "fine diagonal");

        // level 1: Pᵀ K P against the CPU's assembled level
        let c1 = mg::tests_api::level1(&fem);
        let n1 = g.len(1);
        let v = pseudo(n1, 2);
        write(g.lb[1].as_ref().unwrap(), &v);
        let mut prog = vec![];
        g.apply(&mut prog, 1, g.lb[1].as_ref().unwrap(), g.lx[1].as_ref().unwrap());
        g.run(&[&prog]);
        close(
            &g.read_f32(g.lx[1].as_ref().unwrap(), n1).unwrap(),
            &c1.0(&v),
            "level-1 K u",
        );
        close(&g.read_f32(&g.diag[1], n1).unwrap(), &c1.1, "level-1 diagonal");

        // level 2, assembled on the GPU
        let c2 = mg::tests_api::level2(&fem);
        let n2 = g.len(2);
        let v = pseudo(n2, 3);
        write(g.lb[2].as_ref().unwrap(), &v);
        let mut prog = vec![];
        g.apply(&mut prog, 2, g.lb[2].as_ref().unwrap(), g.lx[2].as_ref().unwrap());
        g.run(&[&prog]);
        close(
            &g.read_f32(g.lx[2].as_ref().unwrap(), n2).unwrap(),
            &c2(&v),
            "level-2 K u",
        );

        // transfers
        write(&g.lr[0], &u);
        let mut prog = vec![];
        g.step(
            &mut prog,
            "restrict",
            Params {
                n: dims4(g.dims[0]),
                ..Default::default()
            },
            &[&g.lr[0], g.lb[1].as_ref().unwrap()],
            threads(nodes(g.dims[1])),
        );
        g.run(&[&prog]);
        let mut want = vec![0.0; n1];
        mg::tests_api::restrict(p.n, &u, &mut want);
        close(&g.read_f32(g.lb[1].as_ref().unwrap(), n1).unwrap(), &want, "restrict");
        let v = pseudo(n1, 4);
        write(g.lx[1].as_ref().unwrap(), &v);
        write(&g.z, &u);
        let mut prog = vec![];
        g.step(
            &mut prog,
            "prolong",
            Params {
                n: dims4(g.dims[0]),
                ..Default::default()
            },
            &[g.lx[1].as_ref().unwrap(), &g.diag[0], &g.z],
            threads(nodes(g.dims[0])),
        );
        g.run(&[&prog]);
        let mut want = u.clone();
        mg::tests_api::prolong_add(p.n, &v, &mut want, &inv);
        close(&g.read_f32(&g.z, n0).unwrap(), &want, "prolong");

        // λmax estimates are positive and of the CPU's size
        for l in 0..g.last() {
            let lam = g.lambda(l).unwrap();
            assert!(lam > 0.5 && lam < 10.0, "level {l}: λmax {lam}");
        }
    }

    /// The GPU solve reaches the CPU solution, at a realistic scale: E·h in the thousands, small
    /// nodal loads, a long lever arm (large rigid-body-like displacements), and 1e-9 contrast.
    /// Before the element operator worked on relative displacements, f32 rounding of ke made this
    /// cantilever 0.35% too soft.
    #[test]
    fn gpu_solve_matches_cpu() {
        let Ok(gpu) = crate::gpu::headless() else {
            eprintln!("no GPU; skipped");
            return;
        };
        let n = [40, 4, 6];
        let mut p = problem(n);
        // clamp the x = 0 face only, load the far end lightly
        p.fixed.fill(false);
        p.loads[0].iter_mut().for_each(|l| *l = [0.0; 3]);
        let [nx, ny, nz] = n.map(|v| v + 1);
        for k in 0..nz {
            for j in 0..ny {
                let m = p.node(0, j, k);
                p.fixed[m] = true;
                let m = p.node(nx - 1, j, k);
                p.loads[0][m] = [0.0, 0.0, -0.01];
            }
        }
        p.domain.fill(true);
        let ke = super::super::hex_ke(0.35);
        let active = p.active_nodes();
        let free: Vec<bool> = (0..active.len()).map(|i| active[i] && !p.fixed[i]).collect();
        let fem = Fem {
            p: &p,
            ke: &ke,
            // a hollow beam: solid skin, near-void core
            ee: (0..p.domain.len())
                .map(|e| {
                    let (j, k) = ((e / n[0]) % n[1], e / (n[0] * n[1]));
                    let skin = j == 0 || j == n[1] - 1 || k == 0 || k == n[2] - 1;
                    4550.0 * if skin { 1.0 } else { 1e-9 }
                })
                .collect(),
            free: free.clone(),
        };
        let f: Vec<f64> = p.loads[0].iter().flatten().copied().collect();
        let mut m = mg::Multigrid::new(&fem, &MgSettings::default());
        let mut u = vec![0.0; f.len()];
        fem.solve(&f, &mut u, 1e-10, 1000, &mut |r, z| m.precondition(r, z));
        let g = GpuMg::new(&gpu, &p, &ke, &free, &MgSettings::default()).unwrap();
        g.set_stiffness(&fem.ee).unwrap();
        let its = g.solve(0, 1e-5, 1000).unwrap();
        let ug = g.displacement(0).unwrap();
        let fu = |u: &dyn Fn(usize) -> f64| (0..f.len()).map(|i| f[i] * u(i)).sum::<f64>();
        let (c_cpu, c_gpu) = (fu(&|i| u[i]), fu(&|i| ug[i] as f64));
        eprintln!("{its} GPU CG iterations; compliance {c_gpu:.6e} vs {c_cpu:.6e}");
        assert!((c_gpu - c_cpu).abs() < 1e-4 * c_cpu, "compliance {c_gpu} vs {c_cpu}");
        close(&ug, &u, "GPU u");
    }
}
