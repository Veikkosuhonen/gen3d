//! wgpu device setup shared by the mesher, the headless renderer and the viewer.

use anyhow::{Context, Result, anyhow};

use crate::shape::Compiled;

pub struct Gpu {
    /// Kept alive alongside the device and surfaces created from it.
    _instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Pick the best adapter: discrete > integrated > anything else; software only as a last resort.
/// `WGPU_BACKEND=vulkan|gl|...` and `WGPU_ADAPTER_NAME=...` are honoured.
pub fn create(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>, log: bool) -> Result<Gpu> {
    let adapter = pollster::block_on(async {
        if let Ok(a) = wgpu::util::initialize_adapter_from_env(&instance, surface).await {
            return Ok(a);
        }
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if let Some(s) = surface {
            adapters.retain(|a| a.is_surface_supported(s));
        }
        let rank = |a: &wgpu::Adapter| {
            let info = a.get_info();
            let kind = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
                wgpu::DeviceType::Cpu => 3,
            };
            let backend = match info.backend {
                wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
                _ => 1,
            };
            (kind, backend)
        };
        adapters.sort_by_key(rank);
        adapters
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no GPU adapter found (Vulkan, Metal, DX12 or GL)"))
    })?;
    let info = adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || ["llvmpipe", "lavapipe", "softpipe", "swiftshader"]
            .iter()
            .any(|s| info.name.to_lowercase().contains(s));
    if software {
        eprintln!(
            "warning: no hardware GPU available, using software rendering ({}; slow)",
            info.name
        );
    } else if log {
        println!("GPU: {} ({:?})", info.name, info.backend);
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gen3d"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .context("could not open the GPU device")?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("\x1b[31mgpu error:\x1b[0m {e}")));
    Ok(Gpu {
        _instance: instance,
        adapter,
        device,
        queue,
    })
}

pub fn headless() -> Result<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    create(instance, None, false)
}

impl Gpu {
    /// Create a shader module, catching any wgpu-side validation error instead of panicking.
    ///
    /// By default wgpu wraps every loop in a hidden iteration counter (`force_loop_bounding`) and
    /// guards integer division. On Vulkan that blocks unrolling and adds work to every iteration of
    /// the raymarch loop and every `map()` loop, which is a large slowdown for SDF shaders. Both are
    /// turned off here; buffer bounds checks stay on. The cost: a shape with a genuinely infinite
    /// loop may hang the GPU until the driver resets it, instead of being cut off.
    /// `GEN3D_SHADER_CHECKS=1` restores wgpu's fully checked mode.
    pub fn shader(&self, compiled: &Compiled, label: &str) -> Result<wgpu::ShaderModule> {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(compiled.source.as_str().into()),
        };
        if std::env::var_os("GEN3D_SHADER_CHECKS").is_some_and(|v| v != "0") {
            return self.scoped(|| self.device.create_shader_module(desc));
        }
        let mut checks = wgpu::ShaderRuntimeChecks::checked();
        checks.force_loop_bounding = false;
        checks.int_div_checks = false;
        // SAFETY: shapes are local, user-written SDFs; an infinite loop only stalls the user's own
        // GPU (the driver's watchdog recovers it). Memory bounds checks remain enabled.
        self.scoped(|| unsafe { self.device.create_shader_module_trusted(desc, checks) })
    }

    /// Run `f` inside a validation error scope and turn errors into `Err`.
    pub fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(anyhow!("{e}")),
            None => Ok(out),
        }
    }

    /// Copy `size` bytes of `src` back to the CPU.
    pub fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("gpu poll failed: {e}"))?;
        rx.recv()?.map_err(|e| anyhow!("buffer map failed: {e}"))?;
        let data = staging.get_mapped_range(..).map_err(|e| anyhow!("{e:?}"))?.to_vec();
        staging.unmap();
        Ok(data)
    }
}
