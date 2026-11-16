//! Hardware-dependent choices for the GPU optimizer's kernels (`gpu_engine::Engine`).
//!
//! A profile is picked from the adapter's vendor; `GEN3D_GPU_PROFILE=nvidia|apple|generic`
//! overrides the pick, and `GEN3D_GPU_TUNE="key=value,..."` overrides single fields on top of it
//! (keys as the field names, booleans 0/1). `gen3d bench-topopt` prints the result.
//!
//! The NVIDIA profile is measured on an RTX 3080 (Ampere). The Apple profile keeps the settings the
//! kernels were first tuned with on an M1 Pro; nothing newer has been measured there.

use anyhow::{Result, anyhow, bail};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// profile name, for logs
    pub profile: &'static str,
    /// Compile kernels without wgpu's loop bounding (a hidden counter on every loop, which keeps
    /// drivers from unrolling). Our loops have fixed or data-bounded trip counts. Buffer bounds
    /// checks stay on either way.
    pub unbounded_loops: bool,
    /// Bind the 24×24 element matrix `ke` as a uniform buffer rather than storage. With unrolled
    /// loops its offsets are constants: NVIDIA reads them from the constant bank as FMA operands
    /// instead of issuing one load per entry, which made the fine stencils 2.5-3× faster.
    pub ke_uniform: bool,
}

impl Tuning {
    /// RTX 30xx/40xx-class consumer GPUs (the primary target).
    pub const NVIDIA: Tuning = Tuning {
        profile: "nvidia",
        unbounded_loops: true,
        ke_uniform: true,
    };

    /// Apple silicon: the settings tuned on an M1 Pro.
    pub const APPLE: Tuning = Tuning {
        profile: "apple",
        unbounded_loops: false,
        ke_uniform: false,
    };

    /// Anything else (AMD, Intel, software): the NVIDIA settings, which only remove overhead.
    pub const GENERIC: Tuning = Tuning {
        profile: "generic",
        ..Tuning::NVIDIA
    };

    pub fn named(name: &str) -> Option<Tuning> {
        match name.to_ascii_lowercase().as_str() {
            "nvidia" => Some(Tuning::NVIDIA),
            "apple" => Some(Tuning::APPLE),
            "generic" => Some(Tuning::GENERIC),
            _ => None,
        }
    }

    /// The profile for a PCI vendor id.
    pub fn for_vendor(vendor: u32) -> Tuning {
        match vendor {
            0x10de => Tuning::NVIDIA,
            0x106b => Tuning::APPLE,
            _ => Tuning::GENERIC,
        }
    }

    /// The profile for `info`, with the environment's overrides.
    pub fn detect(info: &wgpu::AdapterInfo) -> Result<Tuning> {
        // Metal adapters may not report Apple's PCI id
        let mut t = if info.backend == wgpu::Backend::Metal && info.vendor != 0x10de {
            Tuning::APPLE
        } else {
            Tuning::for_vendor(info.vendor)
        };
        if let Ok(name) = std::env::var("GEN3D_GPU_PROFILE") {
            t = Tuning::named(&name)
                .ok_or_else(|| anyhow!("GEN3D_GPU_PROFILE={name}: expected nvidia, apple or generic"))?;
        }
        if let Ok(spec) = std::env::var("GEN3D_GPU_TUNE") {
            t.apply(&spec).map_err(|e| anyhow!("GEN3D_GPU_TUNE: {e}"))?;
        }
        Ok(t)
    }

    /// Set fields from `key=value,...`.
    pub fn apply(&mut self, spec: &str) -> Result<()> {
        for kv in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| anyhow!("`{kv}`: expected key=value"))?;
            let flag = || match v.trim() {
                "1" | "true" | "on" => Ok(true),
                "0" | "false" | "off" => Ok(false),
                _ => Err(anyhow!("`{kv}`: expected 0 or 1")),
            };
            match k.trim() {
                "unbounded_loops" => self.unbounded_loops = flag()?,
                "ke_uniform" => self.ke_uniform = flag()?,
                _ => bail!("unknown key `{k}`"),
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Tuning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (unbounded_loops={}, ke_uniform={})",
            self.profile, self.unbounded_loops as u8, self.ke_uniform as u8
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_parse() {
        let mut t = Tuning::NVIDIA;
        t.apply("ke_uniform=0, unbounded_loops=off").unwrap();
        assert!(!t.ke_uniform && !t.unbounded_loops);
        assert!(t.apply("nope=1").is_err());
        assert!(t.apply("ke_uniform=maybe").is_err());
        assert_eq!(Tuning::for_vendor(0x10de), Tuning::NVIDIA);
        assert_eq!(Tuning::named("Apple"), Some(Tuning::APPLE));
    }
}
