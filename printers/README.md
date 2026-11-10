# Printer / slicer profiles

`gen3d slice` loads slicer settings from (first match wins):

1. `--config FILE` on the command line (repeatable),
2. a `//! slicer: ../printers/foo.ini` directive in the shape file (comma-separated list, relative to the shape),
3. `printers/default.ini` in the directory you run `gen3d` from.

With none of these, the slicer's built-in defaults are used (generic printer, 0.4 mm nozzle, PLA). That's
fine for a smoke test, but the G-code won't match your machine.

## PrusaSlicer (recommended)

Set up your printer, filament and print profile in the PrusaSlicer GUI, then
**File → Export → Export Config…** and save it here as `default.ini`. That single `.ini` holds
everything (printer, filament, print settings, start/end G-code).

Override individual settings per run:

    uv run gen3d slice shapes/knob.glsl -- --layer-height 0.15 --fill-density 30% --support-material

## OrcaSlicer

Orca's CLI takes JSON presets. Export your machine / process / filament presets and pass them with
names containing `machine`, `process` and `filament`:

    uv run gen3d slice shapes/knob.glsl --slicer orca \
        -c printers/machine.json -c printers/process.json -c printers/filament.json
