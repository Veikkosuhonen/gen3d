"""Load SDF shape files: resolve #include, parse //! directives, map GLSL errors back to files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LIB_DIR = Path(__file__).parent / "glsl"

INCLUDE_RE = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]')
DIRECTIVE_RE = re.compile(r"^\s*//!\s*(\w+)\s*:?\s*(.*)$")


@dataclass
class Shape:
    path: Path
    source: str  # fully expanded GLSL (without #version)
    files: list[Path]  # index == GLSL source-string number used in #line
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-50.0, -50.0, 0.0),
        (50.0, 50.0, 100.0),
    )
    voxel: float = 0.25  # mm
    directives: dict[str, str] = field(default_factory=dict)

    @property
    def size(self):
        lo, hi = self.bounds
        return tuple(h - l for l, h in zip(lo, hi))

    def deps(self) -> list[Path]:
        return list(self.files)


class ShaderError(Exception):
    pass


def _expand(path: Path, files: list[Path], stack: tuple[Path, ...], seen: set[Path]) -> str:
    path = path.resolve()
    if path in stack:
        raise ShaderError(f"circular #include: {path}")
    if path in seen:  # include guard: each file only once
        return ""
    seen.add(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise ShaderError(f"cannot read {path}: {e}") from e
    idx = len(files)
    files.append(path)
    out = [f"#line 1 {idx}"]
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = INCLUDE_RE.match(line)
        if m:
            name = m.group(1)
            cands = [path.parent / name, LIB_DIR / name]
            target = next((c for c in cands if c.exists()), None)
            if target is None:
                raise ShaderError(f"{path}:{lineno}: include not found: {name}")
            out.append(_expand(target, files, stack + (path,), seen))
            out.append(f"#line {lineno + 1} {idx}")
        elif line.lstrip().startswith("#version"):
            out.append("")  # we supply our own
        else:
            out.append(line)
    return "\n".join(out)


def load_shape(path: str | Path, defines: dict[str, str] | None = None) -> Shape:
    path = Path(path).resolve()
    files: list[Path] = []
    body = _expand(path, files, (), set())
    shape = Shape(path=path, source="", files=files)

    for line in path.read_text().splitlines():
        m = DIRECTIVE_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = re.split(r"\s+(?://|\()", m.group(2))[0].strip()  # drop trailing // or (...) notes
        shape.directives[key] = val
        try:
            if key == "bounds":
                v = [float(x) for x in val.replace(",", " ").split()[:6]]
                if len(v) != 6:
                    raise ValueError("expected 6 numbers: xmin ymin zmin xmax ymax zmax")
                shape.bounds = (tuple(v[:3]), tuple(v[3:]))
            elif key == "voxel":
                shape.voxel = float(val.split()[0])
        except ValueError as e:
            raise ShaderError(f"{path}: bad //! {key}: {e}") from e

    defs = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    shape.source = defs + body + "\n"
    return shape


_ERR_PATTERNS = [
    re.compile(r"(?m)^(\d+)\((\d+)\)"),  # NVIDIA: 0(12) : error ...
    re.compile(r"(?m)^(?:ERROR: )?(\d+):(\d+)"),  # Mesa/AMD: 0:12(3): error ...
]


def map_errors(msg: str, shape: Shape, preamble_files: int = 0) -> str:
    """Rewrite 'N(line)' references into 'file:line'."""

    def sub(m: re.Match) -> str:
        i = int(m.group(1)) - preamble_files
        if 0 <= i < len(shape.files):
            return f"{shape.files[i]}:{m.group(2)}"
        return m.group(0)

    for pat in _ERR_PATTERNS:
        msg = pat.sub(sub, msg)
    return msg
