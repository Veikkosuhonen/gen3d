"""Live raymarching preview window with hot reload."""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import glfw
import moderngl

from .shader import Shape, ShaderError, load_shape, map_errors

VERT = """#version 430
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FRAG_HEAD = """#version 430
"""

FRAG_TAIL = """
#line 1 999
in vec2 v_uv;
out vec4 fragColor;
uniform vec2 u_res;
uniform vec3 u_eye;
uniform vec3 u_target;
uniform vec3 u_bmin;
uniform vec3 u_bmax;
uniform float u_step;      // relaxation factor for non-exact SDFs
uniform int u_clip;        // clip to export bounds
uniform int u_section;     // 0 off, 1 on
uniform float u_cut;       // section plane y
uniform int u_showBounds;

float boxSd(vec3 p, vec3 lo, vec3 hi) {
    vec3 c = 0.5 * (lo + hi), b = 0.5 * (hi - lo);
    vec3 q = abs(p - c) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// returns distance; id: 0 = shape surface, 1 = section cut face
float scene(vec3 p, out int id) {
    float d = map(p);
    if (u_clip == 1) d = max(d, boxSd(p, u_bmin, u_bmax));
    id = 0;
    if (u_section == 1) {
        float c = u_cut - p.y;    // remove y < cut (the half facing the default camera)
        if (c > d) { d = c; id = 1; }
    }
    return d;
}
float sceneD(vec3 p) { int i; return scene(p, i); }

vec3 calcNormal(vec3 p, float h) {
    const vec2 k = vec2(1, -1);
    return normalize(k.xyy * sceneD(p + k.xyy * h) + k.yyx * sceneD(p + k.yyx * h) +
                     k.yxy * sceneD(p + k.yxy * h) + k.xxx * sceneD(p + k.xxx * h));
}

vec2 boxHit(vec3 ro, vec3 rd, vec3 lo, vec3 hi) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (lo - ro) * inv, t1 = (hi - ro) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

float softShadow(vec3 ro, vec3 rd, float tmax) {
    float res = 1.0, t = 0.05;
    for (int i = 0; i < 64 && t < tmax; i++) {
        float h = sceneD(ro + rd * t);
        res = min(res, 10.0 * h / t);
        if (res < 0.01) break;
        t += clamp(h * u_step, 0.05, 5.0);
    }
    return clamp(res, 0.0, 1.0);
}

float ao(vec3 p, vec3 n, float scale) {
    float o = 0.0, w = 1.0;
    for (int i = 1; i <= 5; i++) {
        float h = scale * float(i);
        o += w * (h - sceneD(p + n * h));
        w *= 0.6;
    }
    return clamp(1.0 - o / scale * 0.35, 0.0, 1.0);
}

float gridLine(vec2 p, float spacing, float width) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) * spacing;
    vec2 fw = fwidth(p);
    vec2 l = 1.0 - smoothstep(vec2(0.0), fw * width, g);
    return max(l.x, l.y);
}

float edgeLines(vec3 p, vec3 lo, vec3 hi, float w) {
    // distance to the 12 edges of the bounds box, measured in the plane of the face
    vec3 a = min(abs(p - lo), abs(p - hi));
    vec3 s = step(a, vec3(w));
    return clamp(s.x * s.y + s.y * s.z + s.x * s.z, 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;
    vec3 fw = normalize(u_target - u_eye);
    vec3 rt = normalize(cross(fw, vec3(0, 0, 1)));
    if (length(cross(fw, vec3(0, 0, 1))) < 1e-4) rt = vec3(1, 0, 0);
    vec3 up = cross(rt, fw);
    vec3 ro = u_eye, rd = normalize(fw * 1.8 + uv.x * rt + uv.y * up);

    vec3 diag = u_bmax - u_bmin;
    float scale = length(diag);
    vec3 col = mix(vec3(0.10, 0.11, 0.13), vec3(0.20, 0.22, 0.26), 0.5 + 0.5 * rd.z);

    // ground plane z = 0
    float tPlane = rd.z < 0.0 ? -ro.z / rd.z : 1e9;

    // march only inside an expanded bounds box (or a large region if not clipping)
    vec3 pad = vec3(scale * (u_clip == 1 ? 0.01 : 0.5));
    vec2 tb = boxHit(ro, rd, u_bmin - pad, u_bmax + pad);
    float t = max(tb.x, 0.0), tEnd = min(tb.y, scale * 20.0);
    bool hit = false;
    int id = 0;
    float eps = 1e-4 * scale;
    int i;
    for (i = 0; i < 400 && t < tEnd; i++) {
        float d = scene(ro + rd * t, id);
        if (abs(d) < eps * max(1.0, t / scale)) { hit = true; break; }
        t += d * u_step;
    }

    vec3 light = normalize(vec3(0.6, -0.4, 0.9));
    if (hit && t < tPlane + eps) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p, eps * 2.0);
        vec3 base = id == 1 ? vec3(0.85, 0.25, 0.22) : vec3(0.95, 0.62, 0.22);
        if (id == 1) { // section face: show internal distance field as bands
            float di = map(p);
            base *= 0.75 + 0.25 * smoothstep(-0.1, 0.1, sin(di * 6.2831));
        }
        float dif = clamp(dot(n, light), 0.0, 1.0);
        float sh = softShadow(p + n * eps * 4.0, light, scale * 2.0);
        float occ = ao(p, n, scale * 0.01);
        float back = clamp(dot(n, -light), 0.0, 1.0);
        float rim = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 3.0);
        vec3 h = normalize(light - rd);
        float spe = pow(clamp(dot(n, h), 0.0, 1.0), 48.0) * dif * sh;
        float sky = (0.20 + 0.10 * n.z) * mix(0.35, 1.0, occ);      // hemisphere ambient
        float bounce = 0.10 * clamp(-n.z, 0.0, 1.0);                   // light bounced off the bed
        col = base * (dif * sh + sky + bounce + 0.15 * back * occ) + 0.25 * rim * occ + 0.3 * spe;
    } else if (tPlane < 1e8) {
        vec3 p = ro + rd * tPlane;
        float sh = softShadow(p + vec3(0, 0, eps * 4.0), light, scale * 2.0);
        float minor = gridLine(p.xy, 1.0, 1.0) * 0.25;
        float major = gridLine(p.xy, 10.0, 1.5);
        float axes = 1.0 - smoothstep(0.0, fwidth(p.x) * 2.0, abs(p.x));
        float axesY = 1.0 - smoothstep(0.0, fwidth(p.y) * 2.0, abs(p.y));
        vec3 g = vec3(0.17, 0.18, 0.20);
        g += vec3(0.05) * minor + vec3(0.12) * major;
        g = mix(g, vec3(0.8, 0.25, 0.25), axesY * 0.8);   // x axis (y = 0) red
        g = mix(g, vec3(0.25, 0.75, 0.3), axes * 0.8);    // y axis (x = 0) green
        float fade = exp(-tPlane / (scale * 6.0));
        col = mix(col, g * (0.45 + 0.55 * sh), fade);
    }

    if (u_showBounds == 1) {
        vec2 bh = boxHit(ro, rd, u_bmin, u_bmax);
        if (bh.x < bh.y && bh.y > 0.0) {
            float w = scale * 0.002;
            float tFar = hit ? t : 1e9;
            for (int k = 0; k < 2; k++) {
                float tb2 = k == 0 ? bh.x : bh.y;
                if (tb2 > 0.0 && tb2 < tFar) {
                    float e = edgeLines(ro + rd * tb2, u_bmin, u_bmax, w * max(1.0, tb2 / scale));
                    col = mix(col, vec3(0.4, 0.7, 1.0), e * 0.6);
                }
            }
        }
    }

    col = pow(col, vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
"""


class Viewer:
    def __init__(self, path: Path, defines: dict[str, str]):
        self.path = path
        self.defines = defines
        self.shape: Shape | None = None
        self.prog = None
        self.vao = None
        self.mtimes: dict[Path, float] = {}
        self.error = False
        self.yaw, self.pitch, self.dist = math.radians(-60), math.radians(28), 1.0
        self.target = [0.0, 0.0, 0.0]
        self.drag = None
        self.clip = True
        self.section = False
        self.cut = 0.0
        self.show_bounds = True
        self.busy = False

    # ---------------------------------------------------------------- shader
    def _mt(self, p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    def reload(self, first=False):
        try:
            shape = load_shape(self.path, self.defines)
        except ShaderError as e:
            self._fail(str(e))
            self.mtimes = {self.path: self._mt(self.path)}
            return
        self.mtimes = {f: self._mt(f) for f in shape.files}
        try:
            prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG_HEAD + shape.source + FRAG_TAIL)
        except Exception as e:  # noqa: BLE001
            self._fail(map_errors(str(e), shape))
            return
        if self.prog:
            self.vao.release()
            self.prog.release()
        self.prog = prog
        self.vao = self.ctx.vertex_array(prog, [])
        bounds_changed = self.shape is None or self.shape.bounds != shape.bounds
        self.shape = shape
        self.error = False
        if first or bounds_changed:
            self.reset_camera()
        print(f"[{time.strftime('%H:%M:%S')}] loaded {self.path.name}")

    def _fail(self, msg: str):
        self.error = True
        print(f"\n[{time.strftime('%H:%M:%S')}] \033[31mshader error\033[0m\n{msg.strip()}\n", flush=True)

    def reset_camera(self):
        lo, hi = self.shape.bounds
        self.target = [(a + b) / 2 for a, b in zip(lo, hi)]
        self.dist = 1.6 * math.dist(lo, hi)
        self.yaw, self.pitch = math.radians(-60), math.radians(28)
        self.cut = self.target[1]

    def eye(self):
        cp = math.cos(self.pitch)
        return [self.target[0] + self.dist * cp * math.cos(self.yaw),
                self.target[1] + self.dist * cp * math.sin(self.yaw),
                self.target[2] + self.dist * math.sin(self.pitch)]

    # ---------------------------------------------------------------- input
    def on_key(self, win, key, sc, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        step = (self.shape.size[1] / 50) if self.shape else 1.0
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_R:
            self.reset_camera()
        elif key == glfw.KEY_C:
            self.section = not self.section
        elif key == glfw.KEY_LEFT_BRACKET:
            self.cut -= step
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.cut += step
        elif key == glfw.KEY_B:
            self.clip = not self.clip
        elif key == glfw.KEY_V:
            self.show_bounds = not self.show_bounds
        elif key == glfw.KEY_F5:
            self.reload()
        elif key == glfw.KEY_E:
            self.run_bg(["export", str(self.path)])
        elif key == glfw.KEY_G:
            self.run_bg(["slice", str(self.path)])
        elif key == glfw.KEY_O:
            self.run_bg(["open", str(self.path)])
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            views = {glfw.KEY_1: (-90, 0), glfw.KEY_2: (0, 0), glfw.KEY_3: (-90, 89.9), glfw.KEY_4: (-60, 28)}
            y, p = views[key]
            self.yaw, self.pitch = math.radians(y), math.radians(p)

    def run_bg(self, args):
        if self.busy:
            print("busy, wait for the previous job")
            return
        cmd = [sys.executable, "-m", "gen3d", *args]
        for k, v in self.defines.items():
            cmd += ["-D", f"{k}={v}"]

        def job():
            self.busy = True
            try:
                subprocess.run(cmd)
            finally:
                self.busy = False

        threading.Thread(target=job, daemon=True).start()

    def on_button(self, win, button, action, mods):
        if action == glfw.PRESS:
            self.drag = (button, *glfw.get_cursor_pos(win))
        else:
            self.drag = None

    def on_move(self, win, x, y):
        if not self.drag:
            return
        button, px, py = self.drag
        dx, dy = x - px, y - py
        self.drag = (button, x, y)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.yaw -= dx * 0.008
            self.pitch = max(-1.55, min(1.55, self.pitch + dy * 0.008))
        else:  # pan
            _, h = glfw.get_window_size(win)
            s = self.dist / max(h, 1) * 1.1
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            right = (-sy, cy)
            self.target[0] -= right[0] * dx * s
            self.target[1] -= right[1] * dx * s
            self.target[2] += dy * s

    def on_scroll(self, win, dx, dy):
        self.dist *= 0.9 ** dy

    # ---------------------------------------------------------------- main loop
    def run(self):
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 0)
        win = glfw.create_window(1280, 800, f"gen3d — {self.path.name}", None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("could not open window")
        glfw.make_context_current(win)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context()
        glfw.set_key_callback(win, self.on_key)
        glfw.set_mouse_button_callback(win, self.on_button)
        glfw.set_cursor_pos_callback(win, self.on_move)
        glfw.set_scroll_callback(win, self.on_scroll)
        print(HELP)
        print(f"GL: {self.ctx.info['GL_RENDERER']}")
        self.reload(first=True)
        if self.shape is None:  # first load failed; show bounds of a default shape
            self.shape = Shape(path=self.path, source="", files=[self.path])
            self.reset_camera()

        last_poll = 0.0
        frames, fps_t = 0, time.perf_counter()
        while not glfw.window_should_close(win):
            glfw.poll_events()
            now = time.perf_counter()
            if now - last_poll > 0.25:
                last_poll = now
                if any(self._mt(f) != m for f, m in self.mtimes.items()):
                    time.sleep(0.05)  # let editors finish writing
                    self.reload()

            w, h = glfw.get_framebuffer_size(win)
            self.ctx.viewport = (0, 0, w, h)
            if self.prog is None:
                self.ctx.clear(0.3, 0.05, 0.05)
            else:
                lo, hi = self.shape.bounds
                u = {
                    "u_res": (float(w), float(h)),
                    "u_eye": tuple(self.eye()),
                    "u_target": tuple(self.target),
                    "u_bmin": tuple(map(float, lo)),
                    "u_bmax": tuple(map(float, hi)),
                    "u_step": float(self.shape.directives.get("step", 0.9)),
                    "u_clip": int(self.clip),
                    "u_section": int(self.section),
                    "u_cut": float(self.cut),
                    "u_showBounds": int(self.show_bounds),
                }
                for k, v in u.items():
                    if k in self.prog:
                        self.prog[k].value = v
                self.vao.render(moderngl.TRIANGLES, vertices=3)
            glfw.swap_buffers(win)

            frames += 1
            if now - fps_t > 0.5:
                status = "ERROR (see terminal)" if self.error else f"{frames / (now - fps_t):.0f} fps"
                flags = []
                if self.section:
                    flags.append(f"section y={self.cut:.1f}")
                if not self.clip:
                    flags.append("unclipped")
                if self.busy:
                    flags.append("working…")
                glfw.set_window_title(win, f"gen3d — {self.path.name} — {status} {' '.join(flags)}")
                frames, fps_t = 0, now
        glfw.terminate()


HELP = """gen3d viewer — edit the .glsl file in any editor; it reloads on save.
  mouse: left-drag orbit · right/middle-drag pan · wheel zoom
  1/2/3/4 front/side/top/iso   R reset camera   F5 force reload
  C section view   [ ] move section plane   B toggle clip-to-bounds   V toggle bounds box
  E export mesh    G slice to G-code        O open in slicer GUI      Q/Esc quit
"""


def view(path: Path, defines: dict[str, str]):
    Viewer(path.resolve(), defines).run()


def render_png(path: Path, defines: dict[str, str], out: Path, size=(1024, 768), section=False,
               yaw=-60.0, pitch=28.0):
    """Render the preview offscreen (same shader as the viewer) to a PNG."""
    import imageio.v3 as iio
    import numpy as np

    from .mesher import create_context

    shape = load_shape(path, defines)
    ctx = create_context()
    try:
        prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_HEAD + shape.source + FRAG_TAIL)
    except Exception as e:  # noqa: BLE001
        raise ShaderError(map_errors(str(e), shape)) from None
    w, h = size
    fbo = ctx.simple_framebuffer((w, h), components=4)
    fbo.use()
    v = Viewer(path, defines)
    v.shape = shape
    v.reset_camera()
    v.yaw, v.pitch = math.radians(yaw), math.radians(pitch)
    u = {
        "u_res": (float(w), float(h)), "u_eye": tuple(v.eye()), "u_target": tuple(v.target),
        "u_bmin": tuple(map(float, shape.bounds[0])), "u_bmax": tuple(map(float, shape.bounds[1])),
        "u_step": float(shape.directives.get("step", 0.9)), "u_clip": 1, "u_section": int(section),
        "u_cut": float(v.cut), "u_showBounds": 1,
    }
    for k, val in u.items():
        if k in prog:
            prog[k].value = val
    ctx.vertex_array(prog, []).render(moderngl.TRIANGLES, vertices=3)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)[::-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, img)
    ctx.release()
    return out
