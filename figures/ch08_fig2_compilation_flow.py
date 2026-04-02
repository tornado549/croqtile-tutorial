"""
Figure 2 (ch08): Compilation flow of a .co file.
Shows how the Croqtile compiler splits a .co file into host C++ and device CUDA,
and where __device__, __co__, and host code each live.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class CompilationFlow(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(".co file compilation flow", font_size=20, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        box_h = 0.7
        box_w = 2.8

        def make_box(label, sub, x, y, col, w=box_w):
            r = RoundedRectangle(width=w, height=box_h, corner_radius=0.1,
                                  fill_color=col, fill_opacity=0.2,
                                  stroke_color=col, stroke_width=1.5)
            r.move_to(RIGHT * x + UP * y)
            self.add(r)
            t = Text(label, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r.get_center() + UP * 0.12)
            self.add(t)
            if sub:
                s = Text(sub, font_size=9, color=C["dim"], font="Monospace")
                s.move_to(r.get_center() + DOWN * 0.15)
                self.add(s)
            return r

        def make_arrow(start, end, col=C["arrow"]):
            a = Arrow(start, end, color=col, stroke_width=1.8, buff=0.08,
                      max_tip_length_to_length_ratio=0.15)
            self.add(a)

        # Source .co file (top)
        co_box = make_box("kernel.co", "__co__ + __device__ + host C++", 0, 2.0, C["fg2"])

        # Croqtile compiler (middle)
        compiler = make_box("croqtile compiler", "parse -> transform -> codegen", 0, 0.8, C["green"])
        make_arrow(co_box.get_bottom(), compiler.get_top())

        # Two outputs
        host_box = make_box("Host C++", "main(), launch config", -2.8, -0.6, C["blue"])
        device_box = make_box("Device CUDA", "__co__ -> __global__", 2.8, -0.6, C["orange"])
        make_arrow(compiler.get_bottom() + LEFT * 0.5, host_box.get_top())
        make_arrow(compiler.get_bottom() + RIGHT * 0.5, device_box.get_top())

        # __device__ passthrough annotation
        dev_pass = Text("__device__ passed through as-is", font_size=9, color=C["purple"], font="Monospace")
        dev_pass.move_to(RIGHT * 2.8 + DOWN * 1.2)
        self.add(dev_pass)

        # nvcc
        nvcc_box = make_box("nvcc / clang", "compile + link", 0, -2.0, C["fg3"])
        make_arrow(host_box.get_bottom(), nvcc_box.get_top() + LEFT * 0.8)
        make_arrow(device_box.get_bottom(), nvcc_box.get_top() + RIGHT * 0.8)

        # Binary
        bin_box = make_box("GPU binary", "host + device code", 0, -3.2, C["green"], w=2.2)
        make_arrow(nvcc_box.get_bottom(), bin_box.get_top())

        # Annotations for what goes where
        co_ann = Text("__co__ fn()  ->  generated __global__ kernel", font_size=9, color=C["orange"], font="Monospace")
        co_ann.move_to(RIGHT * 2.8 + UP * 0.1)
        self.add(co_ann)

        call_ann = Text("call device_fn()  ->  calls __device__ directly", font_size=9, color=C["purple"], font="Monospace")
        call_ann.move_to(RIGHT * 2.8 + DOWN * 0.15)
        self.add(call_ann)
