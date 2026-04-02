"""
Figure 1: Croktile DSL wrapping a small __cpp__ island — verbatim injection point
for PTX / raw C++ below the choreographed abstraction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class Ch08Fig1EscapeHatch(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Escape hatch: __cpp__ island inside generated CUDA",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        sub = Text(
            "DSL orchestrates; a raw string splices verbatim at the splice point",
            font_size=13,
            color=C["fg2"],
            font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.12)
        self.add(sub)

        # Outer: Croktile layer
        outer = RoundedRectangle(
            corner_radius=0.15,
            width=10.2,
            height=3.6,
            color=C["green"],
            stroke_width=2.5,
            fill_color=C["fill"],
            fill_opacity=0.55,
        )
        outer.shift(DOWN * 0.15)
        outer_lbl = Text(
            "Croktile (__co__, parallel, foreach, dma, mma, …)",
            font_size=14,
            color=C["green"],
            font="Monospace",
        )
        outer_lbl.next_to(outer, UP, buff=0.18)
        self.add(outer, outer_lbl)

        # Abstract "lines" of DSL (decorative)
        for i, y in enumerate([0.55, 0.15, -0.25]):
            line_w = 3.8 - i * 0.4
            dash = Line(
                LEFT * line_w / 2,
                RIGHT * line_w / 2,
                color=C["dim"],
                stroke_width=2,
                stroke_opacity=0.7,
            )
            dash.move_to(LEFT * 2.2 + DOWN * 0.15 + UP * y)
            self.add(dash)

        dsl_note = Text(
            "structured kernel body",
            font_size=11,
            color=C["fg3"],
            font="Monospace",
        )
        dsl_note.move_to(LEFT * 2.5 + DOWN * 0.15 + DOWN * 0.85)
        self.add(dsl_note)

        # Inner: __cpp__ island
        inner = RoundedRectangle(
            corner_radius=0.12,
            width=4.2,
            height=1.55,
            color=C["orange"],
            stroke_width=2.5,
            fill_color=C["gray_bg"],
            fill_opacity=0.9,
        )
        inner.move_to(RIGHT * 2.35 + DOWN * 0.15 + UP * 0.05)
        inner_title = Text("__cpp__(R\"( … )\")", font_size=13, color=C["orange"], font="Monospace")
        inner_title.next_to(inner, UP, buff=0.12)
        self.add(inner, inner_title)

        asm_lines = [
            'asm volatile("setmaxnreg.',
            '  dec.sync.aligned.u32 40;");',
        ]
        for j, ln in enumerate(asm_lines):
            t = Text(ln, font_size=11, color=C["cmd_c"], font="Monospace")
            t.move_to(inner.get_center() + UP * 0.28 + DOWN * j * 0.38)
            self.add(t)

        island_note = Text(
            "PTX / C++ — not parsed by Croktile",
            font_size=11,
            color=C["orange"],
            font="Monospace",
        )
        island_note.next_to(inner, DOWN, buff=0.18)
        self.add(island_note)

        # Arrow from outer concept to inner
        arr = Arrow(
            start=LEFT * 0.35 + DOWN * 0.15 + UP * 0.1,
            end=inner.get_left() + LEFT * 0.05 + UP * 0.05,
            color=C["arrow_c"],
            stroke_width=3,
            buff=0.08,
        )
        self.add(arr)

        foot = Text(
            "Compiler pastes the string; names inside must match generated C++",
            font_size=11,
            color=C["dim"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.4)
        self.add(foot)
