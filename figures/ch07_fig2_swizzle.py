"""
Ch07 Fig2: Shared-memory bank conflicts without swizzle vs XOR swizzle spreading accesses.
Without swizzle: multiple lanes map to the same bank (serialized). With swizzle: remapped indices hit distinct banks.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class SwizzleBanks(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Bank conflicts: no swizzle vs XOR swizzle",
            font_size=24,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        n_banks = 8
        bank_w = 0.52
        bank_h = 1.35

        def bank_row(y_center, label_above, conflict: bool):
            banks = VGroup()
            for b in range(n_banks):
                col = C["red"] if (conflict and b == 0) else C["grid_c"]
                stroke = C["red"] if (conflict and b == 0) else C["stroke"]
                r = Rectangle(
                    width=bank_w,
                    height=bank_h,
                    fill_color=col,
                    fill_opacity=0.35 if not (conflict and b == 0) else 0.45,
                    stroke_color=stroke,
                    stroke_width=2 if (conflict and b == 0) else 1.2,
                )
                r.move_to(LEFT * (n_banks * bank_w / 2 - bank_w / 2) + RIGHT * b * bank_w)
                r.shift(UP * y_center)
                num = Text(str(b), font_size=12, color=C["fg2"], font="Monospace")
                num.move_to(r.get_bottom() + UP * 0.22)
                banks.add(VGroup(r, num))
            lab = Text(label_above, font_size=15, color=C["fg"], font="Monospace")
            lab.move_to(UP * (y_center + bank_h / 2 + 0.42))
            return VGroup(lab, banks)

        # --- Top: conflict ---
        row1 = bank_row(1.25, "Without swizzle: warp lanes collide on one bank", True)
        self.add(row1[0])

        banks1 = row1[1]
        lane_y = 1.25 + bank_h / 2 + 0.55
        lanes = VGroup()
        for i in range(4):
            dot = Dot(
                point=LEFT * 1.8 + RIGHT * i * 0.35 + UP * lane_y,
                radius=0.09,
                color=C["orange"],
            )
            lbl = Text(f"L{i}", font_size=11, color=C["orange"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.06)
            arr = Arrow(
                dot.get_center(),
                banks1[0][0].get_top() + DOWN * 0.05 + RIGHT * 0.1 * i,
                buff=0.06,
                stroke_width=2,
                color=C["orange"],
                max_tip_length_to_length_ratio=0.18,
            )
            lanes.add(VGroup(dot, lbl, arr))
        self.add(banks1, lanes)

        clash = Text(
            "same bank → serialized accesses (bandwidth / N)",
            font_size=13,
            color=C["red"],
            font="Monospace",
        )
        clash.move_to(UP * 0.15)
        self.add(clash)

        # --- Bottom: swizzled ---
        row2 = bank_row(-1.55, "With swizzle (e.g. XOR remap): lanes map to distinct banks", False)
        self.add(row2[0])

        banks2 = row2[1]
        lane_y2 = -1.55 + bank_h / 2 + 0.55
        map_to = [0, 3, 2, 5]  # illustrative spread after XOR-style remap
        lanes2 = VGroup()
        for i in range(4):
            bi = map_to[i]
            dot = Dot(
                point=LEFT * 1.8 + RIGHT * i * 0.35 + UP * lane_y2,
                radius=0.09,
                color=C["green"],
            )
            lbl = Text(f"L{i}", font_size=11, color=C["green"], font="Monospace")
            lbl.next_to(dot, UP, buff=0.06)
            tgt = banks2[bi][0].get_top() + DOWN * 0.08
            arr = Arrow(
                dot.get_center(),
                tgt,
                buff=0.08,
                stroke_width=2,
                color=C["green"],
                max_tip_length_to_length_ratio=0.16,
            )
            lanes2.add(VGroup(dot, lbl, arr))
        self.add(banks2, lanes2)

        ok = Text(
            "conflict-free warp (full bandwidth)",
            font_size=13,
            color=C["green_ok"],
            font="Monospace",
        )
        ok.move_to(DOWN * 2.65)
        self.add(ok)

        foot = Text(
            "tma.copy.swiz<N> + mma.load.swiz<N> must match — layout is XOR-remapped in shared memory",
            font_size=11,
            color=C["fg3"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.32)
        self.add(foot)
