"""统一可视化样式与调色工具。"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_global_style() -> None:
    """项目级统一绘图风格。"""
    plt.rcParams.update(
        {
            "figure.figsize": (11, 7),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def soft_track_colors(track_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """柔和色板：按轨迹分配颜色。"""
    cmap = plt.get_cmap("Set2")
    return {tid: cmap(i % 8) for i, tid in enumerate(track_ids)}  # type: ignore[return-value]
