#!/usr/bin/env python3
"""快速验证低速强回旋机制：F_n/C*_eff vs speed（旧入口）。

说明：
- 该脚本仍可用，但建议优先使用 `src/cmd/quickViz.py` 作为统一快速验证入口。

用法示例：
  python3 src/cmd/plotAll.py data/final/track1opt.csv
  python3 src/cmd/plotAll.py data/final/track1opt.csv data/final/track2opt.csv --save --out out
  python3 src/cmd/plotAll.py "data/final/track*opt.csv" --save --out out
"""

from __future__ import annotations

import argparse
import sys
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataIO import load_track
from src.visualization.oopVisualization import (
    ForceAnalysisVisualizer,
    TimeSeriesVisualizer,
    TrackDataWrapper,
)


def _expand_track_args(track_args: list[str]) -> list[Path]:
    """Expand user-provided paths, supporting quoted glob patterns."""
    paths: list[Path] = []
    for arg in track_args:
        # If user passes a glob pattern in quotes, the shell won't expand it.
        if any(ch in arg for ch in ("*", "?", "[")):
            matches = sorted(glob.glob(arg))
            if not matches:
                raise FileNotFoundError(f"glob 没匹配到任何文件: {arg}")
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(arg))
    return paths


def _to_dataframe(track_obj) -> pd.DataFrame:
    """Convert load_track output (dict of arrays) into a DataFrame."""
    if isinstance(track_obj, pd.DataFrame):
        return track_obj
    if isinstance(track_obj, dict):
        return pd.DataFrame(track_obj)
    raise TypeError(f"不支持的轨迹数据类型: {type(track_obj)}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boomerang OOP plots (low-speed turning validation)")
    p.add_argument(
        "tracks",
        nargs="+",
        help="一个或多个轨迹CSV路径（建议 data/final/*opt.csv）。可用引号传glob。",
    )
    p.add_argument("--title", default="Low-Speed Turning Mechanism", help="总标题前缀")
    p.add_argument("--out", default="out", help="保存目录（配合 --save）")
    p.add_argument("--save", action="store_true", help="保存图片到 --out")
    p.add_argument("--no-show", action="store_true", help="不弹窗显示（适合批量跑）")
    p.add_argument(
        "--vel-components",
        default="vz",
        help="要画的速度分量，逗号分隔：vx,vy,vz（默认只画vz辅助对齐最高点附近）",
    )
    return p.parse_args()


def main() -> int:
    print("[plotAll] 提示：该入口为兼容保留，推荐改用 src/cmd/quickViz.py")
    args = _parse_args()
    out_dir = Path(args.out)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load tracks
    track_data: dict[str, TrackDataWrapper] = {}
    track_paths = _expand_track_args([str(p) for p in args.tracks])
    for path in track_paths:
        track_dict = load_track(str(path))
        df = _to_dataframe(track_dict)
        required_cols = {"t", "vx", "vy", "vz", "ax", "ay", "az"}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(
                f"{path}: 缺少必要列 {missing}。"
                "建议输入 data/final/*opt.csv（包含 v/a 列）"
            )

        name = path.stem
        track_data[name] = TrackDataWrapper(df, track_name=name)

    # 1) 力分解 + C*_eff（含 optimal v_rot）
    force_viz = ForceAnalysisVisualizer(title=f"{args.title} - Forces/Ceff")
    fig1 = force_viz.plot_aerodynamic_forces_decomposition(track_data)
    if args.save:
        fig1.savefig(out_dir / "forces_decomposition.png", dpi=160)
        plt.close(fig1)

    # 2) C*_eff vs speed（核心判据：低速是否爆炸，v_rot 是否能压平）
    fig2 = force_viz.plot_effective_coefficients_vs_speed(track_data)
    if args.save:
        fig2.savefig(out_dir / "coeffs_vs_speed.png", dpi=160)
        plt.close(fig2)

    # 2.5) 闭环检验：需求向心力 vs 反推 Fn
    fig25 = force_viz.plot_centripetal_force_closure(track_data)
    if args.save:
        fig25.savefig(out_dir / "centripetal_closure.png", dpi=160)
        plt.close(fig25)

    # 3) 辅助对齐：速度分量（默认 vz）
    comps = [c.strip() for c in str(args.vel_components).split(",") if c.strip()]
    time_viz = TimeSeriesVisualizer(title=f"{args.title} - Vel components")
    fig3 = time_viz.plot_velocity_components(track_data, components=comps)
    if args.save:
        fig3.savefig(out_dir / "velocity_components.png", dpi=160)
        plt.close(fig3)

    if not args.no_show and not args.save:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())