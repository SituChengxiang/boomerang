#!/usr/bin/env python3
"""快速验证/配图入口（内置开关版）。

目标：
1) 自动匹配 raw / SMR / opt 三类轨迹；
2) 统一柔和配色（按轨迹）+ 线型区分（同轨迹不同物理量）；
3) 一次性创建所有图，再统一弹窗。
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.aerodynamics import (
    calculate_classic_coefficients,
    calculate_net_aerodynamic_force,
    decompose_aerodynamic_force,
    find_optimal_v_rot,
)
from src.utils.kinematics import analyze_track, compute_centripetal_force_demands
from src.utils.mathUtils import derivatives_smooth, linear_fit_with_ci
from src.utils.physicsCons import MASS
from src.visualization.style import apply_global_style, soft_track_colors

SHOW_WINDOWS = True
SAVE_FIGURES = True
OUTPUT_DIR = PROJECT_ROOT / "out" / "quick_viz"

# 轨迹来源（默认使用 final 的 opt 轨迹集合作为主集合）
OPT_GLOB = str(PROJECT_ROOT / "data" / "final" / "track*opt.csv")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SMR_DIR = PROJECT_ROOT / "data" / "interm"

# 可选：限制轨迹数量，None 表示全部
MAX_TRACKS: int | None = None


def _track_id_from_stem(stem: str) -> str:
    s = stem
    if s.endswith("opt"):
        s = s[:-3]
    if s.endswith("SMR"):
        s = s[:-3]
    return s


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "t" not in df.columns:
        raise ValueError(f"{path}: 缺少 t 列")
    df = df.sort_values("t").reset_index(drop=True)
    return df


def _ensure_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """若缺少 v/a 列，则由 x,y,z 推导；已有则保持不变。"""
    out = df.copy()
    t = np.asarray(out["t"].values, dtype=float)

    need_v = any(c not in out.columns for c in ("vx", "vy", "vz"))
    need_a = any(c not in out.columns for c in ("ax", "ay", "az"))

    if (need_v or need_a) and any(c not in out.columns for c in ("x", "y", "z")):
        missing = [c for c in ("x", "y", "z") if c not in out.columns]
        raise ValueError(f"缺少列 {missing}，无法推导速度/加速度")

    if need_v:
        x = np.asarray(out["x"].values, dtype=float)
        y = np.asarray(out["y"].values, dtype=float)
        z = np.asarray(out["z"].values, dtype=float)
        vx, _ = derivatives_smooth(t, x)
        vy, _ = derivatives_smooth(t, y)
        vz, _ = derivatives_smooth(t, z)
        out["vx"] = vx
        out["vy"] = vy
        out["vz"] = vz

    if need_a:
        vx = np.asarray(out["vx"].values, dtype=float)
        vy = np.asarray(out["vy"].values, dtype=float)
        vz = np.asarray(out["vz"].values, dtype=float)
        ax, _ = derivatives_smooth(t, vx)
        ay, _ = derivatives_smooth(t, vy)
        az, _ = derivatives_smooth(t, vz)
        out["ax"] = ax
        out["ay"] = ay
        out["az"] = az

    if "speed" not in out.columns:
        vx = np.asarray(out["vx"].values, dtype=float)
        vy = np.asarray(out["vy"].values, dtype=float)
        vz = np.asarray(out["vz"].values, dtype=float)
        out["speed"] = np.sqrt(vx**2 + vy**2 + vz**2)

    return out


def _pair_tracks() -> dict[str, dict[str, Path]]:
    """基于 final/track*opt.csv 自动构建 raw/smr/opt 配对。"""
    opt_paths = sorted(Path(p) for p in glob.glob(OPT_GLOB))
    if not opt_paths:
        raise FileNotFoundError(f"没有找到轨迹：{OPT_GLOB}")

    if MAX_TRACKS is not None:
        opt_paths = opt_paths[: max(1, int(MAX_TRACKS))]

    result: dict[str, dict[str, Path]] = {}
    for opt in opt_paths:
        track_id = _track_id_from_stem(opt.stem)
        raw = RAW_DIR / f"{track_id}.csv"
        smr = SMR_DIR / f"{track_id}SMR.csv"
        result[track_id] = {"opt": opt}
        if raw.exists():
            result[track_id]["raw"] = raw
        if smr.exists():
            result[track_id]["smr"] = smr
    return result


def _relative_time(df: pd.DataFrame) -> np.ndarray:
    t = np.asarray(df["t"].values, dtype=float)
    return t - float(t[0])


def _energy_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = _relative_time(df)
    vx = np.asarray(df["vx"].values, dtype=float)
    vy = np.asarray(df["vy"].values, dtype=float)
    vz = np.asarray(df["vz"].values, dtype=float)
    z = np.asarray(df["z"].values, dtype=float)
    e = MASS * 0.5 * (vx**2 + vy**2 + vz**2) + MASS * 9.8 * z
    return t, e


def _compute_aero_core(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """统一气动力计算主干：F_aero, speed, force decomposition。"""
    acc = np.column_stack((df["ax"].values, df["ay"].values, df["az"].values))
    vel = np.column_stack((df["vx"].values, df["vy"].values, df["vz"].values))
    f_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
    f_comp = decompose_aerodynamic_force(f_aero, vel, speed)
    return f_aero, speed, f_comp


def _compute_classic_cl_cd(
    f_aero: np.ndarray,
    speed: np.ndarray,
    f_comp: dict[str, np.ndarray],
    v_rot: float,
) -> tuple[np.ndarray, np.ndarray]:
    """经典口径系数：复用 utils 中的统一计算函数。"""
    coeffs_classic = calculate_classic_coefficients(
        f_aero=f_aero,
        f_components=f_comp,
        speed=speed,
        v_rot=v_rot,
        eps_q=1e-6,
    )
    return coeffs_classic["Cl"], coeffs_classic["Cd"]


def plot_preprocess_need(
    track_id: str,
    raw_df: pd.DataFrame,
    smr_df: pd.DataFrame,
    color: tuple[float, float, float, float],
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图1：说明为何要预处理（速度异常/能量突变）。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    t_raw = _relative_time(raw_df)
    t_smr = _relative_time(smr_df)
    speed_raw = np.asarray(raw_df["speed"].values, dtype=float)
    speed_smr = np.asarray(smr_df["speed"].values, dtype=float)
    _, e_raw = _energy_series(raw_df)
    _, e_smr = _energy_series(smr_df)

    axes[0].plot(t_raw, speed_raw, color=color, linestyle="-", alpha=0.62, label=f"{track_id} raw speed")
    axes[0].plot(t_smr, speed_smr, color=color, linestyle="--", alpha=0.95, label=f"{track_id} SMR speed")
    axes[0].set_title("Why preprocessing is needed: speed stability")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="best")

    axes[1].plot(t_raw, e_raw, color=color, linestyle="-", alpha=0.62, label=f"{track_id} raw E")
    axes[1].plot(t_smr, e_smr, color=color, linestyle="--", alpha=0.95, label=f"{track_id} SMR E")
    axes[1].set_title("Why preprocessing is needed: mechanical energy continuity")
    axes[1].set_xlabel("Time since start (s)")
    axes[1].set_ylabel("Energy (J)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    return fig


def plot_preprocess_comparison_3d_energy(
    track_id: str,
    raw_df: pd.DataFrame,
    smr_df: pd.DataFrame,
    color: tuple[float, float, float, float],
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图2：原始轨迹 vs 平滑轨迹（3D）+ 能量对比。"""
    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    axe = fig.add_subplot(122)

    ax3d.plot(raw_df["x"], raw_df["y"], raw_df["z"], color=color, linestyle="-", alpha=0.55, label=f"{track_id} raw")
    ax3d.plot(smr_df["x"], smr_df["y"], smr_df["z"], color=color, linestyle="--", alpha=0.95, label=f"{track_id} SMR")
    ax3d.set_title("3D trajectory: raw vs SMR")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="best")

    t_raw, e_raw = _energy_series(raw_df)
    t_smr, e_smr = _energy_series(smr_df)
    axe.plot(t_raw, e_raw, color=color, linestyle="-", alpha=0.55, label=f"{track_id} raw E")
    axe.plot(t_smr, e_smr, color=color, linestyle="--", alpha=0.95, label=f"{track_id} SMR E")
    axe.set_title("Energy: raw vs SMR")
    axe.set_xlabel("Time since start (s)")
    axe.set_ylabel("Energy (J)")
    axe.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_low_speed_strong_turning(
    opt_tracks: dict[str, pd.DataFrame],
    colors: dict[str, tuple[float, float, float, float]],
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图3：低速强回旋真实性（三种估计同图）。"""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax_v = ax.twinx()

    for track_id, df in opt_tracks.items():
        data = analyze_track(df, track_id)
        t = np.asarray(data[0], dtype=float)
        t = t - float(t[0])
        a_perp_vec = np.asarray(data[5], dtype=float)
        v_h = np.asarray(data[7], dtype=float)
        a_perp_curv = np.asarray(data[9], dtype=float) if data[9] is not None else np.full_like(v_h, np.nan)
        heading_rate_deg = data[13]
        if heading_rate_deg is None:
            heading_rate_deg = data[14]

        if heading_rate_deg is None:
            a_perp_heading = np.full_like(v_h, np.nan)
        else:
            psi_dot = np.deg2rad(np.asarray(heading_rate_deg, dtype=float))
            a_perp_heading = np.abs(v_h * psi_dot)

        c = colors[track_id]
        ax.plot(t, a_perp_vec, color=c, linestyle="-", alpha=0.9, label=f"{track_id} vector")
        ax.plot(t, a_perp_curv, color=c, linestyle="--", alpha=0.9, label=f"{track_id} curvature")
        ax.plot(t, a_perp_heading, color=c, linestyle=":", alpha=0.95, label=f"{track_id} heading-rate")

        # 右轴叠加速度绝对值（淡色）用于时序对照
        speed_abs = np.asarray(df["speed"].values, dtype=float)
        n = min(len(t), len(speed_abs))
        ax_v.plot(
            t[:n],
            speed_abs[:n],
            color=c,
            linestyle="-.",
            alpha=0.22,
            linewidth=1.0,
            label=f"{track_id} |v|",
        )

    ax.set_title("Low-speed strong turning validation: three estimators + |v|")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(r"$a_{\perp,h}$ (m/s$^2$)")
    ax_v.set_ylabel(r"Speed $|v|$ (m/s)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", ncol=2)
    fig.tight_layout()
    return fig


def plot_centripetal_closure(
    opt_tracks: dict[str, pd.DataFrame],
    colors: dict[str, tuple[float, float, float, float]],
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图4：向心力来源验证（Fn vs m*a_perp）。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax_ts, ax_sc = axes
    x_all = {
        "vector": [],
        "heading": [],
        "curvature": [],
    }
    y_all = {
        "vector": [],
        "heading": [],
        "curvature": [],
    }

    for track_id, df in opt_tracks.items():
        data = analyze_track(df, track_id)
        t = np.asarray(data[0], dtype=float)
        t = t - float(t[0])
        req_dict = compute_centripetal_force_demands(data, MASS)
        req_vec = req_dict["vector"]
        req_heading = req_dict["heading"]
        req_curv = req_dict["curvature"]

        acc = np.column_stack((df["ax"].values, df["ay"].values, df["az"].values))
        vel = np.column_stack((df["vx"].values, df["vy"].values, df["vz"].values))
        f_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
        f_comp = decompose_aerodynamic_force(f_aero, vel, speed)
        fn = np.asarray(f_comp["F_n"], dtype=float)

        c = colors[track_id]
        ax_ts.plot(t, fn, color=c, linestyle="-", alpha=0.9, label=f"{track_id} F_n")
        ax_ts.plot(t, req_curv, color=c, linestyle="--", alpha=0.92, label=f"{track_id} m*a_perp(curv)")
        ax_ts.plot(t, req_heading, color=c, linestyle=":", alpha=0.40, label=f"{track_id} m*a_perp(heading)")

        for key, req in (("vector", req_vec), ("heading", req_heading), ("curvature", req_curv)):
            mask = np.isfinite(fn) & np.isfinite(req)
            x_all[key].append(req[mask])
            y_all[key].append(fn[mask])

    ax_ts.set_title("Centripetal force closure in horizontal plane")
    ax_ts.set_xlabel("Time since start (s)")
    ax_ts.set_ylabel("Force (N)")
    ax_ts.legend(loc="best", ncol=2)

    ax_sc.set_title(r"Scatter closure: $F_n$ vs $m a_{\perp,h}$ (vector/heading/curvature)")
    ax_sc.set_xlabel(r"$m a_{\perp,h}$ (N)")
    ax_sc.set_ylabel(r"$F_n$ (N)")

    style = {
        "vector": ("tab:orange", 0.20, "o"),
        "heading": ("tab:green", 0.24, "^"),
        "curvature": ("tab:blue", 0.30, "s"),
    }

    combined_vals = []
    for key in ("vector", "heading", "curvature"):
        if x_all[key] and y_all[key]:
            x_cat = np.concatenate(x_all[key])
            y_cat = np.concatenate(y_all[key])
            mask = np.isfinite(x_cat) & np.isfinite(y_cat)
            if np.any(mask):
                combined_vals.append(x_cat[mask])
                combined_vals.append(y_cat[mask])

    if combined_vals:
        allv = np.concatenate(combined_vals)
        lo = float(np.nanpercentile(allv, 2))
        hi = float(np.nanpercentile(allv, 98))
        ax_sc.plot([lo, hi], [lo, hi], "k--", alpha=0.35, label="y=x")
        ax_sc.set_xlim(lo, hi)
        ax_sc.set_ylim(lo, hi)

        summary_lines = []
        print("[centripetal_closure] 线性回归结果（Fn vs m*a_perp）")
        for key in ("vector", "heading", "curvature"):
            if not x_all[key] or not y_all[key]:
                continue
            x_cat = np.concatenate(x_all[key])
            y_cat = np.concatenate(y_all[key])
            mask = np.isfinite(x_cat) & np.isfinite(y_cat)
            if not np.any(mask):
                continue

            c, a, mkr = style[key]
            ax_sc.scatter(x_cat[mask], y_cat[mask], s=12, color=c, alpha=a, marker=mkr, label=key)

            stats = linear_fit_with_ci(x_cat[mask], y_cat[mask], alpha=0.05)
            if stats is None:
                continue

            k = stats["k"]
            b = stats["b"]
            ax_sc.plot([lo, hi], [k * lo + b, k * hi + b], color=c, alpha=0.75, linewidth=1.4)
            summary_lines.append(
                f"{key}: k={k:.3f}, b={b:.2e}, R²={stats['r2']:.3f}, n={int(stats['n'])}"
            )

            print(f"  [{key}]")
            print(
                f"    k = {k:.6f} ± {stats['se_k']:.6f} (95%CI: [{stats['k_lo']:.6f}, {stats['k_hi']:.6f}])"
            )
            print(
                f"    b = {b:.6e} ± {stats['se_b']:.6e} N (95%CI: [{stats['b_lo']:.6e}, {stats['b_hi']:.6e}])"
            )
            print(f"    R² = {stats['r2']:.6f}, n = {int(stats['n'])}, dof = {int(stats['dof'])}")

        if summary_lines:
            ax_sc.text(
                0.03,
                0.97,
                "\n".join(summary_lines),
                transform=ax_sc.transAxes,
                va="top",
                ha="left",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
            )
        ax_sc.legend(loc="best", ncol=2)

    fig.tight_layout()
    return fig


def plot_coeffs_vs_speed_no_rot(
    opt_tracks: dict[str, pd.DataFrame],
    colors: dict[str, tuple[float, float, float, float]],
    min_speed: float = 0.2,
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图5a：旋转动压修正前（Cl/Cd vs speed）。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for track_id, df in opt_tracks.items():
        f_aero, speed, f_comp = _compute_aero_core(df)
        cl, cd = _compute_classic_cl_cd(f_aero, speed, f_comp, v_rot=0.0)

        mask = np.isfinite(speed) & (speed >= float(min_speed))
        sp = speed[mask]
        c = colors[track_id]

        for idx, (ax, key) in enumerate(
            ((axes[0], "Cl"), (axes[1], "Cd"))
        ):
            y = cl[mask] if key == "Cl" else cd[mask]
            label = track_id if idx == 0 else "_nolegend_"
            ax.scatter(sp, y, s=10, alpha=0.36, color=c, label=label)

    axes[0].set_title("Aerodynamic coefficients vs speed (no rotation correction)")
    axes[0].set_ylabel("Cl")
    axes[1].set_ylabel("Cd")
    axes[1].set_xlabel("Speed |v| (m/s)")
    axes[0].legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig


def plot_coeffs_vs_speed_with_rot(
    opt_tracks: dict[str, pd.DataFrame],
    colors: dict[str, tuple[float, float, float, float]],
    min_speed: float = 0.2,
) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
    """图5b：旋转动压修正后（Cl/Cd vs speed）。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for track_id, df in opt_tracks.items():
        f_aero, speed, f_comp = _compute_aero_core(df)
        vrot_opt, _ = find_optimal_v_rot(f_comp, speed)
        cl, cd = _compute_classic_cl_cd(f_aero, speed, f_comp, v_rot=vrot_opt)

        mask = np.isfinite(speed) & (speed >= float(min_speed))
        sp = speed[mask]
        c = colors[track_id]

        for idx, (ax, key) in enumerate(
            ((axes[0], "Cl"), (axes[1], "Cd"))
        ):
            y = cl[mask] if key == "Cl" else cd[mask]
            label = f"{track_id} (v_rot={vrot_opt:.2f})" if idx == 0 else "_nolegend_"
            ax.scatter(sp, y, s=10, alpha=0.36, color=c, label=label)

    axes[0].set_title("areodynamic coefficients vs speed (with optimal rotation correction)")
    axes[0].set_ylabel("Cl")
    axes[1].set_ylabel("Cd")
    axes[1].set_xlabel("Speed |v| (m/s)")
    axes[0].legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig


def _save(figs: list[tuple[str, plt.Figure]]) -> None: # pyright: ignore[reportPrivateImportUsage]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in figs:
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=170)


def main() -> int:
    apply_global_style()

    track_map = _pair_tracks()
    track_ids = sorted(track_map.keys())
    if not track_ids:
        print("没有可用轨迹")
        return 1

    # 读取 opt（主分析集）
    opt_tracks: dict[str, pd.DataFrame] = {}
    for tid in track_ids:
        opt_df = _ensure_kinematics(_load_csv(track_map[tid]["opt"]))
        opt_tracks[tid] = opt_df

    colors = soft_track_colors(track_ids)

    # 选择一条同时具备 raw+smr 的轨迹做预处理展示图
    preprocess_track: str | None = None
    for tid in track_ids:
        if "raw" in track_map[tid] and "smr" in track_map[tid]:
            preprocess_track = tid
            break

    figs: list[tuple[str, plt.Figure]] = [] # pyright: ignore[reportPrivateImportUsage]

    if preprocess_track is not None:
        raw_df = _ensure_kinematics(_load_csv(track_map[preprocess_track]["raw"]))
        smr_df = _ensure_kinematics(_load_csv(track_map[preprocess_track]["smr"]))
        c0 = colors[preprocess_track]
        figs.append(
            (
                "01_preprocess_need",
                plot_preprocess_need(preprocess_track, raw_df, smr_df, c0),
            )
        )
        figs.append(
            (
                "02_preprocess_3d_energy",
                plot_preprocess_comparison_3d_energy(preprocess_track, raw_df, smr_df, c0),
            )
        )
    else:
        print("警告：未找到同时存在 raw 与 SMR 的轨迹，跳过图1和图2")

    figs.append(("03_low_speed_strong_turning", plot_low_speed_strong_turning(opt_tracks, colors)))
    figs.append(("04_centripetal_closure", plot_centripetal_closure(opt_tracks, colors)))
    figs.append(("05a_coeff_no_rot", plot_coeffs_vs_speed_no_rot(opt_tracks, colors)))
    figs.append(("05b_coeff_with_rot", plot_coeffs_vs_speed_with_rot(opt_tracks, colors)))

    if SAVE_FIGURES:
        _save(figs)
        print(f"已保存 {len(figs)} 张图到: {OUTPUT_DIR}")

    if SHOW_WINDOWS:
        plt.show()
    else:
        for _, fig in figs:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
