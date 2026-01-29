"""
精简的数据IO模块，用于处理回旋镖轨迹数据。
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[float]]
PathLike = Union[str, pathlib.Path]


def ensure_strictly_increasing_time(
    t: np.ndarray,
    *series: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, ...]:
    """确保时间序列严格递增，移除重复或非递增的时间点。

    重复时间戳会导致数值微分时除以零。

    Args:
        t: 时间数组
        *series: 与时间对应的数据序列
        eps: 最小时间间隔容差

    Returns:
        清理后的(时间, 数据序列...)
    """
    t = np.asarray(t, dtype=float)

    # 验证长度一致
    if any(len(s) != len(t) for s in series):
        raise ValueError("所有数据序列必须与时间序列长度相同")

    # 检查有限值
    finite_mask = np.isfinite(t)
    for s in series:
        finite_mask &= np.isfinite(s)

    if not np.any(finite_mask):
        raise ValueError("所有数据都包含非有限值")

    t = t[finite_mask]
    cleaned = [np.asarray(s, dtype=float)[finite_mask] for s in series]

    if len(t) == 0:
        return t, *cleaned

    # 按时间排序
    order = np.argsort(t)
    t = t[order]
    cleaned = [s[order] for s in cleaned]

    # 移除重复时间点（保留第一个）
    keep_idx = [0]
    last_t = float(t[0])

    for i in range(1, len(t)):
        ti = float(t[i])
        if ti > last_t + eps:
            keep_idx.append(i)
            last_t = ti

    keep = np.asarray(keep_idx, dtype=int)
    t_out = t[keep]
    series_out = [s[keep] for s in cleaned]

    return t_out, *series_out


def load_track(
    filepath: PathLike,
    required_columns: Optional[List[str]] = None,
    sort_by_time: bool = True,
    clean_time: bool = True,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """从CSV文件加载轨迹数据。

    Args:
        filepath: CSV文件路径
        required_columns: 必须存在的列，默认为["t", "x", "y", "z"]
        sort_by_time: 是否按时间排序
        clean_time: 是否清理时间序列
        eps: 时间清理的最小间隔

    Returns:
        数据字典 {列名: 数据数组}
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if required_columns is None:
        required_columns = ["t", "x", "y", "z"]

    try:
        # 加载CSV
        data = np.genfromtxt(filepath, delimiter=",", names=True)

        if data.ndim == 0:
            raise ValueError(f"CSV文件必须包含至少一行数据: {filepath}")

        # 检查必需列
        available_columns = data.dtype.names
        if available_columns is None:
            raise ValueError(f"CSV文件没有列名: {filepath}")

        missing = [col for col in required_columns if col not in available_columns]
        if missing:
            raise ValueError(f"缺少必需列 {missing}。可用列: {available_columns}")

        # 提取数据
        result = {}
        for col in available_columns:
            result[col] = np.asarray(data[col], dtype=float)

        # 检查长度一致
        lengths = {col: len(arr) for col, arr in result.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"数据列长度不一致: {lengths}")

        # 时间排序
        if sort_by_time and "t" in result:
            order = np.argsort(result["t"])
            for col in result:
                result[col] = result[col][order]

        # 时间清理
        if clean_time and "t" in result:
            data_series = [result[col] for col in result if col != "t"]

            cleaned = ensure_strictly_increasing_time(
                result["t"], *data_series, eps=eps
            )

            result["t"] = cleaned[0]
            for i, col in enumerate([c for c in result if c != "t"]):
                result[col] = cleaned[i + 1]

        return result

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"加载文件时出错 {filepath}: {e}")


def save_track(
    filepath: PathLike,
    data: Dict[str, ArrayLike],
    columns: Optional[List[str]] = None,
    fmt: str = "%.8f",
    delimiter: str = ",",
) -> pathlib.Path:
    """保存轨迹数据到CSV文件。

    Args:
        filepath: 输出文件路径
        data: 数据字典 {列名: 数据数组}
        columns: 列顺序，默认为数据字典的键排序
        fmt: 数字格式化
        delimiter: CSV分隔符

    Returns:
        保存的文件路径
    """
    filepath = pathlib.Path(filepath)

    if not data:
        raise ValueError("数据字典为空")

    # 确定列顺序
    if columns is None:
        columns = sorted(data.keys())

    # 验证所有列都存在且长度一致
    lengths = []
    arrays = []

    for col in columns:
        if col not in data:
            raise ValueError(f"缺少列 '{col}'。可用列: {list(data.keys())}")

        arr = np.asarray(data[col], dtype=float)
        lengths.append(len(arr))
        arrays.append(arr)

    if len(set(lengths)) > 1:
        raise ValueError(f"数据列长度不一致: {dict(zip(columns, lengths))}")

    # 保存
    stacked = np.column_stack(arrays)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    header = delimiter.join(columns)
    np.savetxt(
        filepath,
        stacked,
        delimiter=delimiter,
        header=header,
        comments="",
        fmt=fmt,
    )

    return filepath


def validate_track_data(
    data: Dict[str, np.ndarray],
    require_time: bool = True,
    require_position: bool = True,
    min_points: int = 3,
) -> Tuple[bool, List[str]]:
    """验证轨迹数据的完整性。

    Args:
        data: 数据字典
        require_time: 是否要求时间列
        require_position: 是否要求位置列
        min_points: 最小数据点数

    Returns:
        (是否有效, 问题描述列表)
    """
    issues = []

    # 检查基本列
    if require_time and "t" not in data:
        issues.append("缺少时间列 't'")

    if require_position:
        for col in ["x", "y", "z"]:
            if col not in data:
                issues.append(f"缺少位置列 '{col}'")

    # 检查数据点数量
    if data:
        sample_col = next(iter(data))
        n_points = len(data[sample_col])

        if n_points < min_points:
            issues.append(f"数据点太少: {n_points} < {min_points}")

        # 检查所有列长度一致
        for col, arr in data.items():
            if len(arr) != n_points:
                issues.append(f"列 '{col}' 长度不一致: {len(arr)} != {n_points}")

    # 检查时间是否递增
    if "t" in data:
        t = data["t"]
        if len(t) > 1:
            dt = np.diff(t)
            if np.any(dt <= 0):
                issues.append("时间序列不是严格递增的")

            if not np.all(np.isfinite(t)):
                issues.append("时间序列包含非有限值")

    # 检查其他列的有限值
    for col in data:
        if col != "t":
            arr = data[col]
            if not np.all(np.isfinite(arr)):
                issues.append(f"列 '{col}' 包含非有限值")

    return len(issues) == 0, issues


def create_opt_filename(original_path: PathLike) -> pathlib.Path:
    """生成优化后数据的文件名。

    遵循项目约定：track1.csv -> track1opt.csv

    Args:
        original_path: 原始文件路径

    Returns:
        优化后的文件路径
    """
    original_path = pathlib.Path(original_path)
    stem = original_path.stem

    # 如果已经是opt文件，避免双重扩展
    if stem.endswith("opt"):
        stem = stem[:-3]

    opt_stem = f"{stem}opt"
    return original_path.with_name(f"{opt_stem}{original_path.suffix}")


def load_track_simple(
    filepath: PathLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """简化的加载函数，返回(t, x, y, z)元组。

    用于向后兼容需要(t, x, y, z)元组的旧代码。

    Args:
        filepath: CSV文件路径

    Returns:
        (t, x, y, z) 元组
    """
    data = load_track(filepath)

    required = ["t", "x", "y", "z"]
    for col in required:
        if col not in data:
            raise ValueError(f"文件缺少必需列 '{col}'")

    return data["t"], data["x"], data["y"], data["z"]
