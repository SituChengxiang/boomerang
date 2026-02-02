# Utils Module Documentation

This module contains utility functions for processing large-noise physics data.

由于回旋镖运动数据噪声较大，本模块提供了针对性的物理计算优化方案。

---

## 文件说明

### 1. `dataIO.py` (核心数据 I/O)
**依赖**：无
**功能**：
- CSV 文件读取与写入
- 数据完整性检查（最小点数要求）
- 时间序列验证（严格递增、无重复）
- 轨迹数据格式化

**主要函数**：
```python
load_track(filepath, required_columns=None, ...)  # 加载轨迹
save_track(filepath, data, ...)                   # 保存轨迹
validate_track_data(data, ...)                    # 验证数据有效性
```

---

### 2. `standardization.py` (时间标准化)
**依赖**：无（纯数值计算）
**功能**：
- 保证时间严格递增
- 清理重复时间点
- 时间归零（从 0 开始）
- 三次样条插值对齐
- 轨迹重采样

**主要函数**：
```python
normalize_time(t, target_freq=60.0)               # 时间标准化
is_uniform_time(t, rel_tol=1e-3)                  # 判断时间是否均匀
resample_time_series(t, values, new_freq=60.0)    # 重采样
ensure_strictly_increasing_time(t, ...)           # 确保递增
```

---

### 3. `derivatives.py` (微分计算)
**依赖**：`scipy`
**功能**：
针对高噪声场景提供多种求导方法：

#### 3.1 解析三次样条导数
- 处理整体轨迹 → 速度
- 平滑但可能过度平滑

#### 3.2 Savitzky-Golay (SG) 滤波二阶导
- 适用于样条插值分段线性
- 计算加速度效果更好
- **建议**：窗口大小设为数据点的 5%-10%

#### 3.3 状态空间估算
- RTK Smoother 回传参数
- 修正发射角度、初始速度
- 提供最优初始条件

#### 3.4 高斯回归统计（实验性）
- 用于不确定性估计

---

### 4. `smoother.py` (平滑滤波)
**依赖**：`pykalman`
**功能**：
- RTS Kalman Smoother 双向滤波
- 处理测量噪声与过程噪声

**调用示例**：
```python
from src.utils.smoother import RTSKalmanSmoother

smoother = RTSKalmanSmoother(
    process_noise=1e-4,      # 怕过拟合太小，太大平滑过度
    measurement_noise=1e-2   # 根据传感器噪声调整
)
smoothed_trajectory = smoother.smooth(t, x, y, z)
```

---

### 5. `filter.py` (滤波器集)
**依赖**：根据实现变化
**功能**：
- Kalman 滤波
- 低通滤波
- 移动平均滤波

---

### 6. `physicsVerdict.py` (物理判决)
**依赖**：`derivatives.py`, `smoother.py`, `filter.py`
**功能**：
从位置数据计算：
- 速度（微分）
- 机械能（动能 + 势能）
- 蕴含物理单位验证的数值

**关键判断**：
- 能量是否守恒？
- 势能是否趋势合理？
- 数据是否物理自洽？

---

### 7. `visualize.py` (可视化)
**独立使用**
**功能**：
- 绘制轨迹（xy, xz, yz 投影）
- 绘制速度曲线、加速度曲线
- 绘制能量变化
- 对比原始 vs 平滑数据

---

## 使用流程示例

```python
from pathlib import Path
from src.utils.dataIO import load_track, save_track
from src.utils.smoother import RTSKalmanSmoother

# 1. 加载原始数据
data_path = Path("data/raw/track1.csv")
raw_data = load_track(data_path)

# 2. 平滑处理
smoother = RTSKalmanSmoother(process_noise=1e-4, measurement_noise=1e-2)
smooth_t, smooth_x, smooth_y, smooth_z = smoother.smooth(
    raw_data["t"], raw_data["x"], raw_data["y"], raw_data["z"]
)

# 3. 保存平滑后数据
save_track(
    "data/interm/track1_SMR.csv",
    {"t": smooth_t, "x": smooth_x, "y": smooth_y, "z": smooth_z}
)

# 4. 轨迹优化（跳转到 src/fit/optimizeParams.py）
```

---

## 注意事项

### 数据质量检查清单
1. **时间序列**：必须严格递增（使用 `dataIO.ensure_strictly_increasing_time`）
2. **点数要求**：至少 50 个点（`load_track` 默认要求）
3. **数值范围**：避免全为 NaN 或 Inf
4. **物理单位**：位置（m），时间（s），速度（m/s）

### 平滑参数调优建议

**过程噪声 (Process Noise)**  
- 含义：模型对自身预测的不确定度
- 过大 → 过度平滑（轨迹被拉成直线，丢失真实波动）
- 过小 → 过度相信数据，噪声无法滤除
- **推荐值**：`1e-4` 到 `1e-3`

**测量噪声 (Measurement Noise)**  
- 含义：输入数据本身的不确定度（传感器误差）
- 过大 → 过度平滑（尤其末段真实信号弱，更易被平滑掉）
- 过小 → 几乎不做平滑，保留过多噪声
- **推荐值**：`1e-2` 到 `1e-1`

### 速度导数计算

**推荐方法**：
1. **Savitzky-Golay** (savgol)
   - 窗口大小：`int(0.05 * len(data))` 到 `int(0.1 * len(data))`
   - 多项式阶数：3-5 阶
   - 优势：对加速度计算友好

2. **三次样条解析导数**
   - 优势：全局平滑，数学解析
   - 优势：无窗口大小参数，更自动化

**不推荐**：
- 直接有限差分（噪声放大严重）

---

## 维护说明

本模块设计为 **无依赖于项目其他模块**（除了自身内部调用），
确保任何 `import` 都是安全的，不会循环依赖。

如需添加新功能：
1. 确认函数作用域（输入参数/输出格式）
2. 保持 API 一致性（使用 `np.ndarray` 作为标准格式）
3. 添加 docstring 说明物理意义
4. 在本文件中更新说明