# 中国大学生物理学术竞赛（CUPT） Invent Yourself: Paper Boomerang 代码参考

CUPT让我们相遇于此吧（  
这个仓库里有：  
- Go语言的数值计算和GUI模拟
- Rust用于科学计算的尝试
- Python进行数据拟合尝试
- ~~Gunplot绘图代码~~
- Julia尝试更高性能计算
- ~~wxMaxima瞎写的代码~~

（我真是成分复杂啊……）

来看看就好，如果能帮到你，本人不胜荣幸

## 项目结构

因为把研究工作和开发工作放在了一个文件夹里，所以整个项目看起来并不那么常规：
```plaintext
boomerang
|-- README.md
|-- data（数据组）
|   |-- interm（处理时的中间文件）
|   |-- raw（原始数据文件）
|   `-- final（处理后的文件）
|-- out（整个项目输出的一些文件）
`-- src（源码文件）
    |-- visualization（可视化相关）
    |-- fit（轨迹拟合相关）
    |-- cmd（命令行脚本）
    |   `-- preProcess.py（预处理脚本）
    |-- calculation（Rust高性能计算库）
    `-- utils（一些小的工具）
```

## 我踩的坑

1. 一定要先确保处理后的数据是物理自洽的


# Rust 高性能计算模块 (`src/calculation`)

这个 Rust 库用于加速轨迹拟合的核心计算，特别是 ODE 求解和参数优化。Python 处理数据 I/O 和可视化，Rust 提供 5-10 倍的速度提升。

## 功能

- **ODE 求解器**：RK4 和自适应步长积分
- **气动模型**：SPE (Simple Pendulum Extension) 和 BAP (Boomerang Aerodynamics Project) 模型
- **向量运算**：3D 向量、四元数运算
- **损失函数**：MSE、加权多轨迹损失
- **优化器**：参数拟合（开发中）
- **Python 绑定**：通过 PyO3 与 Python 无缝交互

## 编译与使用

### 编译（带 Python 绑定）
```bash
cd src/calculation
cargo build --features python-bindings
```

### 运行测试
```bash
cargo test
```

### 在 Python 中使用
```python
from boomerang_calc import PyOptimizer, PyTrajectoryMetrics

# 创建优化器
optimizer = PyOptimizer()

# 使用优化（Python 端准备好数据）
# params = [Cl, Cd, k_roll, tilt, omega0]
result = optimizer.optimize_single(
    params=[0.5, 0.1, 1.0, 20.0, 80.0],
    initial_state=[0, 0, 0, 10, 2, 0],
    t_eval=t_array,
    reference_metrics=ref_metrics
)
```

## 项目状态

- ✅ ODE 求解器：已完成
- ✅ 气动模型：已完成
- ✅ Python 绑定：已完成
- ⚠️ 优化器：草图版（算法待完善）

---

## 1. 数据预处理 (`src/cmd/preProcess.py`)

这是处理原始 Tracker 导出数据的入口点：

```bash
python src/cmd/preProcess.py
```

**功能**：
- 读取 `data/raw/` 下的 CSV 文件
- 应用 RTSKalmanSmoother 进行轨迹平滑
- 计算速度、加速度
- 保存到 `data/interm/`（中间处理）或 `data/final/`（最终用于拟合）

## 2. 平滑参数调优（详见 README 中的参数说明）

在 `src/utils/readme.md` 中有详细的 `process_noise` 和 `measurement_noise` 调整说明，关键点：
- **过程噪声（Process Noise）**：控制模型与数据的信任度平衡
- **测量噪声（Measurement Noise）**：控制数据噪声的大小

---


## 核心文件：`src/fit/optimizeParams.py`

### 功能
1. 从 `data/final/` 读取已处理的轨迹数据
2. 使用 L-BFGS-B 优化器调整五大参数：
   ```
   Cl   (升力系数)      : 0.020 - 0.200
   Cd   (阻力系数)      : 0.000010 - 0.150
   k_roll (滚转系数)    : -10.0 - 10.0 (预cession方向)
   tilt (初始倾角)      : 0 - 60 度
   omega0 (初始自旋)    : 50 - 120 rad/s
   ```
3. 生成可视化图表（拟合轨迹 vs 真实轨迹）

### 运行单条轨迹优化
```bash
python src/fit/optimizeParams.py
```

### 运行结果
- 单次优化时间：~2-5 秒（Python版）
- 未来目标：0.2-0.5 秒（Rust版）

---


**重要**：优化前必须确保预处理后的数据是物理自洽的。

### 检查清单
1. **时间严格递增**：无重复时间点
2. **速度连续性**：避免突变
3. **能量守恒**：总能量不应无理由跳变
4. **合理范围**：速度 0-20 m/s，位置在视野内

### 工具
```bash
python src/cmd/trackCheck.py
```

---

## 物理模型

这是一个修正了好多好多次的版本了，而且还只是勉强能用，所以总要摸索一会儿的，一个阶段有一个阶段能用的理论。赛场上我甚至没用到这一步。所以，如果你在打比赛的时候又卡在了理论上，不要灰心，先找个能用的版本说清楚就好了。

### 1. 核心思想：飞行陀螺 (Flying Gyroscope)
回旋镖不是一个单纯的质点与空气交互，而是一个**自带旋转动能的飞行翼面**。它的飞行轨迹由以下三个主要机制决定：
1.  **升力矢量控制 (Lift Vectoring)**：回旋镖的升力垂直于其翼平面。由于陀螺进动效应，翼平面会发生滚转（Roll），从而改变升力矢量的指向（从最初的克服重力，逐渐变为指向圆心的向心力），实现转弯。
2.  **旋转动压维持 (Rotational Dynamic Pressure)**：即使平移速度很慢，高速旋转的叶片依然能切割空气产生升力。这解释了为何在顶点（Apex）它不会像石块一样掉下来。
3.  **姿态演变 (Attitude Evolution)**：从发射时的接近直立（Tilt $\approx 10-20^\circ$），到顶点的接近水平（相对于地面是 Knife-edge，但相对于重力是 90度 Bank），再到返回时的平飘。


### 2. 数学表述流程 (Step-by-Step Physics)

在每一个时间步 $t$，已知状态向量 $S = [x, y, z, v_x, v_y, v_z]$ 和自旋速度 $\omega$。

#### **第一步：计算有效动压 (Effective Dynamic Pressure)**
这是为了修正低速下的升力计算。我们不再只看平移速度 $v$，而是引入旋转带来的“等效速度”。

$$v_{trans}^2 = v_x^2 + v_y^2 + v_z^2$$
$$v_{eff}^2 = v_{trans}^2 + (\sigma \cdot \omega \cdot R)^2$$
$$q = \frac{1}{2} \rho S v_{eff}^2$$

*   $\sigma$ (Sigma)：旋转贡献系数（经验值约 0.3~0.5）。
*   $R$：回旋镖半径。

这种处理方式让我们能用一个相对恒定的 $C_L$（常数与攻角相关，但不再需要是速度的狂暴函数）来描述升力。

#### **第二步：确定姿态与升力方向 (Attitude & Lift Direction)**
我们需要一个模型来描述回旋镖的姿态矩阵 $R_{body}$（或者简单的单位法向量 $\vec{n}_{lift}$）。
在仿真中，这通常来自于**欧拉角方程**（受进动矩控制）：

1.  **气动扭矩 (Aerodynamic Torque)**：由于前进侧（Advancing Blade）升力大于后退侧（Retreating Blade），产生一个能够翻转回旋镖的力矩 $\tau_{aero}$。
2.  **陀螺进动 (Gyroscopic Precession)**：力矩 $\tau_{aero}$ 作用在高速旋转的陀螺上，产生成90度相位的进动角速度 $\Omega_{precession}$。
    $$\vec{\Omega}_{precess} \approx \frac{\vec{\tau}_{aero}}{I_z \cdot \omega}$$
    这也是导致回旋镖不断“左倾（Roll）”的原因。
3.  **升力方向**：升力始终沿着回旋镖的自转轴（法向量）方向。
    $$\vec{F}_{Lift} = C_L \cdot q \cdot \vec{n}_{axis}$$
    (注意：这里的 $\vec{n}_{axis}$ 是随时间变化的，从最初指向“上+左”，慢慢变成只想“左+水平”）。

#### **第三步：计算合力 (Total Force)**
$$ \vec{F}_{total} = \vec{F}_{gravity} + \vec{F}_{drag} + \vec{F}_{lift} $$

1.  **重力**：$\vec{F}_g = [0, 0, -mg]$
2.  **阻力**：沿着速度反方向。
    $$\vec{F}_{drag} = - C_D \cdot q \cdot \frac{\vec{v}}{|\vec{v}|}$$
    *(注意：我们也发现了在刀锋飞行阶段，$C_D$ 会显著降低)*
3.  **升力**：如第二步所述，垂直于速度，或者更准确地说是沿着转轴方向。
    $$\vec{F}_{lift} = C_L \cdot q \cdot \vec{n}_{axis}$$

#### **第四步：状态更新 (Integration)**
$$ \vec{a} = \frac{\vec{F}_{total}}{m} $$
$$ \vec{v}_{t+1} = \vec{v}_t + \vec{a} \cdot \Delta t $$
$$ \vec{x}_{t+1} = \vec{x}_t + \vec{v}_t \cdot \Delta t $$
同时更新自旋 $\omega$（线性衰减 $\omega = \omega_0 - k \cdot t$）和姿态角。


### 全新逆向发现 (From Inverse Solve)
通过我们刚刚的逆向求解验证，我们确认了：
1.  **$C_D$ 在 Knife-edge 阶段极低**（几乎消失）。
2.  **$C_L$ 在引入旋转动压修正后非常平稳**（稳定在 0.1 左右）。

这意味着接下来的正向拟合程序，可以直接假设 **$C_L$ 和 $C_D$ 是分段常数或者简单的攻角函数**，而不必使用复杂的高阶多项式去凑数据。

## 一些模块的参数调整说明

### 1. smoother - 平滑模块

利用RTSKalmanSmoother进行双向轨迹平滑，相比kalman滤波更平稳一点
```python
# __init__里的参数
self.process_noise = 0.01      # 过程噪声（模型不确定性）
self.measurement_noise = 0.1    # 测量噪声（数据不确定性）
```
**process_noise（过程噪声）**

- **含义**：模型对自己预测的不确定度
- **风险**：如果**过大** → 会过度平滑
  - Kalman会相信模型（常速模型）而不相信数据
  - 结果：轨迹被"拉"成直线，丢失了回旋镖的真实波动
  - **末段尤其明显**：着陆时回旋镖在减速，速度变化快，但常速模型假设"速度不变"

**measurement_noise（测量噪声）**

- **含义**：数据本身的不确定度（传感器噪声）
- **风险**：如果**过大** → 也会过度平滑
  - 认为数据噪声大，所以多平滑
  - **末段问题**：着陆时真实信号弱，噪声占比高，更容易被平滑掉

```python
def get_smoother_config() -> dict:
    """Return default smoother configuration."""
    return {
        "kalman_process_noise": 1e-4,      # 比你的默认0.01小！（更平滑）
        "kalman_measurement_noise": 1e-2,    # 比你的默认0.1小！（更平滑）
        "savgol_window": 7,                # 窗口大小
        "savgol_polyorder": 3,              # 多项式阶数
    }
```


```markdown
已汇总会话历史记录这是一个非常深刻的问题。这涉及到我们在做“动力学拟合”时最常遇到的核心矛盾：**逆向求解（Inverse Problem）和正向拟合（Forward Fitting）在数学本质上的维度差异。**

简而言之：**逆向是“代数题”，正向是“混沌系统”。**

以下是从第一性原理出发的分析：

### 1. 误差传播机制完全不同

*   **逆向求解（Inverse Solve）是“且听龙吟”：**
    *   你在做 `Inverse Solve` 时，输入是**真实测量**的 $x, v, a$。
    *   方程是 $F_{aero} = ma - mg$。这是一个**代数映射**（Algebraic Mapping）。
    *   在 $t=0.5s$ 时，哪怕你的数据有一点噪声，算出来的 $C_L$ 也就是那个时刻偏一点点。** $t=0.5s$ 的误差绝对不会影响 $t=0.6s$ 的计算。** 它是“即时”的。

*   **正向拟合（Forward Simulation）是“蝴蝶效应”：**
    *   模拟是**初值问题（Initial Value Problem, IVP）**。
    *   $t=0.1s$ 时的状态，是通过积分得到的。如果你的 $k_{roll}$（进动系数）在 $t=0$ 时偏了 1%，导致回旋镖的自旋轴（Spin Axis）多倾斜了 0.1 度。
    *   在这个错误的倾角下，**升力矢量的方向**就偏了。
    *   这个偏差会作为一个错误的加速度，被**二次积分**进位置里。
    *   到了 $t=1.0s$，这个微小的初始偏差，会因为路径的累积效应，导致回旋镖可能飞到几米开外，甚至通过“错误的姿态”进入了完全不同的气动区域（失速或坠落）。

### 2. 也是最关键的：陀螺进动的“转向灾难”

回旋镖不是普通抛射物，它是一个**陀螺**。这是它拟合最难的地方。

*   **普通物体**：阻力算错 10%，它也就是飞得远一点或近一点（一维误差）。
*   **回旋镖**：它是靠**倾斜升力矢量**来转向的（像摩托车压弯）。
    *   进动力矩 $\vec{\tau}$ 决定了自旋轴 $\vec{n}$ 的变化率 $d\vec{n}/dt$。
    *   自旋轴 $\vec{n}$ 决定了升力 $\vec{L}$ 指向哪里。
    *   **如果你没算对力矩（$k_{roll}$），你就没算对它怎么“转弯”。**
    *   一旦转弯的时机或速率错了，后面的所有轨迹都和真实数据**分道扬镳**了。这时候优化器怎么调 $C_L$（升力大小）都没用，因为方向已经错了。

### 3. 为什么“暴力扫描”反而比“加约束”表现好？

你观察到的这个现象非常经典：

*   **优化地形（Optimization Landscape）**：
    *   只用 MSE（暴力）：地形虽然坑坑洼洼，但优化器可以找到一些奇奇怪怪的“虫洞”钻过去。比如它搞出一个巨大的 $k_{roll}$ 让回旋镖疯狂进动，强行把尾巴甩回来。虽然不物理，但数学上残差小。
    *   加了 Heading/Endpoint 约束（精致）：你在地形上竖起了几堵高墙。
        *   优化器刚想往那个“疯狂参数”的山谷走，发现 Heading Error 变大了，被墙挡回来了。
        *   它想往“物理正确”的山谷走，但路途遥远，中间隔着高高的 Loss 分数。
        *   **结果**：它被困在了原地（Local Minimum），只能把 $C_L$ 降到接近 0（既然我飞不对，那我干脆别飞了，躺平算误差最小）。这就是你看到 $C_L=0.02$ 的原因。

### 4. 为什么“6D 相空间 MSE”是正解？

你提到的改用 **6D MSE (Position + Velocity)** 是非常敏锐的直觉，这是解决上述问题的标准手段。

*   **纯位置 MSE ($x, y, z$)**：
    *   优化器可能会找到一条路径，位置是对的，但速度方向是反的，或者速度极快地掠过该点。这在动力学上是荒谬的。
*   **相空间 MSE ($x, v$)**：
    *   你强迫模拟器不仅要在**哪里（Where）**和真实数据重合，还要以**何种姿态势能（How）**通过该点。
    *   这实际上是在约束**能量**和**动量**的流形。这会极大地抹平那些“虽然位置对但物理不合理”的局部极小值坑（Artifact Minima）。

---

### 结论与后续策略

**结论**：之所以难，是因为正向拟合是在**在一个极度敏感的非线性动力学系统中，寻找一条必须同时满足“位置”和“姿态转向”的唯一解**。

**策略**：
1.  **必须要 Fix 变量**：如果你已经通过 Inverse Solve 知道了 `tilt` 大概是多少，**定死它**！减少一个自由度，对于搜索空间的坍缩是指数级的帮助。
2.  **6D Loss**：加上速度残差权重，把优化器锁死在动量守恒的轨道上。
3.  **松弛约束**：去掉那些人为的 Endpoint 权重，让 6D Loss 自己去约束整体形态。

如果你同意这个分析，我们可以开始按照你的思路修改代码：**用 6D MSE，定死 Tilt，放开 Sigma。**
```