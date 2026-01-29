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

因为把研究工作和开发工作放在了一个文件夹里，所以整个项目看起来并不那么常规
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
    |-- preprocess（预处理）
    `-- utils（一些小的工具）
```

## 我踩的坑

1. 一定要先确保处理后的数据是物理自洽的


## 接下来的工作安排

🎯 中优先级（接下来）

### 4. **时间处理模块** (`time_utils.py`)
**重复逻辑**：时间标准化、均匀性检查
```python
# 核心功能
def normalize_time(t, target_freq=60.0)
def is_uniform_time(t, rel_tol=1e-3)
def resample_time_series(t, values, new_freq=60.0)
```

### 5. **轨迹分析模块** (`trajectory.py`)
**重复逻辑**：轨迹特征提取
```python
# 核心功能
def calculate_trajectory_features(t, x, y, z)
def estimate_initial_velocity(t, x, y, z, method='weighted')
def calculate_flight_time(t, z, ground_level=0.0)
```

### 6. **可视化工具** (`plot_utils.py`)
**重复代码**：多个文件中的绘图函数
```python
# 核心功能
def plot_3d_trajectory(x, y, z, title="", ax=None)
def plot_energy_analysis(t, energy, dE_dt, ax=None)
def plot_velocity_components(t, vx, vy, vz, ax=None)
```

 🎯 低优先级（最后）

### 7. **文件发现模块** (`file_utils.py`)
**重复逻辑**：查找特定模式的文件
```python
# 核心功能
def find_track_files(directory, pattern="*opt.csv")
def find_velocity_files(directory, pattern="velocity.csv")
def batch_process_files(directory, process_func, pattern="*.csv")
```

### 8. **配置管理** (`config.py`)
**硬编码值**：物理常数、默认参数
```python
# 核心功能
class BoomerangConfig:
    MASS = 0.00218
    GRAVITY = 9.793
    AIR_DENSITY = 1.225
    # ...
```
