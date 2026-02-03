# 可视化脚本面向对象重构说明

## 🎯 重构目标

解决原有可视化脚本过于冗长、重复代码多的问题，通过面向对象设计提高代码的：

- **可维护性**: 统一接口，减少重复代码
- **可扩展性**: 易于添加新类型的图表
- **可配置性**: 通过配置文件管理图表样式
- **可复用性**: 组件化设计，可在不同场景使用

## 🏗️ 架构设计

### 1. 核心组件

```
src/visualization/
├── plotter_base.py          # 基类和调度器
├── VisualizeData_OO.py      # 重构后的主脚本
└── plot_configs/           # 配置文件目录
    └── plots_config.yaml   # 图表配置
```

### 2. 类层次结构

```python
BasePlotter (抽象基类)
├── TimeSeriesPlotter      # 时间序列图
├── ScatterPlotter         # 散点图
├── MultiAxisPlotter       # 双Y轴图
└── PlotScheduler          # 图表调度器
```

## 🔧 使用方法

### 基本用法

```python
from src.visualization.plotter_base import PlotScheduler, PlotConfig

# 1. 创建调度器
scheduler = PlotScheduler()

# 2. 定义配置
config = PlotConfig(
    name='my_plot',
    title='My Plot Title',
    xlabel='X Axis',
    ylabel='Y Axis'
)

# 3. 注册配置
scheduler.configs['my_plot'] = config

# 4. 生成图表
fig = scheduler.plot_single('my_plot', data)
```

### 配置驱动方式

```python
# 从YAML文件加载配置
scheduler.load_configs_from_yaml('config/plots_config.yaml')

# 批量生成图表
figures = scheduler.plot_batch(
    ['vertical_aero_vs_speed', 'drag_deceleration'],
    tracks_data,
    output_dir=Path('out/plots')
)
```

## 📊 配置文件格式

```yaml
# config/plots_config.yaml
vertical_aero_vs_speed:
  type: scatter
  name: vertical_aero_vs_speed
  title: "Vertical Aero Acceleration vs Horizontal Speed Squared"
  xlabel: "Horizontal Speed Squared (m²/s²)"
  ylabel: "Vertical Acceleration + G (m/s²)"
  figsize: [10, 8]
  grid: true
  legend: true
  ref_lines:
    - y: 9.8
      color: black
      linestyle: "--"
      alpha: 0.3
      label: "1G Hover"
```

## 🚀 优势对比

### 重构前 vs 重构后

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| 代码行数 | ~500行重复代码 | ~50行核心逻辑 |
| 添加新图表 | 复制粘贴100+行 | 10行配置 |
| 修改样式 | 多处修改 | 统一配置文件 |
| 错误定位 | 困难 | 清晰的类结构 |
| 测试覆盖 | 复杂 | 模块化易测试 |

### 具体改进

1. **消除重复代码**: 通用绘图逻辑提取到基类
2. **配置驱动**: 样式和布局通过配置管理
3. **职责分离**: 数据处理、绘图逻辑、配置管理分离
4. **易于扩展**: 新增图表类型只需继承基类

## 🛠️ 扩展指南

### 添加新的图表类型

```python
class CustomPlotter(BasePlotter):
    def plot(self, data: Any, **kwargs) -> plt.Figure:
        self.setup_figure()
        
        # 实现具体的绘图逻辑
        # ...
        
        self.apply_common_styling()
        self.add_legend()
        return self.fig
    
    def _extract_xy_data(self, data: Tuple, kwargs: Dict) -> Tuple:
        # 实现数据解析逻辑
        pass
```

### 添加新的配置选项

在 `PlotConfig` 类中添加新字段：

```python
@dataclass
class PlotConfig:
    # ... 现有字段 ...
    custom_option: str = "default_value"
```

## 📈 性能优化

### 批量处理
```python
# 一次加载配置，批量生成所有图表
scheduler.load_configs_from_yaml('config/plots_config.yaml')
figures = scheduler.plot_batch(all_plot_names, data)
```

### 内存管理
```python
# 及时清理不需要的图形对象
plt.close('all')
del figures
```

## 🧪 测试建议

```python
def test_plotter_creation():
    config = PlotConfig(name='test', title='Test', xlabel='X', ylabel='Y')
    plotter = TimeSeriesPlotter(config)
    assert plotter.config.name == 'test'

def test_plot_generation():
    # 测试基本绘图功能
    pass
```

## 📝 迁移指南

### 从旧脚本迁移

1. **识别重复模式**: 找出相似的绘图函数
2. **提取配置**: 将样式参数提取到配置中
3. **创建对应类**: 选择合适的Plotter子类
4. **逐步替换**: 逐个替换原有函数调用

### 兼容性考虑

- 保持原有的数据接口不变
- 提供包装函数兼容旧调用方式
- 渐进式迁移，避免一次性大改动

## 🎯 最佳实践

1. **配置优先**: 优先使用配置文件而非硬编码
2. **单一职责**: 每个Plotter类只负责一种图表类型
3. **异常处理**: 在调度器中统一处理绘图异常
4. **文档完善**: 为每个配置项添加详细说明
5. **版本控制**: 配置文件也要纳入版本管理

这个重构方案遵循了项目的模块化规范，通过配置驱动和面向对象设计，大大提升了可视化代码的质量和可维护性。