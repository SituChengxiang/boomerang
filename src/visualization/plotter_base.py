#!/usr/bin/env python3
"""
面向对象可视化基类和调度器
提供统一的绘图接口和配置驱动机制
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from pathlib import Path
import yaml


@dataclass
class PlotConfig:
    """图表配置数据类"""
    name: str
    title: str
    xlabel: str
    ylabel: str
    figsize: Tuple[int, int] = (10, 8)
    grid: bool = True
    legend: bool = True
    legend_fontsize: int = 8
    show_zero_line: bool = False
    twin_axis: Optional[str] = None  # 右侧Y轴变量名
    twin_ylabel: Optional[str] = None
    
    # 样式配置
    styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlotConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)


class BasePlotter(ABC):
    """绘图器基类"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.fig = None
        self.ax = None
        
    def setup_figure(self) -> Tuple[Figure, Axes]:
        """设置图形和坐标轴"""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize)
        return self.fig, self.ax
    
    def apply_common_styling(self):
        """应用通用样式"""
        if self.ax is None:
            raise RuntimeError("Must call setup_figure() first")
            
        self.ax.set_title(self.config.title)
        self.ax.set_xlabel(self.config.xlabel)
        self.ax.set_ylabel(self.config.ylabel)
        
        if self.config.grid:
            self.ax.grid(True, alpha=0.3)
            
        if self.config.show_zero_line:
            self.ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    def add_legend(self):
        """添加图例"""
        if self.config.legend and self.ax is not None:
            self.ax.legend(fontsize=self.config.legend_fontsize, loc='best')
    
    @abstractmethod
    def plot(self, data: Any, **kwargs) -> Figure:
        """抽象绘图方法，子类必须实现"""
        pass
    
    def show(self):
        """显示图形"""
        if self.fig is not None:
            plt.tight_layout()
            plt.show()
    
    def save(self, filepath: Union[str, Path], dpi: int = 200):
        """保存图形"""
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


class TimeSeriesPlotter(BasePlotter):
    """时间序列绘图器"""
    
    def plot(self, tracks_data: Dict[str, Tuple], **kwargs) -> Figure:
        """绘制时间序列图"""
        self.setup_figure()
        
        cmap = plt.get_cmap('tab10')
        
        for i, (track_name, data) in enumerate(tracks_data.items()):
            color = cmap(i % 10)
            
            # 解析数据 - 子类可以重写这个方法来处理不同的数据结构
            x_data, y_data = self._extract_xy_data(data, kwargs)
            
            style = self.config.styles.get(track_name, {})
            self.ax.plot(x_data, y_data, 
                        color=color, 
                        label=track_name,
                        linewidth=style.get('linewidth', 1.5),
                        alpha=style.get('alpha', 0.9))
            
            # 可选散点
            if kwargs.get('show_scatter', False):
                self.ax.scatter(x_data, y_data, s=20, color=color, alpha=0.6)
        
        self.apply_common_styling()
        self.add_legend()
        return self.fig
    
    def _extract_xy_data(self, data: Tuple, kwargs: Dict) -> Tuple:
        """提取XY数据，子类可以重写"""
        x_key = kwargs.get('x_key', 't')
        y_key = kwargs.get('y_key', 'value')
        
        # 默认假设data是元组，第一个元素是时间
        if isinstance(data, (tuple, list)) and len(data) > 0:
            if x_key == 't':
                x_data = data[0]  # 第一个元素是时间
            else:
                x_data = data[0]  # 简化处理
            
            # 查找Y数据
            if y_key in ['vz', 'vx', 'vy']:
                # 假设这些在特定位置
                y_index_map = {'vx': 1, 'vy': 2, 'vz': 3}
                y_data = data[y_index_map[y_key]]
            else:
                y_data = data[1]  # 默认第二个元素
                
            return x_data, y_data
        
        raise ValueError(f"Unsupported data format: {type(data)}")


class ScatterPlotter(BasePlotter):
    """散点图绘图器"""
    
    def plot(self, tracks_data: Dict[str, Tuple], **kwargs) -> Figure:
        """绘制散点图"""
        self.setup_figure()
        
        cmap = plt.get_cmap('tab10')
        
        for i, (track_name, data) in enumerate(tracks_data.items()):
            color = cmap(i % 10)
            
            x_data, y_data = self._extract_xy_data(data, kwargs)
            
            style = self.config.styles.get(track_name, {})
            self.ax.scatter(x_data, y_data,
                           s=style.get('s', 5),
                           alpha=style.get('alpha', 0.5),
                           color=color,
                           label=track_name)
        
        # 添加参考线
        if 'ref_lines' in kwargs:
            for line_config in kwargs['ref_lines']:
                self.ax.axhline(**line_config)
        
        self.apply_common_styling()
        self.add_legend()
        return self.fig
    
    def _extract_xy_data(self, data: Tuple, kwargs: Dict) -> Tuple:
        """提取XY数据"""
        x_key = kwargs.get('x_key', 'v_xy_sq')
        y_key = kwargs.get('y_key', 'f_z_aero')
        
        # 根据键名映射到数据索引
        key_to_index = {
            'v_xy_sq': 1,      # 水平速度平方
            'f_z_aero': 2,     # 垂直气动力
            'v_total_sq': 3,   # 总速度平方
            'a_drag_est': 4,   # 阻力估计
        }
        
        x_index = key_to_index.get(x_key, 1)
        y_index = key_to_index.get(y_key, 2)
        
        return data[x_index], data[y_index]


class MultiAxisPlotter(BasePlotter):
    """双Y轴绘图器"""
    
    def plot(self, tracks_data: Dict[str, Tuple], **kwargs) -> Figure:
        """绘制双Y轴图"""
        self.setup_figure()
        
        # 创建右侧Y轴
        ax_right = self.ax.twinx()
        
        cmap = plt.get_cmap('tab10')
        
        for i, (track_name, data) in enumerate(tracks_data.items()):
            color = cmap(i % 10)
            
            # 左侧Y轴数据
            x_data, y_left = self._extract_left_data(data, kwargs)
            left_style = kwargs.get('left_style', {})
            
            self.ax.plot(x_data, y_left,
                        color=color,
                        linewidth=left_style.get('linewidth', 1.5),
                        label=f"{track_name} (left)")
            
            # 右侧Y轴数据（如果有）
            if self.config.twin_axis:
                y_right = self._extract_right_data(data, kwargs)
                right_style = kwargs.get('right_style', {})
                
                ax_right.plot(x_data, y_right,
                             color='k',
                             linestyle=right_style.get('linestyle', '--'),
                             alpha=right_style.get('alpha', 0.3),
                             linewidth=right_style.get('linewidth', 1.0),
                             label=f"{track_name} (right)")
        
        # 设置右侧Y轴标签
        if self.config.twin_ylabel:
            ax_right.set_ylabel(self.config.twin_ylabel)
        
        self.apply_common_styling()
        self.add_legend()
        return self.fig
    
    def _extract_left_data(self, data: Tuple, kwargs: Dict) -> Tuple:
        """提取左侧Y轴数据"""
        return data[0], data[kwargs.get('left_index', 1)]
    
    def _extract_right_data(self, data: Tuple, kwargs: Dict) -> np.ndarray:
        """提取右侧Y轴数据"""
        return data[kwargs.get('right_index', -1)]


class PlotScheduler:
    """绘图调度器 - 统一管理所有图表"""
    
    def __init__(self):
        self.plotters: Dict[str, BasePlotter] = {}
        self.configs: Dict[str, PlotConfig] = {}
    
    def register_plotter(self, name: str, plotter_class: type, config: PlotConfig):
        """注册绘图器"""
        self.configs[name] = config
        # 实际创建会在需要时进行
    
    def load_configs_from_yaml(self, config_file: Union[str, Path]):
        """从YAML文件加载配置"""
        with open(config_file, 'r') as f:
            configs_dict = yaml.safe_load(f)
        
        for name, config_dict in configs_dict.items():
            config = PlotConfig.from_dict(config_dict)
            # 这里可以根据type字段选择不同的plotter类
            plotter_type = config_dict.get('type', 'time_series')
            
            if plotter_type == 'time_series':
                plotter_class = TimeSeriesPlotter
            elif plotter_type == 'scatter':
                plotter_class = ScatterPlotter
            elif plotter_type == 'multi_axis':
                plotter_class = MultiAxisPlotter
            else:
                plotter_class = TimeSeriesPlotter  # 默认
            
            self.register_plotter(name, plotter_class, config)
    
    def create_plotter(self, name: str) -> BasePlotter:
        """创建指定名称的绘图器实例"""
        if name not in self.configs:
            raise ValueError(f"No configuration found for plot '{name}'")
        
        config = self.configs[name]
        # 简化处理 - 实际应该根据配置选择正确的类
        return TimeSeriesPlotter(config)
    
    def plot_single(self, name: str, data: Any, show: bool = True, **kwargs) -> Figure:
        """生成单个图表"""
        plotter = self.create_plotter(name)
        fig = plotter.plot(data, **kwargs)
        if show:
            plotter.show()
        return fig
    
    def plot_batch(self, plot_names: List[str], data: Any, output_dir: Optional[Path] = None):
        """批量生成图表"""
        figures = {}
        
        for name in plot_names:
            print(f"Generating plot: {name}")
            try:
                fig = self.plot_single(name, data, show=False)
                figures[name] = fig
                
                if output_dir:
                    output_file = output_dir / f"{name}.png"
                    fig.savefig(output_file, dpi=200, bbox_inches='tight')
                    print(f"  Saved: {output_file}")
                    
            except Exception as e:
                print(f"  Error generating {name}: {e}")
        
        return figures


# 导出类
__all__ = [
    'PlotConfig',
    'BasePlotter', 
    'TimeSeriesPlotter',
    'ScatterPlotter',
    'MultiAxisPlotter',
    'PlotScheduler'
]