#!/usr/bin/env python3
"""
面向对象重构的可视化脚本
使用PlotScheduler统一管理所有图表生成
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.aerodynamics import (
    calculate_effective_coefficients,
    calculate_net_aerodynamic_force,
    decompose_aerodynamic_force,
    find_optimal_v_rot,
)
from src.utils.energy import calculate_energy_from_dataframe
from src.utils.kinematics import analyze_track
from src.utils.physicsCons import MASS, G

# 导入新的面向对象可视化模块
from src.visualization.plotter_base import PlotScheduler, PlotConfig
from src.utils.visualize import setup_debug_style


def load_and_prepare_data(csv_path: str = "data/interm/velocity.csv"):
    """加载并准备数据"""
    try:
        df_all = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return None, None, None

    tracks = df_all.track.unique()
    print(f"Analyzing {len(tracks)} tracks...")

    # 准备数据
    tracks_data = {}
    energy_data = {}

    for track in tracks:
        df_track = df_all[df_all.track == track].sort_values(by="t")
        if len(df_track) < 10:
            continue

        # 分析轨迹
        track_result = analyze_track(df_track, track)
        tracks_data[track] = track_result

        # 计算能量
        t, total_energy, kinetic_energy, potential_energy = (
            calculate_energy_from_dataframe(df_track)
        )
        energy_data[track] = (t, total_energy, kinetic_energy, potential_energy)

    return df_all, tracks_data, energy_data


def create_scheduler_with_configs():
    """创建配置好的调度器"""
    scheduler = PlotScheduler()
    
    # 直接在代码中定义配置（替代YAML文件）
    configs = {
        'vertical_aero_vs_speed': PlotConfig(
            name='vertical_aero_vs_speed',
            title='Vertical Aero Acceleration vs Horizontal Speed Squared',
            xlabel='Horizontal Speed Squared (m²/s²)',
            ylabel='Vertical Acceleration + G (m/s²)',
            show_zero_line=True
        ),
        'drag_deceleration': PlotConfig(
            name='drag_deceleration',
            title='Drag Deceleration vs Total Speed Squared',
            xlabel='Total Speed Squared (m²/s²)',
            ylabel='Drag Deceleration (m/s²)'
        ),
        'vertical_velocity': PlotConfig(
            name='vertical_velocity',
            title='Vertical Velocity vs Time',
            xlabel='Time (s)',
            ylabel='v_z (m/s)',
            show_zero_line=True
        ),
        'vertical_aero_force': PlotConfig(
            name='vertical_aero_force',
            title='Vertical Aero Force vs Time',
            xlabel='Time (s)',
            ylabel='Lift Acceleration (m/s²)'
        ),
        'mechanical_energy': PlotConfig(
            name='mechanical_energy',
            title='Mechanical Energy vs Time',
            xlabel='Time (s)',
            ylabel='Energy (m²/s²)'
        ),
        'energy_components': PlotConfig(
            name='energy_components',
            title='Energy Components: Kinetic and Potential vs Time',
            xlabel='Time (s)',
            ylabel='Energy per mass (m²/s²)'
        ),
        'energy_change_rate': PlotConfig(
            name='energy_change_rate',
            title='Energy Change Rate vs Time',
            xlabel='Time (s)',
            ylabel='dE/dt per mass (m²/s³)',
            show_zero_line=True
        )
    }
    
    # 注册配置
    for name, config in configs.items():
        scheduler.configs[name] = config
    
    return scheduler


def generate_plots_oo(df_all, tracks_data, energy_data, output_dir: Path = None):
    """使用面向对象方式生成所有图表"""
    
    # 创建调度器
    scheduler = create_scheduler_with_configs()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要生成的图表列表
    plot_sequence = [
        'vertical_aero_vs_speed',
        'drag_deceleration', 
        'vertical_velocity',
        'vertical_aero_force',
        'mechanical_energy',
        'energy_components',
        'energy_change_rate'
    ]
    
    print("=== 生成面向对象重构的可视化图表 ===")
    
    # 批量生成图表
    generated_figures = {}
    
    for plot_name in plot_sequence:
        print(f"\n生成图表: {plot_name}")
        try:
            # 根据图表类型选择数据源
            if plot_name in ['mechanical_energy', 'energy_components', 'energy_change_rate']:
                data_source = energy_data
                use_energy_data = True
            else:
                data_source = tracks_data
                use_energy_data = False
            
            # 创建对应的绘图器并生成图表
            config = scheduler.configs[plot_name]
            
            if config.name in ['vertical_aero_vs_speed', 'drag_deceleration']:
                # 散点图
                from src.visualization.plotter_base import ScatterPlotter
                plotter = ScatterPlotter(config)
                
                if plot_name == 'vertical_aero_vs_speed':
                    fig = plotter.plot(data_source, x_key='v_xy_sq', y_key='f_z_aero')
                else:  # drag_deceleration
                    fig = plotter.plot(data_source, x_key='v_total_sq', y_key='a_drag_est')
                    
            elif plot_name == 'vertical_velocity':
                # 时间序列图 - 需要特殊处理DataFrame数据
                from src.visualization.plotter_base import TimeSeriesPlotter
                plotter = TimeSeriesPlotter(config)
                
                # 重新组织数据格式
                reformatted_data = {}
                for track_name in data_source.keys():
                    df_track = df_all[df_all.track == track_name].sort_values(by="t")
                    if len(df_track) >= 10:
                        reformatted_data[track_name] = (df_track.t.values, df_track.vz.values)
                
                fig = plotter.plot(reformatted_data, show_scatter=False)
                
            elif plot_name == 'vertical_aero_force':
                # 时间序列图
                from src.visualization.plotter_base import TimeSeriesPlotter
                plotter = TimeSeriesPlotter(config)
                fig = plotter.plot(data_source, x_key='t', y_key='f_z_aero')
                
            elif plot_name in ['mechanical_energy', 'energy_components', 'energy_change_rate']:
                # 能量相关图表
                from src.visualization.plotter_base import TimeSeriesPlotter
                plotter = TimeSeriesPlotter(config)
                
                # 重新组织能量数据
                reformatted_energy = {}
                for track_name, (t, total_e, kin_e, pot_e) in energy_data.items():
                    if plot_name == 'mechanical_energy':
                        y_data = total_e
                    elif plot_name == 'energy_components':
                        # 这个比较复杂，需要特殊处理
                        continue  # 暂时跳过
                    else:  # energy_change_rate
                        y_data = np.gradient(total_e, t)
                    
                    reformatted_energy[track_name] = (t, y_data)
                
                if reformatted_energy:  # 只有当有数据时才绘制
                    fig = plotter.plot(reformatted_energy, show_scatter=False)
                else:
                    continue
                    
            # 显示和保存
            plotter.show()
            generated_figures[plot_name] = fig
            
            if output_dir:
                output_file = output_dir / f"{plot_name}.png"
                plotter.save(output_file)
                print(f"  已保存: {output_file}")
                
        except Exception as e:
            print(f"  生成失败: {e}")
            continue
    
    return generated_figures


def main():
    """主函数"""
    # 设置样式
    setup_debug_style()
    
    # 加载数据
    df_all, tracks_data, energy_data = load_and_prepare_data()
    
    if df_all is None:
        return
    
    # 打印分析摘要
    print("\n=== 能量守恒分析 ===")
    print("理想情况（无空气阻力）:")
    print("  - 总机械能E/m应保持恒定")
    print("  - 能量变化率dE/dt应接近0")
    print("实际情况（有空气阻力）:")
    print("  - 总机械能E/m应随时间逐渐减少")
    print("  - 能量变化率dE/dt应为负值（能量耗散）")
    print("  - 动能和势能之间应有明显转换")
    
    # 生成图表
    output_dir = Path("out/plots")
    figures = generate_plots_oo(df_all, tracks_data, energy_data, output_dir)
    
    print(f"\n=== 分析完成 ===")
    print(f"共生成 {len(figures)} 个图表")
    if output_dir.exists():
        print(f"图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()