import numpy as np
from data_utils import read_csv, apply_kalman_filter, analyze_noise_reduction, export_filtered_data
from plot_utils import plot_data_points, plot_curve, plot_comparison

def main():
    # 读取原始数据
    print("读取原始数据...")
    original_data = read_csv()  # 默认读取ps1.csv
    
    # 应用卡尔曼滤波进行降噪
    print("应用卡尔曼滤波进行降噪...")
    filtered_data = apply_kalman_filter(original_data)
    
    # 分析降噪效果
    print("分析降噪效果...")
    noise_stats = analyze_noise_reduction(
        original_data[:, 1:4],     # 原始位置数据
        filtered_data[:, 1:4]      # 滤波后位置数据
    )
    print("\n降噪效果统计：")
    print(f"平均差异: {noise_stats['原始-滤波平均差异']:.6f} m")
    print(f"最大差异: {noise_stats['原始-滤波最大差异']:.6f} m")
    print(f"标准差: {noise_stats['原始-滤波标准差']:.6f} m")
    
    # 绘制原始数据
    print("\n绘制原始数据可视化...")
    plot_data_points(original_data, title="原始回旋镖轨迹数据点")
    plot_comparison(original_data, title="原始回旋镖轨迹分析")
    
    # 绘制滤波后的数据
    print("\n绘制滤波后数据可视化...")
    plot_data_points(filtered_data, title="滤波后的回旋镖轨迹数据点")
    plot_comparison(filtered_data, title="滤波后的回旋镖轨迹分析")

    print("\n8. 导出滤波后的数据...")
    export_filtered_data(filtered_data)

if __name__ == "__main__":
    main()