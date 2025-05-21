import numpy as np
from data_utils import read_csv, apply_kalman_filter, analyze_noise_reduction, export_filtered_data
from plot_utils import plot_3d_points, plot_3d_curve, plot_trajectory_analysis

def main():
    # 读取原始数据
    print("读取原始数据...")
    original_data = read_csv()  # 默认读取ps1.csv1
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
    plot_3d_points(original_data, title="原始回旋镖轨迹数据点")
    plot_trajectory_analysis(original_data, title="原始回旋镖轨迹分析")
    
    # 绘制滤波后的数据
    print("\n绘制滤波后数据可视化...")
    plot_3d_points(filtered_data, title="滤波后的回旋镖轨迹数据点")
    plot_trajectory_analysis(filtered_data, title="滤波后的回旋镖轨迹分析")

    print("\n8. 导出滤波后的数据...")
    export_filtered_data(filtered_data)

def main_with_curve_fit():
    """使用六阶傅里叶-三阶多项式方法进行曲线拟合"""
    from curve_fit_utils import fit_3d_curve
    from plot_utils import plot_fourier_fit
    
    # 读取滤波后的数据
    print("读取滤波后的数据...")
    filtered_data = read_csv("betterps.csv")
    origin_data = read_csv("ps.csv")
    
    # 进行曲线拟合
    print("\n执行六阶傅里叶-三阶多项式拟合...")
    coeffs, fitted_funcs, expressions = fit_3d_curve(filtered_data, n_fourier=6, n_poly=3)
    
    # 打印曲线解析式
    print("\n拟合曲线的解析式：")
    for expr in expressions:
        print(expr)
    
    # 绘制拟合结果
    print("\n绘制拟合结果...")
    plot_fourier_fit(origin_data, fitted_funcs, 
                    title="回旋镖轨迹：原始数据和拟合曲线",
                    save_path="fit_results_fourier_polynomial.png")
    
    print("\n拟合结果已保存到 fit_results_fourier_polynomial.png")

def main_with_dual_trajectories():
    """比较两个轨迹数据"""
    from plot_utils import plot_dual_trajectories
    
    # 读取第一个轨迹数据
    print("读取第一个轨迹数据(ps.csv)...")
    data1_original = read_csv("ps.csv")
    
    # 读取第二个轨迹数据
    print("读取第二个轨迹数据(ps2.csv)...")
    data2_original = read_csv("ps2.csv")
    
    # 应用卡尔曼滤波进行降噪
    print("对第一个轨迹应用卡尔曼滤波...")
    data1_filtered = apply_kalman_filter(data1_original)
    
    print("对第二个轨迹应用卡尔曼滤波...")
    data2_filtered = apply_kalman_filter(data2_original)
    
    # 绘制双轨迹对比图（原始数据）
    print("\n绘制原始轨迹对比图...")
    plot_dual_trajectories(data1_original, data2_original, 
                          title="回旋镖轨迹对比(原始数据)",
                          save_path="trajectory_comparison_original.png")
    
    print("原始对比图已保存到 trajectory_comparison_original.png")
    
    # 绘制双轨迹对比图（滤波后数据）
    print("\n绘制滤波后轨迹对比图...")
    plot_dual_trajectories(data1_filtered, data2_filtered, 
                          title="回旋镖轨迹对比(滤波后)",
                          save_path="trajectory_comparison_filtered.png")
    
    print("滤波后对比图已保存到 trajectory_comparison_filtered.png")

if __name__ == "__main__":
    print("请选择要运行的功能：")
    print("1. 原始数据处理和滤波")
    print("2. 曲线拟合")
    print("3. 轨迹对比")
    choice = input("请输入选项（1、2或3）：")
    
    if choice == "1":
        main()
    elif choice == "2":
        main_with_curve_fit()
    elif choice == "3":
        main_with_dual_trajectories()
    else:
        print("无效的选项！")