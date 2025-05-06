import numpy as np
from scipy.optimize import curve_fit
from data_utils import read_csv, analyze_noise_reduction, params_fixed
from plot_utils import plot_comparison

def y_t(t, R, omega, alpha, vy0, y0):
    """
    y(t) = R * (1 - cos(omega * t)) * exp(-alpha * t) + vy0 * t + y0
    
    参数:
    R: 回转半径
    omega: 角频率
    alpha: 旋转衰减率
    vy0: y方向初速度
    y0: 初始y位置
    """
    return R * (1 - np.cos(omega * t)) * np.exp(-alpha * t) + vy0 * t + y0

def x_t(t, R, omega, alpha, vx0, x0):
    """
    x(t) = R * sin(omega * t) * exp(-alpha * t) + vx0 * t + x0
    
    参数:
    R: 回转半径
    omega: 角频率
    alpha: 旋转衰减率
    vx0: x方向初速度
    x0: 初始x位置
    """
    return R * np.sin(omega * t) * np.exp(-alpha * t) + vx0 * t + x0

def z_t(t, vz0, beta, z0):
    """
    z(t) = vz0 * t - 0.5 * g * t^2 - beta * t^2 + z0
    
    参数:
    vz0: z方向初速度
    beta: 额外衰减系数
    z0: 初始高度
    """
    return vz0 * t - 0.5 * params_fixed['g'] * t**2 - beta * t**2 + z0

def combined_function(t, R, omega, alpha, vx0, vy0, vz0, beta, x0, y0, z0):
    """合并函数，用于三维拟合"""
    x = x_t(t, R, omega, alpha, vx0, x0)
    y = y_t(t, R, omega, alpha, vy0, y0)
    z = z_t(t, vz0, beta, z0)
    return np.column_stack([t, x, y, z])

def main():
    # 读取数据
    print("读取原始数据...")
    data = read_csv()
    
    # 提取时间和坐标数据
    t_data = data[:, 0]
    xyz_data = data[:, 1:4]
    
    # 初始参数猜测
    initial_guess = [
        4.0,    # R - 回转半径
        10.0,   # omega - 角频率
        0.5,    # alpha - 旋转衰减率
        3.5,    # vx0 - x方向初速度
        -0.40,  # vy0 - y方向初速度
        1.7,    # vz0 - z方向初速度
        0.5,    # beta - z方向衰减率
        1.2,    # x0 - 初始x位置
        0.1,    # y0 - 初始y位置
        1.0     # z0 - 初始z位置
    ]
    
    try:
        # 进行拟合
        print("正在拟合数据...")
        params, _ = curve_fit(
            lambda t, *p: combined_function(t, *p)[:, 1:4].ravel(), 
            t_data, 
            xyz_data.ravel(), 
            p0=initial_guess,
            maxfev=1000000
        )
        
        # 提取拟合参数
        R_fit, omega_fit, alpha_fit, vx0_fit, vy0_fit, vz0_fit, beta_fit, x0_fit, y0_fit, z0_fit = params
        
        # 打印拟合参数
        print("\n拟合参数:")
        print(f"R (回转半径): {R_fit:.6f} m")
        print(f"omega (角频率): {omega_fit:.6f} rad/s")
        print(f"alpha (旋转衰减率): {alpha_fit:.6f} 1/s")
        print(f"vx0 (x方向初速度): {vx0_fit:.6f} m/s")
        print(f"vy0 (y方向初速度): {vy0_fit:.6f} m/s")
        print(f"vz0 (z方向初速度): {vz0_fit:.6f} m/s")
        print(f"beta (z方向衰减率): {beta_fit:.6f} m/s²")
        print(f"x0 (初始x位置): {x0_fit:.6f} m")
        print(f"y0 (初始y位置): {y0_fit:.6f} m")
        print(f"z0 (初始z位置): {z0_fit:.6f} m")
        print(f"g (重力加速度): {params_fixed['g']} m/s²")
        print(f"m (回旋镖质量): {params_fixed['m']} kg")
        
        # 生成拟合数据
        fitted_data = combined_function(t_data, *params)
        
        # 分析误差
        stats = analyze_noise_reduction(
            data[:, 1:4],          # 原始位置数据
            fitted_data[:, 1:4],    # 拟合位置数据
            fitted_data[:, 1:4]     # 同时作为模拟数据进行比较
        )
        print("\n拟合误差统计:")
        print("原始数据与拟合数据比较:")
        print(f"   平均差异: {stats['原始-模拟平均差异']:.6f} m")
        print(f"   最大差异: {stats['原始-模拟最大差异']:.6f} m")
        print(f"   标准差: {stats['原始-模拟标准差']:.6f} m")
        print(f"\n拟合改进: {stats['模型改进百分比']:.1f}%")
        
        # 绘制对比图
        print("\n绘制拟合结果...")
        plot_comparison(fitted_data, title="Harmonic Fit Result")
        
    except Exception as e:
        print(f"拟合过程中出错: {e}")

if __name__ == "__main__":
    main()