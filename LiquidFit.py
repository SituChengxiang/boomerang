import numpy as np
from scipy.optimize import curve_fit
from data_utils import read_csv, analyze_noise_reduction, params_fixed
from plot_utils import plot_comparison

def x_t(t, C, k_v, k_omega):
    """
    x(t) = (C / (k_v * (k_v + k_omega))) * (1 - exp(-(2*k_v + k_omega)*t))
    
    参数:
    C: 代表C_L * rho * A * d * r * v_y0^2 / I的综合参数
    k_v: 线速度衰减系数
    k_omega: 角速度衰减系数
    """
    return (C / (k_v * (k_v + k_omega))) * (1 - np.exp(-(2*k_v + k_omega)*t))

def y_t(t, v_y0, k_v):
    """
    y(t) = (v_y0 / k_v) * (1 - exp(-k_v * t))
    
    参数:
    v_y0: y方向初速度
    k_v: 线速度衰减系数
    """
    return (v_y0 / k_v) * (1 - np.exp(-k_v * t))

def z_t(t, vz0, beta, z0):
    """
    z(t) = vz0*t - 0.5*g*t^2 - beta*t^2 + z0
    
    参数:
    vz0: z方向初速度
    beta: 额外的衰减系数
    z0: 初始高度
    """
    g = params_fixed['g']
    return vz0 * t - 0.5 * g * t**2 - beta * t**2 + z0

def combined_function(t, C, k_v, k_omega, v_y0, vz0, beta, z0):
    """合并函数，用于三维拟合"""
    x = x_t(t, C, k_v, k_omega)
    y = y_t(t, v_y0, k_v)
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
        0.2,    # C - 综合物理参数
        0.5,    # k_v - 线速度衰减系数
        0.3,    # k_omega - 角速度衰减系数
        1.0,    # v_y0 - y方向初速度
        1.7,    # vz0 - z方向初速度
        0.5,    # beta - z方向额外衰减
        1.0     # z0 - 初始高度
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
        C_fit, k_v_fit, k_omega_fit, v_y0_fit, vz0_fit, beta_fit, z0_fit = params
        
        # 打印拟合参数
        print("\n拟合参数:")
        print(f"C (综合物理参数): {C_fit:.6f}")
        print(f"k_v (线速度衰减): {k_v_fit:.6f}")
        print(f"k_omega (角速度衰减): {k_omega_fit:.6f}")
        print(f"v_y0 (y方向初速度): {v_y0_fit:.6f}")
        print(f"vz0 (z方向初速度): {vz0_fit:.6f}")
        print(f"beta (z方向额外衰减): {beta_fit:.6f}")
        print(f"z0 (初始高度): {z0_fit:.6f}")
        
        # 生成拟合数据
        fitted_data = combined_function(t_data, *params)
        
        # 分析误差
        stats = analyze_noise_reduction(data, fitted_data)
        print("\n拟合误差统计:")
        print(f"平均偏差: {stats['mean_difference']:.6f} m")
        print(f"最大偏差: {stats['max_difference']:.6f} m")
        print(f"标准差: {stats['std_difference']:.6f} m")
        
        # 绘制对比图
        print("\n绘制拟合结果...")
        plot_comparison(fitted_data, title="Liquid Fit Result")
        
    except Exception as e:
        print(f"拟合过程中出错: {e}")

if __name__ == "__main__":
    main()