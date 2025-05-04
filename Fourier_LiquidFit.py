import numpy as np
from scipy.optimize import curve_fit
from data_utils import read_csv, analyze_noise_reduction, params_fixed
from plot_utils import plot_comparison

def fourier_series(t, n_terms, *coeffs):
    """
    计算n_terms阶傅里叶级数。
    
    参数:
    t: 时间数组
    n_terms: 傅里叶级数的阶数
    coeffs: 系数数组 [a0, a1, b1, a2, b2, ...]
    """
    result = coeffs[0]  # a0项
    for i in range(n_terms):
        k = i + 1
        a_k = coeffs[2*i + 1]
        b_k = coeffs[2*i + 2]
        result += a_k * np.cos(k * t) + b_k * np.sin(k * t)
    return result

def liquid_damping(t, k_v):
    """
    计算流体阻尼效应。
    
    参数:
    t: 时间数组
    k_v: 速度衰减系数
    """
    return np.exp(-k_v * t)

def combined_model(t, n_terms, k_v, vz0, beta, z0, *fourier_coeffs):
    """
    傅里叶级数和流体力学的混合模型。
    
    参数:
    t: 时间数组
    n_terms: 傅里叶级数的阶数
    k_v: 速度衰减系数
    vz0: z方向初速度
    beta: z方向额外衰减系数
    z0: 初始高度
    fourier_coeffs: 傅里叶系数 [ax0,ax1,bx1,...,ay0,ay1,by1,...]
    """
    # 分离x和y方向的傅里叶系数
    n_fourier_coeffs = 2*n_terms + 1  # 每个方向的系数数量
    x_coeffs = fourier_coeffs[:n_fourier_coeffs]
    y_coeffs = fourier_coeffs[n_fourier_coeffs:]
    
    # 计算水平面运动（带阻尼的傅里叶级数）
    damping = liquid_damping(t, k_v)
    x = fourier_series(t, n_terms, *x_coeffs) * damping
    y = fourier_series(t, n_terms, *y_coeffs) * damping
    
    # 计算垂直运动（考虑重力和阻力）
    g = params_fixed['g']
    z = vz0 * t - 0.5 * g * t**2 - beta * t**2 + z0
    
    return np.column_stack([t, x, y, z])

def main():
    # 读取数据
    print("读取原始数据...")
    data = read_csv()
    
    # 提取时间和坐标数据
    t_data = data[:, 0]
    xyz_data = data[:, 1:4]
    
    # 设置傅里叶级数阶数
    n_terms = 3
    
    # 计算每个方向需要的傅里叶系数数量
    n_fourier_coeffs = 2*n_terms + 1  # a0 + (a_k, b_k) pairs
    
    # 初始参数猜测
    initial_guess = [
        0.5,    # k_v - 速度衰减系数
        1.7,    # vz0 - z方向初速度
        0.5,    # beta - z方向额外衰减
        1.0     # z0 - 初始高度
    ]
    
    # 添加x方向的傅里叶系数初始猜测
    initial_guess.extend([0.1] * n_fourier_coeffs)  # [ax0,ax1,bx1,ax2,bx2,...]
    
    # 添加y方向的傅里叶系数初始猜测
    initial_guess.extend([0.1] * n_fourier_coeffs)  # [ay0,ay1,by1,ay2,by2,...]
    
    try:
        # 进行拟合
        print(f"正在拟合数据（使用{n_terms}阶傅里叶级数）...")
        params, _ = curve_fit(
            lambda t, *p: combined_model(t, n_terms, *p)[:, 1:4].ravel(), 
            t_data, 
            xyz_data.ravel(), 
            p0=initial_guess,
            maxfev=1000000
        )
        
        # 提取基本参数
        k_v_fit, vz0_fit, beta_fit, z0_fit = params[:4]
        fourier_coeffs = params[4:]
        
        # 打印拟合参数
        print("\n基本参数:")
        print(f"k_v (速度衰减): {k_v_fit:.6f}")
        print(f"vz0 (z方向初速度): {vz0_fit:.6f} m/s")
        print(f"beta (z方向衰减): {beta_fit:.6f}")
        print(f"z0 (初始高度): {z0_fit:.6f} m")
        
        print("\nX方向傅里叶系数:")
        for i in range(n_terms + 1):
            if i == 0:
                print(f"a{i}: {fourier_coeffs[i]:.6f}")
            else:
                print(f"a{i}: {fourier_coeffs[2*i-1]:.6f}")
                print(f"b{i}: {fourier_coeffs[2*i]:.6f}")
        
        print("\nY方向傅里叶系数:")
        offset = n_fourier_coeffs
        for i in range(n_terms + 1):
            if i == 0:
                print(f"a{i}: {fourier_coeffs[offset+i]:.6f}")
            else:
                print(f"a{i}: {fourier_coeffs[offset+2*i-1]:.6f}")
                print(f"b{i}: {fourier_coeffs[offset+2*i]:.6f}")
        
        # 生成拟合数据
        fitted_data = combined_model(t_data, n_terms, *params)
        
        # 分析误差
        stats = analyze_noise_reduction(data, fitted_data)
        print("\n拟合误差统计:")
        print(f"平均偏差: {stats['mean_difference']:.6f} m")
        print(f"最大偏差: {stats['max_difference']:.6f} m")
        print(f"标准差: {stats['std_difference']:.6f} m")
        
        # 绘制对比图
        print("\n绘制拟合结果...")
        plot_comparison(fitted_data, title=f"Fourier-Liquid Model Fit Result (n={n_terms})")
        
    except Exception as e:
        print(f"拟合过程中出错: {e}")

if __name__ == "__main__":
    main()