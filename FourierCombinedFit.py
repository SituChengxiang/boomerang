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

def physical_base(t, v0, theta, phi, C_L, C_D):
    """
    基础物理模型。
    
    参数:
    v0: 初始速度
    theta: 投掷角度
    phi: 方位角
    C_L: 升力系数
    C_D: 阻力系数
    """
    # 从params_fixed获取物理参数
    a = params_fixed['a']        # 翼展
    d = params_fixed['d']        # 翼宽
    m = params_fixed['m']        # 质量
    rho = params_fixed['rho']    # 空气密度
    g = params_fixed['g']        # 重力加速度
    
    # 计算初始速度分量
    vx0 = v0 * np.cos(theta) * np.cos(phi)
    vy0 = v0 * np.cos(theta) * np.sin(phi)
    vz0 = v0 * np.sin(theta)
    
    # 特征面积
    A = a * d
    
    # 计算空气动力学参数
    k_L = 0.5 * rho * A * C_L
    k_D = 0.5 * rho * A * C_D
    
    # 计算基础轨迹
    x = vx0 * t - (k_D / m) * vx0 * t**2
    y = vy0 * t - (k_D / m) * vy0 * t**2
    z = vz0 * t - 0.5 * g * t**2 - (k_D / m) * vz0 * t**2
    
    return x, y, z

def combined_model(t, n_terms, v0, theta, phi, C_L, C_D, k_damp, *fourier_coeffs):
    """
    物理模型和傅里叶级数的组合模型。
    
    参数:
    t: 时间数组
    n_terms: 傅里叶级数的阶数
    v0, theta, phi: 初始运动参数
    C_L, C_D: 空气动力系数
    k_damp: 阻尼系数
    fourier_coeffs: 傅里叶系数 [ax0,ax1,bx1,...,ay0,ay1,by1,...]
    """
    # 计算物理基础轨迹
    x_phys, y_phys, z_phys = physical_base(t, v0, theta, phi, C_L, C_D)
    
    # 分离x和y方向的傅里叶系数
    n_fourier_coeffs = 2*n_terms + 1
    x_coeffs = fourier_coeffs[:n_fourier_coeffs]
    y_coeffs = fourier_coeffs[n_fourier_coeffs:]
    
    # 计算傅里叶修正项（带阻尼）
    damping = np.exp(-k_damp * t)
    x_corr = fourier_series(t, n_terms, *x_coeffs) * damping
    y_corr = fourier_series(t, n_terms, *y_coeffs) * damping
    
    # 组合物理模型和修正项
    x = x_phys + x_corr
    y = y_phys + y_corr
    z = z_phys
    
    return np.column_stack([t, x, y, z])

def main():
    # 读取数据
    print("读取原始数据...")
    data = read_csv()
    
    # 提取时间和坐标数据
    t_data = data[:, 0]
    xyz_data = data[:, 1:4]
    
    # 设置傅里叶级数阶数
    n_terms = 2
    
    # 计算每个方向需要的傅里叶系数数量
    n_fourier_coeffs = 2*n_terms + 1
    
    # 初始参数猜测
    initial_guess = [
        10.0,   # v0 - 初始速度
        0.2,    # theta - 投掷角度
        0.1,    # phi - 方位角
        1.2,    # C_L - 升力系数
        0.1,    # C_D - 阻力系数
        0.5     # k_damp - 阻尼系数
    ]
    
    # 添加x和y方向的傅里叶系数初始猜测
    initial_guess.extend([0.1] * (2 * n_fourier_coeffs))
    
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
        v0_fit, theta_fit, phi_fit, C_L_fit, C_D_fit, k_damp_fit = params[:6]
        fourier_coeffs = params[6:]
        
        # 打印物理参数
        print("\n物理模型参数:")
        print(f"v0 (初始速度): {v0_fit:.6f} m/s")
        print(f"theta (投掷角度): {theta_fit:.6f} rad ({np.degrees(theta_fit):.2f}°)")
        print(f"phi (方位角): {phi_fit:.6f} rad ({np.degrees(phi_fit):.2f}°)")
        print(f"C_L (升力系数): {C_L_fit:.6f}")
        print(f"C_D (阻力系数): {C_D_fit:.6f}")
        print(f"k_damp (阻尼系数): {k_damp_fit:.6f}")
        
        # 打印傅里叶系数
        print("\nX方向傅里叶修正系数:")
        for i in range(n_terms + 1):
            if i == 0:
                print(f"a{i}: {fourier_coeffs[i]:.6f}")
            else:
                print(f"a{i}: {fourier_coeffs[2*i-1]:.6f}")
                print(f"b{i}: {fourier_coeffs[2*i]:.6f}")
        
        print("\nY方向傅里叶修正系数:")
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
        plot_comparison(fitted_data, title=f"Combined Physics-Fourier Model (n={n_terms})")
        
    except Exception as e:
        print(f"拟合过程中出错: {e}")

if __name__ == "__main__":
    main()