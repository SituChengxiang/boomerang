import numpy as np
from scipy.optimize import curve_fit
from data_utils import read_csv, analyze_noise_reduction, params_fixed
from plot_utils import plot_comparison

def physical_model(t, v0, theta, phi, C_L, C_D):
    """
    基于物理模型的回旋镖轨迹拟合。
    
    参数:
    v0: 初始速度 (m/s)
    theta: 投掷角度 (rad)
    phi: 方位角 (rad)
    C_L: 升力系数
    C_D: 阻力系数
    
    物理参数（从params_fixed获取）:
    a: 翼展 (m)
    d: 翼宽 (m)
    m: 质量 (kg)
    I: 转动惯量 (kg*m²)
    omega: 固定角速度 (rad/s)
    rho: 空气密度 (kg/m³)
    g: 重力加速度 (m/s²)
    """
    # 从params_fixed获取物理参数
    a = params_fixed['a']        # 翼展
    d = params_fixed['d']        # 翼宽
    m = params_fixed['m']        # 质量
    I = params_fixed['I']        # 转动惯量
    omega = params_fixed['omega'] # 固定角速度
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
    
    # 计算轨迹
    # 使用简化的物理模型，考虑升力、阻力和重力
    x = vx0 * t - (k_D / m) * vx0 * t**2
    y = vy0 * t - (k_D / m) * vy0 * t**2
    z = vz0 * t - 0.5 * g * t**2 - (k_D / m) * vz0 * t**2 + (k_L / m) * omega * t**2
    
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
        10.0,   # v0 - 初始速度 (m/s)
        0.2,    # theta - 投掷角度 (rad)
        0.1,    # phi - 方位角 (rad)
        1.2,    # C_L - 升力系数
        0.1     # C_D - 阻力系数
    ]
    
    try:
        # 进行拟合
        print("正在拟合数据...")
        params, _ = curve_fit(
            lambda t, *p: physical_model(t, *p)[:, 1:4].ravel(), 
            t_data, 
            xyz_data.ravel(), 
            p0=initial_guess,
            maxfev=1000000
        )
        
        # 提取拟合参数
        v0_fit, theta_fit, phi_fit, C_L_fit, C_D_fit = params
        
        # 打印拟合参数
        print("\n拟合参数:")
        print(f"v0 (初始速度): {v0_fit:.6f} m/s")
        print(f"theta (投掷角度): {theta_fit:.6f} rad ({np.degrees(theta_fit):.2f}°)")
        print(f"phi (方位角): {phi_fit:.6f} rad ({np.degrees(phi_fit):.2f}°)")
        print(f"C_L (升力系数): {C_L_fit:.6f}")
        print(f"C_D (阻力系数): {C_D_fit:.6f}")
        
        # 打印固定物理参数
        print("\n固定物理参数:")
        print(f"a (翼展): {params_fixed['a']} m")
        print(f"d (翼宽): {params_fixed['d']} m")
        print(f"m (质量): {params_fixed['m']} kg")
        print(f"I (转动惯量): {params_fixed['I']} kg*m²")
        print(f"omega (角速度): {params_fixed['omega']} rad/s")
        print(f"rho (空气密度): {params_fixed['rho']} kg/m³")
        print(f"g (重力加速度): {params_fixed['g']} m/s²")
        
        # 生成拟合数据
        fitted_data = physical_model(t_data, *params)
        
        # 分析误差
        stats = analyze_noise_reduction(data, fitted_data)
        print("\n拟合误差统计:")
        print(f"平均偏差: {stats['mean_difference']:.6f} m")
        print(f"最大偏差: {stats['max_difference']:.6f} m")
        print(f"标准差: {stats['std_difference']:.6f} m")
        
        # 绘制对比图
        print("\n绘制拟合结果...")
        plot_comparison(fitted_data, title="Physical Model Fit Result")
        
    except Exception as e:
        print(f"拟合过程中出错: {e}")

if __name__ == "__main__":
    main()