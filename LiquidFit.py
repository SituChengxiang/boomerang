import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 固定重力加速度为杭州地区的值（单位：m/s²）
G_HANGZHOU = 9.78
# 回旋镖质量（单位：kg）
M_BOOMERANG = 0.002183  # 2.183g

# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 定义x(t)的解析式 - 基于空气动力学模型
def x_t(t, C, k_v, k_omega):
    """
    x(t) = (C / (k_v * (k_v + k_omega))) * (1 - exp(-(2*k_v + k_omega)*t))
    
    参数:
    C: 代表C_L * rho * A * d * r * v_y0^2 / I的综合参数
    k_v: 线速度衰减系数
    k_omega: 角速度衰减系数
    """
    return (C / (k_v * (k_v + k_omega))) * (1 - np.exp(-(2*k_v + k_omega)*t))

# 定义y(t)的解析式 - 基于空气动力学模型
def y_t(t, v_y0, k_v):
    """
    y(t) = (v_y0 / k_v) * (1 - exp(-k_v * t))
    
    参数:
    v_y0: y方向初速度
    k_v: 线速度衰减系数
    """
    return (v_y0 / k_v) * (1 - np.exp(-k_v * t))

# 定义z(t)的解析式，使用固定的重力加速度
def z_t(t, vz0, beta, z0):
    """
    z(t) = vz0*t - 0.5*g*t^2 - beta*t^2 + z0
    
    参数:
    vz0: z方向初速度
    beta: 额外的衰减系数
    z0: 初始高度
    """
    return vz0 * t - 0.5 * G_HANGZHOU * t**2 - beta * t**2 + z0

# 合并的函数用于curve_fit
def combined_function(t, C, k_v, k_omega, v_y0, vz0, beta, z0):
    """
    合并函数，用于三维拟合
    """
    x = x_t(t, C, k_v, k_omega)
    y = y_t(t, v_y0, k_v)
    z = z_t(t, vz0, beta, z0)
    return x, y, z

# 读取数据
data = read_csv('ps.csv')
t_data = data['t'].values
x_data = data['x'].values
y_data = data['y'].values
z_data = data['z'].values

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

# 进行拟合
try:
    params, covariance = curve_fit(
        lambda t, *params: 
        np.concatenate(combined_function(t, *params)), 
        t_data, 
        np.concatenate((x_data, y_data, z_data)), 
        p0=initial_guess,
        maxfev=1000000  # 增加最大迭代次数
    )
    
    # 提取拟合参数
    C_fit, k_v_fit, k_omega_fit, v_y0_fit, vz0_fit, beta_fit, z0_fit = params
    
    # 打印
    print("空气动力学模型拟合参数:")
    print(f"  C: {C_fit} (C_L·ρ·A·d·r·v_y0²/I)")
    print(f"  k_v: {k_v_fit} (线速度衰减系数)")
    print(f"  k_omega: {k_omega_fit} (角速度衰减系数)")
    print(f"  v_y0: {v_y0_fit} (y方向初速度)")
    print(f"  vz0: {vz0_fit} (z方向初速度)")
    print(f"  beta: {beta_fit} (z方向额外衰减)")
    print(f"  z0: {z0_fit} (初始高度)")
    print(f"  g (固定值): {G_HANGZHOU}")
    print(f"回旋镖质量: {M_BOOMERANG} kg")

    # 计算拟合后的曲线点
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    x_fit, y_fit, z_fit = combined_function(t_fit, *params)

    # 可视化
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x_data, y_data, z_data, c='r', marker='o', label='Data Points')
    ax1.plot(x_fit, y_fit, z_fit, 'b-', label='Fit Curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trace')
    ax1.legend()
    
    # X-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t_data, x_data, 'ro', label='Data Points')
    ax2.plot(t_fit, x_fit, 'b-', label='Fit Curve')
    ax2.set_xlabel(' t (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t) ')
    ax2.grid(True)
    ax2.legend()
    
    # Y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t_data, y_data, 'ro', label='Data Points')
    ax3.plot(t_fit, y_fit, 'b-', label='Fit Curve')
    ax3.set_xlabel(' t (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t) ')
    ax3.grid(True)
    ax3.legend()
    
    # Z-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t_data, z_data, 'ro', label='Data Points')
    ax4.plot(t_fit, z_fit, 'b-', label='Fit Curve')
    ax4.set_xlabel(' t (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z(t) ')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('fit_results_aerodynamic.png', dpi=300)
    plt.show()

    # 计算拟合误差
    x_fit_data = x_t(t_data, C_fit, k_v_fit, k_omega_fit)
    y_fit_data = y_t(t_data, v_y0_fit, k_v_fit)
    z_fit_data = z_t(t_data, vz0_fit, beta_fit, z0_fit)
    
    x_error = np.mean((x_fit_data - x_data)**2)
    y_error = np.mean((y_fit_data - y_data)**2)
    z_error = np.mean((z_fit_data - z_data)**2)
    
    print("\n均方误差:")
    print(f"  X均方误差: {x_error}")
    print(f"  Y均方误差: {y_error}")
    print(f"  Z均方误差: {z_error}")
    print(f"  总均方误差: {(x_error + y_error + z_error)/3}")
    
except RuntimeError as e:
    print(f"拟合过程中出现错误: {e}")
    print("请尝试修改初始参数或增加maxfev值")

except ValueError as e:
    print(f"值错误: {e}")
    print("可能是模型参数导致计算溢出，请尝试调整初始参数范围")