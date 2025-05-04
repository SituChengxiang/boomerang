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

# 定义x(t)的解析式 - 更复杂的空气动力学模型
def x_t(t, R_0, omega, k_omega, beta_CL_rho_A):
    """
    x(t) = R_0 * (1 - cos(omega*t)) * exp(-k_omega*t) + beta_CL_rho_A * (omega*R_0)^2 * t^2
    
    参数:
    R_0: 初始回旋半径
    omega: 角频率
    k_omega: 角速度衰减系数
    beta_CL_rho_A: 代表beta*C_L*rho*A的综合参数
    """
    term1 = R_0 * (1 - np.cos(omega * t)) * np.exp(-k_omega * t)
    term2 = beta_CL_rho_A * (omega * R_0)**2 * t**2
    return term1 + term2

# 定义y(t)的解析式 - 更复杂的空气动力学模型
def y_t(t, v0_sin_theta0, k_v, CL_omega_R0_squared):
    """
    y(t) = (v0_sin_theta0 / k_v) * (1 - exp(-k_v*t)) - CL_omega_R0_squared
    
    参数:
    v0_sin_theta0: v_0 * sin(theta_0)，初始速度在y方向的分量
    k_v: 线速度衰减系数
    CL_omega_R0_squared: C_L * omega * R_0^2，常数项
    """
    term1 = (v0_sin_theta0 / k_v) * (1 - np.exp(-k_v * t))
    return term1 - CL_omega_R0_squared

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
def combined_function(t, R_0, omega, k_omega, beta_CL_rho_A, v0_sin_theta0, k_v, CL_omega_R0_squared, vz0, beta, z0):
    """
    合并函数，用于三维拟合
    """
    x = x_t(t, R_0, omega, k_omega, beta_CL_rho_A)
    y = y_t(t, v0_sin_theta0, k_v, CL_omega_R0_squared)
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
    1.0,    # R_0 - 初始回旋半径
    10.0,   # omega - 角频率
    0.5,    # k_omega - 角速度衰减系数
    0.01,   # beta_CL_rho_A - 空气动力学综合参数
    2.0,    # v0_sin_theta0 - 初速度y方向分量
    0.5,    # k_v - 线速度衰减系数
    0.1,    # CL_omega_R0_squared - 常数项
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
    R_0_fit, omega_fit, k_omega_fit, beta_CL_rho_A_fit = params[0:4]
    v0_sin_theta0_fit, k_v_fit, CL_omega_R0_squared_fit = params[4:7]
    vz0_fit, beta_fit, z0_fit = params[7:10]
    
    # 打印result
    print("高级空气动力学模型拟合参数:")
    print("\nX方向参数:")
    print(f"  R_0: {R_0_fit} m (初始回旋半径)")
    print(f"  omega: {omega_fit} rad/s (角频率)")
    print(f"  k_omega: {k_omega_fit} 1/s (角速度衰减系数)")
    print(f"  beta*C_L*rho*A: {beta_CL_rho_A_fit} (空气动力学综合参数)")
    
    print("\nY方向参数:")
    print(f"  v0*sin(theta0): {v0_sin_theta0_fit} m/s (初速度y分量)")
    print(f"  k_v: {k_v_fit} 1/s (线速度衰减系数)")
    print(f"  C_L*omega*R_0^2: {CL_omega_R0_squared_fit} m (常数项)")
    
    print("\nZ方向参数:")
    print(f"  vz0: {vz0_fit} m/s (z方向初速度)")
    print(f"  beta: {beta_fit} m/s² (z方向额外衰减)")
    print(f"  z0: {z0_fit} m (初始高度)")
    print(f"  g (固定值): {G_HANGZHOU} m/s²")
    print(f"回旋镖质量: {M_BOOMERANG} kg")

    # 计算衍生物理参数
    estimated_v0 = v0_sin_theta0_fit / np.sin(np.pi/4)  # 假设发射角为45度
    estimated_CL = CL_omega_R0_squared_fit / (omega_fit * R_0_fit**2)
    
    print("\n估计的物理参数:")
    print(f"  估计初始速度(v0): {estimated_v0:.2f} m/s (假设发射角为45°)")
    print(f"  估计升力系数(C_L): {estimated_CL:.4f}")

    # 计算拟合后的曲线点
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    x_fit, y_fit, z_fit = combined_function(t_fit, *params)

    # 可视化result
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x_data, y_data, z_data, c='r', marker='o', label='data Points')
    ax1.plot(x_fit, y_fit, z_fit, 'b-', label='fit Curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Fourier-Liquid Fit Result')
    ax1.legend()
    
    # X-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t_data, x_data, 'ro', label='data Points')
    ax2.plot(t_fit, x_fit, 'b-', label='fit Curve')
    ax2.set_xlabel('time t (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t) result')
    ax2.grid(True)
    ax2.legend()
    
    # Y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t_data, y_data, 'ro', label='data Points')
    ax3.plot(t_fit, y_fit, 'b-', label='fit Curve')
    ax3.set_xlabel('time t (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t) result')
    ax3.grid(True)
    ax3.legend()
    
    # Z-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t_data, z_data, 'ro', label='data Points')
    ax4.plot(t_fit, z_fit, 'b-', label='fit Curve')
    ax4.set_xlabel('time t (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z(t) result')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('fit_results_advanced_aerodynamic.png', dpi=300)
    plt.show()

    # 计算拟合误差
    x_fit_data = x_t(t_data, R_0_fit, omega_fit, k_omega_fit, beta_CL_rho_A_fit)
    y_fit_data = y_t(t_data, v0_sin_theta0_fit, k_v_fit, CL_omega_R0_squared_fit)
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