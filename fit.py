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

# 定义y(t)和x(t)的解析式，调换了x和y的拟合方式
def y_t(t, R, omega, alpha, vy0, y0):
    return R * (1 - np.cos(omega * t)) * np.exp(-alpha * t) + vy0 * t + y0

def x_t(t, R, omega, alpha, vx0, x0):
    return R * np.sin(omega * t) * np.exp(-alpha * t) + vx0 * t + x0

# 定义z(t)的解析式，使用固定的重力加速度，加入初始位置z0
def z_t(t, vz0, beta, z0):
    return vz0 * t - 0.5 * G_HANGZHOU * t**2 - beta * t**2 + z0

# 合并的函数用于curve_fit，加入初始位置参数
def combined_function(t, R, omega, alpha, vx0, vy0, vz0, beta, x0, y0, z0):
    return x_t(t, R, omega, alpha, vx0, x0), y_t(t, R, omega, alpha, vy0, y0), z_t(t, vz0, beta, z0)

# 读取数据
data = read_csv('ps.csv')
t_data = data['t'].values
x_data = data['x'].values
y_data = data['y'].values
z_data = data['z'].values

# 初始参数猜测
initial_guess = [4,
                 10,  
                 0.5,
                 3.5,  # vx0 (原来是vy0的值)
                 -0.40, # vy0 (原来是vx0的值)
                 1.7, 
                 0.5, 
                 1.2,  # x0 (原来是y0的值)
                 0.1,  # y0 (原来是x0的值)
                 1.0]
# 1 回转半径 2 角频率 3 旋转衰减率 4 x方向初速度 5 y方向初速度 6 z方向初速度 7 z方向衰减率 8 x初始位置 9 y初始位置 10 z初始位置

# 进行拟合
try:
    params, covariance = curve_fit(
        lambda t, R, omega, alpha, vx0, vy0, vz0, beta, x0, y0, z0: 
        np.concatenate(combined_function(t, R, omega, alpha, vx0, vy0, vz0, beta, x0, y0, z0)), 
        t_data, np.concatenate((x_data, y_data, z_data)), 
        p0=initial_guess,
        maxfev=1000000  # 增加最大迭代次数
    )

    # 提取拟合参数
    R_fit, omega_fit, alpha_fit, vx0_fit, vy0_fit, vz0_fit, beta_fit, x0_fit, y0_fit, z0_fit = params

    # 打印结果
    print(f"R: {R_fit}")
    print(f"omega: {omega_fit}")
    print(f"alpha: {alpha_fit}")
    print(f"vx0: {vx0_fit}")
    print(f"vy0: {vy0_fit}")
    print(f"vz0: {vz0_fit}")
    print(f"beta: {beta_fit}")
    print(f"x0: {x0_fit}")
    print(f"y0: {y0_fit}")
    print(f"z0: {z0_fit}")
    print(f"g (固定值): {G_HANGZHOU}")
    print(f"回旋镖质量: {M_BOOMERANG} kg")

    # 计算拟合后的曲线点
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    x_fit, y_fit, z_fit = combined_function(t_fit, *params)

    # 可视化结果
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x_data, y_data, z_data, c='r', marker='o', label='Data Points')
    ax1.plot(x_fit, y_fit, z_fit, 'b-', label='Fit Curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Boomerang 3D Routine')
    ax1.legend()
    
    # X-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t_data, x_data, 'ro', label='Data Points')
    ax2.plot(t_fit, x_fit, 'b-', label='Fit Curve')
    ax2.set_xlabel('time t (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t) result')
    ax2.legend()
    
    # Y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t_data, y_data, 'ro', label='Data Points')
    ax3.plot(t_fit, y_fit, 'b-', label='Fit Curve')
    ax3.set_xlabel('time t (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t) result')
    ax3.legend()
    
    # Z-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t_data, z_data, 'ro', label='Data Points')
    ax4.plot(t_fit, z_fit, 'b-', label='Fit Curve')
    ax4.set_xlabel('time t (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z(t) result')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('fit_results.png', dpi=300)
    plt.show()

    # 计算拟合误差
    x_error = np.mean((x_t(t_data, R_fit, omega_fit, alpha_fit, vx0_fit, x0_fit) - x_data)**2)
    y_error = np.mean((y_t(t_data, R_fit, omega_fit, alpha_fit, vy0_fit, y0_fit) - y_data)**2)
    z_error = np.mean((z_t(t_data, vz0_fit, beta_fit, z0_fit) - z_data)**2)
    
    print(f"X均方误差: {x_error}")
    print(f"Y均方误差: {y_error}")
    print(f"Z均方误差: {z_error}")
    print(f"总均方误差: {(x_error + y_error + z_error)/3}")
    
except RuntimeError as e:
    print(f"拟合过程中出现错误: {e}")
    print("请尝试修改初始参数或增加maxfev值")