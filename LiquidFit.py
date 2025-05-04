import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_utils import read_csv, get_initial_conditions, params_fixed
from plot_utils import plot_comparison

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
    g = params_fixed['g']
    return vz0 * t - 0.5 * g * t**2 - beta * t**2 + z0

# 合并的函数用于curve_fit
def combined_function(t, C, k_v, k_omega, v_y0, vz0, beta, z0):
    """
    合并函数，用于三维拟合
    """
    x = x_t(t, C, k_v, k_omega)
    y = y_t(t, v_y0, k_v)
    z = z_t(t, vz0, beta, z0)
    return x, y, z

# 读取数据并获取初始条件
data = read_csv('ps.csv')
t_data, xyz_data, _ = get_initial_conditions(data)
x_data = xyz_data[:, 0]
y_data = xyz_data[:, 1]
z_data = xyz_data[:, 2]

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
    print(f"  g (固定值): {params_fixed['g']}")
    print(f"回旋镖质量: {params_fixed['m']} kg")

    # 计算拟合后的曲线点
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    x_fit, y_fit, z_fit = combined_function(t_fit, *params)

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

# 绘制对比图
data_3d = xyz_data
data_xt = np.column_stack((t_data, x_data))
data_yt = np.column_stack((t_data, y_data))
data_zt = np.column_stack((t_data, z_data))
plot_comparison(data_3d, data_xt, data_yt, data_zt, title="Liquid Fit Result")