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

# 定义傅里叶-多项式混合函数用于x(t)、y(t)和z(t)拟合
def fourier_poly(t, a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, d0, d1, d2):
    """
    傅里叶-多项式混合函数:
    f(t) = a0 + a1*cos(w*t) + a2*cos(2*w*t) + ... + a6*cos(6*w*t) + 
           b1*sin(w*t) + b2*sin(2*w*t) + ... + b6*sin(6*w*t) +
           c1*t + c2*t^2 + c3*t^3 + d0*exp(-d1*t) + d2
    
    参数:
    a0~a6: 余弦项系数
    b1~b6: 正弦项系数
    c1~c3: 多项式项系数
    d0,d1,d2: 指数衰减项相关参数
    """
    w = 2*np.pi  # 基本频率
    fourier_part = a0 + a1*np.cos(w*t) + a2*np.cos(2*w*t) + a3*np.cos(3*w*t) + \
                  a4*np.cos(4*w*t) + a5*np.cos(5*w*t) + a6*np.cos(6*w*t) + \
                  b1*np.sin(w*t) + b2*np.sin(2*w*t) + b3*np.sin(3*w*t) + \
                  b4*np.sin(4*w*t) + b5*np.sin(5*w*t) + b6*np.sin(6*w*t)
    poly_part = c1*t + c2*t**2 + c3*t**3
    exp_part = d0*np.exp(-d1*t) + d2
    
    return fourier_part + poly_part + exp_part

# 合并的函数用于curve_fit
def combined_function(t, 
                     x_a0, x_a1, x_a2, x_a3, x_a4, x_a5, x_a6, 
                     x_b1, x_b2, x_b3, x_b4, x_b5, x_b6, 
                     x_c1, x_c2, x_c3, x_d0, x_d1, x_d2,
                     
                     y_a0, y_a1, y_a2, y_a3, y_a4, y_a5, y_a6,
                     y_b1, y_b2, y_b3, y_b4, y_b5, y_b6,
                     y_c1, y_c2, y_c3, y_d0, y_d1, y_d2,
                     
                     z_a0, z_a1, z_a2, z_a3, z_a4, z_a5, z_a6,
                     z_b1, z_b2, z_b3, z_b4, z_b5, z_b6,
                     z_c1, z_c2, z_c3, z_d0, z_d1, z_d2):
    x = fourier_poly(t, 
                     x_a0, x_a1, x_a2, x_a3, x_a4, x_a5, x_a6, 
                     x_b1, x_b2, x_b3, x_b4, x_b5, x_b6, 
                     x_c1, x_c2, x_c3, x_d0, x_d1, x_d2)
    
    y = fourier_poly(t, 
                     y_a0, y_a1, y_a2, y_a3, y_a4, y_a5, y_a6,
                     y_b1, y_b2, y_b3, y_b4, y_b5, y_b6,
                     y_c1, y_c2, y_c3, y_d0, y_d1, y_d2)
    
    z = fourier_poly(t, 
                     z_a0, z_a1, z_a2, z_a3, z_a4, z_a5, z_a6,
                     z_b1, z_b2, z_b3, z_b4, z_b5, z_b6,
                     z_c1, z_c2, z_c3, z_d0, z_d1, z_d2)
    
    return x, y, z

# 读取数据
data = read_csv('ps.csv')
t_data = data['t'].values
x_data = data['x'].values
y_data = data['y'].values
z_data = data['z'].values

# 初始参数猜测
initial_guess = [
    # x坐标的傅里叶-多项式参数
    0.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001,  # x_a0 ~ x_a6
    0.5, 0.1, 0.05, 0.01, 0.005, 0.001,       # x_b1 ~ x_b6
    0.5, 0.1, 0.01,                          # x_c1, x_c2, x_c3
    0.5, 0.5, 0.0,                           # x_d0, x_d1, x_d2
    
    # y坐标的傅里叶-多项式参数
    0.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001,  # y_a0 ~ y_a6
    0.5, 0.1, 0.05, 0.01, 0.005, 0.001,       # y_b1 ~ y_b6
    0.5, 0.1, 0.01,                          # y_c1, y_c2, y_c3
    0.5, 0.5, 0.0,                           # y_d0, y_d1, y_d2
    
    # z坐标的傅里叶-多项式参数
    1.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005,  # z_a0 ~ z_a6
    0.1, 0.05, 0.01, 0.005, 0.001, 0.0005,       # z_b1 ~ z_b6
    1.7, -5.0, 0.01,                            # z_c1, z_c2, z_c3
    0.1, 1.0, 1.0                               # z_d0, z_d1, z_d2
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
    x_params = params[0:19]
    y_params = params[19:38]
    z_params = params[38:57]
    
    # 打印result
    param_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
                   'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 
                   'c1', 'c2', 'c3', 'd0', 'd1', 'd2']
    
    print("X方向傅里叶-多项式参数:")
    for i, name in enumerate(param_names):
        print(f"  x_{name}: {x_params[i]}")
        
    print("\nY方向傅里叶-多项式参数:")
    for i, name in enumerate(param_names):
        print(f"  y_{name}: {y_params[i]}")
    
    print("\nZ方向傅里叶-多项式参数:")
    for i, name in enumerate(param_names):
        print(f"  z_{name}: {z_params[i]}")
        
    print(f"\n回旋镖质量: {M_BOOMERANG} kg")

    # 计算拟合后的曲线点
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    x_fit, y_fit, z_fit = combined_function(t_fit, *params)

    # 可视化result
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x_data, y_data, z_data, c='r', marker='o', label='data Points')
    ax1.plot(x_fit, y_fit, z_fit, 'b-', label='fit curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Fourier Fit Result')
    ax1.legend()
    
    # X-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t_data, x_data, 'ro', label='data Points')
    ax2.plot(t_fit, x_fit, 'b-', label='fit curve')
    ax2.set_xlabel('time t (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t) result')
    ax2.grid(True)
    ax2.legend()
    
    # Y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t_data, y_data, 'ro', label='data Points')
    ax3.plot(t_fit, y_fit, 'b-', label='fit curve')
    ax3.set_xlabel('time t (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t) result')
    ax3.grid(True)
    ax3.legend()
    
    # Z-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t_data, z_data, 'ro', label='data Points')
    ax4.plot(t_fit, z_fit, 'b-', label='fit curve')
    ax4.set_xlabel('time t (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z(t) result')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('fit_results_fourier_all_axes.png', dpi=300)
    plt.show()

    # 计算拟合误差
    x_fit_data = fourier_poly(t_data, *x_params)
    y_fit_data = fourier_poly(t_data, *y_params)
    z_fit_data = fourier_poly(t_data, *z_params)
    
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