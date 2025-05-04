import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

# 读取CSV文件中的回力镖飞行数据
print("正在读取回力镖飞行轨迹数据...")
data = pd.read_csv('ps.csv')
t_data = data['t'].values
xyz_data = data[['x', 'y', 'z']].values

# 设置初始状态 (取第一个数据点的位置和估计初始速度)
x0, y0, z0 = xyz_data[0]
# 通过前几个点估计初始速度
if len(t_data) > 1:
    v_x0 = (xyz_data[1][0] - xyz_data[0][0]) / (t_data[1] - t_data[0])
    v_y0 = (xyz_data[1][1] - xyz_data[0][1]) / (t_data[1] - t_data[0])
    v_z0 = (xyz_data[1][2] - xyz_data[0][2]) / (t_data[1] - t_data[0])
else:
    v_x0, v_y0, v_z0 = 1.0, 2.0, 0.0  # 默认值

print(f"数据点数量: {len(t_data)}")
print(f"初始位置: ({x0:.2f}, {y0:.2f}, {z0:.2f})")
print(f"估计初始速度: ({v_x0:.2f}, {v_y0:.2f}, {v_z0:.2f})")

# 改进的数据处理方法
# 1. 放宽RANSAC阈值，适应~5cm左右的偏差
ransac = RANSACRegressor(residual_threshold=0.065)  # 误差阈值，单位m
ransac.fit(t_data.reshape(-1, 1), xyz_data)
inliers = ransac.inlier_mask_
outliers = ~inliers

print(f"初步筛选 - 保留点: {np.sum(inliers)}, 异常点: {np.sum(outliers)}")

# 2. 对轻微偏差的点进行修正而不是直接剔除
xyz_modified = xyz_data.copy()

# 对每个维度分别进行样条平滑
for dim in range(3):  # x, y, z维度
    # 为内点创建样条
    tck = splrep(t_data[inliers], xyz_data[inliers, dim], s=0.01)
    
    # 对于可能的轻微偏差点，检查其与样条的距离
    for i in range(len(t_data)):
        if outliers[i]:
            # 计算此点到样条拟合曲线的距离
            predicted = splev(t_data[i], tck)
            diff = abs(xyz_data[i, dim] - predicted)
            
            # 如果偏差在可接受范围内(5cm)，修正数据点而不是剔除
            if diff < 0.05:  # 5cm
                xyz_modified[i, dim] = predicted * 0.7 + xyz_data[i, dim] * 0.3  # 加权平均
                outliers[i] = False  # 将此点标记为内点

# 更新内点/异常点标记
inliers = ~outliers
xyz_clean = xyz_modified[inliers]
t_clean = t_data[inliers]

print(f"优化后 - 保留数据点: {len(t_clean)}")
print(f"剔除异常值: {len(t_data) - len(t_clean)}")

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 定义动力学模型
def dynamics(t, state, C_L, C_D, theta_deg, params):
    x, y, z, v_x, v_y, v_z = state
    a, d, m, s, omega, rho, g = params['a'], params['d'], params['m'], params['s'], params['omega'], params['rho'], params['g']
    r = a / 2
    I_x = (1/12) * m * s**2
    theta = np.radians(theta_deg)
    v = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    
    # 计算升力与阻力
    F_lift = 0.5 * rho * v**2 * C_L * a * d
    F_drag_x = 0.5 * rho * v_x**2 * C_D * a * d
    F_drag_y = 0.5 * rho * v_y**2 * C_D * a * d
    F_drag_z = 0.5 * rho * v_z**2 * C_D * a * d
    
    # 陀螺进动加速度
    M_z = r * F_lift * np.sin(theta)  # 升力垂直分量引发力矩
    Omega_p = M_z / (I_x * omega)
    a_x = Omega_p * v_y
    
    # 运动方程
    dv_x = a_x - F_drag_x / m
    dv_y = (F_lift * np.cos(theta) - F_drag_y) / m
    dv_z = (F_lift * np.sin(theta) - F_drag_z) / m - g
    
    return [v_x, v_y, v_z, dv_x, dv_y, dv_z]

# 定义目标函数
def objective(params_opt, t_clean, xyz_clean, initial_state, params_fixed):
    C_L, C_D, theta_deg = params_opt
    sol = solve_ivp(
        lambda t, y: dynamics(t, y, C_L, C_D, theta_deg, params_fixed),
        [t_clean[0], t_clean[-1]], initial_state, t_eval=t_clean, rtol=1e-6
    )
    xyz_pred = sol.y[:3].T  # 提取预测轨迹
    residuals = np.linalg.norm(xyz_pred - xyz_clean, axis=1)  # 计算残差
    return np.sum(residuals**2)  # 返回残差平方和

# 初始参数与优化
params_fixed = {
    'a': 0.3, 'd': 0.05, 'm': 0.1, 's': 0.01, 
    'omega': 50, 'rho': 1.225, 'g': 9.793
}
initial_state = [x0, y0, z0, v_x0, v_y0, v_z0]  # 替换为实际初始值
initial_guess = [0.5, 0.1, 30]  # C_L, C_D, theta_deg
bounds = [(0.1, 2.0), (0.01, 0.5), (20, 45)]  # 参数范围限制

result = minimize(
    lambda p: objective(p, t_clean, xyz_clean, initial_state, params_fixed),
    initial_guess, bounds=bounds, method='L-BFGS-B'
)
C_L_opt, C_D_opt, theta_opt = result.x
print(f"Optimized: C_L={C_L_opt:.2f}, C_D={C_D_opt:.2f}, Theta={theta_opt:.2f}°")