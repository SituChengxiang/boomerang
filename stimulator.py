import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from data_utils import read_csv, get_initial_conditions
from plot_utils import plot_comparison

# 从mathematicalMethod中导入动力学方程和初始条件
from mathematicalMethod import dynamics, params_fixed, initial_state, t_data, xyz_data

# 读取数据并获取初始条件
data = read_csv('ps.csv')
t_data, xyz_data, initial_state = get_initial_conditions(data)

# 设置积分时间范围（1秒内）
t_start = t_data[0]  # 起始时间
t_end = t_start + 1.0  # 结束时间
t_eval = np.linspace(t_start, t_end, 100)  # 在1秒内生成100个时间点

# 使用优化后的参数（假设已从mathematicalMethod中获得）
C_L_opt = 0.2  # 示例值
C_D_opt = 0.4  # 示例值
theta_opt = 25  # 示例值

# 数值积分求解
sol = solve_ivp(
    lambda t, y: dynamics(t, y, C_L_opt, C_D_opt, theta_opt, params_fixed),
    [t_start, t_end],
    initial_state,
    t_eval=t_eval,
    method='LSODA',  # 使用更适合刚性问题的积分器
    rtol=1e-3,       # 放宽误差容限
    atol=1e-6
)

# 检查积分是否成功
if not sol.success:
    print("数值积分失败:", sol.message)
    exit()

# 提取预测轨迹
xyz_pred = sol.y[:3].T  # 提取位置 (x, y, z)

# 绘制对比图
data_3d = xyz_data
data_xt = np.column_stack((t_data, xyz_data[:, 0]))
data_yt = np.column_stack((t_data, xyz_data[:, 1]))
data_zt = np.column_stack((t_data, xyz_data[:, 2]))
plot_comparison(data_3d, data_xt, data_yt, data_zt, title="回力镖飞行轨迹对比")