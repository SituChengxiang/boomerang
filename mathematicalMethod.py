import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from data_utils import read_csv, get_initial_conditions, params_fixed
from plot_utils import plot_data_points, plot_comparison

# 读取数据并获取初始条件
data = read_csv('ps.csv')
t_data, xyz_data, initial_state = get_initial_conditions(data)

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

# 使用基于局部平滑的阈值筛选算法优化数据处理

# 对每个维度分别进行平滑滤波
xyz_smooth = np.zeros_like(xyz_data)
for dim in range(3):
    # 使用Savitzky-Golay滤波器进行平滑，调整窗口大小和多项式阶数以适应更多数据
    xyz_smooth[:, dim] = savgol_filter(xyz_data[:, dim], window_length=11, polyorder=3)

# 计算原始数据与平滑数据之间的残差
residuals = np.linalg.norm(xyz_data - xyz_smooth, axis=1)

# 先获取基准残差值作为参考
median_residual = np.median(residuals)
std_residual = np.std(residuals)
print(f"残差中位数: {median_residual:.4f}m, 标准差: {std_residual:.4f}m")

# 1. 使用自适应阈值，考虑数据的整体分布特性
adaptive_threshold = median_residual + 1.5 * std_residual
print(f"自适应阈值: {adaptive_threshold:.4f}m")

# 2. 创建修正后的数据，而不是简单地剔除
xyz_modified = xyz_data.copy()
t_modified = t_data.copy()

# 3. 根据残差大小分级处理
# - 小残差：完全保留
# - 中等残差：进行加权平均修正
# - 大残差：标记为异常值
small_deviation = residuals <= median_residual + 0.5 * std_residual  # 小偏差
medium_deviation = (residuals > median_residual + 0.5 * std_residual) & (residuals <= adaptive_threshold)  # 中等偏差
large_deviation = residuals > adaptive_threshold  # 大偏差

print(f"小偏差点: {np.sum(small_deviation)}, 中等偏差点: {np.sum(medium_deviation)}, 大偏差点: {np.sum(large_deviation)}")

# 4. 对中等偏差点进行加权修正，越接近阈值的点原始数据权重越低
for i in range(len(t_data)):
    if medium_deviation[i]:
        # 计算权重因子 (0.8~0.3)，残差越大，原始数据权重越低
        weight_factor = 0.8 - 0.5 * (residuals[i] - (median_residual + 0.5 * std_residual)) / (adaptive_threshold - (median_residual + 0.5 * std_residual))
        # 对三个坐标分量分别加权平均
        for dim in range(3):
            xyz_modified[i, dim] = weight_factor * xyz_data[i, dim] + (1 - weight_factor) * xyz_smooth[i, dim]

# 5. 只保留小偏差和修正后的中等偏差点
inliers = ~large_deviation
xyz_clean = xyz_modified[inliers]
t_clean = t_data[inliers]

print(f"优化后 - 保留数据点: {len(t_clean)} / {len(t_data)} ({len(t_clean)/len(t_data)*100:.1f}%)")
print(f"剔除异常值: {len(t_data) - len(t_clean)}")

# 6. 可视化数据处理结果（可选）
if False:  # 设置为True以启用可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始数据点
    ax.scatter(xyz_data[small_deviation, 0], xyz_data[small_deviation, 1], xyz_data[small_deviation, 2], 
               c='green', marker='o', label='small error', alpha=0.6)
    ax.scatter(xyz_data[medium_deviation, 0], xyz_data[medium_deviation, 1], xyz_data[medium_deviation, 2], 
               c='blue', marker='o', label='corrected middle', alpha=0.6)
    ax.scatter(xyz_data[large_deviation, 0], xyz_data[large_deviation, 1], xyz_data[large_deviation, 2], 
               c='red', marker='x', label='deleted large error', alpha=0.6)
    
    # 绘制平滑轨迹线
    ax.plot(xyz_smooth[:, 0], xyz_smooth[:, 1], xyz_smooth[:, 2], 'k-', linewidth=2, label='curve')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Data Points and Line')
    ax.legend()
    plt.show()


# dynamics函数定义动力学因素
def dynamics(t, state, C_L, C_D, theta_deg, params, t0=0):
    x, y, z, v_x, v_y, v_z = state
    a, d, m, I, omega, rho, g = params.values()
    theta = np.radians(theta_deg)
    v = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    # 升力与阻力
    F_lift = 0.5 * rho * v**2 * C_L * a * d
    F_drag = 0.5 * rho * v**2 * C_D * a * d

    # 陀螺力矩（绕Y轴）—— 更物理准确的方式
    r = a / 2  # 力臂
    M_gyro_y = r * F_lift * np.sin(theta)  # 力矩 = 力 × 力臂
    alpha = M_gyro_y / I  # 角加速度
    angular_velocity = omega + alpha * (t - t0)  # 使用传入的t0而不是全局变量

    # 线性加速度更新（考虑陀螺进动）
    dv_x = -F_drag / m * (v_x / v)
    dv_y = -F_drag / m * (v_y / v) + angular_velocity * v_z
    dv_z = -F_drag / m * (v_z / v) - g + angular_velocity * v_x

    return [v_x, v_y, v_z, dv_x, dv_y, dv_z]

# 初始参数
# 计算初始状态（完整的6维状态向量）
initial_state = [x0, y0, z0, v_x0, v_y0, v_z0]

# =============================
# 5. 目标函数（误差 + 正则化项）
# =============================
def objective(params_opt, t_clean, xyz_clean, initial_state, params_fixed,
              lambda_CL=0.01, lambda_CD=0.01, lambda_theta=0.01):
    C_L, C_D, theta_deg = params_opt
    t0 = t_clean[0]  # 获取初始时间

    try:
        sol = solve_ivp(
            lambda t, y: dynamics(t, y, C_L, C_D, theta_deg, params_fixed, t0),
            [t_clean[0], t_clean[-1]], initial_state, t_eval=t_clean, rtol=1
        )
        
        # 检查是否成功求解并包含所有时间点
        if not sol.success or sol.y.shape[1] != len(t_clean):
            # 如果求解不成功或时间点数量不匹配，返回一个很大的值
            return 1e10
            
        xyz_pred = sol.y[:3].T  # 获取位置预测
        residuals = np.linalg.norm(xyz_pred - xyz_clean, axis=1)  # 计算残差
        
        # 正则化项（L2）
        reg_term = lambda_CL * C_L**2 + lambda_CD * C_D**2 + lambda_theta * theta_deg**2
        
        return np.sum(residuals**2) + reg_term
        
    except Exception as e:
        print(f"解算错误: {e}")
        return 1e10  # 求解出错时返回很大的值

# =============================
# 6. 参数优化
# =============================
initial_guess = [0.2, 0.4, 25]  # C_L, C_D, theta_deg
bounds = [(0.1, 2.0), (0.01, 0.5), (20, 45)]  # 参数范围限制

result = minimize(
    lambda p: objective(p, t_clean, xyz_clean, initial_state, params_fixed),
    initial_guess, bounds=bounds, method='L-BFGS-B'
)

C_L_opt, C_D_opt, theta_opt = result.x
print(f"\n优化结果:")
print(f"C_L = {C_L_opt:.3f}")
print(f"C_D = {C_D_opt:.3f}")
print(f"Theta = {theta_opt:.3f}°")
print("是否成功:", result.success)

# =============================
# 7. 数值积分求解（用优化后的参数）
# =============================
sol = solve_ivp(
    lambda t, y: dynamics(t, y, C_L_opt, C_D_opt, theta_opt, params_fixed),
    [t_clean[0], t_clean[-1]], initial_state, t_eval=t_clean, rtol=1e-6
)

# 可视化数据处理结果
plot_data_points(xyz_data, title="原始数据点")

# 绘制对比图
data_3d = xyz_data
data_xt = np.column_stack((t_data, xyz_data[:, 0]))
data_yt = np.column_stack((t_data, xyz_data[:, 1]))
data_zt = np.column_stack((t_data, xyz_data[:, 2]))
plot_comparison(data_3d, data_xt, data_yt, data_zt, title="原始数据点")