import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from data_utils import read_csv, get_initial_conditions, params_fixed
from plot_utils import plot_comparison, plot_data_points

def dynamics(t, state, C_L0, C_D0, theta, params):
    """
    回力镖的动力学方程
    
    :param t: 当前时间
    :param state: 状态向量 [x, y, z, vx, vy, vz]
    :param C_L0: 升力系数基值
    :param C_D0: 阻力系数基值
    :param theta: 回力镖角度(度)
    :param params: 固定参数字典
    :return: 状态导数 [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    # 解包状态向量
    x, y, z, vx, vy, vz = state
    
    # 解包参数
    a = params['a']        # 翼展
    m = params['m']        # 质量
    I = params['I']        # 转动惯量
    omega = params['omega'] # 角速度
    rho = params['rho']    # 空气密度
    g = params['g']        # 重力加速度
    d = params['d']        # 翼宽
    A = a * d              # 面积估计
    
    # 计算速度大小
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # 如果速度为零，避免除以零错误
    if v < 1e-10:
        return np.array([vx, vy, vz, 0, 0, -g])
    
    # 角度转换为弧度
    theta_rad = np.radians(theta)
    
    # 计算攻角
    alpha = theta_rad - np.arctan2(vz, vy) if abs(vy) > 1e-10 else theta_rad - np.sign(vz) * np.pi/2
    
    # 升力和阻力系数模型
    C_L_alpha = 2*np.pi  # 升力系数斜率
    k = 0.1              # 阻力系数因子
    
    C_L = C_L0 + C_L_alpha * alpha
    C_D = C_D0 + k * C_L**2
    
    # 计算x方向的加速度（简化模型）
    # 我们使用积分的近似值来代替原方程中的积分计算
    integral_approx = a**3 * omega**2 / 3  # 积分近似值
    dvx_dt = (rho * a * omega * integral_approx / I) * (C_L * np.cos(alpha) - C_D * np.sin(alpha))
    
    # 计算y和z方向的加速度
    force_coef = rho * A * (C_L - C_D) / (2 * m) * v
    dvy_dt = -force_coef * vy
    dvz_dt = force_coef * vz - g
    
    return np.array([vx, vy, vz, dvx_dt, dvy_dt, dvz_dt])

# 读取数据并获取初始条件
data = read_csv('ps1.csv')
initial_state = get_initial_conditions(data)  # 只返回一个数组

# 从data中提取时间和坐标数据
t_data = data[:, 0]  # 提取时间列
xyz_data = data[:, 1:4]  # 提取xyz坐标列

# 设置积分时间范围（1.5秒内）
t_start = t_data[0]  # 起始时间
t_end = t_start + 1.5  # 结束时间
dt = 0.001  # 步长为1毫秒
t_eval = np.arange(t_start, t_end + dt, dt)  # 生成时间点

# 使用估计的参数
C_L0 = 0.5
C_D0 = 0.1
theta = 20  # 角度 (度)

# 数值积分求解
sol = solve_ivp(
    lambda t, y: dynamics(t, y, C_L0, C_D0, theta, params_fixed),
    [t_start, t_end],
    initial_state[1:],  # 只使用[x, y, z, vx, vy, vz]作为初始状态
    t_eval=t_eval,
    method='RK45',  # 使用Runge-Kutta方法
    rtol=1e-6,
    atol=1e-9
)

# 检查积分是否成功
if not sol.success:
    print("数值积分失败:", sol.message)
    exit()

# 提取预测轨迹
t_pred = sol.t
xyz_pred = sol.y[:3].T  # 提取位置 (x, y, z)

# 构建轨迹数据数组
trajectory_data = np.column_stack((t_pred, xyz_pred))

# 打印部分结果
print(f"总模拟时间点: {len(trajectory_data)}")
print(f"前5个时间点的轨迹:")
for i in range(min(5, len(trajectory_data))):
    print(f"t={trajectory_data[i, 0]:.3f}, x={trajectory_data[i, 1]:.3f}, y={trajectory_data[i, 2]:.3f}, z={trajectory_data[i, 3]:.3f}")

# 绘制三维散点图
plot_data_points(trajectory_data, title="回力镖飞行轨迹预测", save_path="trajectory_3d.png")

# 提取实际轨迹数据
real_data = np.column_stack((t_data[:len(xyz_data)], xyz_data))

# 绘制对比图
plot_comparison(real_data, title="回力镖飞行轨迹对比")