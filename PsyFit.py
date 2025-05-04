import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 已知参数（示例值，需替换为实际数据）
a = 0.15      # 翼展 (m)
d = 0.028     # 翼宽 (m)
m = 0.002183      # 质量 (kg)
g = 9.793    # 杭州重力加速度 (m/s²)
rho = 1.225  # 空气密度 (kg/m³)
A = a * d    # 翼面积
I_x = 0.01   # 绕x轴转动惯量 (kg·m²)
omega = 10   # 初始滚转角速度 (rad/s)
r = a / 2    # 翼片到质心距离 (m)

# 动力学方程（只保留一个定义，包含theta_0_deg参数）
def dynamics(t, state, C_L, C_D, theta_0_deg):
    theta_0 = np.radians(theta_0_deg)
    x, y, z, v_x, v_y, v_z = state
    v = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    
    # 计算升力与阻力
    F_lift = 0.5 * rho * v**2 * C_L * A
    F_drag_x = 0.5 * rho * v_x**2 * C_D * A
    F_drag_y = 0.5 * rho * v_y**2 * C_D * A
    F_drag_z = 0.5 * rho * v_z**2 * C_D * A
    
    # 陀螺进动导致的横向加速度
    M_z = r * F_lift * np.sin(theta_0)  # 升力垂直分量引发力矩
    Omega_p = M_z / (I_x * omega)
    a_x = Omega_p * v_y
    
    # 运动方程
    dv_x = a_x - F_drag_x / m
    dv_y = (F_lift * np.cos(theta_0) - F_drag_y) / m
    dv_z = (F_lift * np.sin(theta_0) - F_drag_z) / m - g
    
    return [v_x, v_y, v_z, dv_x, dv_y, dv_z]

# 残差函数
def residuals(params, data, times):
    C_L, C_D, theta_0_deg = params
    # 初始状态 [x, y, z, v_x, v_y, v_z]
    initial_state = [0.0, 0.0, 1.0, 0.0, 10.0, 0.0]
    
    try:
        sol = solve_ivp(lambda t, y: dynamics(t, y, C_L, C_D, theta_0_deg), 
                        [0, times[-1]], initial_state, t_eval=times)
        
        x_pred = sol.y[0]
        y_pred = sol.y[1]
        z_pred = sol.y[2]
        
        # 确保数据维度匹配
        if len(x_pred) != len(data['x']):
            print(f"警告：预测点数 ({len(x_pred)}) 与数据点数 ({len(data['x'])}) 不匹配")
            # 使用最小长度
            min_len = min(len(x_pred), len(data['x']))
            error_x = np.sum((x_pred[:min_len] - data['x'][:min_len])**2)
            error_y = np.sum((y_pred[:min_len] - data['y'][:min_len])**2)
            error_z = np.sum((z_pred[:min_len] - data['z'][:min_len])**2)
        else:
            error_x = np.sum((x_pred - data['x'])**2)
            error_y = np.sum((y_pred - data['y'])**2)
            error_z = np.sum((z_pred - data['z'])**2)
        
        return error_x + error_y + error_z
    except Exception as e:
        print(f"解算出现错误: {e}")
        return 1e6  # 返回一个大数作为惩罚项

# 绘制结果图表
def plot_results(data, times, params):
    C_L, C_D, theta_0_deg = params
    initial_state = [0.0, 0.0, 1.0, 0.0, 10.0, 0.0]  # [x, y, z, v_x, v_y, v_z]
    
    # 使用更密的时间点生成平滑曲线
    t_fit = np.linspace(0, times[-1], 1000)
    
    # 使用优化后的参数求解运动方程
    sol = solve_ivp(lambda t, y: dynamics(t, y, C_L, C_D, theta_0_deg), 
                    [0, times[-1]], initial_state, t_eval=t_fit)
    
    # 提取预测轨迹
    x_fit = sol.y[0]
    y_fit = sol.y[1]
    z_fit = sol.y[2]
    
    # 创建图表
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(data['x'], data['y'], data['z'], c='r', marker='o', label='origin_data')
    ax1.plot(x_fit, y_fit, z_fit, 'b-', label='fit-curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('回旋镖3D轨迹')
    ax1.legend()
    
    # X-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(times, data['x'], 'ro', markersize=3, label='origin_data')
    ax2.plot(sol.t, sol.y[0], 'b-', label='fit-curve')
    ax2.set_xlabel('时间 t (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t) 结果')
    ax2.legend()
    
    # Y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(times, data['y'], 'ro', markersize=3, label='origin_data')
    ax3.plot(sol.t, sol.y[1], 'b-', label='fit-curve')
    ax3.set_xlabel('时间 t (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t) 结果')
    ax3.legend()
    
    # Z-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(times, data['z'], 'ro', markersize=3, label='origin_data')
    ax4.plot(sol.t, sol.y[2], 'b-', label='fit-curve')
    ax4.set_xlabel('时间 t (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z(t) 结果')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('psyfit_results.png', dpi=300)
    plt.show()
    
    # 计算拟合误差
    # 在origin_data点上求解
    sol_orig = solve_ivp(lambda t, y: dynamics(t, y, C_L, C_D, theta_0_deg), 
                         [0, times[-1]], initial_state, t_eval=times)
    
    x_pred = sol_orig.y[0]
    y_pred = sol_orig.y[1]
    z_pred = sol_orig.y[2]
    
    # 确保数据维度匹配
    if len(x_pred) == len(data['x']):
        x_error = np.mean((x_pred - data['x'])**2)
        y_error = np.mean((y_pred - data['y'])**2)
        z_error = np.mean((z_pred - data['z'])**2)
        
        print(f"X均方误差: {x_error:.6f}")
        print(f"Y均方误差: {y_error:.6f}")
        print(f"Z均方误差: {z_error:.6f}")
        print(f"总均方误差: {(x_error + y_error + z_error)/3:.6f}")

def main():
    # 读取轨迹数据
    try:
        if not os.path.exists('ps.csv'):
            print("错误：未找到ps.csv文件")
            return
        
        data = pd.read_csv('ps.csv')
        print(f"成功读取数据，共{len(data)}个数据点")
        
        # 检查数据是否包含 't' 列，如果有则使用，否则创建均匀时间点
        if 't' in data.columns:
            times = data['t'].values
            print("使用数据文件中的时间列")
        else:
            times = np.linspace(0, 2, len(data))  # 假设总飞行时间2秒
            print("数据文件中无时间列，创建等间隔时间点")
        
        # 优化求解
        initial_guess = [0.5, 0.1, 30.0]  # C_L, C_D, theta_0_deg
        bounds = [(0.1, 2.0), (0.01, 0.5), (20.0, 45.0)]  # 参数范围
        
        print("开始优化计算...")
        result = minimize(lambda params: residuals(params, data, times), 
                          initial_guess, bounds=bounds, method='L-BFGS-B')
        
        C_L_opt, C_D_opt, theta_opt = result.x
        print(f"优化结果: C_L={C_L_opt:.4f}, C_D={C_D_opt:.4f}, Theta0={theta_opt:.2f}°")
        print(f"优化状态: {result.success}, 迭代次数: {result.nfev}")
        
        # 绘制结果
        print("正在绘制结果...")
        plot_results(data, times, result.x)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()