import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from data_utils import (
    get_initial_conditions, 
    apply_kalman_filter, 
    analyze_noise_reduction, 
    read_csv,
    params_fixed
)

def velocity_derivatives(t, state, params):
    """
    定义微分方程，描述位置和速度分量的变化率。
    
    :param t: 时间
    :param state: 状态向量 [x, y, z, vx, vy, vz]
    :param params: 参数字典，包含 'D'（几何修正系数）, 'cl0', 'cl1'（升力系数）, 'cd0', 'cd1'（阻力系数）
    :return: 状态导数 [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    
    # 计算速度大小
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    if v < 1e-10:  # 避免除以零
        return [vx, vy, vz, 0, 0, 0]
    
    # 计算速度方向角 phi（与水平面的夹角）
    phi = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
    
    # 计算升力系数和阻力系数（φ的线性函数）
    C_L = params['cl0'] + params['cl1'] * phi
    C_D = params['cd0'] + params['cd1'] * phi
    
    # 计算特征面积（基于翼展和翼宽）
    A = params_fixed['a'] * params_fixed['d']
    
    # 计算空气动力学力
    F_D = -0.5 * params_fixed['rho'] * params['D'] * A * C_D * v**2  # 阻力
    F_L = 0.5 * params_fixed['rho'] * params['D'] * A * C_L * v**2   # 升力
    
    # 分解空气动力到各个方向
    # 阻力沿速度方向相反
    F_Dx = F_D * vx / v
    F_Dy = F_D * vy / v
    F_Dz = F_D * vz / v
    
    # 升力垂直于速度方向，在竖直平面内
    if v * np.cos(phi) < 1e-10:  # 接近垂直运动时
        F_Lx = 0
        F_Ly = 0
        F_Lz = F_L if vz >= 0 else -F_L
    else:
        # 计算升力在 x-y 平面的投影
        F_Lxy = F_L * np.cos(phi)
        F_Lz = -F_L * np.sin(phi) * np.sign(vz)
        # 将 x-y 平面的力分解到 x、y 方向
        F_Lx = -F_Lxy * vy / (v * np.cos(phi))
        F_Ly = F_Lxy * vx / (v * np.cos(phi))
    
    # 计算加速度（牛顿第二定律）
    ax = (F_Dx + F_Lx) / params_fixed['m']
    ay = (F_Dy + F_Ly) / params_fixed['m']
    az = (F_Dz + F_Lz) / params_fixed['m'] - params_fixed['g']
    
    return [vx, vy, vz, ax, ay, az]

def solve_differential_equation(initial_conditions, params, t_span, t_eval):
    """
    根据初始条件和参数求解微分方程。
    
    :param initial_conditions: 初始条件 [x0, y0, z0, vx0, vy0, vz0]
    :param params: 参数字典，包含：
                  - D: 几何修正系数
                  - cl0, cl1: 升力系数的常数项和一次项系数
                  - cd0, cd1: 阻力系数的常数项和一次项系数
    :param t_span: 时间范围 (t0, tf)
    :param t_eval: 时间点数组
    :return: 求解结果，包含时间和状态
    """
    solution = solve_ivp(
        velocity_derivatives,
        t_span,
        initial_conditions,
        args=(params,),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,        # 相对误差容限
        atol=1e-8,        # 绝对误差容限
        max_step=0.1,     # 最大步长
        first_step=0.01   # 初始步长
    )
    
    if not solution.success:
        print(f"警告：微分方程求解失败 - {solution.message}")
    
    return solution

def optimize_parameters(filtered_data, initial_conditions, t_span, t_eval):
    """
    使用最小二乘法优化微分方程中的参数。
    
    :param filtered_data: 滤波后的数据 [t,x,y,z,vx,vy,vz]
    :param initial_conditions: 初始条件 [x0, y0, z0, vx0, vy0, vz0]
    :param t_span: 时间范围
    :param t_eval: 时间点数组
    :return: 优化后的参数
    """
    def objective_function(params_array):
        # 将参数数组转换为字典
        params = {
            'D': params_array[0],     # 几何修正系数
            'cl0': params_array[1],   # 升力系数常数项
            'cl1': params_array[2],   # 升力系数一次项
            'cd0': params_array[3],   # 阻力系数常数项
            'cd1': params_array[4]    # 阻力系数一次项
        }
        
        try:
            # 求解微分方程
            solution = solve_differential_equation(initial_conditions, params, t_span, t_eval)
            if not solution.success:
                return 1e10
            
            # 提取模拟结果
            simulated = solution.y.T
            
            # 计算位置误差（使用相对误差以平衡不同尺度）
            pos_scale = np.max(np.abs(filtered_data[:, 1:4]))
            position_error = np.mean(np.sum((simulated[:, :3] - filtered_data[:, 1:4])**2, axis=1)) / (pos_scale**2)
            
            # 计算速度误差
            vel_scale = np.max(np.abs(filtered_data[:, 4:7]))
            velocity_error = np.mean(np.sum((simulated[:, 3:] - filtered_data[:, 4:7])**2, axis=1)) / (vel_scale**2)
            
            # 总误差（位置误差权重更大）
            total_error = position_error + 0.1 * velocity_error
            
            # 添加物理约束的惩罚项
            penalty = 0
            if params['D'] <= 0:  # D必须为正
                penalty += 1e6
            if abs(params['cl1']) > 2 * abs(params['cl0']):  # 限制升力系数的变化范围
                penalty += 1e6
            if abs(params['cd1']) > 2 * abs(params['cd0']):  # 限制阻力系数的变化范围
                penalty += 1e6
            if params['cd0'] < 0:  # 阻力系数常数项必须为正
                penalty += 1e6
                
            return total_error + penalty
            
        except Exception as e:
            print(f"优化过程出错: {e}")
            return 1e10  # 出错时返回大误差

    # 初始猜测参数 [D, cl0, cl1, cd0, cd1]
    initial_guess = [1.0, 0.3, 0.1, 0.1, 0.05]
    
    # 设置参数边界
    bounds = [
        (0.1, 10.0),    # D: 几何修正系数
        (-1.0, 1.0),    # cl0: 升力系数常数项
        (-0.5, 0.5),    # cl1: 升力系数一次项
        (0.01, 1.0),    # cd0: 阻力系数常数项
        (-0.5, 0.5)     # cd1: 阻力系数一次项
    ]
    
    # 使用 L-BFGS-B 方法进行有界优化
    result = minimize(
        objective_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        print(f"优化警告: {result.message}")
    
    optimized_params = {
        'D': result.x[0],
        'cl0': result.x[1],
        'cl1': result.x[2],
        'cd0': result.x[3],
        'cd1': result.x[4]
    }
    
    return optimized_params

if __name__ == "__main__":
    try:
        print("开始处理回力标轨迹数据...")
        
        # 读取原始数据
        print("\n1. 读取数据...")
        raw_data = read_csv('ps.csv')
        print(f"   读取了 {len(raw_data)} 个数据点")
        
        # 获取初始条件和完整数据（包含速度）
        print("\n2. 计算初始条件和速度...")
        initial_conditions, full_data = get_initial_conditions(raw_data)
        print(f"   初始位置: ({initial_conditions[0]:.2f}, {initial_conditions[1]:.2f}, {initial_conditions[2]:.2f})")
        print(f"   初始速度: ({initial_conditions[3]:.2f}, {initial_conditions[4]:.2f}, {initial_conditions[5]:.2f})")
        
        # 应用卡尔曼滤波
        print("\n3. 应用卡尔曼滤波...")
        filtered_data = apply_kalman_filter(full_data)
        print("   完成卡尔曼滤波")
        
        # 设置时间范围和评估点
        t_span = (filtered_data[0, 0], filtered_data[-1, 0])
        t_eval = filtered_data[:, 0]
        print(f"   时间范围: {t_span[0]:.2f}s 到 {t_span[1]:.2f}s")
        print(f"   采样点数: {len(t_eval)}")
        
        # 优化参数
        print("\n4. 开始参数优化...")
        print("   初始化优化器...")
        optimized_params = optimize_parameters(filtered_data, initial_conditions, t_span, t_eval)
        print("\n优化完成！参数结果:")
        print("   几何修正系数:")
        print(f"      D = {optimized_params['D']:.6f}")
        print("   升力系数:")
        print(f"      CL(φ) = {optimized_params['cl0']:.6f} + {optimized_params['cl1']:.6f}φ")
        print("   阻力系数:")
        print(f"      CD(φ) = {optimized_params['cd0']:.6f} + {optimized_params['cd1']:.6f}φ")
        
        # 使用优化后的参数求解微分方程
        print("\n5. 求解完整轨迹...")
        solution = solve_differential_equation(initial_conditions, optimized_params, t_span, t_eval)
        
        if not solution.success:
            print(f"\n警告：求解器报告问题 - {solution.message}")
        else:
            print("   轨迹求解成功")
        
        # 计算误差统计
        print("\n6. 计算性能指标...")
        simulated = solution.y.T
        actual = filtered_data[:, 1:]  # 去除时间列
        
        # 计算归一化误差
        pos_scale = np.max(np.abs(actual[:, :3]))
        vel_scale = np.max(np.abs(actual[:, 3:]))
        
        # 位置误差分析
        position_errors = np.linalg.norm(simulated[:, :3] - actual[:, :3], axis=1)
        mean_pos_error = np.mean(position_errors)
        max_pos_error = np.max(position_errors)
        std_pos_error = np.std(position_errors)
        rel_pos_error = mean_pos_error / pos_scale * 100  # 相对误差百分比
        
        # 速度误差分析
        velocity_errors = np.linalg.norm(simulated[:, 3:] - actual[:, 3:], axis=1)
        mean_vel_error = np.mean(velocity_errors)
        max_vel_error = np.max(velocity_errors)
        std_vel_error = np.std(velocity_errors)
        rel_vel_error = mean_vel_error / vel_scale * 100  # 相对误差百分比
        
        print("\n性能指标:")
        print("位置拟合:")
        print(f"   平均误差: {mean_pos_error:.3f}m ({rel_pos_error:.1f}%)")
        print(f"   最大误差: {max_pos_error:.3f}m")
        print(f"   标准差: {std_pos_error:.3f}m")
        print("速度拟合:")
        print(f"   平均误差: {mean_vel_error:.3f}m/s ({rel_vel_error:.1f}%)")
        print(f"   最大误差: {max_vel_error:.3f}m/s")
        print(f"   标准差: {std_vel_error:.3f}m/s")
        
        # 分析降噪效果和模型精度
        print("\n7. 评估降噪效果和模型精度...")
        noise_stats = analyze_noise_reduction(
            raw_data[:, 1:4],          # 原始位置数据
            filtered_data[:, 1:4],      # 滤波后的位置数据
            simulated[:, :3]            # 模拟的位置数据
        )
        
        print("\n数据分析结果:")
        print("原始数据与滤波数据比较:")
        print(f"   平均差异: {noise_stats['原始-滤波平均差异']:.3f}m")
        print(f"   最大差异: {noise_stats['原始-滤波最大差异']:.3f}m")
        print(f"   标准差: {noise_stats['原始-滤波标准差']:.3f}m")
        
        print("\n原始数据与模拟数据比较:")
        print(f"   平均差异: {noise_stats['原始-模拟平均差异']:.3f}m")
        print(f"   最大差异: {noise_stats['原始-模拟最大差异']:.3f}m")
        print(f"   标准差: {noise_stats['原始-模拟标准差']:.3f}m")
        
        print("\n滤波数据与模拟数据比较:")
        print(f"   平均差异: {noise_stats['滤波-模拟平均差异']:.3f}m")
        print(f"   最大差异: {noise_stats['滤波-模拟最大差异']:.3f}m")
        print(f"   标准差: {noise_stats['滤波-模拟标准差']:.3f}m")
        
        print(f"\n模型整体改进: {noise_stats['模型改进百分比']:.1f}%")
        
        # 添加可视化展示
        print("\n8. 生成可视化结果...")
        from plot_utils import plot_data_points, plot_comparison
        
        # 显示原始数据点
        print("   绘制原始数据点...")
        plot_data_points(raw_data, title="原始轨迹数据点")
        
        # 显示滤波后的数据点
        print("   绘制滤波后数据点...")
        plot_data_points(filtered_data, title="滤波后的轨迹数据点")
        
        # 创建模拟数据数组（包含时间列）
        simulated_data = np.column_stack((t_eval, simulated[:, :3]))
        print("   绘制模拟数据点...")
        plot_data_points(simulated_data, title="模拟轨迹数据点")
        
        # 创建综合对比图的数据
        print("   生成轨迹对比图...")
        comparison_data = np.column_stack((
            t_eval,                    # 时间
            simulated[:, :3],          # 模拟位置
            filtered_data[:, 1:4],     # 滤波后位置
            raw_data[:, 1:4]           # 原始位置
        ))
        plot_comparison(comparison_data, title="回力标轨迹分析（原始/滤波/模拟）")
        
        print("\n模拟完成！")
        print("建议：检查相对误差是否在可接受范围内（通常<10%）")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())