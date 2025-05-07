import numpy as np
import scipy.optimize as opt
from scipy import stats

# 导入数据处理工具和绘图工具
from data_utils import read_csv, params_fixed, get_initial_conditions
from plot_utils import plot_dynamic_parameters

# 使用已定义的参数
a = params_fixed['a']        # 翼展 (m)
d = params_fixed['d']        # 翼宽 (m)
m = params_fixed['m']        # 质量 (kg)
I = params_fixed['I']        # 转动惯量
omega = params_fixed['omega']  # 固定角速度 (rad/s)
rho = params_fixed['rho']    # 空气密度 (kg/m^3)
g = params_fixed['g']        # 重力加速度 (m/s²)

# 固定参数组合
prm = rho * a * d / (2 * m)

def formula(ax, ay, az, v, vx, vy, vz):
    """定义回旋镖动力学方程组"""
    def equation(var):
        [D, CL, CD] = var
        # f1, f2, f3 = 0 即得原公式
        f1 = ax - D * rho * CL * omega * v * (a**3) * d * vy / (2*I) + prm * CD * v * vx
        f2 = ay + prm * v * vy * (CL - CD)
        f3 = az - prm * v * vz * (CL - CD*0.1 ) + g
        return [f1, f2, f3]
    return equation    

def main():
    """主函数：计算回旋镖动力学参数"""
    # 读取数据
    print("读取原始数据...")
    raw_data = read_csv('ps.csv')
    
    # 应用卡尔曼滤波进行降噪
    print("应用卡尔曼滤波进行降噪...")
    from data_utils import apply_kalman_filter
    filtered_data = apply_kalman_filter(raw_data)

    # 获取轨迹数据并包含速度
    print("计算速度和加速度...")
    _, full_data = get_initial_conditions(filtered_data)
    
    # 提取有用数据
    t = full_data[:, 0]
    x = full_data[:, 1]
    y = full_data[:, 2]
    z = full_data[:, 3]
    v_x = full_data[:, 4]
    v_y = full_data[:, 5]
    v_z = full_data[:, 6]
    
    # 计算速度大小
    v = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    
    # 计算加速度
    print("计算加速度分量...")
    a_x = np.gradient(v_x, t)
    a_y = np.gradient(v_y, t)
    a_z = np.gradient(v_z, t)
    
    # 求解动力学参数
    print("求解动力学参数...")
    solves = []
    for i in range(len(a_x)):
        equation = formula(a_x[i], a_y[i], a_z[i], v[i], v_x[i], v_y[i], v_z[i])
        solves.append(opt.least_squares(equation, [1, 1, 1]))
    
    # 从求解结果中提取参数值
    solves = np.array([solve.x for solve in solves])
    
    # 忽略前两个点
    skip_start = 2
    t_valid = t[skip_start:]
    solves_valid = solves[skip_start:]
    
    # 显示原始数据的统计结果（忽略前两个点）
    print(f"\n\t\tD\t\tCL\t\tCD\t (忽略前{skip_start}个点)")
    print("原始数据平均", np.average(solves_valid, axis=0))
    print("原始数据方差", np.var(solves_valid, axis=0))
    
    # 移除异常值并统计
    print("\n移除异常值后的统计:")
    [D, CL, CD] = np.hsplit(solves_valid, 3)  # 使用去除前两点的数据
    D_z = np.abs(stats.zscore(D))
    CL_z = np.abs(stats.zscore(CL))
    CD_z = np.abs(stats.zscore(CD))
    
    # 只保留 |Z-score| < 3 的数据点
    data = [D[(D_z < 3)], CL[(CL_z < 3)], CD[(CD_z < 3)]]  # 将D的阈值从10改为3，使三个参数筛选标准一致
    print("净化数据平均值:", np.average(data[0]), np.average(data[1]), np.average(data[2]))
    print("净化数据方差:", np.var(data[0]), np.var(data[1]), np.var(data[2]))
    
    # 绘制结果，直接使用有效数据点，不再标记被忽略的点
    plot_dynamic_parameters(t_valid, solves_valid, save_path='boomerang_params.png')
    print("参数变化图已保存到 'boomerang_params.png'")

if __name__ == "__main__":
    main()