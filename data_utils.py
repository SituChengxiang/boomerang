import numpy as np
import csv

params_fixed = {
    'a': 0.15,        # 翼展 (m)
    'd': 0.028,       # 翼宽 (m)
    'm': 0.002183,    # 质量 (kg)
    'I': (5/24) * 0.002183 * (0.15)**2,  # 转动惯量
    'omega': 10.0,    # 固定角速度 (rad/s)
    'rho': 1.225,     # 空气密度 (kg/m^3)
    'g': 9.793        # 重力加速度 (m/s²)
}

def read_csv(file_path='ps.csv'):
    """
    读取CSV文件中的回旋镖轨迹数据。
    
    :param file_path: CSV文件路径，默认为'ps1.csv'
    :return: 数组，结构为[[t,x,y,z],...]
    """
    try:
        # 使用numpy读取CSV文件，跳过header行
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        
        # 检查数据维度
        if data.shape[1] != 4:
            raise ValueError("CSV文件必须包含四列：t,x,y,z")
        
        return data
        
    except Exception as e:
        raise Exception(f"读取CSV文件时出错：{str(e)}")

def get_initial_conditions(data):
    """
    从轨迹数据中提取初始条件，包括初始位置和速度。
    同时计算整个轨迹的速度数据。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :return: tuple (初始条件数组, 带速度的完整数据数组)
    """
    if len(data) < 3:
        raise ValueError("需要至少三个数据点来计算平滑的速度")
    
    # 提取时间和位置数据
    times = data[:, 0]
    positions = data[:, 1:4]
    
    # 使用中心差分计算速度（对内部点）
    velocities = np.zeros_like(positions)
    for i in range(1, len(data)-1):
        dt_forward = times[i+1] - times[i]
        dt_backward = times[i] - times[i-1]
        dt_center = dt_forward + dt_backward
        
        if dt_center <= 0:
            raise ValueError(f"时间差异无效：位置 {i}")
        
        # 使用中心差分计算速度
        velocities[i] = (positions[i+1] - positions[i-1]) / dt_center
    
    # 对边界点使用前向/后向差分
    dt_first = times[1] - times[0]
    dt_last = times[-1] - times[-2]
    
    if dt_first <= 0 or dt_last <= 0:
        raise ValueError("边界时间差异无效")
    
    velocities[0] = (positions[1] - positions[0]) / dt_first
    velocities[-1] = (positions[-1] - positions[-2]) / dt_last
    
    # 创建包含速度的完整数据数组
    full_data = np.column_stack((data, velocities))
    
    # 返回初始条件和完整数据
    initial_conditions = np.array([
        data[0, 1],  # x_0
        data[0, 2],  # y_0
        data[0, 3],  # z_0
        velocities[0, 0],  # v_x0
        velocities[0, 1],  # v_y0
        velocities[0, 2]   # v_z0
    ])
    
    return initial_conditions, full_data

class KalmanFilter:
    """
    实现三维卡尔曼滤波器用于轨迹平滑。
    状态向量：[x, y, z, vx, vy, vz]
    测量向量：[x, y, z]
    """
    def __init__(self, dt=0.1):
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 测量矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差
        self.Q = np.eye(6) * 0.1
        
        # 测量噪声协方差
        self.R = np.eye(3) * 1.0
        
        # 状态估计协方差
        self.P = np.eye(6) * 1.0
        
        # 初始状态
        self.x = np.zeros(6)
        
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]
    
    def update(self, measurement):
        """更新步骤"""
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        # 更新状态协方差
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:3]

def apply_kalman_filter(data):
    """
    对轨迹数据应用卡尔曼滤波。
    
    :param data: 数组，可以是 [t,x,y,z] 或 [t,x,y,z,vx,vy,vz] 格式
    :return: 数组，结构为 [t,x,y,z,vx,vy,vz]，包含滤波后的结果
    """
    # 创建卡尔曼滤波器实例
    kf = KalmanFilter()
    
    # 准备结果数组
    filtered_states = []
    
    # 提取时间和位置数据
    times = data[:, 0]
    positions = data[:, 1:4]
    
    # 对每个时间点的数据进行滤波
    for i, (t, pos) in enumerate(zip(times, positions)):
        # 预测
        kf.predict()
        # 更新
        filtered_state = kf.update(pos)
        # 保存完整状态（位置和速度）
        full_state = np.concatenate(([t], filtered_state, kf.x[3:]))
        filtered_states.append(full_state)
    
    return np.array(filtered_states)

def analyze_noise_reduction(original_data, filtered_data, simulated_data=None):
    """
    分析降噪效果和模拟精度。
    
    :param original_data: 数组，原始位置数据 [x,y,z]
    :param filtered_data: 数组，滤波后位置数据 [x,y,z]
    :param simulated_data: 数组，可选，模拟位置数据 [x,y,z]
    :return: 包含统计信息的字典
    """
    # 计算原始数据和滤波后数据之间的差异
    filter_differences = np.sqrt(np.sum((original_data - filtered_data)**2, axis=1))
    
    # 初始化统计指标
    stats = {
        '原始-滤波平均差异': np.mean(filter_differences),
        '原始-滤波最大差异': np.max(filter_differences),
        '原始-滤波标准差': np.std(filter_differences)
    }
    
    # 如果提供了模拟数据，计算额外的统计信息
    if simulated_data is not None:
        # 计算模拟数据与原始数据的差异
        sim_orig_differences = np.sqrt(np.sum((original_data - simulated_data)**2, axis=1))
        stats.update({
            '原始-模拟平均差异': np.mean(sim_orig_differences),
            '原始-模拟最大差异': np.max(sim_orig_differences),
            '原始-模拟标准差': np.std(sim_orig_differences)
        })
        
        # 计算模拟数据与滤波数据的差异
        sim_filter_differences = np.sqrt(np.sum((filtered_data - simulated_data)**2, axis=1))
        stats.update({
            '滤波-模拟平均差异': np.mean(sim_filter_differences),
            '滤波-模拟最大差异': np.max(sim_filter_differences),
            '滤波-模拟标准差': np.std(sim_filter_differences)
        })
        
        # 计算相对改进百分比
        improvement = (np.mean(sim_orig_differences) - np.mean(sim_filter_differences)) / np.mean(sim_orig_differences) * 100
        stats['模型改进百分比'] = improvement
    
    return stats


def export_filtered_data(filtered_data, output_file='betterps.csv'):
    """
    导出滤波后的数据到CSV文件。
    
    :param filtered_data: 数组，结构为 [t, x, y, z, vx, vy, vz]
    :param output_file: 输出文件名，默认为 'betterps.csv'
    """
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['t', 'x', 'y', 'z'])
            # 写入数据
            for row in filtered_data:
                writer.writerow(row[:4])  # 只导出 t, x, y, z 列
        print(f"滤波后的数据已成功导出到 {output_file}")
    except Exception as e:
        print(f"导出数据时出错：{e}")