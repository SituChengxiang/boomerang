import numpy as np

params_fixed = {
    'a': 0.15,        # 翼展 (m)
    'd': 0.028,       # 翼宽 (m)
    'm': 0.002183,    # 质量 (kg)
    'I': (5/24) * 0.002183 * (0.15)**2,  # 转动惯量
    'omega': 10.0,    # 固定角速度 (rad/s)
    'rho': 1.225,     # 空气密度 (kg/m^3)
    'g': 9.793        # 重力加速度 (m/s²)
}

def read_csv(file_path='ps1.csv'):
    """
    读取CSV文件中的回旋镖轨迹数据。
    
    :param file_path: CSV文件路径，默认为'ps1.csv'
    :return: 数组，结构为[[t,x,y,z],...]
    """
    try:
        # 使用numpy读取CSV文件，跳过header行
        xyz_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        
        # 检查数据维度
        if xyz_data.shape[1] != 3:
            raise ValueError("CSV文件必须包含三列：x,y,z")
        
        # 生成时间数组（假设采样间隔为0.1秒）
        t = np.arange(len(xyz_data)) * 0.1
        
        # 组合时间和坐标数据
        return np.column_stack([t, xyz_data])
        
    except Exception as e:
        raise Exception(f"读取CSV文件时出错：{str(e)}")

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
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :return: 数组，结构为[[t,x,y,z],...]，包含滤波后的结果
    """
    # 创建卡尔曼滤波器实例
    kf = KalmanFilter()
    
    # 准备结果数组
    filtered_positions = []
    
    # 对每个时间点的数据进行滤波
    for point in data:
        t, x, y, z = point
        measurement = np.array([x, y, z])
        kf.predict()
        filtered_pos = kf.update(measurement)
        filtered_positions.append([t, *filtered_pos])
    
    return np.array(filtered_positions)

def analyze_noise_reduction(original_data, filtered_data):
    """
    分析降噪效果。
    
    :param original_data: 数组，结构为[[t,x,y,z],...]，原始数据
    :param filtered_data: 数组，结构为[[t,x,y,z],...]，滤波后数据
    :return: 包含统计信息的字典
    """
    # 计算原始数据和滤波后数据之间的差异
    # 注意：跳过时间列（索引0），只比较xyz坐标（索引1:4）
    differences = np.sqrt(np.sum((original_data[:, 1:4] - filtered_data[:, 1:4])**2, axis=1))
    
    # 计算统计指标
    stats = {
        'mean_difference': np.mean(differences),
        'max_difference': np.max(differences),
        'std_difference': np.std(differences)
    }
    
    return stats