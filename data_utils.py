import pandas as pd
import numpy as np

def read_csv(file_path):
    """
    从CSV文件中读取数据。
    :param file_path: CSV文件路径
    :return: 包含时间和坐标数据的DataFrame
    """
    return pd.read_csv(file_path)

def get_initial_conditions(data):
    """
    根据数据计算回旋镖的初始条件。
    :param data: 包含时间和坐标数据的DataFrame
    :return: 初始位置、初始速度和时间数据
    """
    t_data = data['t'].values
    xyz_data = data[['x', 'y', 'z']].values

    # 初始位置
    x0, y0, z0 = xyz_data[0]

    # 初始速度（通过前两个点估算）
    if len(t_data) > 1:
        v_x0 = (xyz_data[1][0] - xyz_data[0][0]) / (t_data[1] - t_data[0])
        v_y0 = (xyz_data[1][1] - xyz_data[0][1]) / (t_data[1] - t_data[0])
        v_z0 = (xyz_data[1][2] - xyz_data[0][2]) / (t_data[1] - t_data[0])
    else:
        v_x0, v_y0, v_z0 = 1.0, 2.0, 0.0  # 默认值

    initial_state = [x0, y0, z0, v_x0, v_y0, v_z0]
    return t_data, xyz_data, initial_state
