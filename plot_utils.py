import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_data_points(data, title="3D Data Points", save_path=None):
    """
    绘制3D数据点。
    :param data: 包含x, y, z列的DataFrame或数组
    :param title: 图表标题
    :param save_path: 如果提供路径，则保存图像
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o', label='Data Points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_curve(t, x, y, z, title="3D Curve", save_path=None):
    """
    绘制3D曲线。
    :param t: 时间数组
    :param x: x坐标数组
    :param y: y坐标数组
    :param z: z坐标数组
    :param title: 图表标题
    :param save_path: 如果提供路径，则保存图像
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', label='Fit Curve')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_comparison(data_3d, data_xt, data_yt, data_zt, title="Comparison", save_path=None):
    """
    绘制数据点和曲线的对比图，包括一个三维视图和三个二维视图。
    :param data_3d: 三维数据点数组 [[x1, y1, z1], [x2, y2, z2], ...]
    :param data_xt: x-t 数据数组 [[t1, x1], [t2, x2], ...]
    :param data_yt: y-t 数据数组 [[t1, y1], [t2, y2], ...]
    :param data_zt: z-t 数据数组 [[t1, z1], [t2, z2], ...]
    :param title: 图表标题
    :param save_path: 如果提供路径，则保存图像
    """
    fig = plt.figure(figsize=(12, 10))
    
    # 左上角：3D视图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter([p[0] for p in data_3d], [p[1] for p in data_3d], [p[2] for p in data_3d], c='r', marker='o', label='Data Points')
    ax1.plot([p[0] for p in data_3d], [p[1] for p in data_3d], [p[2] for p in data_3d], 'b-', label='Fit Curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View')
    ax1.legend()
    
    # 右上角：z-t图
    ax2 = fig.add_subplot(222)
    ax2.plot([p[0] for p in data_zt], [p[1] for p in data_zt], 'ro', label='Data Points')
    ax2.plot([p[0] for p in data_zt], [p[1] for p in data_zt], 'b-', label='Fit Curve')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Z vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 左下角：y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot([p[0] for p in data_yt], [p[1] for p in data_yt], 'ro', label='Data Points')
    ax3.plot([p[0] for p in data_yt], [p[1] for p in data_yt], 'b-', label='Fit Curve')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # 右下角：x-t图
    ax4 = fig.add_subplot(224)
    ax4.plot([p[0] for p in data_xt], [p[1] for p in data_xt], 'ro', label='Data Points')
    ax4.plot([p[0] for p in data_xt], [p[1] for p in data_xt], 'b-', label='Fit Curve')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X vs Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
