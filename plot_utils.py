import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_data_points(data, title="3D Data Points", new_figure=True, save_path=None):
    """
    绘制3D数据点。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :param title: 图表标题
    :param new_figure: 是否创建新的图形窗口
    :param save_path: 保存图像的路径（可选）
    :return: 如果new_figure为True，返回(fig, ax)元组；否则返回ax
    """
    if new_figure:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gca(projection='3d')
        fig = plt.gcf()
    
    # 提取x,y,z坐标
    xyz = data[:, 1:4]
    
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='Data Points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    if new_figure:
        plt.show()
        return fig, ax
    return ax

def plot_curve(data, title="3D Curve", new_figure=True, save_path=None):
    """
    绘制3D曲线。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :param title: 图表标题
    :param new_figure: 是否创建新的图形窗口
    :param save_path: 保存图像的路径（可选）
    :return: 如果new_figure为True，返回(fig, ax)元组；否则返回ax
    """
    if new_figure:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gca(projection='3d')
        fig = plt.gcf()
    
    # 提取x,y,z坐标
    xyz = data[:, 1:4]
    
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='Fit Curve')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    if new_figure:
        plt.show()
        return fig, ax
    return ax

def plot_comparison(data, title="Comparison", save_path=None):
    """
    绘制数据的综合对比图，包括3D视图和三个时间序列图。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :param title: 图表标题
    :param save_path: 保存图像的路径（可选）
    :return: 包含所有子图的figure对象
    """
    fig = plt.figure(figsize=(12, 10))
    
    # 提取时间和坐标数据
    t = data[:, 0]
    xyz = data[:, 1:4]
    
    # 左上角：3D视图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='Data Points')
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='Fit Curve')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View')
    ax1.legend()
    
    # 右上角：z-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t, xyz[:, 2], 'ro', label='Data Points')
    ax2.plot(t, xyz[:, 2], 'b-', label='Fit Curve')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Z vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 左下角：y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t, xyz[:, 1], 'ro', label='Data Points')
    ax3.plot(t, xyz[:, 1], 'b-', label='Fit Curve')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # 右下角：x-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t, xyz[:, 0], 'ro', label='Data Points')
    ax4.plot(t, xyz[:, 0], 'b-', label='Fit Curve')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X vs Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    return fig