import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

# 设置中文字体  
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN']  # 使用思源黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_dual_trajectories(data1, data2, title="双轨迹对比图", save_path=None):
    """
    绘制两个轨迹的对比图，包括3D视图和三个时间序列图。
    
    :param data1: 第一个轨迹数据，数组，结构为[[t,x,y,z],...]
    :param data2: 第二个轨迹数据，数组，结构为[[t,x,y,z],...]
    :param title: 图表标题
    :param save_path: 保存图像的路径（可选）
    :return: 包含所有子图的figure对象
    """
    fig = plt.figure(figsize=(12, 10))
    
    # 提取时间和坐标数据
    t1 = data1[:, 0]
    xyz1 = data1[:, 1:4]
    
    t2 = data2[:, 0]
    xyz2 = data2[:, 1:4]
    
    # 左上角：3D视图
    ax1 = fig.add_subplot(221, projection='3d')
    # 绘制第一个轨迹
    ax1.scatter(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], c='#145ca0', marker='o', label='轨迹1数据点')
    ax1.plot(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], c='#004f989d', alpha=0.7, label='轨迹1曲线')
    # 绘制第二个轨迹
    ax1.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2], c='#E74C3C', marker='o', label='轨迹2数据点')
    ax1.plot(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2], c='#9d2f239d', alpha=0.7, label='轨迹2曲线')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹对比')
    ax1.legend()
    
    # 右上角：z-t图
    ax2 = fig.add_subplot(222)
    # 绘制第一个轨迹
    ax2.scatter(t1, xyz1[:, 2], c='#145ca0', marker='o', label='轨迹1数据点')
    ax2.plot(t1, xyz1[:, 2], c='#004f989d', alpha=0.7, label='轨迹1曲线')
    # 绘制第二个轨迹
    ax2.scatter(t2, xyz2[:, 2], c='#E74C3C', marker='o', label='轨迹2数据点')
    ax2.plot(t2, xyz2[:, 2], c='#9d2f239d', alpha=0.7, label='轨迹2曲线')
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Z坐标随时间变化')
    ax2.legend()
    ax2.grid(True)
    
    # 左下角：y-t图
    ax3 = fig.add_subplot(223)
    # 绘制第一个轨迹
    ax3.scatter(t1, xyz1[:, 1], c='#145ca0', marker='o', label='轨迹1数据点')
    ax3.plot(t1, xyz1[:, 1], c='#004f989d', alpha=0.7, label='轨迹1曲线')
    # 绘制第二个轨迹
    ax3.scatter(t2, xyz2[:, 1], c='#E74C3C', marker='o', label='轨迹2数据点')
    ax3.plot(t2, xyz2[:, 1], c='#9d2f239d', alpha=0.7, label='轨迹2曲线')
    
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y坐标随时间变化')
    ax3.legend()
    ax3.grid(True)
    
    # 右下角：x-t图
    ax4 = fig.add_subplot(224)
    # 绘制第一个轨迹
    ax4.scatter(t1, xyz1[:, 0], c='#145ca0', marker='o', label='轨迹1数据点')
    ax4.plot(t1, xyz1[:, 0], c='#004f989d', alpha=0.7, label='轨迹1曲线')
    # 绘制第二个轨迹
    ax4.scatter(t2, xyz2[:, 0], c='#E74C3C', marker='o', label='轨迹2数据点')
    ax4.plot(t2, xyz2[:, 0], c='#9d2f239d', alpha=0.7, label='轨迹2曲线')
    
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X坐标随时间变化')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_fourier_fit(data, fitted_funcs, title="傅里叶拟合结果", save_path=None):
    """
    绘制原始数据点和傅里叶拟合曲线的对比图。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :param fitted_funcs: 三个拟合函数的列表 [x(t), y(t), z(t)]
    :param title: 图表标题
    :param save_path: 保存图像的路径（可选）
    :return: 包含所有子图的figure对象
    """
    fig = plt.figure(figsize=(12, 10))
    
    # 提取数据
    t = data[:, 0]
    xyz = data[:, 1:4]
    
    # 生成更密集的时间点用于绘制平滑曲线
    t_smooth = np.linspace(t.min(), t.max(), 500)
    xyz_fit = np.array([f(t_smooth) for f in fitted_funcs]).T
    
    # 左上角：3D视图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='数据点')
    ax1.plot(xyz_fit[:, 0], xyz_fit[:, 1], xyz_fit[:, 2], 'b-', label='拟合曲线')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D视图')
    ax1.legend()
    
    # 右上角：z-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t, xyz[:, 2], 'ro', label='数据点')
    ax2.plot(t_smooth, fitted_funcs[2](t_smooth), 'b-', label='拟合曲线')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Z vs 时间')
    ax2.legend()
    ax2.grid(True)
    
    # 左下角：y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t, xyz[:, 1], 'ro', label='数据点')
    ax3.plot(t_smooth, fitted_funcs[1](t_smooth), 'b-', label='拟合曲线')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y vs 时间')
    ax3.legend()
    ax3.grid(True)
    
    # 右下角：x-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t, xyz[:, 0], 'ro', label='数据点')
    ax4.plot(t_smooth, fitted_funcs[0](t_smooth), 'b-', label='拟合曲线')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X vs 时间')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_3d_points(data, title="3D轨迹数据点", new_figure=True, save_path=None):
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
    
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='数据点')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if new_figure:
        plt.show()
        return fig, ax
    return ax

def plot_3d_curve(data, title="3D轨迹曲线", new_figure=True, save_path=None):
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
    
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='轨迹曲线')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if new_figure:
        plt.show()
        return fig, ax
    return ax

def plot_trajectory_analysis(data, title="轨迹分析", save_path=None):
    """
    绘制轨迹的综合分析图，包括3D视图和三个时间序列图。
    
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
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='数据点')
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', alpha=0.5, label='轨迹曲线')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹')
    ax1.legend()
    
    # 右上角：z-t图
    ax2 = fig.add_subplot(222)
    ax2.plot(t, xyz[:, 2], 'ro', label='数据点')
    ax2.plot(t, xyz[:, 2], 'b-', alpha=0.5, label='轨迹曲线')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Z坐标随时间变化')
    ax2.legend()
    ax2.grid(True)
    
    # 左下角：y-t图
    ax3 = fig.add_subplot(223)
    ax3.plot(t, xyz[:, 1], 'ro', label='数据点')
    ax3.plot(t, xyz[:, 1], 'b-', alpha=0.5, label='轨迹曲线')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y坐标随时间变化')
    ax3.legend()
    ax3.grid(True)
    
    # 右下角：x-t图
    ax4 = fig.add_subplot(224)
    ax4.plot(t, xyz[:, 0], 'ro', label='数据点')
    ax4.plot(t, xyz[:, 0], 'b-', alpha=0.5, label='轨迹曲线')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X坐标随时间变化')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_dynamic_parameters(times, solves, save_path='boomerang_params.png'):
    """
    绘制回旋镖动力学参数（扭矩系数、升力系数和阻力系数）随时间的变化。
    
    :param times: 时间数组
    :param solves: 解数组，结构为[[D, CL, CD], ...]
    :param save_path: 保存图像的路径
    :return: 图形对象
    """
    from scipy import stats
    import numpy as np
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    param_names = ['扭矩系数 (D)', '升力系数 (CL)', '阻力系数 (CD)']
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(times, solves[:, i], 'o-', label=f'原始{name}')
        ax.axhline(y=np.average(solves[:, i]), color='r', linestyle='-', label='平均值')
        
        # 添加去除异常值后的平均值
        param_z = np.abs(stats.zscore(solves[:, i]))
        clean_data = solves[param_z < 3, i]
        ax.axhline(y=np.average(clean_data), color='g', linestyle='--', label='净化平均值')
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel(name)
        ax.set_title(f'{name}随时间的变化')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle('回旋镖动力学参数随时间的变化')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_advdynamic_parameters(times, solves, skip_points=0, save_path='boomerang_params.png'):
    """
    绘制回旋镖动力学参数（扭矩系数、升力系数和阻力系数）随时间的变化。
    
    :param times: 时间数组
    :param solves: 解数组，结构为[[D, CL, CD], ...]
    :param skip_points: 计算统计量时忽略的起始点数量
    :param save_path: 保存图像的路径
    :return: 图形对象
    """
    from scipy import stats
    import numpy as np
    
    # 分离有效数据和忽略的数据
    times_valid = times[skip_points:]
    solves_valid = solves[skip_points:]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    param_names = ['扭矩系数 (D)', '升力系数 (CL)', '阻力系数 (CD)']
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        # 绘制完整数据，但区分有效点和忽略的点
        if skip_points > 0:
            ax.plot(times[:skip_points], solves[:skip_points, i], 'rx', markersize=8, 
                   label=f'忽略的前{skip_points}个点')
        
        # 绘制有效数据点
        ax.plot(times_valid, solves_valid[:, i], 'bo-', label=f'有效{name}数据')
        
        # 计算并显示平均线 (只使用有效数据)
        ax.axhline(y=np.average(solves_valid[:, i]), color='r', linestyle='-', 
                  label='平均值 (有效数据)')
        
        # 添加去除异常值后的平均值
        param_z = np.abs(stats.zscore(solves_valid[:, i]))
        clean_data = solves_valid[param_z < 3, i]
        ax.axhline(y=np.average(clean_data), color='g', linestyle='--', 
                  label='净化平均值 (Z-score < 3)')
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel(name)
        ax.set_title(f'{name}随时间的变化')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle('回旋镖动力学参数随时间的变化')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig