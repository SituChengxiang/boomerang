import numpy as np
from scipy.optimize import curve_fit

def fourier_polynomial_basis(t, n_fourier=3, n_poly=2):
    """
    生成傅里叶-多项式基函数。
    
    :param t: 时间数组
    :param n_fourier: 傅里叶项数
    :param n_poly: 多项式阶数
    :return: 基函数矩阵
    """
    basis = []
    # 添加多项式项
    for i in range(n_poly + 1):
        basis.append(t**i)
    
    # 添加傅里叶项
    for i in range(1, n_fourier + 1):
        basis.extend([np.sin(2*np.pi*i*t), np.cos(2*np.pi*i*t)])
    
    return np.array(basis).T

def fit_3d_curve(data, n_fourier=3, n_poly=2):
    """
    对3D轨迹数据进行傅里叶-多项式拟合。
    
    :param data: 数组，结构为[[t,x,y,z],...]
    :param n_fourier: 傅里叶项数
    :param n_poly: 多项式阶数
    :return: (系数数组, 拟合函数, 解析式)
    """
    t = data[:, 0]
    xyz = data[:, 1:4]
    
    # 生成基函数矩阵
    basis = fourier_polynomial_basis(t, n_fourier, n_poly)
    
    # 对每个坐标分量进行拟合
    coeffs = []
    fitted_funcs = []
    expressions = []
    
    for i, coord in enumerate(['x', 'y', 'z']):
        # 最小二乘拟合
        coeff = np.linalg.lstsq(basis, xyz[:, i], rcond=None)[0]
        coeffs.append(coeff)
        
        # 创建拟合函数
        def create_fit_func(coeff):
            def fit_func(t):
                basis = fourier_polynomial_basis(t, n_fourier, n_poly)
                return basis @ coeff
            return fit_func
        
        fitted_funcs.append(create_fit_func(coeff))
        
        # 生成解析式
        terms = []
        # 多项式项
        for j in range(n_poly + 1):
            if abs(coeff[j]) > 1e-10:  # 忽略接近0的系数
                if j == 0:
                    terms.append(f"{coeff[j]:.6f}")
                elif j == 1:
                    terms.append(f"{coeff[j]:.6f}t")
                else:
                    terms.append(f"{coeff[j]:.6f}t^{j}")
        
        # 傅里叶项
        idx = n_poly + 1
        for j in range(n_fourier):
            if abs(coeff[idx]) > 1e-10:
                terms.append(f"{coeff[idx]:.6f}sin({2*(j+1)}πt)")
            if abs(coeff[idx+1]) > 1e-10:
                terms.append(f"{coeff[idx+1]:.6f}cos({2*(j+1)}πt)")
            idx += 2
        
        expressions.append(f"{coord}(t) = " + " + ".join(terms))
    
    return coeffs, fitted_funcs, expressions