"""
计算穿刺针的形变方程的代码
用于将论文中的公式转化为代码

"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.misc import derivative

#=========================================================================================================================================================
# E =σ/ε=(F / A)/(ΔL/ L 0)= FL 0 /AΔL

# 其中：

# E是杨氏模量，通常以帕斯卡(Pa)表示

# σ是单轴应力

# ε是应变

# F是压缩力或伸展力

# A是横截面积或垂直于作用力的横截面

# ΔL是长度的变化(压缩时为负；拉伸时为正)

# L 0是原始长度

#=========================================================================================================================================================

# 杨氏模量计算
F = 500.0  # 牛顿
A = 0.002  # 平方米
delta_L = -0.001  # 米
L0 = 1.0   # 米

def E_i(F, A, delta_L, L0):
# 检查 ΔL 是否为零
    if delta_L == 0:
        print("错误：长度变化 ΔL 不能为零。")
    else:
        E = (F * L0) / (A * delta_L)
        print(f"计算得到的杨氏模量 E 为：{E} Pa")
    return E
    
    
# 惯性矩计算
def moment_of_inertia(R, sigma):
    integrand = lambda r: r**3
    result, error = quad(integrand, 0, R)
    I = 2 * np.pi * sigma * result
    return I



#=========================================================================================================================================================
# 占位符函数
def v_i(x):
    # 这里应该是 v_i(x) 的实际函数
    return x  # 临时占位符

def v_i_minus_1(x):
    # 这里应该是 v_{i-1}(x) 的实际函数
    return x  # 临时占位符

def P(x, y):
    # 这里应该是 P(x, y) 的实际函数
    return y  # 临时占位符


def second_derivative(func, x, dx=1e-6):
    return derivative(func, x, dx=dx, n=2)

def integrand_N_E_i(x):
    d2v_i_dx2 = second_derivative(v_i, x)
    d2v_i_minus_1_dx2 = second_derivative(v_i_minus_1, x)
    return (d2v_i_dx2**2 - d2v_i_minus_1_dx2**2)





#绕度方程
def V1(x,a0=0,a1=0,a2=0,a3=0):
    return a0 + a1*x + a2*x*x + a3*x*x*x

#绕度方程
def V2(x,a0=0,a1=0,a2=0,a3=0):
    return a0 + a1*x + a2*x*x + a3*x*x*x

x = o = r1 = 0
d1 = d2 = d3 =0

X_new = Y_new = X_new2 = Y_new2 = 0

#第一层组织
while X_new <= d1:
    X_new = x * np.cos(o) -V1(x) * np.sin(o)

    Y_new = V1(x) * np.cos(o) + x * np.sin(o) + r1

#第二层组织
while X_new2 <= d2:
    X_new2 = x * np.cos(o) -V1(x) * np.sin(o) + X_new

    Y_new2 = V1(x) * np.cos(o) + x * np.sin(o) + Y_new

