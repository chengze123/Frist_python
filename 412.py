import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def f(x):
    y = np.tan(x)
    return y


# 创建一个画板和画布
fig, ax = plt.subplots()
x = np.linspace(-np.pi / 2, np.pi / 2, 200)
y = f(x)
# 创建坐标轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
l = ax.plot(x, y)
dot, = ax.plot([], [], 'r.')
xdata, ydata = [], []


# 初始化函数
def init():
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylim(-1, 2)
    return l


# 生成节点函数
def gen_dot():
    tdata, ldata = [], []
    xdata = [-np.pi / 2, np.pi / 2]
    ydata = f(xdata)
    ydata = ydata.tolist()
    a = xdata[0]
    b = xdata[1]
    x1 = a + 0.382 * (b - a)
    x2 = a + 0.618 * (b - a)
    e = 0.0005
    f1 = f(x1)
    f2 = f(x2)
    xdata.append(x1)
    xdata.append(x2)
    ydata.append(f1)
    ydata.append(f2)
    while (b - a > e):
        if (f2 > f1):
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + 0.382 * (b - a)
            f2 = f(x2)
            xdata.append(x2)
            ydata.append(f2)

        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + 0.618 * (b - a)
            f1 = f(x1)
            xdata.append(x1)
            ydata.append(f1)
    l = (a + b) / 2
    xdata.append(l)
    ydata.append(f(l))
    print(l)
    print(f(l))
    for i in range(len(xdata)):
        tdata.append(xdata[i])
        ldata.append(ydata[i])
        newdot = [tdata, ldata]
        yield newdot
    xdata, ydata, tdata, ldata = [], [], [], []


# 更新节点函数
def update_dot(newd):
    dot.set_data(newd[0], newd[1])
    return dot,


# 生成动态图像的节点
ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=1000, init_func=init)
ani.save('sin_dot.gif', writer='imagemagick', fps=30)

plt.show()
