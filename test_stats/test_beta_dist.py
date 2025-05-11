#%matplotlib inline
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
from scipy.stats import beta
import sys

def test_gamma():
    import numpy as np
    from scipy.stats import gamma
    from matplotlib import pyplot as plt

    alpha_values = [1, 2, 3, 3, 3]
    beta_values = [0.5, 0.5, 0.5, 1, 2]
    color = ['b','r','g','y','m']
    x = np.linspace(1E-6, 10, 1000)

    fig, ax = plt.subplots(figsize=(12, 8))

    for k, t, c in zip(alpha_values, beta_values, color):
        dist = gamma(k, 0, t)
        plt.plot(x, dist.pdf(x), c=c, label=r'$\alpha=%.1f,\ \theta=%.1f$' % (k, t))

    plt.xlim(0, 10)
    plt.ylim(0, 2)

    plt.xlabel('$x$')
    plt.ylabel(r'$p(x|\alpha,\beta)$')
    plt.title('Gamma Distribution')

    plt.legend(loc=0)

    plt.savefig("gamma_distribution.png") 
    plt.show()

def plot1():
    # 参数
    alpha = 2 # alpha-1为成功的次数
    beta_param = 5 # beta-1为失败的次数

    # 生成 Beta 分布
    x = np.linspace(0, 1, 1000)
    pdf = beta.pdf(x, alpha, beta_param)


    # 计算期望和方差
    mean = beta.mean(alpha, beta_param)
    variance = beta.var(alpha, beta_param)
    print(f"期望: {mean}")
    print(f"方差: {variance}")

    # 绘制概率密度函数
    plt.plot(x, pdf, label=f'Beta({alpha}, {beta_param})')
    plt.title('Beta Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig("beta_distribution.png") 
    #plt.show() # 我是远程登录的,所以不能显示图形

def plot_all():
    import numpy as np
    from scipy.stats import beta
    from matplotlib import pyplot as plt

    alpha_values = [1/3,2/3,1,1,2,2,4,10,20]
    beta_values = [1,2/3,3,1,1,6,4,30,20]
    colors =  ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    x = np.linspace(0, 1, 1002)[1:-1]

    fig, ax = plt.subplots(figsize=(14,9))

    for a, b, c in zip(alpha_values, beta_values, colors):
        dist = beta(a, b)
        plt.plot(x, dist.pdf(x), c=c,label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

    plt.xlim(0, 1)
    plt.ylim(0, 6)

    plt.xlabel('$x$')
    plt.ylabel(r'$p(x|\alpha,\beta)$')
    plt.title('Beta Distribution')

    ax.annotate('Beta(1/3,1)', xy=(0.014, 5), xytext=(0.04, 5.2),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(10,30)', xy=(0.276, 5), xytext=(0.3, 5.4),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(20,20)', xy=(0.5, 5), xytext=(0.52, 5.4),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(1,3)', xy=(0.06, 2.6), xytext=(0.07, 3.1),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(2,6)', xy=(0.256, 2.41), xytext=(0.2, 3.1),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(4,4)', xy=(0.53, 2.15), xytext=(0.45, 2.6),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(1,1)', xy=(0.8, 1), xytext=(0.7, 2),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(2,1)', xy=(0.9, 1.8), xytext=(0.75, 2.6),
                arrowprops=dict(facecolor='black', arrowstyle='-'))
    ax.annotate('Beta(2/3,2/3)', xy=(0.99, 2.4), xytext=(0.86, 2.8),
                arrowprops=dict(facecolor='black', arrowstyle='-'))

    #plt.legend(loc=0)
    plt.show()
    plt.savefig("beta_distribution_alL.png") 

def plot_function():
    x = np.linspace(-10, 10, 1000)
    y = np.log(1+np.exp(x))+1 # 有点像relu,保证y的输出永远为正数, 这样y可以作为beta(a,b)的伪计数参数

    # 绘制概率密度函数
    plt.plot(x, y, label=f'log(1+exp(x))+1') 
    plt.title('log(1+exp(x))+1')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig("beta_tranform_func.png") 

if __name__=="__main__":
    plot_function()
    sys.exit()
    test_gamma()
    plot_all()
    plot1()