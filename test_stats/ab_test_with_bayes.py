
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

def ab_test():
    people_in_branch = 40 # 如果为4000,则对比更加明显
    # Control is Alpaca, Experiment is Bear
    control, experiment = np.random.rand(2, people_in_branch)
    c_successes = sum(control < 0.16)
    # Bears are about 10% better relative to Alpacas
    e_successes = sum(experiment < 0.176)
    c_failures = people_in_branch - c_successes
    e_failures = people_in_branch - e_successes
    # Our Priors
    prior_successes = 8
    prior_failures = 42
    # For our graph
    fig, ax = plt.subplots(1, 1)
    # Control
    c_alpha, c_beta = c_successes + prior_successes, c_failures + prior_failures
    # Experiment
    e_alpha, e_beta = e_successes + prior_successes, e_failures + prior_failures
    x = np.linspace(0., 0.5, 1000)
    # Generate and plot the distributions!
    c_distribution = beta(c_alpha, c_beta)
    e_distribution = beta(e_alpha, e_beta)
    ax.plot(x, c_distribution.pdf(x))
    ax.plot(x, e_distribution.pdf(x))
    ax.set(xlabel='conversion rate', ylabel='density')
    fig.savefig("ab_test_beta.png")
    fig.show()


    # Arguments are x values so use ppf - the inverse of cdf
    print(c_distribution.ppf([0.025, 0.5, 0.975]))
    print(e_distribution.ppf([0.025, 0.5, 0.975]))
    # [ 0.14443947  0.15530981  0.16661068]
    # [ 0.15770843  0.16897057  0.18064618]

    # 是 SciPy 中概率分布对象的一个方法，全称为 Random Variates Sample，用于从指定的概率分布中生成随机样本。
    sample_size = 100000
    c_samples = pd.Series([c_distribution.rvs() for _ in range(sample_size)])
    e_samples = pd.Series([e_distribution.rvs() for _ in range(sample_size)])
    p_value = 1.0 - sum(e_samples > c_samples)/sample_size # 实验组随机采样的样本转化率优于对照组
    # 0.046830000000000038
    # p-values指标小于0.05，我们有信心相信硬币反面概率更可能较大。


    fig, ax = plt.subplots(1, 1)
    ser = pd.Series(e_samples/c_samples)
    # Make the CDF
    ser = ser.sort_values()
    ser[len(ser)] = ser.iloc[-1] 
    cum_dist = np.linspace(0., 1., len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
    ax.plot(ser_cdf)
    ax.set(xlabel='Bears / Alpacas', ylabel='CDF')
    fig.savefig("ab_test_beta_cdf.png")



if __name__=="__main__":
    ab_test()