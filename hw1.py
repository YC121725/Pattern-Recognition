# 贝叶斯两类决策
from ast import If
import numpy as np
import matplotlib.pyplot as plt



def Bi_Bayes(mean1 = [1,0],
             mean2=[-1,0],
             cov1=np.matrix([[1,0],[0,1]]),
             cov2=np.matrix([[1,0],[0,1]]),
             N=50,
             isshow = True,
             issave = False):
    '''
    Parameter:
    -------------
    mean1,mean2：两类的均值
    cov1,cov2：两类分别的协方差
    N：各多少个样本值
    isshow：是否展示图片，默认True
    issave：是否保存图片，默认False
    '''
    
    np.random.seed(0)

    # N = 50                               # 样本数量
    # mean1 = [1, 0]                       # 第一类的均值
    # mean2 = [-1, 0]                      # 第二类的均值
    # cov = np.matrix([[1, 0], [0, 1]])    # 协方差矩阵

    x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=N)  # 随机高斯抽样
    x2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=N)

    # 把均值矩阵处理
    mean1 = np.matrix(mean1)
    mean2 = np.matrix(mean2)

    X = np.vstack((x1, x2))  # X拼接

    # 由公式计算得到
    # g1x g2x 实现课程中的判别函数
    W1 = mean1.dot(cov1.I).dot(X.T)
    W10 = mean1.dot(cov1.I).dot(mean1.T)
    g1x = W1 - 0.5*W10 + np.log(0.5)

    W2 = mean2.dot(cov2.I).dot(X.T)
    W20 = mean2.dot(cov2.I).dot(mean2.T)
    g2x = W2 - 0.5*W20 + np.log(0.5)

    # 计算
    # 平面方程 W[0,0]X1 + W[0,1]X2 = 0
    W = mean1.dot(cov1.I) - mean2.dot(cov2.I)
    C = 0.5 * (-W10-(-W20))
    # print(C)

    
    
    x_min = min(np.min(x2[:, 0]),np.min(x1[:, 0]))
    x_max = max(np.max(x2[:, 0]),np.max(x1[:, 0]))
    y_min = min(np.min(x2[:, 1]),np.min(x1[:, 1]))
    y_max = max(np.max(x2[:, 1]),np.max(x1[:, 1]))

    Xx1 = np.linspace(x_min-1,x_max+1)
    Xx2 = np.linspace(y_min-1,y_max+1)
    Xx1, Xx2 = np.meshgrid(Xx1, Xx2)
    z = W[0, 0]*Xx1 + W[0, 1]*Xx2 + C[0, 0]
    print('决策面方程为：{}x1 + {}x2 + {} = 0'.format(W[0, 0], W[0, 1],C[0, 0]))
    
    ''' 计算错误率'''
    w1_error = 0
    w2_error = 0
    
    for i in range(N):
        if i 
    
    
    
    
    if isshow:
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.title('Bayesian_Decision')
        plt.xlabel('X分量1')
        plt.ylabel('X分量2')
        # ax.set_xlim(min(x1[:, 0]+x2[:, 0]), max(x1[:, 1]+x2[:, 1]))
        # ax.set_ylim()
        plt.contour(Xx1, Xx2, z, 0)
        plt.scatter(x1[:, 0], x1[:, 1])
        plt.scatter(x2[:, 0], x2[:, 1])
        plt.legend(('Sample1','Sample2'),loc='upper right')
        plt.show()
        if issave:
            plt.savefig('./x1均值{}_{}_x2均值{}_{}_各{}.png'.format(mean1[0,0],mean1[0,1],mean2[0,0],mean2[0,1],N))
              
    return 0


if __name__=='__main__':
    mean1 = [1,0]
    mean2=[-1,0]
    cov1=np.matrix([[1,0],[0,1]])
    cov2=np.matrix([[1,0],[0,1]])
    Bi_Bayes(mean1,mean2,cov1,cov2,100)


