import numpy as np
from numpy.random import randn
from math import log,exp

# 超参数
# TODO: You can change the hyperparameters here
lr = 1e-2  # 学习率
wd = 1e-2  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d*1
    @param bias: 1*1
    @return: wx+b
    """
    # TODO: YOUR CODE HERE
    n,d = np.shape(X)
    a = [np.dot(X[i],weight)+bias for i in range(n)]
    result:np.ndarray = np.array(a)
    return result
    raise NotImplementedError

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d*1
    @param bias: 1*1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: 1*1 由交叉熵损失函数计算得到
        weight: d*1 更新后的weight参数
        bias: 1*1 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n,d = np.shape(X)
    Loss = 0
    wN = len(weight)
    tempW = randn(wN)
    tempB = 0
    haty = predict(X, weight, bias)

    for i in range(n):
        f = (np.dot(X[i],weight)+bias)/1000
        print(f)
        Loss += log(1+np.exp(-Y[i]*f))
        tempW += (1 - sigmoid(Y[i]*f))*Y[i]*X[i]
        tempB += (1 - sigmoid(Y[i]*f))*Y[i]
    Loss = Loss/n + wd*np.linalg.norm(weight)
    weight,bias = weight + lr*(1/n)*tempW - 2*wd*weight, bias + lr*(1/n)*tempB
    return haty,Loss,weight,bias
    raise NotImplementedError
