# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # normalize设置是否将输入图像正规化为0.0～1.0的值。如果将该参数设置为False，则输入图像的像素会保持原来的0～255。
    # 第2个参数flatten设置是否展开输入图像（变成一维数组）​。如果将该参数设置为False，则输入图像为1×28×28的三维数组；
    # 若设置为True，则输入图像会保存为由784个元素构成的一维数组。
    # 第3个参数one_hot_label设置是否将标签保存为one-hot表示(one-hot representation)。
    # one-hot表示是仅正确解标签为1，其余皆为0的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。当one_hot_label为False时，只是像7、2这样简单保存正确解标签；当one_hot_label为True时，标签则保存为one-hot表示。
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:  # 读取学习到的权重参数
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)  # 第一个隐藏层结果
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)  # 第二个隐藏层结果
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)  # 输出层

    return y


# 单个图片处理
# x, t = get_data()
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):  # 逐一取出保存在x中的图像数据
#     y = predict(network, x[i])  # 分类，以Numpy数组的形式输出各个标签对应的概率
#     p= np.argmax(y) # 获取概率最高的元素的索引
#     if p == t[i]:  # 比较神经网络所预测的结果和正确解标签
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 0.9352

# 批处理加速
x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]  # x[i:i+batch_size]从输入数据中抽出批数据。
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 从每一行中找到最大的数，返回索引下标
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 0.9352
