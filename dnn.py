import decimal
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def net_init():
    net = nn.Sequential(
        nn.Linear(768, 768),  # 输入层与第一隐层结点数设置，全连接结构
        nn.Dropout(0.5),
        torch.nn.ReLU(),  # 第一隐层激活函数采用sigmoid
        nn.Linear(768, 768),  # 第一隐层与第二隐层结点数设置，全连接结构
        nn.Dropout(0.5),
        torch.nn.ReLU(),  # 第一隐层激活函数采用sigmoid
        nn.Linear(768, 2),  # 第二隐层与输出层层结点数设置，全连接结构
        torch.nn.Softmax(dim=1)
    )
    return net


def train(net, batch, batch_label):
    # 建立网络
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 优化器使用随机梯度下降，传入网络参数和学习率
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失函数
    loss1 = 0
    # 模型训练
    for i in range(0, len(batch)):
        y_p = net(batch[i])
        loss = loss_func(y_p, batch_label[i].long())  # 计算损失
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 计算梯度，误差回传
        optimizer.step()  # 根据计算的梯度，更新网络中的参数
        loss1 = loss.data.item()
    return loss1


def test_model(net, x_test, y_test):
    y_p = net(x_test[0])
    y_h = y_test[0]
    summ = 0
    y_p_l = torch.zeros((y_h.shape[0]))
    for i in range (0, y_h.shape[0]):
        if y_p[i,0] >= 0.5:
            y_p_l[i] = 0
        else:
            y_p_l[i] = 1
        if y_p_l[i] == y_h[i]:
            summ = summ + 1
    acc = summ / y_h.shape[0]
    return acc