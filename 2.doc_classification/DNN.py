#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 下午2:45
# @Author  : liu yuhan
# @FileName: DNN.py
# @Software: PyCharm

from DNN_model import *
from DNN_utils import *

from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    batch_size = 64
    epochs = 150
    pre_train_model = 'nsp'
    # 载入数据
    data_x, data_y = get_data(pre_train_model)
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.7)

    loader = Data.DataLoader(MyDataSet(data_x_train, data_y_train), batch_size, True)
    # 参数设置
    dim = 768
    h_1 = dim * 2
    h_2 = dim
    d_o = 15
    # 模型初始化
    dnn = Dnn(dim, h_1, h_2, d_o)
    dnn.cuda()
    fun_loss = nn.CrossEntropyLoss()
    fun_loss.cuda()
    optimizer = optim.Adam(dnn.parameters(), lr=0.00001)

    # 计算平均的loss
    ave_loss = []

    # 保存
    model_save_path = '../data/zr_dnn_' + pre_train_model + '.pt'
    evaluator = Evaluator(model_save_path)

    for epoch in tqdm(range(epochs)):
        loss_collector = []
        for i, (x, label) in enumerate(loader):
            optimizer.zero_grad()
            x = x.cuda()
            label = label.cuda()
            output = dnn(x)
            loss = fun_loss(output, label)
            loss.backward()
            optimizer.step()

            loss_collector.append(loss.item())
        ave_loss.append(np.mean(loss_collector))
        evaluator.evaluate(epoch, dnn, data_x_test, data_y_test, np.mean(loss_collector))

    print(evaluator.status_best)
    loss_draw(epochs, ave_loss)
