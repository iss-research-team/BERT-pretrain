#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 下午2:42
# @Author  : liu yuhan
# @FileName: DNN_utils.py
# @Software: PyCharm

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


def get_data(model):
    '''
    针对字典会自动把int形key转换成str的情况
    我只能说 pickle yyds
    :return:
    '''

    x = np.load('../data/x_org_' + model + '.npy')
    y = np.load('../data/y_org_' + model + '.npy')

    ros = RandomOverSampler(random_state=0)
    x, y = ros.fit_resample(x, y)

    return torch.Tensor(x), torch.LongTensor(y)


# loss曲线绘制
def loss_draw(epochs, loss_list):
    plt.plot([i + 1 for i in range(epochs)], loss_list)
    plt.show()


