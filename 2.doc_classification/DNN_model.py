#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 上午9:03
# @Author  : liu yuhan
# @FileName: DNN_model.py
# @Software: PyCharm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from sklearn.metrics import precision_score, recall_score, f1_score


class Dnn(nn.Module):
    def __init__(self, dim, hid_1, hid_2, d_output):
        super(Dnn, self).__init__()
        self.liner_1 = nn.Linear(dim, hid_1)
        self.liner_2 = nn.Linear(hid_1, hid_2)
        self.liner_3 = nn.Linear(hid_2, hid_2)
        self.liner_4 = nn.Linear(hid_2, hid_1)
        self.liner_5 = nn.Linear(hid_1, d_output)

    def forward(self, x):
        output = F.relu(self.liner_1(x))
        output = F.relu(self.liner_2(output))
        output = F.relu(self.liner_3(output))
        output = F.relu(self.liner_4(output))
        output = self.liner_5(output)
        return output


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, x, label):
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


class Evaluator:
    def __init__(self, model_save_path):
        self.f1_best = 0
        self.status_best = []
        self.model_save_path = model_save_path

    def evaluate(self, epoch, model, test_x, test_y, loss):
        torch.set_grad_enabled(False)
        out = model(test_x.cuda())
        prediction = torch.max(out.cpu(), 1)[1]
        pred_y = prediction.data.numpy()
        target_y = test_y.data.numpy()

        precision = precision_score(target_y, pred_y, average='weighted')
        recall = recall_score(target_y, pred_y, average='weighted')
        f1 = f1_score(target_y, pred_y, average='weighted')
        status = ["epoch", epoch, "loss", loss, 'precision:', precision, 'recall:', recall, 'f1:', f1]
        print(status)
        if self.f1_best < f1:
            self.f1_best = f1
            self.status_best = status
            torch.save(model.state_dict(), self.model_save_path)

        torch.set_grad_enabled(True)
