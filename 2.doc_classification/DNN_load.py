#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 下午2:45
# @Author  : liu yuhan
# @FileName: DNN.py
# @Software: PyCharm

from DNN_model import *
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import os

batch_size = 128
epochs = 50
dim = 768
h_1 = dim * 2
h_2 = dim
d_o = 15
model_save_path = '../data/zr_dnn.pt'
MODEL_NAME = "bert-base-chinese"


def model_load(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model


def sentence_split(file_path):
    file = open(file_path, 'r', encoding='UTF-8')
    str_list = file.readlines()
    # str_list = [' '.join(s.strip().split()[:500]) for s in str_list]
    str_list = [s.strip() for s in str_list]
    return str_list


def data_loade():
    # 载入模型
    tokenizer, model = model_load(MODEL_NAME)
    # 载入数据
    file_path = '../data'
    file_list = sorted(os.listdir(file_path))

    train_x = []

    for label, file in enumerate(file_list):
        s_list = sentence_split(os.path.join(file_path, file))
        print(file, len(s_list))
        # emb_save
        for i, s in tqdm(enumerate(s_list)):
            inputs = tokenizer(s, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = torch.sum(outputs.last_hidden_state, dim=1).squeeze()
            train_x.append(last_hidden_states.tolist())

    return torch.Tensor(train_x)


if __name__ == '__main__':
    # 模型初始化
    dnn = Dnn(dim, h_1, h_2, d_o)
    dnn.load_state_dict(torch.load(model_save_path))
    print('model load done.')

    data_x_test = data_loade()
    out = dnn(data_x_test)
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    print(pred_y)
