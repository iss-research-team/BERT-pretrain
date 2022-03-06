#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/13 下午2:33
# @Author  : liu yuhan
# @FileName: src.py
# @Software: PyCharm

from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import os
import numpy as np


def model_load(model_file):
    tokenizer = BertTokenizer.from_pretrained(model_file)
    model = BertModel.from_pretrained(model_file)
    return tokenizer, model


def data_load(filename):
    """
    文件写入
    :param filename:
    :return:
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


def doc_trans(model):
    if model == 'nsp':
        model_path = "../bert_model/checkpoint-nsp-10000"
    elif model == 'ptuning':
        model_path = "../bert_model/checkpoint-ptuning-all"
    elif model == 'origin':
        model_path = "../bert-model/chinese_wwm_ext_pytorch"
    elif model == 'mlm':
        model_path = "../bert_model/checkpoint-mlm-220000"
    else:
        raise ValueError(model + ' is not exist')

    # 载入模型
    tokenizer, bert_model = model_load(model_path)
    # 载入数据
    file_path = '../data/input/train.txt'

    train_x = []
    train_y = []

    s_list = data_load(file_path)
    # emb_save
    for text, label in tqdm(s_list):
        inputs = tokenizer(text, return_tensors="pt", max_length=512)
        outputs = bert_model(**inputs)
        last_hidden_states = torch.sum(outputs.last_hidden_state, dim=1).squeeze()
        train_x.append(last_hidden_states.tolist())
        train_y.append(label)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # 保存结果
    np.save('../data/x_org_' + model, train_x)
    np.save('../data/y_org_' + model, train_y)


if __name__ == '__main__':
    model = 'origin'
    # model = 'ptuning'
    # model = 'nsp'
    # model = 'mlm'

    doc_trans(model)
