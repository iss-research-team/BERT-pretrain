#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/13 下午2:33
# @Author  : liu yuhan
# @FileName: src.py
# @Software: PyCharm

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import os
import numpy as np
import json


def model_load(model_file):
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = AutoModel.from_pretrained(model_file)
    return tokenizer, model


def data_load(filename):
    """
    文件写入
    :param filename:
    :return:
    """
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)

    patent_id_list = [patent_id for patent_id, patent_info in data.items()]
    patent_id2index = {patent_id: index for index, patent_id in enumerate(patent_id_list)}

    data = [(patent_id2index[patent_id], patent_info['doc']) for patent_id, patent_info in data.items()]
    return data


def doc_trans_1():
    # 载入模型
    tokenizer, bert_model = model_load("roberta-large")
    bert_model.eval()
    bert_model.cuda()
    print('模型载入成功')
    # 载入数据
    file_path = '../data/input/doc_dict_patent_230602.json'

    data = data_load(file_path)
    # 将data分为4个部分
    rank = 1
    size = 4
    data = data[rank * len(data) // size: (rank + 1) * len(data) // size]

    print('数据载入成功，rank:%d length:%d' % (rank, len(data)))
    # 16个一组

    id_64 = []
    token_ids_64 = []
    id_128 = []
    token_ids_128 = []
    id_256 = []
    token_ids_256 = []
    id_512 = []
    token_ids_512 = []

    for index, text in tqdm(data, total=len(data)):
        token_ids = tokenizer.encode(text, max_length=512, truncation=True)
        if len(token_ids) <= 64:
            id_64.append(index)
            token_ids_64.append(token_ids)
        elif len(token_ids) <= 128:
            id_128.append(index)
            token_ids_128.append(token_ids)
        elif len(token_ids) <= 256:
            id_256.append(index)
            token_ids_256.append(token_ids)
        else:
            id_512.append(index)
            token_ids_512.append(token_ids)

    print(len(token_ids_64))
    print(len(token_ids_128))
    print(len(token_ids_256))
    print(len(token_ids_512))

    # save
    with open('../data/output/id_64.json', 'w') as f:
        json.dump(id_64, f)
    with open('../data/output/id_128.json', 'w') as f:
        json.dump(id_128, f)
    with open('../data/output/id_256.json', 'w') as f:
        json.dump(id_256, f)
    with open('../data/output/id_512.json', 'w') as f:
        json.dump(id_512, f)

    with open('../data/output/token_ids_64.json', 'w') as f:
        json.dump(token_ids_64, f)
    with open('../data/output/token_ids_128.json', 'w') as f:
        json.dump(token_ids_128, f)
    with open('../data/output/token_ids_256.json', 'w') as f:
        json.dump(token_ids_256, f)
    with open('../data/output/token_ids_512.json', 'w') as f:
        json.dump(token_ids_512, f)

    # for i in tqdm(range(0, data_length, batch_size)):
    #     inputs = tokenizer([text for _, text in data[i:i + batch_size]],
    #                        return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    #     inputs = inputs.to('cuda')
    #     outputs = bert_model(**inputs)
    #     last_hidden_states = torch.sum(outputs.last_hidden_state, dim=1).squeeze()
    #     train_x.extend(last_hidden_states.cpu().tolist())
    # print('数据转化成功')
    # print(len(train_x))
    # train_x = np.array(train_x)
    # # 保存结果
    # np.save('../data/output/patent_feature_230602.npy', train_x)
    # print('数据保存成功')


def doc_trans_2(length=256, batch_size=20):
    print(length, batch_size)
    # model_load
    tokenizer, bert_model = model_load("roberta-large")
    bert_model.to('cuda')
    bert_model.eval()
    with open('../data/output/node1_feature/id_{}.json'.format(length), 'r') as f:
        id_list = json.load(f)

    print(len(id_list))

    with open('../data/output/token_ids_{}.json'.format(length), 'r') as f:
        token_ids_list = json.load(f)

    attention_mask_list = [[1] * len(token_ids) for token_ids in token_ids_list]
    # padding
    token_ids_list = [token_ids + [0] * (length - len(token_ids)) for token_ids in token_ids_list]
    attention_mask_list = [attention_mask + [0] * (length - len(attention_mask)) for attention_mask in
                           attention_mask_list]
    # encode
    train_x = []
    data_length = len(token_ids_list)
    for i in tqdm(range(0, data_length, batch_size)):
        inputs = token_ids_list[i:i + batch_size]
        attention_mask = attention_mask_list[i:i + batch_size]
        inputs = torch.tensor(inputs).to('cuda')
        attention_mask = torch.tensor(attention_mask).to('cuda')
        outputs = bert_model(inputs, attention_mask=attention_mask)
        last_hidden_states = torch.sum(outputs.last_hidden_state, dim=1).squeeze()
        if last_hidden_states.shape != torch.Size([batch_size, 1024]):
            print(last_hidden_states.shape)

        train_x.extend(last_hidden_states.cpu().tolist())

    print('数据转化成功')
    print(len(train_x))
    train_x = np.array(train_x)
    # 保存结果
    np.save('../data/output/patent_feature_{}.npy'.format(length), train_x)
    print('数据保存成功')


def check(node='node1'):
    """
    检查embedding是否完整
    :param node:
    测试通过，没有问题。
    """
    length_list = [64, 128, 256, 512]
    result_path = '../data/output/{}_feature'.format(node)
    for length in length_list:
        print(node, length)
        id_path = os.path.join(result_path, 'id_{}.json'.format(length))
        with open(id_path, 'r') as f:
            id_list = json.load(f)
        print('id_list', len(id_list))
        feature_path = os.path.join(result_path, 'patent_feature_{}.npy'.format(length))
        feature = np.load(feature_path, allow_pickle=True)
        print('feature shape', feature.shape)


if __name__ == '__main__':
    # doc_trans_2()
    for node in ['node1']:
        check(node=node)
