#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 下午9:37
# @Author  : liu yuhan
# @FileName: 4.inst2vec.py
# @Software: PyCharm

import json

import numpy as np
from tqdm import tqdm
from collections import defaultdict


def str2set(inf_str):
    inf_list = inf_str.split(' | ')
    inf_list = [inf.lower() for inf in inf_list]
    inf_set = set(filter(None, inf_list))
    return inf_set


def get_patent2id():
    patent2id_path = '../data/input/patent2index.json'
    with open(patent2id_path, 'r', encoding='UTF-8') as file:
        patent2id = json.load(file)
    return patent2id


def inst2patent():
    patent2id = get_patent2id()

    patent2node_path = '../data/input/relationship_dict_patent_tmp.json'
    with open(patent2node_path, 'r', encoding='UTF-8') as file:
        patent2node = json.load(file)

    inst2patent = defaultdict(set)

    for patent, node_inf in tqdm(patent2node.items()):
        inst_inf = node_inf['institution']
        inst_set_dwpi = str2set(inst_inf['inst-dwpi'])
        inst_set_original = str2set(inst_inf['inst-original'])
        author_inf = node_inf['author']
        author_set_dwpi = str2set(author_inf['author-dwpi'])
        author_set_original = str2set(author_inf['author-original'])

        # inst - author
        inst_list_dwpi = inst_set_dwpi - author_set_dwpi
        inst_list_original = inst_set_original - author_set_original

        for inst in list(inst_list_dwpi | inst_list_original):
            inst2patent[inst].add(patent2id[patent])

    inst2patent = {inst: list(patent_set) for inst, patent_set in inst2patent.items()}

    with open('../data/output/inst2patent.json', 'w', encoding='UTF-8') as file:
        json.dump(inst2patent, file)


def get_inst2vec_mean():
    with open('../data/output/inst2patent.json', 'r', encoding='UTF-8') as file:
        inst2patent = json.load(file)

    inst2index = {inst: index for index, inst in enumerate(inst2patent.keys())}
    inst2vec = np.zeros((len(inst2patent), 1024))
    for node in ['node1_feature', 'node2_feature', 'node3_feature', 'node4_feature']:
        for length in [64, 128, 256, 512]:
            with open(f'../data/output/{node}/id_{length}.json', 'r', encoding='UTF-8') as file:
                patent_id_list = json.load(file)
            feature = np.load(f'../data/output/{node}/patent_feature_{length}.npy')
            patent2feature = {patent: feature for patent, feature in zip(patent_id_list, feature)}

            for inst, patent_list in tqdm(inst2patent.items(), desc=f'{node}_{length}'):
                for patent in patent_list:
                    try:
                        inst2vec[inst2index[inst]] += patent2feature[patent]
                    except KeyError:
                        continue

    # mean
    index2num = {inst2index[inst]: len(patent_list) for inst, patent_list in inst2patent.items()}
    for index, num in index2num.items():
        inst2vec[index] /= num

    # save
    np.save('../data/output/inst2vec_mean.npy', inst2vec)


def get_inst2vec_max():
    with open('../data/output/inst2patent.json', 'r', encoding='UTF-8') as file:
        inst2patent = json.load(file)

    inst2index = {inst: index for index, inst in enumerate(inst2patent.keys())}
    inst2vec = np.zeros((len(inst2patent), 1024))
    for node in ['node1_feature', 'node2_feature', 'node3_feature', 'node4_feature']:
        for length in [64, 128, 256, 512]:
            with open(f'../data/output/{node}/id_{length}.json', 'r', encoding='UTF-8') as file:
                patent_id_list = json.load(file)
            feature = np.load(f'../data/output/{node}/patent_feature_{length}.npy')
            patent2feature = {patent: feature for patent, feature in zip(patent_id_list, feature)}

            for inst, patent_list in tqdm(inst2patent.items(), desc=f'{node}_{length}'):
                for patent in patent_list:
                    try:
                        inst2vec[inst2index[inst]] = np.maximum(inst2vec[inst2index[inst]], patent2feature[patent])
                    except KeyError:
                        continue

    # save
    np.save('../data/output/inst2vec_max.npy', inst2vec)


if __name__ == '__main__':
    get_inst2vec_mean()
