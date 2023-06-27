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
            feature = np.load(f'../data/output/{node}/patent_feature_{length}.npy').tolist()
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
            feature = np.load(f'../data/output/{node}/patent_feature_{length}.npy').tolist()
            patent2feature = {patent: feature for patent, feature in zip(patent_id_list, feature)}

            for inst, patent_list in tqdm(inst2patent.items(), desc=f'{node}_{length}'):
                for patent in patent_list:
                    try:
                        inst2vec[inst2index[inst]] = np.maximum(inst2vec[inst2index[inst]], patent2feature[patent])
                    except KeyError:
                        continue

    # save
    np.save('../data/output/inst2vec_max.npy', inst2vec)


def inst_vec_combine():
    node_list_sc_path = '../data/input/node_list_sc_0615.json'
    with open(node_list_sc_path, 'r', encoding='UTF-8') as file:
        node_list_sc = json.load(file)

    node_list_base_path = '../data/input/node_list_base_0615.json'
    with open(node_list_base_path, 'r', encoding='UTF-8') as file:
        node_list_base = json.load(file)

    node_list_invest_path = '../data/input/node_list_invest_0615.json'
    with open(node_list_invest_path, 'r', encoding='UTF-8') as file:
        node_list_invest = json.load(file)

    index2node_path = '../data/input/index2node_0615.json'
    with open(index2node_path, 'r', encoding='UTF-8') as file:
        index2node = json.load(file)
    inst2patent_path = '../data/output/inst2patent.json'
    with open(inst2patent_path, 'r', encoding='UTF-8') as file:
        inst2patent = json.load(file)

    inst2index = {inst: index for index, inst in enumerate(inst2patent.keys())}

    count_inst = 0
    count_patent = 0
    node_list_all = node_list_sc + node_list_base + node_list_invest
    node_list_all = list(set(node_list_all))
    print(len(node_list_all))
    print('num of node_list_sc:', len(node_list_sc))
    print('num of node_list_base:', len(node_list_base))
    print('num of node_list_invest:', len(node_list_invest))
    print('base | invest:', len(set(node_list_base) | set(node_list_invest)))
    print('(base | invest) & sc:', len((set(node_list_base) | set(node_list_invest)) & set(node_list_sc)))
    print('base & sc:', len(set(node_list_base) & set(node_list_sc)))

    node_list_sc_plus = (set(node_list_base) | set(node_list_invest)) & set(node_list_sc)

    node2inst = defaultdict(list)
    for node in node_list_all:
        if node not in node_list_sc_plus:
            continue
        inst_list = index2node[str(node)]
        for inst in inst_list:
            if inst in inst2index:
                node2inst[node].append(inst2index[inst])
                count_inst += 1
                count_patent += len(inst2patent[inst])

    print(count_inst)
    print(count_patent)
    print(len(node2inst))

    # count_patent_sc_plus = 0
    # for node in node2inst:
    #     if node in node_list_sc_plus:
    #         count_patent_sc_plus += 1
    # print('patent_node', count_patent_sc_plus)

    # save node2inst
    with open('../data/output/node2inst_tech_0621.json', 'w', encoding='UTF-8') as file:
        json.dump(node2inst, file, ensure_ascii=False, indent=4)

    # node2index_tmp
    node2index_tmp = {node: index for index, node in enumerate(node2inst.keys())}
    # mean
    node2vec_mean = np.zeros((len(node2inst), 1024))
    inst2feature_mean = np.load(f'../data/output/inst2vec_mean.npy')
    print(inst2feature_mean.shape)
    inst2feature_mean = {inst: feature for inst, feature in enumerate(inst2feature_mean)}

    for node, inst_list in tqdm(node2inst.items()):
        for inst in inst_list:
            node2vec_mean[node2index_tmp[node]] += inst2feature_mean[inst]
        node2vec_mean[node2index_tmp[node]] /= len(inst_list)
    # save
    np.save('../data/output/node2vec_mean_0621.npy', node2vec_mean)

    # max
    node2vec_max = np.zeros((len(node2inst), 1024))
    inst2feature_max = np.load(f'../data/output/inst2vec_max.npy')
    print(inst2feature_max.shape)
    inst2feature_max = {inst: feature for inst, feature in enumerate(inst2feature_max)}

    for node, inst_list in tqdm(node2inst.items()):
        for inst in inst_list:
            node2vec_max[node2index_tmp[node]] = np.maximum(node2vec_max[node2index_tmp[node]], inst2feature_max[inst])
    # save
    np.save('../data/output/node2vec_max_0621.npy', node2vec_max)


if __name__ == '__main__':
    # get_inst2vec_mean()
    inst_vec_combine()
