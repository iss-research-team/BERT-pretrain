#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 下午5:39
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm

import os
import random


def get_label():
    """
    把label写进去
    :return:
    """
    file_path = '../data'
    file_list = os.listdir(file_path)

    data_label = []

    for label, file in enumerate(file_list):
        f = open(os.path.join(file_path, file), 'r', encoding="UTF-8")
        for each_line in f:
            data_label.append([each_line.strip(), str(label)])

    random.shuffle(data_label)

    f_w_1 = open('../data/input/train.txt', 'w', encoding='UTF-8')
    f_w_2 = open('../data/input/test.txt', 'w', encoding='UTF-8')
    f_w_3 = open('../data/input/valid.txt', 'w', encoding='UTF-8')

    cut_1 = int(0.6 * len(data_label))
    cut_2 = int(0.8 * len(data_label))

    # 写入
    # 训练集
    for each_line in data_label[:cut_1]:
        f_w_1.write('\t'.join(each_line) + '\n')
    for each_line in data_label[cut_1:cut_2]:
        f_w_2.write('\t'.join(each_line) + '\n')
    for each_line in data_label[cut_2:]:
        f_w_3.write('\t'.join(each_line) + '\n')


def get_label_temper():
    """
    把label写进去
    :return:
    """
    file_path = '../data'
    file_list = os.listdir(file_path)

    data_label = []

    for label, file in enumerate(file_list):
        f = open(os.path.join(file_path, file), 'r', encoding="UTF-8")
        for each_line in f:
            data_label.append([each_line.strip(), str(label)])

    random.shuffle(data_label)

    f_w_1 = open('input.txt', 'w', encoding='UTF-8')

    # 写入
    for each_line in data_label:
        f_w_1.write(each_line[0] + '\n')


if __name__ == '__main__':
    get_label_temper()
