#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 下午5:39
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm

"""
数据处理，处理完的数据
"""
import json
import os
import re
import jieba
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def sentence_clean(s):
    """
    句子的处理
    :param s:
    :return:
    """
    # 正则去除括号中的内容
    pattern = re.compile(r'[\(].+[\)]')
    s = re.sub(pattern, '', s)
    pattern = re.compile(r'[（].+[）]')
    s = re.sub(pattern, '', s)
    # 开头的部分
    # 大写数字
    pattern_2 = re.compile(r'[零一二三四五六七八九○０Ｏ]+、')
    s = re.sub(pattern_2, '', s)
    pattern_2 = re.compile(r'[零一二三四五六七八九○０Ｏ]+是')
    s = re.sub(pattern_2, '', s)
    # 小写数字
    pattern_2 = re.compile(r'[0-9]+[、．.]')
    s = re.sub(pattern_2, '', s)
    pattern_2 = re.compile(r'^[0-9]+')
    s = re.sub(pattern_2, '', s)
    # 条
    pattern_2 = re.compile(r'[第].+[条章]')
    s = re.sub(pattern_2, '', s)
    # 年月日
    pattern_3 = re.compile(r'[零一二三四五六七八九○０Ｏ]+年[一二三四五六七八九十]+月[一二三四五六七八九十]+日')
    s = re.sub(pattern_3, '', s)
    pattern_3 = re.compile(r'[0-9０]+年[0-9]+月[0-9０]+日')
    s = re.sub(pattern_3, '', s)
    # 去除空格
    s = ' '.join(s.split())
    if len(s) <= 10:
        s = ''
    return s


def get_keyword_seq(kw_list, s):
    """
    通过每一句话获取节点的序列
    :param s:
    :return:
    """
    node_list = []
    node_trans_list = []

    for node in kw_list:
        if node not in s:
            continue
        else:
            node_list.append(node)
            node_length = len(node)
            bit = 0
            for i in range(s.count(node)):
                bit = s.find(node, bit)
                node_trans = [k for k in range(bit, bit + node_length)]
                node_trans_list.append(node_trans)
                bit += 1

    return node_list, node_trans_list


def overlap_check(node_list):
    """
    判断两个节点是否重叠，不重叠返回1，重叠返回0
    :param node2:
    :param node1:
    :return:
    """
    index_list = []
    for node in node_list:
        index_list += node

    if len(set(index_list)) == len(index_list):
        # 无重叠
        return True
    else:
        return False


def deal(file_path, keywords_path, output_path_clean):
    """

    :param file_path:
    :param keywords_path:
    :param output_path_clean:
    :return:
    """
    make_path(output_path_clean)

    keywords_list = open(keywords_path, 'r', encoding='GB2312').readlines()
    keywords_list = sorted([''.join(keywords.split()) for keywords in keywords_list],
                           key=lambda x: len(x),
                           reverse=True)
    print(keywords_list)
    results = []
    count = 0
    count_kw = 0

    file_list = os.listdir(file_path)  # 二级目录列表
    for file in file_list:
        doc_file_path = os.path.join(file_path, file)
        # 文件处理
        doc_file_list = os.listdir(doc_file_path)
        for doc_file in tqdm(doc_file_list):
            # 文本拆分成句子
            s_list = sentence_cut(os.path.join(doc_file_path, doc_file))
            for s in s_list:
                s_clean = sentence_clean(s)
                # 重新梳理一下
                if not s_clean:
                    # 如果s_clean是空的直接跳过
                    continue
                count += 1
                # 获取关键词
                kw_list, kw_trans_list = get_keyword_seq(keywords_list, s_clean)
                if kw_list and overlap_check(kw_trans_list):
                    count_kw += 1
                results.append({'text': s_clean,
                                'kw': kw_list})

    with open('data_clean.json', 'w', encoding="UTF-8") as file:
        json.dump(results, file)
    print(count, count_kw, count_kw / count)


def sentence_cut(doc_path):
    """
    将文本拆分成句子，引入长句拆分，希望效果会好一些
    :param doc_path:
    :return:
    """
    s_list = []
    with open(doc_path, 'r', encoding='UTF-8') as file:
        # 每一行去除前两行(政策文本的特殊性)
        next(file)
        next(file)
        for line in file:
            s_list += re.split(r"[；。？！：]", line)
    return s_list


if __name__ == '__main__':
    input_path = '../../data-origin/input/政策文本检索'
    keywords_path = '../../词典构建相关工作/术语筛选.txt'
    output_path_clean = '../data/input'
    deal(input_path, keywords_path, output_path_clean)
