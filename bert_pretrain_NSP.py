#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 上午11:21
# @Author  : liu yuhan
# @FileName: bert_pretrain_NSP.py
# @Software: PyCharm


import torch
import random
from transformers import BertConfig, DataCollatorForLanguageModeling, BertForNextSentencePrediction
from transformers import BertTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm


def load_data(filename):
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


def get_random_weight(label, num_label):
    random_weight = [1] * num_label
    random_weight[label] = 0
    return random_weight


class DataMaker:
    def __init__(self, tokenizer, label_path, max_len, max_label_len):
        self.label_path = label_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_label_len = max_label_len
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

    def label_trans(self, label):
        """
        这个地方可能会有多种写法，解决label长度不同的问题
        :param label:
        :return:
        """
        return self.tokenizer.encode(label)[1:]

    def get_label(self):
        """
        将所有label载入进来
        :return:
        """
        f = open(self.label_path, 'r', encoding='UTF-8')
        class_id_list = []
        for label in f:
            label = label.split()[1]
            class_id_list.append(self.label_trans(label))
        return class_id_list

    def data_trans(self, file_path):
        """
        完成编码
        :param file_path:
        :return:
        """

        # 准备类的转码结果
        class_id_list = self.get_label()
        num_label = len(class_id_list)
        # 载入数据
        data = load_data(file_path)
        source_list, target_list = [], []

        for text, label in data:
            # token
            token_ids = tokenizer.encode(text, truncation=True, max_length=self.max_len)
            token_ids_pos = token_ids + class_id_list[label]
            token_ids_neg = token_ids + random.choices(class_id_list, weights=get_random_weight(label, num_label))[0]
            # 长度不满的补充到max_len+max_label_len
            if len(token_ids_pos) <= self.max_len + self.max_label_len:
                token_ids_pos += [0] * (self.max_len + self.max_label_len - len(token_ids_pos))
            if len(token_ids_neg) <= self.max_len + self.max_label_len:
                token_ids_neg += [0] * (self.max_len + self.max_label_len - len(token_ids_neg))

            source_list.append(token_ids_pos)
            target_list.append(1)
            source_list.append(token_ids_neg)
            target_list.append(0)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}


if __name__ == '__main__':
    # 相关参数
    label_path = 'input/label_MLM.txt'
    max_len = 128
    max_label_len = 30
    # model path
    bert_file = "chinese_wwm_ext_pytorch"

    config = BertConfig.from_pretrained(bert_file)
    tokenizer = BertTokenizer.from_pretrained(bert_file)

    data_maker = DataMaker(tokenizer, label_path, max_len, max_label_len)
    data_train = data_maker.data_trans("input/train.txt")
    data_train = Dataset.from_dict(data_train)
    data_train.shuffle()

    data_eval = data_maker.data_trans("input/valid.txt")
    data_eval = Dataset.from_dict(data_eval)
    data_eval.shuffle()

    model = BertForNextSentencePrediction.from_pretrained(bert_file)
    print('No of parameters: ', model.num_parameters())

    training_args = TrainingArguments(
        output_dir='./outputs/',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        save_steps=10000,
        do_train=True,
        do_eval=True,
        prediction_loss_only=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_eval
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
