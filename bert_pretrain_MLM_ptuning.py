#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 上午11:21
# @Author  : liu yuhan
# @FileName: bert_pretrain_MLM.py
# @Software: PyCharm


import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertTokenizer, LineByLineTextDataset, TrainingArguments, Trainer


class DataMaker:
    def __init__(self, tokenizer, label_path):
        self.label_path = label_path
        self.tokenizer = tokenizer

    def label_trans(self, label):
        """
        这个地方可能会有多种写法，解决label长度不同的问题
        :param label:
        :return:
        """
        label_id = self.tokenizer.encode(label)[0][1:-1]
        if len(label_id) < 10:
            add_length = 10 - len(label_id)
            if add_length % 2 == 0:
                label_id = [77] * int(add_length / 2) + label_id + [77] * int(add_length / 2)
            else:
                label_id = [77] * int((add_length - 1) / 2) + label_id + [77] * int((add_length + 1) / 2)
        return label_id

    def get_label(self):
        f = open(self.label_path, 'r', encoding='UTF-8')
        class_id_list = []
        for label in f:
            label = label.split()[1]
            class_id_list.append(self.label_trans(label))
        return class_id_list


if __name__ == '__main__':
    bert_file = "bert-base-uncased"
    # bert_file = "prajjwal1/bert-tiny"
    # bert_file = "ckiplab/albert-base-chinese"
    # bert_file = "clue/albert_chinese_medium"
    # bert_file = "chinese_roberta_wwm_ext_L-12_H-768_A-12"

    config = BertConfig.from_pretrained(bert_file)
    tokenizer = BertTokenizer.from_pretrained(bert_file)
    model = BertForMaskedLM.from_pretrained(bert_file)
    print('No of parameters: ', model.num_parameters())

    # 相关参数
    label_path = 'input/label.txt'

    data_maker = DataMaker(tokenizer, label_path)
    class_id_list = data_maker.get_label()
    print(class_id_list)

# dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='input.txt', block_size=512)
# print(dataset[0])
#
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# print('No. of lines: ', len(dataset))
#
# training_args = TrainingArguments(
#     output_dir='./outputs/',
#     overwrite_output_dir=True,
#     num_train_epochs=50,
#     per_device_train_batch_size=4,
#     save_steps=100000,
# )
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )
# trainer.train()
# trainer.save_model('./outputs/')
#
# # 这里考虑把p—tuning的思想放进来
# desc = ['[unused%s]' % i for i in range(1, 41)]
# desc = desc[:20] + ['[mask]'] * 10 + desc[20:]
# desc_ids = [tokenizer.token_to_id(t) for t in desc]
