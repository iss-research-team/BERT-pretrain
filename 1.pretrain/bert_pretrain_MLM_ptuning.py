#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 上午11:21
# @Author  : liu yuhan
# @FileName: bert_pretrain_MLM.py
# @Software: PyCharm


import torch
from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
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


class DataMaker:
    def __init__(self, tokenizer, label_path, max_len, prompt_size):
        self.label_path = label_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_size = prompt_size
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

    def label_trans(self, label):
        """
        这个地方可能会有多种写法，解决label长度不同的问题
        :param label:
        :return:
        """
        label_id = self.tokenizer.encode(label)[1:-1]
        if len(label_id) < 10:
            add_length = 10 - len(label_id)
            if add_length % 2 == 0:
                label_id = [77] * int(add_length / 2) + label_id + [77] * int(add_length / 2)
            else:
                label_id = [77] * int((add_length - 1) / 2) + label_id + [77] * int((add_length + 1) / 2)
        return label_id

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
        完成编码，转换成p-tuning的形式
        :param file_path:
        :return:
        """

        desc_1 = [i for i in range(1, 1 + self.prompt_size)]
        desc_2 = [i for i in range(self.prompt_size + 1, self.prompt_size * 2 + 1)]
        # 准备类的转码结果
        class_id_list = self.get_label()
        # 载入数据
        data = load_data(file_path)
        source_list, target_list = [], []

        for text, label in data:
            # token
            token_ids = tokenizer.encode(text, truncation=True, max_length=self.max_len)
            # 长度不满的补充到128
            if len(token_ids) <= self.max_len:
                token_ids = token_ids + [0] * (self.max_len - len(token_ids))
            token_ids_masked = self.data_collator([token_ids])['input_ids'][0].tolist()

            source_ids = token_ids_masked[:1] + desc_1 + [103] * 10 + desc_2 + token_ids_masked[1:]
            label_ids = token_ids[:1] + desc_1 + class_id_list[label] + desc_2 + token_ids[1:]

            source_list.append(source_ids)
            target_list.append(label_ids)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}


if __name__ == '__main__':
    # 相关参数
    label_path = '../data/input/label_MLM.txt'
    max_len = 128
    prompt_size = 15
    # model path
    bert_file = "albert-base-v2"

    config = AlbertConfig.from_pretrained(bert_file)
    tokenizer = AlbertTokenizer.from_pretrained(bert_file, config)

    data_maker = DataMaker(tokenizer, label_path, max_len, prompt_size)
    data_train = data_maker.data_trans("input/train.txt")
    data_train = Dataset.from_dict(data_train)

    data_eval = data_maker.data_trans("input/valid.txt")
    data_eval = Dataset.from_dict(data_eval)

    model = AlbertForMaskedLM.from_pretrained(bert_file, config)
    print('No of parameters: ', model.num_parameters())

    training_args = TrainingArguments(
        output_dir='../data/outputs/',
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
