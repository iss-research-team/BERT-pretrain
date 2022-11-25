#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 上午11:21
# @Author  : liu yuhan
# @FileName: bert_pretrain_MLM.py
# @Software: PyCharm

import json
import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm


def load_data(filename):
    """
    数据的输入
    :param filename:
    :return:
    """
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    return data


class DataMaker:
    def __init__(self, tokenizer, max_len, if_kw=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        self.if_kw = if_kw

    def data_trans(self, file_path):
        """
        数据处理
        source是原始的ids
        target是mask的ids
            target在有术语词典和没有术语词典的写法不同
        :param file_path:
        :return:
        """
        # 载入数
        source_list, target_list = [], []
        data = load_data(file_path)

        print('data processing...')
        for text_inf in tqdm(data):
            text = text_inf['text']
            kw_list = text_inf['kw']
            # token
            token_ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_len, padding='max_length')
            # mask
            if self.if_kw and kw_list:
                # 有字典的情景 且 关键词非空的情景
                token_ids_masked = self.kw_mask(token_ids, kw_list)
            else:
                # 不考虑字典 或 关键词为空
                token_ids_masked = self.data_collator([token_ids])['input_ids'][0].tolist()

            source_list.append(token_ids_masked)
            target_list.append(token_ids)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}

    def kw_mask(self, token_ids, kw_list):
        """
        在有术语词典的情景下进行mask
        :param token_ids:
        :param kw_list:
        :return:
        """
        # 字典非空的情景
        token_ids_masked = token_ids.copy()
        token_ids_trans = ' ' + ' '.join([str(ids) for ids in token_ids]) + ' '
        # 考虑有关键词的情景
        for kw in kw_list:
            kw_ids = self.tokenizer.encode(kw)[1:-1]
            kw_ids_trans = ' ' + ' '.join([str(ids) for ids in kw_ids]) + ' '
            bit = 0
            for _ in range(token_ids_trans.count(kw_ids_trans)):
                # 找到起始的空格的位置
                bit = token_ids_trans.find(kw_ids_trans, bit)
                start_bit = token_ids_trans[:bit].count(' ')
                for i in range(start_bit, start_bit + len(kw_ids)):
                    token_ids_masked[i] = self.tokenizer.mask_token_id
                bit += 1
        return token_ids_masked


if __name__ == '__main__':
    # 相关参数
    max_len = 128
    model_path = "/home/logan-02/bert-model/chinese_roberta_wwm_ext_pytorch"

    tokenizer = BertTokenizer.from_pretrained(model_path)

    data_maker = DataMaker(tokenizer, max_len, if_kw=True)
    data_train = data_maker.data_trans("data_clean.json")
    data_train = Dataset.from_dict(data_train)

    model = BertForMaskedLM.from_pretrained(model_path)
    print('No of parameters: ', model.num_parameters())

    training_args = TrainingArguments(
        output_dir='../data/outputs/',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        save_steps=10000,
        do_train=True,
        do_eval=False,
        prediction_loss_only=False,
        # eval_accumulation_steps=10000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        # eval_dataset=data_eval
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
    # # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate()
