#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 上午11:21
# @Author  : liu yuhan
# @FileName: bert_pretrain_MLM.py
# @Software: PyCharm


import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertTokenizer, LineByLineTextDataset, TrainingArguments, Trainer

# bert_file = "bert-base-uncased"
# bert_file = "prajjwal1/bert-tiny"
bert_file = "ckiplab/albert-base-chinese"
# bert_file = "clue/albert_chinese_medium"

config = BertConfig.from_pretrained(bert_file)
tokenizer = BertTokenizer.from_pretrained(bert_file)
model = BertForMaskedLM.from_pretrained(bert_file)
print('No of parameters: ', model.num_parameters())

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='input.txt', block_size=512)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
print('No. of lines: ', len(dataset))

training_args = TrainingArguments(
    output_dir='./outputs/',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    save_steps=100000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('./outputs/')
