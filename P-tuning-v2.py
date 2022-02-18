#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/4 上午10:25
# @Author  : liu yuhan
# @FileName: P-tuning.py
# @Software: PyCharm


import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Embedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

maxlen = 128
batch_size = 32
config_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('input/train.txt')
valid_data = load_data('input/valid.txt')
test_data = load_data('input/test.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
# 针对政策文本，label众多

mask_idx = 21
desc = ['[unused%s]' % i for i in range(1, 41)]
desc = desc[:20] + ['[mask]'] * 10 + desc[20:]
desc_ids = [tokenizer.token_to_id(t) for t in desc]


def label_trans(label):
    """
    这个地方可能会有多种写法，解决label长度不同的问题
    :param label:
    :return:
    """
    label_id = tokenizer.encode(label)[0][1:-1]
    if len(label_id) < 10:
        add_length = 10 - len(label_id)
        if add_length % 2 == 0:
            label_id = [77] * int(add_length / 2) + label_id + [77] * int(add_length / 2)
        else:
            label_id = [77] * int((add_length - 1) / 2) + label_id + [77] * int((add_length + 1) / 2)
    return label_id


def get_label():
    f = open('input/label.txt', 'r', encoding='UTF-8')
    class_id_list = []
    for label in f:
        label = label.split()[1]
        class_id_list.append(label_trans(label))
    return class_id_list


class_id_list = get_label()


def random_masking(token_ids):
    """
    对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            # 植入模板
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids = token_ids[:1] + desc_ids + token_ids[1:]
            segment_ids = [0] * len(desc_ids) + segment_ids
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]

            source_ids[mask_idx] = source_ids[:mask_idx] + [tokenizer._token_mask_id] * 10 + source_ids[mask_idx:]
            target_ids[mask_idx] = target_ids[:mask_idx] + class_id_list[label] + target_ids[mask_idx:]

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                          batch_token_ids, batch_segment_ids, batch_output_ids
                      ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


class PtuningEmbedding(Embedding):
    """新定义Embedding层，只优化部分Token
    """

    def call(self, inputs, mode='embedding'):
        embeddings = self.embeddings
        embeddings_sg = K.stop_gradient(embeddings)
        mask = np.zeros((K.int_shape(embeddings)[0], 1))
        mask[1:51] += 1  # 只优化id为1～8的token
        self.embeddings = embeddings * mask + embeddings_sg * (1 - mask)
        outputs = super(PtuningEmbedding, self).call(inputs, mode)
        self.embeddings = embeddings
        return outputs


class PtuningBERT(BERT):
    """
    替换原来的Embedding
    """

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        if layer is Embedding:
            layer = PtuningEmbedding
        return super(PtuningBERT,
                     self).apply(inputs, layer, arguments, **kwargs)


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=PtuningBERT,
    with_mlm=True
)

for layer in model.layers:
    if layer.name != 'Embedding-Token':
        layer.trainable = False

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
output = keras.layers.Lambda(lambda x: x[:, :10])(model.output)
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(6e-4))
train_model.summary()

# 预测模型
model = keras.models.Model(model.inputs, output)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        print('y_pred', y_pred)
        # y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
        # y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_bert.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=1000,
        callbacks=[evaluator]
    )

else:
    model.load_weights('best_model_bert.weights')