# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/26 11:44
@Auth ： hcb
@File ：data_loader.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import os
import jieba
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class BaseData():

    __doc__ = "生产训练集和测试集数据迭代器"

    def __init__(self, args):
        self.base_dir = os.path.join(os.path.dirname(__file__), "raw_data")
        self.raw_data_path = os.path.join(self.base_dir, "toutiao_prepared.txt")
        # self.prepared_data_path = os.path.join(self.base_dir, "toutiao_prepared.txt")
        self.use_char = True

        self.word2id = {}
        self.id2word = {}
        self.label2id = {}
        self.id2label = {}

        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.enforced_sorted = True
        self.train_dataloader = None
        self.test_dataloader = None
        self.trainset_idx, self.testset_idx = self.obtain_dataset()  # 主程序
        self.obtain_dataloader()

    def obtain_dataset(self):
        """
        处理数据
        :return: 训练集和测试集的索引矩阵
        """
        with open(self.raw_data_path, "r", encoding="utf8") as reader:
            all_lines = reader.readlines()
        # 处理成样本和标签
        dataset = []
        for line in tqdm(all_lines, desc="处理数据"):
            sample_text, sample_label = self.clean_data(line)
            dataset.append((sample_text, sample_label))
        # 划分训练集和测试集
        train_set, test_set = train_test_split(dataset, test_size=0.5, random_state=10)  # 选总数据一半作为数据集
        train_set, test_set = train_test_split(train_set, test_size=0.15, random_state=10)
        # 根据训练集构建vocab
        self.build_vocab(train_set)
        trainset_idx = self.trans_data(train_set)
        testset_idx = self.trans_data(test_set)

        return trainset_idx, testset_idx

    def obtain_dataloader(self):
        """
        根据索引矩阵生产数据的迭代器
        :return:
        train_dataloader： 训练集迭代器
        test_dataloader： 测试集迭代器
        """
        train_dataset = MyData(self.trainset_idx)
        test_dataset = MyData(self.testset_idx)
        # droplast设为True 防止最后一个batch数量不足
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True,
                                           collate_fn=self.coll_batch)
        self.test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True,
                                          collate_fn=self.coll_batch)

    def clean_data(self, line):
        """
        分词并清洗数据
        :param line:
        :return:
        sample_text:  ["刘亦菲", "漂亮"，“美女”]
        label: "娱乐"
        """
        text, label = line.split("\t")[0], line.split("\t")[1]
        if self.use_char:
            sample_text = list(text)
        else:
            sample_text = jieba.lcut(text)
        return sample_text, label

    def build_vocab(self, data_info):
        """
        构建词汇表字典
        :param data_info:
        :return:
        """
        tokens = []
        labels = set()
        for text, label in data_info:
            tokens.extend(text)
            labels.add(label)

        tokens = sorted(set(tokens))
        tokens.insert(0, "<pad>")
        tokens.insert(1, "<unk>")
        labels = sorted(labels)

        self.word2id = {word:idx for idx, word in enumerate(tokens)}
        self.id2word = {idx:word for idx, word in enumerate(tokens)}
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for idx, label in enumerate(labels)}

    def trans_data(self, data_set):
        """
        根据词汇表字典将文本转成索引矩阵
        :param data_set:
        :return:
        """
        data_set_idx = []
        for text, label in data_set:
            text_idx = [self.word2id[word] if word in self.word2id else self.word2id["<unk>"] for word in text]
            label_idx = self.label2id[label]
            data_set_idx.append((text_idx, label_idx))
        return data_set_idx

    def coll_batch(self, batch):
        """
        对每个batch进行处理
        :param batch:
        :return:
        """
        # 每条样本的长度
        current_len = [len(data[0]) for data in batch]
        if self.enforced_sorted:
            index_sort = list(reversed(np.argsort(current_len)))
            batch = [batch[index] for index in index_sort]
            current_len = [min(current_len[index], self.max_seq_len) for index in index_sort]
        # 对每个batch进行padding

        max_length = min(max(current_len), self.max_seq_len)
        batch_x = []
        batch_y = []
        for item in batch:
            sample = item[0]
            if len(sample) > max_length:
                sample = sample[0:max_length]
            else:
                sample.extend([0] * (max_length-len(sample)))
            batch_x.append(sample)
            batch_y.append([item[1]])
        return {"sample": torch.tensor(batch_x), "label": torch.tensor(batch_y), "length": current_len}


class MyData(Dataset):
    def __init__(self, data_set):
        self.data = data_set

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#     data_obj = BaseData(args=1)

