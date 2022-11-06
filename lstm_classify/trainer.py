# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/26 14:38
@Auth ： hcb
@File ：trainer.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import argparse
import os
from data_loader import BaseData
import lstm_model
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report


def train(args):
    # 定义模型优化器 损失函数等
    model = lstm_model.LSTMClassifier(args)
    if args.use_cuda:
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss()

    train_dataloader = args.dataloader.train_dataloader
    test_dataloader = args.dataloader.test_dataloader
    model.train()
    for epoch in tqdm(range(args.epoch_num)):
        print(f"epoch {epoch}...")
        for train_info in tqdm(train_dataloader):
            optimizer.zero_grad()
            # model.hidden = model.init_hidden()
            data = train_info["sample"]
            label = train_info["label"]
            length = train_info["length"]
            if args.use_cuda:
                data = data.cuda()
                label = label.cuda()
            # print("data_size", data.size())
            predict_label = model(data)
            label = label.view(args.batch_size,)  # [30, 1] --> [30]
            loss_batch = loss_function(predict_label, label)
            loss_batch.backward()
            # print("loss", loss_batch)

            optimizer.step()
        print(f"evaluation...epoch_{epoch}:")
        true_label, pred_label = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for test_info in test_dataloader:
                data = test_info["sample"]
                label = test_info["label"]
                length = test_info["length"]
                # 保存真实标签
                label_list = label.view(1, -1).squeeze().numpy().tolist()
                true_label.extend(label_list)
                if args.use_cuda:
                    data = data.cuda()
                    label = label.cuda()

                predict_label = model(data)
                predict_label_list = torch.argmax(predict_label, dim=1).cpu().numpy().tolist()
                pred_label.extend(predict_label_list)

                label = label.view(args.batch_size, )
                loss_sum += loss_function(predict_label, label)
        print(classification_report(true_label, pred_label))
        print(f"epoch:{epoch} test data loss: {loss_sum}.")


def main():
    args = argparse.ArgumentParser()

    args.add_argument("--model", default="lstm", choices=["textcnn", "lstm"])
    args.add_argument("--batch_size", type=int, default=50)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--max_seq_len", type=int, default=80)
    args.add_argument("--enforced_sorted", type=bool, default=True)
    args.add_argument("--embedding_dim", type=int, default=128)
    args.add_argument("--hidden_dim", type=int, default=128)
    args.add_argument("--num_layer", type=int, default=2)
    args.add_argument("--epoch_num", type=int, default=2)
    args.add_argument("--use_cuda", type=bool, default=False)

    args = args.parse_args()

    data_load = BaseData(args)
    setattr(args, "dataloader", data_load)
    setattr(args, "vocab_num", len(data_load.word2id))
    setattr(args, "class_num", len(data_load.label2id))

    train(args)


if __name__ == '__main__':
    main()
