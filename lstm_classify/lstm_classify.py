import torch
import numpy
import tensorflow as tf
from ast import literal_eval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def main():
    # 训练并保存模型
    # trainModel()

    # 测试数据集
    x_test, y_test = getData('raw_data/key_points.txt')
    result = predict(x_test)
    for i in range(len(result)):
        print(result[i])
        print(y_test[i])

def trainModel():
    x_train, y_train = getData('raw_data/key_points.txt')
    # print(x_train)
    # print(y_train)

    # 创建模型
    # 循环神经网络的数据输入必须是3维数据
    cell_size = len(x_train)
    model = Sequential([
        LSTM(units=cell_size, input_shape=(len(x_train[0]), len(x_train[0][0])), return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    adam = Adam(lr=1e-3)
    # 二分类问题
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # 训练
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    # 评估
    score = model.evaluate(x_train, y_train, batch_size=32)
    # 保存模型文件
    model.save('model/lstm_model.h5')

def predict(x_data):
    model = load_model('model/lstm_model.h5')
    result = model.predict(x_data)
    return result

def getData(path):
    x_train = [];
    y_train = [];
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            my_list = literal_eval(line)
            x_train.append(my_list[0])
            y_train.append(my_list[1])
    return x_train, y_train

if __name__ == '__main__':
    main()
