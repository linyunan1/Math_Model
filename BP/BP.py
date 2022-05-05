import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.client import device_lib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Lables = {0: 'crack',
          1: 'inclusion',
          2: 'pitted'}


def print_history(history):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy&loss')
    # plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
    # plt.legend(['Train_loss', 'Val_loss'])
    plt.show()


def BP(lr):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(7,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.build(input_shape=(None, 7))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # 这里是数据和标签，数据的shape是7，标签one-hot，shape是3

    data = []
    # 读取数据
    raw = pd.read_csv('hu.csv')
    raw_data = raw.values
    raw_feature = raw_data[:, 0:7]

    # 数据归一化
    scaler = MinMaxScaler()
    scaler.fit(raw_feature)
    scaler.data_max_
    raw_feature = scaler.transform(raw_feature)

    # 将最后一列的缺陷类别转成one-hot编码形式
    x = []
    y = []
    for i in range(len(raw_feature)):
        x.append(list(raw_feature[i]))
        if raw_data[i][7] == 'crack':
            y.append([1, 0, 0])
        elif raw_data[i][7] == 'inclusion':
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])


    # 随机打乱数据
    x = np.array(x)
    y = np.array(y)
    permutation = np.random.permutation(len(x))
    x = x[permutation]
    y = y[permutation]
    # 选取打乱后的前240个数据作为训练数据和验证数据
    train_data = x[0:240]
    train_label = y[0:240]
    # 选取打乱后的后60个作为测试数据
    test_data = x[240:]
    test_label = y[240:]


    lr = 0.001  #学习率初值，可动态下降
    bp_model = BP(lr=lr)

    bp_model.summary()

    #学子率动态衰减
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.8, patience=5,
                                                      min_lr=0.5e-6)

    # 早停法，保存训练中的最优参数
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                          verbose=0, patience=100, min_delta=0.0001,
                                          restore_best_weights='True')

    history = bp_model.fit(train_data, train_label, batch_size=10, epochs=10000, verbose=1,
                           callbacks=[lr_reducer, es], validation_split=0.25, shuffle=False)

    # 画出四条曲线（训练集和验证集的loss和accuracy缺陷）
    print_history(history)
    # 训练好的模型，在测试集上的准确率
    print('loss, acc:', bp_model.evaluate(test_data, test_label, batch_size=10, verbose=0))
