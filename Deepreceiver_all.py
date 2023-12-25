from __future__ import print_function

import os.path
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
import densenet_signal
import numpy as np
import sklearn.metrics as metrics
from getData_all import load_data_all  # 数据集中提取数据
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers import Input


batch_size = 36     # 一次训练的样本数
nb_epoch = 10       # 训练次数
from keras.models import load_model
from keras.models import Model

nb_classes = 2      # 分类数，代表 1 0 两类
depth = 40          # DenseNet中的数量或层数
nb_dense_block = 4  # 添加到末尾的dense_block数
growth_rate = 12    # 每个dense_block添加的过滤器数量
nb_filter = -1      # 初始过滤器个数。-1表示过滤器初始个数为2 * growth_rate
dropout_rate = 0.0  # 0.0 for data augmentation

base_model = densenet_signal.DenseNet(input_shape=(448, 2, 1), classes=nb_classes, depth=depth,
                                      nb_dense_block=nb_dense_block,
                                      growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                                      weights=None)     # 构建基础的DenseNet网络

input = base_model.input
output = base_model(input)

predict1 = Dense(nb_classes, activation="softmax", name="out1")(output)     # 每一层都加一个全连接层
predict2 = Dense(nb_classes, activation="softmax", name="out2")(output)     # 预测输出32个比特位
predict3 = Dense(nb_classes, activation="softmax", name="out3")(output)
predict4 = Dense(nb_classes, activation="softmax", name="out4")(output)
predict5 = Dense(nb_classes, activation="softmax", name="out5")(output)
predict6 = Dense(nb_classes, activation="softmax", name="out6")(output)
predict7 = Dense(nb_classes, activation="softmax", name="out7")(output)
predict8 = Dense(nb_classes, activation="softmax", name="out8")(output)
predict9 = Dense(nb_classes, activation="softmax", name="out9")(output)
predict10 = Dense(nb_classes, activation="softmax", name="out10")(output)
predict11 = Dense(nb_classes, activation="softmax", name="out11")(output)
predict12 = Dense(nb_classes, activation="softmax", name="out12")(output)
predict13 = Dense(nb_classes, activation="softmax", name="out13")(output)
predict14 = Dense(nb_classes, activation="softmax", name="out14")(output)
predict15 = Dense(nb_classes, activation="softmax", name="out15")(output)
predict16 = Dense(nb_classes, activation="softmax", name="out16")(output)
predict17 = Dense(nb_classes, activation="softmax", name="out17")(output)
predict18 = Dense(nb_classes, activation="softmax", name="out18")(output)
predict19 = Dense(nb_classes, activation="softmax", name="out19")(output)
predict20 = Dense(nb_classes, activation="softmax", name="out20")(output)
predict21 = Dense(nb_classes, activation="softmax", name="out21")(output)
predict22 = Dense(nb_classes, activation="softmax", name="out22")(output)
predict23 = Dense(nb_classes, activation="softmax", name="out23")(output)
predict24 = Dense(nb_classes, activation="softmax", name="out24")(output)
predict25 = Dense(nb_classes, activation="softmax", name="out25")(output)
predict26 = Dense(nb_classes, activation="softmax", name="out26")(output)
predict27 = Dense(nb_classes, activation="softmax", name="out27")(output)
predict28 = Dense(nb_classes, activation="softmax", name="out28")(output)
predict29 = Dense(nb_classes, activation="softmax", name="out29")(output)
predict30 = Dense(nb_classes, activation="softmax", name="out30")(output)
predict31 = Dense(nb_classes, activation="softmax", name="out31")(output)
predict32 = Dense(nb_classes, activation="softmax", name="out32")(output)

model = Model(inputs=input,
              outputs=[predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9,
                       predict10,
                       predict11, predict12, predict13, predict14, predict15, predict16, predict17, predict18,
                       predict19, predict20,
                       predict21, predict22, predict23, predict24, predict25, predict26, predict27, predict28,
                       predict29, predict30,
                       predict31, predict32])

print("Model created")

model.summary()  # 输出层的结构
optimizer = Adam(lr=1e-4)  # 使用Adam代替SGD加速训练；Adam收敛速度快，泛化效果差，稳定性高；
model.compile(loss={
    'out1': 'binary_crossentropy',  # 二进制交叉熵
    'out2': 'binary_crossentropy',
    'out3': 'binary_crossentropy',
    'out4': 'binary_crossentropy',
    'out5': 'binary_crossentropy',
    'out6': 'binary_crossentropy',

    'out7': 'binary_crossentropy',
    'out8': 'binary_crossentropy',

    'out9': 'binary_crossentropy',
    'out10': 'binary_crossentropy',
    'out11': 'binary_crossentropy',
    'out12': 'binary_crossentropy',
    'out13': 'binary_crossentropy',
    'out14': 'binary_crossentropy',
    'out15': 'binary_crossentropy',
    'out16': 'binary_crossentropy',

    'out17': 'binary_crossentropy',
    'out18': 'binary_crossentropy',

    'out19': 'binary_crossentropy',
    'out20': 'binary_crossentropy',
    'out21': 'binary_crossentropy',
    'out22': 'binary_crossentropy',
    'out23': 'binary_crossentropy',
    'out24': 'binary_crossentropy',
    'out25': 'binary_crossentropy',
    'out26': 'binary_crossentropy',

    'out27': 'binary_crossentropy',
    'out28': 'binary_crossentropy',

    'out29': 'binary_crossentropy',
    'out30': 'binary_crossentropy',
    'out31': 'binary_crossentropy',
    'out32': 'binary_crossentropy',
},  # 对数损失函数
    optimizer=optimizer,  # 优化器
    metrics=["accuracy"])  # 模型训练指标
print("Finished compiling")
print("Building model...")  # 模型建立完成
import argparse
import sys

testX, trainX, Y_test, Y_Train, snrTest, snrTrain = load_data_all()

trainX = np.reshape(trainX, [20000 * 9, 448, 2, 1])
testX = np.reshape(testX, [10000 * 9, 448, 2, 1])

Y_Train = np.reshape(Y_Train, [20000 * 9, 32, 2])
Y_test = np.reshape(Y_test, [10000 * 9, 32, 2])

Y_Train = np.split(Y_Train, 32, axis=1)  # 按列分割，分成32个部分
Y_test = np.split(Y_test, 32, axis=1)  # 此时该三维数组会变成四维，分割出的二维数组被设为三维数组，因此总共四维

for i in range(32):  #
    Y_Train[i] = np.reshape(Y_Train[i], [20000 * 9, 2])  # python中，第一个[]就代表第一级数组，
    Y_test[i] = np.reshape(Y_test[i], [10000 * 9, 2])

'''截取小样本做程序测试'''
# trainx = trainX[:100]
# testx = testX[:10]
# trainy = []
# testy = []
# for i in range(32):  #
#     trainy.append(Y_Train[i][:100])
#     testy.append(Y_test[i][:10])
'''小样本测试结束'''

# Load model
weights_file = "./weights_all/deepreceiver_weights.h5"
try:
    # model = load_model(weights_file)
    model.load_weights(weights_file)
    print("模型加载成功")
except:
    print("模型加载失败")

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5)
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
# model_checkpoint = ModelCheckpoint(weights_file, save_best_only=True, save_weights_only=True, verbose=1)

callbacks = [lr_reducer, model_checkpoint]

# model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
#                     steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
#                     callbacks=callbacks,
#                     validation_data=(testX, Y_test),
#                     validation_steps=testX.shape[0] // batch_size, verbose=1)

# history1 = model.fit(trainX, Y_Train, batch_size=batch_size, epochs=1,
#                      callbacks=callbacks,
#                      validation_data=(testX, Y_test), verbose=1)    # 训练模型 模型直接更新到model 返回history记录训练过程
history1 = model.fit(trainX, Y_Train, batch_size=batch_size, epochs=nb_epoch, shuffle=True,
                     callbacks=callbacks,
                     validation_data=(testX, Y_test), verbose=1)    # 训练模型 模型直接更新到model 返回history记录训练过程
model.save_weights("./weights_all/deepreceiver_weights.h5")

