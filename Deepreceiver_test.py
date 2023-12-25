#%%初始化
from __future__ import print_function

import os.path
import os
import time
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import matplotlib.pyplot as plt

batch_size = 36     # 一次训练的样本数
nb_epoch = 10       # 训练次数
from keras.models import load_model
from keras.models import Model

#%%构造网络
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

#%%加载网络参数
weights_file = "./weights_all/deepreceiver_weights.h5"
# model = load_model(weights_file)
try:
    model.load_weights(weights_file)
    print("模型加载成功")
except:
    print("模型加载失败")

#%%加载数据集
testX, trainX, Y_test, Y_Train, snrTest, snrTrain = load_data_all()
print("数据集加载完成")
# train_num = 180000  # 训练的样本数
# test_num = 90000    # 测试的样本数
# channel_iq = 2      # iq两个通道
# spot_num = 448      # 每个通道的采样点数
# bit_num = 32        # 比特位数
#%%数据预处理
trainX = np.reshape(trainX, [20000 * 9, 448, 2, 1])
testX = np.reshape(testX, [10000 * 9, 448, 2, 1])

Y_Train = np.reshape(Y_Train, [20000 * 9, 32, 2])
Y_test = np.reshape(Y_test, [10000 * 9, 32, 2])

snrTrain = np.reshape(snrTrain, [1, 20000 * 9])
snrTest = np.reshape(snrTest, [1, 10000 * 9])

# Y_Train = np.split(Y_Train, 32, axis=1)  # 按列分割，分成32个部分
# Y_test = np.split(Y_test, 32, axis=1)  # 此时该三维数组会变成四维，分割出的二维数组被设为三维数组，因此总共四维
#
# for i in range(32):  #
#     Y_Train[i] = np.reshape(Y_Train[i], [20000 * 9, 2])  # python中，第一个[]就代表第一级数组，
#     Y_test[i] = np.reshape(Y_test[i], [10000 * 9, 2])

#%%随机截取部分数据
star = time.time()
random.seed(100)    # 特定的随机种子
rate = 0.1      # 每类数据取1%
num = 10000     # 每类数据的数量
select_num = int(num * rate)    # 每个类别选择的总数
snr_num = 9     # 9个信噪比类别
all_num = select_num*snr_num    # 选择的总数

testsnr_list = np.array(list(set(snrTest[0]))).reshape(1, len(set(snrTest[0])))     # 信噪比 及其对应的 位置
temp = np.array([[0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]])      # 不同信噪比对应的索引位置
testsnr_list = np.concatenate((testsnr_list, temp), axis=0)     # 合并
# print("testsnr_list", testsnr_list)

randomindex = []        # 随机索引汇总
for i in range(snr_num):      # 九个类别
    temp_list = list(range(int(testsnr_list[1][i]), int(testsnr_list[1][i]) + num))  # 创建索引列表0~9999 10000~19999 ...
    random.shuffle(temp_list)   # 打乱索引
    temp_list = temp_list[:select_num]  # 取前一部分
    randomindex = randomindex + [temp_list]     # 随机索引汇总
# print("randomindex", randomindex)

for i in range(snr_num):
    for j in range(select_num):
        if i == 0 and j == 0:
            testx = testX[randomindex[i][j]].reshape(1, 448, 2, 1)
            testy = Y_test[randomindex[i][j]].reshape(1, 32, 2)
            continue
        temp = testX[randomindex[i][j]].reshape(1, 448, 2, 1)
        testx = np.concatenate((testx, temp), axis=0)       # 根据索引 将x数据汇总

        temp = Y_test[randomindex[i][j]].reshape(1, 32, 2)
        testy = np.concatenate((testy, temp), axis=0)       # 根据索引 将y标签汇总

print("数据预处理时间：", time.time() - star)

#%%测试
sta = time.time()
# testx(10,448,2,1)
result = model.predict(x=testx)     # result=(32,数量,2) 预测9w个数据400s
resulty = []
for i in range(32):
    if i == 0:
        resulty = result[i].argmax(axis=1).reshape(1, all_num)
        continue
    temp = result[i].argmax(axis=1).reshape(1, all_num)
    resulty = np.concatenate((resulty, temp), axis=0)
resulty = resulty.T

righty = []
for i in range(all_num):
    if i == 0:
        righty = testy[i].argmax(axis=1).reshape(1, 32)
        continue
    temp = testy[i].argmax(axis=1).reshape(1, 32)
    righty = np.concatenate((righty, temp), axis=0)
size = resulty.size     # 元素总数
aaaa = resulty - righty
count = np.sum((resulty - righty) != 0)
all_error = count / size
print("总误码率：{:.4f}".format(all_error))

error = []
for i in range(9):
    temp = aaaa[i*select_num:(i+1)*select_num]
    count = np.sum(temp != 0)
    size = temp.size
    error.append(count/size)
error = np.array(error).reshape(1,9)
print("误码率为{0}".format(error))
end = time.time()
print("数据处理花了{:.8f}秒".format(end - sta))
#%%
plt.title("DeepReciver_baseline", fontsize=12)
plt.xlabel("Eb/E0(db)", fontsize=12)
plt.ylabel("BER(%)", fontsize=12)
plt.plot(np.arange(9), error[0]*100, "--ob", markerfacecolor='none')
plt.savefig('/home/NewDisk/gejie/program/SR2CNN/CEIC-36/Code-CEIC36/SR2CNN_CEIC36_NEWDATA/实验图片/deepreciver_baseline.png')
plt.show()

# print(righty)
# sta = time.time()
# righty = []
# resulty = []
# for i in range(select_num):
#     if i == 0:
#         righty = np.array(testy)[..., i, :].argmax(axis=1).reshape(1,32)
#         resulty = np.array(result)[..., i, :].argmax(axis=1).reshape(1,32)
#         continue
#     temp = np.array(testy)[..., i, :].argmax(axis=1).reshape(1,32)
#     righty = np.concatenate((righty, temp), axis=0)
#
#     temp = np.array(result)[..., i, :].argmax(axis=1).reshape(1,32)
#     resulty = np.concatenate((resulty, temp), axis=0)
# size = resulty.size     # 元素总数
# count = np.sum((resulty - righty) != 0)
# error = count / size
# print("error", error)
# end = time.time()
# print("数据处理花了{}秒",format(end - sta))


