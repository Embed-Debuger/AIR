from keras import backend, losses, Input
import numpy as np

from getData import load_data
from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)


def cal_PAPR(data1):
    print(np.shape(data1))
    result1 = 2*np.power(np.max(data1), 2) / (
            (1 / (448*batchsize)) * np.sum(np.power(data1, 2)))
    return result1


def mifgsm(model, signal, true_label, T, eps=0.1):
    print("源信号功率为%f" % (
        (np.sum(np.power(signal, 2)) / (448 * batchsize))))
    print("eps", eps)
    pred = model.output
    label = Input(shape=(32,None,2))
    # true_label: 目标真实值的张量。
    # pred: 目标预测值的张量。


    # 第一轮将原图输入，并设定步长alph，和动量g
    adv_sample = signal
    alph = eps / T
    mu = 1
    g = 0
    sess = backend.get_session()


    # 在迭代中进行优化功率和PAPR
    for t in range(T):
        loss = 0
        for i in range(32):
            loss = loss - losses.categorical_crossentropy(label[0][i], pred[i])

            gradient = backend.gradients(loss, model.input)
            gradient = gradient[0]
            adv_noise = backend.sign(gradient) * eps / T
            adv = model.input + adv_noise  # mifgsm算法

        adv,loss1 = sess.run([adv,loss], feed_dict={model.input: adv_sample,
                                       label: 1-true_label})  # 注意这里传递参数的情况
        print("第%d次迭代" %(t))
        # 计算生成的扰动功率和PAPR
        print("噪声功率为%f" % ((np.sum(np.power(adv - signal, 2)) / (448 * batchsize))))

        # 进行对抗噪声功率归一化
        power = np.power(10, arg.POWER / 10)
        noise_gyh = (adv - adv_sample) / (
            np.sqrt(np.sum(np.power(adv - adv_sample, 2)) / (448 * batchsize))) * np.sqrt(power)

        papr = np.power(10, arg.PAPR / 10)
        max = np.sqrt((np.sum(np.power(noise_gyh, 2)) / (
                    448 * batchsize)) * papr)
        print("归一化后噪声功率为%f" % (
        (np.sum(np.power(noise_gyh, 2)) / (448 * batchsize))))

        noise_gyh = np.clip(noise_gyh, -max, max)

        max = np.sqrt((np.sum(np.power(noise_gyh, 2)) / (
                448 * batchsize)) * papr)

        noise_gyh = np.clip(noise_gyh, -max, max)


        print("最大幅值应该为%f" % (max))
        print("实际最大幅值为%f" % (np.max(noise_gyh)))



        print("PAPR为%f" % (cal_PAPR(noise_gyh)))


        adv_sample = adv_sample+noise_gyh
    return adv_sample

def plot_signal(signal1,signal2,limit,db):
    # 取其中一个
    signal1 = signal1[0].reshape(2,448)
    signal2 = signal2[0].reshape(2, 448)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.subplot(211)
    plt.plot(signal2[0], label="original signal")
    # plt.subplot(212)
    plt.plot(signal1[0], label="adversial signal")

    plt.title("Images of signal before and after PGD attack. PAPR = "+ str(arg.PARA)+" POWER = "+str(arg.POWER))

    plt.legend()
    plt.show()
# fgsm攻击 函数调用
# 下面部分代码(图像处理)需要根据自己的攻击图像的实际情况进行修改
def mifgsm_attack(signal, label,model,T):


    # 加载准备攻击的模型，对要攻击的信号进行预测
    lpr_model = model
    # img_convert = cv2.resize(image, (x, y))  # 这里的x/y根据要求进行修改
    ret_predict = lpr_model.predict(signal) # 进行预测，得到预测结果



    # 将db转为比值，并×平均功率0.5后开方，得到最大幅值
    # limit = arg.limit1
    # print("limit限制")
    # epsilons = np.linspace(0, limit, num=2)[1:]
    print("开始攻击")


    signal_attack = mifgsm(model=lpr_model, signal=signal, true_label = label, T=T)

            # 迭代完成后获得对抗噪声
            # 计算初始噪声功率
    noise = signal_attack-signal
    print("噪声功率为%f"%((np.sum(np.power(noise, 2)) / (448*batchsize))))

    print("噪声PAPR为%f"%(cal_PAPR(noise)))

    attack = lpr_model.predict(signal_attack)
    attack_num = 0
    for i in range(32):
        for t in range(batchsize):
            if np.argmax(attack[i][t]) != np.argmax(ret_predict[i][t]):
                # print('第 '+str(i)+'位 分类结果：', ret_predict[i][t])
                # print('第 '+str(i)+'位 攻击成功，攻击后的结果为：', attack[i][t])
                attack_num = attack_num+1

    ASR = attack_num/(32*batchsize)

    return ASR

import argparse
import sys
if __name__ == "__main__":
    def parse_arguments(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--db', type=int,
                            help='攻击的源信号的信噪比')
        parser.add_argument('--PAPR', type=float,
                            help='攻击后生成噪声的PAPR')
        parser.add_argument('--POWER', type=float,
                            help='攻击后生成噪声的功率')
        # parser.add_argument('--limit1', type=float,
        #                     help='Directory with aligned face thumbnails.')
        return parser.parse_args(argv)

    arg = parse_arguments(sys.argv[1:])


    # 加载数据
    testX, trainX, Y_test, Y_Train, snrTest, snrTrain = load_data(arg.db)
    trainX = np.reshape(trainX, [20000, 448, 2, 1])
    testX = np.reshape(testX, [10000, 448, 2, 1])
    Y_Train = np.reshape(Y_Train, [20000, 32, 1, 2])
    Y_test = np.reshape(Y_test, [10000, 32, 1, 2])
    label_shape = np.shape(Y_Train[0])


    # 加载模型
    model = load_model(
        "/home/zhanglongyuan/zhanglongyuan/DeepReceiver_keras/weights_all/deepreceiver.h5")


    # 设定攻击时候需要的条件，如攻击的迭代次数
    T = 5
    ASR = 0
    batchsize = 1
    # filename = "./MIFGSM_ASR"+str(arg.limit1)+".txt"

    # 攻击的批次数
    for i in range(int(100)):
        print("第 " + str(i) + " 批次图片")
        asr = mifgsm_attack(trainX[i:i + batchsize],Y_Train[i:i + batchsize], model, T=T)
        print(asr)

        # ASR = ASR + asr / int(batchsize)

    # f = open(
    #     filename, 'w',
    #     encoding='utf - 8')  ##ffilename可以是原来的txt文件，也可以没有然后把写入的自动创建成txt文件
    # f.write(str(asr))
    # f.close()
