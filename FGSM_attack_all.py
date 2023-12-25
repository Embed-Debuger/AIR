from keras import backend, losses, Input
import numpy as np
from sklearn.utils import shuffle

from getData_all import load_data_all
from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf
import argparse
import sys
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
def fgsm(model, image, y_true, eps=0.1, limit=0.15):
    y_pred = model.output
    label = Input(shape=(32,None,2))
    # y_true: 目标真实值的张量。
    # y_pred: 目标预测值的张量。
    print("eps", eps)
    loss = 0
    for i in range(32):
        # print(y_true[0][i])
        # print(np.shape(y_true[0][i]))
        loss = loss + losses.categorical_crossentropy(label[0][i], y_pred[i])
        a = y_pred[i]


    gradient = backend.gradients(loss, model.input)
    gradient = gradient[0]
    print("生成adv")
    adv_noise= backend.sign(gradient) * eps
    adv = image + adv_noise  # fgsm算法
    sess = backend.get_session()
    adv = sess.run(adv, feed_dict={model.input: image, label:y_true})  # 注意这里传递参数的情况
    # print(adv-image)
    # 没有必要再限制噪声的幅值，再eps的时候已经限制了
    # pert = np.clip(adv-image, -limit, limit)  # 有的像素点会超过255，需要处理
    # adv = pert + image

    return adv

def plot_signal(signal1,signal2,limit,db):
    # 取其中一个
    signal1 = signal1[0].reshape(2,448)
    signal2 = signal2[0].reshape(2, 448)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.subplot(211)
    plt.plot(signal1[0], label="original signal")
    # plt.subplot(212)
    plt.plot(signal2[0], label="adversial signal")

    plt.title("Images of signal before and after FGSM attack. epsilon = "+ str(db))

    plt.legend()
    plt.show()
# fgsm攻击 函数调用
# 下面部分代码(图像处理)需要根据自己的攻击图像的实际情况进行修改
def fgsm_attack(signal, label,model, epsilons=10):
    # 加载准备攻击的模型，对要攻击的图形进行转换
    lpr_model = model
    # img_convert = cv2.resize(image, (x, y))  # 这里的x/y根据要求进行修改
    ret_predict = lpr_model.predict(signal) # 进行预测

    # 计算eps的值
    db = arg.PARA
    # 将db转为比值，并×平均功率0.5后开方，得到最大幅值
    limit = db
    print("limit限制")
    epsilons = np.linspace(0, limit, num=2)[1:]
    print("开始使用fgsm进行攻击")
    for eps in epsilons:
        img_attack = fgsm(lpr_model, signal, label, eps=eps ,limit = limit)
        print(
            np.linalg.norm(np.reshape(img_attack[0] - signal[0], [448 * 2, 1]),
                           ord=2))
        print(
            np.linalg.norm(np.reshape(img_attack[0] - signal[0], [448 * 2, 1]),
                           ord=np.inf))
        attack = lpr_model.predict(img_attack)
        # 当识别的结果超过100%不同的时候，就算攻击成功
        attack_num = 0
        for i in range(32):
            for t in range(batchsize):
                if np.argmax(attack[i][t]) != np.argmax(ret_predict[i][t]):
                    print('第 '+str(i)+'位 分类结果：', ret_predict[i][t])
                    print('第 '+str(i)+'位 攻击成功，攻击后的结果为：', attack[i][t])
                    attack_num = attack_num+1
    plot_signal(img_attack,signal,limit,db)


    print(attack_num)
    print(attack_num/(32*batchsize))
        # return True
    return attack_num/(32*batchsize)


if __name__ == "__main__":
    def parse_arguments(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--PARA', type=float,
                            help='Directory with aligned face thumbnails.')
        return parser.parse_args(argv)


    arg = parse_arguments(sys.argv[1:])

    testX, trainX, Y_test, Y_Train, snrTest, snrTrain = load_data_all()
    print(np.shape(testX))


    trainX = np.reshape(trainX, [20000*9, 448, 2, 1])
    # testX = np.reshape(testX, [10000*9, 448, 2, 1])

    Y_Train = np.reshape(Y_Train, [20000*9, 32,1, 2])
    # Y_test = np.reshape(Y_test, [10000*9, 32,1, 2])

    # 打乱训练集
    index = [i for i in range(len(trainX))]
    # 下面这种写法也可以
    # index = np.arange(len(dataset))
    np.random.shuffle(index)  # 打乱索引

    trainX = trainX[index]
    Y_Train = Y_Train[index]

    label_shape = np.shape(Y_Train[0])
    model = load_model(
        "/home/zhanglongyuan/zhanglongyuan/DeepReceiver_keras/weights_all/deepreceiver.h5")
    batchsize = 64
    ASR = 0
    filename = "./weights_all/"+str(arg.PARA)+"FGSM_ASR.txt"
    for i in range(int(1)):
        print("第 " + str(i) + " 批次图片")
        asr=fgsm_attack(trainX[i:i + batchsize], Y_Train[i:i + batchsize], model)
        ASR = ASR + asr/int(1)


    f = open(
        filename, 'w', encoding ='utf - 8')  ##ffilename可以是原来的txt文件，也可以没有然后把写入的自动创建成txt文件
    f.write(str(ASR))
    f.close()
