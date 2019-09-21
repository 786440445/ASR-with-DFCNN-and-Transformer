# coding=utf-8
import random
import sys
import os
import tensorflow as tf
import warnings
import numpy as np
import datetime

sys.path.append('../')
from src.model.cnn_ctc import CNNCTCModel
from src.model.language_model import Language_Model

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from src.data import GetData
from src.utils import GetEditDistance, acoustic_vocab, language_vocab
from src.const import Const
from src.hparams import DataHparams, AmHparams,  LmHparams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def pred_pinyin(model, inputs, input_length):
    pred = model.predict(inputs, input_length)
    text = []
    for k in pred:
        text.append(acoustic_vocab[k])
    pinyin = ' '.join(text)
    return pred, pinyin


def speech_test(am_model, lm_model, test_data, num, sess):
    # 3. 进行测试-------------------------------------------
    num_data = len(test_data.pny_lst)
    length = test_data.data_length
    if length == None:
        length = num_data
    ran_num = random.randint(0, length - 1)
    words_num = 0
    word_error_num = 0
    han_num = 0
    han_error_num = 0
    data = ''
    for i in range(num):
        print('\nthe ', i+1, 'th example.')
        # 载入训练好的模型，并进行识别
        index = (ran_num + i) % num_data
        try:
            hanzi = test_data.han_lst[index]
            hanzi_vec = [language_vocab.index(idx) for idx in hanzi]
            inputs, input_length, label, _ = test_data.get_data(index)
            pred, pinyin = pred_pinyin(am_model, inputs, input_length)
            y = test_data.pny_lst[index]

            # 语言模型预测
            with sess.as_default():
                py_in = pred.reshape(1, -1)
                han_pred = sess.run(lm_model.preds, {lm_model.x: py_in})
                han = ''.join(language_vocab[idx] for idx in han_pred[0])
        except ValueError:
            continue
        print('原文汉字结果:', ''.join(hanzi))
        print('原文拼音结果:', ''.join(y))
        print('预测拼音结果:', pinyin)
        print('预测汉字结果:', han)
        data += '原文汉字结果:' + ''.join(hanzi) + '\n'
        data += '原文拼音结果:' + ''.join(y) + '\n'
        data += '预测拼音结果:' + pinyin + '\n'
        data += '预测汉字结果:' + han + '\n'

        words_n = label.shape[0]
        words_num += words_n  # 把句子的总字数加上
        py_edit_distance = GetEditDistance(label, pred)
        # 拼音距离
        if (py_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
            word_error_num += py_edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            word_error_num += words_n  # 就直接加句子本来的总字数就好了

        # 汉字距离
        words_n = np.array(hanzi_vec).shape[0]
        han_num += words_n  # 把句子的总字数加上
        han_edit_distance = GetEditDistance(np.array(hanzi_vec), han_pred[0])
        if (han_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
            han_error_num += han_edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            han_error_num += words_n  # 就直接加句子本来的总字数就好了

    data += '*[Test Result] Speech Recognition ' + test_data.data_type + ' set word accuracy ratio: ' + str((1 - word_error_num / words_num) * 100) + '%'
    filename = str(datetime.datetime.now()) + '_' + str(num)
    with open(os.path.join(Const.PredResultFolder, filename), 'w') as f:
        f.writelines(data)
    print('*[Test Result] Speech Recognition ' + test_data.data_type + ' set 拼音 word accuracy ratio: ',
          (1 - word_error_num / words_num) * 100, '%')
    print('*[Test Result] Speech Recognition ' + test_data.data_type + ' set 汉字 word accuracy ratio: ',
          (1 - han_error_num / han_num) * 100, '%')


def main():
    # 测试长度
    # 1. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试
    hparams = DataHparams()
    parser = hparams.parser
    data_hp = parser.parse_args()
    data_hp.data_type = 'test'
    data_hp.shuffle = True
    data_hp.data_length = None
    test_count = 500

    # 2.声学模型-----------------------------------
    hparams = AmHparams()
    parser = hparams.parser
    am_hp = parser.parse_args()
    am_model = CNNCTCModel(am_hp)

    test_data = GetData(data_hp, am_hp.batch_size, am_hp.feature_dim, am_hp.feature_max_length)

    print('loading acoustic model...')
    am_model.load_model('model_04-14.91')

    # 3.语言模型-----------------------------------
    hparams = LmHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    print('loading language model...')
    lm_model = Language_Model(hp)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
    with lm_model.graph.as_default():
        saver = tf.train.Saver()
    latest = tf.train.latest_checkpoint(Const.LmModelFolder)
    print(latest)
    saver.restore(sess, latest)
    speech_test(am_model, lm_model, test_data, test_count, sess)


if __name__ == '__main__':
    main()