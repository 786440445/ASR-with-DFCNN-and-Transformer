# coding=utf-8
import random
import os
import tensorflow as tf
import warnings
import numpy as np
import datetime

os.sys.path.append('../')
from src.model.transformer import Transformer
from src.data import prepare_data
from src.hparams import TransformerHparams
from src.utils import GetEditDistance, hanzi_vocab
from src.const import Const
from src.data import han2id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def get_pred_han(logits):
    logits = logits[0]
    return ''.join(hanzi_vocab[idx] for idx in logits[:-1])


def transformer_test(hp, test_data):
    model = Transformer(hp)
    with model.graph.as_default():
        saver = tf.train.Saver()

    num_data = len(test_data.pny_lst)
    length = test_data.data_length
    if length == None:
        length = num_data
    num = hp.count
    ran_num = random.randint(0, length - 1)
    han_num = 0
    han_error_num = 0
    data = ''

    with tf.Session(graph=model.graph) as sess:
        latest = tf.train.latest_checkpoint(Const.TransformerFolder)
        saver.restore(sess, latest)
        for i in range(num):
            try:
                print('\nthe ', i + 1, 'th example.')
                index = (ran_num + i) % num_data
                X, Y = test_data.get_transformer_data(index)
                hanzi = test_data.han_lst[index]
                tar_get = han2id(hanzi, hanzi_vocab)
                preds = sess.run(model.preds, feed_dict={model.x: X, model.y: Y})
                print(preds)
                han_pred = get_pred_han(preds)
            except ValueError:
                continue
            print('原文汉字结果:', ''.join(hanzi))
            print('预测汉字结果:', han_pred)
            data += '原文汉字结果:' + ''.join(hanzi) + '\n'
            data += '预测汉字结果:' + han_pred + '\n'

            # 汉字距离
            words_n = np.array(preds).shape[0]
            han_num += words_n  # 把句子的总字数加上
            han_edit_distance = GetEditDistance(np.array(tar_get), preds[0])
            if (han_edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
                han_error_num += han_edit_distance  # 使用编辑距离作为错误字数
            else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                han_error_num += words_n  # 就直接加句子本来的总字数就好了

        data += '*[Test Result] Speech Recognition ' + test_data.data_type + ' set word accuracy ratio: ' + str(
            (1 - han_error_num / han_num) * 100) + '%'
        filename = str(datetime.datetime.now()) + '_' + str(num)
        with open(os.path.join(Const.PredResultFolder, filename), 'w') as f:
            f.writelines(data)
        print('*[Test Result] Speech Recognition ' + test_data.data_type + ' set 汉字 word accuracy ratio: ',
              (1 - han_error_num / han_num) * 100, '%')

        pass


def main():
    hparams = TransformerHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    # 数据准备工作
    test_data = prepare_data('test', hp, shuffle=True, length=None)
    test_data.feature_dim = hp.feature_dim
    transformer_test(hp, test_data)


if __name__ == '__main__':
    main()