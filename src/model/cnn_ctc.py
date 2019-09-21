from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Reshape, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import numpy as np
from src.utils import decode_ctc, acoustic_vocab_size
from src.const import Const


class CNNCTCModel():
    def __init__(self, args):
        # 神经网络最终输出的每一个字符向量维度的大小
        self.vocab_size = acoustic_vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.lr
        self.feature_length = 200
        self.is_training = args.is_training
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, self.feature_length, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        self.h5 = cnn_cell(128, self.h4, pool=False)

        # [10, 200, 25, 128]
        self.h6 = Reshape((-1, 3200))(self.h5)
        self.h6 = Dropout(0.3)(self.h6)
        self.h7 = dense(128)(self.h6)
        self.h7 = Dropout(0.3)(self.h7)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)

        # 采用全局平均池化代替Dense
        # self.h6 = nin(self.h5, self.vocab_size)
        # [10, 200, 25, 1424]
        # self.h7 = global_avg_pool(self.h6)
        # self.model.outputs = tf.nn.softmax(self.h7)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    def _ctc_init(self):
        # 这里input_length指的是网络softmax输出后的结果长度，也就是经过ctc计算的loss的输入长度。
        # 由于网络的时域维度由1600经过三个池化变成200，因此output的长度为200，因此input_length<=200
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs, self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0, epsilon=10e-8)
        # sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        if self.gpu_nums > 1:
            self.ctc_model=multi_gpu_model(self.ctc_model, gpus=self.gpu_nums)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=adam)

    def predict(self, data_input, length, batch_size=1):
        ''' 返回预测结果
        :param date_input:
        :param input_len:
        :return:
        '''
        x_in = np.zeros((batch_size, 1600, self.feature_length, 1), dtype=np.float32)
        for i in range(batch_size):
            if len(data_input) > 1600:
                x_in[i, 0:1600] = data_input[:1600]
            else:
                x_in[i, 0:len(data_input)] = data_input
        # 通过输入，得到预测的值，预测的值即为网络中softmax的输出
        # shape = [1, 200, 1424]
        # 还需要通过ctc_loss的网络进行解码
        pred = self.model.predict(x_in, steps=1)
        return decode_ctc(pred, length)

    def load_model(self, model):
        self.ctc_model.load_weights(Const.AmModelFolder + model + '.hdf5')

    def save_model(self, model):
        self.ctc_model.save_weights(Const.AmModelFolder + model + '.hdf5')


# ============================模型组件=================================
def conv1x1(size):
    return Conv2D(size, (1, 1), use_bias=True, activation='relu',
                  padding='same', kernel_initializer='he_normal')

def conv2d(size):
    return Conv2D(size, (3, 3), use_bias=True, activation='relu',
                  padding='same', kernel_initializer='he_normal')

def norm(x):
    return BatchNormalization(axis=-1)(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

def dense(units, activation='relu'):
    return Dense(units, activation=activation, use_bias=True,
                 kernel_initializer='he_normal')

def dropout(x, num):
    if num == None:
        return x
    else:
        return Dropout(num)(x)

def cnn_cell(size, x, nin_flag=False, nin_size=32, pool=True):
    x = norm(conv2d(size)(x))
    if nin_flag:
        x = nin(x, nin_size)
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x

def nin(x, size):
    x = norm(conv1x1(size)(x))
    return x

def global_avg_pool(x):
    return GlobalAveragePooling2D(data_format='channels_last')(x)

# CTC_loss计算公式,通过K.ctc_batch_cost传入四个参数:
# labels： 真实y值标签
# y_pred： softmax输出y标签
# input_length：ctc输入长度
# label_length：真实y长度
# y_pred由网络输出得到的softmax输出结果，即代表，拼音的分类。
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
