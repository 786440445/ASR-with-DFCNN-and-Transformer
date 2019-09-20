from keras.utils import Sequence
import numpy as np
import math
import os
import soundfile as sf

from src.const import Const
from src.wav_util import compute_fbank_from_api
from src.data import pny2id


class DataGenerator(Sequence):
    def __init__(self, datas, hp):
        self.batch_size = hp.batch_size
        self.paths = datas.path_lst
        self.py_labels = datas.pny_lst
        self.hz_labels = datas.han_lst
        self.feature_max_length = hp.feature_max_length
        self.indexes = np.arange(len(self.paths))
        self.shuffle = datas.shuffle
        self.data_path = datas.data_path
        self.type = datas.data_type

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.paths[k] for k in batch_indexs]
        py_label_datas = [self.py_labels[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_datas, py_label_datas)
        return X, y

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.paths) / float(self.batch_size))

    def data_generation(self, batch_datas, py_label_datas):
        # batch_wav_data.shape = (10 1600 200 1), inputs_length.shape = (10,)
        batch_wav_data = np.zeros((self.batch_size, self.feature_max_length, 200, 1), dtype=np.float)
        # batch_label_data.shape = (10 64) ,label_length.shape = (10,)
        batch_label_data = np.zeros((self.batch_size, 64), dtype=np.int64)
        # length
        input_length = []
        label_length = []
        error_count = []
        # 随机选取batch_size个wav数据组成一个batch_wav_data
        for i, path in enumerate(batch_datas):
            # Fbank特征提取函数(从feature_python)
            try:
                file1 = os.path.join(self.data_path, path)
                file2 = os.path.join(Const.NoiseOutPath, path)
                if os.path.isfile(file1):
                    signal, sample_rate = sf.read(file1)
                elif os.path.isfile(file2):
                    signal, sample_rate = sf.read(file2)
                else:
                    print("file path Error")
                    return 0
                fbank = compute_fbank_from_api(signal, sample_rate)
                input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
                data_length = input_data.shape[0] // 8 + 1
                label = pny2id(py_label_datas[i])
                label = np.array(label)
                len_label = len(label)
                # 将错误数据进行抛出异常,并处理
                if input_data.shape[0] > self.feature_max_length:
                    raise ValueError
                if len_label > 64 or len_label > data_length:
                    raise ValueError

                input_length.append([data_length])
                label_length.append([len_label])
                batch_wav_data[i, 0:len(input_data)] = input_data
                batch_label_data[i, 0:len_label] = label
            except ValueError:
                error_count.append(i)
                continue
        # 删除异常语音信息
        if error_count != []:
            batch_wav_data = np.delete(batch_wav_data, error_count, axis=0)
            batch_label_data = np.delete(batch_label_data, error_count, axis=0)
        label_length = np.mat(label_length)
        input_length = np.mat(input_length)
        # CTC 输入长度0-1600//8+1
        # label label真实长度
        inputs = {'the_inputs': batch_wav_data,
                  'the_labels': batch_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros((self.batch_size - len(error_count), 1), dtype=np.float32)}
        return inputs, outputs