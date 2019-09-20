import pandas as pd
import os

from utils import pinyin_vocab, hanzi_vocab
from random import shuffle
from wav_util import *
from const import Const


class GetData():
    def __init__(self, data_args, batch_size, feature_dim, feature_max_length=1600):
        self.start = 0
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.feature_max_length = feature_max_length

        self.data_type = data_args.data_type
        self.data_path = Const.SpeechDataPath

        self.thchs30 = data_args.thchs30
        self.aishell = data_args.aishell
        self.stcmd = data_args.stcmd
        self.aidatatang = data_args.aidatatang
        self.prime = data_args.prime

        self.noise = data_args.noise
        self.data_length = data_args.data_length
        self.shuffle = data_args.shuffle

        self.lfr_m = data_args.lfr_m
        self.lfr_n = data_args.lfr_n

        self.path_lst = []
        self.pny_lst = []
        self.han_lst = []
        self.source_init()

    def source_init(self):
        """
        txt文件初始化，加载
        :return:
        """
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.stcmd == True:
                read_files.append('stcmd_train.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.noise == True:
                read_files.append('noise_data.txt')

        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
            if self.stcmd == True:
                read_files.append('stcmd_dev.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_dev.txt')

        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
            if self.stcmd == True:
                read_files.append('stcmd_test.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_test.txt')

        for file in read_files:
            print('load ', file, ' data...')
            sub_file = '../data/' + file
            data = pd.read_table(sub_file, header=None)
            paths = data.iloc[:, 0].tolist()
            pny = data.iloc[:, 1].tolist()
            hanzi = data.iloc[:, 2].tolist()
            self.path_lst.extend(paths)
            self.pny_lst.extend(pny)
            self.han_lst.extend(hanzi)
        if self.data_length:
            self.path_lst = self.path_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]

    def get_data(self, index, label_type='pinyin'):
        """
        获取一条语音数据的Fbank信息
        :param index: 索引位置
        :return:
            input_data: 语音特征数据
            data_length: 语音特征数据长度
            label: 语音标签的向量
        """
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(self.data_path, self.path_lst[index])
            if os.path.isfile(file):
                signal, sample_rate = sf.read(self.data_path + self.path_lst[index])
            else:
                signal, sample_rate = sf.read(Const.NoiseOutPath + self.path_lst[index])
            fbank = compute_fbank_from_api(signal, sample_rate, nfilt=self.feature_dim)

            if label_type == 'pinyin':
                input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
                data_length = input_data.shape[0] // 8 + 1
                label = pny2id(self.pny_lst[index])
                label = np.array(label)
                len_label = len(label)
                # 将错误数据进行抛出异常,并处理
                if input_data.shape[0] > self.feature_max_length:
                    raise ValueError
                if len_label > 64 or len_label > data_length:
                    raise ValueError
                return input_data, data_length, label, len_label
            else:
                input_data = fbank
                data_length = input_data.shape[0] // 8 + 1
                label = han2id(self.han_lst[index])
                label.insert(0, Const.SOS)
                tar_label = han2id(self.han_lst[index])
                tar_label.append(Const.EOS)

                label = np.array(label)
                tar_label = np.array(tar_label)
                len_label = len(label)
                # 将错误数据进行抛出异常,并处理
                return input_data, data_length, tar_label, label, len_label
        except ValueError:
            raise ValueError

    def get_am_batch(self):
        """
        一个batch数据生成器，充当fit_general的参数
        :return:
            inputs: 输入数据
            outputs: 输出结果
        """
        # 数据列表长度
        shuffle_list = [i for i in range(len(self.path_lst))]
        while True:
            if self.shuffle == True:
                shuffle(shuffle_list)
            # batch_wav_data.shape = (10 1600 200 1), inputs_length.shape = (10,)
            batch_wav_data = np.zeros((self.batch_size, self.feature_max_length, 200, 1), dtype=np.float)
            # batch_label_data.shape = (10 64) ,label_length.shape = (10,)
            batch_label_data = np.zeros((self.batch_size, 64), dtype=np.int64)
            # length
            input_length = []
            label_length = []
            error_count = []
            for i in range(len(self.path_lst) // self.batch_size):
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    try:
                        # 随机选取一个batch
                        input_data, data_length, label, len_label, = self.get_data(index)
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
                yield inputs, outputs

    # 训练语言模型batch数据，拼音到汉字
    def get_lm_batch(self):
        shuffle_list = [i for i in range(len(self.pny_lst))]
        if self.shuffle == True:
            shuffle(shuffle_list)
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            index_list = shuffle_list[begin:end]
            max_len = max([len(self.pny_lst[index]) for index in index_list])
            input_data = []
            label_data = []
            for i in index_list:
                try:
                    py_vec = pny2id(self.pny_lst[i])\
                             + [0] * (max_len - len(self.pny_lst[i].strip().split(' ')))
                    han_vec = han2id(self.han_lst[i]) + [0] * (max_len - len(self.han_lst[i].strip()))
                    input_data.append(py_vec)
                    label_data.append(han_vec)
                except ValueError:
                    continue
            input_data = np.array(input_data)
            label_data = np.array(label_data)
            yield input_data, label_data
        pass

    # transformer训练batch数据
    def get_transformer_batch(self):
        wav_length = len(self.path_lst)
        shuffle_list = [i for i in range(wav_length)]
        if self.shuffle == True:
            shuffle(shuffle_list)
        while 1:
            # 随机选取batch_size个wav数据组成一个batch_wav_data
            for i in range(wav_length // self.batch_size):
                # length
                wav_data_lst = []
                label_data_lst = []
                target_label_lst = []
                error_count = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                # 随机选取一个batch
                for index in sub_list:
                    try:
                        input_data, _, target_label, label, len_label, = self.get_data(index, label_type='hanzi')
                        input_data = build_LFR_features(input_data, self.lfr_m, self.lfr_n)
                        wav_data_lst.append(input_data)
                        label_data_lst.append(label)
                        target_label_lst.append(target_label)
                    except ValueError:
                        error_count.append(i)
                        continue
                # label为decoder的输入，ground_truth为decoder的输出
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst, Const.EOS)
                pad_target_data, _ = self.label_padding(target_label_lst, Const.PAD)
                # 删除异常语音信息
                if error_count != []:
                    pad_wav_data = np.delete(pad_wav_data, error_count, axis=0)
                    pad_label_data = np.delete(pad_label_data, error_count, axis=0)
                    pad_target_data = np.delete(pad_target_data, error_count, axis=0)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length.reshape(-1, ),
                          'label_length': label_length.reshape(-1, ),  # batch中的每个utt的真实长度
                          'ground_truth': pad_target_data
                          }
                yield inputs
        pass


    def get_transformer_data(self, index):
        """
        获取一条语音数据的Fbank信息
        :param index: 索引位置
        :return:
            input_data: 语音特征数据
            data_length: 语音特征数据长度
            label: 语音标签的向量
        """
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(self.data_path, self.path_lst[index])
            if os.path.isfile(file):
                fbank = compute_transformer_fbank(self.data_path + self.path_lst[index])
            else:
                fbank = compute_transformer_fbank(Const.NoiseOutPath + self.path_lst[index])

            label = pny2id(self.pny_lst[index])
            return fbank, label
        except ValueError:
            raise ValueError

    def wav_padding(self, wav_data_lst):
        feature_dim = wav_data_lst[0].shape[1]
        # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        wav_lens = np.array([len(data) for data in wav_data_lst])
        # 取一个batch中的最长
        wav_max_len = max(wav_lens)
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, feature_dim), dtype=np.float32)
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst, pad_idx):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
        new_label_data_lst += pad_idx
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def downsample(feature, contact):
    add_len = (contact - feature.shape[0] % contact) % contact
    pad_zero = np.zeros((add_len, feature.shape[1]), dtype=np.float)
    feature = np.append(feature, pad_zero, axis=0)
    print(feature.shape)
    feature = np.reshape(feature, (feature.shape[0] / 4, feature.shape[1] * 4))
    return feature


def pny2id(line):
    """
    拼音转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
    :param line:
    :param vocab:
    :return:
    """
    try:
        line = line.strip()
        line = line.split(' ')
        return [pinyin_vocab.index(pin) for pin in line]
    except ValueError:
        raise ValueError


def han2id(line):
    """
    文字转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
    :param line:
    :param vocab:
    :return:
    """
    try:
        line = line.strip()
        res = []
        for han in line:
            if han == Const.PAD_FLAG:
                res.append(Const.PAD)
            elif han == Const.SOS_FLAG:
                res.append(Const.SOS)
            elif han == Const.EOS_FLAG:
                res.append(Const.EOS)
            else:
                res.append(hanzi_vocab.index(han))
        return res
    except ValueError:
        raise ValueError


def ctc_len(label):
    add_len = 0
    label_len = len(label)
    for i in range(label_len - 1):
        if label[i] == label[i + 1]:
            add_len += 1
    return label_len + add_len


if __name__ == "__main__":
    pass