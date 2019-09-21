import os

from random import shuffle
from src.wav_util import *
from src.utils import *


def prepare_data(type, hp, shuffle=True, length=None,):
    """
    数据准备接口
    :param type: 数据类型
    :param batch_size: batch大小
    :param is_shuffle: 是否乱序
    :param length: 数据长度
    :return:
    """
    # 0.准备训练所需数据------------------------------
    hparams = DataHparams()
    parser = hparams.parser
    data_hp = parser.parse_args()

    data_hp.data_type = type
    data_hp.shuffle = shuffle
    data_hp.data_length = length

    batch_size = hp.batch_size
    feature_dim = hp.feature_dim
    data = GetData(data_hp, batch_size, feature_dim)
    return data


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

    def get_fbank_and_pinyin_data(self, index):
        """
        获取一条语音数据的Fbank与拼音信息
        :param index: 索引位置
        :return:
            input_data: 语音特征数据
            data_length: 语音特征数据长度
            label: 语音标签的向量
        """
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(self.data_path, self.path_lst[index])
            noise_file = Const.NoiseOutPath + self.path_lst[index]
            fbank = compute_fbank_from_file(file) if os.path.isfile(file) else\
                compute_fbank_from_file(noise_file)
            input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
            data_length = input_data.shape[0] // 8 + 1
            label = pny2id(self.pny_lst[index], acoustic_vocab)
            label = np.array(label)
            len_label = len(label)
            # 将错误数据进行抛出异常,并处理
            if input_data.shape[0] > self.feature_max_length:
                raise ValueError
            if len_label > 64 or len_label > data_length:
                raise ValueError
            return input_data, data_length, label, len_label
        except ValueError:
            raise ValueError

    def get_fbank_and_hanzi_data(self, index):
        '''
        获取一条语音数据的Fbank与拼音信息
        :param index: 索引位置
        :return: 返回相应信息
        '''
        try:
            # Fbank特征提取函数(从feature_python)
            file = os.path.join(self.data_path, self.path_lst[index])
            noise_file = Const.NoiseOutPath + self.path_lst[index]
            input_data = compute_fbank_from_file(file, feature_dim=self.feature_dim) if os.path.isfile(file) else \
                compute_fbank_from_file(noise_file, feature_dim=self.feature_dim)
            label = han2id(self.han_lst[index], hanzi_vocab)
            tar_label = label
            label.insert(0, Const.SOS)
            tar_label.append(Const.EOS)

            label = np.array(label)
            tar_label = np.array(tar_label)
            len_label = len(label)
            # 将错误数据进行抛出异常,并处理
            return input_data, tar_label, label, len_label
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
                        input_data, data_length, label, len_label, = self.get_fbank_and_pinyin_data(index)
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

    def get_lm_batch(self):
        '''
        训练语言模型batch数据，拼音到汉字
        :return:
        '''
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
                    py_vec = pny2id(self.pny_lst[i], acoustic_vocab)\
                             + [0] * (max_len - len(self.pny_lst[i].strip().split(' ')))
                    han_vec = han2id(self.han_lst[i], language_vocab) + [0] * (max_len - len(self.han_lst[i].strip()))
                    input_data.append(py_vec)
                    label_data.append(han_vec)
                except ValueError:
                    continue
            input_data = np.array(input_data)
            label_data = np.array(label_data)
            yield input_data, label_data
        pass

    def get_transformer_batch(self):
        '''
        # transformer训练batch数据
        :return:
        '''
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
                        input_data, target_label, label, len_label, = self.get_fbank_and_hanzi_data(index)
                        input_data = build_LFR_features(input_data, self.lfr_m, self.lfr_n)
                        wav_data_lst.append(input_data)
                        label_data_lst.append(label)
                        target_label_lst.append(target_label)
                    except ValueError:
                        error_count.append(i)
                        continue
                # label为decoder的输入，ground_truth为decoder的输出
                pad_wav_data, input_length = wav_padding(wav_data_lst)
                pad_label_data, label_length = label_padding(label_data_lst, Const.EOS)
                pad_target_data, _ = label_padding(target_label_lst, Const.PAD)
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


    def get_transformer_data_from_file(self, file):
        try:
            fbank = compute_fbank_from_file(file, feature_dim=self.feature_dim)
            input_data = build_LFR_features(fbank, self.lfr_m, self.lfr_n)
            input_data = np.expand_dims(input_data, axis=0)
            label = np.array([[Const.SOS for _ in range(input_data.shape[1])]])
            return input_data, label,
        except ValueError:
            raise ValueError

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
            noise_file = os.path.join(Const.NoiseOutPath, self.path_lst[index])
            X, y = self.get_transformer_data_from_file(file) if os.path.isfile(file) else \
                self.get_transformer_data_from_file(noise_file)
            return X, y
        except ValueError:
            raise ValueError

if __name__ == "__main__":
    pass