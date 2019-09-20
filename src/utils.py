import difflib
import numpy as np
from keras import backend as K
import pandas as pd
import tensorflow as tf
from src.hparams import DataHparams
from src.const import Const

hparams = DataHparams()
parser = hparams.parser
hp = parser.parse_args()

# 加载所有的拼音类别
def get_py_vocab_list():
    text = pd.read_table(hp.pinyin_dict, header=None)
    symbol = text.iloc[:, 0].tolist()
    # 是否需要把pad位置放在第一个位置，然后重新训练声学模型
    # symbol = Const.PAD_FLAG.split(' ')
    # symbol.extend(symbol)
    symbol.extend(Const.PAD_FLAG.split(' '))
    symbol_num = len(symbol)
    return symbol_num, symbol


# 加载所有的汉字类别
def get_hz_vocab_list():
    text = pd.read_table(hp.hanzi_dict, header=None)
    list_hanzi = text.iloc[:, 0].tolist()
    list = Const.PAD_FLAG + ' ' + Const.SOS_FLAG + ' ' + Const.EOS_FLAG
    list = list.split(' ')
    list.extend(list_hanzi)
    hanzi_num = len(list)
    return hanzi_num, list


# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost


# 定义解码器------------------------------------
def decode_ctc(num_result, input_length):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = input_length
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=100, top_paths=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    r1 = r[0][0].eval(session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    tf.reset_default_graph()  # 然后重置tf图，这句很关键
    r1 = r1[0]
    return r1


pinyin_vocab_size, pinyin_vocab = get_py_vocab_list()
hanzi_vocab_size, hanzi_vocab = get_hz_vocab_list()