from enum import IntEnum
import os

# 外网机
# SpeechDataPath 音频数据文件目录
# '../../../speech_data/'
# NoiseOutPath 噪声文件目录
# '/usr/corpus/noise_data'

# server
# SpeechDataPath 音频数据文件目录
# '/data/speech_data/'
# NoiseOutPath 噪声文件目录
# '/data/speech_data/noise_data'

# mac
# SpeechDataPath 音频数据文件目录
# '../../../speech_data/'
# NoiseOutPath 噪声文件目录
# '../../../speech_data/noise_data'

ServerId = 0


class ServerIndex(IntEnum):
    Linux = 0
    Server = 1
    Mac = 2


class Const:
    # SOS为起始标识符
    # EOS为结束标志符
    PAD = 0
    SOS = 1
    EOS = 2
    PAD_FLAG = '<pad>'
    SOS_FLAG = '<sos>'
    EOS_FLAG = '</sos>'

    # 噪声文件
    NoiseDataTxT = '../data/noise_data.txt'

    # 声学模型文件
    AmModelFolder = '../model_and_log/logs_am/checkpoint/'
    AmModelTensorBoard = '../model_and_log/logs_am/tensorboard/'
    # 语言模型文件
    LmModelFolder = '../model_and_log/logs_lm/checkpoint/'
    LmModelTensorboard = '../model_and_log/logs_lm/tensorboard/'

    # Seq2Seq-transformer
    TransformerFolder = '../model_and_log/logs_seq2seq/checkpoint/'
    TransformerTensorboard = '../model_and_log/logs_seq2seq/tensorboard/'

    # 预测结果保存路径
    PredResultFolder = '../model_and_log/pred/'

    if ServerId == ServerIndex.Linux:
        SpeechDataPath = '../../../speech_data/'
        NoiseOutPath = '/usr/corpus/noise_data/'

    elif ServerId == ServerIndex.Server:
        SpeechDataPath = '/data/speech_data'
        NoiseOutPath = '/data/noise_data/'

    elif ServerId == ServerIndex.Mac:
        SpeechDataPath = '../../../speech_data/'
        NoiseOutPath = '../../../speech_data/noise_data/'


if ServerId == ServerIndex.Server:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
