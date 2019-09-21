import wave
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.fftpack import fft
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
from sklearn import preprocessing

from src.noise import add_noise


def compute_fbank_from_file(file, feature_dim=200):
    signal, sample = sf.read(file)
    feature = compute_fbank_from_api(signal, sample_rate, nfilt=feature_dim)
    return feature


def compute_fbank_from_api(signal, sample_rate, nfilt=200):
    """
    Fbank特征提取, 结果进行零均值归一化操作
    :param wav_file: 文件路径
    :return: feature向量
    """
    feature = logfbank(signal, sample_rate, nfilt=nfilt)
    feature = preprocessing.scale(feature)
    return feature


def read_wav_data(filename):
    wav = wave.open(filename, "rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    #wave_data = wave_data
    return wave_data, framerate


# 获取信号的时频图
def compute_fbank(file):
    # 汉明窗
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    # 预加重
    pre_emphasis = 0.97
    emphasized_signal = np.append(wavsignal[0], wavsignal[1:] - pre_emphasis * wavsignal[:-1])
    # 计算循环终止的位置，也就是最终生成的窗数
    range0_end = int(len(emphasized_signal) / fs * 1000 - time_window) // 10 + 1
    # 用于存放最终的频率特征数据
    data_input = np.zeros((range0_end, 200), dtype=np.float)
    data_line = np.zeros((1, 400), dtype=np.float)

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        # 加窗
        data_line = data_line * w
        data_line = np.abs(fft(data_line))

        # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        data_input[i] = data_line[0:200]
    feature = np.log(data_input + 1)

    # 0均值方差归一化处理
    return preprocessing.scale(feature)


def compute_fbank_from_asrt(file):
    # 汉明窗
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    wavsignal, fs = read_wav_data(file)

    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]
    # 预加重
    pre_emphasis = 0.97
    emphasized_signal = np.append(wavsignal[0], wavsignal[1:] - pre_emphasis * wavsignal[:-1])
    # 计算循环终止的位置，也就是最终生成的窗数
    range0_end = int(len(emphasized_signal) / fs * 1000 - time_window) // 10
    # 用于存放最终的频率特征数据
    data_input = np.zeros((range0_end, 200), dtype=np.float)
    data_line = np.zeros((1, 400), dtype=np.float)

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[0, p_start:p_end]
        # 加窗
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length

        # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        data_input[i] = data_line[0:200]
    data_input = np.log(data_input + 1)
    return data_input


def wav_show(wave_data, fs): # 显示出来声音波形
    time = np.arange(0, len(wave_data)) * (1.0/fs)  # 计算声音的播放时间，单位为秒
    plt.plot(time, wave_data)


def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()


# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2, fft_size/2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()


# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()


if __name__ == "__main__":
    file = "../../../speech_data/data_thchs30/data/A2_0.wav"
    signal1, sample_rate = sf.read(file)
    signal2 = add_noise(signal1, sample_rate=16000, n_to_add=1, dB='random', type_noise='0.0')
    signal2 = signal2[0]

    plot_time(signal1, sample_rate)
    plot_time(signal2, sample_rate)

    plot_freq(signal1, sample_rate)
    plot_freq(signal2, sample_rate)

    plt.show()
    # # sample_rate, signal1 = wav.read(file) # 读取的数字为[-2^15, 2^15]之间，除以2^15就可以转化为下面的方法
    # # signal2, sr = sf.read(file)  # 读取的数字为[-1, 1]之间
    # # signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    # # print('sample rate:', sample_rate, ', frame length:', len(signal))
    # # 原始信号波形图
    # plot_time(signal1, sample_rate)
    # # fbank from api画图
    # signal, sample_rate = sf.read(file)
    # feature = logfbank(signal, sample_rate, nfilt=200)
    # feature1 = preprocessing.scale(feature)
    #
    # # 比较三个方法频谱图
    # filter_banks1 = compute_fbank_from_api(file)
    # plot_spectrogram(filter_banks1.T, 'Filter Banks_from_api')
    # # # fbank 画图
    # filter_banks2 = compute_fbank(file)
    # plot_spectrogram(feature.T, 'Filter Banks')
    # plt.show()
    # receive_wav()