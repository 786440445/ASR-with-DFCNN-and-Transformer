import librosa as lrs
import numpy as np
import os
import random
from tqdm import tqdm


'''
有色噪声定义
返回：
    一条有色噪声
参数：
    len_noise : 噪声采样点数, 需要和待扩增信号长度相同;
    type_noise: 有色噪声类型, -1至1之间的浮点数, 其中0表示白噪声, 小于0表示红移噪声类, 大于0表示蓝移.
'''


def color_noise(len_noise, type_noise):
    x_random = np.random.normal(0, 1, len_noise)
    mid_frame = int(np.ceil((len_noise + 1) / 2))
    x_fft = np.fft.fft(x_random)
    x_fft_half = x_fft[:mid_frame]
    n = np.arange(1, mid_frame + 1)
    x_fft_half = x_fft_half * (n ** type_noise)
    if len_noise % 2 == 0:
        x_fft_half_ = np.conj(x_fft_half[-2:0:-1])
    else:
        x_fft_half_ = np.conj(x_fft_half[-1:0:-1])
    noise = np.concatenate([x_fft_half, x_fft_half_])
    noise = np.real(np.fft.ifft(noise))
    noise = noise - np.mean(noise)
    noise = noise / np.max(noise)

    noise = noise.astype(np.float32)
    return noise


'''
噪声加权系数定义
返回：
    噪声加权系数K
参数：
    signal: 待加噪声信号;
    noise : 噪声信号;
    dB    : 信噪比（SNR）, 一般取大于5dB较好.
'''


def SNR2K(signal, noise, dB):
    energe_s = np.sum(signal * signal) / len(signal)
    energe_n = np.sum(noise * noise) / len(noise)
    K = np.sqrt(energe_s / energe_n) * (10 ** (-dB / 20))
    return K


'''
批量添加噪声定义
返回：
    若未指定out_path, （即默认值None） 则以列表形式返回加噪后的信号; 若指定out_path, 则将加噪后信号保存至目标目录, 返回空列表.
参数：
    signal_path: str或list, 加噪前信号存储路径(相对路径或绝对路径均可, 传入文件夹(str)或文件名列表(list)均可)
    n_to_add   : int, 每条信号加噪的数量, 默认为1;
    sample_rate: int, 采样率, 默认16000;
    out_path   : str, 加噪后信号存储文件夹位置, 可选参数;
    dB         : str, 指定信噪比, 默认为random, 即5-10之间的随机数;
    type_noise : str, 指定噪声类型, 可选参数, 默认random, 即-1到1之间的随机数.
    keep_bits  : bool, 确定是否保持比特率不变，默认False时，保存的wav文件大小会扩大4倍，设置为True时，大小不变，但额外需要pydub库。
'''


def add_noise(signal_path, n_to_add=1, sample_rate=16000, out_path=None, dB='random', type_noise='random',
              keep_bits=False):
    if isinstance(signal_path, list):
        if os.path.isfile(signal_path[0]):
            signal_files = signal_path
        else:
            print('Error signal_path!')
            return 0
    elif os.path.isdir(signal_path):
        signal_files = list(os.listdir(signal_path))
        for n in range(len(signal_files)):
            signal_files[n] = os.path.join(signal_path, signal_files[n])
    else:
        print('Error signal_path!')
        return 0

    signal_added_list = []
    length = len(signal_files)
    l = 0
    name_list = []
    for file in tqdm(signal_files):
        n = 0
        signal, _ = lrs.load(file, sr=sample_rate)

        while n < n_to_add:
            if dB == 'random':
                snr_dB = random.randint(5, 10)
            else:
                snr_dB = int(dB)
            if type_noise == 'random':
                type_n = random.randint(-10, 10) / 10
            else:
                type_n = str(type_noise)
                if np.abs(float(type_n)) > 1:
                    print('Error noise type! Please given a float belongs to [-1, 1] !')
                    return 0
            noise = color_noise(len(signal), type_noise=type_n)
            K = SNR2K(signal, noise, dB=snr_dB)
            signal_added = (signal + K * noise).astype(np.float32)
            if out_path is not None:
                # 加噪信号命名为: 索引_噪声类型_信噪比_dB.wav
                # 例: data/mixed/1_0.7_8dB.wav
                path = out_path + '/' + str(l) + '_' + str(n) + '_' + str(type_n) + '_' + str(snr_dB) + '_dB.wav'
                name_list.append(path)
                # 仅当加噪后失真时才做归一化处理，否则保持原有的数据分布
                if max(np.abs(signal_added)) > 1:
                    if_norm = True
                else:
                    if_norm = False
                lrs.output.write_wav(path, y=signal_added, sr=sample_rate, norm=if_norm)
                if keep_bits:
                    from pydub import AudioSegment
                    signal = AudioSegment.from_wav(path)
                    print(signal)
                    signal.export(path, format='wav', bitrate='128')
            else:
                signal_added_list.append(signal_added)
            n += 1
        l += 1
    return signal_added_list, name_list