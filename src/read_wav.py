import pyaudio
import wave
import wav_util
import tensorflow as tf
import soundfile as sf
import numpy as np

from src.model.cnn_ctc import CNNCTCModel
from src.test import pred_pinyin
from src.hparams import AmHparams, LmHparams
from src.model.language_model import Language_Model
from src.utils import hanzi_vocab
from const import Const


def receive_wav():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 10

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf = wave.open('../wav_file/input_file.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def main():
    # 1.声学模型-----------------------------------
    hparams = AmHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    am_model = CNNCTCModel(hp)
    print('loading acoustic model...')
    select_model_step = 'model_05-7.64'
    am_model.load_model(select_model_step)

    # 2.语言模型-----------------------------------
    hparams = LmHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    hp.is_training = False
    print('loading language model...')
    lm_model = Language_Model(hp)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
    # wav_util.plot_time(signal, sample_rate)
    with lm_model.graph.as_default():
        saver = tf.train.Saver()
    with sess.as_default():
        latest = tf.train.latest_checkpoint(Const.LmModelFolder)
        saver.restore(sess, latest)

    # 现场输入识别
    receive_wav()
    signal, sample_rate = sf.read('../wav_file/input_file.wav')
    fbank = wav_util.compute_fbank_from_api(signal, sample_rate)
    inputs = fbank.reshape(fbank.shape[0], fbank.shape[1], 1)
    input_length = inputs.shape[0] // 8 + 1
    pred, pinyin = pred_pinyin(am_model, inputs, input_length)
    print("预测拼音结果为：", pinyin)

    # 语言模型预测
    with sess.as_default():
        pred = np.array(pred)
        han_in = pred.reshape(1, -1)
        han_vec = sess.run(lm_model.preds, {lm_model.x: han_in})
        han_pred = ''.join(hanzi_vocab[idx] for idx in han_vec[0])
        print("中文预测结果为：", han_pred)


if __name__ == '__main__':
    main()