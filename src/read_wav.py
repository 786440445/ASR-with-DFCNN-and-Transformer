import pyaudio
import tensorflow as tf
from src.wav_util import *
from src.model.cnn_ctc import CNNCTCModel
from src.test import pred_pinyin
from src.hparams import AmHparams, LmHparams, TransformerHparams
from src.model.language_model import Language_Model
from src.model.transformer import Transformer
from src.utils import language_vocab, hanzi_vocab
from src.const import Const
from src.data import prepare_data


def receive_wav(file):
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
    wf = wave.open(file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def dfcnn_speech(sess, am_model, lm_model, file):
    fbank = compute_fbank_from_file(file, feature_dim=200)
    inputs = fbank.reshape(fbank.shape[0], fbank.shape[1], 1)
    input_length = inputs.shape[0] // 8 + 1
    pred, pinyin = pred_pinyin(am_model, inputs, input_length)
    print("拼音预测结果为：", pinyin)
    pred, pinyin = pred_pinyin(am_model, inputs, input_length)
    # 语言模型预测
    with sess.as_default():
        pred = np.array(pred)
        han_in = pred.reshape(1, -1)
        han_vec = sess.run(lm_model.preds, {lm_model.x: han_in})
        han_pred = ''.join(language_vocab[idx] for idx in han_vec[0])
        print("中文预测结果为：", han_pred)


def transformer_speech(sess, model, train_data, file):
    with sess.as_default():
        X, Y = train_data.get_transformer_data_from_file(file)
        preds = sess.run(model.preds, feed_dict={model.x: X, model.y: Y})
        han_pred =  ''.join(hanzi_vocab[idx] for idx in preds[0][:-1])
        print('中文预测结果：', han_pred)


def recognition(type='dfcnn'):

    file = '../wav_file/input1.wav'

    # 现场输入识别
    receive_wav(file)
    if type == 'dfcnn':
        # 1.声学模型-----------------------------------
        hparams = AmHparams()
        parser = hparams.parser
        hp = parser.parse_args()
        am_model = CNNCTCModel(hp)
        print('loading acoustic model...')
        select_model_step = 'model_04-14.91'
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
        with lm_model.graph.as_default():
            saver = tf.train.Saver()
        with sess.as_default():
            latest = tf.train.latest_checkpoint(Const.LmModelFolder)
            saver.restore(sess, latest)
        while(True):
            dfcnn_speech(sess, am_model, lm_model, file)

    if type == 'transformer':
        hparams = TransformerHparams()
        parser = hparams.parser
        hp = parser.parse_args()
        hp.is_training = False
        train_data = prepare_data('train', hp, shuffle=True, length=None)

        model = Transformer(hp)
        with model.graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=model.graph) as sess:
            latest = tf.train.latest_checkpoint(Const.TransformerFolder)
            saver.restore(sess, latest)
        while(True):
            transformer_speech(sess, model, train_data, file)


if __name__ == '__main__':
    recognition('dfcnn')
