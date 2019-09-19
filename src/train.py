import keras
import os
import warnings
import sys
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('../')
from src.const import Const
from src.model.cnn_ctc import CNNCTCModel
from src.data import GetData
from src.model.language_model import Language_Model
from src.generator import DataGenerator
from src.hparams import DataHparams, AmHparams, LmHparams

warnings.filterwarnings('ignore')


def prepare_data(type, shuffle=True, length=None):
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
    hp = parser.parse_args()

    hp.data_type = type
    hp.shuffle = shuffle
    hp.data_length = length
    data = GetData(hp)
    return data


def acoustic_model(train_data, dev_data):
    """
    声学模型
    :param train_data: 训练数据集合
    :param dev_data: 验证数据集合
    :return:
    """
    hparams = AmHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    epochs = hp.epochs
    model = CNNCTCModel(hp)

    save_step = len(train_data.path_lst) // train_data.batch_size
    latest = tf.train.latest_checkpoint(Const.AmModelFolder)
    select_model = 'model_05-7.64'
    if os.path.exists(Const.AmModelFolder + select_model + '.hdf5'):
        print('load acoustic model...')
        model.load_model(select_model)

    generator = DataGenerator(train_data)
    # 获取一个batch数据
    dev_generator = DataGenerator(dev_data)

    ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
    cpCallBack = ModelCheckpoint(os.path.join(Const.AmModelFolder, ckpt), verbose=1, save_best_only=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=Const.AmModelTensorBoard, histogram_freq=0, write_graph=True,
                                             write_images=True, update_freq='epoch')
    model.ctc_model.fit_generator(generator,
                                  steps_per_epoch=save_step,
                                  validation_data=dev_generator,
                                  validation_steps=20,
                                  epochs=epochs,
                                  workers=10,
                                  use_multiprocessing=True,
                                  callbacks=[cpCallBack,
                                             tbCallBack]
                                  )
    pass


def language_model(train_data):
    """
    语言模型
    :param train_data: 训练数据
    :return:
    """
    hparams = LmHparams()
    parser = hparams.parser
    hp = parser.parse_args()
    epochs = hp.epochs
    lm_model = Language_Model(hp)

    batch_num = len(train_data.path_lst) // train_data.batch_size
    with lm_model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(graph=lm_model.graph) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        if os.path.exists(Const.LmModelFolder):
            print('loading language model...')
            latest = tf.train.latest_checkpoint(Const.LmModelFolder)
            if latest != None:
                add_num = int(latest.split('_')[-2])
                saver.restore(sess, latest)
        writer = tf.summary.FileWriter(Const.LmModelTensorboard, tf.get_default_graph())
        for k in range(epochs):
            total_loss = 0
            batch = train_data.get_lm_batch()
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                cost, _ = sess.run([lm_model.mean_loss, lm_model.train_op], feed_dict=feed)
                total_loss += cost
                if i % 10 == 0:
                    print("epoch: %d step: %d/%d  train loss=6%f" % (k+1, i, batch_num, cost))
                    if i % 5000 == 0:
                        rs = sess.run(merged, feed_dict=feed)
                        writer.add_summary(rs, k * batch_num + i)
            print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
            saver.save(sess, Const.LmModelFolder + 'model_%d_%.3f.ckpt' % (k + 1 + add_num, total_loss / batch_num))
        writer.close()
    pass


def main():
    train_data = prepare_data('train', shuffle=True, length=None)
    dev_data = prepare_data('dev', shuffle=True, length=None)
    print('//-----------------------start acoustic model-----------------------//')
    # acoustic_model(train_data, dev_data)
    print('//-----------------------start language model-----------------------//')
    language_model(train_data)


if __name__ == '__main__':
    main()