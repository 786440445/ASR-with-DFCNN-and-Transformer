import os
import warnings
import sys
import tensorflow as tf

sys.path.append('../')
from src.const import Const
from src.model.transformer import Transformer
from src.train import prepare_data
from src.hparams import TransformerHparams

warnings.filterwarnings('ignore')


# transformer训练Seq2Seq ASR
def transformer_train(args, train_data):
    model = Transformer(args)
    epochs = args.epochs
    batch_num = len(train_data.path_lst) // train_data.batch_size
    with model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=50)
    with tf.Session(graph=model.graph) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        if os.path.exists(Const.TransformerFolder):
            print('loading transformer model...')
            latest = tf.train.latest_checkpoint(Const.TransformerFolder)
            if latest != None:
                add_num = int(latest.split('_')[-2])
                saver.restore(sess, latest)
        writer = tf.summary.FileWriter(Const.TransformerTensorboard, tf.get_default_graph())
        generator = train_data.get_transformer_batch()
        for k in range(epochs):
            total_loss = 0
            # 输入的数据是拼音数据
            # 输出的是文字数据
            for i in range(batch_num):
                data = next(generator)
                X = data["the_inputs"]
                Y = data["the_labels"]
                y_target = data["ground_truth"]
                feed = {model.x: X, model.y: Y, model.target: y_target}
                cost, _ = sess.run([model.mean_loss, model.train_op], feed_dict=feed)
                total_loss += cost
                if i % 10 == 0:
                    print("epoch: %d step: %d/%d  train loss=6%f" % (k + 1, i, batch_num, cost))
                    if i % 5000 == 0:
                        rs = sess.run(merged, feed_dict=feed)
                        writer.add_summary(rs, k * batch_num + i)
            print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
            saver.save(sess, Const.TransformerFolder + 'transformer_model_%d_%.3f.ckpt' % (k + 1 + add_num, total_loss / batch_num))
        writer.close()
    pass


# 直接构建transformer实现语音转汉字过程
def main():
    hparams = TransformerHparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # 数据准备工作
    train_data = prepare_data('train', hp, shuffle=True, length=None)
    transformer_train(hp, train_data)


if __name__ == '__main__':
    main()