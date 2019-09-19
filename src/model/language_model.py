from src.model.modules.transformer import *
from src.const import Const
from src.utils import pinyin_vocab_size
from src.utils import hanzi_vocab_size


class Language_Model():
    def __init__(self, arg):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = arg.is_training
            self.hidden_units = arg.hidden_units
            self.input_vocab_size = pinyin_vocab_size
            self.label_vocab_size = hanzi_vocab_size
            self.num_heads = arg.num_heads
            self.num_blocks = arg.num_blocks
            self.position_max_length = arg.position_max_length
            self.lr = arg.lr
            self.dropout_rate = arg.dropout_rate

            # input
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.int32, shape=(None, None))
            # embedding
            # X的embedding + position的embedding
            self.emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                 scope="enc_embed")
            self.enc = self.emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.position_max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")

            # Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   d_model=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False)

                    # Feed Forward
                    self.outputs = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, Const.PAD))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if self.is_training:
                # Loss label平滑
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
                # loss计算，预测结果与平滑值的交叉熵
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                # 平均loss
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()