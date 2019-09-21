from src.model.modules.transformer import *
from src.utils import hanzi_vocab_size
from src.const import Const


class Transformer():
    def __init__(self, arg):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = arg.is_training
            self.d_model = arg.d_model
            self.label_vocab_size = hanzi_vocab_size
            self.num_heads = arg.num_heads
            self.num_blocks = arg.num_blocks
            self.position_max_length = arg.position_max_length
            self.lr = arg.lr
            self.dropout_rate = arg.dropout_rate
            self.feature_dim = arg.feature_dim
            self.concat = arg.concat
            # input
            # x.shape = [N, L, 80]
            self.x = tf.placeholder(tf.float32, shape=(None, None, self.feature_dim * self.concat))
            # y.shape = [N, Ls]
            self.y = tf.placeholder(tf.int32, shape=(None, None))
            self.target = tf.placeholder(tf.int32, shape=(None, None))
            self.memory = self.encoder()
            self.logits, self.preds = self.decoder(self.memory)
            self.istarget = tf.cast(tf.not_equal(self.target, Const.PAD), tf.float32)
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('transformer_acc', self.acc)

            if self.is_training:
                # Loss label平滑,pad target and logits loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.target, depth=self.label_vocab_size))
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

    def encoder(self):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            seq_len = tf.shape(self.x)[1]
            enc = tf.layers.dense(self.x, 512, kernel_initializer='glorot_normal')
            position_embed = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.position_max_length, num_units=self.d_model, zero_pad=False, scale=False, scope="enc_pe")
            enc = enc + tf.cast(position_embed[:, :seq_len, :], dtype=self.x.dtype)
            # Dropout
            enc = tf.layers.dropout(enc, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              d_model=self.d_model,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training,
                                              causality=False)

                    # Feed Forward
                    enc = feedforward(enc, num_units=[4 * self.d_model, self.d_model])

        return enc

    def decoder(self, memory):
        seq_len = tf.shape(self.y)[1]
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            emb = embedding(self.y, vocab_size=self.label_vocab_size, num_units=self.d_model, scale=True,
                            scope="enc_embed")
            position_emb = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.y)[1]), 0), [tf.shape(self.y)[0], 1]),
                vocab_size=self.position_max_length, num_units=self.d_model, zero_pad=False, scale=False, scope="enc_pe")
            dec = emb + tf.cast(position_emb[:, :seq_len, :], self.x.dtype)
            # Dropout
            dec = tf.layers.dropout(dec,
                                    rate=self.dropout_rate,
                                    training=tf.convert_to_tensor(self.is_training))

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              d_model=self.d_model,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training,
                                              causality=True)

                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              d_model=self.d_model,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training,
                                              causality=False)

                    dec = feedforward(dec, num_units=[4 * self.d_model, self.d_model])

                logits = tf.layers.dense(dec, self.label_vocab_size)
                preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return logits, preds




