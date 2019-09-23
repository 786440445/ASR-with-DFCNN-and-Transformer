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
            self.input_label = tf.placeholder(tf.int32, shape=(None, None))
            self.target = tf.placeholder(tf.int32, shape=(None, None))
            self.memory = self.encoder()
            self.logits, self.preds = self.decoder(self.memory)
            self.istarget = tf.cast(tf.not_equal(self.target, 0), tf.float32)
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.target)) * self.istarget) / (
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
        seq_len = tf.shape(self.input_label)[1]
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            emb = embedding(self.input_label, vocab_size=self.label_vocab_size, num_units=self.d_model, scale=True,
                            scope="enc_embed")
            position_emb = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_label)[1]), 0), [tf.shape(self.input_label)[0], 1]),
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


    def beam_search(self, encoder_output, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._hp.test.beam_size, tf.shape(encoder_output)[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, feat_len, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        # [batch_size * beam_size, feat_len, hidden_units]
        encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
        bias = tf.zeros_like(scores, dtype=tf.bool)  # 是否结束的标识位
        # 缓存的历史结果，[batch_size * beam_size, 0, num_blocks , hidden_units ]
        cache = tf.zeros([batch_size * beam_size, 0, self._hp.num_blocks, self._hp.hidden_units])

        def step(i, bias, preds, scores, cache):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            # [batch_size * beam_size, step, hidden_size]
            decoder_output = self.decoder(encoder_output, preds, is_training=False, reuse=reuse)
            last_preds, last_k_preds, last_k_scores = self.test_output(decoder_output, reuse=reuse)

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)

            # Update scores.
            scores = scores[:, None] + last_k_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]

            # Pruning.
            scores, k_indices = tf.nn.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
            base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]

            # Update predictions.
            last_k_preds = tf.gather(tf.reshape(last_k_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(preds, indices=k_indices / beam_size)
            # cache = tf.gather(cache, indices=k_indices / beam_size)
            preds = tf.concat((preds, last_k_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, cache

        def not_finished(i, bias, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._hp.test.max_target_length])
                )
            )

        i, bias, preds, scores, cache = tf.while_loop(cond=not_finished,
                                                      body=step,
                                                      loop_vars=[0, bias, preds, scores, cache],
                                                      shape_invariants=[
                                                          tf.TensorShape([]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None, None, None])],
                                                      back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        lengths = tf.reduce_sum(tf.to_float(tf.not_equal(preds, 3)), axis=-1)  # [batch_size, beam_size]
        lp = tf.pow((5 + lengths) / (5 + 1), self._hp.test.lp_alpha)  # Length penalty
        scores /= lp  # following GNMT
        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds


