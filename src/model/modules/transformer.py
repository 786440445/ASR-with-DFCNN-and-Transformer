import tensorflow as tf
from keras.layers import Dense, Conv1D, Softmax


def layer_norm(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """
    LayerNorm
    :param inputs:
    :param epsilon:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        # 取最后一个维度
        params_shape = inputs_shape[-1:]
        # 计算均值和方差
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="embedding", reuse=None):
    """
    向量的embedding
    :param inputs:
    :param vocab_size: 语料库大小
    :param num_units:
    :param zero_pad: 是否补0
    :param scale:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    if type in ("k", "key", "keys"):
        # 遮掩Key中为0的信息
        # Key Masking (-1,0,1), 当x<0,=0,>0时
        # 如果最后一个维度加起来为0，表示该长度上<MaxLength是没有拼音的。需要mask
        # 加起来不等于0，表示该长度上有拼音
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(inputs) * (-2 ** 32 + 1)
        # [80, 1164, 1164]内存爆炸
        outputs = tf.where(tf.equal(key_masks, 0), paddings, inputs)  # (h*N, T_q, T_k)

    elif type in ("q", "query", "querys"):
        # 遮掩Q中为0的信息
        # query中有信息的部分为1，没有信息的部分为0
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs = inputs * query_masks  # broadcasting. (h*N, T_q, T_k)

    elif type in ("f", "future", "right"):
        # 遮掩未来的信息, 实现一个三角矩阵，对未来的信息进行mask
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (h*N, T_q, T_k)
    else:
        print("Check of upi entered type correctly !")
    return outputs


def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0, is_training=True, scope="scaled_dot_product_attention"):
    # Multiplication Q乘以K的转置
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale 缩放降低维度 除以d(k)的平方根
    outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

    outputs = mask(outputs, Q, K, type='key')

    # 遮掩未来的信息
    if causality:
        outputs = mask(outputs, Q, K, type="future")

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    attention = tf.transpose(outputs, [0, 2, 1])
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # Query Masking
    outputs = mask(outputs, Q, K, type="query")
    # Dropouts
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum 加权平均
    outputs = tf.matmul(outputs, V)  # ( h*N, T_q, C/h)
    return outputs


def multihead_attention(queries, keys,
                        d_model=None, num_heads=8, dropout_rate=0,
                        is_training=True, causality=False, scope="multihead_attention", reuse=None):
    """
    多头注意力机制
    :param emb:
    :param queries: Q
    :param keys: K
    :param num_units: 层数
    :param num_heads: head个数, 通常为8个
    :param dropout_rate:
    :param is_training:
    :param causality:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        if d_model is None:
            d_model = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_k, C)

        # Split and concat 从第三个维度上划分成多头的QKV
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, is_training)

        # Restore shape 多头机制结合
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection 参差链接，将query加起来。
        outputs += queries
        # Normalize layerNorm
        outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs, num_units, scope="positionwise_ffnn", reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        # 1维卷积 构成了全连接
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection 残差+LN
        outputs += inputs
        # Normalize
        outputs = layer_norm(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    """
    平滑
    :param inputs:
    :param epsilon:
    :return:
    """
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)
