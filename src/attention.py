import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # One fully connected layer with non-linear activation for each of the hidden states;
    #  the shape of `v` is (B*T,A), where A=attention_size
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.expand_dims(b_omega, 0))
    # For each of the B*T hidden states its vector of size A from `v` is reduced with `u` vector
    vu = tf.matmul(v, tf.expand_dims(u_omega, -1))   # (B*T, 1) shape
    vu = tf.reshape(vu, tf.shape(inputs)[:2])        # (B,T) shape
    alphas = tf.nn.softmax(vu)                       # (B,T) shape also

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
