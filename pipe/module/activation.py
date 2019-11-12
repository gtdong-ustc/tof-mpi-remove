def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.nn.sigmoid(x) - 0.5

