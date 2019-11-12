import sys

sys.path.insert(0, './module/')
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32

def kpn_subnet_raw(x, flg, regular, batch_size, deformable_range):
    output = kpn(x, flg, regular)
    biass = output[:, :, :, -9::]
    kers = output[:, :, :, 0:-9]
    kers = tf.reshape(kers, [-1, tf.shape(x)[1], tf.shape(x)[2], 1 * 1 * 9, 9])

    #
    x_new = []
    for i in range(9):
        ker = kers[:, :, :, :, i]
        bias = biass[:, :, :, i]
        x_new.append(tf.reduce_sum(ker * x, -1))
    x_new = tf.stack(x_new, -1)

    return x_new

def kpn_raw(x, flg, regular, batch_size, deformable_range):
    # inputs 9 channel raw measurements, float32
    # outputs 9 channel raw measurements, float32
    x_shape = [-1, 384, 512, 9]
    y_shape = [-1, 384, 512, 9]

    output = kpn_subnet_raw(x, flg)
    biass = output[:,:,:,-9::]
    kers = output[:,:,:,0:-9]
    kers = tf.reshape(kers,[-1, tf.shape(x)[1], tf.shape(x)[2], 1*1*9, 9])

    #
    x_new = []
    for i in range(9):
        ker = kers[:,:,:,:,i]
        bias = biass[:,:,:,i]
        # x_new.append(tf.reduce_sum(ker * x,-1)+bias)
        x_new.append(tf.reduce_sum(ker * x,-1))
    x_new = tf.stack(x_new, -1)

    return x_new, output