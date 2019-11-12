import sys

sys.path.insert(0, './module/')
import numpy as np
import tensorflow as tf
from activation import *

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32



def kpn_subnet(x, flg, regular):
    x_shape = [None, 424, 512, 9]
    y_shape = [None, 424, 512, 1 * 1 * 9 * 9 + 9]
    pref = 'kpn_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [ \
        64, \
        64, 64, 64,
        128, 128, 128,
        256, 256, 256,
        512,
    ]
    filter_sizes = [ \
        None,
        7, 5, 5,
        5, 3, 3,
        3, 3, 3,
        3,
    ]
    pool_sizes = [ \
        None,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
    ]
    pool_strides = [ \
        None,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
    ]
    skips = [ \
        False,
        False, False, True,
        False, False, True,
        False, False, True,
        False,
    ]
    filter_sizes_skips = [ \
        3,
        3, 3, 3,
        3, 3, 3,
        3, 3, 3,
        3,
    ]

    n_output = y_shape[-1]
    n_filters_mix = [n_output, n_output, n_output, n_output]
    filter_sizes_mix = [3, 3, 3, 3]

    # initializer
    min_init = -1
    max_init = 1

    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(1, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # convolution
        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            # activation=leaky_relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )
        current_input = tf.layers.batch_normalization(current_input, training=train_ae, name=name + 'BN')
        current_input = leaky_relu(current_input)
        conv.append(current_input)
        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            pool.append(conv[-1])
        else:
            pool.append( \
                tf.layers.max_pooling2d( \
                    inputs=conv[-1],
                    pool_size=[pool_sizes[i], pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref + "pool_" + str(i)
                )
            )
        current_input = pool[-1]

    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range(len(n_filters) - 1, 0, -1):
        name = pref + "upsample_" + str(i - 1)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # upsampling
        current_input = tf.layers.conv2d_transpose( \
            inputs=current_input,
            filters=n_filters[i - 1],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            # activation=leaky_relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )

        current_input = tf.layers.batch_normalization(current_input, training=train_ae, name=name + 'BN')
        current_input = leaky_relu(current_input)

        # skip connection
        if skips[i - 1] == True:
            name = pref + "skip_conv_" + str(i - 1)

            # define the initializer
            if name + '_bias' in inits:
                bias_init = eval(name + '_bias()')
            else:
                bias_init = tf.zeros_initializer()
            if name + '_kernel' in inits:
                kernel_init = eval(name + '_kernel()')
            else:
                kernel_init = None

            tmp = [current_input]
            tmp.append(pool[i - 1])
            current_input = tf.concat(tmp, -1)
            current_input = tf.layers.conv2d( \
                inputs=current_input,
                filters=n_filters[i - 1] * 2,
                kernel_size=[filter_sizes_skips[i - 1], filter_sizes_skips[i - 1]],
                padding="same",
                # activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
            current_input = tf.layers.batch_normalization(current_input, training=train_ae, name=name + 'BN')
            current_input = leaky_relu(current_input)
        upsamp.append(current_input)

    # mix
    mix = []
    for i in range(1, len(n_filters_mix)):
        name = pref + "mix_conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        if i == (len(n_filters_mix) - 1):
            activation = None
        else:
            activation = leaky_relu

        # convolution

        mix.append( \
            tf.layers.conv2d( \
                inputs=current_input,
                filters=n_filters_mix[i],
                kernel_size=[filter_sizes_mix[i], filter_sizes_mix[i]],
                padding="same",
                activation=activation,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        current_input = mix[-1]

    ae_outputs = tf.identity(current_input, name="ae_output")
    return ae_outputs

def kpn(x, flg, regular):
    # inputs 9 channel raw measurements, float32
    # outputs 9 channel raw measurements, float32
    x_shape = [-1, 384, 512, 9]
    y_shape = [-1, 384, 512, 9]

    output = kpn(x, flg)
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
