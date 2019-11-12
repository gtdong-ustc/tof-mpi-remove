import sys
sys.path.insert(0, './module/')
import tensorflow as tf
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32
def doformable_subnet_raw(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """
    y_shape = [None, 424, 512, 1 * 1 * 3 * 2 * 9]
    pref = 'deformable_subnet_raw_'

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
        conv.append( \
            tf.layers.conv2d( \
                inputs=current_input,
                filters=n_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
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
            activation=leaky_relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )

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
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        upsamp.append(current_input)

    ###conv out
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
    offsets = current_input
    features = tf.identity(upsamp[-1], name='ae_output')
    offsets = tf.identity(offsets, name="offsets_output")
    return features, offsets
def weight_subnet_raw(inputs, flg, regular):  ## x (B,H,W,9), features:(B,H,W,64), samples:(B,H,W,81)
    pref = 'weight_subnet_'

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

    # change space
    # inputs = tf.concat([x, features, samples], axis=-1)
    wt_inputs = tf.identity(inputs, name='wt_inputs')
    # prepare input
    current_input = tf.identity(wt_inputs, name="current_wt_input")
    ### weight_conv
    name = pref + "wt_conv_"
    # define the initializer
    if name + '_bias' in inits:
        bias_init = eval(name + '_bias()')
    else:
        bias_init = tf.zeros_initializer()
    if name + '_kernel' in inits:
        kernel_init = eval(name + '_kernel()')
    else:
        kernel_init = None
    current_input = tf.layers.conv2d( \
        inputs=current_input,
        filters=81,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=leaky_relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(0),
    )

    current_input = tf.layers.conv2d( \
        inputs=current_input,
        filters=81,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        # activation=relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(1),
    )
    weights = tf.identity(current_input, name='wt_output')
    return weights

def deformable_kpn_raw(x, flg, regular, batch_size, deformable_range):
    """
    :param x:
    :param flg:
    :param regular:
    :param batch_size:
    :param range:
    :return:
    """

    N = 9
    batch_size = batch_size
    x_list = tf.split(x, N, axis=-1)
    h_max = x.shape.as_list()[1]
    w_max = x.shape.as_list()[2]
    samples_set = []
    coords_h_pos_set = []
    coords_w_pos_set = []
    features, offsets = doformable_subnet_raw(x, flg, regular)
    offsets = tf.reshape(offsets, shape=[batch_size, h_max, w_max, 3, 2 * N])

    for i in range(3):
        offsets_temp = offsets[:, :, :, i, :]
        # print(offsets_temp)
        for j in range(3):
            samples, coords_h_pos, coords_w_pos = bilinear_interpolation(x_list[i * 3 + j], offsets_temp, N, batch_size,
                                                                         deformable_range)
            samples_set.append(samples)
            coords_h_pos_set.append(coords_h_pos)
            coords_w_pos_set.append(coords_w_pos)
    samples = tf.concat(samples_set, axis=-1)
    inputs = tf.concat([x, features, samples], axis=-1)
    weights = weight_subnet_raw(inputs, flg, regular)
    output = tf.reshape(samples * weights, shape=[batch_size, h_max, w_max, 9, 9])
    output = tf.reduce_sum(output, axis=-1)
    output = x + output
    return output, offsets