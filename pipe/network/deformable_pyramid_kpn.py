import sys

sys.path.insert(0, './module/')
import tensorflow as tf
from dataset import *
from activation import *

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32


def feature_extractor_subnet(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'feature_extractor_subnet_'

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
    n_filters = [
        16, 16,
        32, 32,
        64, 64,
        96, 96,
        128, 128,
    ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
        3, 3,
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    skips = [ \
        False, False,
        True, False,
        True, False,
        True, False,
        True, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: feature extractor
    feature = []
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
                activation=relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            feature.append(current_input)
        else:
            feature.append(
                tf.layers.max_pooling2d( \
                    inputs=current_input,
                    pool_size=[pool_sizes[i], pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref + "pool_" + str(i)
                )
            )
        current_input = feature[-1]
    return feature

def depth_regresssion_subnet(x, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'depth_regression_subnet_' + str(subnet_num) + '_'

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
    n_filters = [
        128, 96,
        64, 32,
        16, 1,
    ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
        1, 1,
        1, 1,
    ]
    pool_strides = [
        1, 1,
        1, 1,
        1, 1,
    ]
    skips = [ \
        False, False,
        False, False,
        False, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
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
            activation=relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            feature.append(current_input)
        else:
            feature.append(
                tf.layers.max_pooling2d( \
                    inputs=current_input,
                    pool_size=[pool_sizes[i], pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref + "pool_" + str(i)
                )
            )
        current_input = feature[-1]
    return x

def unet_subnet(x, flg, regular, N, subnet_num):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'unet_subnet_' + str(subnet_num) + '_'

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
    n_filters = [
        16, 16,
        32, 32,
        64, 64,
        128, 128,
    ]
    filter_sizes = [
        7, 7,
        5, 5,
        5, 5,
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    skips = [ \
        False, False,
        True, False,
        True, False,
        True, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(0, len(n_filters)):
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
    ####################################################################################################################
    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range((len(n_filters) - 1) - 1, 0, -1):
        name = pref + "upsample_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        ## change the kernel size in upsample process
        if skips[i] == False and skips[i + 1] == True:
            filter_sizes[i] = 4
        # upsampling
        current_input = tf.layers.conv2d_transpose( \
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            activation=relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )
        upsamp.append(current_input)
        current_input = tf.layers.batch_normalization(
            inputs=current_input,
            training=train_ae,
            name=pref + "upsamp_bn_" + str(i))
        # skip connection
        if skips[i] == False and skips[i - 1] == True:
            current_input = tf.concat([current_input, pool[i + 1]], axis=-1)
    ####################################################################################################################
    features = tf.identity(upsamp[-1], name='ae_output')
    return features

def weight_subnet(inputs, flg, regular, N, subnet_num):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
    pref = 'weight_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg
    current_input = inputs
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

    n_filters_mix = [N]
    filter_sizes_mix = [1]
    mix = []
    for i in range(1, len(n_filters_mix)):
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

        if i == (len(n_filters_mix) - 1):
            activation = sigmoid
        else:
            activation = relu

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
    weights = tf.identity(current_input, name='wt_output')
    return weights

def offset_subnet(inputs, flg, regular, N, subnet_num):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
    pref = 'offset_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg
    current_input = inputs
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

    n_filters_mix = [N * 2]
    filter_sizes_mix = [1]
    mix = []
    for i in range(1, len(n_filters_mix)):
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

        if i == (len(n_filters_mix) - 1):
            activation = None
        else:
            activation = relu

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

    offsets = tf.identity(current_input, name='offset_output')
    return offsets
def deformable_kpn_rgb_subnet(x, flg, regular, batch_size, deformable_range, subnet_num):
    N = 9
    features = unet_subnet(x, flg, regular, N, subnet_num=subnet_num)
    offsets = offset_subnet(x, flg, regular, N, subnet_num=subnet_num)
    weights = weight_subnet(features, flg, regular, N, subnet_num=subnet_num)
    return offsets, weights

def deformable_kpn_depth_subnet(x, rgb_offsets, rgb_weights, flg, regular, batch_size, deformable_range, subnet_num):
    N = 9
    features = unet_subnet(x, flg, regular, N, subnet_num=subnet_num)
    offsets = offset_subnet(x, flg, regular, N, subnet_num=subnet_num)

    offsets = offsets * rgb_offsets
    samples, coords_h_pos, coords_w_pos = bilinear_interpolation(x, offsets, N, deformable_range)
    weights = weight_subnet(features, flg, regular, N, subnet_num=subnet_num)
    weights = weights * rgb_weights
    weights = weights - tf.reduce_mean(weights, axis=-1, keep_dims=True)
    depth_residual = weights * samples
    depth_output = tf.reduce_sum(depth_residual, axis=-1, keep_dims=True)
    depth_output = x + depth_output

    return depth_output, offsets

def deformable_pyramid_kpn(x, flg, regular, batch_size, deformable_range):

    depth_and_amplitude = x[:,:,:,0:2]
    rgb = x[:,:,:,2:4]

    rgb_offsets, rgb_weights = deformable_kpn_rgb_subnet(rgb, flg, regular, batch_size, deformable_range, subnet_num=0)
    depth_output, offsets = deformable_kpn_depth_subnet(depth_and_amplitude, rgb_offsets, rgb_weights, flg, regular, batch_size, deformable_range, subnet_num=0)
    return depth_output, offsets