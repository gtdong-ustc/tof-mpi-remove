import sys

sys.path.insert(0, './module/')
import tensorflow as tf
from dataset import *
from activation import *
from conv import conv
from dfus_block import dfus_block

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
        96, 96,
        128, 128,
        256, 256,
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
    features = []
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
            current_input = current_input
            if (i == len(n_filters) - 1) or (pool_sizes[i + 1] == 2 and pool_strides[i + 1] == 2):
                features.append(current_input)
        else:
            current_input = tf.layers.max_pooling2d( \
                inputs=current_input,
                pool_size=[pool_sizes[i], pool_sizes[i]],
                strides=pool_strides[i],
                name=pref + "pool_" + str(i)
            )
    return features


def depth_residual_regresssion_subnet(x, flg, regular, subnet_num):
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
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution
        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
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

    depth_coarse = tf.identity(feature[-1], name='depth_coarse_output')
    return depth_coarse


def residual_output_subnet(x, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'residual_output_subnet_' + str(subnet_num) + '_'

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
        1
    ]
    filter_sizes = [
        1
    ]
    pool_sizes = [ \
        1
    ]
    pool_strides = [
        1
    ]
    skips = [ \
        False
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
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
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution
        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
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

    depth_residual_coarse = tf.identity(feature[-1], name='depth_coarse_residual_output')
    return depth_residual_coarse

def unet_subnet(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'unet_subnet_'

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
        # current_input = tf.layers.batch_normalization(
        #     inputs=current_input,
        #     training=train_ae,
        #     name=pref + "upsamp_BN_" + str(i))
        # skip connection
        if skips[i] == False and skips[i - 1] == True:
            current_input = tf.concat([current_input, pool[i + 1]], axis=-1)
    ####################################################################################################################
    features = tf.identity(upsamp[-1], name='ae_output')
    return features

def depth_output_subnet(inputs, flg, regular, kernel_size):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
    pref = 'depth_output_subnet_'

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

    n_filters_mix = [kernel_size ** 2]
    filter_sizes_mix = [1]
    mix = []
    for i in range(len(n_filters_mix)):
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
    return current_input

def dear_kpn(x, flg, regular):

    kernel_size = 3
    features = unet_subnet(x, flg, regular)
    weights = depth_output_subnet(features, flg, regular, kernel_size=kernel_size)
    weights = weights / tf.reduce_sum(tf.abs(weights) + 1e-6, axis=-1, keep_dims=True)
    column = im2col(x, kernel_size=kernel_size)
    current_output = tf.reduce_sum(column * weights, axis=-1, keep_dims=True)
    depth_output = tf.identity(current_output, name='depth_output')

    return depth_output

def sample_pyramid_add_kpn_FiveLevel(x, flg, regular, batch_size, deformable_range):
    depth_residual = []
    depth_residual_input = []
    h_max = tf.shape(x)[1]
    w_max = tf.shape(x)[2]

    depth = tf.expand_dims(x[:, :, :, 0], axis=-1)
    depth_and_amplitude = x[:, :, :, 0:2]
    rgb = x[:, :, :, 2:5]

    features = feature_extractor_subnet(depth_and_amplitude, flg, regular)

    for i in range(1, len(features) + 1):

        if i == 1:
            inputs = features[len(features) - i]
        else:
            feature_input = features[len(features) - i]
            h_max_low_scale = tf.shape(feature_input)[1]
            w_max_low_scale = tf.shape(feature_input)[2]
            depth_coarse_input = tf.image.resize_bicubic(depth_residual[-1], size=(h_max_low_scale, w_max_low_scale),
                                                         align_corners=True)
            inputs = tf.concat([feature_input, depth_coarse_input], axis=-1)
        current_depth_residual = depth_residual_regresssion_subnet(inputs, flg, regular, subnet_num=i)
        depth_residual.append(current_depth_residual)

        current_depth_residual_input = tf.image.resize_bicubic(current_depth_residual, size=(h_max, w_max),
                                                               align_corners=True)
        depth_residual_input.append(current_depth_residual_input)

    depth_coarse_residual_input = tf.concat(depth_residual_input, axis=-1)
    final_depth_residual_output = residual_output_subnet(depth_coarse_residual_input, flg, regular, subnet_num=0)

    current_final_depth_output = depth + final_depth_residual_output
    final_depth_output = dear_kpn(current_final_depth_output, flg, regular)
    depth_residual_input.append(final_depth_residual_output)
    depth_residual_input.append(final_depth_output - current_final_depth_output)
    depth_residual_input.append(final_depth_output - current_final_depth_output)
    return final_depth_output, depth_residual_input