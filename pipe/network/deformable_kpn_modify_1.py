import sys

sys.path.insert(0, './module/')
import tensorflow as tf
from dataset import *
from activation import *

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32

def dof_subnet(inputs, flg, regular, subnet_num):
    pref = 'dof_subnet_' + str(subnet_num) + '_'
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

    n_filters_mix = [9, 9]
    filter_sizes_mix = [1, 1]
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

    dof_output = tf.identity(current_input, name='dof_output')
    return dof_output

def weight_subnet(inputs, flg, regular, subnet_num):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
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

    n_filters_mix = [9, 9]
    filter_sizes_mix = [3, 3]
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

    weights = tf.identity(current_input, name='wt_output')
    return weights

def offset_subnet(inputs, flg, regular, subnet_num):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
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

    n_filters_mix = [18, 18]
    filter_sizes_mix = [3, 3]
    gain_offset = []
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
            activation = None
        else:
            activation = relu

        # convolution
        gain_offset.append( \
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
        current_input = gain_offset[-1]

    offset = tf.identity(gain_offset[-1], name='offset_output')
    return offset


def deformable_subnet(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'deformable_subnet_'

    # whether to train flag
    train_ae = flg
    N = 9

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
        16, 16,
        32, 32,
        32, 32,
        64, 64,
        64,
        128,
    ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
        3, 3,
        3, 3,
        3,
        3,
    ]
    pool_sizes = [ \
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2,
        1,
    ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2,
        1,
    ]
    skips = [ \
        False, False,
        True, False,
        True, False,
        True, False,
        True, False,
        True,
        False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    offset_pyramid = []
    weight_pyramid = []
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
    for i in range((len(n_filters) - 1) - 1, (0 - 1), -1):
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
        # current_input = tf.layers.batch_normalization(
        #     inputs=current_input,
        #     training=train_ae,
        #     name=pref + "upsamp_BN_" + str(i))
        # skip connection
        if skips[i] == True:
            current_input = tf.concat([current_input, pool[i]], axis=-1)
        upsamp.append(current_input)

        if (i == (len(n_filters) - 1) - 1) or (i == 0) or (skips[i] == False and skips[i - 1] == True):
            if i == (len(n_filters) - 1) - 1:
                offset_pyramid.append(offset_subnet(pool[-1], flg=train_ae, regular=regular, subnet_num=i + 1))
                weight_pyramid.append(weight_subnet(pool[-1], flg=train_ae, regular=regular, subnet_num=i + 1))
            else:
                offset_pyramid.append(offset_subnet(upsamp[-1], flg=train_ae, regular=regular, subnet_num=i))
                weight_pyramid.append(weight_subnet(upsamp[-1], flg=train_ae, regular=regular, subnet_num=i))

    ####################################################################################################################
    features = tf.identity(upsamp[-1], name='ae_output')
    # offset_pyramid_output = tf.identity(offset_pyramid, name='offset_pyramid_output')
    # weight_pyramid_output = tf.identity(weight_pyramid, name='weight_pyramid_output')
    return features, offset_pyramid, weight_pyramid

def deformable_kpn_modify_1(x, flg, regular, batch_size, deformable_range):
    N = 9
    h_max = x.shape.as_list()[1]
    w_max = x.shape.as_list()[2]
    # depth pyramid
    depth_pyramid = []
    depth_residual_pyramid = []
    depth_residual_scale = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    scale_num = len(depth_residual_scale)

    for i in range(5, -1, -1):
        depth_pyramid.append(
            tf.image.resize_images(x, [tf.cast(h_max / 2**i, dtype=tf.int32), tf.cast(w_max / 2**i, dtype=tf.int32)])
        )

    features, offset_pyramid, weight_pyramid = deformable_subnet(x, flg, regular)

    for i in range(scale_num):
        current_depth = depth_pyramid[i]
        current_offset = offset_pyramid[i]
        current_weight = weight_pyramid[i]
        current_samples, current_coords_h_pos, current_coords_w_pos = bilinear_interpolation(current_depth, current_offset, N, batch_size, deformable_range)
        current_samples = dof_subnet(current_samples, flg, regular, subnet_num=i)

        current_depth_residual = current_samples * current_weight
        current_depth_residual = tf.reduce_sum(current_depth_residual, axis=-1, keep_dims=True)

        current_depth_residual = tf.image.resize_images(current_depth_residual, [h_max, w_max])

        depth_residual_pyramid.append(current_depth_residual * depth_residual_scale[i])

    depth_residual = tf.concat(depth_residual_pyramid, axis=-1)
    depth_residual = tf.reduce_sum(depth_residual, axis=-1, keep_dims=True)
    depth_output = x + depth_residual

    return depth_output, current_offset