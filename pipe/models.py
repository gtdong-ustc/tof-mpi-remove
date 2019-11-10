import sys

sys.path.insert(0, '../sim/')
import numpy as np
import tensorflow as tf
import os, json, glob
import imageio
import matplotlib
import math
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from tof_class import *
import pdb
import pickle
import time
import scipy.misc
from scipy import sparse
import scipy.interpolate
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from kinect_spec import *
from dataset import *
import cv2
from numpy import linalg as LA

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32


# tof_cam = kinect_real_tf()


def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.nn.sigmoid(x) - 0.5


def samples_visualizaiton(x, offsets, h_selected_pos, w_selected_pos, batch_size, N):
    """
     This function is to visualize the sample positon
    :param x: depth map
    :param offsets:  the offsets of kernel
    :param h_pos:  the height positon is to visualize
    :param w_pos:  the weight position is to visualize
    :param batch_size:
    :param N: the kernel size
    :return: the visualization of samples position
    """
    offsets_size = tf.shape(offsets)
    h_w_reshape_size = [offsets_size[0], offsets_size[1], offsets_size[2], N, 2]
    # print(h_w_reshape_size)

    offsets = tf.reshape(offsets, h_w_reshape_size)
    coords_h, coords_w = tf.split(offsets, [1, 1], axis=-1)
    coords_h = tf.squeeze(coords_h, [4])
    coords_w = tf.squeeze(coords_w, [4])
    coords_h = tf.cast(coords_h, dtype=tf.float32)
    coords_w = tf.cast(coords_w, dtype=tf.float32)

    h0 = tf.floor(coords_h)
    h1 = h0 + 1
    w0 = tf.floor(coords_w)
    w1 = w0 + 1

    h_max_idx = x.shape.as_list()[1]
    w_max_idx = x.shape.as_list()[2]

    tensor_h = list(range(384))
    tensor_w = list(range(512))

    w_pos, h_pos = tf.meshgrid(tensor_w, tensor_h)

    w_pos = tf.expand_dims(tf.expand_dims(w_pos, 0), -1)
    h_pos = tf.expand_dims(tf.expand_dims(h_pos, 0), -1)
    w_pos = tf.tile(w_pos, multiples=[batch_size, 1, 1, N])
    h_pos = tf.tile(h_pos, multiples=[batch_size, 1, 1, N])
    w_pos = tf.cast(w_pos, dtype=tf.float32)
    h_pos = tf.cast(h_pos, dtype=tf.float32)

    ih0 = h0 + h_pos
    iw0 = w0 + w_pos

    ih1 = h1 + h_pos
    iw1 = w1 + w_pos

    mask_outside_sum = tf.cast(0 <= ih0, dtype=tf.float32) + tf.cast(ih1 < h_max_idx, dtype=tf.float32) + \
                       tf.cast(0 <= iw0, dtype=tf.float32) + tf.cast(iw1 < w_max_idx, dtype=tf.float32)

    mask_outside = mask_outside_sum > 3.0
    mask_outside = tf.cast(mask_outside, dtype=tf.float32)

    ih0 = ih0 * mask_outside
    iw0 = iw0 * mask_outside
    ih1 = ih1 * mask_outside
    iw1 = iw1 * mask_outside

    tensor_batch = list(range(batch_size))
    tensor_batch = tf.convert_to_tensor(tensor_batch)
    tensor_batch = tf.reshape(tensor_batch, [batch_size, 1, 1, 1])
    tensor_batch = tf.tile(tensor_batch, multiples=[1, h_max_idx, w_max_idx, 9])
    tensor_batch = tf.cast(tensor_batch, dtype=tf.float32)

    samples_visual = tf.constant([batch_size, h_max_idx, w_max_idx])
    fill_ones = tf.ones([batch_size * N], dtype=tf.int32)
    tensor_batch_pos = tensor_batch[:, h_selected_pos, w_selected_pos, :]
    ih0_pos = ih0[:, h_selected_pos, w_selected_pos, :]
    iw0_pos = iw0[:, h_selected_pos, w_selected_pos, :]
    ih1_pos = ih1[:, h_selected_pos, w_selected_pos, :]
    iw1_pos = iw1[:, h_selected_pos, w_selected_pos, :]

    idx00 = tf.stack([tensor_batch_pos, ih0_pos, iw0_pos], axis=-1)
    idx01 = tf.stack([tensor_batch_pos, ih0_pos, iw1_pos], axis=-1)
    idx10 = tf.stack([tensor_batch_pos, ih1_pos, iw0_pos], axis=-1)
    idx11 = tf.stack([tensor_batch_pos, ih1_pos, iw1_pos], axis=-1)

    idx00 = tf.reshape(tf.cast(idx00, dtype=tf.int32), [-1, 3])
    idx01 = tf.reshape(tf.cast(idx01, dtype=tf.int32), [-1, 3])
    idx10 = tf.reshape(tf.cast(idx10, dtype=tf.int32), [-1, 3])
    idx11 = tf.reshape(tf.cast(idx11, dtype=tf.int32), [-1, 3])

    # print(idx00)
    # print(fill_ones)
    # print(samples_visual)

    samples_visual_00 = tf.scatter_nd(idx00, fill_ones, samples_visual)
    samples_visual_01 = tf.scatter_nd(idx01, fill_ones, samples_visual)
    samples_visual_10 = tf.scatter_nd(idx10, fill_ones, samples_visual)
    samples_visual_11 = tf.scatter_nd(idx11, fill_ones, samples_visual)

    output = tf.add_n([
        samples_visual_00, samples_visual_01,
        samples_visual_10, samples_visual_11
    ])

    output = tf.expand_dims(output, axis=-1)
    output = tf.cast(output, tf.float32)
    # output = x - x * output

    return output


def bilinear_interpolation(input, offsets, N, batch_size, deformable_range):
    """

    :param input:
    :param offsets:
    :param N:
    :param batch_size:
    :return:
    """
    # input_size = tf.shape(input)
    h_max_idx = input.shape.as_list()[1]
    w_max_idx = input.shape.as_list()[2]
    offsets_size = tf.shape(offsets)
    """
    xiugai h_w_reshape_size = [offsets_size[0], offsets_size[1], offsets_size[2], N, 2]
    """
    h_w_reshape_size = [offsets_size[0], offsets_size[1], offsets_size[2], 2, N]

    offsets = tf.reshape(offsets, h_w_reshape_size)
    coords_h, coords_w = tf.split(offsets, [1, 1], axis=3)
    coords_h = tf.squeeze(coords_h, [3])
    coords_w = tf.squeeze(coords_w, [3])
    coords_h = tf.cast(coords_h, dtype=tf.float32)
    coords_w = tf.cast(coords_w, dtype=tf.float32)

    h0 = tf.floor(coords_h)
    h1 = h0 + 1
    w0 = tf.floor(coords_w)
    w1 = w0 + 1

    ## this may be have some questions
    w_pos, h_pos = tf.meshgrid(list(range(w_max_idx)), list(range(h_max_idx)))

    w_pos = tf.expand_dims(tf.expand_dims(w_pos, 0), -1)
    h_pos = tf.expand_dims(tf.expand_dims(h_pos, 0), -1)
    w_pos = tf.tile(w_pos, multiples=[batch_size, 1, 1, N])
    h_pos = tf.tile(h_pos, multiples=[batch_size, 1, 1, N])
    w_pos = tf.cast(w_pos, dtype=tf.float32)
    h_pos = tf.cast(h_pos, dtype=tf.float32)

    ih0 = h0 + h_pos
    iw0 = w0 + w_pos

    # print('*************************************')
    # print(h0)
    # print(h_pos)

    ih1 = h1 + h_pos
    iw1 = w1 + w_pos

    coords_h_pos = coords_h + h_pos
    coords_w_pos = coords_w + w_pos

    mask_inside_sum = tf.cast(0 <= ih0, dtype=tf.float32) + tf.cast(ih1 <= h_max_idx, dtype=tf.float32) + \
                      tf.cast(0 <= iw0, dtype=tf.float32) + tf.cast(iw1 <= w_max_idx, dtype=tf.float32) + \
                      tf.cast(tf.abs(h1) <= deformable_range, dtype=tf.float32) + tf.cast(
        tf.abs(w1) <= deformable_range, dtype=tf.float32)

    mask_outside = mask_inside_sum < 6.0
    mask_inside = mask_inside_sum > 5.0

    mask_outside = tf.cast(mask_outside, dtype=tf.float32)
    mask_inside = tf.cast(mask_inside, dtype=tf.float32)

    ih0 = ih0 * mask_inside
    iw0 = iw0 * mask_inside
    ih1 = ih1 * mask_inside
    iw1 = iw1 * mask_inside

    # coords_h_pos = coords_h_pos * mask_inside + tensor_original_iw * mask_outside
    # coords_w_pos = coords_w_pos * mask_inside + tensor_original_iw * mask_outside

    tensor_batch = list(range(batch_size))
    tensor_batch = tf.convert_to_tensor(tensor_batch)
    tensor_batch = tf.reshape(tensor_batch, [batch_size, 1, 1, 1])
    tensor_batch = tf.tile(tensor_batch, multiples=[1, h_max_idx, w_max_idx, N])
    tensor_batch = tf.cast(tensor_batch, dtype=tf.float32)

    tensor_channel = tf.zeros(shape=[N], dtype=tf.float32)
    tensor_channel = tf.reshape(tensor_channel, [1, 1, 1, N])
    tensor_channel = tf.tile(tensor_channel, multiples=[batch_size, h_max_idx, w_max_idx, 1])
    tensor_channel = tf.cast(tensor_channel, dtype=tf.float32)

    idx00 = tf.stack([tensor_batch, ih0, iw0, tensor_channel], axis=-1)
    idx01 = tf.stack([tensor_batch, ih0, iw1, tensor_channel], axis=-1)
    idx10 = tf.stack([tensor_batch, ih1, iw0, tensor_channel], axis=-1)
    idx11 = tf.stack([tensor_batch, ih1, iw1, tensor_channel], axis=-1)

    idx00 = tf.reshape(idx00, [-1, 4])
    idx01 = tf.reshape(idx01, [-1, 4])
    idx10 = tf.reshape(idx10, [-1, 4])
    idx11 = tf.reshape(idx11, [-1, 4])

    im00 = tf.gather_nd(input, tf.cast(idx00, dtype=tf.int32))
    im01 = tf.gather_nd(input, tf.cast(idx01, dtype=tf.int32))
    im10 = tf.gather_nd(input, tf.cast(idx10, dtype=tf.int32))
    im11 = tf.gather_nd(input, tf.cast(idx11, dtype=tf.int32))

    im00 = tf.reshape(im00, [batch_size, h_max_idx, w_max_idx, N])
    im01 = tf.reshape(im01, [batch_size, h_max_idx, w_max_idx, N])
    im10 = tf.reshape(im10, [batch_size, h_max_idx, w_max_idx, N])
    im11 = tf.reshape(im11, [batch_size, h_max_idx, w_max_idx, N])

    im00 = tf.cast(im00, dtype=tf.float32)
    im01 = tf.cast(im01, dtype=tf.float32)
    im10 = tf.cast(im10, dtype=tf.float32)
    im11 = tf.cast(im11, dtype=tf.float32)

    wt_w0 = w1 - coords_w
    wt_w1 = coords_w - w0
    wt_h0 = h1 - coords_h
    wt_h1 = coords_h - h0

    w00 = wt_h0 * wt_w0
    w01 = wt_h0 * wt_w1
    w10 = wt_h1 * wt_w0
    w11 = wt_h1 * wt_w1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    output = output * mask_inside
    return output, coords_h_pos, coords_w_pos


def kpn(x, flg, regular):
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


def bottleneck(x, flg, regular, inits, i):
    pref = 'bottleneck_'
    block_filters = [
        64, 64, 256
    ]
    filter_sizes = [
        1, 3, 1
    ]
    skips = [
        False, False, True
    ]

    # change space
    ae_inputs = tf.identity(x, name='ae_' + pref + str(i) + '_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name='bottleneck_' + str(i) + '_input')
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(1, len(block_filters)):
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
            filters=block_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=relu,
            trainable=flg,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )
        current_input = tf.layers.batch_normalization(current_input, trainable=flg, training=flg, momentum=0.9)

        if skips[i] == True:
            shortcut = tf.layers.conv2d(
                inputs=ae_inputs,
                filters=block_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=relu,
                trainable=flg,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
            current_input = current_input + shortcut
        current_input = relu(current_input)
        conv.append(current_input)

        return conv[-1]


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
        64, 64, 64,
        128, 128, 128,
        256, 256, 256,
        512, 512, 512,
        512,
        512,
    ]
    filter_sizes = [
        3, 3, 3,
        3, 3, 3,
        3, 3, 3,
        3, 3, 3,
        3,
        3,
    ]
    pool_sizes = [ \
        1, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
        1,
    ]
    pool_strides = [
        1, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
        1,
    ]
    skips = [ \
        False, False, False,
        True, False, False,
        True, False, False,
        True, False, False,
        True,
        False,
    ]

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
                activation=leaky_relu,
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
        if skips[i - 1] == False and skips[i] == True:
            filters = n_filters[i - 1] * 2
        else:
            filters = n_filters[i - 1]
        current_input = tf.layers.conv2d_transpose( \
            inputs=current_input,
            filters=filters,
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            activation=leaky_relu,
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
        if skips[i - 1] == True:
            # current_input = current_input + pool[i - 1]
            current_input = tf.concat([current_input, pool[i - 1]], axis=-1)
        upsamp.append(current_input)

    ###conv out
    name = pref + "convout_"

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
        inputs=upsamp[-1],
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=leaky_relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(0),
    )

    offsets = tf.layers.conv2d( \
        inputs=current_input,
        filters=18,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        # activation=sigmoid,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(1),
    )
    features = tf.identity(upsamp[-1], name='ae_output')
    offsets = tf.identity(offsets, name="offsets_output")
    return features, offsets


def deformable_half_subnet(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'deformable_half_subnet_'

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
        32, 32, 32,
        32, 32, 32,
        64, 64, 64,
        64, 64, 64,
        128, 128, 128,
        128, 128, 128,
        256,
        256,
    ]
    filter_sizes = [
        5, 5, 5,
        5, 5, 5,
        5, 5, 5,
        5, 5, 5,
        5, 5, 5,
        5, 5, 5,
        5,
        5,
    ]
    pool_sizes = [ \
        1, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
        1,
    ]
    pool_strides = [
        1, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2, 1, 1,
        2,
        1,
    ]
    skips = [ \
        False, False, False,
        True, False, False,
        True, False, False,
        True, False, False,
        True, False, False,
        True, False, False,
        True,
        False,
    ]

    n_filters_mix = [32, 32, 32, 18]
    filter_sizes_mix = [5, 5, 5, 5]

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
        if skips[i - 1] == False and skips[i] == True:
            filters = n_filters[i - 1] * 2
        else:
            filters = n_filters[i - 1]
        current_input = tf.layers.conv2d_transpose( \
            inputs=current_input,
            filters=filters,
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
        if skips[i - 1] == True:
            # current_input = current_input + pool[i - 1]
            current_input = tf.concat([current_input, pool[i - 1]], axis=-1)
        upsamp.append(current_input)

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
    features = tf.identity(upsamp[-1], name='ae_output')
    offsets = tf.identity(current_input, name="offsets_output")
    return features, offsets


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


def weight_subnet(inputs, flg, regular):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
    pref = 'weight_subnet_'

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

    n_filters_mix = [9, 9, 9, 9]
    filter_sizes_mix = [5, 5, 5, 5]
    mix = []
    for i in range(1, len(n_filters_mix)):
        name = pref + "_conv_" + str(i)

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


def dof_subnet(inputs, flg, regular):
    pref = 'dof_subnet_'
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

    n_filters_mix = [9, 9, 9, 9]
    filter_sizes_mix = [1, 1, 1, 1]
    mix = []
    for i in range(1, len(n_filters_mix)):
        name = pref + "_conv_" + str(i)

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

    dof_output = tf.identity(current_input, name='wt_output')
    return dof_output


def coherent_demodulation(inputs, phi, theta):
    return inputs


def unet(x, flg, regular, batch_size, deformable_range, z_multiplier):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'unet_'

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
        16, 16,
        16, 16,
        32, 32,
        32, 32,
        32, 32,
        32
    ]
    filter_sizes = [
        5, 5,
        5, 5,
        5, 5,
        5, 5,
        5, 5,
        5, 5,
        5
    ]
    pool_sizes = [ \
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2
    ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2
    ]
    skips = [ \
        False, True,
        False, True,
        False, True,
        False, True,
        False, True,
        False, True,
        False
    ]

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
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        # conv[-1] = tf.layers.batch_normalization(
        #     inputs = conv[-1],
        #     training = train_ae,
        #     name = pref + "BN_" + str(i))
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
        # current_input = tf.layers.batch_normalization(
        #     inputs=current_input,
        #     training=train_ae,
        #     name=pref + "upsamp_BN_" + str(i))
        # skip connection
        if skips[i - 1] == True:
            current_input = current_input + pool[i - 1]
        upsamp.append(current_input)

    ###conv out
    name = pref + "convout_" + str(i - 1)

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
        inputs=upsamp[-1],
        filters=1,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        activation=leaky_relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name,
    )

    # current_input = tf.squeeze(current_input, [3])
    output = tf.identity(current_input, name="ae_output")
    return output


def kpn_raw(x, flg, regular, batch_size, deformable_range, z_multiplier):
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


def deformable_kpn(x, flg, regular, batch_size, range, z_multiplier):
    N = 9
    batch_size = batch_size
    features, offsets = deformable_subnet(x, flg, regular)

    # offsets = offsets * range

    samples, coords_h_pos, coords_w_pos = bilinear_interpolation(x, offsets, N, batch_size, range)

    inputs = tf.concat([x, features, samples], axis=-1)

    weights = weight_subnet(inputs, flg, regular)

    depth_output = weights * samples
    depth_output = tf.reduce_sum(depth_output, axis=-1, keep_dims=True)
    depth_output = x + depth_output
    # print(depth_output)

    return depth_output, offsets


def deformable_kpn_half(x, flg, regular, batch_size, range, z_multiplier):
    N = 9
    features, offsets = deformable_half_subnet(x, flg, regular)

    samples, coords_h_pos, coords_w_pos = bilinear_interpolation(x, offsets, N, batch_size, range)

    # dof_sample = dof_computer(dist=x, samples=samples, batch_size=batch_size, z_multiplier=z_multiplier, coords_h_pos=coords_h_pos, coords_w_pos=coords_w_pos)

    # inputs = tf.concat([x, features, samples], axis=-1)

    samples = dof_subnet(samples, flg, regular)

    weights = weight_subnet(features, flg, regular)
    weights = weights - tf.reduce_mean(weights)
    depth_output = weights * samples
    depth_output = tf.reduce_sum(depth_output, axis=-1, keep_dims=True)
    depth_output = x + depth_output
    # print(depth_output)

    return depth_output, offsets


def deformable_kpn_raw(x, flg, regular, batch_size, deformable_range, z_multiplier):
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


NETWORK_NAME = {
    'deformable_kpn': deformable_kpn,
    'deformable_kpn_raw': deformable_kpn_raw,
    'deformable_kpn_half': deformable_kpn_half,
    'unet': unet,
    'kpn_raw': kpn_raw,
}

ALL_NETWORKS = dict(NETWORK_NAME)


def get_network(name, x, flg, regular, batch_size, range, z_multiplier):
    if name not in NETWORK_NAME.keys():
        print('Unrecognized network, pick one among: {}'.format(ALL_NETWORKS.keys()))
        raise Exception('Unknown network selected')
    selected_network = ALL_NETWORKS[name]
    return selected_network(x, flg, regular, batch_size, range, z_multiplier)
