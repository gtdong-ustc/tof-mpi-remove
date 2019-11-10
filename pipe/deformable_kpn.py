# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
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
import cv2
from numpy import linalg as LA

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

from kinect_init import *
# from testing_MRM_LF2 import data_augment_kinect
# from testing_MRM_LF2 import data_augment

PI = 3.14159265358979323846
flg = False
dtype = tf.float32


def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg

def relu(x):
    return  tf.nn.relu(x)

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
    ae_inputs = tf.identity(x, name='ae_' + pref + str (i) + '_inputs')

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
        filters=18,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=relu,
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
        activation=relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(1),
    )
    features = tf.identity(upsamp[-1], name='ae_output')
    offsets = tf.identity(offsets, name="offsets_output")
    return features, offsets

def weight_subnet(x, features, samples, flg, regular): ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
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
    inputs = tf.concat([x, features, samples], axis=-1)
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
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(0),
    )

    current_input = tf.layers.conv2d( \
        inputs=current_input,
        filters=9,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=relu,
        trainable=train_ae,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name + str(1),
    )
    weights = tf.identity(current_input, name='wt_output')
    return weights

def bilinear_interpolation(input, offsets, N, batch_size):
    input_size = tf.shape(input)
    offsets_size = tf.shape(offsets)
    h_w_reshape_size = [offsets_size[0], offsets_size[1], offsets_size[2], N, 2]

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

    h_max_idx = input.shape.as_list()[1]
    w_max_idx = input.shape.as_list()[2]

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

    ih1 = h1 + h_pos
    iw1 = w1 + w_pos


    mask_outside_sum = tf.cast(0 <= ih0, dtype=tf.float32) + tf.cast(ih1 < h_max_idx, dtype=tf.float32) +\
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

    tensor_channel = list(range(9))
    tensor_channel = tf.convert_to_tensor(tensor_channel)
    tensor_channel = tf.reshape(tensor_channel, [1, 1, 1, 9])
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

    im00 = tf.reshape(im00, [batch_size, h_max_idx, w_max_idx, 9])
    im01 = tf.reshape(im01, [batch_size, h_max_idx, w_max_idx, 9])
    im10 = tf.reshape(im10, [batch_size, h_max_idx, w_max_idx, 9])
    im11 = tf.reshape(im11, [batch_size, h_max_idx, w_max_idx, 9])

    wt_w0 = w1 - coords_w
    wt_w1 = coords_w - w0
    wt_h0 = h1 - coords_h
    wt_h1 = coords_h - h0

    w00 = wt_w0 * wt_h0
    w01 = wt_w0 * wt_h1
    w10 = wt_w1 * wt_h0
    w11 = wt_w1 * wt_h1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    output = output * mask_outside
    return output

def deformable_deeptof(x, flg, regular, batch_size):

    N = 9
    batch_size = batch_size
    features, offsets = deformable_subnet(x, flg, regular)

    samples = bilinear_interpolation(x, offsets, N, batch_size)

    weights = weight_subnet(x, features, samples, flg, regular)

    depth_output = weights * samples
    depth_output = tf.reduce_sum(depth_output, axis=-1)
    depth_output = tf.expand_dims(depth_output, axis=-1)

    print(depth_output)

    return depth_output


def processPixelStage1(m):
    # m is (None,424, 512, 9)
    # the first three is the first frequency
    tmp = []
    tmp.append(processMeasurementTriple(m[:, :, :, 0:3], prms['ab_multiplier_per_frq'][0], trig_table0))
    tmp.append(processMeasurementTriple(m[:, :, :, 3:6], prms['ab_multiplier_per_frq'][1], trig_table1))
    tmp.append(processMeasurementTriple(m[:, :, :, 6:9], prms['ab_multiplier_per_frq'][2], trig_table2))

    m_out = [ \
        tmp[0][:, :, :, 0], tmp[1][:, :, :, 0], tmp[2][:, :, :, 0],
        tmp[0][:, :, :, 1], tmp[1][:, :, :, 1], tmp[2][:, :, :, 1],
        tmp[0][:, :, :, 2], tmp[1][:, :, :, 2], tmp[2][:, :, :, 2],
    ]
    m_out = tf.stack(m_out, -1)

    # return processMeasurementTriple(m[:,:,:,0:3], prms['ab_multiplier_per_frq'][0], trig_table0)
    return m_out


def processPixelStage1_mat(m):
    # if not saturated
    cos_tmp0 = np.stack([trig_table0[:, :, 0], trig_table1[:, :, 0], trig_table2[:, :, 0]], -1)
    cos_tmp1 = np.stack([trig_table0[:, :, 1], trig_table1[:, :, 1], trig_table2[:, :, 1]], -1)
    cos_tmp2 = np.stack([trig_table0[:, :, 2], trig_table1[:, :, 2], trig_table2[:, :, 2]], -1)

    sin_negtmp0 = np.stack([trig_table0[:, :, 3], trig_table1[:, :, 3], trig_table2[:, :, 3]], -1)
    sin_negtmp1 = np.stack([trig_table0[:, :, 4], trig_table1[:, :, 4], trig_table2[:, :, 4]], -1)
    sin_negtmp2 = np.stack([trig_table0[:, :, 5], trig_table1[:, :, 5], trig_table2[:, :, 5]], -1)

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0, 0)
    cos_tmp1 = np.expand_dims(cos_tmp1, 0)
    cos_tmp2 = np.expand_dims(cos_tmp2, 0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0, 0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1, 0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2, 0)

    #
    abMultiplierPerFrq = np.expand_dims(np.expand_dims(np.expand_dims(prms['ab_multiplier_per_frq'], 0), 0), 0)

    ir_image_a = cos_tmp0 * m[:, :, :, 0::3] + cos_tmp1 * m[:, :, :, 1::3] + cos_tmp2 * m[:, :, :, 2::3]
    ir_image_b = sin_negtmp0 * m[:, :, :, 0::3] + sin_negtmp1 * m[:, :, :, 1::3] + sin_negtmp2 * m[:, :, :, 2::3]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq
    ir_amplitude = tf.sqrt(ir_image_a ** 2 + ir_image_b ** 2) * prms['ab_multiplier']

    return ir_image_a, ir_image_b, ir_amplitude


def processMeasurementTriple(m, abMultiplierPerFrq, trig_table):
    # m is (None,424,512,3)
    zmultiplier = tf.constant(z_table, dtype=dtype)

    # judge where saturation happens
    saturated = tf.cast(tf.less(tf.abs(m), 1.0), dtype=dtype)
    saturated = 1 - saturated[:, :, :, 0] * saturated[:, :, :, 1] * saturated[:, :, :, 2]

    # if not saturated
    cos_tmp0 = trig_table[:, :, 0]
    cos_tmp1 = trig_table[:, :, 1]
    cos_tmp2 = trig_table[:, :, 2]

    sin_negtmp0 = trig_table[:, :, 3]
    sin_negtmp1 = trig_table[:, :, 4]
    sin_negtmp2 = trig_table[:, :, 5]

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0, 0)
    cos_tmp1 = np.expand_dims(cos_tmp1, 0)
    cos_tmp2 = np.expand_dims(cos_tmp2, 0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0, 0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1, 0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2, 0)

    ir_image_a = cos_tmp0 * m[:, :, :, 0] + cos_tmp1 * m[:, :, :, 1] + cos_tmp2 * m[:, :, :, 2]
    ir_image_b = sin_negtmp0 * m[:, :, :, 0] + sin_negtmp1 * m[:, :, :, 1] + sin_negtmp2 * m[:, :, :, 2]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq

    ir_amplitude = tf.sqrt(ir_image_a ** 2 + ir_image_b ** 2) * prms['ab_multiplier']

    m_out = tf.stack([ir_image_a, ir_image_b, ir_amplitude], -1)

    # # mask out the saturated pixel
    # zero_mat = tf.zeros(tf.shape(ir_image_a))
    # full_mat = tf.ones(tf.shape(ir_amplitude))
    # m_out_sat = tf.stack([zero_mat, zero_mat, full_mat], -1)
    # saturated = tf.expand_dims(saturated,-1)
    # m_out = saturated * m_out_sat + (1 - saturated) * m_out

    return m_out


def processPixelStage2(ira, irb, iramp):
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude
    ratio = 100
    tmp0 = tf.atan2(ratio * (irb + 1e-10), ratio * (ira + 1e-10))
    flg = tf.cast(tf.less(tmp0, 0.0), dtype)
    tmp0 = flg * (tmp0 + PI * 2) + (1 - flg) * tmp0

    tmp1 = tf.sqrt(ira ** 2 + irb ** 2) * prms['ab_multiplier']

    ir_sum = tf.reduce_sum(tmp1, -1)

    # disable disambiguation
    ir_min = tf.reduce_min(tmp1, -1)

    # phase mask
    phase_msk1 = tf.cast( \
        tf.greater(ir_min, prms['individual_ab_threshold']),
        dtype=dtype
    )
    phase_msk2 = tf.cast( \
        tf.greater(ir_sum, prms['ab_threshold']),
        dtype=dtype
    )
    phase_msk_t = phase_msk1 * phase_msk2

    # compute phase
    t0 = tmp0[:, :, :, 0] / (2.0 * PI) * 3.0
    t1 = tmp0[:, :, :, 1] / (2.0 * PI) * 15.0
    t2 = tmp0[:, :, :, 2] / (2.0 * PI) * 2.0

    t5 = tf.floor((t1 - t0) * 0.3333333 + 0.5) * 3.0 + t0
    t3 = t5 - t2
    t4 = t3 * 2.0

    c1 = tf.cast(tf.greater(t4, -t4), dtype=dtype)
    f1 = c1 * 2.0 + (1 - c1) * (-2.0)
    f2 = c1 * 0.5 + (1 - c1) * (-0.5)
    t3 = t3 * f2
    t3 = (t3 - tf.floor(t3)) * f1

    c2 = tf.cast(tf.less(0.5, tf.abs(t3)), dtype=dtype) * \
         tf.cast(tf.less(tf.abs(t3), 1.5), dtype=dtype)
    t6 = c2 * (t5 + 15.0) + (1 - c2) * t5
    t7 = c2 * (t1 + 15.0) + (1 - c2) * t1
    t8 = (tf.floor((t6 - t2) * 0.5 + 0.5) * 2.0 + t2) * 0.5

    t6 /= 3.0
    t7 /= 15.0

    # transformed phase measurements (they are transformed and divided
    # by the values the original values were multiplied with)
    t9 = t8 + t6 + t7
    t10 = t9 / 3.0  # some avg

    t6 = t6 * 2.0 * PI
    t7 = t7 * 2.0 * PI
    t8 = t8 * 2.0 * PI

    t8_new = t7 * 0.826977 - t8 * 0.110264
    t6_new = t8 * 0.551318 - t6 * 0.826977
    t7_new = t6 * 0.110264 - t7 * 0.551318

    t8 = t8_new
    t6 = t6_new
    t7 = t7_new

    norm = t8 ** 2 + t6 ** 2 + t7 ** 2
    mask = tf.cast(tf.greater(t9, 0.0), dtype)
    t10 = t10

    slope_positive = float(0 < prms['ab_confidence_slope'])

    ir_min_ = tf.reduce_min(tmp1, -1)
    ir_max_ = tf.reduce_max(tmp1, -1)

    ir_x = slope_positive * ir_min_ + (1 - slope_positive) * ir_max_

    ir_x = tf.log(ir_x)
    ir_x = (ir_x * prms['ab_confidence_slope'] * 0.301030 + prms['ab_confidence_offset']) * 3.321928
    ir_x = tf.exp(ir_x)
    ir_x = tf.maximum(prms['min_dealias_confidence'], ir_x)
    ir_x = tf.minimum(prms['max_dealias_confidence'], ir_x)
    ir_x = ir_x ** 2

    mask2 = tf.cast(tf.greater(ir_x, norm), dtype)

    t11 = t10

    mask3 = tf.cast( \
        tf.greater(prms['max_dealias_confidence'] ** 2, norm),
        dtype
    )
    t10 = t10
    phase = t11

    # mask out dim regions
    phase = phase

    # phase to depth mapping
    zmultiplier = z_table
    xmultiplier = x_table

    phase_msk = tf.cast(tf.less(0.0, phase), dtype)
    phase = phase_msk * (phase + prms['phase_offset']) + (1 - phase_msk) * phase

    depth_linear = zmultiplier * phase
    depth = depth_linear
    max_depth = phase * prms['unambiguous_dist'] * 2

    cond1 = tf.cast(tf.less(0.0, depth_linear), dtype) * \
            tf.cast(tf.less(0.0, max_depth), dtype)

    # xmultiplier = (xmultiplier * 90) / (max_depth**2 * 8192.0)

    # depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)

    # depth_fit = tf.maximum(depth_fit, 0.0)
    # depth = cond1 * depth_fit + (1 - cond1) * depth_linear

    depth_out = depth
    ir_sum_out = ir_sum
    ir_out = tf.minimum( \
        tf.reduce_sum(iramp, -1) * 0.33333333 * prms['ab_output_multiplier'],
        65535.0
    )

    msk_out = cond1 * phase_msk_t * mask * mask2 * mask3
    return depth_out, ir_sum_out, ir_out, msk_out


def filterPixelStage1(m):
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude

    #
    norm2 = m[:, :, :, 0:3] ** 2 + m[:, :, :, 3:6] ** 2
    inv_norm = 1.0 / tf.sqrt(norm2)

    # get rid of those nan
    inv_norm = tf.minimum(inv_norm, 1e10)

    m_normalized = tf.stack([m[:, :, :, 0:3] * inv_norm, m[:, :, :, 3:6] * inv_norm], -1)

    threshold = prms['joint_bilateral_ab_threshold'] ** 2 / prms['ab_multiplier'] ** 2
    joint_bilateral_exp = prms['joint_bilateral_exp']
    threshold = tf.constant(threshold, dtype=dtype)
    joint_bilateral_exp = tf.constant(joint_bilateral_exp, dtype=dtype)

    # set the parts with norm2 < threshold to be zero
    norm_flag = tf.cast(tf.less(norm2, threshold), dtype=dtype)
    threshold = (1 - norm_flag) * threshold
    joint_bilateral_exp = (1 - norm_flag) * joint_bilateral_exp

    # guided bilateral filtering
    gauss = prms['gaussian_kernel']
    weight_acc = tf.ones(tf.shape(m_normalized)[0:4]) * gauss[1, 1]
    weighted_m_acc0 = gauss[1, 1] * m[:, :, :, 0:3]
    weighted_m_acc1 = gauss[1, 1] * m[:, :, :, 3:6]

    # coefficient for bilateral space
    m_n = m_normalized

    # proxy for other m normalized
    m_l = tf.concat([m_n[:, :, 1::, :], m_n[:, :, 0:1, :]], 2)
    m_r = tf.concat([m_n[:, :, -1::, :], m_n[:, :, 0:-1, :]], 2)
    m_u = tf.concat([m_n[:, 1::, :, :], m_n[:, 0:1, :, :]], 1)
    m_d = tf.concat([m_n[:, -1::, :, :], m_n[:, 0:-1, :, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :, :], m_l[:, 0:1, :, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :, :], m_r[:, 0:1, :, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :, :], m_l[:, 0:-1, :, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :, :], m_r[:, 0:-1, :, :]], 1)

    m_n_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_n_shift = tf.stack(m_n_shift, -1)

    # proxy of other_norm2
    norm2_l = tf.concat([norm2[:, :, 1::, :], norm2[:, :, 0:1, :]], 2)
    norm2_r = tf.concat([norm2[:, :, -1::, :], norm2[:, :, 0:-1, :]], 2)
    norm2_u = tf.concat([norm2[:, 1::, :, :], norm2[:, 0:1, :, :]], 1)
    norm2_d = tf.concat([norm2[:, -1::, :, :], norm2[:, 0:-1, :, :]], 1)
    norm2_lu = tf.concat([norm2_l[:, 1::, :, :], norm2_l[:, 0:1, :, :]], 1)
    norm2_ru = tf.concat([norm2_r[:, 1::, :, :], norm2_r[:, 0:1, :, :]], 1)
    norm2_ld = tf.concat([norm2_l[:, -1::, :, :], norm2_l[:, 0:-1, :, :]], 1)
    norm2_rd = tf.concat([norm2_r[:, -1::, :, :], norm2_r[:, 0:-1, :, :]], 1)
    other_norm2 = tf.stack([ \
        norm2_rd, norm2_d, norm2_ld, norm2_r,
        norm2_l, norm2_ru, norm2_u, norm2_lu,
    ], -1)

    dist = [ \
        m_rd * m_n, m_d * m_n, m_ld * m_n, m_r * m_n,
        m_l * m_n, m_ru * m_n, m_u * m_n, m_lu * m_n,
    ]
    dist = -tf.reduce_sum(tf.stack(dist, -1), -2)
    dist += 1.0
    dist *= 0.5

    # color filtering
    gauss_f = gauss.flatten()
    gauss_f = np.delete(gauss_f, [4])
    joint_bilateral_exp = tf.tile(tf.expand_dims(joint_bilateral_exp, -1), [1, 1, 1, 1, 8])
    weight_f = tf.exp(-1.442695 * joint_bilateral_exp * dist)
    weight = tf.stack([gauss_f[k] * weight_f[:, :, :, :, k] for k in range(weight_f.shape[-1])], -1)

    # if (other_norm2 >= threshold)...
    threshold = tf.tile(tf.expand_dims(threshold, -1), [1, 1, 1, 1, 8])
    wgt_msk = tf.cast(tf.less(threshold, other_norm2), dtype=dtype)
    weight = wgt_msk * weight
    dist = wgt_msk * dist

    # coefficient for bilateral space
    ms = tf.stack([m[:, :, :, 0:3], m[:, :, :, 3:6]], -1)

    # proxy for other m normalized
    m_l = tf.concat([ms[:, :, 1::, :], ms[:, :, 0:1, :]], 2)
    m_r = tf.concat([ms[:, :, -1::, :], ms[:, :, 0:-1, :]], 2)
    m_u = tf.concat([ms[:, 1::, :, :], ms[:, 0:1, :, :]], 1)
    m_d = tf.concat([ms[:, -1::, :, :], ms[:, 0:-1, :, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :, :], m_l[:, 0:1, :, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :, :], m_r[:, 0:1, :, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :, :], m_l[:, 0:-1, :, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :, :], m_r[:, 0:-1, :, :]], 1)
    m_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)

    weighted_m_acc0 += tf.reduce_sum(weight * m_shift[:, :, :, :, 0, :], -1)
    weighted_m_acc1 += tf.reduce_sum(weight * m_shift[:, :, :, :, 1, :], -1)

    dist_acc = tf.reduce_sum(dist, -1)
    weight_acc += tf.reduce_sum(weight, -1)

    # test the edge
    bilateral_max_edge_test = tf.reduce_prod(tf.cast( \
        tf.less(dist_acc, prms['joint_bilateral_max_edge']),
        dtype
    ), -1)

    m_out = []
    wgt_acc_msk = tf.cast(tf.less(0.0, weight_acc), dtype=dtype)
    m_out.append(wgt_acc_msk * weighted_m_acc0 / weight_acc)
    m_out.append(wgt_acc_msk * weighted_m_acc1 / weight_acc)
    m_out.append(m[:, :, :, 6:9])

    m_out = tf.concat(m_out, -1)

    # mask out the edge
    # do not filter the edge
    edge_step = 1
    edge_msk = np.zeros(m.shape[1:3])
    edge_msk[0:0 + edge_step, :] = 1
    edge_msk[-1 - edge_step + 1::, :] = 1
    edge_msk[:, 0:0 + edge_step] = 1
    edge_msk[:, -1 - edge_step + 1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(tf.expand_dims(edge_msk, -1), 0), [tf.shape(m)[0], 1, 1, 9])

    m_out = edge_msk * m + (1 - edge_msk) * m_out

    return m_out, bilateral_max_edge_test


def filterPixelStage2(raw_depth, raw_depth_edge, ir_sum):
    # raw depth is the raw depth prediction
    # raw_depth_edge is roughly the same as raw depth, except some part are zero if
    # don't want to do edge filtering
    # mask out depth that is out of region
    depth_msk = tf.cast(tf.greater(raw_depth, prms['min_depth']), dtype) * \
                tf.cast(tf.less(raw_depth, prms['max_depth']), dtype)
    # mask out the edge
    # do not filter the edge of the image
    edge_step = 1
    edge_msk = np.zeros(raw_depth.shape[1:3])
    edge_msk[0:0 + edge_step, :] = 1
    edge_msk[-1 - edge_step + 1::, :] = 1
    edge_msk[:, 0:0 + edge_step] = 1
    edge_msk[:, -1 - edge_step + 1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(edge_msk, 0), [tf.shape(raw_depth)[0], 1, 1])

    #
    knl = tf.constant(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), dtype=dtype)
    knl = tf.expand_dims(tf.expand_dims(knl, -1), -1)
    ir_sum_exp = tf.expand_dims(ir_sum, -1)
    ir_sum_acc = tf.nn.conv2d(ir_sum_exp, knl, strides=[1, 1, 1, 1], padding='SAME')
    squared_ir_sum_acc = tf.nn.conv2d(ir_sum_exp ** 2, knl, strides=[1, 1, 1, 1], padding='SAME')
    ir_sum_acc = tf.squeeze(ir_sum_acc, -1)
    squared_ir_sum_acc = tf.squeeze(squared_ir_sum_acc, -1)
    min_depth = raw_depth
    max_depth = raw_depth

    # min_depth, max_depth
    m_n = raw_depth_edge
    m_l = tf.concat([m_n[:, :, 1::], m_n[:, :, 0:1]], 2)
    m_r = tf.concat([m_n[:, :, -1::], m_n[:, :, 0:-1]], 2)
    m_u = tf.concat([m_n[:, 1::, :], m_n[:, 0:1, :]], 1)
    m_d = tf.concat([m_n[:, -1::, :], m_n[:, 0:-1, :]], 1)
    m_lu = tf.concat([m_l[:, 1::, :], m_l[:, 0:1, :]], 1)
    m_ru = tf.concat([m_r[:, 1::, :], m_r[:, 0:1, :]], 1)
    m_ld = tf.concat([m_l[:, -1::, :], m_l[:, 0:-1, :]], 1)
    m_rd = tf.concat([m_r[:, -1::, :], m_r[:, 0:-1, :]], 1)
    m_shift = [ \
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)
    nonzero_msk = tf.cast(tf.greater(m_shift, 0.0), dtype=dtype)
    m_shift_min = nonzero_msk * m_shift + (1 - nonzero_msk) * 99999999999
    min_depth = tf.minimum(tf.reduce_min(m_shift_min, -1), min_depth)
    max_depth = tf.maximum(tf.reduce_max(m_shift, -1), max_depth)

    #
    tmp0 = tf.sqrt(squared_ir_sum_acc * 9.0 - ir_sum_acc ** 2) / 9.0
    edge_avg = tf.maximum( \
        ir_sum_acc / 9.0, prms['edge_ab_avg_min_value']
    )
    tmp0 /= edge_avg

    #
    abs_min_diff = tf.abs(raw_depth - min_depth)
    abs_max_diff = tf.abs(raw_depth - max_depth)

    avg_diff = (abs_min_diff + abs_max_diff) * 0.5
    max_abs_diff = tf.maximum(abs_min_diff, abs_max_diff)

    cond0 = []
    cond0.append(tf.cast(tf.less(0.0, raw_depth), dtype))
    cond0.append(tf.cast(tf.greater_equal(tmp0, prms['edge_ab_std_dev_threshold']), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_close_delta_threshold'], abs_min_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_far_delta_threshold'], abs_max_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_max_delta_threshold'], max_abs_diff), dtype))
    cond0.append(tf.cast(tf.less(prms['edge_avg_delta_threshold'], avg_diff), dtype))

    cond0 = tf.reduce_prod(tf.stack(cond0, -1), -1)

    depth_out = (1 - cond0) * raw_depth

    # !cond0 part
    edge_test_msk = 1 - tf.cast(tf.equal(raw_depth_edge, 0.0), dtype)
    depth_out = raw_depth * (1 - cond0) * edge_test_msk

    # mask out the depth out of the range
    depth_out = depth_out * depth_msk

    # mask out the edge
    depth_out = edge_msk * raw_depth + (1 - edge_msk) * depth_out

    # msk_out
    msk_out = edge_msk + (1 - edge_msk) * depth_msk * (1 - cond0) * edge_test_msk

    return depth_out, msk_out


def get_loss(depth, gt, ori_depth_msk, gt_msk):
    """
    Get the depth loss and raw measurement loss
    """
    """
        Loss function: L2 and so on. 
    """
    """ Args: depth is the depth prediction from netword, is a 4-D tensor (BxHxWx1)
              gt is the depth ground truth, is a 4-D tensor (BxHxWx1)
              full is the raw measurement with noise from Kinect, is a 4-D tensor (BxHxWx9)
              ideal is the groundtruth of the raw measurement , is a 4-D tensor (BxHxWx9)
              mask reprsent the area that compute the loss
        Return: loss is 1-D Tensor 
    """

    # depth_flat = tf.layers.flatten(depth)
    # gt_flat = tf.layers.flatten(gt)
    ## depth loss : l2
    # depth_loss = (tf.reduce_sum((tf.abs(depth - gt) * gt_msk) ** 2)
    #               / (tf.reduce_sum(gt_msk))) ** (1 / 2)

    depth_loss = tf.reduce_mean(tf.abs(depth - gt) ** 2) ** (1 / 2)

    ## raw loss : l2
    # raw_loss = tf.reduce_mean(tf.abs(full - ideal) ** 2) ** (1 / 2)

    loss = depth_loss
    return loss

def get_metrics(depth, ori_depth, gt, depth_msk, ori_msk, gt_msk):

    ori_mse, update_op_ori = tf.metrics.mean_squared_error(gt, ori_depth)
    ori_psnr = 10 * (tf.log(25.0 / ori_mse) / tf.log(10.0))
    update_op_ori = 10 * (tf.log(25.0 / update_op_ori) / tf.log(10.0))
    pre_mse, update_op_pre = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr = 10 * (tf.log(25.0 / pre_mse) / tf.log(10.0))
    update_op_pre = 10 * (tf.log(25.0 / update_op_pre) / tf.log(10.0))

    depth = depth * gt_msk
    ori_depth = ori_depth * gt_msk
    ori_mse_dm, update_op_ori_dm = tf.metrics.mean_squared_error(gt * ori_msk, ori_depth)
    ori_psnr_dm = 10 * (tf.log(25.0 / ori_mse_dm) / tf.log(10.0))
    update_op_ori_dm = 10 * (tf.log(25.0 / update_op_ori_dm) / tf.log(10.0))
    # pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt * depth_msk, depth)
    pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr_dm = 10 * (tf.log(25.0 / pre_mse_dm) / tf.log(10.0))
    update_op_pre_dm = 10 * (tf.log(25.0 / update_op_pre_dm) / tf.log(10.0))

    return (ori_psnr, update_op_ori), (pre_psnr, update_op_pre),\
           (ori_psnr_dm, update_op_ori_dm), (pre_psnr_dm, update_op_pre_dm)

def kinect_mask_tensor():
    # return the kinect mask that creates the positive-negative interval
    mask = np.zeros((424, 512))
    idx = 1
    for i in range(mask.shape[0]):
        mask[i, :] = idx
        if i != (mask.shape[0] / 2 - 1):
            idx = -idx

    mask = tf.convert_to_tensor(mask)
    mask = tf.cast(mask, tf.float32)
    return mask


def preprocessing(features, labels):
    msk = kinect_mask_tensor()
    meas = features['full']
    meas = [meas[:, :, i] * msk / tof_cam.cam['map_max'] for i in
            range(meas.shape[-1])]  ##tof_cam.cam['map_max'] == 3500
    meas = tf.stack(meas, -1)
    meas_p = meas[20:-20, :, :]

    ideal = labels['ideal']
    ideal = [ideal[:, :, i] * msk / tof_cam.cam['map_max'] for i in range(ideal.shape[-1])]
    ideal = tf.stack(ideal, -1)
    ideal_p = ideal[20:-20, :, :]
    gt = labels['gt']
    gt = tf.image.resize_images(gt, [meas.shape[0], meas.shape[1]])
    gt_p = gt[20:-20, :, :]
    features['full'] = meas_p
    labels['ideal'] = ideal_p
    labels['gt'] = gt_p
    return features, labels


def kinect_pipeline(meas):
    ## Kinect Pipeline, use the algorithm of Kinect v2 to compute the denoised depth

    # convert to the default data type
    x_kinect = tf.cast(meas, tf.float32)
    # make the size to be 424,512 (padding 0)
    y_idx = int((424 - int(x_kinect.shape[1])) / 2)
    zero_mat = tf.zeros([tf.shape(x_kinect)[0], y_idx, tf.shape(x_kinect)[2], 9])
    x_kinect = tf.concat([zero_mat, x_kinect, zero_mat], 1)

    msk = kinect_mask_tensor()
    msk = tf.expand_dims(tf.expand_dims(msk, 0), -1)
    x_kinect = x_kinect * msk * tof_cam.cam['map_max']

    # final depth prediction: kinect pipeline
    ira, irb, iramp = processPixelStage1_mat(x_kinect)
    depth_outs, ir_sum_outs, ir_outs, msk_out1 = processPixelStage2(ira, irb, iramp)

    # creates the mask
    ms = tf.concat([ira, irb, iramp], -1)
    bilateral_max_edge_tests = filterPixelStage1(ms)[1]
    depth_out_edges = depth_outs * bilateral_max_edge_tests
    msk_out2 = filterPixelStage2(depth_outs, depth_out_edges, ir_outs)[1]
    msk_out3 = tf.cast(tf.greater(depth_outs, prms['min_depth']), dtype=dtype)
    msk_out4 = tf.cast(tf.less(depth_outs, prms['max_depth']), dtype=dtype)
    depth_msk = tf.cast(tf.greater(msk_out2 * msk_out3 * msk_out4, 0.5), dtype=dtype)

    depth_outs /= 1000.0
    depth_outs *= depth_msk

    # baseline correction
    depth_outs = depth_outs * base_cor['k'] + base_cor['b']

    depth_outs = depth_outs[:, 20:-20, :]
    depth_msk = depth_msk[:, 20:-20, :]

    return depth_outs, depth_msk


def tof_net_func(features, labels, mode, params):

    gt = labels['gt']
    full = features['full']
    ideal = labels['ideal']
    depth_kinect, depth_kinect_msk = kinect_pipeline(full)

    ## use the network to denoise and de-MPI
    depth_kinect = tf.expand_dims(depth_kinect, -1)
    depth_kinect_msk = tf.expand_dims(depth_kinect_msk, axis=-1)
    depth_outs = deformable_deeptof(depth_kinect, flg=mode == tf.estimator.ModeKeys.TRAIN, regular=0.1, batch_size=params['batch_size'])

    depth_msk = depth_outs > 1e-4
    depth_msk = tf.cast(depth_msk, tf.float32)
    ## first compute the mask and make the same in dtype and deimension

    gt_msk = gt > 1e-4
    gt_msk = tf.cast(gt_msk, tf.float32)

    pre_depth_error_map = tf.abs(depth_outs - gt)
    kinect_depth_error_map = tf.abs(depth_kinect - gt)
    pre_depth_error_map = tf.identity(pre_depth_error_map, 'pre_depth_error_map')
    kinect_depth_error_map = tf.identity(kinect_depth_error_map, 'kinect_depth_error_map')
    ## Summary
    tf.summary.image('PRE_DEPTH_ERROR_MAP', pre_depth_error_map)
    tf.summary.image('KINECT_DEPTH_ERROR_MAP', kinect_depth_error_map)
    # compute loss (for TRAIN and EVAL modes)
    loss = None
    train_op = None
    metrics = None
    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        loss = get_loss(depth_outs, gt, depth_kinect_msk, gt_msk)
        loss = tf.identity(loss, name="loss")
        # configure the training op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('TRAINING_LOSS', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            global_step = tf.train.get_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            # global batch_size
            ori_psnr, pre_psnr , ori_psnr_dm, pre_psnr_dm = get_metrics(depth_outs, depth_kinect, gt,
                                                                       depth_msk, depth_kinect_msk, gt_msk)
            metrics = {
                        "ori_PSNR": ori_psnr,
                        "pre_PSNR": pre_psnr,
                        "ori_PSNR_dm": ori_psnr_dm,
                        "pre_PSNR_dm": pre_psnr_dm,
                       }

            # metrics = {
            #     'accuracy': accuracy
            # }
            # return tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     loss=loss,
            #     eval_metric_ops=metrics
            # )
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "depth": depth_outs,
        }
    else:
        predictions = None

    # return a ModelFnOps object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
    )


def imgs_input_fn(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    def _parse_function(serialized, height=height, width=width):
        features = \
            {
                'meas': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string),
                'ideal': tf.FixedLenFeature([], tf.string)
            }

        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        meas_shape = tf.stack([height, width, 9])
        gt_shape = tf.stack([height * 4, width * 4, 1])
        ideal_shape = tf.stack([height, width, 9])

        meas_raw = parsed_example['meas']
        gt_raw = parsed_example['gt']
        ideal_raw = parsed_example['ideal']

        # decode the raw bytes so it becomes a tensor with type

        meas = tf.decode_raw(meas_raw, tf.int32)
        meas = tf.cast(meas, tf.float32)
        meas = tf.reshape(meas, meas_shape)

        gt = tf.decode_raw(gt_raw, tf.float32)
        gt = tf.reshape(gt, gt_shape)

        ideal = tf.decode_raw(ideal_raw, tf.int32)
        ideal = tf.cast(ideal, tf.float32)
        ideal = tf.reshape(ideal, ideal_shape)

        features = {'full': meas}
        labels = {'gt': gt, 'ideal': ideal}

        return features, labels

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda features, labels: preprocessing(features, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)  # Repeat the dataset this time
    batch_dataset = dataset.batch(batch_size)  # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()  # Make an iterator
    batch_features, batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels

    return batch_features, batch_labels

def dataset_evaluate(evaluate_data_path, model_dir, learning_rate, batch_size, shuffle):
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        # keep_checkpoint_max=6,
        # save_checkpoints_steps=evaluate_steps,
        log_step_count_steps=1)  # set the frequency of logging steps for loss function
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration, params={'learning_rate': learning_rate})
    data = tof_net.evaluate(input_fn=lambda: imgs_input_fn(filenames=evaluate_data_path, height=424, width=512,
                                                    shuffle=shuffle, repeat_count=1, batch_size=batch_size))
    # data = list(tof_net.evaluate(input_fn=lambda: imgs_input_fn(filenames=evaluate_data_path, height=424, width=512,
    #                                     shuffle=True, repeat_count=1, batch_size=batch_size)))
    print(data['ori_PSNR'])
    print(data['pre_PSNR'])

def dataset_training(train_data_path, evaluate_data_path, model_dir, learning_rate, batch_size, traing_steps, evaluate_steps):
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=6,
        save_checkpoints_steps=evaluate_steps,
        log_step_count_steps=10)  # set the frequency of logging steps for loss function

    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, params={'learning_rate': learning_rate, 'batch_size': batch_size},
                                     config=configuration)


    train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(filenames=train_data_path, height=424, width=512,
                                                               shuffle=True, repeat_count=-1, batch_size=batch_size),
                                        max_steps=traing_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(filenames=evaluate_data_path, height=424, width=512,
                                                             shuffle=True, repeat_count=1, batch_size=batch_size),
                                      steps=evaluate_steps)
    tf.estimator.train_and_evaluate(tof_net, train_spec, eval_spec)


if __name__ == '__main__':
    # data
    model_dir = './models/kinect/deformable_deeptof/x5_dep_l2nomsk_1e-5'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train_data_path = '../FLAT/kinect_tfrecords/x5_full_ideal_gt/x5_full_ideal_gt_train.tfrecords'
    evaluate_data_path = '../FLAT/kinect_tfrecords/x5_full_ideal_gt/x5_full_ideal_gt_eval.tfrecords'

    # initialize the camera model
    tof_cam = kinect_real_tf()

    dataset_training(train_data_path=train_data_path, evaluate_data_path=evaluate_data_path,
                     model_dir=model_dir, learning_rate=1e-5, batch_size=5, traing_steps=16000,
                     evaluate_steps=400
                     )
