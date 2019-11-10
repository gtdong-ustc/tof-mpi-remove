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


def smoothness(x, y):
    """
    Smoothness constraint between predicted and image
    Args:
        x: disparity
        y: image
    """

    def gradient_x(image):
        sobel_x = tf.Variable(initial_value=[[1, 0, -1], [2, 0, -2], [1, 0, -1]], trainable=False, dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        if image.get_shape()[-1].value == 3:
            sobel_x = tf.concat([sobel_x, sobel_x, sobel_x], axis=2)
        return tf.nn.conv2d(image, sobel_x, [1, 1, 1, 1], padding='SAME')

    def gradient_y(image):
        sobel_y = tf.Variable(initial_value=[[1, 2, -1], [0, 0, 0], [-1, -2, -1]], trainable=False, dtype=tf.float32)
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        if image.get_shape()[-1].value == 3:
            sobel_y = tf.concat([sobel_y, sobel_y, sobel_y], axis=2)
        return tf.nn.conv2d(image, sobel_y, [1, 1, 1, 1], padding='SAME')

    # normalize image and disp in a fixed range
    x = x / 255
    y = y / 255

    disp_gradients_x = gradient_x(x)
    disp_gradients_y = gradient_y(x)

    image_gradients_x = tf.reduce_mean(gradient_x(y), axis=-1, keepdims=True)
    image_gradients_y = tf.reduce_mean(gradient_y(y), axis=-1, keepdims=True)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = tf.abs(disp_gradients_x) * weights_x
    smoothness_y = tf.abs(disp_gradients_y) * weights_y

    return tf.reduce_mean(smoothness_x + smoothness_y)


def mean_l1(x, y, mask=None):
    """
    Mean reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.abs(x - y)) / tf.reduce_sum(mask)


def mean_l2(x, y, mask):
    """
    Mean squarred error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask*tf.square(x - y)) / tf.reduce_sum(mask)


def sum_l1(x, y, mask=None):
    """
    Sum of the reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.abs(x - y))


def sum_l2(x, y, mask=None):
    """
    Sum squarred error
    Args:
        x: predicted image
        y: target image
        mask: compute only on those points
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)
    return tf.reduce_sum(mask * tf.square(x - y))


SUPERVISED_LOSS = {
    'mean_l1': mean_l1,
    'sum_l1': sum_l1,
    'mean_l2': mean_l2,
    'sum_l2': sum_l2,
    'smoothness': smoothness,
}

ALL_LOSSES = dict(SUPERVISED_LOSS)

def get_supervised_loss(name, x, y, mask=None):
    if name not in ALL_LOSSES.keys():
        print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
        raise Exception('Unknown loss function selected')
    base_loss_function = ALL_LOSSES[name]
    return base_loss_function(x, y, mask)

def get_mae(depth_outs, gt, msk):
    return tf.reduce_sum(msk * tf.abs(depth_outs - gt)) / tf.reduce_sum(msk)

def get_metrics_psnr(depth, ori_depth, gt, depth_msk, ori_msk, gt_msk):
    ori_mse, update_op_ori = tf.metrics.mean_squared_error(gt, ori_depth)
    ori_psnr = 10 * (tf.log(25.0 / ori_mse) / tf.log(10.0))
    update_op_ori = 10 * (tf.log(25.0 / update_op_ori) / tf.log(10.0))
    pre_mse, update_op_pre = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr = 10 * (tf.log(25.0 / pre_mse) / tf.log(10.0))
    update_op_pre = 10 * (tf.log(25.0 / update_op_pre) / tf.log(10.0))

    depth = depth * gt_msk
    ori_depth = ori_depth * gt_msk
    ori_mse_dm, update_op_ori_dm = tf.metrics.mean_squared_error(gt, ori_depth)
    ori_psnr_dm = 10 * (tf.log(25.0 / ori_mse_dm) / tf.log(10.0))
    update_op_ori_dm = 10 * (tf.log(25.0 / update_op_ori_dm) / tf.log(10.0))
    # pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt * depth_msk, depth)
    pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr_dm = 10 * (tf.log(25.0 / pre_mse_dm) / tf.log(10.0))
    update_op_pre_dm = 10 * (tf.log(25.0 / update_op_pre_dm) / tf.log(10.0))

    return (ori_psnr, update_op_ori), (pre_psnr, update_op_pre), \
           (ori_psnr_dm, update_op_ori_dm), (pre_psnr_dm, update_op_pre_dm)

def get_metrics_mae(depth, ori_depth, gt, depth_msk, ori_msk, gt_msk):
    ori_mae, update_ori_mae = tf.metrics.mean_absolute_error(gt, ori_depth)
    pre_mae, update_pre_mae = tf.metrics.mean_absolute_error(gt, depth)

    gt_msk_k = tf.reduce_mean(gt_msk)
    depth = depth * gt_msk
    ori_depth = ori_depth * gt_msk
    ori_mae_dm, update_ori_mae_dm = tf.metrics.mean_absolute_error(gt / gt_msk_k, ori_depth / gt_msk_k)
    pre_mae_dm, update_pre_mae_dm = tf.metrics.mean_absolute_error(gt / gt_msk_k, depth / gt_msk_k)
    return (ori_mae, update_ori_mae), (pre_mae, update_pre_mae), \
           (ori_mae_dm, update_ori_mae_dm), (pre_mae_dm, update_pre_mae_dm)
