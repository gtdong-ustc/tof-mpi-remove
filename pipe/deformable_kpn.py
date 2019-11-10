# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys

sys.path.insert(0, '../sim/')
import argparse
import numpy as np
import tensorflow as tf
import os, json, glob
import imageio
import matplotlib
import math
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

from loss import *
from models import *
from kinect_option import *
from dataset import *

def tof_net_func(features, labels, mode, params):
    depth_kinect = None
    depth_kinect_msk = None
    raw_new = None
    offsets = None
    depth_outs = None
    depth_msk = None
    gt = labels['gt']
    full = features['full']
    ideal = labels['ideal']

    model_name_list = params['model_name'].split('_')
    gt_msk = gt > 1e-4
    gt_msk = tf.cast(gt_msk, tf.float32)
    depth_kinect, depth_kinect_msk = kinect_pipeline(full)
    depth_kinect = tf.expand_dims(depth_kinect, -1)
    depth_kinect_msk = tf.expand_dims(depth_kinect_msk, axis=-1)

    if model_name_list[-1] != 'raw':
        if model_name_list[0] == 'deformable':
            depth_outs, offsets = get_network(name=params['model_name'], x=depth_kinect,
                                              flg=mode == tf.estimator.ModeKeys.TRAIN, regular=0.1,
                                              batch_size=params['batch_size'], range=params['deformable_range'])

            samples_position = samples_visualizaiton(depth_outs, offsets, h_selected_pos=params['selected_position'][0],
                                                     w_selected_pos=params['selected_position'][1],
                                                     batch_size=params['batch_size'], N=9)
            samples_position_test = samples_visualizaiton(depth_outs, offsets, h_selected_pos=300,
                                                          w_selected_pos=400,
                                                          batch_size=params['batch_size'], N=9)
            ## Summary
            samples_position_map = tf.identity(samples_position + samples_position_test, 'samples_position')
            tf.summary.image('samples_position', samples_position_map)
        else:
            depth_outs = get_network(name=params['model_name'], x=depth_kinect, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                     regular=0.1, batch_size=params['batch_size'], range=params['deformable_range'])

        depth_msk = depth_outs > 1e-4
        depth_msk = tf.cast(depth_msk, tf.float32)
    else:
        if model_name_list[0] == 'deformable':
            raw_new, offsets = get_network(name=params['model_name'], x=full, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                     regular=0.1, batch_size=params['batch_size'], range=params['deformable_range'])
            # samples_position = samples_visualizaiton(raw_new, offsets, h_selected_pos=params['selected_position'][0],
            #                                          w_selected_pos=params['selected_position'][1], batch_size=params['batch_size'], N=9)
            # samples_position_test = samples_visualizaiton(raw_new, offsets, h_selected_pos=300,
            #                                               w_selected_pos=400,batch_size=params['batch_size'], N=9)
            # ## Summary
            # samples_position_map = tf.identity(samples_position + samples_position_test, 'samples_position')
            # tf.summary.image('samples_position', samples_position_map)
        else:
            raw_new = get_network(name=params['model_name'], x=full, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                           regular=0.1, batch_size=params['batch_size'],range=params['deformable_range'])

        depth_outs, depth_msk = kinect_pipeline(raw_new)
        depth_outs = tf.expand_dims(depth_outs, -1)
        depth_msk = tf.expand_dims(depth_msk, axis=-1)
    ## Summary
    depth_gt_map = tf.identity(gt, 'depth_gt')
    depth_outs_map = tf.identity(depth_outs, 'depth_outs')
    depth_kinectr_map = tf.identity(depth_kinect, 'depth_kinect')
    tf.summary.image('depth_gt', depth_gt_map)
    tf.summary.image('depth_outs', depth_outs_map)
    tf.summary.image('depth_kinect', depth_kinectr_map)
    # compute loss (for TRAIN and EVAL modes)
    loss = None
    train_op = None
    metrics = None
    eval_summary_hook = None
    evaluation_hooks = []
    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):

        if model_name_list[-1] == 'raw':
            loss_raw = get_supervised_loss(params['loss_fn'], raw_new, ideal)
            loss_depth = get_supervised_loss(params['loss_fn'], depth_outs, gt, gt_msk)

            loss = 0.9 * loss_raw + 0.1 * loss_depth
            loss = tf.identity(loss, name="loss")
        else:
            loss = get_supervised_loss(params['loss_fn'], depth_outs, gt, gt_msk)
        # configure the training op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('training_loss', loss)
            # tf.summary.scalar('training_psnr', training_psnr)
            # tf.summary.scalar('training_mae', training_mae)
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            global_step = tf.train.get_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            ori_mae, pre_mae, ori_mae_dm, pre_mae_dm = get_metrics_mae(depth_outs, depth_kinect, gt,
                                                                       depth_msk, depth_kinect_msk, gt_msk)

            metrics = {
                "ori_MAE": ori_mae,
                "pre_MAE": pre_mae,
                "ori_MAE_dm": ori_mae_dm,
                "pre_MAE_dm": pre_mae_dm,
            }
            eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=1,
                                output_dir= params['model_dir'] + "/eval",
                                summary_op=tf.summary.merge_all())
            evaluation_hooks.append(eval_summary_hook)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics,
                evaluation_hooks=evaluation_hooks
            )

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

def dataset_training(train_data_path, evaluate_data_path, model_dir, loss_fn, learning_rate, batch_size, traing_steps, evaluate_steps, deformable_range, selected_position, model_name):
    session_config = tf.ConfigProto(device_count={'GPU': 0, 'GPU': 1})
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=6,
        save_checkpoints_steps=200,
        session_config=session_config,
        save_summary_steps=5,
        log_step_count_steps=5
    )  # set the frequency of logging steps for loss function

    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
                                     params={'learning_rate': learning_rate, 'batch_size': batch_size, 'model_dir': model_dir,
                                             'loss_fn': loss_fn, 'deformable_range': deformable_range,
                                             'selected_position': selected_position, 'model_name': model_name})


    train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(filenames=train_data_path, height=424, width=512,
                                                               shuffle=True, repeat_count=-1, batch_size=batch_size),
                                        max_steps=traing_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(filenames=evaluate_data_path, height=424, width=512,
                                                             shuffle=True, repeat_count=1, batch_size=batch_size),
                                      steps=None, throttle_secs=evaluate_steps)
    tf.estimator.train_and_evaluate(tof_net, train_spec, eval_spec)

def dataset_testing(evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range, loss_fn, selected_position, model_name):
    session_config = tf.ConfigProto(device_count={'GPU': 0, 'GPU': 1})
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=session_config,
        log_step_count_steps = 10,
        save_summary_steps = 5,)
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
        params={'learning_rate': 1e-4, 'batch_size': batch_size, 'model_dir': model_dir,
                'deformable_range': deformable_range, 'loss_fn':loss_fn,
                'selected_position': selected_position, 'model_name': model_name})
    tof_net.evaluate(input_fn=lambda: imgs_input_fn(filenames=evaluate_data_path, height=424, width=512,
        shuffle=False, repeat_count=1, batch_size=batch_size), checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deformable KPN Network')
    parser.add_argument("-t", "--trainingSet", help='the name to the list file with training set', default = 'x5_reflection_ideal_gt', type=str)

    parser.add_argument("-m", "--modelName", help="name of the denoise model to be used", default="deformable_kpn")
    parser.add_argument("-l", "--lr", help="initial value for learning rate", default=1e-5, type=float)
    parser.add_argument("-i", "--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[424, 512])
    parser.add_argument("-b", "--batchSize", help='batch size to use during training', type=int, default=4)
    parser.add_argument("-s", "--steps", help='number of training steps', type=int, default=4000)
    parser.add_argument("-e", "--evalSteps", help='after the number of training steps to eval', type=int, default=100)
    parser.add_argument("-o", '--lossType', help="Type of supervised loss to use", default="mean_l2", type=str)
    parser.add_argument("-d", "--deformableRange", help="the range of deformable kernel", default=192.0, type=float)
    parser.add_argument("-f", '--flagEval', help="The flag to identity the test", default=False, type=bool)
    parser.add_argument("-p", '--postfix', help="the postfix of the training task", default=None, type=str)
    parser.add_argument('-n', '--selectedPosition', help='the position of selected pixel', default=[191, 255], type=int, nargs='+')
    parser.add_argument("-c", '--checkpointSteps', help="", default="800", type=str)
    args = parser.parse_args()
    # parser.add_argument("--decayStep", help="halve learning rate after this many steps", type=int, default=50000)
    # parser.add_argument("-e", "--evaluateSet", help="path to the list file with the validation set", default=None, type=str)
    # parser.add_argument("-m", "--modelDir", help="path to the output folder where the results will be saved",
    #                     required=True)


    model_dir = './models/kinect/' + args.modelName
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.modelName[0:10] == 'deformable':
        mkdir_name = args.trainingSet.split('_')[0] + '_' + args.lossType + '_dR' + str(args.deformableRange)
    else:
        mkdir_name = args.trainingSet.split('_')[0] + '_' + args.lossType
    if args.postfix is not None:
        model_dir = os.path.join(model_dir, mkdir_name + '_' + args.postfix)
    else:
        model_dir = os.path.join(model_dir, mkdir_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train_data_path = os.path.join('../FLAT/kinect_tfrecords/' + args.trainingSet, args.trainingSet + '_train.tfrecords')
    evaluate_data_path = os.path.join('../FLAT/kinect_tfrecords/' + args.trainingSet, args.trainingSet + '_eval.tfrecords')
    # initialize the camera model

    if args.flagEval is not True:
        dataset_training(train_data_path=train_data_path, evaluate_data_path=evaluate_data_path, loss_fn=args.lossType,
                         model_dir=model_dir, learning_rate=args.lr, batch_size=args.batchSize, traing_steps=args.steps,
                         evaluate_steps=args.evalSteps, deformable_range = args.deformableRange, selected_position = args.selectedPosition,
                         model_name=args.modelName
                         )
    else:
        dataset_testing(evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        batch_size=args.batchSize, checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange,
                        selected_position=args.selectedPosition, model_name = args.modelName)
