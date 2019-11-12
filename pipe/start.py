# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys

sys.path.insert(0, '../sim/')
import argparse
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from loss import *
from model import *
from kinect_pipeline import *
from dataset import *
from metric import *

tof_cam = kinect_real_tf()

gpu_device_number_list = [
{'GPU': 0},
{'GPU': 0, 'GPU': 1},
{'GPU': 0, 'GPU': 1, 'GPU': 2},
{'GPU': 0, 'GPU': 1, 'GPU': 2, 'GPU': 3},
{'GPU': 0, 'GPU': 1, 'GPU': 2, 'GPU': 3, 'GPU': 4},
{'GPU': 0, 'GPU': 1, 'GPU': 2, 'GPU': 3, 'GPU': 4, 'GPU': 5},
{'GPU': 0, 'GPU': 1, 'GPU': 2, 'GPU': 3, 'GPU': 4, 'GPU': 5, 'GPU': 6},
{'GPU': 0, 'GPU': 1, 'GPU': 2, 'GPU': 3, 'GPU': 4, 'GPU': 5, 'GPU': 6, 'GPU': 7}
]

def tof_net_func(features, labels, mode, params):
    depth_kinect = None
    depth_kinect_msk = None
    raw_new = None
    offsets = None
    depth_outs = None
    depth_msk = None
    z_multiplier = None
    loss_msk = None
    loss_mask_dict = {}

    if params['training_set'] == 'FLAT_reflection_s5' or params['training_set'] == 'FLAT_full_s5':
        if params['output_flg'] == True:
            gt = None
            ideal = None
        else:
            gt = labels['gt']
            ideal = labels['ideal']
            gt_msk = gt > 1e-4
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['full']
        depth_kinect, depth_kinect_msk = kinect_pipeline(full)
        depth_kinect = tf.expand_dims(depth_kinect, -1)
        depth_kinect_msk = tf.expand_dims(depth_kinect_msk, axis=-1)
    elif params['training_set'] == 'deeptof_reflection':
        if params['output_flg'] == True:
            gt = None
        else:
            gt = labels['depth_ref']
            gt_msk = gt > 1e-4
            gt_msk = tf.cast(gt_msk, tf.float32)

            ### add gt msk
            loss_mask_dict['gt_msk'] = gt_msk
        full = features['depth']
        amps = features['amps']
        depth_kinect = full
        depth_kinect_msk = depth_kinect > 1e-4

    ### add kinect_msk
    loss_mask_dict['depth_kinect_msk'] = depth_kinect_msk


    model_name_list = params['model_name'].split('_')
    if model_name_list[-1] == 'raw':
        if model_name_list[0] == 'deformable':
            raw_new, offsets = get_network(name=params['model_name'], x=full, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                     regular=0.1, batch_size=params['batch_size'], range=params['deformable_range'])
        else:
            raw_new = get_network(name=params['model_name'], x=full, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                           regular=0.1, batch_size=params['batch_size'],range=params['deformable_range'])

        depth_outs, depth_msk = kinect_pipeline(raw_new)
        depth_outs = tf.expand_dims(depth_outs, -1)
        depth_msk = tf.expand_dims(depth_msk, axis=-1)
    else:
        if model_name_list[0] == 'deformable':
            depth_outs, offsets = get_network(name=params['model_name'], x=depth_kinect,
                                              flg=mode == tf.estimator.ModeKeys.TRAIN, regular=0.1,
                                              batch_size=params['batch_size'], range=params['deformable_range'])

        else:
            depth_outs = get_network(name=params['model_name'], x=depth_kinect, flg=mode == tf.estimator.ModeKeys.TRAIN,
                                     regular=0.1, batch_size=params['batch_size'], range=params['deformable_range'])

        depth_msk = depth_outs > 1e-4
        depth_msk = tf.cast(depth_msk, tf.float32)

    ## get the msk needed in compute loss and metrics
    loss_mask_dict['depth_msk'] = depth_msk
    loss_msk = loss_mask_dict[params['loss_mask']]

    # compute loss (for TRAIN and EVAL modes)
    loss = None
    train_op = None
    metrics = None
    eval_summary_hook = None
    evaluation_hooks = []

    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):

        if model_name_list[-1] == 'raw':
            loss_raw = get_supervised_loss(params['loss_fn'], raw_new, ideal)
            loss_depth = get_supervised_loss(params['loss_fn'], depth_outs, gt, loss_msk)

            loss = 0.9 * loss_raw + 0.1 * loss_depth
            loss = tf.identity(loss, name="loss")
        else:
            loss = get_supervised_loss(params['loss_fn'], depth_outs, gt, loss_msk)
        # configure the training op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('training_loss', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            global_step = tf.train.get_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            if loss_msk == None:
                depth_gt_map = tf.identity(gt, 'depth_gt')
                depth_outs_map = tf.identity(depth_outs, 'depth_outs')
                depth_kinect_map = tf.identity(depth_kinect, 'depth_kinect')
                depth_outs_error = tf.identity(tf.abs(gt - depth_outs), 'depth_outs_error')
                depth_kinect_error = tf.identity(tf.abs(gt - depth_kinect), 'depth_kinect_error')
            else:
                depth_gt_map = tf.identity(gt, 'depth_gt')
                depth_outs_map = tf.identity(depth_outs * loss_msk, 'depth_outs')
                depth_kinect_map = tf.identity(depth_kinect * loss_msk, 'depth_kinect')
                depth_outs_error = tf.identity(tf.abs(gt * loss_msk - depth_outs * loss_msk), 'depth_outs_error')
                depth_kinect_error = tf.identity(tf.abs(gt * loss_msk - depth_kinect * loss_msk), 'depth_kinect_error')

            tf.summary.image('depth_gt', colorize_img(depth_gt_map, vmin=0.5, vmax=5.0, cmap='jet'))
            tf.summary.image('depth_outs', colorize_img(depth_outs_map, vmin=0.5, vmax=5.0, cmap='jet'))
            tf.summary.image('depth_kinect', colorize_img(depth_kinect_map, vmin=0.5, vmax=5.0, cmap='jet'))
            tf.summary.image('depth_outs_error', colorize_img(depth_outs_error, vmin=0.0, vmax=0.2, cmap='jet'))
            tf.summary.image('depth_kinect_error', colorize_img(depth_kinect_error, vmin=0.0, vmax=0.2, cmap='jet'))

            ## get metrics
            ori_mae, pre_mae = get_metrics_mae(depth_outs, depth_kinect, gt, loss_msk)
            metrics = {
                "ori_MAE": ori_mae,
                "pre_MAE": pre_mae,
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
            "offset": offsets
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

def dataset_training(train_data_path, evaluate_data_path, model_dir, loss_fn, learning_rate, batch_size, traing_steps,
                     evaluate_steps, deformable_range, model_name, checkpoint_steps, loss_mask, gpu_Number, training_set, image_shape):
    for i in range(gpu_Number):
        session_config = tf.ConfigProto(device_count=gpu_device_number_list[i])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=10,
        save_checkpoints_steps=200,
        session_config=session_config,
        save_summary_steps=5,
        log_step_count_steps=5
    )  # set the frequency of logging steps for loss function

    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
                                     params={'learning_rate': learning_rate, 'batch_size': batch_size, 'model_dir': model_dir,
                                             'loss_fn': loss_fn, 'deformable_range': deformable_range, 'model_name': model_name,
                                             'loss_mask': loss_mask, 'output_flg':False, 'training_set': training_set})


    train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_input_fn(training_set=training_set, filenames=train_data_path, height=image_shape[0], width=image_shape[1],
                                                               shuffle=True, repeat_count=-1, batch_size=batch_size),
                                        max_steps=traing_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                                                             shuffle=False, repeat_count=1, batch_size=batch_size),
                                      steps=None, throttle_secs=evaluate_steps)
    tf.estimator.train_and_evaluate(tof_net, train_spec, eval_spec)

def dataset_testing(evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                    loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape):
    for i in range(gpu_Number):
        session_config = tf.ConfigProto(device_count=gpu_device_number_list[i])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=session_config,
        log_step_count_steps = 10,
        save_summary_steps = 5,)
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
        params={'learning_rate': 1e-4, 'batch_size': batch_size, 'model_dir': model_dir,
                'deformable_range': deformable_range, 'loss_fn':loss_fn,
                'model_name': model_name, 'loss_mask': loss_mask, 'output_flg':False, 'training_set': training_set})
    tof_net.evaluate(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
        shuffle=False, repeat_count=1, batch_size=batch_size), checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps)

def dataset_output(result_path, evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                   loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape):
    for i in range(gpu_Number):
        session_config = tf.ConfigProto(device_count=gpu_device_number_list[i])
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=session_config,
        log_step_count_steps = 10,
        save_summary_steps = 5,)
    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration,
        params={'learning_rate': 1e-4, 'batch_size': batch_size, 'model_dir': model_dir,
                'deformable_range': deformable_range, 'loss_fn':loss_fn,
                'model_name': model_name, 'loss_mask': loss_mask, 'output_flg':True, 'training_set': training_set})
    result = list(tof_net.predict(input_fn=lambda: get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                shuffle=False, repeat_count=1, batch_size=batch_size), checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps))

    for i in range(len(result)):
        pre_depth = result[i]['depth']
        offset = result[i]['offset']

        pre_depth = np.reshape(pre_depth, -1).astype(np.float32)
        offset = np.reshape(offset, -1).astype(np.float32)

        pre_depth.tofile(result_path + '/pre_depth_'+str(i))
        offset.tofile(result_path + '/offset_'+str(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deformable KPN Network')
    parser.add_argument("-t", "--trainingSet", help='the name to the list file with training set', default = 'FLAT_reflection_s5', type=str)

    parser.add_argument("-m", "--modelName", help="name of the denoise model to be used", default="deformable_kpn")
    parser.add_argument("-l", "--lr", help="initial value for learning rate", default=1e-5, type=float)
    parser.add_argument("-i", "--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[424, 512])
    parser.add_argument("-b", "--batchSize", help='batch size to use during training', type=int, default=4)
    parser.add_argument("-s", "--steps", help='number of training steps', type=int, default=4000)
    parser.add_argument("-e", "--evalSteps", help='after the number of training steps to eval', type=int, default=100)
    parser.add_argument("-o", '--lossType', help="Type of supervised loss to use, such as mean_l2, mean_l1, sum_l2, sum_l1, smoothness, SSIM", default="mean_l2", type=str)
    parser.add_argument("-d", "--deformableRange", help="the range of deformable kernel", default=192.0, type=float)
    parser.add_argument("-f", '--flagMode', help="The flag that select the runing mode, such as train, eval, output", default='train', type=str)
    parser.add_argument("-p", '--postfix', help="the postfix of the training task", default=None, type=str)
    parser.add_argument("-c", '--checkpointSteps', help="select the checkpoint of the model", default="800", type=str)
    parser.add_argument("-k", '--lossMask', help="the mask used in compute loss", default='gt_msk', type=str)
    parser.add_argument("-g", '--gpuNumber', help="The number of GPU used in training", default=2, type=int)
    args = parser.parse_args()

    dataset_dir = '/userhome/dataset/tfrecords'
    model_dir = './result/kinect/' + args.modelName
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.modelName[0:10] == 'deformable':
        mkdir_name = args.trainingSet + '_' + args.lossType + '_dR' + str(args.deformableRange)
    else:
        mkdir_name = args.trainingSet + '_' + args.lossType
    if args.postfix is not None:
        model_dir = os.path.join(model_dir, mkdir_name + '_' + args.postfix)
    else:
        model_dir = os.path.join(model_dir, mkdir_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    output_dir = os.path.join(model_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dataset_path = os.path.join(dataset_dir, args.trainingSet)
    train_data_path = os.path.join(dataset_path, args.trainingSet + '_train.tfrecords')
    evaluate_data_path = os.path.join(dataset_path, args.trainingSet + '_eval.tfrecords')


    if args.flagMode == 'train':
        dataset_training(train_data_path=train_data_path, evaluate_data_path=evaluate_data_path, loss_fn=args.lossType,
                         model_dir=model_dir, learning_rate=args.lr, batch_size=args.batchSize, traing_steps=args.steps,
                         evaluate_steps=args.evalSteps, deformable_range = args.deformableRange, model_name=args.modelName,
                         checkpoint_steps=args.checkpointSteps, loss_mask = args.lossMask, gpu_Number = args.gpuNumber,
                         training_set = args.trainingSet, image_shape = args.imageShape
                         )
    elif args.flagMode == 'eval':
        dataset_testing(evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,batch_size=args.batchSize,
                        checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange, model_name = args.modelName,
                        loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet, image_shape = args.imageShape)

    else:
        dataset_output(result_path=output_dir,evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        batch_size=args.batchSize, checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange,
                        model_name = args.modelName, loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet,
                        image_shape = args.imageShape)