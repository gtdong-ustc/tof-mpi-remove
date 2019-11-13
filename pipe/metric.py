import sys

sys.path.insert(0, '../sim/')
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_metrics_psnr(depth, ori_depth, gt, msk):
    """
    This function be used to return the PSNR of depth and gt in eval mode
    :param depth:
    :param ori_depth:
    :param gt:
    :param msk:
    :return:
    """
    ori_mse, update_op_ori = tf.metrics.mean_squared_error(gt, ori_depth)
    ori_psnr = 10 * (tf.log(25.0 / ori_mse) / tf.log(10.0))
    update_op_ori = 10 * (tf.log(25.0 / update_op_ori) / tf.log(10.0))
    pre_mse, update_op_pre = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr = 10 * (tf.log(25.0 / pre_mse) / tf.log(10.0))
    update_op_pre = 10 * (tf.log(25.0 / update_op_pre) / tf.log(10.0))

    depth = depth * msk
    ori_depth = ori_depth * msk
    ori_mse_dm, update_op_ori_dm = tf.metricKs.mean_squared_error(gt, ori_depth)
    ori_psnr_dm = 10 * (tf.log(25.0 / ori_mse_dm) / tf.log(10.0))
    update_op_ori_dm = 10 * (tf.log(25.0 / update_op_ori_dm) / tf.log(10.0))
    # pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt * depth_msk, depth)
    pre_mse_dm, update_op_pre_dm = tf.metrics.mean_squared_error(gt, depth)
    pre_psnr_dm = 10 * (tf.log(25.0 / pre_mse_dm) / tf.log(10.0))
    update_op_pre_dm = 10 * (tf.log(25.0 / update_op_pre_dm) / tf.log(10.0))

    return (ori_psnr, update_op_ori), (pre_psnr, update_op_pre), \
           (ori_psnr_dm, update_op_ori_dm), (pre_psnr_dm, update_op_pre_dm)

def get_metrics_mae(depth, ori_depth, gt, msk):
    """
    This function be used to return the MAE of depth and gt in eval mode
    :param depth:
    :param ori_depth:
    :param gt:
    :param msk:
    :return:
    """
    if msk == None:
        ori_mae, update_ori_mae = tf.metrics.mean_absolute_error(gt , ori_depth)
        pre_mae, update_pre_mae = tf.metrics.mean_absolute_error(gt , depth)
    else:
        msk_one = tf.ones_like(gt, dtype=tf.float32)

        gt = gt * msk
        depth = depth * msk
        ori_depth = ori_depth * msk

        msk_coeff = tf.reduce_sum(msk_one) / tf.reduce_sum(msk)

        ori_mae, update_ori_mae = tf.metrics.mean_absolute_error(gt * msk_coeff, ori_depth * msk_coeff)
        pre_mae, update_pre_mae = tf.metrics.mean_absolute_error(gt * msk_coeff, depth * msk_coeff)
    return (ori_mae, update_ori_mae), (pre_mae, update_pre_mae)
