import sys

sys.path.insert(0, '../sim/')
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def SSIM(x, y):
    """
    SSIM dissimilarity measure
    Args:
        x: predicted image
        y: target image
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')
    mu_y = tf.nn.avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')

    sigma_x = tf.nn.avg_pool(x ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
def mean_SSIM(x,y):
	"""
	Mean error over SSIM reconstruction
	"""
	return tf.reduce_mean(SSIM(x,y))


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
    ori_mse_dm, update_op_ori_dm = tf.metrics.mean_squared_error(gt, ori_depth)
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
        pre_mae_percent_25, update_pre_mae_percent_25 = tf.metrics.mean_absolute_error(gt , depth)
        pre_mae_percent_50, update_pre_mae_percent_50 = tf.metrics.mean_absolute_error(gt , depth)
        pre_mae_percent_75, update_pre_mae_percent_75 = tf.metrics.mean_absolute_error(gt , depth)

    else:
        msk_one = tf.ones_like(gt, dtype=tf.float32)

        gt_tmp = gt * msk
        depth_tmp = depth * msk
        ori_depth_tmp = ori_depth * msk
        msk_one_sum = tf.reduce_sum(msk_one)
        msk_sum = tf.reduce_sum(msk)

        msk_coeff = msk_one_sum / msk_sum

        msk_one_sum = tf.cast(msk_one_sum, dtype=tf.int32)
        msk_sum = tf.cast(msk_sum, dtype=tf.int32)
        percent_25 = tf.cast(msk_sum / 4, dtype=tf.int32)
        percent_50 = percent_25 * 2
        percent_75 = percent_25 * 3
        msk_sum_diff = msk_one_sum - msk_sum
        error_array = tf.abs(ori_depth_tmp-gt_tmp)
        error_tensor = tf.reshape(error_array, [-1])
        error_tensor = tf.contrib.framework.sort(error_tensor, direction='ASCENDING')

        # max_value = tf.reduce_max(error_tensor)
        # min_value = tf.reduce_min(error_tensor)
        min_error_value = tf.gather(error_tensor, indices=msk_sum_diff)
        percent_25_error_value = tf.gather(error_tensor, indices=msk_sum_diff + percent_25)
        percent_50_error_value = tf.gather(error_tensor, indices=msk_sum_diff + percent_50)
        percent_75_error_value = tf.gather(error_tensor, indices=msk_sum_diff + percent_75)
        # min_error_value = min_value
        # percent_25_error_value = min_value + 0.25 * (max_value-min_value)
        # percent_50_error_value = min_value + 0.5 * (max_value-min_value)
        # percent_75_error_value = min_value + 0.75 * (max_value-min_value)

        # min_error_array = min_error_value * msk_one
        # percent_25_error_array = percent_25_error_value * msk_one
        # percent_50_error_array = percent_50_error_value * msk_one
        # percent_75_error_array = percent_75_error_value * msk_one
        # msk_percent_25 = tf.cast(tf.greater(error_array, min_error_array),dtype=tf.float32) * \
        #                  tf.cast(tf.greater(percent_25_error_array, error_array),dtype=tf.float32)
        # msk_percent_50 = tf.cast(tf.greater(error_array, percent_25_error_array),dtype=tf.float32) * \
        #                  tf.cast(tf.greater(percent_50_error_array, error_array),dtype=tf.float32)
        # msk_percent_75 = tf.cast(tf.greater(error_array, percent_50_error_array),dtype=tf.float32) * \
        #                  tf.cast(tf.greater(percent_75_error_array, error_array),dtype=tf.float32)

        msk_percent_25 = tf.cast(error_array > min_error_value, dtype=tf.float32) * \
                         tf.cast(error_array < percent_25_error_value, dtype=tf.float32)
        msk_percent_50 = tf.cast(error_array > percent_25_error_value, dtype=tf.float32) * \
                         tf.cast(error_array < percent_50_error_value, dtype=tf.float32)
        msk_percent_75 = tf.cast(error_array > percent_50_error_value, dtype=tf.float32) * \
                         tf.cast(error_array < percent_75_error_value, dtype=tf.float32)

        # msk_percent_25 = tf.cast(error_array > percent_25_error_value, dtype=tf.float32)
        # msk_percent_50 = tf.cast(error_array > percent_50_error_value, dtype=tf.float32)
        # msk_percent_75 = tf.cast(error_array > percent_75_error_value, dtype=tf.float32)

        # one = tf.ones_like(error_array, dtype=tf.float32)
        # zero = tf.zeros_like(error_array, dtype=tf.float32)
        # msk_percent_25 = tf.where(error_array < percent_25_error_value, x =one, y=zero)
        # msk_percent_50 = tf.where(error_array < percent_50_error_value, x=one, y=zero)
        # msk_percent_75 = tf.where(error_array < percent_75_error_value, x=one, y=zero)


        # msk_one_sum = tf.cast(msk_one_sum, dtype=tf.float32)
        # msk_coeff_percent_25 = msk_one_sum / tf.reduce_sum(msk_percent_25)
        # msk_coeff_percent_50 = msk_one_sum / tf.reduce_sum(msk_percent_50)
        # msk_coeff_percent_75 = msk_one_sum / tf.reduce_sum(msk_percent_75)
        # msk_coeff_percent_25 = msk_one_sum / msk_one_sum
        # msk_coeff_percent_50 = msk_one_sum / msk_one_sum
        # msk_coeff_percent_75 = msk_one_sum / msk_one_sum

        msk_coeff_percent_25 = 4 * msk_coeff
        msk_coeff_percent_50 = 4 * msk_coeff
        msk_coeff_percent_75 = 4 * msk_coeff
        gt_tmp_percent_25 = gt * msk_percent_25
        depth_tmp_percent_25 = depth * msk_percent_25
        gt_tmp_percent_50 = gt * msk_percent_50
        depth_tmp_percent_50 = depth * msk_percent_50
        gt_tmp_percent_75 = gt * msk_percent_75
        depth_tmp_percent_75 = depth * msk_percent_75


        ori_mae, update_ori_mae = tf.metrics.mean_absolute_error(gt_tmp * msk_coeff, ori_depth_tmp * msk_coeff)
        pre_mae, update_pre_mae = tf.metrics.mean_absolute_error(gt_tmp * msk_coeff, depth_tmp * msk_coeff)
        pre_mae_percent_25, update_pre_mae_percent_25 = tf.metrics.mean_absolute_error(gt_tmp_percent_25 * msk_coeff_percent_25,
                                                                                       depth_tmp_percent_25 * msk_coeff_percent_25)
        pre_mae_percent_50, update_pre_mae_percent_50 = tf.metrics.mean_absolute_error(gt_tmp_percent_50 * msk_coeff_percent_50,
                                                                                       depth_tmp_percent_50 * msk_coeff_percent_50)
        pre_mae_percent_75, update_pre_mae_percent_75 = tf.metrics.mean_absolute_error(gt_tmp_percent_75 * msk_coeff_percent_75,
                                                                                       depth_tmp_percent_75 * msk_coeff_percent_75)
    return (ori_mae, update_ori_mae), (pre_mae, update_pre_mae), (pre_mae_percent_25, update_pre_mae_percent_25),\
           (pre_mae_percent_50, update_pre_mae_percent_50), (pre_mae_percent_75, update_pre_mae_percent_75)
