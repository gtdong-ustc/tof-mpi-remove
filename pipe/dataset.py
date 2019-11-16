import sys

sys.path.insert(0, '../sim/')
import tensorflow as tf
import matplotlib
from tof_class import *
from kinect_spec import *

from kinect_pipeline import kinect_mask_tensor

tf.logging.set_verbosity(tf.logging.INFO)
from kinect_init import *

tof_cam = kinect_real_tf()

PI = 3.14159265358979323846
flg = False
dtype = tf.float32

def colorize_img(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size,height, width,1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')

    Returns a 3D tensor of shape [batch_size,height, width,3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # quantize
    indices = tf.to_int32(tf.round(value[:, :, :, 0] * 255))

    # gather
    color_map = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = color_map(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value

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
    gt = tof_cam.dist_to_depth(gt)
    gt_p = gt[20:-20, :, :]
    features['full'] = meas_p
    labels['ideal'] = ideal_p
    labels['gt'] = gt_p
    return features, labels

def preprocessing_deeptof(features, labels):
    """
    not konw some preprocess pipeline needed to use
    :param features:
    :param labels:
    :return:
    """
    return features, labels

def preprocessing_tof_FT3(features, labels):
    """
    not konw some preprocess pipeline needed to use
    :param features:
    :param labels:
    :return:
    """
    rgb_list = []
    rgb = features['rgb']
    for i in range(3):
        rgb_list.append(rgb[:,:,i] - tf.reduce_mean(rgb[:,:,i]))

    rgb_p = tf.stack(rgb_list, axis=-1)
    features['rgb'] = rgb_p
    return features, labels

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

def imgs_input_fn_deeptof(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    def _parse_function(serialized, height=height, width=width):
        features = \
            {
                'amps': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
                'depth_ref': tf.FixedLenFeature([], tf.string)
            }

        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        amps_shape = tf.stack([height, width, 1])
        depth_shape = tf.stack([height , width , 1])
        depth_ref_shape = tf.stack([height, width, 1])

        amps_raw = parsed_example['amps']
        depth_raw = parsed_example['depth']
        depth_ref_raw = parsed_example['depth_ref']

        # decode the raw bytes so it becomes a tensor with type

        amps = tf.decode_raw(amps_raw, tf.float32)
        amps = tf.cast(amps, tf.float32)
        amps = tf.reshape(amps, amps_shape)

        depth = tf.decode_raw(depth_raw, tf.float32)
        depth = tf.cast(depth, tf.float32)
        depth = tf.reshape(depth, depth_shape)

        depth_ref = tf.decode_raw(depth_ref_raw, tf.float32)
        depth_ref = tf.cast(depth_ref, tf.float32)
        depth_ref = tf.reshape(depth_ref, depth_ref_shape)

        features = {'amps': amps, 'depth': depth}
        labels = {'depth_ref': depth_ref}

        return features, labels

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda features, labels: preprocessing_deeptof(features, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)  # Repeat the dataset this time
    batch_dataset = dataset.batch(batch_size)  # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()  # Make an iterator
    batch_features, batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels

    return batch_features, batch_labels

def imgs_input_fn_FT3(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    def _parse_function(serialized, height=height, width=width):
        features = \
            {
                'noisy': tf.FixedLenFeature([], tf.string),
                'intensity': tf.FixedLenFeature([], tf.string),
                'rgb': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }

        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        noisy_shape = tf.stack([height, width, 1])
        intensity_shape = tf.stack([height , width , 1])
        rgb_shape = tf.stack([height, width, 3])
        gt_shape = tf.stack([height, width, 1])

        noisy_raw = parsed_example['noisy']
        intensity_raw = parsed_example['intensity']
        rgb_raw = parsed_example['rgb']
        gt_raw = parsed_example['gt']

        # decode the raw bytes so it becomes a tensor with type

        noisy = tf.decode_raw(noisy_raw, tf.float32)
        noisy = tf.cast(noisy, tf.float32)
        noisy = tf.reshape(noisy, noisy_shape)

        intensity = tf.decode_raw(intensity_raw, tf.float32)
        intensity = tf.cast(intensity, tf.float32)
        intensity = tf.reshape(intensity, intensity_shape)

        rgb = tf.decode_raw(rgb_raw, tf.float32)
        rgb = tf.cast(rgb, tf.float32)
        rgb = tf.reshape(rgb, rgb_shape)

        gt = tf.decode_raw(gt_raw, tf.float32)
        gt = tf.cast(gt, tf.float32)
        gt = tf.reshape(gt, gt_shape)

        features = {'noisy': noisy, 'intensity': intensity, 'rgb': rgb}
        labels = {'gt': gt}

        return features, labels

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda features, labels: preprocessing_tof_FT3(features, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)  # Repeat the dataset this time
    batch_dataset = dataset.batch(batch_size)  # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()  # Make an iterator
    batch_features, batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels

    return batch_features, batch_labels

def bilinear_interpolation(input, offsets, N, batch_size, deformable_range):
    """
    This function used to sample from depth map, a simple tf version of bilinear interpolation function.
    :param input:
    :param offsets:
    :param N:
    :param batch_size:
    :param deformable_range:
    :return:
    """
    # input_size = tf.shape(input)
    h_max_idx = input.shape.as_list()[1]
    w_max_idx = input.shape.as_list()[2]
    offsets_size = tf.shape(offsets)

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

def im2col(input, kernel_size = 3, batch_size_input = 1):

    h_pos_list = []
    w_pos_list = []

    h_max = input.shape.as_list()[1]
    w_max = input.shape.as_list()[2]
    batch_size = tf.shape(input)[0]

    padding_size = int((kernel_size - 1) / 2)
    input_padding = tf.pad(input, paddings=[[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]])
    w_pos, h_pos = tf.meshgrid(list(range(1, w_max + 1)), list(range(1, h_max + 1)))
    w_pos = tf.expand_dims(tf.expand_dims(w_pos, 0), -1)
    h_pos = tf.expand_dims(tf.expand_dims(h_pos, 0), -1)
    w_pos = tf.cast(w_pos, dtype=tf.float32)
    h_pos = tf.cast(h_pos, dtype=tf.float32)

    for i in range(0-padding_size, padding_size + 1, 1):
        for j in range(0-padding_size, padding_size + 1, 1):
            h_pos = h_pos + tf.cast(i, dtype=tf.float32)
            w_pos = w_pos + tf.cast(j, dtype=tf.float32)
            h_pos_list.append(h_pos)
            w_pos_list.append(w_pos)

    h_pos = tf.concat(h_pos_list, axis=-1)
    w_pos = tf.concat(w_pos_list, axis=-1)
    h_pos = tf.tile(h_pos, multiples=[batch_size, 1, 1, 1])
    w_pos = tf.tile(w_pos, multiples=[batch_size, 1, 1, 1])

    # tensor_batch = list(range(batch_size))
    # tensor_batch = tf.convert_to_tensor(tensor_batch)
    tensor_batch = tf.range(batch_size)
    tensor_batch = tf.reshape(tensor_batch, [batch_size, 1, 1, 1])
    tensor_batch = tf.tile(tensor_batch, multiples=[1, h_max, w_max, kernel_size ** 2])
    tensor_batch = tf.cast(tensor_batch, dtype=tf.float32)

    tensor_channel = tf.zeros(shape=[kernel_size ** 2], dtype=tf.float32)
    tensor_channel = tf.reshape(tensor_channel, [1, 1, 1, kernel_size ** 2])
    tensor_channel = tf.tile(tensor_channel, multiples=[batch_size, h_max, w_max, 1])
    tensor_channel = tf.cast(tensor_channel, dtype=tf.float32)

    idx = tf.stack([tensor_batch, h_pos, w_pos, tensor_channel], axis=-1)

    idx = tf.reshape(idx, [-1, 4])

    im = tf.gather_nd(input_padding, tf.cast(idx, dtype=tf.int32))

    output = tf.reshape(im, [batch_size, h_max, w_max, kernel_size ** 2])
    return output
"""
###
This function has been temporarily deleted
###
def dof_computer(dist, samples, batch_size, z_multiplier, coords_h_pos, coords_w_pos):
    N = samples.shape.as_list()[-1]
    XX_s, YY_s, ZZ_s = map2mesh_samples(samples, tof_cam.cam, batch_size, z_multiplier, yy_coords=coords_h_pos, xx_coords=coords_w_pos)
    XX, YY, ZZ = map2mesh(dist, tof_cam.cam, batch_size, z_multiplier)
    XX = tf.tile(XX, multiples=[1,1,1,N])
    YY = tf.tile(YY, multiples=[1, 1, 1, N])
    ZZ = tf.tile(ZZ, multiples=[1, 1, 1, N])
    dist = tf.tile(dist, multiples=[1, 1, 1, N])
    dof_samp_cur = tf.sqrt((XX-XX_s)**2 + (YY-YY_s)**2 + (ZZ-ZZ_s)**2)
    dof_samples = dof_samp_cur + samples + dist
    return dof_samples
"""



ALL_INPUT_FN = {
    'FLAT_reflection_s5': imgs_input_fn,
    'FLAT_full_s5': imgs_input_fn,
    'deeptof_reflection': imgs_input_fn_deeptof,
    'tof_FT3': imgs_input_fn_FT3
}

def get_input_fn(training_set, filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    base_input_fn = ALL_INPUT_FN[training_set]
    return base_input_fn(filenames, height, width, shuffle=shuffle, repeat_count=repeat_count, batch_size=batch_size)


