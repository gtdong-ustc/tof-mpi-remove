# import tensorflow as tf
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import cv2
import argparse



def random_crop_array_and_upsample(input, h_idx, w_idx, scale):
    h_max = input.shape[0]
    w_max = input.shape[1]
    channel_number = input.shape[2]

    h_stride = int(h_max / scale)
    w_stride = int(w_max / scale)

    h_stride_med = int(h_stride / 2)
    w_stride_med = int(w_stride / 2)

    input_region = input[h_idx - h_stride_med : h_idx + h_stride_med, w_idx - w_stride_med : w_idx + w_stride_med, :]
    input_region_upsample = resize(input_region, (h_max, w_max, channel_number))

    return input_region_upsample

def get_offset_map_amd_upsample(offsets, h_idx, w_idx, scale): ######## offsets : H,W,C=18
    N = 9
    offsets_scale = offsets * scale * scale
    h_max = offsets_scale.shape[0]
    w_max = offsets_scale.shape[1]

    h_med = int(h_max /2)
    w_med = int(w_max /2)

    h_w_reshape_size = [h_max, w_max, N, 2]

    offsets_scale = np.reshape(offsets_scale, h_w_reshape_size)
    coords_h = offsets_scale[:,:,:,0]
    coords_w = offsets_scale[:,:,:,1]
    print('done')
    coords_h_samples = []
    coords_w_samples = []
    for i in range(N):
        coords_h_samples.append(coords_h[h_idx, w_idx, i])
        coords_w_samples.append(coords_w[h_idx, w_idx, i])


    h0_samples = np.floor(coords_h_samples)
    h1_samples = h0_samples + 1
    w0_samples = np.floor(coords_w_samples)
    w1_samples = w0_samples + 1

    h0_idx_samples = h0_samples + h_med
    h1_idx_samples = h1_samples + h_med

    w0_idx_samples = w0_samples + w_med
    w1_idx_samples = w1_samples + w_med


    inside_msk_sum = np.array(0 <= h0_idx_samples).astype(np.float32) + np.array(0 <= w0_idx_samples).astype(np.float32) + \
                     np.array(h1_idx_samples < h_max).astype(np.float32) + np.array(w1_idx_samples < w_max).astype(np.float32)
    inside_msk = inside_msk_sum > 3
    h0_idx_samples = h0_idx_samples * inside_msk
    h1_idx_samples = h1_idx_samples * inside_msk
    w0_idx_samples = w0_idx_samples * inside_msk
    w1_idx_samples = w1_idx_samples * inside_msk


    samples_visulization = np.zeros((h_max, w_max, 1), dtype=np.float32)
    for i in range(N):
        samples_visulization[int(h0_idx_samples[i]), int(w0_idx_samples[i]), 0] = 1.5
        samples_visulization[int(h0_idx_samples[i]), int(w1_idx_samples[i]), 0] = 1.5
        samples_visulization[int(h1_idx_samples[i]), int(w0_idx_samples[i]), 0] = 1.5
        samples_visulization[int(h1_idx_samples[i]), int(w1_idx_samples[i]), 0] = 1.5

    samples_visulization[int(h_med), int(w_med), 0] = 2.0
    return samples_visulization


def visulization(input, offset,scale,h_coords, w_coords): ###input : H,W,C
    input_region = random_crop_array_and_upsample(input, h_coords, w_coords, scale)
    sample_coords = get_offset_map_amd_upsample(offset, h_coords, w_coords, scale)
    input_region = np.squeeze(input_region)
    sample_coords = np.squeeze(sample_coords)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('pre_depth')
    plt.imshow(input_region)
    plt.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('samples')
    plt.imshow(sample_coords)
    plt.axis('off')
    plt.show()

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        visulization(pre_depth, offset, scale=scale, h_coords=y, w_coords=x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deformable KPN Network')
    parser.add_argument("-n", "--numberMap", help='select the depth map that need to visulization', default = 1, type=int)
    parser.add_argument("-s", "--scale", help='select the expand scale that used in visulization', default=1, type=int)
    args = parser.parse_args()

    number = args.numberMap
    scale = args.scale

    offset_path = 'D:/tensorboard_file/FLAT/deformable_kpn_half/x5_mean_l2_dR192.0_depth_kinect_msk/output/offset_' + number
    pre_depth_path = 'D:/tensorboard_file/FLAT/deformable_kpn_half/x5_mean_l2_dR192.0_depth_kinect_msk/output/pre_depth_' + number
    result_path = 'D:/tensorboard_file/FLAT/deformable_kpn_half/x5_mean_l2_dR192.0_depth_kinect_msk/output/png/' + number

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(offset_path,'rb') as f:
        offset = np.fromfile(f, dtype=np.float32)
    with open(pre_depth_path,'rb') as f:
        pre_depth = np.fromfile(f, dtype=np.float32)
    offset = np.reshape(offset, (384, 512, 18))
    pre_depth = np.reshape(pre_depth, (384, 512, 1))
    if not os.path.exists(result_path + '/' + 'pre_depth.png'):
        pre_depth_png = np.squeeze(pre_depth)
        cv2.imwrite(result_path + '/' + 'pre_depth.png', pre_depth_png * 75)

    img = cv2.imread(result_path + '/' + 'pre_depth.png')
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)

    while (True):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyAllWindows()
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


