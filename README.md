# The program that remove the multi-path inference in ToF depth

This repository provides a deformable kernel denoise methods for time-of-flight (ToF) artifacts removal.

## How to use the code

environment
```
tensorflow-gpu==1.12.0 
Pillow
scikit-image
six
```

the project starts with the 'start.py'. Through this file, you can select different models, data sets and loss functions for training, and you can switch between train, eval and output modes by adjusting parameters
The parameters available are as follows

```
Arg
├───modelName		# Select the model required during training
│   ├───deeptof               # Completed
│   ├───deformable_kpn        # Completed
│   ├───deformable_kpn_raw    # Completed but not debug (raw express the netword is trained by raw measurement)
│   ├───kpn_raw               # Completed but not debug (FLAT: MRM)
│   ├───kpn                   # Not completed 
│   ├───deformable_ddfn_kpn   # Not completed (DDFN in "Real-world Image Denoising with Deep Boosting")
│   ├───tof_kpn               # Not completed (network in "Deep End-to-End Alignment and Refinement for Time-of-Flight RGB-D Module")
│   ├───jdpn                  # Not completed (network in "Deformable Kernel Network for Joint Image Filtering")
│   └───coarse_fine_unet      # Not completed (network in "Deep Learning for MPI Removal in ToF Sensors")
├───trainingSet		# Select the dataset required during training
│   ├───FLAT_reflection_s5    # noise: MPI ,image shape 424 512
│   ├───FLAT_full_s5          # noise: MPI + shotnoise ,image shape 424 512
│   ├───deeptof_reflection    # noise: MPI ,image shape 256 256
│   └───ToFFlyingThings3D     # noise: MPI ,image shape 384 512 (not completed)
├───flagMode		# Select the running mode of the code
│   ├───train                 # train model
│   ├───eval_ED               # evaluate model in test sets
│   ├───eval_TD               # evaluate model in training sets
│   └───output                # output depth prediction, offsets, weight
├───gpuNumber		# The number of GPU used in training
├───addGradient		# weather add the gradient loss function
├───decayEpoch		# after n epochs, decay the learning rate
├───lossType		# Select the loss function in training
│   ├───mean_l1               # the mean of L1 loss between input and gt
│   ├───mean_l2               # the mean of L2 loss
│   ├───sum_l1                # the sum of L1 loss
│   ├───sum_l2                # the sum of L2 loss
│   ├───smoothness            # depth map smoothness loss
│   ├───SSIM                  # the sum of structural similarity loss 
│   ├───SSIM_l1               # the sum of structural similarity loss + L1 loss
│   ├───mean_SSIM             # the mean of structural similarity loss 
│   ├───ZNCC                  # Zero Mean Normalized Cross-Correlation loss
│   ├───cos_similarity        # cos_similarity
│   ├───mean_huber            # the mean of huber loss
│   └───sum_huber             # the sum of huber loss
└───lossMask	    # Select the loss mask to be used during training
    ├───gt_msk                # Non-zero region in groundtruth
    ├───depth_kinect_msk      # Non-zero region in depth input
    └───depth_kinect_with_gt_msk      # gt_msk with depth_kinect_msk
```
For example

```
python start.py -b 2 -s 200000 -m sample_pyramid_add_kpn -p size384 -k depth_kinect_with_gt_msk -l 0.0004 -t tof_FT3 -i 480 640 -o mean_l1 --addGradient sobel_gradient -g 4 -e 1200
```
   

