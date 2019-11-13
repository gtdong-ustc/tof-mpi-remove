# The program that remove the multi-path inference in ToF depth

This repository provides a deformable kernel denoise methods for time-of-flight (ToF) artifacts removal.

## How to use the code

the project starts with the 'start.py'. Through this file, you can select different models, data sets and loss functions for training, and you can switch between train, eval and output modes by adjusting parameters
```
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
    parser.add_argument("-c", '--checkpointSteps', help="select the checkpoint of the model in finetune or evaluate", default="800", type=str)
    parser.add_argument("-k", '--lossMask', help="the mask used in compute loss", default='gt_msk', type=str)
    parser.add_argument("-g", '--gpuNumber', help="The number of GPU used in training", default=2, type=int)

```

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
│   ├───eval                  # evaluate model
│   └───output                # output depth prediction, offsets, weight
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
    └───depth_kinect_msk      # Non-zero region in depth input
```
For example

```
python start -b 8 -s 1000 -l 0.0001 -t FLAT_reflection_s5 -m deformable_kpn -o mean_l2 
```
   

