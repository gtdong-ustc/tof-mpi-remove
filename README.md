# The program that remove the multi-path inference in ToF depth

This repository provides a deformable kernel denoise methods for time-of-flight (ToF) artifacts removal.

## Organization and access of the FLAT dataset
 The FLAT dataset is organized in the following way:
```
FLAT
├───deeptof		# simulation of DeepToF
│   ├───full			# raw measurements
│   └───list			# lists of files to use for each task, automatically generated from ./sim/deeptof_prepare.py
├───kinect		# simulation of kinect 2
│   ├───full			# raw measurements
│   ├───noise			# raw measurements without multi path interference (MPI), with noise
│   ├───ideal			# raw measurements without MPI and noise
│   ├───reflection		# raw measurements with MPI, and without noise
│   ├───gt			# true depth
│   ├───list			# lists of files to use for each task, automatically generated from ./sim/kinect_prepare.py
│   │   ├───all				# visualization of all scenes 
│   │   ├───motion_background		# visualization of scenes for a certain task
│   │   ├───motion_foreground		# visualization of scenes for a certain task
│   │   ├───motion_real			# visualization of scenes for a certain task
│   │   ├───shot_noise_test		# visualization of scenes for a certain task
│   │   ├───test			# visualization of scenes for a certain task
│   │   ├───test_dyn			# visualization of scenes for a certain task
│   │   ├───train			# visualization of scenes for a certain task
│   │   └───val				# visualization of scenes for a certain task
│   └───msk			# the mask of background
├───phasor		# simulation of Phasor
│   ├───full			# raw measurements
│   │   ├───FILE_ID		
│   └───list			# lists of files to use for each task, automatically generated from ./sim/phasor_prepare.py
└───trans_render	# transient rendering files
    ├───dyn			# dynamic scenes
    └───static			# static scenes
```

##How to use the code
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
│   ├───deeptof			   # Completed
│   ├───deformable_kpn		   # Completed
│   ├───deformable_kpn_raw	   # Completed but not debug (raw express the netword is trained by raw measurement)
│   ├───kpn_raw		       # Completed but not debug (FLAT: MRM)
│   ├───kpn		           # Not completed 
│   ├───deformable_ddfn_kpn   # Not completed (DDFN in "Real-world Image Denoising with Deep Boosting")
│   ├───tof_kpn               # Not completed (network in "Deep End-to-End Alignment and Refinement for Time-of-Flight RGB-D Module")
│   └───coarse_fine_unet	   # Not completed (network in "Deep Learning for MPI Removal in ToF Sensors")
├───trainingSet		# Select the dataset required during training
│   ├───FLAT_reflection_s5	   # noise: MPI ,image shape 424 512
│   ├───FLAT_full_s5		   # noise: MPI + shotnoise ,image shape 424 512
│   ├───deeptof_reflection	   # noise: MPI ,image shape 256 256
│   └───ToFFlyingThings3D	   # noise: MPI ,image shape 384 512 (not completed)
├───flagMode		# Select the running mode of the code
│   ├───train			      # train model
│   ├───eval			      # evaluate model
│   └───output			      # output depth prediction, offsets, weight
└───lossMask	    # Select the loss mask to be used during training
    ├───gt_msk			      # Non-zero region in groundtruth
    └───depth_kinect_msk	  # Non-zero region in depth input
```

   

