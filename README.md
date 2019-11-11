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
