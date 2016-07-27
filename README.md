# Depth Based Object Tracking Library (dbot)

The core of this library are two probabilistic tracker
 * A Non-Parametric Tracker Based on Rao-Blackwellized Coordinate Descent Particle Filter
 
  [M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal. Probabilistic Object Tracking using a Range Camera IEEE Intl Conf on Intelligent Robots and Systems, 2013]

  http://arxiv.org/abs/1505.00241
  
  This tracker can run on a pure CPU system or optionally run on GPU using CUDA 6.5 or later

 * A Parametric Tracker Based on Robust Multi-Sensor Gaussian Filter Tracker
 
   [J. Issac, M. Wuthrich, C. Garcia Cifuentes, J. Bohg, S. Trimpe, S. Schaal
   Depth-Based Object Tracking Using a Robust Gaussian Filter
   IEEE Intl Conf on Robotics and Automation, 2016] 

   http://arxiv.org/abs/1602.06157

# Requirements
 * Ubuntu 12.04
 * C++0x or C++11 Compiler (gcc-4.6 or later)
 * [CUDA](https://developer.nvidia.com/cuda-downloads) 6.5 or later (optional)
 
## Dependecies
 * [Filtering library](https://github.com/filtering-library/fl) (fl)
 * [Eigen](http://eigen.tuxfamily.org/) 3.2.1 or later
 
# Compiling
 The cmake package uses [Catkin](https://github.com/ros/catkin). If you have installed ROS groovy or later, then you most likely have catkin installed already.

     $ cd $HOME
     $ mkdir -p projects/tracking/src  
     $ cd projects/tracking/src
     $ git clone git@github.com:bayesian-object-tracking/dbot.git
     $ cd ..
     $ catkin_make -DCMAKE_BUILD_TYPE=Release -DDBOT_BUILD_GPU=On

If no CUDA enabled device is available, you can deactivate the GPU implementation via 

     $ catkin_make -DCMAKE_BUILD_TYPE=Release -DDBOT_BUILD_GPU=Off


# How to use dbot

Checkout the ros nodes of each tracker in [dbot_ros](https://github.com/bayesian-object-tracking/dbot_ros) package for exact usage of the filters.

 
