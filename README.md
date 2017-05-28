# Depth Based Object Tracking Library (dbot)

The core of this library are two probabilistic tracker
 * A Non-Parametric Tracker Based on Rao-Blackwellized Coordinate Descent Particle Filter
   
   http://arxiv.org/abs/1505.00241
   
   This tracker can run on a pure CPU system or optionally run on GPU using CUDA 6.5 or later

```
   inproceedings{wuthrich-iros-2013,
     title = {Probabilistic Object Tracking Using a Range Camera},
     author = {W{\"u}thrich, M. and Pastor, P. and Kalakrishnan, M. and Bohg, J. and Schaal, S.},
     booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
     pages = {3195-3202},
     publisher = {IEEE},
     month = nov,
     year = {2013},
     month_numeric = {11}
  }
```
 
 * A Parametric Tracker Based on Robust Multi-Sensor Gaussian Filter Tracker

   http://arxiv.org/abs/1602.06157
   
```
   @inproceedings{jan_ICRA_2016,
     title = {Depth-based Object Tracking Using a Robust Gaussian Filter},
     author = {Issac, Jan and W{\"u}thrich, Manuel and Garcia Cifuentes, Cristina and Bohg, Jeannette and Trimpe, Sebastian and Schaal, Stefan},
     booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) 2016},
     publisher = {IEEE},
     month = may,
     year = {2016},
     url = {http://arxiv.org/abs/1602.06157},
     month_numeric = {5}
   }
```

All trackers require mesh models of the tracked objects in Wavefront (.obj) format.

# Requirements
 * Ubuntu 14.04
 * C++11 Compiler (gcc-4.7 or later)
 * [CUDA](https://developer.nvidia.com/cuda-downloads) 6.5 or later (optional)
 
## Dependecies
 * [Filtering library](https://github.com/filtering-library/fl) (fl)
 * [Eigen](http://eigen.tuxfamily.org/) 3.2.1 or later
 
# Compiling
 The cmake package uses [Catkin](https://github.com/ros/catkin). If you have installed ROS groovy or later, then you most likely have catkin installed already.

     $ cd $HOME
     $ mkdir -p projects/tracking/src  
     $ cd projects/tracking/src
     $ git clone git@github.com:filtering-library/fl.git
     $ git clone git@github.com:bayesian-object-tracking/dbot.git
     $ cd ..
     $ catkin_make -DCMAKE_BUILD_TYPE=Release -DDBOT_BUILD_GPU=On

If no CUDA enabled device is available, you can deactivate the GPU implementation via 

     $ catkin_make -DCMAKE_BUILD_TYPE=Release -DDBOT_BUILD_GPU=Off


# How to use dbot

Checkout the ros nodes of each tracker in [dbot_ros](https://github.com/bayesian-object-tracking/dbot_ros) package for exact usage of the filters.

 
