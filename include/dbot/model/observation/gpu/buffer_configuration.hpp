/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file object_resterizer.hpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date March 2016
 */

#include <dbot/model/observation/gpu/object_rasterizer.hpp>
#include <dbot/model/observation/gpu/cuda_likelihood_evaluator.hpp>

class BufferConfiguration {
public:
  BufferConfiguration(ObjectRasterizer* rasterizer,
                      CudaEvaluator* evaluator,
                      int nr_max_poses);


private:
  ObjectRasterizer* rasterizer_;
  CudaEvaluator* evaluator_;

  int nr_max_poses_;
  int nr_max_poses_per_row_;
  int nr_max_poses_per_col_;
  int nr_poses_;
  int nr_poses_per_row_;
  int nr_poses_per_column_;
  int nr_cols_;
  int nr_rows_;

  // GPU contraints
  int max_texture_size_opengl_;
  cudaDeviceProp cuda_device_properties_;
};
