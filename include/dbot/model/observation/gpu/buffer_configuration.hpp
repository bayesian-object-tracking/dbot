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
 * \file buffer_configuration.hpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date March 2016
 */


#pragma once


#include <dbot/model/observation/gpu/object_rasterizer.hpp>
#include <dbot/model/observation/gpu/cuda_likelihood_evaluator.hpp>
#include "boost/shared_ptr.hpp"

class BufferConfiguration {
public:
  BufferConfiguration(boost::shared_ptr<ObjectRasterizer> rasterizer,
                      boost::shared_ptr<CudaEvaluator> evaluator,
                      int max_nr_poses,
                      int nr_rows,
                      int nr_cols);

  bool allocate_memory(int max_nr_poses,
                       int& new_max_nr_poses);
  bool set_resolution(const int nr_rows, const int nr_cols,
                       int& new_max_nr_poses);
  bool set_nr_of_poses(const int nr_poses,
                       int& new_nr_poses);
  bool set_number_of_threads(const int nr_threads,
                             int& new_nr_threads);

  /// Enable automatic adaptation to GPU constraints
  /** \param [in] should_adapt whether or not the number of poses should
   *  be downgraded in case the GPU hardware limits are reached
   */
  void set_adapt_to_constraints(bool should_adapt);

private:

  enum message_type {WARNING, ERROR};

  bool check_against_texture_size_constraint(const int nr_poses,
                                             const int nr_poses_per_row,
                                             const int nr_poses_per_col,
                                             int& new_nr_poses);
  void compute_grid_layout(const int nr_poses,
                           int& nr_poses_per_row,
                           int& nr_poses_per_col);
  bool check_against_global_memory_constraint(const int nr_poses,
                                              int& new_nr_poses,
                                              int& new_nr_poses_per_row,
                                              int& new_nr_poses_per_col);
  bool check_against_thread_constraint(const int nr_threads,
                                        int& new_nr_threads);
  void issue_message(const BufferConfiguration::message_type foo,
                     const std::string problem_quantity,
                     const std::string constraint,
                     const std::string original_value,
                     const std::string new_value = "");

  boost::shared_ptr<ObjectRasterizer> rasterizer_;
  boost::shared_ptr<CudaEvaluator> evaluator_;

  int max_nr_poses_;
  int max_nr_poses_per_row_;
  int max_nr_poses_per_col_;
  int nr_poses_;
  int nr_poses_per_row_;
  int nr_poses_per_column_;
  int nr_cols_;
  int nr_rows_;
  int nr_threads_;

  bool adapt_to_constraints_;

  // GPU contraints
  int max_texture_size_opengl_;
  cudaDeviceProp cuda_device_properties_;

};
