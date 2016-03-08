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

/**
 * \brief This class takes care of synchronizing the number of poses and other
 * common variables between the object rasterizer and the cuda evaluator.
 *
 * You can enable automatic adaptation to GPU constraints by calling
 * set_adapt_to_constraints(true). This will then automatically downscale
 * the number of poses to fit within the constraints the GPU has.
 */
class BufferConfiguration {
public:

  /**
   * \brief Constructor which needs a pointer to the rasterizer and evaluator.
   * \param [in] rasterizer a pointer to the object rasterizer
   * \param [in] evaluator a pointer to the cuda evaluator
   * \param [in] max_nr_poses_ the maximum number of poses that will ever be
   * evaluated in one call
   * \param [in] nr_rows the vertical resolution
   * \param [in] nr_cols the horizontal resolution
   */
  BufferConfiguration(boost::shared_ptr<ObjectRasterizer> rasterizer,
                      boost::shared_ptr<CudaEvaluator> evaluator,
                      int max_nr_poses,
                      int nr_rows,
                      int nr_cols);

  /**
   * \brief This function allocates GPU memory space through OpenGL and CUDA
   * for the number of poses specified. This should be the maximum number of
   * poses you ever want to evaluate in one frame, as allocation is only done once.
   * \param [in] max_nr_poses the maximum number of poses you want to evaluate
   * \param [out] new_max_nr_poses the reduced maximum number of poses that you
   * are allowed to evaluate per frame. This will only differ from max_nr_poses
   * if you enabled adaptation to constraints.
   * \return whether memory allocation was successful or not
   */
  bool allocate_memory(int max_nr_poses,
                       int& new_max_nr_poses);
  /**
   * \brief Set the resolution to a new value.
   * \param [in] nr_rows the new vertical resolution
   * \param [in] nr_cols the new horizontal resolution
   * \param new_max_nr_poses [out] the reduced maximum number of poses that you
   * are allowed to evaluate per frame. This will only differ from max_nr_poses
   * if you enabled adaptation to constraints.
   * \return whether resetting the resolution was successful or not
   */
  bool set_resolution(const int nr_rows, const int nr_cols,
                       int& new_max_nr_poses);

  /**
   * \brief Set the number of poses to be evaluated in the next frame
   * \param [in] nr_poses the number of poses that you want to evaluate in
   * the next frame. Should be less then what you specified with max_nr_poses.
   * \param [out] new_nr_poses the reduced number of poses that you are allowed
   * to evaluate in the next frame. This will only differ from nr_poses if you
   * enabled adaptation to constraints.
   * \return whether setting the number of poses succeeded or not
   */
  bool set_nr_of_poses(const int nr_poses,
                       int& new_nr_poses);

  /**
   * \brief Set the number of threads to the desired amount. Should be a multiple
   * of the warp size (usually 32).
   * \param [in] nr_threads the desired amount of threads
   * \param [out] new_nr_threads the reduced amount of threads that will be used
   * in case your request exceeds the maximum number of threads allowed. Will only
   * differ from nr_threads if you enabled adaptation to constraints.
   * \return whether setting the number of threads succeeded or not
   */
  bool set_number_of_threads(const int nr_threads,
                             int& new_nr_threads);

  /** \brief Enable automatic adaptation to constraints. This includes GPU
   *  constraints, but also self-made constraints like the maximum number of poses.
   *  \param [in] should_adapt whether or not the number of poses should
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
