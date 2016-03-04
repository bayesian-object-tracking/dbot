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

#include <dbot/model/observation/gpu/buffer_configuration.hpp>

BufferConfiguration::BufferConfiguration(ObjectRasterizer* rasterizer,
                                         CudaEvaluator* evaluator,
                                         int nr_max_poses,
                                         int nr_cols,
                                         int nr_rows) :
    rasterizer_(rasterizer),
    evaluator_(evaluator),
    nr_max_poses_(nr_max_poses),
    nr_cols_(nr_cols),
    nr_rows_(nr_rows)
{
    max_texture_size_opengl_ = rasterizer_->get_max_texture_size();
    cuda_device_properties_ = evaluator_->getDeviceProperties();
}

bool BufferConfiguration::allocate_memory(int max_nr_poses) {

    int max_nr_poses_per_row, max_nr_poses_per_col;

    compute_grid_layout(max_nr_poses, max_nr_poses_per_row, max_nr_poses_per_col);

    int new_max_nr_poses;

    bool constrained_by_texture_size =
        check_against_texture_size_constraint(max_nr_poses, max_nr_poses_per_row,
                                              max_nr_poses_per_col, new_max_nr_poses);

    if (constrained_by_texture_size) {
        if (adapt_to_contraints) {
            max_nr_poses_ = new_max_nr_poses;
            issue_warning();
        } else {
            issue_error();
            return false;
        }
    }

    int new_max_nr_poses_per_row, new_max_nr_poses_per_col;

    bool constrained_by_global_memory =
        check_against_global_memory_constraint(max_nr_poses_, new_max_nr_poses_per_row,
                                               new_max_nr_poses_per_col, new_max_nr_poses);


    if (constrained_by_global_memory) {
        if (adapt_to_constraints) {
            max_nr_poses_ = new_max_nr_poses;
            max_nr_poses_per_row = new_max_nr_poses_per_row;
            max_nr_poses_per_col = new_max_nr_poses_per_col;
            issue_warning();
        } else {
            issue_error();
            return false;
        }
    }

    evaluator_->allocate_memory_for_max_poses(max_nr_poses_,
                                              max_nr_poses_per_row,
                                              max_nr_poses_per_col);

    rasterizer_->allocate_textures_for_max_poses(max_nr_poses_,
                                                 max_nr_poses_per_row,
                                                 max_nr_poses_per_col);

    // TODO: When to pass nr_cols, nr_rows? Pass inside this allocation call?

    return true;
}

bool BufferConfiguration::set_resolution(int nr_rows, int nr_cols,
                                         int& new_nr_rows, int& new_nr_cols) {

    // limit in height
    int max_nr_rows = min(nr_rows, max_texture_size_opengl_,
                          cuda_device_properties_.maxTexture2D[1]);

    // resulting width
    float ratio = nr_cols / (float) nr_rows;
    int tmp_nr_cols = ratio * max_nr_rows;

    // limit in width
    int max_nr_cols = min(tmp_nr_cols, max_texture_size_opengl_,
                          cuda_device_properties_.maxTexture2D[0]);

    // update resulting height
    max_nr_rows = max_nr_cols / ratio;

    // scale down such that resolution is a power of two
    new_nr_cols = pow(2, floor(log(max_nr_cols)));
    new_nr_rows = pow(2, floor(log(max_nr_rows)));


    bool constrained_by_texture_size = (nr_rows > new_nr_rows) ||
                                       (nr_cols > new_nr_cols);

    if (constrained_by_texture_size) {
        if (adapt_to_constraints) {
            nr_rows_ = new_nr_rows;
            nr_cols_ = new_nr_cols;
            issue_warning();
        } else {
            issue_error();
            return false;
        }
    } else {
        nr_rows_ = nr_rows;
        nr_cols_ = nr_cols;
    }

    bool successfully_allocated_memory = allocate_memory(max_nr_poses_);

    return successfully_allocated_memory;
}


bool BufferConfiguration::check_against_texture_size_constraint(int nr_poses,
                                                                int nr_poses_per_row,
                                                                int nr_poses_per_col,
                                                                int& new_nr_poses) {

    new_nr_poses = nr_poses_per_row * nr_poses_per_col;

    return new_nr_poses < nr_poses;
}


void BufferConfiguration::compute_grid_layout(int nr_poses,
                                              int& nr_poses_per_row,
                                              int& nr_poses_per_col) {

    int max_texture_size_x = min(max_texture_size_opengl_,
                                 cuda_device_properties_.maxTexture2D[0],
                                 cuda_device_properties_.maxGridSize[0]);
    int max_texture_size_y = min(max_texture_size_opengl_,
                                 cuda_device_properties_.maxTexture2D[1],
                                 cuda_device_properties_.maxGridSize[1]);

    nr_poses_per_row = floor(max_texture_size_x / nr_cols_);
    nr_poses_per_col = min(floor(max_texture_size_y / nr_rows_),
                           ceil(nr_poses / (float) poses_per_row));
}


bool BufferConfiguration::check_against_global_memory_constraint(int nr_poses,
                                                                 int& new_nr_poses,
                                                                 int& new_nr_poses_per_row,
                                                                 int& new_nr_poses_per_col) {

    int constant_need_rasterizer, per_pose_need_rasterizer;
    rasterizer_->get_memory_need_parameters(nr_rows_, nr_cols_,
                                            constant_need_rasterizer,
                                            per_pose_need_rasterizer);

    int constant_need_evaluator, per_pose_need_evaluator;
    evaluator_->get_memory_need_parameters(nr_rows_, nr_cols_,
                                           constant_need_evaluator,
                                           per_pose_need_evaluator);

    int constant_needs = constant_need_rasterizer + constant_need_evaluator;
    int per_pose_needs = per_pose_need_rasterizer + per_pose_need_evaluator;

    int memory_needs = constant_needs + per_pose_needs * nr_poses;

    new_nr_poses = min(nr_poses,
                       (cuda_device_properties_.totalGlobalMem - constant_needs)
                        / per_pose_needs);

    compute_grid_layout(new_nr_poses, new_nr_poses_per_row, new_nr_poses_per_col);


    return memory_needs > cuda_device_properties_.totalGlobalMem;
}


bool BufferConfiguration::set_number_of_threads(int nr_threads) {
    int new_nr_threads;
    bool constrained_by_thread_limit
        = check_against_thread_constraints(nr_threads, new_nr_threads);

    if (constrained_by_thread_limit) {
        if (adapt_to_constraints) {
            nr_threads_ = new_nr_threads;
            issue_warning();
        } else {
            issue_error();
            return false;
        }
    } else {
        nr_threads_ = nr_threads;
    }

    evaluator_->set_nr_threads(nr_threads_);
    return true;
}


bool BufferConfiguration::check_against_thread_constraints(int nr_threads,
                                                           int& new_nr_threads) {

    new_nr_threads = min(nr_threads, cuda_device_properties_.maxThreadsDim[0]);
    return new_nr_threads < nr_threads;
}


void BufferConfiguration::issue_warning() {

}

    if (allocated_poses > allocated_poses_per_row * allocated_poses_per_column) {
        if (adapt_to_constraints) {
            std::cout << "WARNING (OPENGL): The space for the number of maximum poses you requested (" << allocated_poses << ") cannot be allocated. "
                      << "The limit is OpenGL texture size: " << max_texture_size_opengl_ << ". Current resolution is (" << nr_cols_ << ", "
                      << nr_rows_ << ") , which means a maximum of (" << max_poses_per_row << ", " << max_poses_per_column << ") poses. "
                      << "As a result, space for " << allocated_poses_per_row * allocated_poses_per_column << " poses will be allocated "
                      << "in the form of (" << allocated_poses_per_row << ", " << allocated_poses_per_column << ")." << std::endl;
        } else {
            std::cout << "ERROR (OPENGL): The number of poses you requested cannot be rendered. The limit is the maximum OpenGL texture size: "
                      << max_texture_size_opengl_ << " x " << max_texture_size_opengl_ << ". You requested a resolution of " << nr_cols_  << " x " << nr_rows_
                      << " and " << allocated_poses << " poses." << std::endl;
            exit(-1);
        }
    }

    allocated_poses = allocated_poses_per_row * allocated_poses_per_column;

    max_nr_poses_ = allocated_poses;
    nr_poses_ = allocated_poses;
    max_nr_poses_per_row_ = allocated_poses_per_row;
    nr_poses_per_row_ = allocated_poses_per_row;
    max_nr_poses_per_col_ = allocated_poses_per_column;
    nr_poses_per_column_ = allocated_poses_per_column;



    // check limitation by texture size
    if (cuda_device_properties_.maxTexture2D[0] <= allocated_poses_per_row * nr_cols_) {
        if (adapt_to_constraints) {

            std::cout << "WARNING (CUDA): The max poses you requested (" << allocated_poses << ") could not be allocated." << std::endl;

            allocated_poses_per_row = cuda_device_properties_.maxTexture2D[0] / nr_cols_;
            allocated_poses_per_column = ceil(allocated_poses / allocated_poses_per_row);

            if (cuda_device_properties_.maxTexture2D[1] <= allocated_poses_per_column * nr_rows_) {
                allocated_poses_per_column = cuda_device_properties_.maxTexture2D[1] / nr_rows_;
            }

            allocated_poses = min(allocated_poses, allocated_poses_per_row * allocated_poses_per_column);

            std::cout << "The limit is max texture size (" << cuda_device_properties_.maxTexture2D[0]
                      << ", " << cuda_device_properties_.maxTexture2D[1] << ") retrieved from CUDA properties. "
                      << "Number of poses was reduced to (" << allocated_poses_per_row << ", "
                      << allocated_poses_per_column << "), a total of " << allocated_poses << std::endl;


        } else {
            std::cout << "ERROR (CUDA): The max poses you requested (" << allocated_poses << ") could not be allocated."
                      << "The limit is max texture size (" << cuda_device_properties_.maxTexture2D[0]
                      << ", " << cuda_device_properties_.maxTexture2D[1] << ") retrieved from CUDA properties. " << std::endl;
            exit(-1);
        }
    }



    // check limitation by global memory
    size_t size_of_log_likelihoods = sizeof(float) * allocated_poses;
    size_t size_of_resampling_indices = sizeof(int) * allocated_poses;
    size_t size_of_occlusion_indices = sizeof(int) * allocated_poses;
    size_t size_of_occlusion_probs = nr_rows_ * nr_cols_ * allocated_poses * sizeof(float);
    size_t size_of_opengl_textures = size_of_occlusion_probs * 2;
    size_t size_of_observations = nr_cols_ * nr_rows_ * sizeof(float);

    size_t total_size = size_of_log_likelihoods + size_of_resampling_indices + size_of_occlusion_indices
                      + size_of_occlusion_probs * 2 + size_of_opengl_textures + size_of_observations;

    if (total_size > cuda_device_properties_.totalGlobalMem) {
        if (adapt_to_constraints) {
            std::cout << "WARNING (CUDA): The space (" << total_size << " B) for the number of maximum poses you requested (" << allocated_poses << ") cannot be allocated. "
                      << "The limit is global memory size (" << cuda_device_properties_.totalGlobalMem
                      << " B) retrieved from CUDA properties." << std::endl;

            size_t size_depending_on_nr_poses = (sizeof(float) + sizeof(int) * 2 + nr_rows_ * nr_cols_ * sizeof(float) * 4);
            allocated_poses = min(allocated_poses, (int) floor((cuda_device_properties_.totalGlobalMem - size_of_observations) / size_depending_on_nr_poses));
            allocated_poses_per_column = ceil(allocated_poses / allocated_poses_per_row);

            std::cout << "Instead, space (" << allocated_poses * size_depending_on_nr_poses + size_of_observations << " B) for " << allocated_poses << " poses was allocated. " << std::endl;
        } else {
            std::cout << "ERROR (CUDA): The space (" << total_size << " B) for the number of maximum poses you requested (" << allocated_poses << ") cannot be allocated. "
                      << "The limit is global memory size (" << cuda_device_properties_.totalGlobalMem
                      << " B) retrieved from CUDA properties." << std::endl;
            exit(-1);
        }
    }

}


