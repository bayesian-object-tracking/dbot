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
 * \file buffer_configuration.cpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date March 2016
 */

#include <boost/lexical_cast.hpp>
#include <dbot/gpu/buffer_configuration.h>
#include <iostream>
#include <sstream>

#define STR(x) boost::lexical_cast<std::string>(x)

BufferConfiguration::BufferConfiguration(
    boost::shared_ptr<ObjectRasterizer> rasterizer,
    boost::shared_ptr<CudaEvaluator> evaluator,
    const int max_nr_poses,
    const int nr_rows,
    const int nr_cols)
    : rasterizer_(rasterizer),
      evaluator_(evaluator),
      max_nr_poses_(max_nr_poses),
      nr_cols_(nr_cols),
      nr_rows_(nr_rows)
{
    max_texture_size_opengl_ = rasterizer_->get_max_texture_size();
    cuda_device_properties_ = evaluator_->get_device_properties();
    adapt_to_constraints_ = false;
}

bool BufferConfiguration::allocate_memory(const int max_nr_poses,
                                          int& new_max_nr_poses)
{
    max_nr_poses_ = max_nr_poses;

    int max_nr_poses_per_row, max_nr_poses_per_col;

    compute_grid_layout(
        max_nr_poses_, max_nr_poses_per_row, max_nr_poses_per_col);

    int tmp_max_nr_poses_1;
    bool constrained_by_texture_size =
        check_against_texture_size_constraint(max_nr_poses_,
                                              max_nr_poses_per_row,
                                              max_nr_poses_per_col,
                                              tmp_max_nr_poses_1);

    if (constrained_by_texture_size)
    {
        if (adapt_to_constraints_)
        {
            issue_message(WARNING,
                          "maximum number of poses (at resolution " +
                              STR(nr_rows_) + " x " + STR(nr_cols_) + ")",
                          "maximum texture size",
                          STR(max_nr_poses_),
                          STR(tmp_max_nr_poses_1));
            max_nr_poses_ = tmp_max_nr_poses_1;
        }
        else
        {
            issue_message(ERROR,
                          "maximum number of poses (at resolution " +
                              STR(nr_rows_) + " x " + STR(nr_cols_) + ")",
                          "maximum texture size",
                          STR(max_nr_poses));
            return false;
        }
    }

    int new_max_nr_poses_per_row, new_max_nr_poses_per_col;

    int tmp_max_nr_poses_2;
    bool constrained_by_global_memory =
        check_against_global_memory_constraint(max_nr_poses_,
                                               tmp_max_nr_poses_2,
                                               new_max_nr_poses_per_row,
                                               new_max_nr_poses_per_col);

    if (constrained_by_global_memory)
    {
        if (adapt_to_constraints_)
        {
            issue_message(WARNING,
                          "maximum number of poses",
                          "global memory size",
                          STR(max_nr_poses_),
                          STR(tmp_max_nr_poses_2));
            max_nr_poses_ = tmp_max_nr_poses_2;
            max_nr_poses_per_row = new_max_nr_poses_per_row;
            max_nr_poses_per_col = new_max_nr_poses_per_col;
        }
        else
        {
            issue_message(ERROR,
                          "maximum number of poses",
                          "global memory size",
                          STR(max_nr_poses));
            return false;
        }
    }

    evaluator_->allocate_memory_for_max_poses(
        max_nr_poses_, max_nr_poses_per_row, max_nr_poses_per_col);

    rasterizer_->allocate_textures_for_max_poses(
        max_nr_poses_, max_nr_poses_per_row, max_nr_poses_per_col);

    new_max_nr_poses = max_nr_poses_;
    return true;
}

bool BufferConfiguration::set_resolution(const int nr_rows,
                                         const int nr_cols,
                                         int& new_max_nr_poses)
{
    nr_rows_ = nr_rows;
    nr_cols_ = nr_cols;

    evaluator_->set_resolution(nr_rows, nr_cols);
    rasterizer_->set_resolution(nr_rows, nr_cols);

    bool successfully_allocated_memory =
        allocate_memory(max_nr_poses_, new_max_nr_poses);

    return successfully_allocated_memory;
}

bool BufferConfiguration::set_nr_of_poses(const int nr_poses, int& new_nr_poses)
{
    new_nr_poses = nr_poses;

    if (nr_poses > max_nr_poses_)
    {
        if (adapt_to_constraints_)
        {
            new_nr_poses = max_nr_poses_;
            issue_message(WARNING,
                          "number of poses",
                          "maximum number of poses",
                          STR(nr_poses),
                          STR(new_nr_poses));
        }
        else
        {
            issue_message(ERROR,
                          "number of poses",
                          "maximum number of poses",
                          STR(nr_poses));
            return false;
        }
    }

    evaluator_->set_number_of_poses(new_nr_poses);

    return true;
}

bool BufferConfiguration::set_number_of_threads(const int nr_threads,
                                                int& new_nr_threads)
{
    bool constrained_by_thread_limit =
        check_against_thread_constraint(nr_threads, new_nr_threads);

    if (constrained_by_thread_limit)
    {
        if (adapt_to_constraints_)
        {
            nr_threads_ = new_nr_threads;
            issue_message(WARNING,
                          "number of threads",
                          "maximum number of threads",
                          STR(nr_threads),
                          STR(nr_threads_));
        }
        else
        {
            issue_message(ERROR,
                          "number of threads",
                          "maximum number of threads",
                          STR(nr_threads));
            return false;
        }
    }
    else
    {
        nr_threads_ = nr_threads;
    }

    evaluator_->set_nr_threads(nr_threads_);
    return true;
}

void BufferConfiguration::set_adapt_to_constraints(bool should_adapt)
{
    adapt_to_constraints_ = should_adapt;
}

// ========================================================== //
// ========= Functions for checking GPU constraints ========= //
// ========================================================== //

bool BufferConfiguration::check_against_texture_size_constraint(
    const int nr_poses,
    const int nr_poses_per_row,
    const int nr_poses_per_col,
    int& new_nr_poses)
{
    new_nr_poses = std::min(nr_poses_per_row * nr_poses_per_col, nr_poses);

    return new_nr_poses < nr_poses;
}

void BufferConfiguration::compute_grid_layout(const int nr_poses,
                                              int& nr_poses_per_row,
                                              int& nr_poses_per_col)
{
    int max_texture_size_x =
        std::min(std::min(max_texture_size_opengl_,
                          cuda_device_properties_.maxTexture2D[0]),
                 cuda_device_properties_.maxGridSize[0]);
    int max_texture_size_y =
        std::min(std::min(max_texture_size_opengl_,
                          cuda_device_properties_.maxTexture2D[1]),
                 cuda_device_properties_.maxGridSize[1]);

    nr_poses_per_row = floor(max_texture_size_x / nr_cols_);
    nr_poses_per_col = std::min<int>(floor(max_texture_size_y / nr_rows_),
                                ceil(nr_poses / (float)nr_poses_per_row));
}

bool BufferConfiguration::check_against_global_memory_constraint(
    const int nr_poses,
    int& new_nr_poses,
    int& new_nr_poses_per_row,
    int& new_nr_poses_per_col)
{
    int constant_need_rasterizer, per_pose_need_rasterizer;
    rasterizer_->get_memory_need_parameters(
        nr_rows_, nr_cols_, constant_need_rasterizer, per_pose_need_rasterizer);

    int constant_need_evaluator, per_pose_need_evaluator;
    evaluator_->get_memory_need_parameters(
        nr_rows_, nr_cols_, constant_need_evaluator, per_pose_need_evaluator);

    int constant_needs = constant_need_rasterizer + constant_need_evaluator;
    int per_pose_needs = per_pose_need_rasterizer + per_pose_need_evaluator;

    int memory_needs = constant_needs + per_pose_needs * nr_poses;

    new_nr_poses = std::min(
        nr_poses,
        (int)(cuda_device_properties_.totalGlobalMem - constant_needs) /
            per_pose_needs);

    compute_grid_layout(
        new_nr_poses, new_nr_poses_per_row, new_nr_poses_per_col);

    return memory_needs > cuda_device_properties_.totalGlobalMem;
}

bool BufferConfiguration::check_against_thread_constraint(const int nr_threads,
                                                          int& new_nr_threads)
{
    new_nr_threads =
        std::min(nr_threads, cuda_device_properties_.maxThreadsDim[0]);
    return new_nr_threads < nr_threads;
}

void BufferConfiguration::issue_message(
    const BufferConfiguration::message_type foo,
    const std::string problem_quantity,
    const std::string constraint,
    const std::string original_value,
    const std::string new_value)
{
    std::string type;
    switch (foo)
    {
        case WARNING:
            type = "WARNING";
            break;
        case ERROR:
            type = "ERROR";
            break;
    }

    std::cout << std::endl
              << type << ": "
              << "The " << problem_quantity << " you requested ("
              << original_value << ") could not be set. It is constrained by "
              << constraint << ".";

    if (foo == WARNING)
    {
        std::cout << ". The " << problem_quantity << " was reduced to "
                  << new_value << ".";
    }

    std::cout << std::endl << std::endl;
}
