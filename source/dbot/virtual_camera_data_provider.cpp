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
 * \file ros_camera_data_provider.h
 * \author Jan Issc (jan.issac@gmail.com)
 * \date December 2015
 */

#include <fl/util/profiling.hpp>
#include <dbot/virtual_camera_data_provider.h>

namespace dbot
{
VirtualCameraDataProvider::VirtualCameraDataProvider(
    int downsampling_factor,
    const std::string& frame_id)
    : downsampling_factor_(downsampling_factor),
      frame_id_(frame_id)
{
    native_resolution_.width = 640;
    native_resolution_.height = 480;

    camera_matrix_.setZero(3, 3);
    camera_matrix_(0, 0) = 580.0 / downsampling_factor_; // fx
    camera_matrix_(1, 1) = 580.0 / downsampling_factor_; // fy
    camera_matrix_(2, 2) = 1.0;
    camera_matrix_(0, 2) = 320 / downsampling_factor_;   // cx
    camera_matrix_(1, 2) = 240 / downsampling_factor_;   // cy
}

Eigen::MatrixXd VirtualCameraDataProvider::depth_image() const
{
    return depth_image_;
}

Eigen::VectorXd VirtualCameraDataProvider::depth_image_vector() const
{
    Eigen::VectorXd image(depth_image_.size());

    for (int i = 0, k = 0; i < depth_image_.rows(); ++i)
    {
        for (int j = 0; j < depth_image_.cols(); ++j)
        {
            image[k++] = depth_image_(i, j);
        }
    }

    return image;
}

Eigen::Matrix3d VirtualCameraDataProvider::camera_matrix() const
{
    return camera_matrix_;
}

std::string VirtualCameraDataProvider::frame_id() const
{
    return frame_id_;
}

int VirtualCameraDataProvider::downsampling_factor() const
{
    return downsampling_factor_;
}

CameraData::Resolution VirtualCameraDataProvider::native_resolution() const
{
    return native_resolution_;
}
}
