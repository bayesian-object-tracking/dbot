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
 * \file simple_camera_data_provider.h
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <dbot/simple_camera_data_provider.h>

namespace dbot
{
SimpleCameraDataProvider::SimpleCameraDataProvider(
    const std::string& camera_frame_id,
    const Eigen::Matrix3d& camera_mat,
    const CameraData::Resolution& native_res)
    : frame_id_(camera_frame_id),
      camera_matrix_(camera_mat),
      native_res_(native_res)
{
}

Eigen::Matrix3d SimpleCameraDataProvider::camera_matrix() const
{
    return camera_matrix_;
}

std::string SimpleCameraDataProvider::frame_id() const
{
    return frame_id_;
}

CameraData::Resolution SimpleCameraDataProvider::native_resolution() const
{
    return native_res_;
}

}
