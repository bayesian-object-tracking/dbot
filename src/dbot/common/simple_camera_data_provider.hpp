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

#pragma once

#include <string>

#include <Eigen/Dense>

#include <dbot/common/camera_data_provider.hpp>

namespace dbot
{
class SimpleCameraDataProvider : public CameraDataProvider
{
public:
    SimpleCameraDataProvider(const std::string& camera_frame_id,
                           const Eigen::Matrix3d& camera_mat,
                           const CameraData::Resolution& native_res);

    Eigen::Matrix3d camera_matrix() const;
    std::string frame_id() const;
    CameraData::Resolution native_resolution() const;

protected:
    std::string frame_id_;
    Eigen::Matrix3d camera_matrix_;
    CameraData::Resolution native_res_;
};
}
