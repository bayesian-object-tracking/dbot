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

#include <dbot/util/camera_data_provider.hpp>

namespace dbot
{
class SimpleCameraDataProvider : public CameraDataProvider
{
public:
    SimpleCameraDataLoader(const std::string& camera_frame_id,
                           const Eigen::Matrix3d& camera_mat)
        : frame_id_(camera_frame_id), camera_matrix_(camera_mat)
    {
    }

public:
    Eigen::Matrix3d camera_matrix() const { return camera_matrix_; }
    std::string frame_id() const { return frame_id_; }

private:
    std::string frame_id_;
    Eigen::Matrix3d camera_matrix_;
};
}
