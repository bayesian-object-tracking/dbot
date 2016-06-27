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
 * \file camera_data.cpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date December 2015
 */

#include <string>

#include <Eigen/Dense>

#include <dbot/camera_data.hpp>
#include <dbot/camera_data_provider.hpp>

namespace dbot
{
CameraData::CameraData(const std::shared_ptr<CameraDataProvider> &data_provider)
    : data_provider_(std::move(data_provider))
{
}

Eigen::MatrixXd CameraData::depth_image() const
{
    return data_provider_->depth_image();
}

Eigen::VectorXd CameraData::depth_image_vector() const
{
   return data_provider_->depth_image_vector();
}

std::string CameraData::frame_id() const
{
    return data_provider_->frame_id();
}

Eigen::Matrix3d CameraData::camera_matrix() const
{
    return data_provider_->camera_matrix();
}

int CameraData::downsampling_factor() const
{
    return data_provider_->downsampling_factor();
}

CameraData::Resolution CameraData::resolution() const
{
    Resolution res = native_resolution();

    res.width = res.width / downsampling_factor();
    res.height = res.height / downsampling_factor();

    return res;
}

CameraData::Resolution CameraData::native_resolution() const
{
    return data_provider_->native_resolution();
}

int CameraData::pixels() const
{
    auto res = resolution();
    return res.height * res.width;
}
}
