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
 * \file camera_data_provider.hpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date December 2015
 */

#pragma once

#include <string>
#include <Eigen/Dense>

#include <dbot/common/camera_data.hpp>

namespace dbot
{

/**
 * \brief Represents the interface of a CameraData implementation.
 */
class CameraDataProvider
{
public:
    /**
     * \brief Default virtual destructor
     */
    virtual ~CameraDataProvider() {  }

    /**
     * \brief returns an obtained depth image as an Eigen matrix
     */
    virtual Eigen::MatrixXd depth_image() const = 0;

    /**
     * \brief returns an obtained depth image as an Eigen vector
     */
    virtual Eigen::VectorXd depth_image_vector() const = 0;

    /**
     * \brief Obtains the camera matrix
     */
    virtual Eigen::Matrix3d camera_matrix() const = 0;

    /**
     * \brief Returns the frame_id name of the camera
     */
    virtual std::string frame_id() const = 0;

    /**
     * \brief Returns the resolution integer downsampling factor
     */
    virtual int downsampling_factor() const = 0;

    /**
     * \brief Returns the camera native resolution. That is, width and height in
     *        pixels
     */
    virtual CameraData::Resolution native_resolution() const = 0;
};

}
