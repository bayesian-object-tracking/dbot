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
 * \file virtual_camera_data_provider.hpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date January 2016
 */

#pragma once

#include <string>
#include <Eigen/Dense>

#include <dbot/util/camera_data_provider.hpp>

namespace dbot
{
/**
 * \brief Represents the interface of a CameraData implementation.
 */
class VirtualCameraDataProvider
        : public CameraDataProvider
{
public:
    VirtualCameraDataProvider(int downsampling_factor,
                              const std::string& frame_id);

    /**
     * \brief Default virtual destructor
     */
    virtual ~VirtualCameraDataProvider() {}
    /**
     * \brief returns an obtained depth image as an Eigen matrix
     */
    virtual Eigen::MatrixXd depth_image() const;

    /**
     * \brief returns an obtained depth image as an Eigen vector
     */
    virtual Eigen::VectorXd depth_image_vector() const;

    /**
     * \brief Obtains the camera matrix
     */
    virtual Eigen::Matrix3d camera_matrix() const;

    /**
     * \brief Returns the frame_id name of the camera
     */
    virtual std::string frame_id() const;

    /**
     * \brief Returns the resolution integer downsampling factor
     */
    virtual int downsampling_factor() const;

    /**
     * \brief Returns the camera native resolution. That is, width and height in
     *        pixels
     */
    virtual CameraData::Resolution native_resolution() const;

protected:
    int downsampling_factor_;
    std::string frame_id_;
    Eigen::Matrix3d camera_matrix_;
    CameraData::Resolution native_resolution_;
    Eigen::MatrixXd depth_image_;
};
}
