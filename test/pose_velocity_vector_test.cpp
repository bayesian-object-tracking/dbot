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
 * \date 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <osr/pose_velocity_vector.hpp>

using namespace osr;
typedef double Real;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 3, 1> Vector;

Real epsilon = 0.000000001;

TEST(pose_velocity_vector, consistency)
{
    PoseVelocityVector vector1 = PoseVelocityVector::Random();
    PoseVelocityVector vector2;

    vector2.pose() = vector1.pose();
    vector2.linear_velocity() = vector1.linear_velocity();
    vector2.angular_velocity() = vector1.angular_velocity();

    EXPECT_TRUE(vector1.isApprox(vector2));
}

TEST(pose_velocity_vector, setting_pose_properties)
{
    PoseVelocityVector vector = PoseVelocityVector::Random();
    PoseVector pose = PoseVector::Random();

    vector.pose().position() = pose.position();
    vector.pose().orientation() = pose.orientation();

    EXPECT_TRUE(vector.pose().isApprox(pose));
}
