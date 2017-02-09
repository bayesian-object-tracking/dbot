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
#include <dbot/pose/pose_vector.hpp>

using namespace osr;
typedef double Real;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 4, 4> HomogeneousMatrix;
typedef Eigen::Matrix<Real, 6, 1> Vector6d;
typedef Eigen::Matrix<Real, 3, 1> Vector3d;
typedef Eigen::Matrix<Real, 4, 1> Vector4d;
typedef PoseVector::Affine Affine;

Real epsilon = 0.000000001;

TEST(pose_vector, equality)
{
    Vector6d vector = Vector6d::Random();
    PoseVector pose_vector = vector;

    EXPECT_TRUE(pose_vector.isApprox(vector));
}

TEST(pose_vector, position)
{
    Vector3d vector = Vector3d::Random();
    PoseVector pose_vector;
    pose_vector.position() = vector;

    EXPECT_TRUE(pose_vector.position().isApprox(vector));
}

TEST(pose_vector, euler_vector)
{
    Vector3d vector = Vector3d::Random();
    PoseVector pose_vector;
    pose_vector.orientation() = vector;

    EXPECT_TRUE(pose_vector.orientation().isApprox(vector));
}

TEST(pose_vector, quaternion)
{
    EulerVector euler = EulerVector::Random();
    PoseVector pose_vector;

    pose_vector.orientation().quaternion(euler.quaternion());

    EXPECT_TRUE(pose_vector.orientation().isApprox(euler));
}

TEST(pose_vector, get_homogeneous)
{
    PoseVector pose_vector = PoseVector::Random();

    Vector3d va = Vector3d::Random();
    Vector4d vb;
    vb.topRows(3) = va;
    vb(3) = 1;

    va = pose_vector.orientation().rotation_matrix() * va +
         pose_vector.position();

    vb = pose_vector.homogeneous() * vb;

    EXPECT_TRUE(va.isApprox(vb.topRows(3)));
}

TEST(pose_vector, set_homogeneous)
{
    PoseVector pose_vector1 = PoseVector::Random();
    PoseVector pose_vector2;
    pose_vector2.homogeneous(pose_vector1.homogeneous());

    EXPECT_TRUE(pose_vector1.isApprox(pose_vector2));
}

TEST(pose_vector, get_affine)
{
    PoseVector pose_vector = PoseVector::Random();

    Vector3d va = Vector3d::Random();
    Vector3d vb = va;

    pose_vector.position() = Vector3d::Zero();

    va = pose_vector.orientation().rotation_matrix() * va +
         pose_vector.position();
    vb = pose_vector.affine() * vb;

    EXPECT_TRUE(va.isApprox(vb));
}

TEST(pose_vector, set_affine)
{
    PoseVector pose_vector1 = PoseVector::Random();
    PoseVector pose_vector2;
    pose_vector2.affine(pose_vector1.affine());

    EXPECT_TRUE(pose_vector1.isApprox(pose_vector2));
}

TEST(pose_vector, product)
{
    PoseVector v1 = PoseVector::Random();
    PoseVector v2 = PoseVector::Random();

    PoseVector correct_result;
    correct_result.orientation().rotation_matrix(
        v2.orientation().rotation_matrix() *
        v1.orientation().rotation_matrix());
    correct_result.position() =
        v2.orientation().rotation_matrix() * v1.position() + v2.position();

    PoseVector operator_result = v2 * v1;

    EXPECT_TRUE(correct_result.isApprox(operator_result));
}

TEST(pose_vector, inverse)
{
    PoseVector v = PoseVector::Random();

    PoseVector result = v * v.inverse();
    EXPECT_TRUE(result.norm() < 0.00001);

    result = v.inverse() * v;
    EXPECT_TRUE(result.norm() < 0.00001);
}
