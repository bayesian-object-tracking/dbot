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
#include <dbot/pose/euler_vector.hpp>

using namespace osr;
typedef double Real;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 3, 1> Vector;

Real epsilon = 0.000000001;

TEST(euler_vector, zero_angle)
{
    EulerVector euler_vector = EulerVector::Zero();

    EXPECT_TRUE(euler_vector.rotation_matrix().isIdentity());
}

TEST(euler_vector, constructor)
{
    Vector vector = Vector::Random();
    EulerVector euler_vector(vector);

    EXPECT_TRUE((vector - euler_vector).norm() < epsilon);
}

TEST(euler_vector, equality_operator)
{
    Vector vector = Vector::Random();
    EulerVector euler_vector;

    euler_vector = vector;

    EXPECT_TRUE((vector - euler_vector).norm() < epsilon);
}

TEST(euler_vector, multiplication)
{
    EulerVector euler_vector_1 = EulerVector::Random();
    EulerVector euler_vector_2 = EulerVector::Random();

    EulerVector euler_vector = euler_vector_1 * euler_vector_2;
    RotationMatrix rotation_matrix =
        euler_vector_1.rotation_matrix() * euler_vector_2.rotation_matrix();

    RotationMatrix delta =
        rotation_matrix.inverse() * euler_vector.rotation_matrix();

    EXPECT_TRUE(delta.isIdentity());
}

TEST(euler_vector, inverse)
{
    EulerVector euler_vector1 = EulerVector::Random();
    EulerVector euler_vector2;

    euler_vector2 = euler_vector1 * euler_vector1.inverse();
    EXPECT_TRUE(euler_vector2.rotation_matrix().isIdentity());

    euler_vector2 = euler_vector1.inverse() * euler_vector1;
    EXPECT_TRUE(euler_vector2.rotation_matrix().isIdentity());
}

TEST(euler_vector, angle_axis_basic)
{
    Vector axis = Vector::Random();
    axis.normalize();
    Real angle = 1.155435;

    EulerVector euler_vector;
    euler_vector.angle_axis(angle, axis);

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle) < epsilon);
    EXPECT_TRUE(axis.isApprox(euler_vector.axis()));
}

TEST(euler_vector, angle_axis_set)
{
    Vector axis = Vector::Random();
    axis.normalize();
    Real angle = 1.155435;

    EulerVector euler_vector;
    euler_vector.angle_axis(AngleAxis(angle, axis));

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle) < epsilon);
    EXPECT_TRUE(axis.isApprox(euler_vector.axis()));
}

TEST(euler_vector, angle_axis_get)
{
    EulerVector euler_vector = EulerVector::Random();

    EXPECT_TRUE(std::fabs(euler_vector.angle() -
                          euler_vector.angle_axis().angle()) < epsilon);
    EXPECT_TRUE(euler_vector.axis().isApprox(euler_vector.angle_axis().axis()));
}

TEST(euler_vector, quaternion_get)
{
    EulerVector euler_vector = EulerVector::Random();
    AngleAxis angle_axis = AngleAxis(euler_vector.quaternion());

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle_axis.angle()) < epsilon);
    EXPECT_TRUE(angle_axis.axis().isApprox(euler_vector.axis()));
}

TEST(euler_vector, quaternion_set)
{
    EulerVector euler_vector = EulerVector::Random();
    AngleAxis angle_axis = euler_vector.angle_axis();
    euler_vector.quaternion(Quaternion(angle_axis));

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle_axis.angle()) < epsilon);
    EXPECT_TRUE(angle_axis.axis().isApprox(euler_vector.axis()));
}

TEST(euler_vector, rotation_matrix_get)
{
    EulerVector euler_vector = EulerVector::Random();
    AngleAxis angle_axis = AngleAxis(euler_vector.rotation_matrix());

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle_axis.angle()) < epsilon);
    EXPECT_TRUE(angle_axis.axis().isApprox(euler_vector.axis()));
}

TEST(euler_vector, rotation_matrix_set)
{
    EulerVector euler_vector = EulerVector::Random();
    AngleAxis angle_axis = euler_vector.angle_axis();
    euler_vector.rotation_matrix(RotationMatrix(angle_axis));

    EXPECT_TRUE(std::fabs(euler_vector.angle() - angle_axis.angle()) < epsilon);
    EXPECT_TRUE(angle_axis.axis().isApprox(euler_vector.axis()));
}

TEST(euler_vector, rescale)
{
    for (int i = 0; i < 1000; i++)
    {
        EulerVector vector = Eigen::Vector3i::Random().cast<Real>();

        EulerVector rescaled_vector = vector;
        rescaled_vector.rescale();

        EXPECT_TRUE(rescaled_vector.angle() <= M_PI);

        Eigen::Vector4d rescaled_coeffs = rescaled_vector.quaternion().coeffs();
        Eigen::Vector4d coeffs = vector.quaternion().coeffs();

        EXPECT_TRUE(rescaled_coeffs.isApprox(coeffs, 0.0001) ||
                    rescaled_coeffs.isApprox(-coeffs, 0.0001));
    }

    for (int i = 0; i < 1000; i++)
    {
        EulerVector vector = EulerVector::Random();

        EulerVector rescaled_vector = vector;
        rescaled_vector.rescale();

        EXPECT_TRUE(rescaled_vector.angle() <= M_PI);

        Eigen::Vector4d rescaled_coeffs = rescaled_vector.quaternion().coeffs();
        Eigen::Vector4d coeffs = vector.quaternion().coeffs();

        EXPECT_TRUE(rescaled_coeffs.isApprox(coeffs, 0.0001) ||
                    rescaled_coeffs.isApprox(-coeffs, 0.0001));
    }
}
