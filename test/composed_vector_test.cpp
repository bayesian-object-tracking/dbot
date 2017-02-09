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

#include <dbot/pose/composed_vector.hpp>
#include <dbot/pose/pose_vector.hpp>

using namespace osr;
typedef double Real;

TEST(composed_vector, count_dynamic_size)
{
    typedef Eigen::VectorXd Vector;
    typedef Eigen::VectorBlock<Vector, 4> Block;

    ComposedVector<Block, Vector> vector;

    vector.resize(12);
    EXPECT_TRUE(vector.count() == 3);

    vector.resize(15);
    EXPECT_TRUE(vector.count() == 3);

    vector.resize(16);
    EXPECT_TRUE(vector.count() == 4);
}

TEST(composed_vector, count_fixed_size)
{
    typedef Eigen::Matrix<Real, 16, 1> Vector;
    typedef Eigen::VectorBlock<Vector, 4> Block;

    ComposedVector<Block, Vector> vector;

    EXPECT_TRUE(vector.count() == 4);
}

TEST(composed_vector, recount_dynamic_size)
{
    typedef Eigen::VectorXd Vector;
    typedef Eigen::VectorBlock<Vector, 4> Block;

    ComposedVector<Block, Vector> vector;

    vector.recount(3);
    EXPECT_TRUE(vector.size() == 12);

    vector.recount(15);
    EXPECT_TRUE(vector.size() == 60);
}

TEST(composed_vector, mutators_dynamic_size)
{
    typedef Eigen::VectorXd Vector;
    typedef PoseBlock<Vector> Block;

    ComposedVector<Block, Vector> vector;
    vector.recount(3);

    PoseVector pose0 = PoseVector::Random();
    PoseVector pose1 = PoseVector::Random();
    PoseVector pose2 = PoseVector::Random();

    vector.component(0) = pose0;
    vector.component(1).orientation() = pose1.orientation();
    vector.component(1).position() = pose1.position();
    vector.component(2).orientation().quaternion(
        pose2.orientation().quaternion());
    vector.component(2).position() = pose2.position();

    EXPECT_TRUE(pose0.isApprox(vector.component(0)));
    EXPECT_TRUE(pose1.isApprox(vector.component(1)));
    EXPECT_TRUE(pose2.isApprox(vector.component(2)));
}

TEST(composed_vector, mutators_fixed_size)
{
    typedef Eigen::Matrix<Real, 3 * PoseVector::SizeAtCompileTime, 1> Vector;
    typedef PoseBlock<Vector> Block;

    ComposedVector<Block, Vector> vector;

    PoseVector pose0 = PoseVector::Random();
    PoseVector pose1 = PoseVector::Random();
    PoseVector pose2 = PoseVector::Random();

    vector.component(0) = pose0;
    vector.component(1).orientation() = pose1.orientation();
    vector.component(1).position() = pose1.position();
    vector.component(2).orientation().quaternion(
        pose2.orientation().quaternion());
    vector.component(2).position() = pose2.position();

    EXPECT_TRUE(pose0.isApprox(vector.component(0)));
    EXPECT_TRUE(pose1.isApprox(vector.component(1)));
    EXPECT_TRUE(pose2.isApprox(vector.component(2)));
}

TEST(composed_vector, accessors_dynamic_size)
{
    typedef Eigen::VectorXd Vector;
    typedef PoseBlock<Vector> Block;

    ComposedVector<Block, Vector> vector;
    vector.recount(3);
    vector = ComposedVector<Block, Vector>::Random(vector.size());

    PoseVector pose0, pose1, pose2;
    pose0 = vector.component(0);
    pose1.orientation() = vector.component(1).orientation();
    pose1.position() = vector.component(1).position();
    pose2.position() = vector.component(2).position();
    pose2.orientation().quaternion(
        vector.component(2).orientation().quaternion());

    EXPECT_TRUE(vector.component(0).isApprox(pose0));
    EXPECT_TRUE(vector.component(1).isApprox(pose1));
    EXPECT_TRUE(vector.component(2).isApprox(pose2));
}

TEST(composed_vector, accessors_fixed_size)
{
    typedef Eigen::Matrix<Real, 3 * PoseVector::SizeAtCompileTime, 1> Vector;
    typedef PoseBlock<Vector> Block;

    ComposedVector<Block, Vector> vector =
        ComposedVector<Block, Vector>::Random();

    PoseVector pose0, pose1, pose2;
    pose0 = vector.component(0);
    pose1.orientation() = vector.component(1).orientation();
    pose1.position() = vector.component(1).position();
    pose2.position() = vector.component(2).position();
    pose2.orientation().quaternion(
        vector.component(2).orientation().quaternion());

    EXPECT_TRUE(vector.component(0).isApprox(pose0));
    EXPECT_TRUE(vector.component(1).isApprox(pose1));
    EXPECT_TRUE(vector.component(2).isApprox(pose2));
}
