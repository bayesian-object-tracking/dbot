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
 * \file pose_velocity_vector.h
 * \date August 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>
#include <dbot/pose/pose_vector.h>
#include <dbot/pose/euler_vector.h>
namespace dbot
{
/// basic functionality for both vectors and blocks ****************************
template <typename Base>
class PoseVelocityBase : public Base
{
public:
    enum
    {
        POSE_INDEX = 0,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9,

        POSE_SIZE = 6,
        VELOCITY_SIZE = 3,
    };

    // types *******************************************************************
    typedef Eigen::Matrix<Real, VELOCITY_SIZE, 1> VelocityVector;
    typedef Eigen::VectorBlock<Base, VELOCITY_SIZE> VelocityBlock;

    // constructor and destructor **********************************************
    PoseVelocityBase(const Base& vector) : Base(vector) {}
    virtual ~PoseVelocityBase() noexcept {}
    // operators ***************************************************************
    template <typename T>
    void operator=(const Eigen::MatrixBase<T>& vector)
    {
        *((Base*)(this)) = vector;
    }

    // accessors ***************************************************************
    virtual PoseVector pose() const
    {
        return this->template middleRows<POSE_SIZE>(POSE_INDEX);
    }
    virtual PoseVector::HomogeneousMatrix homogeneous() const
    {
        return pose().homogeneous();
    }
    virtual PoseVector::Affine affine() const { return pose().affine(); }
    virtual PoseVector::Vector position() const { return pose().position(); }
    virtual EulerVector orientation() const { return pose().orientation(); }
    virtual VelocityVector linear_velocity() const
    {
        return this->template middleRows<VELOCITY_SIZE>(LINEAR_VELOCITY_INDEX);
    }
    virtual VelocityVector angular_velocity() const
    {
        return this->template middleRows<VELOCITY_SIZE>(ANGULAR_VELOCITY_INDEX);
    }

    // mutators ****************************************************************
    PoseBlock<Base> pose() { return PoseBlock<Base>(*this, POSE_INDEX); }
    virtual void homogeneous(
        const typename PoseBlock<Base>::HomogeneousMatrix& H)
    {
        pose().homogeneous(H);
    }
    virtual void affine(const typename PoseBlock<Base>::Affine& A)
    {
        pose().affine(A);
    }
    virtual void set_zero()
    {
        pose().setZero();
        set_zero_velocity();
    }
    virtual void set_zero_pose()
    {
        pose().setZero();
    }
    virtual void set_zero_velocity()
    {
        linear_velocity() = Eigen::Vector3d::Zero();
        angular_velocity() = Eigen::Vector3d::Zero();
    }

    typename PoseBlock<Base>::PositionBlock position()
    {
        return pose().position();
    }
    typename PoseBlock<Base>::OrientationBlock orientation()
    {
        return pose().orientation();
    }
    VelocityBlock linear_velocity()
    {
        return VelocityBlock(*this, LINEAR_VELOCITY_INDEX);
    }
    VelocityBlock angular_velocity()
    {
        return VelocityBlock(*this, ANGULAR_VELOCITY_INDEX);
    }

    template <typename PoseType>
    void apply_delta(const PoseType& delta_pose)
    {
        pose().apply_delta(delta_pose);
        linear_velocity() = delta_pose.linear_velocity();
        angular_velocity() = delta_pose.angular_velocity();
    }

    template <typename PoseType>
    void subtract(const PoseType& mean)
    {
        pose().subtract(mean);
    }
};

/// implementation for vectors *************************************************
class PoseVelocityVector : public PoseVelocityBase<Eigen::Matrix<Real, 12, 1>>
{
public:
    typedef PoseVelocityBase<Eigen::Matrix<Real, 12, 1>> Base;

    // constructor and destructor **********************************************
    PoseVelocityVector() : Base(Base::Zero()) {}
    template <typename T>
    PoseVelocityVector(const Eigen::MatrixBase<T>& vector)
        : Base(vector)
    {
    }

    virtual ~PoseVelocityVector() noexcept {}
};

/// implementation for blocks **************************************************
template <typename Vector>
class PoseVelocityBlock
    : public PoseVelocityBase<Eigen::VectorBlock<Vector, 12>>
{
public:
    typedef Eigen::VectorBlock<Vector, 12> Block;
    typedef PoseVelocityBase<Block> Base;

    using Base::operator=;

    // constructor and destructor **********************************************
    PoseVelocityBlock(const Block& block) : Base(block) {}
    PoseVelocityBlock(Vector& vector, int start) : Base(Block(vector, start)) {}
    virtual ~PoseVelocityBlock() noexcept {}
};
}
