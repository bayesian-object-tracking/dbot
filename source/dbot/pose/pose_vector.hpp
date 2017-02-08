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
 * \file pose_vector.hpp
 * \date August 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include "euler_vector.hpp"

namespace osr
{
/// basic functionality for both vectors and blocks ****************************
template <typename Base>
class PoseBase : public Base
{
public:
    enum
    {
        BLOCK_SIZE = 3,
        POSITION_INDEX = 0,
        EULER_VECTOR_INDEX = 3
    };

    // types *******************************************************************
    typedef Eigen::Matrix<Real, 3, 1> Vector;
    typedef Eigen::Matrix<Real, 4, 4> HomogeneousMatrix;
    typedef typename Eigen::Transform<Real, 3, Eigen::Affine> Affine;

    typedef Eigen::VectorBlock<Base, BLOCK_SIZE> PositionBlock;
    typedef EulerBlock<Base> OrientationBlock;

    typedef PoseBase<Eigen::Matrix<Real, 6, 1>> PoseVector;

    // constructor and destructor **********************************************
    PoseBase(const Base& vector) : Base(vector) {}
    virtual ~PoseBase() noexcept {}
    // operators ***************************************************************
    template <typename T>
    void operator=(const Eigen::MatrixBase<T>& vector)
    {
        *((Base*)(this)) = vector;
    }

    // accessors ***************************************************************
    virtual Vector position() const
    {
        return this->template middleRows<BLOCK_SIZE>(POSITION_INDEX);
    }
    virtual EulerVector orientation() const
    {
        return this->template middleRows<BLOCK_SIZE>(EULER_VECTOR_INDEX);
    }
    virtual HomogeneousMatrix homogeneous() const
    {
        HomogeneousMatrix H(HomogeneousMatrix::Identity());
        H.topLeftCorner(3, 3) = orientation().rotation_matrix();
        H.topRightCorner(3, 1) = position();

        return H;
    }
    virtual Affine affine() const
    {
        Affine A;
        A.linear() = orientation().rotation_matrix();
        A.translation() = position();

        return A;
    }
    virtual PoseVector inverse() const
    {
        PoseVector inv(PoseVector::Zero());
        inv.homogeneous(this->homogeneous().inverse());
        return inv;
    }

    // mutators ****************************************************************
    PositionBlock position() { return PositionBlock(*this, POSITION_INDEX); }
    OrientationBlock orientation()
    {
        return OrientationBlock(*this, EULER_VECTOR_INDEX);
    }
    virtual void homogeneous(const HomogeneousMatrix& H)
    {
        orientation().rotation_matrix(H.topLeftCorner(3, 3));
        position() = H.topRightCorner(3, 1);
    }
    virtual void affine(const Affine& A)
    {
        orientation().rotation_matrix(A.rotation());
        position() = A.translation();
    }
    virtual void set_zero() { this->setZero(); }
    template <typename PoseType>
    void apply_delta(const PoseType& delta_pose)
    {
        position() = orientation().rotation_matrix() * delta_pose.position()
                + position();
        orientation() = orientation() * delta_pose.orientation();
    }

    /// \todo: these subtract and apply_delta functions are a bit confusing,
    /// they should be made clearer

    // note: we do not apply the inverse pose, but we treat position and
    // orientation separately
    template <typename PoseType>
    void subtract(const PoseType& mean)
    {
        position() = mean.orientation().inverse().rotation_matrix()
                * (position() - mean.position());
        orientation() = mean.orientation().inverse() * orientation();
    }

    // operators ***************************************************************
    template <typename T>
    PoseVector operator*(const PoseBase<T>& factor)
    {
        PoseVector product(PoseVector::Zero());
        product.homogeneous(this->homogeneous() * factor.homogeneous());
        return product;
    }
};

/// implementation for vectors *************************************************
class PoseVector : public PoseBase<Eigen::Matrix<Real, 6, 1>>
{
public:
    typedef PoseBase<Eigen::Matrix<Real, 6, 1>> Base;

    // constructor and destructor **********************************************
    PoseVector() : Base(Base::Zero()) {}
    template <typename T>
    PoseVector(const Eigen::MatrixBase<T>& vector)
        : Base(vector)
    {
    }

    virtual ~PoseVector() noexcept {}
};

/// implementation for blocks **************************************************
template <typename Vector>
class PoseBlock : public PoseBase<Eigen::VectorBlock<Vector, 6>>
{
public:
    typedef Eigen::VectorBlock<Vector, 6> Block;
    typedef PoseBase<Block> Base;

    using Base::operator=;

    // constructor and destructor **********************************************
    PoseBlock(const Block& block) : Base(block) {}
    PoseBlock(Vector& vector, int start) : Base(Block(vector, start)) {}
    virtual ~PoseBlock() noexcept {}
};
}
