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
 * \file free_floating_rigid_bodies_state.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <dbot/pose/rigid_bodies_state.hpp>

namespace osr
{
template <int BodyCount>
struct FreeFloatingRigidBodiesStateTypes
{
    enum
    {
        BODY_SIZE = 12,
        POSE_SIZE = 6,
    };

    typedef RigidBodiesState<BodyCount == -1 ? -1 : BodyCount * BODY_SIZE> Base;
};

template <int BodyCount = -1>
class FreeFloatingRigidBodiesState
    : public FreeFloatingRigidBodiesStateTypes<BodyCount>::Base
{
public:
    typedef FreeFloatingRigidBodiesStateTypes<BodyCount> Types;
    typedef typename Types::Base Base;
    enum
    {
        BODY_SIZE = Types::BODY_SIZE,
        POSE_SIZE = Types::POSE_SIZE,
    };

    typedef typename Base::State State;
    typedef Eigen::Matrix<Real, BodyCount == -1 ? -1 : BodyCount * POSE_SIZE, 1>
        Poses;
    typedef typename Base::PoseVelocityBlock PoseVelocityBlock;

    // constructor and destructor **********************************************
    FreeFloatingRigidBodiesState() {}
    FreeFloatingRigidBodiesState(unsigned count_bodies)
        : Base(State::Zero(count_bodies * BODY_SIZE))
    {
    }

    template <typename T>
    FreeFloatingRigidBodiesState(const Eigen::MatrixBase<T>& state_vector)
        : Base(state_vector)
    {
    }
    virtual ~FreeFloatingRigidBodiesState() noexcept {}
    // accessors ***************************************************************
    virtual osr::PoseVelocityVector component(int index) const
    {
        return PoseVelocityBlock(*((State*)(this)),
                                 index * PoseVelocityBlock::SizeAtCompileTime);
    }
    virtual Poses poses() const
    {
        Poses poses_(count() * POSE_SIZE);
        for (int body_index = 0; body_index < count(); body_index++)
        {
            poses_.template middleRows<POSE_SIZE>(body_index * POSE_SIZE) =
                component(body_index).pose();
        }
        return poses_;
    }

    int count() const
    {
        return this->size() / PoseVelocityBlock::SizeAtCompileTime;
    }

    // mutators ****************************************************************
    PoseVelocityBlock component(int index)
    {
        return PoseVelocityBlock(*((State*)(this)),
                                 index * PoseVelocityBlock::SizeAtCompileTime);
    }
    virtual void poses(const Poses& poses_)
    {
        for (int body_index = 0; body_index < count(); body_index++)
        {
            component(body_index).pose() =
                poses_.template middleRows<POSE_SIZE>(body_index * POSE_SIZE);
        }
    }
    void recount(int new_count)
    {
        return this->resize(new_count * PoseVelocityBlock::SizeAtCompileTime);
    }

    template <typename StateType>
    void apply_delta(const StateType& delta)
    {
        recount(delta.count());
        for (size_t i = 0; i < delta.count(); i++)
        {
            component(i).apply_delta(delta.component(i));
        }
    }

    template <typename StateType>
    void subtract(const StateType& mean)
    {
        for (size_t i_obj = 0; i_obj < mean.count(); i_obj++)
        {
            component(i_obj).subtract(mean.component(i_obj));
        }
    }

    virtual void set_zero()
    {
        for (size_t i = 0; i < count(); i++)
        {
            component(i).set_zero();
        }
    }

    virtual void set_zero_pose()
    {
        for (size_t i = 0; i < count(); i++)
        {
            component(i).set_zero_pose();
        }
    }

    virtual void set_zero_velocity()
    {
        for (size_t i = 0; i < count(); i++)
        {
            component(i).set_zero_velocity();
        }
    }
};
}
