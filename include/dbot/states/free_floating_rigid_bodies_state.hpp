/*************************************************************************
This software allows for filtering in high-dimensional observation and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/


#ifndef POSE_TRACKING_STATES_FREE_FLOATING_RIGID_BODIES_STATE_HPP
#define POSE_TRACKING_STATES_FREE_FLOATING_RIGID_BODIES_STATE_HPP

#include <Eigen/Dense>
#include <vector>

#include <fl/util/types.hpp>

#include <dbot/states/rigid_bodies_state.hpp>


namespace dbot
{

template<int BodyCount>
struct FreeFloatingRigidBodiesStateTypes
{
    enum
    {
        BODY_SIZE = 12,
        POSE_SIZE = 6,
    };

    typedef RigidBodiesState<BodyCount == -1 ? -1 : BodyCount * BODY_SIZE> Base;
};


template<int BodyCount = -1>
class FreeFloatingRigidBodiesState:
                public FreeFloatingRigidBodiesStateTypes<BodyCount>::Base
{
public:
    typedef FreeFloatingRigidBodiesStateTypes<BodyCount> Types;
    typedef typename Types::Base Base;
    enum
    {
        BODY_SIZE = Types::BODY_SIZE,
        POSE_SIZE = Types::POSE_SIZE,
    };

    typedef typename Base::State    State;
    typedef Eigen::Matrix<fl::Real,
                        BodyCount == -1 ? -1 : BodyCount * POSE_SIZE, 1> Poses;
    typedef typename Base::PoseVelocityBlock PoseVelocityBlock;


    // constructor and destructor **********************************************
    FreeFloatingRigidBodiesState() { }
    FreeFloatingRigidBodiesState(unsigned count_bodies):
                            Base(State::Zero(count_bodies * BODY_SIZE)) { }

    template <typename T>
    FreeFloatingRigidBodiesState(const Eigen::MatrixBase<T>& state_vector):
                                                          Base(state_vector) { }
    virtual ~FreeFloatingRigidBodiesState() noexcept {}

    // accessors ***************************************************************
    virtual fl::PoseVelocityVector component(int index) const
    {
        return PoseVelocityBlock(*((State*)(this)),
                                 index * PoseVelocityBlock::SizeAtCompileTime);
    }
    virtual Poses poses() const
    {
        Poses poses_(count()*POSE_SIZE);
        for(int body_index = 0; body_index < count(); body_index++)
        {
             poses_.template middleRows<POSE_SIZE>(body_index * POSE_SIZE)
                                = component(body_index).pose();
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
        for(int body_index = 0; body_index < count(); body_index++)
        {
             component(body_index).pose()
                = poses_.template middleRows<POSE_SIZE>(body_index * POSE_SIZE);
        }
    }
    void recount(int new_count)
    {
        return this->resize(new_count * PoseVelocityBlock::SizeAtCompileTime);
    }
};

}

#endif
