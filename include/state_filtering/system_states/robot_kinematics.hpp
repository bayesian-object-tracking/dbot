/*************************************************************************
This software allows for filtering in high-dimensional measurement and
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


#ifndef ROBOT_KINEMATICS_HPP_
#define ROBOT_KINEMATICS_HPP_

#include <state_filtering/system_states/rigid_body_system.hpp>
#include <state_filtering/tools/macros.hpp>

#include <Eigen/Dense>
#include <boost/static_assert.hpp>

#include <vector>

/// TODO: all of this is just a copy of he floating body stuff. in here the robot kinematics have to be implemented

template<int size_bodies>
struct RobotKinematicsTypes
{
    enum
    {
        COUNT_PER_BODY = 12,
        BLOCK_COUNT = 3,
        POSITION_INDEX = 0,
        ORIENTATION_INDEX = 3,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9,
    };

    typedef RigidBodySystem<size_bodies == -1 ? -1 : size_bodies * COUNT_PER_BODY> Base;
};


template<int size_bodies = -1>
class RobotKinematics: public RobotKinematicsTypes<size_bodies>::Base
{
public:
    typedef RobotKinematicsTypes<size_bodies> Types;
    typedef typename Types::Base Base;

    typedef typename Base::Scalar   Scalar;
    typedef typename Base::State    State;
    typedef typename Base::Vector   Vector;

    typedef typename Base::AngleAxis            AngleAxis;
    typedef typename Base::Quaternion           Quaternion;
    typedef typename Base::RotationMatrix       RotationMatrix;
    typedef typename Base::HomogeneousMatrix    HomogeneousMatrix;

    enum
    {
        SIZE_BODIES = size_bodies,
        SIZE_STATE = Base::SIZE_STATE,
        COUNT_PER_BODY = Types::COUNT_PER_BODY,
        BLOCK_COUNT = Types::BLOCK_COUNT,
        POSITION_INDEX = Types::POSITION_INDEX,
        ORIENTATION_INDEX = Types::ORIENTATION_INDEX,
        LINEAR_VELOCITY_INDEX = Types::LINEAR_VELOCITY_INDEX,
        ANGULAR_VELOCITY_INDEX = Types::ANGULAR_VELOCITY_INDEX,
    };

    typedef Eigen::VectorBlock<State, BLOCK_COUNT>      Block;
    typedef Eigen::VectorBlock<State, COUNT_PER_BODY>   BodyBlock;

    // give access to base member functions (otherwise it is shadowed)
    using Base::quaternion;
    using Base::count_state;

    // constructor for fixed size without initial value
    RobotKinematics():
        Base(State::Zero(SIZE_STATE)),
        count_bodies_(SIZE_BODIES)
    {
        assert_fixed_size<true>();
    }

    // constructor for dynamic size without initial value
    RobotKinematics(unsigned count_bodies):
        Base(State::Zero(count_bodies * COUNT_PER_BODY)),
        count_bodies_(count_bodies)
    {
        assert_dynamic_size<true>();
    }

    // constructor with initial value
    template <typename T> RobotKinematics(const Eigen::MatrixBase<T>& state_vector):
        Base(state_vector),
        count_bodies_(state_vector.rows()/COUNT_PER_BODY){ }

    virtual ~RobotKinematics() {}

    // read
    virtual Vector position(const size_t& object_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(object_index * COUNT_PER_BODY + POSITION_INDEX);
    }
    virtual Vector euler_vector(const size_t& object_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(object_index * COUNT_PER_BODY + ORIENTATION_INDEX);
    }
    virtual Vector linear_velocity(const size_t& object_index = 0) const
    {
        TO_BE_IMPLEMENTED
    }
    virtual Vector angular_velocity(const size_t& object_index = 0) const
    {
        TO_BE_IMPLEMENTED
    }


    virtual unsigned count_bodies() const
    {
        return count_bodies_;
    }
private:
    unsigned count_bodies_;

    template <bool dummy> void assert_fixed_size() const {BOOST_STATIC_ASSERT(size_bodies > -1);}
    template <bool dummy> void assert_dynamic_size() const {BOOST_STATIC_ASSERT(size_bodies == -1);}
};



#endif
