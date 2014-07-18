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


#ifndef FLOATING_BODY_SYSTEM_HPP_
#define FLOATING_BODY_SYSTEM_HPP_

#include <state_filtering/system_states/rigid_body_system.hpp>

#include <Eigen/Dense>
#include <boost/static_assert.hpp>

#include <vector>



template<int size_bodies>
struct FloatingBodySystemTypes
{
    enum
    {
        COUNT_PER_BODY = 12,
        POSE_COUNT = 6,
        BLOCK_COUNT = 3,
        POSITION_INDEX = 0,
        POSE_INDEX = 0,
        ORIENTATION_INDEX = 3,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9,
    };

    typedef RigidBodySystem<size_bodies == -1 ? -1 : size_bodies * COUNT_PER_BODY> Base;
    typedef Eigen::Matrix<typename Base::Scalar, size_bodies == -1 ? -1 : size_bodies * POSE_COUNT, 1> Poses;
};


template<int size_bodies = -1>
class FloatingBodySystem: public FloatingBodySystemTypes<size_bodies>::Base
{
public:
    typedef FloatingBodySystemTypes<size_bodies> Types;
    typedef typename Types::Base Base;
    enum
    {
        SIZE_BODIES = size_bodies,
        SIZE_STATE = Base::SIZE_STATE,
        COUNT_PER_BODY = Types::COUNT_PER_BODY,
        POSE_COUNT = Types::POSE_COUNT,
        BLOCK_COUNT = Types::BLOCK_COUNT,
        POSITION_INDEX = Types::POSITION_INDEX,
        POSE_INDEX = Types::POSE_INDEX,
        ORIENTATION_INDEX = Types::ORIENTATION_INDEX,
        LINEAR_VELOCITY_INDEX = Types::LINEAR_VELOCITY_INDEX,
        ANGULAR_VELOCITY_INDEX = Types::ANGULAR_VELOCITY_INDEX,
    };

    typedef typename Base::Scalar   Scalar;
    typedef typename Base::State    State;
    typedef typename Base::Vector   Vector;
    typedef typename Eigen::Matrix<Scalar, POSE_COUNT, 1> Pose;
    typedef typename Types::Poses Poses;

    typedef typename Base::AngleAxis            AngleAxis;
    typedef typename Base::Quaternion           Quaternion;
    typedef typename Base::RotationMatrix       RotationMatrix;
    typedef typename Base::HomogeneousMatrix    HomogeneousMatrix;
    typedef typename Eigen::Transform<Scalar, 3, Eigen::Affine> Affine;

    typedef Eigen::VectorBlock<State, BLOCK_COUNT>      Block;
    typedef Eigen::VectorBlock<State, POSE_COUNT>       PoseBlock;
    typedef Eigen::VectorBlock<State, COUNT_PER_BODY>   BodyBlock;

    // give access to base member functions (otherwise it is shadowed)
    using Base::quaternion;
    using Base::state_size;

    // constructor for fixed size without initial value
    FloatingBodySystem():
        Base(State::Zero(SIZE_STATE)),
        count_bodies_(SIZE_BODIES)
    {
        assert_fixed_size<true>();
    }

    // constructor for dynamic size without initial value
    FloatingBodySystem(unsigned count_bodies):
        Base(State::Zero(count_bodies * COUNT_PER_BODY)),
        count_bodies_(count_bodies)
    {
        assert_dynamic_size<true>();
    }

    // constructor with initial value
    template <typename T> FloatingBodySystem(const Eigen::MatrixBase<T>& state_vector):
        Base(state_vector),
        count_bodies_(state_vector.rows()/COUNT_PER_BODY){ }

    virtual ~FloatingBodySystem() {}

  virtual void update() const
  {}
  
    // read
    virtual Vector position(const size_t& body_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(body_index * COUNT_PER_BODY + POSITION_INDEX);
    }
    virtual Vector euler_vector(const size_t& body_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(body_index * COUNT_PER_BODY + ORIENTATION_INDEX);
    }
    virtual Vector linear_velocity(const size_t& body_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(body_index * COUNT_PER_BODY + LINEAR_VELOCITY_INDEX);
    }
    virtual Vector angular_velocity(const size_t& body_index = 0) const
    {
        return this->template middleRows<BLOCK_COUNT>(body_index * COUNT_PER_BODY + ANGULAR_VELOCITY_INDEX);
    }
    virtual Pose pose(const size_t& body_index = 0) const
    {
        return this->template middleRows<POSE_COUNT>(body_index * COUNT_PER_BODY + POSE_INDEX);
    }
    virtual Poses poses() const
    {
        Poses poses_(bodies_size()*POSE_COUNT);
        for(size_t body_index = 0; body_index < bodies_size(); body_index++)
        {
             poses_.template middleRows<POSE_COUNT>(body_index * POSE_COUNT) = pose(body_index);
        }
        return poses_;
    }

    // write
    virtual void poses(const Poses& poses_)
    {
        for(size_t body_index = 0; body_index < bodies_size(); body_index++)
        {
             pose(body_index) = poses_.template middleRows<POSE_COUNT>(body_index * POSE_COUNT);
        }
    }
    Block position(const size_t& body_index = 0)
    {
      return Block(this->derived(), body_index * COUNT_PER_BODY + POSITION_INDEX);
    }
    Block euler_vector(const size_t& body_index = 0)
    {
      return Block(this->derived(), body_index * COUNT_PER_BODY + ORIENTATION_INDEX);
    }
    Block linear_velocity(const size_t& body_index = 0)
    {
      return Block(this->derived(), body_index * COUNT_PER_BODY + LINEAR_VELOCITY_INDEX);
    }
    Block angular_velocity(const size_t& body_index = 0)
    {
      return Block(this->derived(), body_index * COUNT_PER_BODY + ANGULAR_VELOCITY_INDEX);
    }
    PoseBlock pose(const size_t& body_index = 0)
    {
      return PoseBlock(this->derived(), body_index * COUNT_PER_BODY + POSE_INDEX);
    }
    BodyBlock operator [](const size_t& body_index)
    {
      return BodyBlock(this->derived(), body_index * COUNT_PER_BODY);
    }

    // other representations
    virtual void quaternion(const Quaternion& quaternion, const size_t& body_index = 0)
    {
        AngleAxis angle_axis(quaternion.normalized());
        euler_vector(body_index) = angle_axis.angle()*angle_axis.axis();
    }
    virtual void pose(const Affine& affine, const size_t& body_index = 0)
    {
       quaternion(Quaternion(affine.rotation()), body_index);
       position(body_index) = affine.translation();
    }


    virtual unsigned bodies_size() const
    {
        return count_bodies_;
    }
private:
    unsigned count_bodies_;

    template <bool dummy> void assert_fixed_size() const {BOOST_STATIC_ASSERT(size_bodies > -1);}
    template <bool dummy> void assert_dynamic_size() const {BOOST_STATIC_ASSERT(size_bodies == -1);}
};



#endif
