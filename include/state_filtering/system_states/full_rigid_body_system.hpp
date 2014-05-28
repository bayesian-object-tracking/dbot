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


#ifndef FULL_RIGID_BODY_SYSTEM_HPP_
#define FULL_RIGID_BODY_SYSTEM_HPP_

#include <state_filtering/system_states/rigid_body_system.hpp>

#include <Eigen/Dense>
#include <boost/static_assert.hpp>


#include <vector>



static const unsigned count_per_object = 13;

enum sub_index{translation_index = 0, orientation_index = 3, linear_velocity_index = 7, angular_velocity_index = 10};


template<int body_size = -1>
class FullRigidBodySystem: public RigidBodySystem<body_size == -1 ? -1 : body_size * count_per_object>
{
public:
    static const int size_bodies_ = body_size;

    typedef FullRigidBodySystem<body_size> this_type;
    typedef typename RigidBodySystem<this_type::size_states_>::StateVectorType StateVectorType;

    // constructor for static size without initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FullRigidBodySystem():
        RigidBodySystem<this_type::size_states_> (StateVectorType::Zero(this_type::size_states_), body_size)
    {
        assert_fixed_size<true>();
        reset_quaternions();
    }
    // constructor for dynamic size without initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FullRigidBodySystem(unsigned object_count):
        RigidBodySystem<this_type::size_states_> (StateVectorType::Zero(object_count * count_per_object), object_count)
    {
        assert_dynamic_size<true>();
        reset_quaternions();
    }
    // constructor with initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    template <typename T> FullRigidBodySystem(const Eigen::MatrixBase<T>& state_vector):
        RigidBodySystem<this_type::size_states_> (state_vector, state_vector.rows()/count_per_object){ }

    virtual ~FullRigidBodySystem() {}

    // implementation of get fcts from parent class --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    virtual Eigen::Quaterniond get_quaternion(const size_t& object_index = 0) const
    {
        return Eigen::Quaterniond(this->template middleRows<4>(object_index * count_per_object + orientation_index));
    }
    virtual Eigen::Matrix3d get_rotation_matrix(const size_t& object_index = 0) const
    {
        return Eigen::Matrix3d(get_quaternion(object_index));
    }
    virtual Eigen::Matrix4d get_homogeneous_matrix(const size_t& object_index = 0) const
    {
        Eigen::Matrix4d homogeneous_matrix(Eigen::Matrix4d::Identity());
        homogeneous_matrix.topLeftCorner(3, 3) = get_rotation_matrix(object_index);
        homogeneous_matrix.topRightCorner(3, 1) = get_translation(object_index);

        return homogeneous_matrix;
    }
    virtual Eigen::Vector3d get_translation(const size_t& object_index = 0) const
    {
        return this->template middleRows<3>(object_index * count_per_object + translation_index);
    }
    virtual Eigen::Vector3d get_linear_velocity(const size_t& object_index = 0) const
    {
        return this->template middleRows<3>(object_index * count_per_object + linear_velocity_index);
    }
    virtual Eigen::Vector3d get_angular_velocity(const size_t& object_index = 0) const
    {
        return this->template middleRows<3>(object_index * count_per_object + angular_velocity_index);
    }

    // child specific fcts ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Eigen::VectorBlock<StateVectorType, 3> translation(const size_t& object_index = 0)
    {
      return Eigen::VectorBlock<StateVectorType, 3>(this->derived(), object_index * count_per_object + translation_index);
    }
    Eigen::VectorBlock<StateVectorType, 4> orientation(const size_t& object_index = 0)
    {
      return Eigen::VectorBlock<StateVectorType, 4>(this->derived(), object_index * count_per_object + orientation_index);
    }
    Eigen::VectorBlock<StateVectorType, 3> linear_velocity(const size_t& object_index = 0)
    {
      return Eigen::VectorBlock<StateVectorType, 3>(this->derived(), object_index * count_per_object + linear_velocity_index);
    }
    Eigen::VectorBlock<StateVectorType, 3> angular_velocity(const size_t& object_index = 0)
    {
      return Eigen::VectorBlock<StateVectorType, 3>(this->derived(), object_index * count_per_object + angular_velocity_index);
    }

    Eigen::VectorBlock<StateVectorType, count_per_object> operator [](const size_t& object_index)
    {
      return Eigen::VectorBlock<StateVectorType, count_per_object>(this->derived(), object_index * count_per_object);
    }


private:
    template <bool dummy> void assert_fixed_size() const {BOOST_STATIC_ASSERT(body_size > -1);}
    template <bool dummy> void assert_dynamic_size() const {BOOST_STATIC_ASSERT(body_size == -1);}

    void reset_quaternions()
    {
        for(size_t object_index = 0; object_index < this->count_bodies_; object_index++)
            this->template middleRows<4>(object_index * count_per_object +  3) = Eigen::Quaterniond::Identity().coeffs();
    }
};



#endif
