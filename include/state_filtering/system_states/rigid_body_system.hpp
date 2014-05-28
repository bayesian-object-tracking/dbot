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


#ifndef RIGID_BODY_SYSTEM_HPP_
#define RIGID_BODY_SYSTEM_HPP_

#include <Eigen/Dense>
#include <vector>
#include <boost/static_assert.hpp>



template<int state_size = -1>
class RigidBodySystem: public Eigen::Matrix<double, state_size, 1>
{
public:
    typedef Eigen::Matrix<double, state_size, 1> StateVectorType;

    static const int size_states_ = state_size;
    const unsigned count_states_;
    const unsigned count_bodies_;

//    // constructor for static size without initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//    RigidBodySystem(const unsigned& object_count): state_count_(state_size), body_count_(object_count)
//    {
//        assert_fixed_size<true>();
//    }
//    // constructor for dynamic size without initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//    RigidBodySystem(const unsigned& state_count, const unsigned& object_count): state_count_(state_count), body_count_(object_count)
//    {
//        assert_dynamic_size<true>();
//    }
    // constructor with initial value -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    template <typename T> RigidBodySystem(const Eigen::MatrixBase<T>& state_vector, const unsigned& object_count):
        count_states_(state_vector.rows()),
        count_bodies_(object_count)
    {
        *((Eigen::Matrix<double, state_size, 1>*)(this)) = state_vector;
    }
    virtual ~RigidBodySystem() {}


    virtual void set_state(const Eigen::Matrix<double, state_size, 1>& state_vector)
    {
        *((Eigen::Matrix<double, state_size, 1>*)(this)) = state_vector;
    }
    // get functions ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    virtual Eigen::Quaterniond get_quaternion(const size_t& object_index = 0) const = 0;
    virtual Eigen::Matrix3d get_rotation_matrix(const size_t& object_index = 0) const = 0;
    virtual Eigen::Matrix4d get_homogeneous_matrix(const size_t& object_index = 0) const = 0;
    virtual Eigen::Vector3d get_translation(const size_t& object_index = 0) const = 0;

    virtual Eigen::Vector3d get_linear_velocity(const size_t& object_index = 0) const = 0;
    virtual Eigen::Vector3d get_angular_velocity(const size_t& object_index = 0) const = 0;

//private:
//    template <bool dummy> void assert_fixed_size() const {BOOST_STATIC_ASSERT(state_size > -1);}
//    template <bool dummy> void assert_dynamic_size() const {BOOST_STATIC_ASSERT(state_size == -1);}
};






#endif
