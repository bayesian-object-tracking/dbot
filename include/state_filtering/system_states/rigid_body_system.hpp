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
#include <iostream>



template<int size_state = -1>
class RigidBodySystem: public Eigen::Matrix<double, size_state, 1>
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, size_state, 1> State;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector;

    // rotation types
    typedef Eigen::AngleAxis<Scalar> AngleAxis;
    typedef Eigen::Quaternion<Scalar>   Quaternion;
    typedef Eigen::Matrix<Scalar, 3, 3> RotationMatrix;
    typedef Eigen::Matrix<Scalar, 4, 4> HomogeneousMatrix;


    enum
    {
        SIZE_STATE = size_state
    };

    // constructor and destructor
    template <typename T> RigidBodySystem(const Eigen::MatrixBase<T>& state_vector,
                                          const unsigned& count_bodies):
        count_state_(state_vector.rows()),
        count_bodies_(count_bodies)
    {
        *((State*)(this)) = state_vector;
    }
    virtual ~RigidBodySystem() {}

    // set state
    virtual void set_state(const State& state_vector)
    {
        *((State*)(this)) = state_vector;
    }

    // interfaces
    virtual Vector  get_position            (const size_t& object_index = 0) const = 0;
    virtual Vector  get_euler_vector        (const size_t& object_index = 0) const = 0;
    virtual Vector  get_linear_velocity     (const size_t& object_index = 0) const = 0;
    virtual Vector  get_angular_velocity    (const size_t& object_index = 0) const = 0;

    // other representations for orientation
    virtual Quaternion get_quaternion(const size_t& object_index = 0) const
    {
        Scalar angle = get_euler_vector(object_index).norm();
        Vector axis = get_euler_vector(object_index).normalized();
        if(std::isfinite(axis.norm()))
            return Quaternion(AngleAxis(angle, axis));
        else
            return Quaternion::Identity();
    }
    virtual RotationMatrix get_rotation_matrix(const size_t& object_index = 0) const
    {
        return RotationMatrix(get_quaternion(object_index));
    }
    // homo geneous matrix
    virtual HomogeneousMatrix get_homogeneous_matrix(const size_t& object_index = 0) const
    {
        HomogeneousMatrix homogeneous_matrix(HomogeneousMatrix::Identity());
        homogeneous_matrix.topLeftCorner(3, 3) = get_rotation_matrix(object_index);
        homogeneous_matrix.topRightCorner(3, 1) = get_position(object_index);

        return homogeneous_matrix;
    }

    // counts
    virtual unsigned countState() const
    {
        return count_state_;
    }
    virtual unsigned countBodies() const
    {
        return count_bodies_;
    }

private:
    unsigned count_state_;
    unsigned count_bodies_;
};


#endif
