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

#ifndef FAST_FILTERING_STATES_RIGID_BODY_SYSTEM_HPP_
#define FAST_FILTERING_STATES_RIGID_BODY_SYSTEM_HPP_

#include <Eigen/Dense>
#include <vector>

namespace ff
{

template<int SIZE_STATE_ = -1>
class RigidBodySystem: public Eigen::Matrix<double, SIZE_STATE_, 1>
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, SIZE_STATE_, 1>       State;
    typedef Eigen::Matrix<Scalar, 3, 1>                 Vector;

    // rotation types
    typedef Eigen::AngleAxis<Scalar>    AngleAxis;
    typedef Eigen::Quaternion<Scalar>   Quaternion;
    typedef Eigen::Matrix<Scalar, 3, 3> RotationMatrix;
    typedef Eigen::Matrix<Scalar, 4, 4> HomogeneousMatrix;

    enum
    {
        SIZE_STATE = SIZE_STATE_
    };

    // constructor and destructor
    template <typename T> RigidBodySystem(const Eigen::MatrixBase<T>& state_vector)
    {
        *this = state_vector;
    }

    virtual ~RigidBodySystem() {}

    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& state_vector)
    {
        count_state_ = state_vector.rows();
        *((State*)(this)) = state_vector;
    }
  
    // read
    virtual Vector position(const size_t& object_index = 0) const = 0;
    virtual Vector euler_vector(const size_t& object_index = 0) const = 0;
    virtual void update() const = 0;

    // other representations
    virtual Quaternion quaternion(const size_t& object_index = 0) const
    {
        Scalar angle = euler_vector(object_index).norm();
        Vector axis = euler_vector(object_index).normalized();

        if(std::isfinite(axis.norm())) 
        {
            return Quaternion(AngleAxis(angle, axis));
        }

        return Quaternion::Identity();
    }
    virtual RotationMatrix rotation_matrix(const size_t& object_index = 0) const
    {
        return RotationMatrix(quaternion(object_index));
    }
    virtual HomogeneousMatrix homogeneous_matrix(const size_t& object_index = 0) const
    {
        HomogeneousMatrix homogeneous_matrix(HomogeneousMatrix::Identity());
        homogeneous_matrix.topLeftCorner(3, 3) = rotation_matrix(object_index);
        homogeneous_matrix.topRightCorner(3, 1) = position(object_index);

        return homogeneous_matrix;
    }

    // counts
    virtual unsigned state_size() const
    {
        return count_state_;
    }

    virtual unsigned bodies_size() const = 0;

private:
    // this is the actual state size, since the SIZE_STATE can also be -1 for dynamic size
    unsigned count_state_;
};

}

#endif
