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

#ifndef POSE_TRACKING_STATES_RIGID_BODIES_STATE_HPP
#define POSE_TRACKING_STATES_RIGID_BODIES_STATE_HPP

#include <Eigen/Dense>
#include <vector>
#include <fl/util/math/pose_velocity_vector.hpp>

namespace ff
{

template<int Dimension = -1>
class RigidBodiesState: public Eigen::Matrix<double, Dimension, 1>
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Dimension, 1>       State;
    typedef Eigen::Matrix<Scalar, 3, 1>               Vector;

    // rotation types
    typedef Eigen::AngleAxis<Scalar>    AngleAxis;
    typedef Eigen::Quaternion<Scalar>   Quaternion;
    typedef Eigen::Matrix<Scalar, 3, 3> RotationMatrix;
    typedef Eigen::Matrix<Scalar, 4, 4> HomogeneousMatrix;

    typedef fl::PoseVelocityBlock<State> PoseVelocityBlock;

    // constructor and destructor
    RigidBodiesState() { }
    template <typename T> RigidBodiesState(const Eigen::MatrixBase<T>& state_vector)
    {
        *this = state_vector;
    }

    virtual ~RigidBodiesState() {}

    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& state_vector)
    {
        *((State*)(this)) = state_vector;
    }
  
    // read
    virtual Vector position(const size_t& object_index = 0) const = 0;
    virtual Vector euler_vector(const size_t& object_index = 0) const = 0;

    // other representations
//    virtual Quaternion quaternion(const size_t& object_index = 0) const
//    {
//        Scalar angle = euler_vector(object_index).norm();
//        Vector axis = euler_vector(object_index).normalized();

//        if(std::isfinite(axis.norm()))
//        {
//            return Quaternion(AngleAxis(angle, axis));
//        }
//        return Quaternion::Identity();
//    }
//    virtual RotationMatrix rotation_matrix(const size_t& object_index = 0) const
//    {
//        return RotationMatrix(quaternion(object_index));
//    }
//    virtual HomogeneousMatrix homogeneous_matrix(const size_t& object_index = 0) const
//    {
//        HomogeneousMatrix homogeneous_matrix(HomogeneousMatrix::Identity());
//        homogeneous_matrix.topLeftCorner(3, 3) = rotation_matrix(object_index);
//        homogeneous_matrix.topRightCorner(3, 1) = position(object_index);

//        return homogeneous_matrix;
//    }

    virtual unsigned body_count() const = 0;


    virtual fl::PoseVelocityVector component(int index) const = 0;

    virtual int count() const = 0;
};

}

#endif
