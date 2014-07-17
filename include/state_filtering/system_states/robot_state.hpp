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


#ifndef ROBOT_STATE_HPP_
#define ROBOT_STATE_HPP_

#include <state_filtering/system_states/rigid_body_system.hpp>
#include <state_filtering/tools/macros.hpp>
#include <state_filtering/tools/kinematics_from_urdf.hpp>

#include <Eigen/Dense>
#include <boost/static_assert.hpp>

#include <vector>

template<int size_joints>
struct RobotStateTypes
{
  enum
    {
      // For the joints
      COUNT_PER_JOINT = 1,
      JOINT_ANGLE_INDEX = 0,
      //JOINT_VELOCITY_INDEX = 1,
      // TODO: Check if obsolete
      // for the links
      COUNT_PER_BODY = 12,
      BLOCK_COUNT = 3,
      POSITION_INDEX = 0,
      ORIENTATION_INDEX = 3,
      LINEAR_VELOCITY_INDEX = 6,
      ANGULAR_VELOCITY_INDEX = 9,
    };

  typedef RigidBodySystem<size_joints == -1 ? -1 : size_joints * COUNT_PER_JOINT> Base;
};


template<int size_joints = -1, int size_bodies = -1>
class RobotState: public RobotStateTypes<size_joints>::Base
{
public:
  typedef RobotStateTypes<size_joints> Types;
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
      // for the joints
      SIZE_JOINTS = size_joints,
      COUNT_PER_JOINT = Types::COUNT_PER_JOINT,
      JOINT_ANGLE_INDEX = Types::JOINT_ANGLE_INDEX,
      //JOINT_VELOCITY_INDEX = Types::JOINT_VELOCITY_INDEX,
      // For the links
      SIZE_BODIES = size_bodies,
      SIZE_STATE = Base::SIZE_STATE,
      COUNT_PER_BODY = Types::COUNT_PER_BODY,
      BLOCK_COUNT = Types::BLOCK_COUNT,
      POSITION_INDEX = Types::POSITION_INDEX,
      ORIENTATION_INDEX = Types::ORIENTATION_INDEX,
      LINEAR_VELOCITY_INDEX = Types::LINEAR_VELOCITY_INDEX,
      ANGULAR_VELOCITY_INDEX = Types::ANGULAR_VELOCITY_INDEX,
    };

  // give access to base member functions (otherwise it is shadowed)
  using Base::quaternion;
  using Base::state_size;

  // constructor for fixed size without initial value
  RobotState():
    Base(State::Zero(SIZE_STATE)),
    num_bodies_(SIZE_BODIES),
    num_joints_(SIZE_JOINTS)
  {
    assert_fixed_size<true>();
  }

  // constructor for dynamic size without initial value
  RobotState(unsigned num_bodies, 
	     unsigned num_joints,
	     const boost::shared_ptr<KinematicsFromURDF> &kinematics_ptr):
    Base(State::Zero(num_joints * COUNT_PER_JOINT)),
    num_bodies_(num_bodies),
    num_joints_(num_joints),
    kinematics_(kinematics_ptr)
  {
    
    assert_dynamic_size<true>();
  }

  // constructor with initial value
  template <typename T> RobotState(const Eigen::MatrixBase<T>& state_vector):
    Base(state_vector),
    num_joints_(state_vector.rows()/COUNT_PER_JOINT)
  { }


  virtual ~RobotState() {}

  template <typename T> void operator = (const Eigen::MatrixBase<T>& state_vector)
  {
    std::cout << "EQUAL OPERATOR IN ROBOT STATE" << std::endl;
    Base::count_state_ = state_vector.rows();
    kinematics_->InitKDLData(state_vector);
    *((State*)(this)) = state_vector;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// these are the two main functions which have to implement the robot state.
  /// here it simply accesses the eigen vector, from which the class is derived and returns the correct
  /// part of the vector, since it directly encodes the pose of the object. for the robot arm this will
  /// be a bit more complicated, since the eigen vector will contain the joint angles, and we will have
  /// to compute the pose of the link from the joint angles.
  // this returns the position (translation) part of the pose of the link.
  virtual Vector position(const size_t& object_index = 0) const
  {
    return kinematics_->GetLinkPosition(object_index);
  }
  // this returns the orientation part of the pose of the link. the format is euler vector, the norm of the vector is the
  // angle, and the direction is the rotation axis. below there is a function that converts from Quaternion2EulerVector
  // which implements the transformation
  virtual Vector euler_vector(const size_t& object_index = 0) const
  {
    return Quaternion2EulerVector(kinematics_->GetLinkOrientation(object_index));
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  virtual Vector Quaternion2EulerVector(const Quaternion& quaternion) const 
  {
    AngleAxis angle_axis(quaternion);
    return angle_axis.angle()*angle_axis.axis();
  }


  virtual unsigned bodies_size() const
  {
    return num_bodies_;
  }

  virtual unsigned joints_size() const
  {
    return num_joints_;
  }
private:
  unsigned num_bodies_;
  unsigned num_joints_;

  // pointer to the robot kinematic
  const boost::shared_ptr<KinematicsFromURDF>  kinematics_;

  template <bool dummy> void assert_fixed_size() const {BOOST_STATIC_ASSERT(size_joints > -1);}
  template <bool dummy> void assert_dynamic_size() const {BOOST_STATIC_ASSERT(size_joints == -1);}
};



#endif
