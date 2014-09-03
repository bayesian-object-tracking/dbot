/*********************************************************************
 *
 *  Copyright (c) 2014, Jeannette Bohg - MPI for Intelligent System
 *  (jbohg@tuebingen.mpg.de)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Jeannette Bohg nor the names of MPI
 *     may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/


#ifndef STATE_FILTERING_UTILS_KINEMATICS_FROM_URDF_HPP_
#define STATE_FILTERING_UTILS_KINEMATICS_FROM_URDF_HPP_

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <list>
#include <vector>
#include <list>
#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

// tools
#include <tracking/utils/part_mesh_model.hpp>

class KinematicsFromURDF
{
public:
  
  KinematicsFromURDF();
  ~KinematicsFromURDF();
  
  // outputs a list of mesh model objects
  void GetPartMeshes(std::vector<boost::shared_ptr<PartMeshModel> > &part_meshes);

  // Initialises the KDL data and specifically the camera pose
  void InitKDLData(const Eigen::VectorXd& joint_state);

  // Get the position of the robot link with index idx
  Eigen::VectorXd GetLinkPosition( int idx);

  // Get the orientation of the robot link with index idx
  Eigen::Quaternion<double> GetLinkOrientation( int idx);
  
  // Get initial samples around input joint state
  std::vector<Eigen::VectorXd> GetInitialSamples(const sensor_msgs::JointState &state,
						 int initial_sample_count,
						 float ratio_std = 0.1);

  // Convert Joint message to Eigen vector
  std::vector<Eigen::VectorXd> GetInitialJoints(const sensor_msgs::JointState &state);

  // Get dependencies
  void GetDependencies(std::vector<std::vector<size_t> >& dependencies);
  
  // return the KDL kinematic tree
  KDL::Tree GetTree();

  // Get the number of joints
  int num_joints();

  std::string GetLinkName(int idx);

  std::vector<std::string> GetJointMap();

  std::string GetRootFrameID();

private:
  
  // compute the camera frame for the current joint angles
  void SetCameraTransform();
  
  // compute the transformations for all the links in one go
  void ComputeLinkTransforms();
    
  // get the joint index in state array 
  int GetJointIndex(const std::string &name);
  
  // get a random pertubation of a joint within its limits
  double GetRandomPertubation(int jnt_index, double jnt_angle, double ratio);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_priv_;
  //std::string tf_correction_root_;
  std::string description_path_;

  // model as constructed form the robot urdf description
  urdf::Model urdf_;
  // KDL kinematic tree
  KDL::Tree kin_tree_;
  // KDL Kinematic chain from camera to robot base
  KDL::Chain base_2_cam_;

  // maps joint indices to joint names and joint limits
  std::vector<std::string> joint_map_;
  std::vector<float> lower_limit_;
  std::vector<float> upper_limit_;

  // maps mesh indices to link names
  std::vector<std::string> part_mesh_map_;
  // maps link names to KDL frames
  std::map<std::string, KDL::Frame> frame_map_;

  // KDL segment map connecting link segments to joints
  KDL::SegmentMap segment_map_;
  // Forward kinematics solver
  KDL::TreeFkSolverPos_recursive *tree_solver_;
  KDL::ChainFkSolverPos_recursive *chain_solver_;

  // KDL copy of the joint state
  KDL::JntArray jnt_array_;
  // Contains Camera pose relative to base
  KDL::Frame    cam_frame_;
  
  // random generator for joint angle sampling
  boost::mt19937 generator_;

};

#endif
