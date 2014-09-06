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

#include <pose_tracking_interface/utils/kinematics_from_urdf.hpp>

#include <boost/random/normal_distribution.hpp>

KinematicsFromURDF::KinematicsFromURDF()
  : nh_priv_("~")
{
  // Load robot description from parameter server
  std::string desc_string;
  if(!nh_.getParam("robot_description", desc_string))
    ROS_ERROR("Could not get urdf from param server at %s", desc_string.c_str());

  // Initialize URDF object from robot description
  if (!urdf_.initString(desc_string))
    ROS_ERROR("Failed to parse urdf");
 
  // set up kinematic tree from URDF
  if (!kdl_parser::treeFromUrdfModel(urdf_, kin_tree_)){
    ROS_ERROR("Failed to construct kdl tree");
    return;
  }  

  // setup path for robot description and root of the tree
  nh_priv_.param<std::string>("robot_description_package_path", description_path_, "..");
  //nh_priv_.param<std::string>("tf_correction_root", tf_correction_root_, "L_SHOULDER" );


  // create segment map for correct ordering of joints
  segment_map_ =  kin_tree_.getSegments();
  boost::shared_ptr<const urdf::Joint> joint;
  joint_map_.resize(kin_tree_.getNrOfJoints());
  lower_limit_.resize(kin_tree_.getNrOfJoints());
  upper_limit_.resize(kin_tree_.getNrOfJoints());
  for (KDL::SegmentMap::const_iterator seg_it = segment_map_.begin(); seg_it != segment_map_.end(); ++seg_it)
    {
	
      if (seg_it->second.segment.getJoint().getType() != KDL::Joint::None)
	{
	  joint = urdf_.getJoint(seg_it->second.segment.getJoint().getName().c_str());
	  // check, if joint can be found in the URDF model of the object/robot
	  if (!joint)
	    {
	      ROS_FATAL("Joint '%s' has not been found in the URDF robot model! Aborting ...", joint->name.c_str());
	      return;
	    }
	  // extract joint information
	  if (joint->type != urdf::Joint::UNKNOWN && joint->type != urdf::Joint::FIXED)
	    {
	      joint_map_[seg_it->second.q_nr] = joint->name;
	      lower_limit_[seg_it->second.q_nr] = joint->limits->lower;
	      upper_limit_[seg_it->second.q_nr] = joint->limits->upper;
	    }
	}
    }
  
  std::string cam_frame, base_frame;
  nh_priv_.param<std::string>("camera_frame", cam_frame, "XTION");
  nh_priv_.param<std::string>("kinematic_frame", base_frame, "BASE" );
  // create chain from base to camera
  if(kin_tree_.getChain(cam_frame, base_frame, base_2_cam_))
    ROS_INFO("Sucessfully created chain from %s to %s", cam_frame.c_str(), base_frame.c_str());
  else 
    ROS_ERROR("Could not create chain from %s to %s", cam_frame.c_str(), base_frame.c_str());
  chain_solver_ = new KDL::ChainFkSolverPos_recursive(base_2_cam_);
  
  // initialise kinematic tree solver
  tree_solver_ = new KDL::TreeFkSolverPos_recursive(kin_tree_);
}

KinematicsFromURDF::~KinematicsFromURDF()
{
  delete tree_solver_;
  delete chain_solver_;
}

void KinematicsFromURDF::GetPartMeshes(std::vector<boost::shared_ptr<PartMeshModel> > &part_meshes)
{
  //Load robot mesh for each link
  std::vector<boost::shared_ptr<urdf::Link> > links;
  urdf_.getLinks(links);
  std::string global_root =  urdf_.getRoot()->name;
  for (unsigned i=0; i< links.size(); i++)
    {
      // keep only the links descending from our root
      boost::shared_ptr<urdf::Link> tmp_link = links[i];
      while(//tmp_link->name.compare(tf_correction_root_)!=0 && 
	    tmp_link->name.compare(global_root)!=0)
	{
	  tmp_link = tmp_link->getParent();
	}
      
      boost::shared_ptr<PartMeshModel> part_ptr(new PartMeshModel(links[i], description_path_, i));
      if(part_ptr->proper_)
	{
	  // if the link has an actual mesh file to read
	  std::cout << "link " << links[i]->name << " is descendant of " << tmp_link->name << std::endl;
	  part_meshes.push_back(part_ptr);
	  // Produces an index map for the links
	  part_mesh_map_.push_back(part_ptr->get_name());
	}
    }
}

void KinematicsFromURDF::InitKDLData(const Eigen::VectorXd& joint_state)
{
  // Internally, KDL array use Eigen Vectors


  // DEBUG: randomly shuffle Vector to see if it has a noticable effect
  /*  std::vector<int> vec(joint_state.rows());
  for (int i = 0; i < vec.size(); ++i)
    vec[i] = i;
  std::random_shuffle(vec.begin(),vec.end());
  
  jnt_array_.data.resize(joint_state.rows());
  for (int i = 0; i < vec.size(); ++i)
    jnt_array_.data(i) = joint_state(vec[i]);
  */
  jnt_array_.data = joint_state;
  // Get the transform from the robot base to the camera frame
  SetCameraTransform();
  // Given the new joint angles, compute all link transforms in one go
  ComputeLinkTransforms();
}

void KinematicsFromURDF::SetCameraTransform()
{
  // loop over all the joints in the chain from base to camera
  KDL::JntArray chain_jnt_array(base_2_cam_.getNrOfJoints());
  std::vector<std::string >::const_iterator location;
  int j=0;
  int n_segments = base_2_cam_.getNrOfSegments();
  for( int i=0;i<n_segments;i++){
    // for each valid joint in the chain from base to camera
    if(base_2_cam_.getSegment(i).getJoint().getType()!=KDL::Joint::None){
      // get the name of the joints
      std::string name = base_2_cam_.getSegment(i).getJoint().getName();
      // find its index in the joint map
      location = std::find( joint_map_.begin(), joint_map_.end(), name );
      if ( location == joint_map_.end() )
	ROS_ERROR("Joint in chain not in JointState. This should never happen.\n");
      // fill the associated joint array accordingly
      chain_jnt_array(j) = jnt_array_(location-joint_map_.begin());
      j++;
    }
  }
  
  // get transform from base to camera frame
  if(chain_solver_->JntToCart(chain_jnt_array, cam_frame_)<0)
    ROS_ERROR("Could get transform from base to camera\n");
}

void KinematicsFromURDF::ComputeLinkTransforms( )
{
  // loop over all segments to compute the link transformation
  for (KDL::SegmentMap::const_iterator seg_it = segment_map_.begin(); 
       seg_it != segment_map_.end(); ++seg_it)
    {
      if (std::find(part_mesh_map_.begin(), 
		    part_mesh_map_.end(), 
		    seg_it->second.segment.getName()) != part_mesh_map_.end())
	{
	  KDL::Frame frame;
	  if(tree_solver_->JntToCart(jnt_array_, frame, seg_it->second.segment.getName())<0)
	    ROS_ERROR("TreeSolver returned an error for link %s", 
		      seg_it->second.segment.getName().c_str());
	  frame_map_[seg_it->second.segment.getName()] = cam_frame_ * frame;
	}
    }
}

Eigen::VectorXd KinematicsFromURDF::GetLinkPosition( int idx)
{
  Eigen::VectorXd pos(3);
  pos << frame_map_[part_mesh_map_[idx]].p.x(), frame_map_[part_mesh_map_[idx]].p.y(),frame_map_[part_mesh_map_[idx]].p.z(); 
  return pos;
}

Eigen::Quaternion<double> KinematicsFromURDF::GetLinkOrientation( int idx)
{
  Eigen::Quaternion<double> quat;
  frame_map_[part_mesh_map_[idx]].M.GetQuaternion(quat.x(), quat.y(), quat.z(), quat.w());
  return quat;
}

std::vector<Eigen::VectorXd> KinematicsFromURDF::GetInitialSamples(const sensor_msgs::JointState &state,
								   int initial_sample_count,
								   float ratio_std)
{
  std::vector<Eigen::VectorXd> samples;
  samples.reserve(initial_sample_count);
  for(int i=0; i<initial_sample_count; ++i)
    {
      Eigen::VectorXd sample(state.position.size());
      // loop over all joint and fill in KDL array
      for(std::vector<double>::const_iterator jnt = state.position.begin(); 
	  jnt !=state.position.end(); ++jnt)
	{
	  int tmp_index = GetJointIndex(state.name[jnt-state.position.begin()]);
	  if (tmp_index >=0)
	    {
	      double new_jnt;
	      std::string name = state.name[jnt-state.position.begin()];
	      new_jnt = GetRandomPertubation( tmp_index, *jnt, ratio_std);
	      sample(tmp_index) = new_jnt;
	    } else 
	    ROS_ERROR("i: %d, No joint index for %s", 
		      (int)(jnt-state.position.begin()), 
		      state.name[jnt-state.position.begin()].c_str());
	}
      samples.push_back(sample);
    }
  return samples;
}

std::vector<Eigen::VectorXd> KinematicsFromURDF::GetInitialJoints(const sensor_msgs::JointState &state)
{
  std::vector<Eigen::VectorXd> samples;
  Eigen::VectorXd sample(num_joints());
  // loop over all joint and fill in KDL array
  for(std::vector<double>::const_iterator jnt = state.position.begin(); 
      jnt !=state.position.end(); ++jnt)
    {
      int tmp_index = GetJointIndex(state.name[jnt-state.position.begin()]);

      if (tmp_index >=0) 
	sample(tmp_index) = *jnt;
	else 
	ROS_ERROR("i: %d, No joint index for %s", 
		  (int)(jnt-state.position.begin()), 
		  state.name[jnt-state.position.begin()].c_str());
    }
  samples.push_back(sample);

  return samples;
}

void KinematicsFromURDF::GetDependencies(std::vector<std::vector<size_t> >& dependencies)
{
  // only one fully dependent object -> the robot 
  std::vector<size_t> robot_deps;
  for(int i=0; i<num_joints(); ++i)
    robot_deps.push_back(i);
  dependencies.push_back(robot_deps);
}

KDL::Tree KinematicsFromURDF::GetTree()
{
  return kin_tree_;
}

double KinematicsFromURDF::GetRandomPertubation(int jnt_index, double jnt_angle, double ratio)
{
  double mean = jnt_angle;
  double range = upper_limit_[jnt_index]-lower_limit_[jnt_index];
  double std  = ratio * range;
  boost::normal_distribution<double> normal(mean, std);
  double val = normal(generator_);
  
  // clip the values to the limits
  if(val>upper_limit_[jnt_index])
    val = upper_limit_[jnt_index];
  
  if(val<lower_limit_[jnt_index])
    val = lower_limit_[jnt_index];
    
  return val;
  
}

int KinematicsFromURDF::GetJointIndex(const std::string &name)
{
    for (unsigned int i=0; i < joint_map_.size(); ++i)
      if (joint_map_[i] == name)
	return i;
    return -1;
}

std::string KinematicsFromURDF::GetLinkName(int idx)
{
  return part_mesh_map_[idx];
}

int KinematicsFromURDF::num_joints()
{
  return kin_tree_.getNrOfJoints();
}

int KinematicsFromURDF::num_links()
{
    return part_mesh_map_.size();
}


std::vector<std::string> KinematicsFromURDF::GetJointMap()
{
  return joint_map_;
}

std::string KinematicsFromURDF::GetRootFrameID()
{
  return kin_tree_.getRootSegment()->first;
}
