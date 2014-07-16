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

#include <state_filtering/tools/kinematics_from_urdf.hpp>

KinematicsFromURDF::KinematicsFromURDF()
  : nh_priv_("~")
{
  // Load robot description from parameter server
  std::string desc_string;
  if(!nh_.getParam("robot_description", desc_string))
    ROS_ERROR("Could not get urdf from param server at %s", desc_string.c_str());

  // Initialize URDF object from robot description
  if (!urdf_.initFile(desc_string))
    ROS_ERROR("Failed to parse urdf");
 
  // set up kinematic tree from URDF
  if (!kdl_parser::treeFromUrdfModel(urdf_, kin_tree_)){
    ROS_ERROR("Failed to construct kdl tree");
    return;
  }  

  // setup path fro robot description and root of the tree
  nh_priv_.param<std::string>("robot_description_package_path", description_path_, "..");
  nh_priv_.param<std::string>("tf_correction_root", tf_correction_root_, "L_SHOULDER" );


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
  nh_priv_.param<std::string>("camera_frame", cam_frame, "XTION" );
  nh_priv_.param<std::string>("kinematic_frame", base_frame, "BASE" );
  // create chain from base to camera
  kin_tree_.getChain(cam_frame, base_frame, cam_2_base_);
  chain_solver_ = new KDL::ChainFkSolverPos_recursive(cam_2_base_);
  
  // initialise kinematic tree solver
  tree_solver_ = new KDL::TreeFkSolverPos_recursive(kin_tree_);
}

void KinematicsFromURDF::Get_part_meshes(std::vector<boost::shared_ptr<PartMeshModel> > &part_meshes)
{
  //Load robot mesh for each link
  std::vector<boost::shared_ptr<urdf::Link> > links;
  urdf_.getLinks(links);
  std::string global_root =  urdf_.getRoot()->name;
  for (unsigned i=0; i< links.size(); i++)
    {
      // keep only the links descending from our root
      boost::shared_ptr<urdf::Link> tmp_link = links[i];
      while(tmp_link->name.compare(tf_correction_root_)!=0 && 
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
	}
    }
}

int KinematicsFromURDF::Get_number_joints()
{
  return kin_tree_.getNrOfJoints();
}


