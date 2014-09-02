/*********************************************************************
 *
 *  Copyright (c) 2013, Jeannette Bohg - MPI for Intelligent System
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

#ifndef _PART_MESH_MODEL_
#define _PART_MESH_MODEL_

#include <boost/shared_ptr.hpp>
#include <urdf/model.h>

#include <Eigen/Dense>

#ifdef HAVE_V2
#include "assimp/assimp.h"
#include "assimp/aiPostProcess.h"
#include "assimp/aiScene.h"
#elif defined HAVE_V3
#include "assimp/cimport.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#endif

class PartMeshModel
{
public:
  PartMeshModel(const boost::shared_ptr<urdf::Link> p_link,
		const std::string &p_description_path,
		unsigned p_index)
    : proper_(false) 
    , link_(p_link)
    , name_(p_link->name)
    , vertices_(new std::vector<Eigen::Vector3d>)
    , indices_(new std::vector<std::vector<int> >)
  {
    
    if ((link_->visual.get() != NULL) &&
	(link_->visual->geometry.get() != NULL) && 
	(link_->visual->geometry->type == urdf::Geometry::MESH))
      { 
        boost::shared_ptr<urdf::Mesh> mesh = boost::dynamic_pointer_cast<urdf::Mesh> (link_->visual->geometry);
        std::string filename (mesh->filename);
	filename_ = filename;

        if (filename.substr(filename.size() - 4 , 4) == ".stl" || 
	    filename.substr(filename.size() - 4 , 4) == ".dae")
	  {
	    if (filename.substr(filename.size() - 4 , 4) == ".dae")
	      filename.replace(filename.size() - 4 , 4, ".stl");
	    filename.erase(0,25);
	    filename = p_description_path + filename;
	    	    
	    scene_ =  aiImportFile(filename.c_str(),
				   aiProcessPreset_TargetRealtime_Quality);
	    numFaces_ = scene_->mMeshes[0]->mNumFaces;

	    original_transform_.linear() = Eigen::Quaterniond(link_->visual->origin.rotation.w,
							      link_->visual->origin.rotation.x, 
							      link_->visual->origin.rotation.y, 
							      link_->visual->origin.rotation.z).toRotationMatrix(); 
	    original_transform_.translation() = Eigen::Vector3d(link_->visual->origin.position.x, 
								link_->visual->origin.position.y, 
								link_->visual->origin.position.z);

	    proper_=true;
	  }
      }
  }

  boost::shared_ptr<std::vector<Eigen::Vector3d> > get_vertices()
  {
    const struct aiMesh* mesh = scene_->mMeshes[0];
    unsigned num_vertices = mesh->mNumVertices;
    vertices_->resize(num_vertices); 
    for(unsigned v=0;v<num_vertices;++v)
      {
    Eigen::Vector3d point;
	point(0) = mesh->mVertices[v].x;
	point(1) = mesh->mVertices[v].y;
	point(2) = mesh->mVertices[v].z;
	vertices_->at(v) = original_transform_ * point;
      }
    return vertices_;
  }

  boost::shared_ptr<std::vector<std::vector<int> > > get_indices()
  {
    const struct aiMesh* mesh = scene_->mMeshes[0];
    unsigned num_faces = mesh->mNumFaces;
    unsigned size_of_face = 3; // assuming triangles, check!
    indices_->resize(num_faces); 
    for (unsigned t = 0; t < num_faces; ++t)
      {
	const struct aiFace* face_ai = &mesh->mFaces[t];
	
	// Check for triangle
	if(face_ai->mNumIndices!=size_of_face) {
	  std::cerr << "not a triangle!" << std::endl;
	  exit(-1);
	}
	
	// fill a triangle with indices
	std::vector<int> triangle(size_of_face);
	for(unsigned j = 0; j < face_ai->mNumIndices; j++)
	  triangle[j] = face_ai->mIndices[j];
	indices_->at(t) = triangle;
      }
    return indices_;
  }

  const std::string& get_name(){return name_;}

  bool proper_;

private:
  const boost::shared_ptr<urdf::Link> link_;
  const struct aiScene* scene_;
  unsigned numFaces_;
  
  boost::shared_ptr<std::vector<Eigen::Vector3d> > vertices_;
  boost::shared_ptr<std::vector<std::vector<int> > > indices_;

  Eigen::Affine3d original_transform_;

  std::string name_;

  std::string filename_;
  
};


#endif // _PART_MESH_MODEL_
