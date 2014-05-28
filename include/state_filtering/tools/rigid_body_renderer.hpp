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


#ifndef POSE_FILTERING_TRIANGLE_OBJEC_tMODEL_HPP_
#define POSE_FILTERING_TRIANGLE_OBJEC_tMODEL_HPP_

#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>
#include <state_filtering/system_states/rigid_body_system.hpp>


namespace obj_mod
{

class RigidBodyRenderer
{
public:
    RigidBodyRenderer(
            const std::vector<std::vector<Eigen::Vector3d> >& vertices,
            const std::vector<std::vector<std::vector<int> > >& indices,
            const boost::shared_ptr<RigidBodySystem<-1> >& rigid_body_system);

    virtual ~RigidBodyRenderer();

	void PredictObservation(
			Eigen::Matrix3d camera_matrix,
			int n_rows, int n_cols,
			std::vector<int> &intersec_tindices,
			std::vector<float> &depth) const;


    virtual void set_state(const Eigen::VectorXd& state)
    {
        _rigid_body_system->set_state(state);
        _R.resize(_rigid_body_system->count_bodies_);
        _t.resize(_rigid_body_system->count_bodies_);
        for(size_t part_index = 0; part_index < _rigid_body_system->count_bodies_; part_index++)
        {
            _R[part_index] = _rigid_body_system->get_rotation_matrix(part_index);
            _t[part_index] = _rigid_body_system->get_translation(part_index);
        }
    }

    Eigen::VectorXd get_rigid_body_system() const;
	std::vector<Eigen::Matrix4d> get_hom() const;
	std::vector<Eigen::Matrix3d> get_R() const;
	std::vector<Eigen::Vector3d> get_t() const;

	std::vector<std::vector<Eigen::Vector3d> > get_vertices() const;

	Eigen::Vector3d get_com() const;

    Eigen::Vector3d  get_coms(const size_t& index) const;

protected:
	std::vector<std::vector<Eigen::Vector3d> > _vertices;
	std::vector<std::vector<Eigen::Vector3d> > _normals;
	std::vector<std::vector<std::vector<int> > > _indices;

	std::vector<Eigen::Matrix3d> _R;
	std::vector<Eigen::Vector3d> _t;

    const boost::shared_ptr<RigidBodySystem<-1> > _rigid_body_system;

	std::vector<Eigen::Vector3d> _coms;
	std::vector<float> _com_weights;
};

}

#endif
