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
    typedef boost::shared_ptr<RigidBodyRenderer> Ptr;
    typedef RigidBodySystem<-1>     State;
    typedef Eigen::Vector3d         Vector;
    typedef Eigen::Matrix3d         Matrix;

    RigidBodyRenderer(const std::vector<std::vector<Eigen::Vector3d> >&     vertices,
                      const std::vector<std::vector<std::vector<int> > >&   indices,
                      const boost::shared_ptr<State >&                      state_ptr);

    virtual ~RigidBodyRenderer();

    void Render(Matrix camera_matrix,
                int n_rows, int n_cols,
                std::vector<int> &intersec_tindices,
                std::vector<float> &depth) const;

    // get functions
    std::vector<std::vector<Vector> > vertices() const;
    Vector system_center() const;
    Vector object_center(const size_t& index) const;

    // set function
    virtual void state(const Eigen::VectorXd& state);

protected:
    // triangles
    std::vector<std::vector<Vector> >               vertices_;
    std::vector<std::vector<Vector> >               normals_;
    std::vector<std::vector<std::vector<int> > >    indices_;

    // state
    const boost::shared_ptr<State>  state_;
    std::vector<Matrix>             R_;
    std::vector<Vector>             t_;

    // cached center of mass
    std::vector<Vector>     coms_;
    std::vector<float>      com_weights_;
};

}

#endif
