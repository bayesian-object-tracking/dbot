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


#ifndef POSE_TRACKING_UTILS_RIGID_BODY_RENDERER_HPP
#define POSE_TRACKING_UTILS_RIGID_BODY_RENDERER_HPP

#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>
#include <pose_tracking/states/rigid_bodies_state.hpp>


namespace fl
{

class RigidBodyRenderer
{
public:
    typedef boost::shared_ptr<RigidBodyRenderer> Ptr;
    typedef fl::RigidBodiesState<Eigen::Dynamic> State;
    typedef Eigen::Vector3d Vector;
    typedef Eigen::Matrix3d Matrix;

    RigidBodyRenderer(const std::vector<std::vector<Eigen::Vector3d> >& vertices,
                      const std::vector<std::vector<std::vector<int> > >& indices,
                      const boost::shared_ptr<State>& state_ptr);

    RigidBodyRenderer(const std::vector<std::vector<Eigen::Vector3d> >& vertices,
                      const std::vector<std::vector<std::vector<int> > >& indices,
                      const boost::shared_ptr<State>& state_ptr,
                      Matrix camera_matrix,
                      int n_rows,
                      int n_cols);        

    virtual ~RigidBodyRenderer();

    void Render(Matrix camera_matrix,
                int n_rows,
                int n_cols,
                std::vector<int> &intersect_indices,
                std::vector<float> &depth) const;

    void Render(Matrix camera_matrix,
                int n_rows,
                int n_cols,
                std::vector<float>& depth_image) const;

    void Render(std::vector<float>& depth_image) const;

    // get functions
    std::vector<std::vector<Vector> > vertices() const;
    Vector system_center() const;
    Vector object_center(const size_t& index) const;

    // set function
    virtual void state(const Eigen::VectorXd& state);

    void parameters(Matrix camera_matrix,
                    int n_rows,
                    int n_cols);

private:
    /**
     * Because c++0x on gcc.4.6 do not implement delegating constructors
     *
     *
     *
     *
     *
     *                            (-_-)
     *
     *
     *
     *
     */
    void init(const std::vector<std::vector<Eigen::Vector3d> >& vertices,
              const std::vector<std::vector<std::vector<int> > >& indices,
              const boost::shared_ptr<State>& state_ptr);

protected:
    // triangles
    std::vector<std::vector<Vector> >               vertices_;
    std::vector<std::vector<Vector> >               normals_;
    std::vector<std::vector<std::vector<int> > >    indices_;

    // state
    boost::shared_ptr<State> state_;
    std::vector<Matrix>             R_;
    std::vector<Vector>             t_;

    // cached center of mass
    std::vector<Vector> coms_;
    std::vector<float>  com_weights_;

    Matrix camera_matrix_;
    int n_rows_;
    int n_cols_;
};

}

#endif
