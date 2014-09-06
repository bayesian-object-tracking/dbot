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


#include <pose_tracking/utils/rigid_body_renderer.hpp>
#include <iostream>

#include <limits>

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace Eigen;

using namespace ff;

RigidBodyRenderer::RigidBodyRenderer(const std::vector<std::vector<Eigen::Vector3d> >&   vertices,
                                     const std::vector<std::vector<std::vector<int> > >& indices,
                                     const boost::shared_ptr<State>&                     state_ptr)
:vertices_(vertices), indices_(indices), state_(state_ptr)
{
    state(*state_ptr);

	float total_weight = 0;
    coms_.resize(vertices.size());
    com_weights_.resize(vertices.size());
    for(size_t part_index = 0; part_index < indices_.size(); part_index++)
	{
        com_weights_[part_index] = vertices[part_index].size();
        total_weight += com_weights_[part_index];

        coms_[part_index] = Vector3d::Zero();
		for(size_t vertex_index = 0; vertex_index < vertices[part_index].size(); vertex_index++)
            coms_[part_index] += vertices[part_index][vertex_index];
        coms_[part_index] /= float(vertices[part_index].size());
	}
    for(size_t i = 0; i < com_weights_.size(); i++)
        com_weights_[i] /= total_weight;

    normals_.clear();
    for(size_t part_index = 0; part_index < indices_.size(); part_index++)
	{
        vector<Vector3d> part_normals(indices_[part_index].size());
        for(int triangle_index = 0; triangle_index < int(part_normals.size()); triangle_index++)
		{
			//compute the three cross products and make sure that they yield the same normal
			vector<Vector3d> temp_normals(3);
			for(int vertex_index = 0; vertex_index < 3; vertex_index++)
                temp_normals[vertex_index] = ((vertices_[part_index][ indices_[part_index][triangle_index][(vertex_index+1)%3] ]-vertices_[part_index][ indices_[part_index][triangle_index][vertex_index] ]).cross(
                        vertices_[part_index][ indices_[part_index][triangle_index][(vertex_index+2)%3] ]-vertices_[part_index][ indices_[part_index][triangle_index][(vertex_index+1)%3] ])).normalized();

			for(int vertex_index = 0; vertex_index < 3; vertex_index++)
				if(!temp_normals[vertex_index].isApprox(temp_normals[(vertex_index+1)%3]))
				{
                    cout << "error, part_normals are not equal, probably the triangle is degenerate."<< endl;
					cout << "normal 1 " << endl << temp_normals[vertex_index] << endl;
					cout << "normal 2 " << endl << temp_normals[(vertex_index+1)%3] << endl;
					exit(-1);
				}
            part_normals[triangle_index] = temp_normals[0];
		}
        normals_.push_back(part_normals);
	}
}

RigidBodyRenderer::~RigidBodyRenderer() {}

// todo: does not handle the case properly when the depth is around zero or negative
void RigidBodyRenderer::Render( Matrix camera_matrix,
                                int n_rows, int n_cols,
                                std::vector<int> &intersec_tindices,
                                std::vector<float> &depth) const
{
	Matrix3d inv_camera_matrix = camera_matrix.inverse();

	// we project all the points into image space --------------------------------------------------------
    vector<vector<Vector3d> > trans_vertices(vertices_.size());
    vector<vector<Vector2d> > image_vertices(vertices_.size());

    for(int part_index = 0; part_index < int(vertices_.size()); part_index++)
	{
        image_vertices[part_index].resize(vertices_[part_index].size());
        trans_vertices[part_index].resize(vertices_[part_index].size());
        for(int point_index = 0; point_index < int(vertices_[part_index].size()); point_index++)
		{
            trans_vertices[part_index][point_index] = R_[part_index] * vertices_[part_index][point_index] + t_[part_index];
            image_vertices[part_index][point_index] =
                    (camera_matrix * trans_vertices[part_index][point_index]/trans_vertices[part_index][point_index](2)).topRows(2);

		}
	}

	// we find the intersections with the triangles and the depths ---------------------------------------------------
    vector<float> depth_image(n_rows*n_cols, numeric_limits<float>::max());
	intersec_tindices.clear(); depth.clear();
    for(int part_index = 0; part_index < int(indices_.size()); part_index++)
	{
        for(int triangle_index = 0; triangle_index < int(indices_[part_index].size()); triangle_index++)
		{
			vector<Vector2d> vertices(3);
			Vector2d center(Vector2d::Zero());

			// find the min and max indices to be checked ------------------------------------------------------------
			int min_row = numeric_limits<int>::max();
			int max_row = -numeric_limits<int>::max();
			int min_col = numeric_limits<int>::max();
			int max_col = -numeric_limits<int>::max();
            bool behind_camera = false;
			for(int i = 0; i < 3; i++)
			{
                vertices[i] = image_vertices[part_index][indices_[part_index][triangle_index][i]];
				center += vertices[i]/3.;
				min_row =  ceil(float(vertices[i](1))) < min_row ?  ceil(float(vertices[i](1))) : min_row;
				max_row = floor(float(vertices[i](1))) > max_row ? floor(float(vertices[i](1))) : max_row;
				min_col =  ceil(float(vertices[i](0))) < min_col ?  ceil(float(vertices[i](0))) : min_col;
				max_col = floor(float(vertices[i](0))) > max_col ? floor(float(vertices[i](0))) : max_col;

                // how should this be handled properly? for now if some vertex in a triangle comes to lie behind camera
                // we just discard that triangle.
                if(trans_vertices[part_index][indices_[part_index][triangle_index][i]](2) < 0.001)
                    behind_camera = true;
			}
            if(behind_camera)
                continue;


			// make sure all of them are inside of image -----------------------------------------------------------------
			min_row = min_row >= 0 ? min_row : 0;
			max_row = max_row < n_rows ? max_row : (n_rows - 1);
			min_col = min_col >= 0 ? min_col : 0;
			max_col = max_col < n_cols ? max_col : (n_cols - 1);


			// check whether triangle is inside image ----------------------------------------------------------------------
			if(max_row < 0 || min_row >= n_rows || max_col < 0 || min_col >= n_cols || max_row < min_row || max_col < min_col)
				continue;

			// we find the line params of the triangle sides ---------------------------------------------------------------
			vector<float> slopes(3);
			vector<bool> boundary_type(3); const bool upper = true; const bool lower = false;

			for(int i = 0; i < 3; i++)
			{
				Vector2d side = vertices[(i+1)%3]-vertices[i];
				slopes[i] = side(1)/side(0);

				// we determine whether the line limits the triangle on top or on the bottom
				if(vertices[i](1) + slopes[i]*(center(0)-vertices[i](0)) > center(1)) boundary_type[i] = upper;
				else boundary_type[i] = lower;
			}

			if(boundary_type[0] == boundary_type[1] && boundary_type[0] == boundary_type[2]) //if triangle is degenerate we continue
				continue;

			for(int col = min_col; col <= max_col; col++)
			{
				float min_row_given_col = -numeric_limits<float>::max();
				float max_row_given_col = numeric_limits<float>::max();

				// the min_row is the max lower boundary at that column, and the max_row is the min upper boundary at that column
				for(int i = 0; i < 3; i++)
				{
					if(boundary_type[i] == lower)
					{
						float lowe_Rboundary = ceil(float(vertices[i](1)  + slopes[i]*(float(col)-vertices[i](0))) );
						min_row_given_col = lowe_Rboundary > min_row_given_col ? lowe_Rboundary : min_row_given_col;
					}
					else
					{
						float upper_boundary = floor(float(vertices[i](1)  + slopes[i]*(float(col)-vertices[i](0)) ));
						max_row_given_col = upper_boundary < max_row_given_col ? upper_boundary : max_row_given_col;
					}
				}


				// we push back the indices of the intersections and the corresponding depths ------------------------------------
                Vector3d normal = R_[part_index]*normals_[part_index][triangle_index];
                float offset = normal.dot(trans_vertices[part_index][indices_[part_index][triangle_index][0]]);
				for(int row = int(min_row_given_col); row <= int(max_row_given_col); row++)
					if(row >= 0 && row < n_rows && col >= 0 && col < n_cols)
					{
						//						intersec_tindices.push_back(row*n_cols + col);
						// we find the intersection between the ray and the triangle --------------------------------------------
						Vector3d line_vector = inv_camera_matrix * Vector3d(col, row, 1); // the depth is the z component
                        float depth = std::fabs(offset/normal.dot(line_vector));
						//if(depth > 0.5)
						  depth_image[row*n_cols + col] =
						    depth < depth_image[row*n_cols + col] ? depth : depth_image[row*n_cols + col];
                    }
			}
		}
	}

	// fill the depths into the depth vector -------------------------------
	intersec_tindices.resize(n_rows*n_cols);
	depth.resize(n_rows*n_cols);
	int count = 0;
	for(int row = 0; row < n_rows; row++)
		for(int col = 0; col < n_cols; col++)
			if(depth_image[row*n_cols + col] != numeric_limits<float>::max())
			{
				intersec_tindices[count] = row*n_cols + col;
				depth[count] = depth_image[row*n_cols + col];
				count++;
			}
	intersec_tindices.resize(count);
	depth.resize(count);

}

// get functions
std::vector<std::vector<RigidBodyRenderer::Vector> > RigidBodyRenderer::vertices() const
{
    vector<vector<Vector3d> > trans_vertices(vertices_.size());

    for(int part_index = 0; part_index < int(vertices_.size()); part_index++)
	{

        trans_vertices[part_index].resize(vertices_[part_index].size());
        for(int point_index = 0; point_index < int(vertices_[part_index].size()); point_index++)
      trans_vertices[part_index][point_index] = R_[part_index] * vertices_[part_index][point_index] + t_[part_index];
	}
	return trans_vertices;
}
RigidBodyRenderer::Vector RigidBodyRenderer::system_center() const
{
	Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for(size_t i = 0; i < coms_.size(); i++)
        com += com_weights_[i] * (R_[i]*coms_[i] + t_[i]);

    com = R_[0].inverse() * (com - t_[0]);

	return com;
}
RigidBodyRenderer::Vector RigidBodyRenderer::object_center(const size_t& index) const
{
    return R_[index]*coms_[index] + t_[index];
}

// set state
void RigidBodyRenderer::state(const Eigen::VectorXd& state)
{
    *state_ = state;
    state_->update();
    R_.resize(state_->body_count());
    t_.resize(state_->body_count());
    for(size_t part_index = 0; part_index < state_->body_count(); part_index++)
    {
        R_[part_index] = state_->rotation_matrix(part_index);
        t_[part_index] = state_->position(part_index);
    }
}


// test the enchilada

//VectorXd initial_rigid_bodies_state = VectorXd::Zero(15);
//initial_rigid_bodies_state.middleRows(3, 4) = Quaterniond::Identity().coeffs();
//
//
//obj_mod::LargeTrimmersModel objec_tmodel_enchilada(vertices, indices);
//objec_tmodel_enchilada.set_state(initial_rigid_bodies_state);
//vector<std::vector<Eigen::Vector3d> > visualize_vertices;
//while(ros::ok())
//{
//	visualize_vertices = objec_tmodel_enchilada.get_vertices();
//	vis::CloudVisualizer cloud_vis;
//	for(size_t i = 0; i < visualize_vertices.size(); i++)
//		cloud_vis.add_cloud(visualize_vertices[i]);
//	cloud_vis.add_point(objec_tmodel_enchilada.get_com().cast<float>());
//	cloud_vis.show(true);
//
//
//
//	// get keyboard input =======================================================================
//	float d_angle = 2*M_PI / 10;
//	float d_trans = 0.01;
//
//	Matrix3d R = objec_tmodel_enchilada.get_R()[0];
//	Vector3d t = objec_tmodel_enchilada.get_t()[0];
//	float alpha = objec_tmodel_enchilada.get_alpha();
//
//	char c;
//	cin >> c;
//
//	if(c == 'q') R = R * AngleAxisd(d_angle, Vector3d(1,0,0));
//	if(c == 'a') R = R * AngleAxisd(-d_angle, Vector3d(1,0,0));
//	if(c == 'w') R = R * AngleAxisd(d_angle, Vector3d(0,1,0));
//	if(c == 's') R = R * AngleAxisd(-d_angle, Vector3d(0,1,0));
//	if(c == 'e') R = R * AngleAxisd(d_angle, Vector3d(0,0,1));
//	if(c == 'd') R = R * AngleAxisd(-d_angle, Vector3d(0,0,1));
//
//	if(c == 'r') t = t + d_trans*Vector3d(1,0,0);
//	if(c == 'f') t = t - d_trans*Vector3d(1,0,0);
//	if(c == 't') t = t + d_trans*Vector3d(0,1,0);
//	if(c == 'g') t = t - d_trans*Vector3d(0,1,0);
//	if(c == 'y') t = t + d_trans*Vector3d(0,0,1);
//	if(c == 'h') t = t - d_trans*Vector3d(0,0,1);
//
//	if(c == 'o') alpha = alpha + d_angle;
//	if(c == 'l') alpha = alpha - d_angle;
//
//	R.col(0).normalize();
//	R.col(1).normalize();
//	R.col(2).normalize();
//
//
//	objec_tmodel_enchilada.set_state(R, t, alpha);
//}

