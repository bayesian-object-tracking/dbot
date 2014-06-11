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


#ifndef IMAGE_OBSERVATION_MODEL_
#define IMAGE_OBSERVATION_MODEL_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <state_filtering/observation_models/cpu_image_observation_model/kinect_measurement_model.hpp>
#include <state_filtering/process_model/occlusion_process_model.hpp>
#include <state_filtering/system_states/rigid_body_system.hpp>
#include <state_filtering/tools/rigid_body_renderer.hpp>

namespace obs_mod
{
class ImageObservationModel
{
public:
    typedef std::vector<float> MeasurementType;

    typedef Eigen::Matrix<double, -1, -1> Image;




	ImageObservationModel(
			const Eigen::Matrix3d& camera_matrix,
			const size_t& n_rows,
			const size_t& n_cols,
			const float& initial_visibility_prob,
            const size_t& max_sample_count,
            const boost::shared_ptr<RigidBodySystem<-1> >& rigid_body_system
            ):
			camera_matrix_(camera_matrix),
			n_rows_(n_rows),
			n_cols_(n_cols),
			initial_visibility_prob_(initial_visibility_prob),
            max_sample_count_(max_sample_count),
            rigid_body_system_(rigid_body_system){}

	virtual ~ImageObservationModel() {}

	virtual std::vector<float> Evaluate(
			const std::vector<Eigen::VectorXd >& states,
			std::vector<size_t>& occlusion_indices,
			const bool& update_occlusions = false) = 0;


//    virtual std::vector<float> Evaluate_test(
//            const std::vector<Eigen::VectorXd >& states,
//            std::vector<size_t>& occlusion_indices,
//            const bool& update_occlusions = false,
//            std::vector<std::vector<int> > intersect_indices = 0,
//            std::vector<std::vector<float> > predictions = 0) = 0;

	// set and get functions =============================================================================================================================================================================================================================================================================================
    virtual void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                  std::vector<std::vector<float> > &depth) = 0;
    virtual const std::vector<float> get_occlusions(size_t index) const = 0 ;
	virtual void set_occlusions(const float& visibility_prob = -1) = 0;

    virtual void measurement(const Image& image, const double& time)
    {
        vector<float> std_measurement(image.size());

        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                std_measurement[row*image.cols() + col] = image(row, col);

        measurement(std_measurement, time);
    }



    virtual void measurement(const MeasurementType& observations, const double& time) = 0;

protected:
	// constant parameters ===========================================================================================================================================================================================================================================================================================================================
	const Eigen::Matrix3d camera_matrix_;
	const size_t n_rows_;
	const size_t n_cols_;
	const float initial_visibility_prob_;
	const size_t max_sample_count_;
    const boost::shared_ptr<RigidBodySystem<-1> > rigid_body_system_;
};

}
#endif
