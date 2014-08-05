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

#ifndef CPU_IMAGE_OBSERVATION_MODEL_
#define CPU_IMAGE_OBSERVATION_MODEL_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <state_filtering/models/measurement/cpu_image_observation_model/kinect_measurement_model.hpp>
#include <state_filtering/utils/rigid_body_renderer.hpp>
#include <state_filtering/models/process/implementations/occlusion_process.hpp>

#include <state_filtering/models/measurement/image_observation_model.hpp>

#include <state_filtering/states/floating_body_system.hpp>


namespace distributions
{

struct CPUImageObservationModelTypes
{
    typedef double                              ScalarType;
    typedef FloatingBodySystem<-1>              VectorType;
    typedef Eigen::Matrix<ScalarType, -1, -1>   MeasurementType;

    typedef RaoBlackwellMeasurementModel<ScalarType, VectorType, MeasurementType> RaoBlackwellMeasurementModelType;
};




class CPUImageObservationModel: public CPUImageObservationModelTypes::RaoBlackwellMeasurementModelType
{
public:
    typedef typename CPUImageObservationModelTypes::ScalarType      ScalarType;
    typedef typename CPUImageObservationModelTypes::VectorType      VectorType;
    typedef typename CPUImageObservationModelTypes::MeasurementType MeasurementType;








    typedef boost::shared_ptr<obj_mod::RigidBodyRenderer> ObjectModel;
    typedef boost::shared_ptr<distributions::KinectMeasurementModel> PixelObservationModel;
	typedef boost::shared_ptr<proc_mod::OcclusionProcessModel> OcclusionProcessModel;

	CPUImageObservationModel(
			const Eigen::Matrix3d& camera_matrix,
			const size_t& n_rows,
			const size_t& n_cols,
			const size_t& max_sample_count,
            const boost::shared_ptr<RigidBodySystem<-1> >& rigid_body_system,
			const ObjectModel object_model,
			const PixelObservationModel observation_model,
			const OcclusionProcessModel occlusion_process_model,
			const float& initial_visibility_prob);

	~CPUImageObservationModel();

	std::vector<float> Evaluate(
			const std::vector<Eigen::VectorXd >& states,
			std::vector<size_t>& occlusion_indices,
			const bool& update_occlusions = false);

//    std::vector<float> Evaluate_test(
//            const std::vector<Eigen::VectorXd >& states,
//            std::vector<size_t>& occlusion_indices,
//            const bool& update_occlusions = false,
//            std::vector<std::vector<int> > intersect_indices = 0,
//            std::vector<std::vector<float> > predictions = 0);

	// set and get functions =============================================================================================================================================================================================================================================================================================
    void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                          std::vector<std::vector<float> > &depth);
    const std::vector<float> get_occlusions(size_t index) const;
	void set_occlusions(const float& visibility_prob = -1);
    void measurement(const std::vector<float>& observations, const double& delta_time);

    size_t state_size();

    size_t measurement_rows();
    size_t measurement_cols();


    virtual void Reset();



private:
    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;
    const boost::shared_ptr<RigidBodySystem<-1> > rigid_body_system_;




	// models ============================================================================================================================================================================================================================================================
	ObjectModel object_model_;
	PixelObservationModel observation_model_;
	OcclusionProcessModel occlusion_process_model_;

	// occlusion parameters ===========================================================================================================================================================================================================================================================================================================================
	std::vector<std::vector<float> > visibility_probs_;
	std::vector<std::vector<double> > visibility_update_times_;

	// observed data ==================================================================================================================================================================
	std::vector<float> observations_;
	double observation_time_;

    // internal copy of states
    std::vector<Eigen::VectorXd> states_;
};

}
#endif
