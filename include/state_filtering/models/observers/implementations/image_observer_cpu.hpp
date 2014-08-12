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

#ifndef MODELS_OBSERVERS_IMPLEMENTATIONS_IMAGE_observer_CPU_HPP
#define MODELS_OBSERVERS_IMPLEMENTATIONS_IMAGE_observer_CPU_HPP

#include <vector>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <state_filtering/models/observers/implementations/kinect_observer.hpp>
#include <state_filtering/utils/rigid_body_renderer.hpp>
#include <state_filtering/models/processes/implementations/occlusion_process.hpp>

#include <state_filtering/models/observers/features/rao_blackwell_observer.hpp>

#include <state_filtering/states/floating_body_system.hpp>


namespace distributions
{

struct ImageObserverCPUTypes
{
    typedef double                              ScalarType;
    typedef RigidBodySystem<-1>                 StateType;
    typedef Eigen::Matrix<ScalarType, -1, -1>   ObservationType;
    typedef size_t                              IndexType;

    typedef RaoBlackwellObserver<ScalarType, StateType, ObservationType, IndexType>
                                            RaoBlackwellObserverType;
};




class ImageObserverCPU: public ImageObserverCPUTypes::RaoBlackwellObserverType
{
public:
    typedef typename ImageObserverCPUTypes::ScalarType      ScalarType;
    typedef typename ImageObserverCPUTypes::StateType       StateType;
    typedef typename ImageObserverCPUTypes::ObservationType ObservationType;

    typedef boost::shared_ptr<obj_mod::RigidBodyRenderer> ObjectRenderer;
    typedef boost::shared_ptr<distributions::KinectObserver> PixelObservationModel;
    typedef boost::shared_ptr<proc_mod::OcclusionProcess> OcclusionProcessModel;

    // TODO: DO WE NEED ALL OF THIS IN THE CONSTRUCTOR??
    ImageObserverCPU(
			const Eigen::Matrix3d& camera_matrix,
			const size_t& n_rows,
			const size_t& n_cols,
			const size_t& max_sample_count,
            const ObjectRenderer object_renderer,
			const PixelObservationModel observation_model,
			const OcclusionProcessModel occlusion_process_model,
			const float& initial_visibility_prob);

    ~ImageObserverCPU();

    std::vector<ScalarType> Loglikes(const std::vector<const StateType*>& states,
                                     std::vector<IndexType>&        indices,
                                     const bool&                    update = false);

    void Observation(const ObservationType& image, const ScalarType& delta_time);

    virtual void Reset();

    //TODO: TYPES
    const std::vector<float> Occlusions(size_t index) const;

private:
    // TODO: GET RID OF THIS
    void Occlusions(const float& visibility_prob = -1);
    void Observation(const std::vector<float>& observations, const ScalarType& delta_time);

    // TODO: WE PROBABLY DONT NEED ALL OF THIS
    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;
    const boost::shared_ptr<RigidBodySystem<-1> > rigid_body_system_;

    // models
    ObjectRenderer object_model_;
	PixelObservationModel observation_model_;
	OcclusionProcessModel occlusion_process_model_;

    // occlusion parameters
	std::vector<std::vector<float> > visibility_probs_;
	std::vector<std::vector<double> > visibility_update_times_;

    // observed data
	std::vector<float> observations_;
	double observation_time_;
};

}
#endif
