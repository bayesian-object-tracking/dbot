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

#include <state_filtering/utils/traits.hpp>
#include <state_filtering/utils/rigid_body_renderer.hpp>

#include <state_filtering/models/observers/kinect_observer.hpp>
#include <state_filtering/models/processes/occlusion_process.hpp>
#include <state_filtering/models/observers/features/rao_blackwell_observer.hpp>
#include <state_filtering/states/floating_body_system.hpp>


namespace sf
{

// Forward declarations
template <typename Scalar, typename State> class ImageObserverCPU;

namespace internal
{
/**
 * ImageObserverCPU distribution traits specialization
 * \internal
 */
template <typename Scalar, typename State>
struct Traits<ImageObserverCPU<Scalar, State> >
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef RaoBlackwellObserver<State, Observation> RaoBlackwellObserverBase;

    typedef boost::shared_ptr<obj_mod::RigidBodyRenderer> ObjectRendererPtr;
    typedef boost::shared_ptr<sf::KinectObserver> PixelObservationModelPtr;
    typedef boost::shared_ptr<sf::OcclusionProcess> OcclusionProcessModelPtr;
};
}

/**
 * \class ImageObserverCPU
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
template <typename Scalar, typename State>
class ImageObserverCPU:
        public internal::Traits<ImageObserverCPU<Scalar, State> >::RaoBlackwellObserverBase
{
public:
    typedef internal::Traits<ImageObserverCPU<Scalar, State> > Traits;

    typedef typename Traits::Observation              Observation;
    typedef typename Traits::ObjectRendererPtr        ObjectRendererPtr;
    typedef typename Traits::PixelObservationModelPtr PixelObservationModelPtr;
    typedef typename Traits::OcclusionProcessModelPtr OcclusionProcessModelPtr;

    // TODO: DO WE NEED ALL OF THIS IN THE CONSTRUCTOR??
    ImageObserverCPU(
			const Eigen::Matrix3d& camera_matrix,
			const size_t& n_rows,
			const size_t& n_cols,
			const size_t& max_sample_count,
            const ObjectRendererPtr object_renderer,
            const PixelObservationModelPtr observation_model,
            const OcclusionProcessModelPtr occlusion_process_model,
            const float& initial_visibility_prob):
        camera_matrix_(camera_matrix),
        n_rows_(n_rows),
        n_cols_(n_cols),
        initial_visibility_prob_(initial_visibility_prob),
        max_sample_count_(max_sample_count),
        object_model_(object_renderer),
        observation_model_(observation_model),
        occlusion_process_model_(occlusion_process_model),
        observation_time_(0)
    {
        Reset();
    }

    ~ImageObserverCPU() { }

    std::vector<Scalar> Loglikes_(const std::vector<const State*>& states,
                                 std::vector<size_t>& indices,
                                 const bool& update = false)
    {
        std::vector<std::vector<float> > new_visibility_probs(states.size());
        std::vector<std::vector<double> > new_visibility_update_times(states.size());
        std::vector<Scalar> loglikes(states.size(),0);
        for(size_t state_index = 0; state_index < size_t(states.size()); state_index++)
        {
            if(update)
            {
                new_visibility_probs[state_index] = visibility_probs_[indices[state_index]];
                new_visibility_update_times[state_index] = visibility_update_times_[indices[state_index]];
            }
            // we predict observations_ ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            std::vector<int> intersect_indices;
            std::vector<float> predictions;
            //TODO: DOES THIS MAKE SENSE? THE OBJECT MODEL SHOULD KNOW ABOUT THE STATE...
            object_model_->state(*states[state_index]);
            object_model_->Render(camera_matrix_, n_rows_, n_cols_, intersect_indices, predictions);

            // we loop through all the pixels which intersect the object model ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            for(size_t i = 0; i < size_t(predictions.size()); i++)
            {
                if(isnan(observations_[intersect_indices[i]]))
                    loglikes[state_index] += log(1.);
                else
                {
                    float visibility_prob;
                    // we predict the visiblity probability and set the time of the last update time to current
                    visibility_prob = 1. -
                            occlusion_process_model_->Propagate(
                                1. - visibility_probs_[indices[state_index]][intersect_indices[i]],
                                observation_time_ - visibility_update_times_[indices[state_index]][intersect_indices[i]]);

                    observation_model_->Condition(predictions[i], false);
                    float p_obsIpred_vis =
                            observation_model_->Probability(observations_[intersect_indices[i]]) * visibility_prob;

                    observation_model_->Condition(predictions[i], true);
                    float p_obsIpred_occl =
                            observation_model_->Probability(observations_[intersect_indices[i]]) * (1-visibility_prob);

                    observation_model_->Condition(std::numeric_limits<float>::infinity(), true);
                    float p_obsIinf = observation_model_->Probability(observations_[intersect_indices[i]]);

                    loglikes[state_index] += log((p_obsIpred_vis + p_obsIpred_occl)/p_obsIinf);

                    // we update the visibiliy (occlusion) with the observations
                    if(update)
                    {
                        new_visibility_probs[state_index][intersect_indices[i]] = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
                        new_visibility_update_times[state_index][intersect_indices[i]] = observation_time_;
                    }
                }
            }
        }
        if(update)
        {
            visibility_probs_ = new_visibility_probs;
            visibility_update_times_ = new_visibility_update_times;
            for(size_t state_index = 0; state_index < indices.size(); state_index++)
                indices[state_index] = state_index;
        }
        return loglikes;
    }

    void SetObservation(const Observation& image, const Scalar& delta_time)
    {
        std::vector<float> std_measurement(image.size());

        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                std_measurement[row*image.cols() + col] = image(row, col);

        SetObservation(std_measurement, delta_time);
    }

    virtual void Reset()
    {
        Occlusions();
        observation_time_ = 0;
    }

    //TODO: TYPES
    const std::vector<float> Occlusions(size_t index) const
    {
        return visibility_probs_[index];
    }

private:
    // TODO: GET RID OF THIS
    void Occlusions(const float& visibility_prob = -1)
    {
        float p = visibility_prob == -1 ? initial_visibility_prob_ : visibility_prob;
        visibility_probs_ = std::vector<std::vector<float> >(1, std::vector<float>(n_rows_*n_cols_, p));
        visibility_update_times_ = std::vector<std::vector<double> >(1, std::vector<double>(n_rows_*n_cols_, 0));
    }

    void SetObservation(const std::vector<float>& observations, const Scalar& delta_time)
    {
        observations_ = observations;
        observation_time_ += delta_time;

    }




    // TODO: WE PROBABLY DONT NEED ALL OF THIS
    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;
    const boost::shared_ptr<RigidBodySystem<-1> > rigid_body_system_;

    // models
    ObjectRendererPtr object_model_;
    PixelObservationModelPtr observation_model_;
    OcclusionProcessModelPtr occlusion_process_model_;

    // occlusion parameters
	std::vector<std::vector<float> > visibility_probs_;
	std::vector<std::vector<double> > visibility_update_times_;

    // observed data
	std::vector<float> observations_;
	double observation_time_;
};

}
#endif
