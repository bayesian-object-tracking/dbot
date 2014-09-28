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

#ifndef POSE_TRACKING_MODELS_OBSERVATION_MODELS_APPROXIMATE_KINECT_PIXEL_OBSERVATION_MODEL_HPP
#define POSE_TRACKING_MODELS_OBSERVATION_MODELS_APPROXIMATE_KINECT_PIXEL_OBSERVATION_MODEL_HPP

#include <cmath>
#include <tuple>

#include <Eigen/Dense>

#include <fast_filtering/distributions/interfaces/evaluation.hpp>
#include <fast_filtering/distributions/exponential_distribution.hpp>
#include <fast_filtering/distributions/uniform_distribution.hpp>
#include <fast_filtering/distributions/truncated_gaussian.hpp>
#include <fast_filtering/utils/helper_functions.hpp>

#include <pose_tracking/models/observation_models/kinect_pixel_observation_model.hpp>
#include <pose_tracking/utils/rigid_body_renderer.hpp>
#include <pose_tracking/utils/hash_mapping.hpp>

#include <boost/unordered_map.hpp>

namespace ff
{

/**
 * \class KinectObservationModel
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
template <typename State>
class ApproximateKinectPixelObservationModel:
        public GaussianMap<double, Eigen::Matrix<double, 1, 1> >,
        public Evaluation<double, double>
{
public:
    typedef Eigen::Matrix<double, 1, 1> Noise;
    typedef double Scalar;
    typedef double Observation;

    typedef boost::shared_ptr<RigidBodyRenderer> ObjectRendererPtr;

    typedef boost::unordered_map<Noise, int> NoiseIndexMap;

    ApproximateKinectPixelObservationModel(
            const ObjectRendererPtr object_renderer,
            const Eigen::Matrix3d& camera_matrix,
            const size_t n_rows,
            const size_t n_cols,
            const Scalar sensor_failure_probability = 0.01,
            const Scalar object_model_sigma = 0.003,
            const Scalar sigma_factor = 0.00142478,
            const Scalar half_life_depth = 1.0,
            const Scalar max_depth = 6.0,
            const Scalar min_depth = 0.0,
            const Scalar approximation_depth = 1.5,
            const size_t depth_count = 10000,
            const size_t occlusion_count = 100)
        : object_renderer_(object_renderer),
          camera_matrix_(camera_matrix),
          n_rows_(n_rows),
          n_cols_(n_cols),
          occluded_observation_model_(sensor_failure_probability,
                                     object_model_sigma,
                                     sigma_factor,
                                     half_life_depth,
                                     max_depth),
          visible_observation_model_(occluded_observation_model_),
          exponential_distribution_(-log(0.5)/half_life_depth, min_depth),
          object_model_noise_(0.0, object_model_sigma),
          sensor_failure_distribution_(min_depth, max_depth),
          sensor_failure_probability_(sensor_failure_probability),
          sigma_factor_(sigma_factor),
          max_depth_(max_depth),
          min_depth_(min_depth),
          approximation_depth_(approximation_depth),
          occlusion_step_(1.0 / double(occlusion_count - 1)),
          depth_step_((max_depth_ - min_depth_) / double(depth_count - 1)),
          sample_cache_(occlusion_count)
    {
        for(size_t occlusion_index = 0; occlusion_index < occlusion_count; occlusion_index++)
        {
            std::vector<double> log_probs(depth_count);
            for(size_t depth_index = 0; depth_index < depth_count; depth_index++)
            {
                Condition(approximation_depth_,
                          hf::Logit(double(occlusion_index) * occlusion_step_));
                log_probs[depth_index] = LogProbability(
                            min_depth_ + double(depth_index) * depth_step_);
            }
            samplers_.push_back(hf::DiscreteDistribution(log_probs));
        }
    }


    virtual ~ApproximateKinectPixelObservationModel() {}

    std::vector<NoiseIndexMap> sample_cache_;

    virtual Observation MapStandardGaussian(const Noise& sample) const
    {
        // original content
//        int depth_index = samplers_[occlusion_index_].MapStandardGaussian(sample(0));
        double depth_index = samplers_[occlusion_index_].MapStandardGaussian(sample(0));

        return rendered_depth_ - approximation_depth_ + min_depth_
                + depth_step_ * depth_index;
    }

    virtual Observation MapStandardGaussian_NONCONST(const Noise& sample)
    {
        NoiseIndexMap& cache = sample_cache_[occlusion_index_];

        if (cache.find(sample) == cache.end())
        {
            std::cout << "CACHE MISSSSSSSSSSSSSSSSSSSSSSSSS" << std::endl;

            cache[sample] = samplers_[occlusion_index_].MapStandardGaussian(sample(0));
        }

        return rendered_depth_ - approximation_depth_ + min_depth_
                + depth_step_ * double(cache[sample]);
    }



    virtual Scalar Probability(const Observation& observation) const
    {
        Scalar probability_given_occluded =
                occluded_observation_model_.Probability(observation);

        Scalar probability_given_visible =
                visible_observation_model_.Probability(observation);

        return probability_given_occluded * occlusion_probability_ +
                probability_given_visible * (1.0 - occlusion_probability_);
    }

    virtual Scalar LogProbability(const Observation& observation) const
    {
        return std::log(Probability(observation));
    }

    virtual void ClearCache(size_t size)
    {
        predictions_.resize(size);
        for (auto& prediction: predictions_)
        {
            prediction.first = false;
        }
    }

    virtual void Condition(const State& state,                           
                           const Scalar& occlusion,
                           size_t state_index,
                           size_t occlusion_index)
    {
        if (!predictions_[state_index].first)
        {
            object_renderer_->state(state);
            object_renderer_->Render(camera_matrix_,
                                     n_rows_,
                                     n_cols_,
                                     predictions_[state_index].second);

            predictions_[state_index].first = true;
        }

        Condition(predictions_[state_index].second[occlusion_index], occlusion);
    }

    virtual void Condition(const Scalar& rendered_depth,
                           const Scalar& occlusion)
    {
        rendered_depth_ = rendered_depth;
        occlusion_probability_ = hf::Sigmoid(occlusion);
        occlusion_index_ = occlusion_probability_ / occlusion_step_;

        occluded_observation_model_.Condition(rendered_depth, true);
        visible_observation_model_.Condition(rendered_depth, false);
    }

private:
    ObjectRendererPtr object_renderer_;

    Eigen::Matrix3d camera_matrix_;
    size_t n_rows_;
    size_t n_cols_;

    std::vector<std::pair<bool, std::vector<float> > > predictions_;

    KinectPixelObservationModel occluded_observation_model_;
    KinectPixelObservationModel visible_observation_model_;

    Scalar rendered_depth_;
    Scalar occlusion_probability_;

    TruncatedGaussian object_model_noise_;
    UniformDistribution uniform_distribution_;
    ExponentialDistribution exponential_distribution_;
    UniformDistribution sensor_failure_distribution_;

    // parameters
    const Scalar sensor_failure_probability_, sigma_factor_,
            max_depth_, min_depth_, approximation_depth_;


    std::vector<hf::DiscreteDistribution> samplers_;

    double occlusion_step_;
    double depth_step_;
    int occlusion_index_;
};

}



#endif
