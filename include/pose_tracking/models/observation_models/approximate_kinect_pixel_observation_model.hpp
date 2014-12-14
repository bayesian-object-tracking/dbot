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

#include <fl/distribution/interface/evaluation.hpp>
#include <fl/util/traits.hpp>

#include <ff/distributions/exponential_distribution.hpp>
#include <ff/distributions/uniform_distribution.hpp>
#include <ff/distributions/truncated_gaussian.hpp>
#include <ff/utils/helper_functions.hpp>

#include <pose_tracking/models/observation_models/kinect_pixel_observation_model.hpp>
#include <pose_tracking/utils/rigid_body_renderer.hpp>

#include <boost/unordered_map.hpp>

namespace fl
{

// Base template
template <typename State_a_,
          typename State_b_ = double,
          internal::SpaceType space_type = internal::Scalar>
class ApproximateKinectPixelObservationModel { };

/**
 * Scalar implementation of the approximate continuous observation model
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
template <typename State_a_, typename State_b_>
class ApproximateKinectPixelObservationModel<State_a_,
                                             State_b_,
                                             internal::Scalar>:
        public GaussianMap<double, double>,
        public Evaluation<double, double>
{
public:
    typedef State_a_ State_a;
    typedef State_b_ State_b;
    typedef double Scalar;
    typedef double Noise;
    typedef double Observation;

    typedef boost::shared_ptr<RigidBodyRenderer> ObjectRendererPtr;

    ApproximateKinectPixelObservationModel(
            const ObjectRendererPtr object_renderer,
            const Eigen::Matrix3d& camera_matrix,
            const size_t n_rows,
            const size_t n_cols,
            const Scalar tail_weight = 0.01,
            const Scalar model_sigma = 0.003,
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
          occluded_observation_model_(tail_weight,
                                     model_sigma,
                                     sigma_factor,
                                     half_life_depth,
                                     max_depth),
          visible_observation_model_(occluded_observation_model_),
          exponential_distribution_(-log(0.5)/half_life_depth, min_depth, max_depth),
          max_depth_(max_depth),
          min_depth_(min_depth),
          approximation_depth_(approximation_depth),
          occlusion_step_(1.0 / double(occlusion_count - 1)),
          depth_step_((max_depth - min_depth) / double(depth_count - 1))
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

    virtual Observation MapStandardGaussian(const Noise& sample) const
    {
        if(std::isinf(rendered_depth_))
        {
            return exponential_distribution_.MapStandardGaussian(sample);
        }

        int depth_index = samplers_[occlusion_index_].MapStandardGaussian(sample);

        return rendered_depth_ - approximation_depth_ + min_depth_
                + depth_step_ * double(depth_index);
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

    virtual void ClearCache()
    {
        for (auto& prediction: predictions_)
        {
            prediction.first = false;
        }
    }

    virtual void Condition(const State_a& state,
                           const State_b& occlusion,
                           size_t state_index,
                           size_t pixel_index)
    {
        if (state_index + 1 > predictions_.size())
        {
            predictions_.resize(state_index + 1, {false, std::vector<float>()});
        }

        if (!predictions_[state_index].first)
        {
            object_renderer_->state(state);
            object_renderer_->Render(camera_matrix_,
                                     n_rows_,
                                     n_cols_,
                                     predictions_[state_index].second);

            predictions_[state_index].first = true;
        }

        Condition(predictions_[state_index].second[pixel_index], occlusion);

        // if this was the last pixel, reset cache
        if (pixel_index + 1 == n_rows_*n_cols_)
        {
            ClearCache();
        }
    }

    virtual void Condition(const Scalar& rendered_depth,
                           const State_b& occlusion)
    {
        assert(!std::isnan(occlusion));


        rendered_depth_ = rendered_depth;
        occlusion_probability_ = hf::Sigmoid(occlusion);

        occlusion_index_ = occlusion_probability_ / occlusion_step_;

        occluded_observation_model_.Condition(rendered_depth, true);
        visible_observation_model_.Condition(rendered_depth, false);
    }

    virtual size_t Dimension() const
    {
        return 1;
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

    UniformDistribution uniform_distribution_;
    ExponentialDistribution exponential_distribution_;

    // parameters
    const Scalar max_depth_, min_depth_, approximation_depth_;

    std::vector<hf::DiscreteDistribution> samplers_;

    double occlusion_step_;
    double depth_step_;
    int occlusion_index_;
};



/**
 * \internal
 */
template <typename State_a_, typename State_b_>
struct Traits <ApproximateKinectPixelObservationModel<State_a_,
                                                      State_b_,
                                                      internal::Vectorial> >
{
    enum
    {
        Dimension = 1
    };

    typedef State_a_ State_a;
    typedef State_b_ State_b;
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 1, 1> Noise;
    typedef Eigen::Matrix<Scalar, 1, 1> Observation;

    typedef GaussianMap<Observation, Noise> GaussianMapBase;
    typedef Evaluation<Observation, Scalar> EvaluationBase;
};

/**
 * Vectorial specialization of the scalar model
 */
template <typename State_a_, typename State_b_>
class ApproximateKinectPixelObservationModel<State_a_, State_b_, internal::Vectorial>:
        public Traits<ApproximateKinectPixelObservationModel<State_a_, State_b_, internal::Vectorial> >::GaussianMapBase,
        public Traits<ApproximateKinectPixelObservationModel<State_a_, State_b_, internal::Vectorial> >::EvaluationBase
{
public:
    typedef ApproximateKinectPixelObservationModel<State_a_, State_b_, internal::Vectorial> This;

    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::State_a State_a;
    typedef typename Traits<This>::State_b State_b;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Observation Observation;

    typedef boost::shared_ptr<RigidBodyRenderer> ObjectRendererPtr;

    ApproximateKinectPixelObservationModel(
            const ObjectRendererPtr object_renderer,
            const Eigen::Matrix3d& camera_matrix,
            const size_t n_rows,
            const size_t n_cols,
            const Scalar tail_weight = 0.01,
            const Scalar model_sigma = 0.003,
            const Scalar sigma_factor = 0.00142478,
            const Scalar half_life_depth = 1.0,
            const Scalar max_depth = 6.0,
            const Scalar min_depth = 0.0,
            const Scalar approximation_depth = 1.5,
            const size_t depth_count = 10000,
            const size_t occlusion_count = 100):
        scalar_model_(object_renderer, camera_matrix,
                        n_rows,
                        n_cols,
                        tail_weight,
                        model_sigma,
                        sigma_factor,
                        half_life_depth,
                        max_depth,
                        min_depth,
                        approximation_depth,
                        depth_count,
                        occlusion_count)
    { }

    virtual ~ApproximateKinectPixelObservationModel() { }

    virtual Observation MapStandardGaussian(const Noise& sample) const
    {
        Observation observation;
        observation(0) = scalar_model_.MapStandardGaussian(sample(0));
        return observation;
    }

    virtual Scalar Probability(const Observation& observation) const
    {
        return scalar_model_.Probability(observation(0));
    }

    virtual Scalar LogProbability(const Observation& observation) const
    {
        return scalar_model_.LogProbability(observation(0));
    }

    virtual void Condition(const State_a& state,
                           const State_b& occlusion,
                           size_t state_index,
                           size_t pixel_index)
    {
        scalar_model_.Condition(state, occlusion(0), state_index, pixel_index);
    }

    virtual void Condition(const Scalar& rendered_depth,
                           const State_b& occlusion)
    {
        scalar_model_.Condition(rendered_depth, occlusion(0));
    }

    virtual size_t Dimension() const
    {
        return Traits<This>::Dimension;
    }

private:
    ApproximateKinectPixelObservationModel<
    State_a, double, internal::Scalar> scalar_model_;
};

}

#endif
