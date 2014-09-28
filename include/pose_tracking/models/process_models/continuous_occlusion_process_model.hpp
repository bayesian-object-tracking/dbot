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

#ifndef POSE_TRACKING_MODELS_PROCESS_MODELS_CONTINUOUS_OCCLUSION_PROCESS_MODEL_HPP
#define POSE_TRACKING_MODELS_PROCESS_MODELS_CONTINUOUS_OCCLUSION_PROCESS_MODEL_HPP

#include <fast_filtering/utils/helper_functions.hpp>

#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>
#include <fast_filtering/distributions/interfaces/gaussian_map.hpp>
#include <fast_filtering/distributions/truncated_gaussian.hpp>

#include <pose_tracking/models/process_models/occlusion_process_model.hpp>

namespace ff
{

// Forward declarations
class ContinuousOcclusionProcessModel;

namespace internal
{
template <>
struct Traits<ContinuousOcclusionProcessModel>
{
    typedef double                      Scalar;
    typedef Eigen::Matrix<Scalar, 1, 1> State;
    typedef Eigen::Matrix<Scalar, 1, 1> Noise;

    typedef StationaryProcessModel<State> ProcessModelBase;
    typedef GaussianMap<State, Noise>     GaussianMapBase;
};
}

class ContinuousOcclusionProcessModel:
        public internal::Traits<ContinuousOcclusionProcessModel>::ProcessModelBase,
        public internal::Traits<ContinuousOcclusionProcessModel>::GaussianMapBase
{
public:
    typedef internal::Traits<ContinuousOcclusionProcessModel> Traits;
    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::State  State;
    typedef typename Traits::Noise  Noise;

	// the prob of source being object given source was object one sec ago,
	// and prob of source being object given one sec ago source was not object
    ContinuousOcclusionProcessModel(Scalar p_occluded_visible,
                                    Scalar p_occluded_occluded,
                                    Scalar sigma):
        mean_(p_occluded_visible, p_occluded_occluded),
        sigma_(sigma) { }

    virtual ~ContinuousOcclusionProcessModel() {}

    virtual void Condition(const double& delta_time,
                           const State& occlusion)
    {
        double initial_occlusion_probability = hf::Sigmoid(occlusion(0, 0));

        mean_.Condition(delta_time, initial_occlusion_probability);
        double mean = mean_.MapStandardGaussian();

        occlusion_probability_.SetParameters(mean,
                                             sigma_ * std::sqrt(delta_time),
                                             0.0, 1.0);
    }

    virtual State MapStandardGaussian(const Noise& sample) const
    {
        State l;
        l(0, 0) = hf::Logit(
                    occlusion_probability_.MapStandardGaussian(sample(0,0)));
        return l;
    }

    virtual size_t Dimension()
    {
        return 1;
    }

private:
    OcclusionProcessModel mean_;
    TruncatedGaussian occlusion_probability_;
    double sigma_;
};

}

#endif
