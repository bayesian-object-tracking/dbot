/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
 * Probabilistic Object Tracking using a Range Camera
 * IEEE Intl Conf on Intelligent Robots and Systems, 2013
 * http://arxiv.org/abs/1505.00241
 *
 */

/**
 * \file kinect_image_model.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <osr/pose_vector.hpp>
#include <fl/util/assertions.hpp>
#include <dbot/common/traits.hpp>
#include <osr/free_floating_rigid_bodies_state.hpp>
#include <dbot/model/rao_blackwell_observation_model.hpp>

#include <dbot/common/rigid_body_renderer.hpp>
#include <dbot/model/kinect_pixel_model.hpp>
#include <dbot/model/occlusion_model.hpp>

namespace dbot
{
/// \todo get rid of traits
// Forward declarations
template <typename Scalar, typename State, int OBJECTS>
class KinectImageModel;

namespace internal
{
/**
 * ImageObservationModelCPU distribution traits specialization
 * \internal
 */
template <typename Scalar, typename State, int OBJECTS>
struct Traits<KinectImageModel<Scalar, State, OBJECTS>>
{
    typedef RbObservationModel<State> ObservationModelBase;
    typedef typename ObservationModelBase::Observation Observation;

    typedef std::shared_ptr<dbot::RigidBodyRenderer> ObjectRendererPtr;
    typedef std::shared_ptr<dbot::KinectPixelModel>
        PixelObservationModelPtr;
    typedef std::shared_ptr<dbot::OcclusionModel>
        OcclusionModelPtr;
};
}

/**
 * \class ImageObservationModelCPU
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
template <typename Scalar, typename State, int OBJECTS = -1>
class KinectImageModel
    : public internal::Traits<
          KinectImageModel<Scalar, State>>::ObservationModelBase
{
public:
    typedef internal::Traits<KinectImageModel<Scalar, State>>
        Traits;

    typedef typename Traits::ObservationModelBase Base;
    typedef typename Traits::Observation Observation;
    typedef typename Traits::ObjectRendererPtr ObjectRendererPtr;
    typedef typename Traits::PixelObservationModelPtr PixelObservationModelPtr;
    typedef typename Traits::OcclusionModelPtr OcclusionModelPtr;

    typedef typename Base::StateArray StateArray;
    typedef typename Base::RealArray RealArray;
    typedef typename Base::IntArray IntArray;

    typedef typename Eigen::Transform<fl::Real, 3, Eigen::Affine> Affine;

    // TODO: DO WE NEED ALL OF THIS IN THE CONSTRUCTOR??
    KinectImageModel(
        const Eigen::Matrix3d& camera_matrix,
        const size_t& n_rows,
        const size_t& n_cols,
        const ObjectRendererPtr object_renderer,
        const PixelObservationModelPtr observation_model,
        const OcclusionModelPtr occlusion_process_model,
        const float& initial_occlusion,
        const double& delta_time)
        : camera_matrix_(camera_matrix),
          n_rows_(n_rows),
          n_cols_(n_cols),
          initial_occlusion_(initial_occlusion),
          object_model_(object_renderer),
          observation_model_(observation_model),
          occlusion_process_model_(occlusion_process_model),
          observation_time_(0),
          Base(delta_time)
    {
        static_assert_base(State, osr::RigidBodiesState<OBJECTS>);

        this->default_poses_.recount(object_model_->vertices().size());
        this->default_poses_.setZero();

        reset();
    }

    virtual ~KinectImageModel() noexcept {}
    RealArray loglikes(const StateArray& deltas,
                       IntArray& indices,
                       const bool& update = false)
    {
        std::vector<std::vector<float>> new_occlusions(deltas.size());
        std::vector<std::vector<double>> new_occlusion_times(deltas.size());

        RealArray log_likes = RealArray::Zero(deltas.size());
        for (size_t i_state = 0; i_state < size_t(deltas.size()); i_state++)
        {
            if (update)
            {
                new_occlusions[i_state] = occlusions_[indices[i_state]];
                new_occlusion_times[i_state] =
                    occlusion_times_[indices[i_state]];
            }

            // render the object model -----------------------------------------
            int body_count = deltas[i_state].count();
            std::vector<Affine> poses(body_count);
            for (size_t i_obj = 0; i_obj < body_count; i_obj++)
            {
                auto pose_0 = this->default_poses_.component(i_obj);
                auto delta = deltas[i_state].component(i_obj);

                osr::PoseVector pose;
                pose.orientation() = delta.orientation() * pose_0.orientation();
                pose.position() = delta.position() + pose_0.position();

                poses[i_obj] = pose.affine();
            }
            object_model_->set_poses(poses);
            std::vector<int> intersect_indices;
            std::vector<float> predictions;
            object_model_->Render(camera_matrix_,
                                  n_rows_,
                                  n_cols_,
                                  intersect_indices,
                                  predictions);

            // compute likelihoods ---------------------------------------------
            for (size_t i = 0; i < size_t(predictions.size()); i++)
            {
                if (isnan(observations_[intersect_indices[i]]))
                {
                    log_likes[i_state] += log(1.);
                }
                else
                {
                    double delta_time = observation_time_ -
                                        occlusion_times_[indices[i_state]]
                                                        [intersect_indices[i]];

                    occlusion_process_model_->Condition(
                        delta_time,
                        occlusions_[indices[i_state]][intersect_indices[i]]);

                    float occlusion =
                        occlusion_process_model_->MapStandardGaussian();

                    observation_model_->Condition(predictions[i], false);
                    float p_obsIpred_vis =
                        observation_model_->Probability(
                            observations_[intersect_indices[i]]) *
                        (1.0 - occlusion);

                    observation_model_->Condition(predictions[i], true);
                    float p_obsIpred_occl =
                        observation_model_->Probability(
                            observations_[intersect_indices[i]]) *
                        occlusion;

                    observation_model_->Condition(
                        std::numeric_limits<float>::infinity(), true);
                    float p_obsIinf = observation_model_->Probability(
                        observations_[intersect_indices[i]]);

                    log_likes[i_state] +=
                        log((p_obsIpred_vis + p_obsIpred_occl) / p_obsIinf);

                    // we update the occlusion with the observations
                    if (update)
                    {
                        new_occlusions[i_state][intersect_indices[i]] =
                            p_obsIpred_occl /
                            (p_obsIpred_vis + p_obsIpred_occl);
                        new_occlusion_times[i_state][intersect_indices[i]] =
                            observation_time_;
                    }
                }
            }
        }
        if (update)
        {
            occlusions_ = new_occlusions;
            occlusion_times_ = new_occlusion_times;
            for (size_t i_state = 0; i_state < indices.size(); i_state++)
                indices[i_state] = i_state;
        }
        return log_likes;
    }

    void set_observation(const Observation& image)
    {
        assert(image.rows() == image.size());
        assert(image.cols() == 1);

        std::vector<float> std_measurement(image.size());        

        for (int i = 0; i < image.size(); ++i)
        {
            std_measurement[i] = image(i, 0);
        }

        set_observation(std_measurement, this->delta_time_);
    }

    virtual void reset()
    {
        occlusions_.resize(1);
        occlusions_[0] =
            std::vector<float>(n_rows_ * n_cols_, initial_occlusion_);
        occlusion_times_.resize(1);
        occlusion_times_[0] = std::vector<double>(n_rows_ * n_cols_, 0);
        observation_time_ = 0;
    }

    // TODO: TYPES
    const std::vector<float> Occlusions(size_t index) const
    {
        return occlusions_[index];
    }

private:
    void set_observation(const std::vector<float>& observations,
                         const Scalar& delta_time)
    {
        observations_ = observations;
        observation_time_ += delta_time;
    }

    // TODO: WE PROBABLY DONT NEED ALL OF THIS
    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_occlusion_;
    const std::shared_ptr<osr::RigidBodiesState<-1>> rigid_bodies_state_;

    // models
    ObjectRendererPtr object_model_;
    PixelObservationModelPtr observation_model_;
    OcclusionModelPtr occlusion_process_model_;

    // occlusion parameters
    std::vector<std::vector<float>> occlusions_;
    std::vector<std::vector<double>> occlusion_times_;

    // observed data
    std::vector<float> observations_;
    double observation_time_;
};
}
