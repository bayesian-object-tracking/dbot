/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

#include <pose_tracking/utils/rigid_body_renderer.hpp>

#include "vector_hashing.hpp"

namespace fl
{
// forward declaration
template <
    typename CameraObservationModel,
    typename State,
    int ResRows,
    int ResCols>
class DepthObservationModel;

/**
 * Traits of DepthObservationModel
 */
template <
    typename CameraObservationModel,
    typename State_,
    int ResRows,
    int ResCols>
struct Traits<
           DepthObservationModel<
               CameraObservationModel, State_, ResRows, ResCols>>
{        
    typedef State_ State;
    typedef typename State::Scalar Scalar;
    typedef typename Traits<CameraObservationModel>::State StateInternal;
    typedef typename Traits<CameraObservationModel>::Observation Observation;
    typedef typename Traits<CameraObservationModel>::Noise Noise;

    typedef ObservationModelInterface<
                Observation,
                State_,
                Noise
            > ObservationModelBase;
};



/**
 * \class DepthObservationModel
 */
template <
    typename CameraObservationModel,
    typename State,
    int ResRows = Eigen::Dynamic,
    int ResCols = Eigen::Dynamic>
class DepthObservationModel
    : public Traits<
                 DepthObservationModel<
                    CameraObservationModel, State, ResRows, ResCols>
             >::ObservationModelBase
{
public:
    typedef DepthObservationModel<
                CameraObservationModel, State, ResRows, ResCols
            > This;

    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::StateInternal StateInternal;

public:
    DepthObservationModel(
            std::shared_ptr<CameraObservationModel> camera_obsrv_model,
            std::shared_ptr<fl::RigidBodyRenderer> renderer,
            size_t pose_state_dimension,
            size_t parameters_dimension)
        : camera_obsrv_model_(camera_obsrv_model),
          renderer_(renderer),
          pose_state_dimension_(pose_state_dimension),
          parameters_dimension_(parameters_dimension)
    {
        assert(pose_state_dimension_ > 0);
        assert(parameters_dimension_ > 0);
    }


    ~DepthObservationModel() { }

    /**
     * \return Prediction assuming non-additive noise
     */
    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Eigen::MatrixXd pose = state.topRows(6);

        if (predictions_cache_.find(pose) == predictions_cache_.end())
        {
            map(state, predictions_cache_[pose]);
        }

        return camera_obsrv_model_->predict_observation(
                    predictions_cache_[pose],
                    noise,
                    delta_time);
    }

    virtual size_t observation_dimension() const
    {
        return camera_obsrv_model_->observation_dimension();
    }

    virtual size_t state_dimension() const
    {
        return pose_state_dimension_ + parameters_dimension_;
    }

    virtual size_t noise_dimension() const
    {
        return camera_obsrv_model_->noise_dimension();
    }

    virtual void clear_cache()
    {
        predictions_cache_.clear();
    }

public:
    /** \cond INTERNAL */
    void map(const State& state, StateInternal& state_internal)
    {
        renderer_->state(state.topRows(pose_state_dimension_));
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, state, state_internal);
    }

    void convert(const std::vector<float>& depth,
                 const State& state,
                 StateInternal& state_internal)
    {
        const int pixel_count = depth.size();
        state_internal.resize(2 * pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            state_internal(2 * i) = (std::isinf(depth[i]) ? 7 : depth[i]);
            state_internal(2 * i + 1) = state(pose_state_dimension_ + i);
        }
    }

    /** \endcond */

protected:
    std::shared_ptr<CameraObservationModel> camera_obsrv_model_;
    std::shared_ptr<fl::RigidBodyRenderer> renderer_;

    std::vector<float> depth_rendering_;
    size_t parameters_dimension_;
    size_t pose_state_dimension_;

    std::unordered_map<Eigen::MatrixXd,
                       StateInternal,
                       VectorHash<Eigen::MatrixXd>> predictions_cache_;
};




// Forward declarations
template <
    typename PixelObservationModel,
    typename State,
    int Pixels
>
class ExperimentalObservationModel;

/**
 * Traits of ExperimentalObservationModel
 */
template <
    typename PixelObservationModel,
    typename State,
    int Pixels
>
struct Traits<
           ExperimentalObservationModel<PixelObservationModel, State, Pixels>
        >
{
    static constexpr int IIDPixels = Pixels;

    typedef typename Traits<PixelObservationModel>::Scalar Scalar;
    typedef typename Traits<PixelObservationModel>::State LocalState;
    typedef typename Traits<PixelObservationModel>::Observation LocalObservation;
    typedef typename Traits<PixelObservationModel>::Noise LocalNoise;

    typedef Eigen::Matrix<
                Scalar,
                FactorSizes<LocalObservation::RowsAtCompileTime, Pixels>::Size,
                1
            > Observation;

    typedef Eigen::Matrix<
                Scalar,
                FactorSizes<LocalNoise::RowsAtCompileTime, Pixels>::Size,
                1
            > Noise;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <
    typename PixelObservationModel,
    typename State,
    int Pixels = Eigen::Dynamic>
class ExperimentalObservationModel
    : public Traits<
                 ExperimentalObservationModel<
                     PixelObservationModel,
                     State,
                     Pixels
                 >
             >::ObservationModelBase
{
public:
    typedef ExperimentalObservationModel<
                PixelObservationModel,
                State,
                Pixels
            > This;

    typedef typename Traits<This>::LocalState LocalState;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::Noise Noise;

public:
    ExperimentalObservationModel(
            const std::shared_ptr<PixelObservationModel>& pixel_obsrv_model,
            size_t state_dimension,
            int pixels = Pixels)
        : pixel_obsrv_model_(pixel_obsrv_model),
          state_dimension_(state_dimension),
          pixels_(pixels)
    { }

    ~ExperimentalObservationModel() { }

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Observation y = Observation::Zero(observation_dimension(), 1);

        int obsrv_dim = pixel_obsrv_model_->observation_dimension();
        int noise_dim = pixel_obsrv_model_->noise_dimension();
        int state_dim = pixel_obsrv_model_->state_dimension();

        LocalState local_state(state_dim, 1);
        local_state(0) = state(0);

        for (int i = 0; i < pixels_; ++i)
        {
            local_state(1) = state(1 + i);

            y.middleRows(i * obsrv_dim, obsrv_dim) =
                pixel_obsrv_model_->predict_observation(
                    local_state,
                    noise.middleRows(i * noise_dim, noise_dim),
                    delta_time);
        }

        return y;
    }

    virtual size_t observation_dimension() const
    {
        return pixel_obsrv_model_->observation_dimension() * pixels_;
    }

    virtual size_t state_dimension() const
    {
        return state_dimension_;
    }

    virtual size_t noise_dimension() const
    {
        return pixel_obsrv_model_->noise_dimension() * pixels_;
    }

    const std::shared_ptr<PixelObservationModel>& pixel_observation_model()
    {
        return pixel_obsrv_model_;
    }

protected:
    std::shared_ptr<PixelObservationModel> pixel_obsrv_model_;
    size_t state_dimension_;
    size_t pixels_;
};


}
