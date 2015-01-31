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

#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/model/observation/factorized_iid_observation_model.hpp>

#include "vector_hashing.hpp"

namespace fl
{


template <
    typename State,
    typename Scalar,
    int ResRows,
    int ResCols>
class DepthObservationModel;

template <
    typename State_,
    typename Scalar,
    int ResRows,
    int ResCols>
struct Traits<DepthObservationModel<State_, Scalar, ResRows, ResCols>>
{
    enum
    {
        PixelObsrvDim = 1,
        PixelStateDim = 1
    };

    // [y  y^2] kernel space?
    typedef Eigen::Matrix<Scalar, PixelObsrvDim, 1> PixelObsrv;

    // [h_i(x) h_i(x)^2] rendered pixel
    typedef Eigen::Matrix<Scalar, PixelStateDim, 1> PixelState;

    // local linear gaussian observation model
    typedef LinearGaussianObservationModel<
                PixelObsrv,
                PixelState
            > PixelObsrvModel;

    typedef typename Traits<PixelObsrvModel>::SecondMoment PixelCov;
    typedef typename Traits<PixelObsrvModel>::SensorMatrix PixelSensorMatrix;

    // Holistic observation model
    typedef FactorizedIIDObservationModel<
                PixelObsrvModel,
                FactorSize<ResRows, ResCols>::Size
            > CameraObservationModel;

    typedef State_ State;
    typedef typename Traits<CameraObservationModel>::State StateInternal;
    typedef typename Traits<CameraObservationModel>::Observation Observation;
    typedef typename Traits<CameraObservationModel>::Noise Noise;

    typedef ANObservationModelInterface<
                Observation,
                State_,
                Noise
            > ANObservationModelBase;

    typedef ObservationModelInterface<
                Observation,
                State_,
                Noise
            > ObservationModelBase;
};


template <
    typename State,
    typename Scalar,
    int ResRows,
    int ResCols>
class DepthObservationModel
    : public Traits<
                 DepthObservationModel<State, Scalar, ResRows, ResCols>
             >::ANObservationModelBase,
      public Traits<
                 DepthObservationModel<State, Scalar, ResRows, ResCols>
             >::ObservationModelBase
{
public:
    typedef DepthObservationModel<State, Scalar, ResRows, ResCols> This;

    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::Noise Noise;

    enum
    {
        PixelObsrvDim =  Traits<This>::PixelObsrvDim,
        PixelStateDim =  Traits<This>::PixelStateDim
    };
    typedef typename Traits<This>::PixelObsrvModel PixelObsrvModel;
    typedef typename Traits<This>::PixelCov PixelCov;
    typedef typename Traits<This>::PixelSensorMatrix PixelSensorMatrix;

    typedef typename Traits<This>::StateInternal StateInternal;
    typedef typename Traits<This>::CameraObservationModel CameraObservationModel;

public:
    DepthObservationModel(std::shared_ptr<fl::RigidBodyRenderer> renderer,
                          Scalar camera_sigma,
                          Scalar model_sigma,
                          size_t state_dimension = DimensionOf<State>(),
                          int res_rows = ResRows,
                          int res_cols = ResCols)
        : camera_obsrv_model_(
              std::make_shared<PixelObsrvModel>(
                  PixelCov::Identity(PixelObsrvDim, PixelObsrvDim)
                  * ((camera_sigma*camera_sigma) + (model_sigma*model_sigma))),
              (res_rows*res_cols)),
          model_sigma_(model_sigma),
          camera_sigma_(camera_sigma),
          renderer_(renderer),
          state_dimension_(state_dimension)
    {
        assert(res_rows > 0);
        assert(res_cols > 0);
        assert(state_dimension_ > 0);

        depth_rendering_.resize(res_rows * res_cols);
    }


    ~DepthObservationModel() { }

    /**
     * \return Prediction assuming additive noise
     */
    virtual Observation predict_observation(const State& state,
                                            double delta_time)
    {
        Eigen::MatrixXd pose = state.topRows(6);

        if (predictions_cache_.find(pose) == predictions_cache_.end())
        {
            map(state, predictions_cache_[pose]);
        }

        return predictions_cache_[pose];
    }

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

        return camera_obsrv_model_.predict_observation(
                    predictions_cache_[pose],
                    noise,
                    delta_time);

//        map(state, s_i_);

//        return camera_obsrv_model_.predict_observation(
//                    s_i_,
//                    noise,
//                    delta_time);
    }

    virtual size_t observation_dimension() const
    {
        return camera_obsrv_model_.observation_dimension();
    }

    virtual size_t state_dimension() const
    {
        return state_dimension_;
    }

    virtual size_t noise_dimension() const
    {
        return camera_obsrv_model_.noise_dimension();
    }

    virtual Noise noise_covariance_vector() const
    {
        return Noise::Ones(noise_dimension(), 1) *
               (model_sigma_ * model_sigma_ + camera_sigma_ * camera_sigma_);
    }

    virtual void clear_cache()
    {
        predictions_cache_.clear();
    }

public:
    /** \cond INTERNAL */
    void map(const State& state, StateInternal& state_internal)
    {
        renderer_->state(state);
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, state_internal);
    }

    void convert(const std::vector<float>& depth,
                 StateInternal& state_internal)
    {
        const int pixel_count = depth.size();
        state_internal.resize(pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            state_internal(i, 0) = (std::isinf(depth[i]) ? 7 : depth[i]);
        }
    }

    /** \endcond */

protected:
    CameraObservationModel camera_obsrv_model_;
    Scalar model_sigma_;
    Scalar camera_sigma_;
    std::shared_ptr<fl::RigidBodyRenderer> renderer_;
    std::vector<float> depth_rendering_;
    size_t state_dimension_;
    StateInternal s_i_;

    std::unordered_map<Eigen::MatrixXd,
                       StateInternal,
                       VectorHash<Eigen::MatrixXd>> predictions_cache_;
};

}
