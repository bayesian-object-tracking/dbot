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
 * \date March 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <Eigen/Dense>

#include <fl/distribution/gaussian.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

#include "../trackers/dev_test_tracker/vector_hashing.hpp"
#include "../trackers/dev_test_tracker/virtual_object.hpp"

namespace fl
{

template <typename State_> class PixelModel;

template <typename State_>
struct Traits<PixelModel<State_>>
{
    enum
    {
        ObsrvDim = 1,
        NoiseDim = 1,
        ParamDim = 1,
        StateDim = 1
    };

    typedef State_ State;
    typedef typename State::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;    

    typedef ObservationModelInterface<Obsrv, State, Noise> ObservationModelBase;
    typedef AdaptiveModel<Param> AdaptiveModelBase;
    typedef Gaussian<Noise> GaussianBase;

    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
};

template <typename State>
class PixelModel
    : public Traits<PixelModel<State>>::ObservationModelBase,
      public Traits<PixelModel<State>>::AdaptiveModelBase,
      public Traits<PixelModel<State>>::GaussianBase
{
private:
    typedef PixelModel This;

    typedef from_traits(Obsrv);
    typedef from_traits(Noise);
    typedef from_traits(Param);

public:
    using Traits<This>::GaussianBase::square_root;

public:
    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        Obsrv y;        
        y(0) = state(0) + sigma(param_(0)) * noise(0);
        return y;
    }

    double sigma(double b) { return std::exp(b) * square_root()(0); }

    virtual const Param& param() const { return param_; }
    virtual void param(Param params) { param_ = params; }


    virtual int obsrv_dimension() const { return Traits<This>::ObsrvDim; }
    virtual int noise_dimension() const { return Traits<This>::NoiseDim; }
    virtual int state_dimension() const { return Traits<This>::StateDim; }
    virtual int param_dimension() const { return Traits<This>::ParamDim; }

protected:
    Param param_;
};





template <typename State_> class DepthPixelModel;

template <typename State_>
struct Traits<DepthPixelModel<State_>>
{
    enum
    {
        ObsrvDim = 1,
        NoiseDim = 1,
        ParamDim = 1,
        StateDim = State_::SizeAtCompileTime
    };

    typedef State_ State;
    typedef typename State::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;

    typedef ObservationModelInterface<Obsrv, State, Noise> ObservationModelBase;
    typedef AdaptiveModel<Param> AdaptiveModelBase;
    typedef Gaussian<Noise> GaussianBase;

    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
};

template <typename State>
class DepthPixelModel
    : public Traits<DepthPixelModel<State>>::ObservationModelBase,
      public Traits<DepthPixelModel<State>>::AdaptiveModelBase,
      public Traits<DepthPixelModel<State>>::GaussianBase
{
private:
    typedef DepthPixelModel This;

    typedef from_traits(Obsrv);
    typedef from_traits(Noise);
    typedef from_traits(Param);

public:
    using Traits<This>::GaussianBase::square_root;

public:
    DepthPixelModel(std::shared_ptr<fl::RigidBodyRenderer> renderer,
                    int state_dimension = DimensionOf<State>())
        : renderer_(renderer),
          state_dimension_(state_dimension)
    {
        delta_time_ = -1;
        id_ = 0;
        pose_.setZero(6, 1);
    }

    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        if (delta_time_ != delta_time)
        {
            predictions_cache_.clear();
            delta_time_ = delta_time;
        }

        // render object of needed
        Eigen::MatrixXd pose = state.topRows(6);
        if (pose_ != pose || pose_.isZero())
        {
            pose_ = pose;

            if (predictions_cache_.find(pose) == predictions_cache_.end())
            {
                map(state, predictions_cache_[pose]);
            }
        }

        if (predictions_cache_.find(pose) == predictions_cache_.end())
        {
            std::cout << "exiting because of " << pose.transpose() << std::endl;
            exit(-1);
        }

        double depth = predictions_cache_[pose](id());

        Obsrv y(1, 1);
        if (depth > 0)
        {
            y(0) = depth + sigma(param_(0)) * square_root()(0) * noise(0);
        }
        else
        {
            y(0) = bg_depth_ + sigma(param_(0)) * bg_sigma_ * noise(0);
        }

        return y;
    }

    double sigma(double b) { return std::exp(b); }

    virtual const Param& param() const { return param_; }
    virtual void param(Param params) { param_ = params; }

    virtual int obsrv_dimension() const { return Traits<This>::ObsrvDim; }
    virtual int noise_dimension() const { return Traits<This>::NoiseDim; }
    virtual int param_dimension() const { return Traits<This>::ParamDim; }
    virtual int state_dimension() const { return state_dimension_; }

    virtual int id() const { return id_; }
    virtual void id(int new_id) { id_ = new_id; }

public:
    /** \cond INTERNAL */
    void map(const State& state, Eigen::MatrixXd& obsrv_image)
    {
        renderer_->state(state);
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, obsrv_image);
    }

    void convert(const std::vector<float>& depth,
                 Eigen::MatrixXd& obsrv_image)
    {
        const int pixel_count = depth.size();
        obsrv_image.resize(pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            obsrv_image(i) = (!std::isinf(depth[i]) ? depth[i] : -1);
        }
    }
    /** \endcond */

private:
    int id_;
    double delta_time_;
    Eigen::MatrixXd pose_;

protected:
    Param param_;

public:

    double bg_depth_;
    double bg_sigma_;

    std::vector<float> depth_rendering_;
    size_t parameters_dimension_;
    size_t state_dimension_;

    std::shared_ptr<fl::RigidBodyRenderer> renderer_;

    std::unordered_map<Eigen::MatrixXd,
                       Eigen::MatrixXd,
                       VectorHash<Eigen::MatrixXd>> predictions_cache_;
};




}
