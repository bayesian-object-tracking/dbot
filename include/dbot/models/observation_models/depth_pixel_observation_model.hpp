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
 * \file depth_pixel_observation_model.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <Eigen/Dense>

#include <fl/util/descriptor.hpp>
#include <fl/util/scalar_matrix.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/uniform_distribution.hpp>
#include <fl/distribution/cauchy_distribution.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

#include <dbot/utils/pose_hashing.hpp>
#include <dbot/utils/rigid_body_renderer.hpp>

namespace fl
{

template <typename State_>
class DepthPixelObservationModel
    : public ObservationFunction<Vector1d, State_, Vector1d>,
      public ObservationDensity<Vector1d, State_>,
      public Descriptor
{
public:
    typedef Vector1d Obsrv;
    typedef Vector1d Noise;
    typedef State_   State;

public:
    DepthPixelObservationModel(
        const std::shared_ptr<dbot::RigidBodyRenderer>& renderer,
        Real bg_depth,
        Real fg_sigma,
        Real bg_sigma,
        int state_dim = DimensionOf<State>::Value)
        : state_dim_(state_dim),
          renderer_(renderer),
          id_(0)
    {
        // setup backgroud density
        auto bg_mean = Obsrv(1);

        bg_mean(0) =
            bg_depth < 0. ? std::numeric_limits<Real>::infinity() : bg_depth;
        bg_density_.mean(bg_mean);
        bg_density_.square_root(bg_density_.square_root() * bg_sigma);

        // setup backgroud density
        fg_density_.square_root(fg_density_.square_root() * fg_sigma);
    }

    Real log_probability(const Obsrv& obsrv, const State& state) const override
    {
        return density(state).log_probability(obsrv);
    }

    Real probability(const Obsrv& obsrv, const State& state) const override
    {
        return density(state).probability(obsrv);
    }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = density(state).map_standard_normal(noise);
        return y;
    }

    virtual int obsrv_dimension() const { return 1; }
    virtual int noise_dimension() const { return 1; }
    virtual int state_dimension() const { return state_dim_; }

    virtual int id() const { return id_; }
    virtual void id(int new_id) { id_ = new_id; }

    void nominal_pose(const State& p)
    {
        render_cache_.clear();
        nominal_pose_= p;
    }

    virtual std::string name() const
    {
        return "DepthPixelObservationModel";
    }

    virtual std::string description() const
    {
        return "DepthPixelObservationModel";
    }

private:
    /** \cond internal */
    void map(const State& pose, Eigen::VectorXd& obsrv_image) const
    {
        renderer_->set_poses({pose.affine()});
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, obsrv_image);
    }

    void convert(
        const std::vector<float>& depth,
        Eigen::VectorXd& obsrv_image) const
    {
        const int pixel_count = depth.size();
        obsrv_image.resize(pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            if (!std::isinf(depth[i]))
            {
                obsrv_image(i, 0) = depth[i];
            }
            else
            {
                obsrv_image(i, 0) = std::numeric_limits<double>::infinity();
            }
        }
    }

    const Gaussian<Obsrv>& density(const State& state) const
    {
        Obsrv y = depth(state);

        if (std::isinf(y(0)))
        {
            return bg_density_;
        }

       fg_density_.mean(y);
       return fg_density_;
    }


    Obsrv depth(const State& current_state) const
    {
        if (render_cache_.find(current_state) == render_cache_.end())
        {
            State current_pose;
            current_pose.orientation() =
                current_state.orientation() * nominal_pose_.orientation();

            current_pose.position() =
                current_state.position() + nominal_pose_.position();

            map(current_pose, render_cache_[current_state]);
        }

        assert (render_cache_.find(current_state) != render_cache_.end());

        Obsrv depth;
        depth(0) = render_cache_[current_state](id_);

        return depth;
    }
    /** \endcond */

public:
    int state_dim_;

    mutable Gaussian<Obsrv> fg_density_;
    mutable Gaussian<Obsrv> bg_density_;

    mutable std::vector<float> depth_rendering_;
    std::shared_ptr<dbot::RigidBodyRenderer> renderer_;

    mutable std::unordered_map<
                State,
                Eigen::VectorXd,
                PoseHash<State>
            > render_cache_;

public:
    int id_;
    mutable State nominal_pose_;
};

}
