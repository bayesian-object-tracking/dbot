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
 * J. Issac, M. Wuthrich, C. Garcia Cifuentes, J. Bohg, S. Trimpe, S. Schaal
 * Depth-Based Object Tracking Using a Robust Gaussian Filter
 * IEEE Intl Conf on Robotics and Automation, 2016
 * http://arxiv.org/abs/1602.06157
 *
 */

/**
 * \file depth_pixel_model.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <mutex>
#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <Eigen/Dense>

#include <fl/util/descriptor.hpp>
#include <fl/util/scalar_matrix.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/uniform_distribution.hpp>
#include <fl/distribution/cauchy_distribution.hpp>
#include <fl/model/sensor/interface/sensor_density.hpp>
#include <fl/model/sensor/interface/sensor_function.hpp>

#include <dbot/rigid_body_renderer.hpp>

#include <osr/pose_hashing.hpp>

namespace fl
{
template <typename State_>
class DepthPixelModel
    : public SensorFunction<Vector1d, State_, Vector1d>,
      public SensorDensity<Vector1d, State_>,
      public Descriptor
{
public:
    typedef Vector1d Obsrv;
    typedef Vector1d Noise;
    typedef State_ State;

    typedef std::unordered_map<State, State, osr::PoseHash<State>> PoseCacheMap;

    typedef std::unordered_map<State, Eigen::VectorXd, osr::PoseHash<State>>
        RenderCacheMap;

public:
    DepthPixelModel(
        const std::shared_ptr<dbot::RigidBodyRenderer>& renderer,
        Real bg_depth,
        Real fg_sigma,
        Real bg_sigma,
        int state_dim = DimensionOf<State>::Value)
        : state_dim_(state_dim), renderer_(renderer), id_(0)
    {
        mutex = std::make_shared<std::mutex>();
        render_cache_ = std::make_shared<RenderCacheMap>();
        poses_cache_ = std::make_shared<PoseCacheMap>();

        // setup backgroud density
        auto bg_mean = Obsrv(1);

        bg_mean(0) =
            bg_depth < 0. ? std::numeric_limits<Real>::infinity() : bg_depth;
        bg_density_.mean(bg_mean);
        bg_density_.square_root(bg_density_.square_root() * bg_sigma);

        fg_density_.square_root(fg_density_.square_root() * fg_sigma);
    }

    DepthPixelModel(const DepthPixelModel& other)
    {
        state_dim_ = other.state_dim_;
        renderer_ = other.renderer_;
        id_ = other.id_;
        bg_density_ = other.bg_density_;
        fg_density_ = other.fg_density_;
        mutex = other.mutex;
        nominal_pose_ = other.nominal_pose_;
        render_cache_ = other.render_cache_;
        poses_cache_ = other.poses_cache_;
    }

    virtual ~DepthPixelModel() noexcept {}
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
        std::lock_guard<std::mutex> lock(*mutex);

        render_cache_->clear();
        nominal_pose_ = p;
    }

    virtual std::string name() const { return "DepthPixelModel"; }
    virtual std::string description() const
    {
        return "DepthPixelModel";
    }

private:
    /** \cond internal */
    void map(const State& pose, Eigen::VectorXd& obsrv_image) const
    {
        renderer_->set_poses({pose.component(0).affine()});
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, obsrv_image);
    }

    void convert(const std::vector<float>& depth,
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
        RenderCacheMap& render_cache_ = *this->render_cache_;
        PoseCacheMap& poses_cache_ = *this->poses_cache_;

        if (render_cache_.find(current_state) == render_cache_.end())
        {
            State current_pose = current_state;


            /// \todo: this transformation should not be done in here

            current_pose.component(0).position() =
                nominal_pose_.component(0).orientation().rotation_matrix() *
                    current_state.component(0).position() +
                nominal_pose_.component(0).position();

            current_pose.component(0).orientation() =
                nominal_pose_.component(0).orientation() *
                current_state.component(0).orientation();



            std::lock_guard<std::mutex> lock(*mutex);
            map(current_pose, render_cache_[current_state]);
            poses_cache_[current_state] = current_pose;
        }

        assert(render_cache_.find(current_state) != render_cache_.end());

        Obsrv depth;
        depth(0) = render_cache_[current_state](id_);

        return depth;
    }
    /** \endcond */

private:
    int state_dim_;

    mutable Gaussian<Obsrv> fg_density_;
    mutable Gaussian<Obsrv> bg_density_;

    mutable std::shared_ptr<std::mutex> mutex;
    mutable std::vector<float> depth_rendering_;
    std::shared_ptr<dbot::RigidBodyRenderer> renderer_;

private:
    int id_;
    mutable State nominal_pose_;

public:
    mutable std::shared_ptr<RenderCacheMap> render_cache_;
    mutable std::shared_ptr<PoseCacheMap> poses_cache_;
};
}
