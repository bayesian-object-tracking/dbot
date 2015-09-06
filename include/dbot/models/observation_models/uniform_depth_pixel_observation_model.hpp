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
 * \file uniform_depth_pixel_observation_model.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <Eigen/Dense>

#include <fl/util/descriptor.hpp>
#include <fl/distribution/uniform_distribution.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

template <typename State_>
class UniformDepthPixelObservationModel
    : public ObservationFunction<Vector1d, State_, Vector1d>,
      public ObservationDensity<Vector1d, State_>,
      public Descriptor
{
public:
    typedef Vector1d Obsrv;
    typedef Vector1d Noise;
    typedef State_   State;

public:
    UniformDepthPixelObservationModel(
        Real min_depth,
        Real max_depth,
        int state_dim = DimensionOf<State>::Value)
        : state_dim_(state_dim),
          density_(min_depth, max_depth)
    { }

    Real log_probability(const Obsrv& obsrv, const State& state) const override
    {
        return density_.log_probability(obsrv);
    }

    Real probability(const Obsrv& obsrv, const State& state) const override
    {
        return density_.probability(obsrv);
    }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = density_.map_standard_normal(noise);
        return y;
    }

    virtual int obsrv_dimension() const { return 1; }
    virtual int noise_dimension() const { return 1; }
    virtual int state_dimension() const { return state_dim_; }

    virtual std::string name() const
    {
        return "UniformDepthPixelObservationModel";
    }

    virtual std::string description() const
    {
        return "UniformDepthPixelObservationModel";
    }


private:
    int state_dim_;
    UniformDistribution density_;
};

}
