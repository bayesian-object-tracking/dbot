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
 * \file squared_feature_policy.hpp
 * \date March 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef SQUARED_FEATURE_POLICY_HPP
#define SQUARED_FEATURE_POLICY_HPP

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/util/math.hpp>
#include <fl/filter/gaussian/feature_policy.hpp>

namespace fl
{

template <typename ...> class SquaredFeaturePolicy;

template <typename Obsrv>
struct Traits<SquaredFeaturePolicy<Obsrv>>
{
    enum : signed int
    {
        SizeFactor = 2,
        FeatureDim = ExpandSizes<Obsrv::RowsAtCompileTime, SizeFactor>::Size
    };

    /**
     * Observation feature vector type
     */
    typedef Eigen::Matrix<typename Obsrv::Scalar, FeatureDim, 1> ObsrvFeature;

    /**
     * FeaturePolicy interface
     */
    typedef FeaturePolicyInterface<Obsrv, ObsrvFeature> FeaturePolicyBase;
};


template <typename Obsrv>
class SquaredFeaturePolicy<Obsrv>
    : public Traits<SquaredFeaturePolicy<Obsrv>>::FeaturePolicyBase
{
public:
    typedef SquaredFeaturePolicy This;
    typedef from_traits(ObsrvFeature);

    enum : signed int { SizeFactor = Traits<This>::SizeFactor };

    /**
     * \return Feature of the given observation
     */
    virtual ObsrvFeature extract(const Obsrv& obsrv,
                                 const Obsrv& expected_obsrv,
                                 const Obsrv& var_obsrv)
    {
        const int obsrv_dim = obsrv.rows();

        ObsrvFeature feature(feature_dimension(obsrv_dim), 1);

        for (int i = 0; i < obsrv_dim; ++i)
        {
            double d = obsrv(i); // - expected_obsrv(i);
            feature(SizeFactor * i) = d;
            feature(SizeFactor * i + 1) = d * d;
        }

        return feature;
    }

    static constexpr int feature_dimension(int obsrv_dimension)
    {
        return obsrv_dimension * SizeFactor;
    }
};





template <typename ...> class SigmoidFeaturePolicy;

template <typename Obsrv>
struct Traits<SigmoidFeaturePolicy<Obsrv>>
{
    enum : signed int
    {
        SizeFactor = 2,
        FeatureDim = ExpandSizes<Obsrv::RowsAtCompileTime, SizeFactor>::Size
    };

    /**
     * Observation feature vector type
     */
    typedef Eigen::Matrix<typename Obsrv::Scalar, FeatureDim, 1> ObsrvFeature;

    /**
     * FeaturePolicy interface
     */
    typedef FeaturePolicyInterface<Obsrv, ObsrvFeature> FeaturePolicyBase;
};


template <typename Obsrv>
class SigmoidFeaturePolicy<Obsrv>
    : public Traits<SigmoidFeaturePolicy<Obsrv>>::FeaturePolicyBase
{
public:
    typedef SigmoidFeaturePolicy This;
    typedef from_traits(ObsrvFeature);

    enum : signed int { SizeFactor = Traits<This>::SizeFactor };

    /**
     * \return Feature of the given observation
     */
    virtual ObsrvFeature extract(const Obsrv& obsrv,
                                 const Obsrv& expected_obsrv,
                                 const Obsrv& var_obsrv)
    {
        const int obsrv_dim = obsrv.rows();

        ObsrvFeature feature(feature_dimension(obsrv_dim), 1);

        for (int i = 0; i < obsrv_dim; ++i)
        {
            double d =
                fl::sigmoid(

                        (obsrv(i) - expected_obsrv(i)) / std::sqrt(var_obsrv(i))

                );
            feature(SizeFactor * i) = d;
            feature(SizeFactor * i + 1) = d * d;
        }

        return feature;
    }

    static constexpr int feature_dimension(int obsrv_dimension)
    {
        return obsrv_dimension * SizeFactor;
    }
};

}

#endif
