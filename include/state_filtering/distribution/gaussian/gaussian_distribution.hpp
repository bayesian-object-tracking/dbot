/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 05/25/2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#ifndef STATE_FILTERING_DISTRIBUTION_GAUSSIAN_GAUSSIAN_DISTRIBUTION_HPP
#define STATE_FILTERING_DISTRIBUTION_GAUSSIAN_GAUSSIAN_DISTRIBUTION_HPP

#include <Eigen/Dense>

#include <boost/assert.hpp>

#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/sampleable.hpp>
#include <state_filtering/distribution/gaussian_mappable.hpp>
#include <state_filtering/distribution/probability_density_function.hpp>

namespace filter
{

/**
 * @brief Traits of GaussianDistribution. This class defines the vector and matrix type.
 */
template <typename ScalarType, int size>
class GaussianDistributionTraits:
        public DistributionTraits<ScalarType, size>
{
public:
    typedef Eigen::Matrix<ScalarType, size, size> CovarianceType;
};


/**
 * @brief GaussianDistribution is a parametrized distribution
 */
template <typename Traits>
class GaussianDistribution:
        public Distribution<Traits>,
        public Sampleable<Traits>,
        public ProbabilityDensityFunction<Traits>,
        public GaussianMappable<Traits>
{
public:
    typedef typename Traits::ScalarType ScalarType;
    typedef typename Traits::SampleType SampleType;
    typedef typename Traits::CovarianceType CovarianceType;

    GaussianDistribution()
    {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(SampleType);
        EIGEN_STATIC_ASSERT_FIXED_SIZE(CovarianceType);
    }

    virtual ~GaussianDistribution() { }

    virtual SampleType sample()
    {
        return mean_;
    }

    virtual SampleType mapFromGaussian(const SampleType& sample) const
    {
        return mean_ + L_ * sample;
    }

    inline virtual void setNormal()
    {
        BOOST_ASSERT_MSG(mean_.rows() > 0 || mean_.cols() > 0,
                         "Calling setNormal on dynamic sized gaussian without initialization");

        mean_.setZero();
        covariance_.setIdentity();
    }

    inline virtual void mean(const SampleType& mean)
    {
        mean_ = mean;
    }

    virtual void covariance(const CovarianceType& covariance)
    {
        // TODO: it should be able to deal with SEMI definite matrices as well!!
        // check the rank
        BOOST_ASSERT_MSG(covariance.colPivHouseholderQr().rank() != covariance.rows() ||
                         covariance.rows() != covariance.cols(),
                         "covariance matrix is not full rank");

        covariance_ = covariance;
        precision_ = covariance_.inverse();
        L_ = covariance_.llt().matrixL();

        // check the rank
        BOOST_ASSERT_MSG(!covariance_.isApprox(L_*L_.transpose()),
                         "LLT decomposition went wrong. Matrix might not positive definite!");

        log_normalizer_ = -0.5
                * ( log(covariance_.determinant()) + double(covariance.rows()) * log(2.0 * M_PI) );
    }

    inline virtual SampleType mean() const
    {
        return mean_;
    }

    inline virtual CovarianceType covariance() const
    {
        return covariance_;
    }


    virtual ScalarType logProbability(const SampleType& sample) const
    {
        return log_normalizer_ - 0.5 * (sample - mean_).transpose() * precision_ * (sample - mean_);
    }

protected:
    SampleType mean_;
    CovarianceType covariance_;
    CovarianceType precision_;
    CovarianceType L_;
    double log_normalizer_;
};


}

#endif
