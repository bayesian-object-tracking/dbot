/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *    Jan Issac (jan.issac@gmail.com)
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
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef STATE_FILTERING_DISTRIBUTION_IMPLEMENTATIONS_GAUSSIAN_DISTRIBUTION_HPP
#define STATE_FILTERING_DISTRIBUTION_IMPLEMENTATIONS_GAUSSIAN_DISTRIBUTION_HPP

// eigen
#include <Eigen/Dense>

// boost
#include <boost/assert.hpp>
#include <boost/utility/enable_if.hpp>

// state_filtering
#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/features/moments_solvable.hpp>
#include <state_filtering/distribution/features/evaluable.hpp>
#include <state_filtering/distribution/features/gaussian_mappable.hpp>

namespace filter
{

/**
 * @brief GaussianDistribution is a parametrized distribution
 */
template <typename ScalarType_, int SIZE>
class GaussianDistribution:
        public MomentsSolvable<ScalarType_, SIZE>,
        public Evaluable<ScalarType_, SIZE>,
        public GaussianMappable<ScalarType_, SIZE, SIZE>
{
public: /* distribution traits */
    typedef MomentsSolvable<ScalarType_, SIZE>        MomentsBaseType;
    typedef GaussianMappable<ScalarType_, SIZE, SIZE>   MappableBaseType;

    typedef typename MomentsBaseType::Scalar        Scalar;
    typedef typename MomentsBaseType::Variable      Variable;
    typedef typename MomentsBaseType::CovarianceType    CovarianceType;
    typedef typename MappableBaseType::Sample      Sample;

public:
    GaussianDistribution()
    {
        DISABLE_IF_DYNAMIC_SIZE(Variable);

        setNormal();
    }

    explicit GaussianDistribution(int variable_size)
    {
        DISABLE_IF_FIXED_SIZE(Variable);

        mean_.resize(variable_size, 1);
        covariance_.resize(variable_size, variable_size);
        precision_.resize(variable_size, variable_size);
        cholesky_factor_.resize(variable_size, variable_size);

        setNormal();
    }

    virtual ~GaussianDistribution() { }

    virtual Variable mapNormal(const Sample& sample) const
    {
        return mean_ + cholesky_factor_ * sample;
    }

    virtual void setNormal()
    {
        full_rank_ = true;
        mean(Variable::Zero(variable_size()));
        covariance(CovarianceType::Identity(variable_size(), variable_size()));
    }

    virtual void mean(const Variable& mean)
    {
        mean_ = mean;
    }

    virtual void covariance(const CovarianceType& covariance)
    {
        covariance_ = covariance;

        // we assume that the input matrix is positive semidefinite
        Eigen::LDLT<CovarianceType> ldlt;
        ldlt.compute(covariance_);
        CovarianceType L = ldlt.matrixL();
        Variable D_sqrt = ldlt.vectorD();
        for(size_t i = 0; i < D_sqrt.rows(); i++)
            D_sqrt(i) = std::sqrt(std::fabs(D_sqrt(i)));
        cholesky_factor_ = ldlt.transpositionsP().transpose()*L*D_sqrt.asDiagonal();

        if(covariance.colPivHouseholderQr().rank() == covariance.rows())
        {
            full_rank_ = true;
            precision_ = covariance_.inverse();
            log_normalizer_ = -0.5 * ( log(covariance_.determinant()) + double(covariance.rows()) * log(2.0 * M_PI) );
        }
        else
            full_rank_ = false;
    }

    virtual Variable mean() const
    {
        return mean_;
    }

    virtual CovarianceType covariance() const
    {
        return covariance_;
    }

    virtual Scalar LogProbability(const Variable& sample) const
    {
        if(full_rank_)
            return log_normalizer_ - 0.5 * (sample - mean_).transpose() * precision_ * (sample - mean_);
        else
            return -std::numeric_limits<Scalar>::infinity();
    }

    virtual int variable_size() const
    {
        return mean_.rows();
    }

    virtual int sample_size() const
    {
        return variable_size();
    }

protected:
    Variable mean_;
    CovarianceType covariance_;
    bool full_rank_;
    CovarianceType precision_;
    CovarianceType cholesky_factor_;
    double log_normalizer_;
};

}

#endif
