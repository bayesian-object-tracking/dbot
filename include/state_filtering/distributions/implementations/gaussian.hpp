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

#ifndef DISTRIBUTIONS_IMPLEMENTATIONS_GAUSSIAN_HPP
#define DISTRIBUTIONS_IMPLEMENTATIONS_GAUSSIAN_HPP

// eigen
#include <Eigen/Dense>

// state_filtering
#include <state_filtering/distributions/features/moments_solvable.hpp>
#include <state_filtering/distributions/features/evaluable.hpp>
#include <state_filtering/distributions/features/gaussian_mappable.hpp>

namespace distributions
{




template <typename ScalarType_, int DIMENSION_EIGEN>
struct GaussianTypes
{
    typedef ScalarType_                                                     ScalarType;
    typedef Eigen::Matrix<ScalarType, DIMENSION_EIGEN, 1>                   VectorType;
    typedef Eigen::Matrix<ScalarType, DIMENSION_EIGEN, DIMENSION_EIGEN>     OperatorType;

    typedef MomentsSolvable<ScalarType, VectorType, OperatorType>           MomentsSolvableType;
    typedef Evaluable<ScalarType, VectorType>                               EvaluableType;
    typedef GaussianMappable<ScalarType, VectorType, DIMENSION_EIGEN>       GaussianMappableType;

    typedef typename GaussianMappableType::NoiseType                        InputType;
};






template <typename ScalarType_, int DIMENSION_EIGEN>
class Gaussian: public GaussianTypes<ScalarType_, DIMENSION_EIGEN>::MomentsSolvableType,
                public GaussianTypes<ScalarType_, DIMENSION_EIGEN>::EvaluableType,
                public GaussianTypes<ScalarType_, DIMENSION_EIGEN>::GaussianMappableType
{
public:
    typedef GaussianTypes<ScalarType_, DIMENSION_EIGEN> Types;

    typedef typename Types::ScalarType    ScalarType;
    typedef typename Types::VectorType    VectorType;
    typedef typename Types::OperatorType  OperatorType;
    typedef typename Types::InputType     NoiseType;

public:
    Gaussian()
    {
        DISABLE_IF_DYNAMIC_SIZE(VectorType);

        SetUnit();
    }

    explicit Gaussian(const unsigned& dimension): Types::GaussianMappableType(dimension)
    {
        DISABLE_IF_FIXED_SIZE(VectorType);

        mean_.resize(dimension, 1);
        covariance_.resize(dimension, dimension);
        precision_.resize(dimension, dimension);
        cholesky_factor_.resize(dimension, dimension);

        SetUnit();
    }

    virtual ~Gaussian() { }

    virtual VectorType MapGaussian(const NoiseType& sample) const
    {
        return mean_ + cholesky_factor_ * sample;
    }

    virtual void SetUnit()
    {
        full_rank_ = true;
        Mean(VectorType::Zero(Dimension()));
        Covariance(OperatorType::Identity(Dimension(), Dimension()));
    }

    virtual void Mean(const VectorType& mean)
    {
        mean_ = mean;
    }

    virtual void Covariance(const OperatorType& covariance)
    {
        covariance_ = covariance;

        // we assume that the input matrix is positive semidefinite
        Eigen::LDLT<OperatorType> ldlt;
        ldlt.compute(covariance_);
        OperatorType L = ldlt.matrixL();
        VectorType D_sqrt = ldlt.vectorD();
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

    virtual VectorType Mean() const
    {
        return mean_;
    }

    virtual OperatorType Covariance() const
    {
        return covariance_;
    }

    virtual ScalarType LogProbability(const VectorType& vector) const
    {
        if(full_rank_)
            return log_normalizer_ - 0.5 * (vector - mean_).transpose() * precision_ * (vector - mean_);
        else
            return -std::numeric_limits<ScalarType>::infinity();
    }

    virtual int Dimension() const
    {
        return this->NoiseDimension(); // all dimensions are the same
    }


private:
    VectorType mean_;
    OperatorType covariance_;
    bool full_rank_;
    OperatorType precision_;
    OperatorType cholesky_factor_;
    ScalarType log_normalizer_;
};

}

#endif
