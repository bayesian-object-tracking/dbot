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
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef STATE_FILTERING_DISTRIBUTIONS_SUM_OF_DELTAS_HPP
#define STATE_FILTERING_DISTRIBUTIONS_SUM_OF_DELTAS_HPP

// eigen
#include <Eigen/Dense>

// std
#include <vector>

// state_filtering
#include <state_filtering/utils/traits.hpp>
#include <state_filtering/distributions/interfaces/moments_interface.hpp>

namespace sf
{

// Forward declarations
template <typename Vector> class SumOfDeltas;

namespace internal
{
/**
 * SumOfDeltas distribution traits specialization
 * \internal
 */
template <typename Vector>
struct Traits<SumOfDeltas<Vector> >
{
    enum { Dimension = VectorTraits<Vector>::Dimension };

    typedef typename internal::VectorTraits<Vector>::Scalar Scalar;
    typedef Eigen::Matrix<Scalar,Dimension, Dimension>      Operator;

    typedef std::vector<Vector>                      Deltas;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Weights;

    typedef MomentsInterface<Vector, Operator> MomentsInterfaceBase;
};
}

/**
 * \class SumOfDeltas
 * \ingroup distributions
 */
template <typename Vector>
class SumOfDeltas:
        public internal::Traits<SumOfDeltas<Vector> >::MomentsInterfaceBase
{
public:
    typedef internal::Traits<SumOfDeltas<Vector> > Traits;

    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::Operator   Operator;
    typedef typename Traits::Deltas     Deltas;
    typedef typename Traits::Weights    Weights;

public:
    explicit SumOfDeltas(const unsigned& dimension = Traits::Dimension)
    {
        deltas_ = Deltas(1, Vector::Zero(dimension == Eigen::Dynamic? 0 : dimension));
        weights_ = Weights::Ones(1);
    }

    virtual ~SumOfDeltas() { }

    virtual void SetDeltas(const Deltas& deltas, const Weights& weights)
    {
        deltas_ = deltas;
        weights_ = weights.normalized();
    }

    virtual void SetDeltas(const Deltas& deltas)
    {
        deltas_ = deltas;
        weights_ = Weights::Ones(deltas_.size())/Scalar(deltas_.size());
    }

    virtual void GetDeltas(Deltas& deltas, Weights& weights) const
    {
        deltas = deltas_;
        weights = weights_;
    }

    virtual Vector Mean() const
    {
        Vector mean(Vector::Zero(Dimension()));
        for(size_t i = 0; i < deltas_.size(); i++)
            mean += weights_[i] * deltas_[i];

        return mean;
    }

    virtual Operator Covariance() const
    {
        Vector mean = Mean();
        Operator covariance(Operator::Zero(Dimension(), Dimension()));
        for(size_t i = 0; i < deltas_.size(); i++)
            covariance += weights_[i] * (deltas_[i]-mean) * (deltas_[i]-mean).transpose();

        return covariance;
    }

    virtual int Dimension() const
    {
        return deltas_[0].rows();
    }

protected:
    Deltas  deltas_;
    Weights weights_;
};

}

#endif
