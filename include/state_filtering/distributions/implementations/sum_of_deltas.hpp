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

#ifndef DISTRIBUTIONS_IMPLEMENTATIONS_SUM_OF_DELTAS_HPP
#define DISTRIBUTIONS_IMPLEMENTATIONS_SUM_OF_DELTAS_HPP

// eigen
#include <Eigen/Dense>

// std
#include <vector>

// state_filtering
#include <state_filtering/distributions/features/moments_solvable.hpp>

namespace sf
{

namespace internal
{
template <typename Scalar_, typename Vector_>
struct SumOfDeltasTypes
{
    typedef Scalar_ Scalar;
    typedef Vector_ Vector;
    typedef Eigen::Matrix<Scalar,
                          Vector::SizeAtCompileTime,
                          Vector::SizeAtCompileTime> Operator;
    typedef MomentsSolvable<Scalar,
                            Vector,
                            Operator> MomentsSolvableType;
};
}

template <typename Scalar_, typename Vector_>
class SumOfDeltas:
        public internal
                ::SumOfDeltasTypes<Scalar_, Vector_>
                ::MomentsSolvableType
{
public:
    typedef internal::SumOfDeltasTypes<Scalar_, Vector_> Base;
    typedef typename Base::Scalar   Scalar;
    typedef typename Base::Vector   Vector;
    typedef typename Base::Operator Operator;

    typedef std::vector<Vector>                         Deltas;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>    Weights;

public:
    /**
     * Constructs a fixed or dynamic sized sum of deltas distribution resembling
     * a Gaussian distribution with the deltas as its statistics.
     */
    SumOfDeltas()
    {
        // initialize with one delta at zero
        int dimension = Vector::SizeAtCompileTime == Eigen::Dynamic?
                    1 : Vector::SizeAtCompileTime;
        SetDeltas(Deltas(1, Vector::Zero(dimension)), Weights::Ones(1));
    }

    /**
     * Constructs dynamic sized sum of deltas distribution resembling a
     * Gaussian distribution with the deltas as its statistics.
     */
    explicit SumOfDeltas(const unsigned& dimension)
    {
        DISABLE_IF_FIXED_SIZE(Vector);

        // initialize with one delta at zero
        SetDeltas(Deltas(1, Vector::Zero(dimension)), Weights::Ones(1));
    }

    /**
     * Default customizable destructor
     */
    virtual ~SumOfDeltas() { }

    /**
     * Sets the distribution deltas and the associated weights and recomputes
     * the deltas' mean and covariance.
     *
     * @param [in] deltas               Distribution delta set
     * @param [in] weights              Deltas' weights
     * @param [in] covariance_weights   Deltas' second momement weights. If not
     *                                  set, it's equal to first moment weights
     *                                  specified above.
     * @param [in] normalize            Set true normalize the weights
     */
    virtual void SetDeltas(const Deltas& deltas,
                           const Weights& weights,
                           const Weights& covariance_weights = Weights(),
                           bool normalize = true)
    {
        deltas_ = deltas;
        mean_weights_ = normalize ? weights.normalized() : weights;
        if (covariance_weights.rows() > 0)
        {
            covariance_weights_ = normalize ?
                        covariance_weights.normalized() : covariance_weights;
        }
        else
        {
            covariance_weights_ = mean_weights_;
        }

        // recompute the mean and covariance
        mean_.setZero(Dimension());
        for(size_t i = 0; i < deltas_.size(); i++)
        {
            mean_ += mean_weights_[i] * deltas_[i];
        }

//        covariance_.setZero(Dimension(), Dimension());
//        for(size_t i = 0; i < deltas_.size(); i++)
//        {
//            covariance_ += covariance_weights_[i]
//                    * (deltas_[i]-mean) * (deltas_[i]-mean).transpose();
//        }

        // approx. 5x faster covariance by avoiding sum of dyadic products
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
                scatter_matrix(Dimension(), deltas_.size());

        for (int i = 0; i < deltas_.size(); i++)
        {
            scatter_matrix.col(i) = deltas_[i] - mean_;
        }

        covariance_ = scatter_matrix * scatter_matrix.transpose();
    }

    /**
     * Sets the distribution deltas and the associated weights and recomputes
     * the deltas' mean and covariance. This assumes equally weighted deltas.
     *
     * @param [in] deltas   Distribution delta set
     */
    virtual void SetDeltas(const Deltas& deltas)
    {
        SetDeltas(deltas,
                  Weights::Ones(deltas_.size())/Scalar_(deltas_.size()));
    }

    /**
     * Returns the set of deltas and their weights
     * @param [out]  deltas
     * @param [out]  weights
     */
    virtual void GetDeltas(Deltas& deltas, Weights& weights) const
    {
        deltas = deltas_;
        weights = mean_weights_;
    }

    /**
     * Returns the set of deltas and their first and second moment weights
     * @param [out]  deltas
     * @param [out]  weights
     */
    virtual void GetDeltas(Deltas& deltas,
                           Weights& mean_weights,
                           Weights& covariance_weights) const
    {
        deltas = deltas_;
        mean_weights = mean_weights_;
        covariance_weights = covariance_weights_;
    }

    /**
     * Returns the weighted mean of the deltas
     */
    virtual Vector Mean() const
    {
        return mean_;
    }

    /**
     * Returns sencond centered moment of the deltas
     */
    virtual Operator Covariance() const
    {
        return covariance_;
    }

    /**
     * Returns the dimension of distribution
     */
    virtual int Dimension() const
    {
        return deltas_[0].rows();
    }

protected:
    Deltas deltas_;
    Weights mean_weights_;
    Weights covariance_weights_;
    Vector mean_;
    Operator covariance_;
};

}

#endif
