/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
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
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 *   University of Southern California
 */

#ifndef STATE_FILTERING_DISTRIBUTIONS_STATISTICAL_GAUSSIAN_HPP
#define STATE_FILTERING_DISTRIBUTIONS_STATISTICAL_GAUSSIAN_HPP

#include <Eigen/Dense>

#include <state_filtering/utils/traits.hpp>
#include <state_filtering/utils/macros.hpp>
#include <state_filtering/distributions/interfaces/moments_interface.hpp>

namespace sf
{

// Forward declarations
template <typename Vector> class StatisticalGaussian;

namespace internal
{
/**
 * StatisticalGaussian distribution traits specialization
 * \internal
 */
template <typename Vector>
struct Traits<StatisticalGaussian<Vector> >
{
    enum { Dimension = VectorTraits<Vector>::Dimension };

    typedef typename internal::VectorTraits<Vector>::Scalar  Scalar;
    typedef Eigen::Matrix<Scalar, Dimension, Dimension>      Operator;

    typedef Eigen::Vector<double, Eigen::Dynamic>            WeightVector;
    typedef Eigen::Matrix<Scalar, Dimension, Eigen::Dynamic> PointMatrix;

    typedef MomentsInterface<Vector, Operator> MomentsInterfaceBase;
};
}

/**
 * \class StatisticalGaussian
 * \ingroup distributions
 */
template <typename Vector>
class StatisticalGaussian:
        public internal::Traits<StatisticalGaussian<Vector> >::MomentsInterfaceBase
{
public:
    typedef internal::Traits<StatisticalGaussian<Vector> > Traits;

    typedef typename Traits::Scalar          Scalar;
    typedef typename Traits::Operator        Operator;
    typedef typename Traits::WeightVector    WeightVector;
    typedef typename Traits::PointMatrix     PointMatrix;

    /**
     * Constructs a matrix instance
     *
     * @param number_of_points
     */
    explicit StatisticalGaussian(int number_of_points = 1)
    {
        if (number_of_points > 0)
        {
            mean_weights_.resize(number_of_points);
            covariance_weights_.resize(number_of_points);
        }

        point_matrix_.resize(1, 1);
    }

    /**
     * Customizable destructor
     */
    virtual ~StatisticalGaussian() { }

    /**
     * Resizes the matrix and its weight vectors
     *
     * @param dimension         Rows of the matrix
     * @param number_of_points    Columns of the matrix
     */
    virtual void Resize(int number_of_points, int dimension = 1)
    {
        point_matrix_.resize(dimension, number_of_points);
        mean_weights_.resize(number_of_points);
        covariance_weights_.resize(number_of_points);
    }

    /**
     * Sets the sigma point and its related weights
     *
     * @param [in] index        Sigma point index
     * @param [in] point        sigma point column vector
     * @param [in] meanWeight   mean related weight
     * @param [in] covWeight    covariance matrix related weight
     */
    virtual void SetPoint(int index,
                          const Vector& point,
                          double mean_weight,
                          double covariance_weight)
    {
        if (col(index).rows() != point.rows())
        {
            ROS_ASSERT_MSG(index == 0,
                           "Only the first column can induce a change in "\
                           "dimension.");

            point_matrix_.resize(point.rows(), Eigen::NoChange);
        }

        col(index) = point;
        mean_weights_[index] = mean_weight;
        covariance_weights_[index] = covariance_weight;
    }   

    /**
     * Computes the weighted mean of the sigma points
     *
     * @param [out] mean    Mean of the sigma points
     */
    virtual void Mean(Vector& mean) const
    {
        mean.resize(point.rows(), 1);

        for (int i = 0; i < point_matrix_.cols(); i++)
        {
            mean += mean_weights_[i] * col(i);
        }
    }

    /**
     * Computes the weighted covariance matrix of the sigma points
     *
     * @param [out] covariance    Covariance of the sigma points
     */
    virtual void Covariance(Operator& covariance) const
    {
        PointMatrix covariance_factor;
        CovarianceFactor(covariance_factor);

        covariance =  covariance_factor
                        * covariance_weights_.asDiagonal()
                        * covariance_factor.transpose();
    }

    /**
     * Comutes the covariance factor matrix. A covariance factor matrix A
     * is the matrix containing the zero mean points defining the covariance
     * C=AWA', where W is the weight diagonal matrix
     *
     * @param [out] covariance_factor   The covariance factor matrix.
     */
    virtual void CovarianceFactor(PointMatrix& covariance_factor) const
    {
        Vector mean;
        Mean(mean);

        covariance_factor.resize(point_matrix_.rows(), point_matrix_.cols());
        for (int i = 0; i < point_matrix_.cols(); i++)
        {
            covariance_factor.col(i) = point_matrix_.col(i) - mean;
        }
    }

    /**
     * Returns the point matrix
     */
    virtual const PointMatrix& point_matrix() const
    {
        return point_matrix_;
    }

    /**
     * Returns the mean weight vector
     *
     * @return mean weight vector
     */
    virtual const WeightVector& mean_weights() const
    {
        return mean_weights_;
    }

    /**
     * Returns the covariance weight vector
     *
     * @return covariance weight vetcor
     */
    virtual const WeightVector& covariance_weights() const
    {
        return covariance_weights_;
    }

    /**
     * Returns the point at the specified index
     *
     * @param i     Index of the requested point
     */
    virtual const Vector& point(int i) const
    {
        return point_matrix_.col(i);
    }

    /**
     * Weight of the sample at the specified index used to compute the means
     *
     * @param index      Column index number
     * @return           Mean weight of the requested sigma point
     */
    virtual double mean_weight(int index) const
    {
        return mean_weights_[index];
    }

    /**
     * Weight of the sample at the specified index used to compute the
     * covariance
     *
     * @param index      Column index number
     * @return           Covariance weight of the requested sigma point
     */
    virtual double covariance_weight(int index) const
    {
        return covariance_weights_[index];
    }

    /**
     * Comutes the weighte covariance factor matrix. A weighted covariance
     * factor matrix B is the matrix containing the weighted zero mean points
     * defining the covariance C=BB'
     *
     * @param [out] covariance_factor   The covariance factor matrix.
     */
//    virtual void WeightedCovarianceFactor(PointMatrix& covariance_factor)
//    {

//    }

protected:                
    /**
     * Weights of samples to compute the mean
     */
    WeightVector mean_weights_;

    /**
     * Weights of samples to compute the covariance
     */
    WeightVector covariance_weights_;

    /**
     * The point matrix containing the data points
     */
    PointMatrix point_matrix_;
};

}

#endif

