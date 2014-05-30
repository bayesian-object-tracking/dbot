
#ifndef FILTERS_UKF_UKF_SIGMA_POINT_SAMPLER_HPP
#define FILTERS_UKF_UKF_SIGMA_POINT_SAMPLER_HPP

#include <boost/shared_ptr.hpp>

#include <Eigen/Eigen>

#include <state_filtering/filter/kalman/spkf/types.hpp>
#include <state_filtering/filter/kalman/spkf/sigma_point_matrix.hpp>
#include <state_filtering/filter/kalman/spkf/distribution_descriptor.hpp>

namespace filter
{
    /**
     * Generic sigma point sampler interface
     */
    class SigmaPointTransform
    {
    public:
        typedef boost::shared_ptr<SigmaPointTransform> Ptr;

    public:
        /**
         * @brief forward Samples sigma points from the given gaussian distribution
         *
         * @param [in]  distDesc        Input distribution descriptor
         * @param [out] sigmaPoints     Sampled sigma points
         *
         * Samples 2n+1 sigma points from the given gaussian distribution defined by the mean vector
         * and its covariance matrix. n is the dimension of the state
         */
        virtual void forward(const DistributionDescriptor& distDesc,
                             SigmaPointMatrix& sigmaPoints) = 0;

        /**
         * @brief Calculates the gaussian distribution out of the given points
         *
         * Uses the 2n+1 sigma points to determine the mean and the covariance of the gaussian
         *
         * @param [in]  sigmaPoints  Given sigma points
         * @param [out] distDesc     Reconstructed gaussian distribution description
         */
        virtual void backward(const SigmaPointMatrix& sigmaPoints,
                              DistributionDescriptor& distDesc,
                              int subDimension = 0,
                              int marginBegin = 0) = 0;
    };
}

#endif
