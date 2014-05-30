

#ifndef FILTERS_SPKF_SPKF_DISTRIBUTION_DESCRIPTOR_HPP
#define FILTERS_SPKF_SPKF_DISTRIBUTION_DESCRIPTOR_HPP

#include <Eigen/Eigen>

#include <boost/shared_ptr.hpp>

#include "filters/spkf/types.hpp"

namespace filter
{
    class SigmaPointMatrix;

    /**
     * State descriptor interface
     *
     * Different Kalman filters use different representations of the state. This generic interface
     * provides access to the state without putting any resitrictions on the way it is
     * representated.
     */
    class DistributionDescriptor
    {
    public:
        typedef boost::shared_ptr<DistributionDescriptor> Ptr;

    public:
        virtual ~DistributionDescriptor() { }

        /**
         * @brief state Returns the first moment vector
         *
         * @return current first moment estimate (mean)
         */
        virtual DynamicVector& mean() = 0;

        /**
         * @brief @see DistributionDescriptor::mean()
         *
         * @return @see DistributionDescriptor::mean()
         */
        virtual const DynamicVector& mean() const = 0;

        /**
         * @brief covariance Returns second central moment matrix
         *
         * @return current second central moment matrix (covariance matrix)
         */
        virtual CovarianceMatrix& covariance() = 0;

        /**
         * @brief @see DistributionDescriptor::covariance()
         *
         * @return @see DistributionDescriptor::covariance()
         */
        virtual const CovarianceMatrix& covariance() const = 0;

        /**
         * @brief time Returns the timestamp at which the state was updated
         *
         * @return timestamp of update
         */
        virtual double timestamp() const = 0;

        /**
         * @brief dimension Returns the state dimension
         *
         * @return state dimension
         */
        virtual int dimension() const = 0;

        /**
         * @brief sigmaPoints Returns the sigma points representing the state
         *
         * @return sigmaPoints of the state
         */
        virtual SigmaPointMatrix& sigmaPoints() = 0;

        /**
         * @brief @see DistributionDescriptor::sigmaPoints()
         *
         * @return @see DistributionDescriptor::sigmaPoints()
         */
        virtual const SigmaPointMatrix& sigmaPoints() const = 0;

        /**
         * @brief state Sets the state vector
         *
         * @param _state New state
         */
        virtual void mean(const DynamicVector& _state) = 0;

        /**
         * @brief covariance Sets the LLT UKF state covariance  matrix
         *
         * @param _covariance New state covariance matrix
         */
        virtual void covariance(const CovarianceMatrix& _covariance) = 0;
        /**
         * @brief timestamp Updates the state timestamp
         *
         * @param _time New update timestamp
         */
        virtual void timestamp(double _time) = 0;
    };
}

#endif
