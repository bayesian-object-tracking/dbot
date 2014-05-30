
#ifndef FILTERS_SPKF_UKF_UKF_DISTRIBUTION_DESCRIPTOR_HPP
#define FILTERS_SPKF_UKF_UKF_DISTRIBUTION_DESCRIPTOR_HPP

#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include <Eigen/Eigen>

#include <vector>

#include <boost/shared_ptr.hpp>

#include <state_filtering/filter/kalman/spkf/sigma_point_matrix.hpp>
#include <state_filtering/filter/kalman/spkf/sigma_point_kalman_filter.hpp>
#include <state_filtering/filter/kalman/spkf/ukf/unscented_transform.hpp>

namespace filter
{
    class UkfDistributionDescriptor:
            public DistributionDescriptor
    {
    public:
        UkfDistributionDescriptor()
        {
        }

        virtual ~UkfDistributionDescriptor()
        {

        }

        /**
         * @see DistributionDescriptor::state()
         */
        virtual DynamicVector& mean()
        {
            return state_;
        }

        /**
         * @see DistributionDescriptor::const_mean()
         */
        virtual const DynamicVector& mean() const
        {
            return state_;
        }

        /**
         *
         * @see DistributionDescriptor::covariance()
         */
        virtual CovarianceMatrix& covariance()
        {
            return covariance_;
        }

        /**
         * @see DistributionDescriptor::const_covariance
         */
        virtual const CovarianceMatrix& covariance() const
        {
            return covariance_;
        }

        /**
         * @see StateDescriptor::timestamp()
         */
        virtual double timestamp() const
        {
            return timestamp_;
        }

        /**
         * @see StateDescriptor::dimension()
         */
        virtual int dimension() const
        {
            if (state_.rows() == 0)
            {
                sigmaPointsMatrix_.rows();
            }

            return state_.rows();
        }

        /**
         * @see StateDescriptor::sigmaPoints()
         */
        virtual SigmaPointMatrix& sigmaPoints()
        {
            return sigmaPointsMatrix_;
        }

        /**
         * @brief @see DistributionDescriptor::sigmaPoints()
         *
         * @return @see DistributionDescriptor::sigmaPoints()
         */
        virtual const SigmaPointMatrix& sigmaPoints() const
        {
            return sigmaPointsMatrix_;
        }

        /**
         * @see DistributionDescriptor::state(const StateType)
         */
        virtual void mean(const DynamicVector& _state)
        {
            state_ = _state;
        }

        /**
         * @brief covariance Sets the LLT UKF state covariance matrix
         *
         * @param _covariance New state covariance matrix
         */
        virtual void covariance(const CovarianceMatrix& _covariance)
        {
            covariance_ = _covariance;
        }

        /**
         * @see StateDescriptor::timestamp(double)
         */
        virtual void timestamp(double _timestamp)
        {
            timestamp_ = _timestamp;
        }

        /**
         * @brief segmentsDimensions
         *
         * @return Returns the segments list
         */
        virtual const std::vector<int>& segmentsDimensions() const
        {
            return segmentsDimensions_;
        }

        virtual void segmentsDimensions(const std::vector<int>& _segmentsDimensions)
        {
            int totalSegmentsDimension = 0;
            std::vector<int>::const_iterator it = _segmentsDimensions.begin();
            while(it != _segmentsDimensions.end())
            {
                totalSegmentsDimension += *it;

                ++it;
            }

            ROS_ASSERT(totalSegmentsDimension == state_.rows());

            segmentsDimensions_ = _segmentsDimensions;
        }

    protected:
        DynamicVector state_;
        CovarianceMatrix covariance_;
        SigmaPointMatrix sigmaPointsMatrix_;
        double timestamp_;
        std::vector<int> segmentsDimensions_;
    };
}

#endif
