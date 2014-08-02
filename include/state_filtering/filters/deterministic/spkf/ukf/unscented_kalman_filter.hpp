

#ifndef FILTERS_SPKF_UKF_UNSCENTED_KALMAN_FILTER_HPP
#define FILTERS_SPKF_UKF_UNSCENTED_KALMAN_FILTER_HPP

#include <Eigen/Eigen>

#include <boost/shared_ptr.hpp>

#include <state_filtering/filters/deterministic/spkf/sigma_point_kalman_filter.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/unscented_kalman_filter_base.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/unscented_transform.hpp>

namespace distributions
{
    class UkfInternals:
            public UkfInternalsBase
    {
    public:
        typedef boost::shared_ptr<UkfInternals> Ptr;

    public: /* SpkfInternals interface */
        /**
         * @see SpkfInternals::
         */
        virtual void update(DistributionDescriptor& predictedStateDesc,
                            DistributionDescriptor& measurementDesc,
                            const MeasurementModel::MeasurementVector& measurement,
                            DistributionDescriptor& updatedStateDesc);

        /**
         * @see SpkfInternals::
         */
        virtual void onBeginUpdate(DistributionDescriptor &updatedStatem,
                                   DistributionDescriptor& measurementDesc);

        /**
         * @see SpkfInternals::
         */
        virtual void onFinalizeUpdate(DistributionDescriptor& measurementDesc,
                                      DistributionDescriptor& updatedState);

    public:/* UKF internal specifics */
        /**
         *
         */
        virtual void applyMeasurementUncertainty(DistributionDescriptor& measurementDesc);

        /**
         *
         */
        virtual void computeCrossCovariance(const SigmaPointMatrix& predictedStateSigmaPoints,
                                           const SigmaPointMatrix& measurementSigmaPoints,
                                           const DynamicVector& stateMean,
                                           const MeasurementVector& measurementMean,
                                           CovarianceMatrix& measureStateCovariance);

        /**
         *
         */
        virtual void computeKalmanGain(const CovarianceMatrix &measurementCovariance,
                                       const CovarianceMatrix& measureStateCovariance,
                                       DynamicMatrix& kalmanGain);

        /**
         *
         */
        virtual void update(const DistributionDescriptor& predictedStateDesc,
                            const DistributionDescriptor& measurementDesc,
                            const DynamicMatrix& kalmanGain,
                            const MeasurementModel::MeasurementVector& measurement,
                            DistributionDescriptor& updatedStateDesc);
    };
}

#endif
