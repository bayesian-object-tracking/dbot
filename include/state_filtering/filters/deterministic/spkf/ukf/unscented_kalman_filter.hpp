

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
                            DistributionDescriptor& observationDesc,
                            const Observer::ObservationVector& observation,
                            DistributionDescriptor& updatedStateDesc);

        /**
         * @see SpkfInternals::
         */
        virtual void onBeginUpdate(DistributionDescriptor &updatedStatem,
                                   DistributionDescriptor& observationDesc);

        /**
         * @see SpkfInternals::
         */
        virtual void onFinalizeUpdate(DistributionDescriptor& observationDesc,
                                      DistributionDescriptor& updatedState);

    public:/* UKF internal specifics */
        /**
         *
         */
        virtual void applyObservationUncertainty(DistributionDescriptor& observationDesc);

        /**
         *
         */
        virtual void computeCrossCovariance(const SigmaPointMatrix& predictedStateSigmaPoints,
                                           const SigmaPointMatrix& observationSigmaPoints,
                                           const DynamicVector& stateMean,
                                           const ObservationVector& observationMean,
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
                            const DistributionDescriptor& observationDesc,
                            const DynamicMatrix& kalmanGain,
                            const Observer::ObservationVector& observation,
                            DistributionDescriptor& updatedStateDesc);
    };
}

#endif
