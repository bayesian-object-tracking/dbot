

#ifndef FILTERS_SPKF_UKF_FACTORIZED_UNSCENTED_KALMAN_FILTER_HPP
#define FILTERS_SPKF_UKF_FACTORIZED_UNSCENTED_KALMAN_FILTER_HPP

#include <Eigen/Eigen>

#include <boost/shared_ptr.hpp>

#include <state_filtering/filters/deterministic/spkf/sigma_point_kalman_filter.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/unscented_kalman_filter_base.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/unscented_transform.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/ukf_distribution_descriptor.hpp>

namespace distributions
{
    /**
     * @brief The PUkfInternals implements a version of the UKF using the Sherman-Morrison-Woodbury
     * identity to express the update step in a parallel way.
     */
    class FactorizedUkfInternals:
            public UkfInternalsBase
    {
    public:
        typedef boost::shared_ptr<FactorizedUkfInternals> Ptr;

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

    public:/* P-UKF internal specifics */
        virtual void inverteDiagonalMatrix(const DynamicVector& R, DynamicVector& invR);

        virtual void computeMatrixC(const DynamicMatrix& zmY,
                                    const DynamicVector& invW,
                                    const DynamicMatrix& invR,
                                    DynamicMatrix& innovationMatrix);

    public:
        DynamicMatrix C;
        DynamicVector invR;
        DynamicMatrix zmX;
        DynamicMatrix zmY;
        DynamicVector invW;
        DynamicVector predictedState;
        DynamicVector predictedMeasurement;
        DynamicVector innovation;

        double sigma;
    };
}

#endif
