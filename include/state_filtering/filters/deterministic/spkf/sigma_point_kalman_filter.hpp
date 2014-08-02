
#ifndef FILTERS_SPKF_SIGMA_POINT_KALMAN_FILTER_HPP
#define FILTERS_SPKF_SIGMA_POINT_KALMAN_FILTER_HPP

#include <boost/shared_ptr.hpp>

#include <Eigen/Eigen>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/sigma_point_transform.hpp>
#include <state_filtering/filters/deterministic/spkf/distribution_descriptor.hpp>
#include <state_filtering/filters/deterministic/spkf/validation_gate_base.hpp>

namespace distributions
{
    class SpkfInternals
    {
    public:
        typedef boost::shared_ptr<SpkfInternals> Ptr;

    public:
        virtual void process(DistributionDescriptor& currentStateDesc,
                             const ProcessModel::InputType& controlInput,
                             const double deltaTime,
                             DistributionDescriptor& predictedStateDesc) = 0;

        virtual void update(DistributionDescriptor& predictedStateDesc,
                            DistributionDescriptor& measurementDesc,
                            const MeasurementModel::MeasurementVector& measurement,
                            DistributionDescriptor& updatedStateDesc) = 0;

        virtual void predictMeasurement(const SigmaPointMatrix& predictedStateSigmaPoints,
                                        SigmaPointMatrix& measurementSigmaPoints) = 0;

        virtual void onBeginPredict(const DistributionDescriptor& stateDesc,
                                    DistributionDescriptor& predictedStateDesc) = 0;

        virtual void onFinalizePredict(const DistributionDescriptor& stateDesc,
                                       DistributionDescriptor& predictedStateDesc) = 0;

        virtual void onBeginUpdate(DistributionDescriptor& updatedStatem,
                                   DistributionDescriptor& measurementDesc) = 0;

        virtual void onFinalizeUpdate(DistributionDescriptor& measurementDesc,
                                      DistributionDescriptor& updatedState) = 0;

        /**
         * Creates an instance of DistributionDescriptor
         */
        virtual DistributionDescriptor::Ptr createDistributionDescriptor() = 0;

        /**
         * Sets the used observation model
         *
         * @param measurementModel  Used measurement model model
         */
        virtual void measurementModel(const MeasurementModel::Ptr& _measurementModel)
        {
            measurementModel_ = _measurementModel;
        }

        /**
         * Sets the process model
         *
         * @param processModel Used process model
         */
        virtual void processModel(const ProcessModelPtr& _processModel)
        {
            processModel_ = _processModel;
        }

        /**
         * @brief stateSigmaPointSampler Sets the state sigma point sampler
         *
         * @param _stateSigmaPointSampler Filter state sigma point sampled used be the prediction
         *                                step
         */
        virtual void sigmaPointTransform(SigmaPointTransform::Ptr _sigmaPointTransform)
        {
            sigmaPointTransform_ = _sigmaPointTransform;
        }

        /**
         * @brief validationGate Sets a validation gate
         */
        virtual void validationGate(ValidationGate::Ptr validationGate)
        {
            validationGate_ = validationGate;
        }

        /**
         * Returns the used observation model
         */
        virtual MeasurementModel::Ptr measurementModel()
        {
            return measurementModel_;
        }

        /**
         * Returns the process model
         */
        virtual ProcessModelPtr processModel()
        {
            return processModel_;
        }

        /**
         * @brief stateSigmaPointSampler Returns the state sigma point sampler
         *
         * @return Point of the state sigma point sampler
         */
        virtual SigmaPointTransform::Ptr sigmaPointTransform()
        {
            return sigmaPointTransform_;
        }

        /**
         * @brief validationGate Returns the validation gate
         *
         * @return Pointer to the validation gate
         */
        virtual ValidationGate::Ptr validationGate()
        {
            return validationGate_;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        SpkfInternals::Ptr internalFilter;
        ProcessModelPtr processModel_;
        MeasurementModel::Ptr measurementModel_;
        SigmaPointTransform::Ptr sigmaPointTransform_;
        ValidationGate::Ptr validationGate_;
    };
}

#endif
