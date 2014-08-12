
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
                            DistributionDescriptor& observationDesc,
                            const Observer::ObservationVector& observation,
                            DistributionDescriptor& updatedStateDesc) = 0;

        virtual void predictObservation(const SigmaPointMatrix& predictedStateSigmaPoints,
                                        SigmaPointMatrix& observationSigmaPoints) = 0;

        virtual void onBeginPredict(const DistributionDescriptor& stateDesc,
                                    DistributionDescriptor& predictedStateDesc) = 0;

        virtual void onFinalizePredict(const DistributionDescriptor& stateDesc,
                                       DistributionDescriptor& predictedStateDesc) = 0;

        virtual void onBeginUpdate(DistributionDescriptor& updatedStatem,
                                   DistributionDescriptor& observationDesc) = 0;

        virtual void onFinalizeUpdate(DistributionDescriptor& observationDesc,
                                      DistributionDescriptor& updatedState) = 0;

        /**
         * Creates an instance of DistributionDescriptor
         */
        virtual DistributionDescriptor::Ptr createDistributionDescriptor() = 0;

        /**
         * Sets the used observation model
         *
         * @param Observer  Used observation model model
         */
        virtual void Observer(const Observer::Ptr& _Observer)
        {
            Observer_ = _Observer;
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
        virtual Observer::Ptr Observer()
        {
            return Observer_;
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
        Observer::Ptr Observer_;
        SigmaPointTransform::Ptr sigmaPointTransform_;
        ValidationGate::Ptr validationGate_;
    };
}

#endif
