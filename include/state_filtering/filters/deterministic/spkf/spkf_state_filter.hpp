
#ifndef FILTERS_SPKF_SPKF_STATE_FILTER_HPP
#define FILTERS_SPKF_SPKF_STATE_FILTER_HPP

#include <boost/shared_ptr.hpp>

#include <Eigen/Eigen>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/sigma_point_transform.hpp>
#include <state_filtering/filters/deterministic/spkf/distribution_descriptor.hpp>
#include <state_filtering/filters/deterministic/spkf/sigma_point_kalman_filter.hpp>

namespace distributions
{
    /**
     * @class SpkfStateFilter
     * @brief Sigma Point Kalman State Filter Interface
     *
     * Sigma Point Kalman State Filter Interface for object state estimation and non-linear dynamic
     * systems with non-additive noise.
     */
    class SpkfStateFilter
    {
    public:
        typedef boost::shared_ptr<SpkfStateFilter> Ptr;

    public:
        /**
         * Filter constructor
         */
        SpkfStateFilter(SpkfInternals::Ptr _internals);

        /**
         * Filter destructor
         */
        virtual ~SpkfStateFilter();

        /**
         * @brief predict UKF prediction step returning the predicted distribution.
         *
         * UKF prediction step returning the predicted distribution using the inverse unscented
         * transform.
         *
         * @param [in]  currentState                Current state
         * @param [in]  control                     Input control vector
         * @param [in]  deltaTime                   Time passed since last prediction
         * @param [out] predictedState              Predicted state
         * @param [out] predictedStateSigmaPoints   Predicted state sigma points
         */
        virtual void predict(const DistributionDescriptor &stateDesc,
                             const ProcessModel::ControlVector& control,
                             double deltaTime,
                             DistributionDescriptor& predictedStateDesc);

        /**
         * @brief update UKF update step using a observation
         *
         * @param [in]  observation             Observation
         * @param [in]  predictedState          Predicted state distribution
         * @param [out] updatedState            Updated state after incorporating the observation
         */
        virtual void update(const ObservationVector& observation,
                            DistributionDescriptor& predictedState,
                            DistributionDescriptor& updatedState);

    public: /* Getters and DI setters */
        /**
         * Returns the used observation model
         */
        virtual Observer::Ptr Observer();

        /**
         * Returns the process model
         */
        virtual ProcessModel::Ptr processModel();

        /**
         * @brief sigmaPointTransform Returns the state sigma point sampler
         *
         * @return Point of the state sigma point sampler
         */
        virtual SigmaPointTransform::Ptr sigmaPointTransform();

        /**
         * Sets the used observation model
         *
         * @param Observer  Used observation model model
         */
        virtual void Observer(const Observer::Ptr& Observer_);

        /**
         * Sets the process model
         *
         * @param processModel Used process model
         */
        virtual void processModel(const ProcessModel::Ptr& processModel_);

        /**
         * @brief sigmaPointTransform Sets the state sigma point sampler
         *
         * @param _sigmaPointTransform Filter state sigma point sampled used be the prediction
         *                                step
         */
        virtual void sigmaPointTransform(SigmaPointTransform::Ptr _sigmaPointTransform);

        /**
         * @brief validationGate Sets a validation gate
         */
        virtual void validationGate(ValidationGate::Ptr validationGate);

        /**
         * @brief internals returns the filter internal implementation
         *
         * @return pointer to the algorithm internals
         */
        virtual SpkfInternals::Ptr internals();

        /**
         * @brief augmentedStateDescriptor Returns a reference to the augmented state descriptor
         *
         * Returns a reference to the augmented state descriptor used internally. This provides
         * access to intermediate results.
         *
         * @return
         */
        virtual const DistributionDescriptor& augmentedStateDescriptor() const;

        /**
         * @brief observationDescriptor Returns the observation descriptor
         *
         * Returns a reference to the observation descriptor used internally. This provides
         * access to intermediate results.
         *
         * @return
         */
        virtual const DistributionDescriptor& observationDescriptor() const;

        /**
         * @brief validationGate Returns the validation gate
         *
         * @return Pointer to the validation gate
         */
        virtual ValidationGate::Ptr validationGate();

    protected:
        SpkfInternals::Ptr internals_;
        DistributionDescriptor::Ptr augmentedStateDesc;
        DistributionDescriptor::Ptr observationDesc;        
    };
}

#endif
