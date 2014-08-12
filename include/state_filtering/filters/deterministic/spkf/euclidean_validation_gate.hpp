
#ifndef FILTERS_SPKF_EUCLIDEAN_VALIDATION_GATE_HPP
#define FILTERS_SPKF_EUCLIDEAN_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/validation_gate_base.hpp>

namespace distributions
{
    class EuclideanValidationGate:
            public ValidationGate
    {
    public:
        typedef boost::shared_ptr<EuclideanValidationGate> Ptr;

    public:
        EuclideanValidationGate(double distance_threshold, double invalidating_variance = 1.e-12);

        /**
         * @brief validate Passthrough all observation using a euclidean distance threshold
         *
         * @see ValidationGate::validate()
         */
        virtual void validate(const distributions::DynamicVector& residual,
                              const distributions::DynamicVector& invCov,
                              distributions::DynamicVector& validInvCov);

    protected:
        double distance_threshold_;
    };
}

#endif
