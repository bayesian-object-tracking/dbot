
#ifndef FILTERS_SPKF_ZERO_VALIDATION_GATE_HPP
#define FILTERS_SPKF_ZERO_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/validation_gate_base.hpp>

namespace distributions
{
    class ZeroValidationGate:
            public ValidationGate
    {
    public:
        typedef boost::shared_ptr<ZeroValidationGate> Ptr;

    public:
        /**
         * @brief validate Passthrough all measurement without validation
         *
         * @see ValidationGate::validate()
         */
        virtual void validate(const distributions::DynamicVector& residual,
                              const distributions::DynamicVector& invCov,
                              distributions::DynamicVector& validInvCov){ }
    };
}

#endif
