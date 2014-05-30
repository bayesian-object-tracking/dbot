
#ifndef FILTERS_SPKF_ZERO_VALIDATION_GATE_HPP
#define FILTERS_SPKF_ZERO_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include "filters/spkf/types.hpp"
#include "filters/spkf/validation_gate_base.hpp"

namespace filter
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
        virtual void validate(const filter::DynamicVector& residual,
                              const filter::DynamicVector& invCov,
                              filter::DynamicVector& validInvCov){ }
    };
}

#endif
