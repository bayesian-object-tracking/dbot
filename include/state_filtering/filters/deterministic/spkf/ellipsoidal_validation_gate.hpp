
#ifndef FILTERS_SPKF_ELLIPSOIDAL_VALIDATION_GATE_HPP
#define FILTERS_SPKF_ELLIPSOIDAL_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/validation_gate_base.hpp>

namespace distributions
{
    class EllipsoidalValidationGate:
            public ValidationGate
    {
    public:
        typedef boost::shared_ptr<EllipsoidalValidationGate> Ptr;

    public:
        EllipsoidalValidationGate(double acceptance_prob, double invalidating_variance_ = 1.e-12);

        /**
         * @brief validate Passthrough all measurement using a euclidean distance threshold
         *
         * @see ValidationGate::validate()
         */
        virtual void validate(const distributions::DynamicVector& residual,
                              const distributions::DynamicVector& invCov,
                              distributions::DynamicVector& validInvCov);

    protected:
        double acceptance_prob_;
    };
}

#endif
