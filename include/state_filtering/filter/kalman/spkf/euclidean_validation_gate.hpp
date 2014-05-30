
#ifndef FILTERS_SPKF_EUCLIDEAN_VALIDATION_GATE_HPP
#define FILTERS_SPKF_EUCLIDEAN_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include "filters/spkf/types.hpp"
#include "filters/spkf/validation_gate_base.hpp"

namespace filter
{
    class EuclideanValidationGate:
            public ValidationGate
    {
    public:
        typedef boost::shared_ptr<EuclideanValidationGate> Ptr;

    public:
        EuclideanValidationGate(double distance_threshold, double invalidating_variance = 1.e-12);

        /**
         * @brief validate Passthrough all measurement using a euclidean distance threshold
         *
         * @see ValidationGate::validate()
         */
        virtual void validate(const filter::DynamicVector& residual,
                              const filter::DynamicVector& invCov,
                              filter::DynamicVector& validInvCov);

    protected:
        double distance_threshold_;
    };
}

#endif
