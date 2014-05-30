
#ifndef FILTERS_SPKF_VALIDATION_GATE_HPP
#define FILTERS_SPKF_VALIDATION_GATE_HPP

#include <boost/shared_ptr.hpp>

#include <state_filtering/filter/kalman/spkf/types.hpp>

namespace filter
{
    class ValidationGate
    {
    public:
        typedef boost::shared_ptr<ValidationGate> Ptr;

    public:
        ValidationGate(double invalidating_variance = 1.0e-12):
            invalidating_variance_(invalidating_variance) { }

        /**
         * @brief ValidationGate Overridable default descructor
         */
        virtual ~ValidationGate() { }

        /**
         * @brief validate Determines wheather to pass a measurement or not
         *
         * @param residual          Residual error or innovation value
         * @param invCov            Inverse of the covariance matrix
         * @param validInvCov       Updated inverse covariance matrix including rejected measurements
         */
        virtual void validate(const filter::DynamicVector& residual,
                              const filter::DynamicVector& invCov,
                              filter::DynamicVector& validInvCov) = 0;

    protected:
        double invalidating_variance_;
    };
}

#endif
