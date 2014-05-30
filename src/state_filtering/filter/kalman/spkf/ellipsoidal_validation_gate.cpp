
#include "filters/spkf/ellipsoidal_validation_gate.hpp"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>

using namespace filter;

EllipsoidalValidationGate::EllipsoidalValidationGate(double acceptance_prob,
                                                     double invalidating_variance):
    ValidationGate(invalidating_variance),
    acceptance_prob_(acceptance_prob)
{

}

void EllipsoidalValidationGate::validate(const DynamicVector& residual,
                                       const DynamicVector& invCov,
                                       DynamicVector& validInvCov)
{
    boost::math::inverse_chi_squared chi2(1);

    double d2;
    int n_measurements = residual.rows();

    for (int i = 0; i < n_measurements; i++)
    {
        d2 = residual(i, 0) * invCov(i, 0) * residual(i, 0);

        if (boost::math::cdf(chi2, d2) > acceptance_prob_)
        {
            validInvCov(i, 0) = invalidating_variance_;
        }
    }
}
