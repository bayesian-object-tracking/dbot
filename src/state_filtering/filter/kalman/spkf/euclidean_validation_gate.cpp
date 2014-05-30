
#include "filters/spkf/euclidean_validation_gate.hpp"

using namespace filter;

EuclideanValidationGate::EuclideanValidationGate(double distance_threshold,
                                                 double invalidating_variance):
    ValidationGate(invalidating_variance),
    distance_threshold_(distance_threshold)
{

}

void EuclideanValidationGate::validate(const DynamicVector& residual,
                                       const DynamicVector& invCov,
                                       DynamicVector& validInvCov)
{
    for (int i = 0; i < residual.rows(); i++)
    {
        if (!(std::abs(residual(i, 0)) < distance_threshold_))
        {
            validInvCov(i, 0) = invalidating_variance_;
        }
    }
}
