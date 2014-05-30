

#include "filters/spkf/ukf/kappa_unscented_transform.hpp"

using namespace filter;

KappaUnscentedTransform::KappaUnscentedTransform(double _kappa):
    kappa_(_kappa)
{
}

KappaUnscentedTransform::~KappaUnscentedTransform()
{

}

void KappaUnscentedTransform::forward(const DistributionDescriptor& distDesc,
                                 SigmaPointMatrix& sigmaPoints)
{
    // get state dimension and determine the number of samples
    int dimension = distDesc.dimension();
    SigmaPointMatrix::size_type numberOfSamples = numberOfSigmaPoints(dimension);

    double gamma = gammaFactor(dimension);
    double sampleWeight = 1 / (2. * (double(dimension) + kappa_));

    // sample 2n+1 points
    sigmaPoints.resize(numberOfSamples);

    // first sample is the mean
    sigmaPoints.point(0,
                      distDesc.mean(),
                      kappa_ / (double(dimension) + kappa_),
                      kappa_ / (double(dimension) + kappa_));

    CovarianceMatrix covarianceSqr = distDesc.covariance().llt().matrixL();
    covarianceSqr = covarianceSqr * gamma;
    DynamicVector pointShift;

    // sample the 2n points
    for (int i = 1; i <= dimension; i++)
    {
        pointShift = covarianceSqr.col(i - 1);

        sigmaPoints.point(i,
                          distDesc.mean() + pointShift,
                          sampleWeight,
                          sampleWeight);

        sigmaPoints.point(dimension + i,
                          distDesc.mean() - pointShift,
                          sampleWeight,
                          sampleWeight);
    }
}

void KappaUnscentedTransform::backward(const SigmaPointMatrix &sigmaPoints,
                                       DistributionDescriptor& distDesc,
                                       int subDimension,
                                       int marginBegin)
{
    sigmaPoints.mean(distDesc.mean(), subDimension, marginBegin);
    sigmaPoints.covariance(distDesc.mean(), distDesc.covariance(), subDimension, marginBegin);
}

unsigned int KappaUnscentedTransform::numberOfSigmaPoints(unsigned int dimension) const
{
    return 2 * dimension + 1;
}

double KappaUnscentedTransform::gammaFactor(unsigned int dimension)
{
    return std::sqrt((double(dimension) + kappa_));
}

double KappaUnscentedTransform::kappa() const { return kappa_; }

void KappaUnscentedTransform::kappa(double _kappa) { kappa_ = _kappa; }
