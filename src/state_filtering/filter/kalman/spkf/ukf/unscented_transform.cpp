

#include "filters/spkf/ukf/unscented_transform.hpp"

using namespace filter;

UnscentedTransform::UnscentedTransform(double _alpha, double _beta, double _kappa):
    alpha_(_alpha),
    beta_(_beta),
    kappa_(_kappa)
{
}

UnscentedTransform::~UnscentedTransform()
{

}

void UnscentedTransform::forward(const DistributionDescriptor& distDesc,
                                 SigmaPointMatrix& sigmaPoints)
{
    // get state dimension and determine the number of samples
    int dimension = distDesc.dimension();
    SigmaPointMatrix::size_type numberOfSamples = numberOfSigmaPoints(dimension);

    double lambda = lambdaScalar(dimension);
    double gamma = gammaFactor(dimension);
    double sampleWeight = 1. / (2. * (double(dimension) + lambda));

    // sample 2n+1 points
    sigmaPoints.resize(numberOfSamples);

    // first sample is the mean
    sigmaPoints.point(0,
                      distDesc.mean(),
                      lambda / (double(dimension) + lambda),
                      lambda / (double(dimension) + lambda) + (1 - alpha_ * alpha_ + beta_));

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

void UnscentedTransform::backward(const SigmaPointMatrix &sigmaPoints,
                                  DistributionDescriptor& distDesc,
                                  int subDimension,
                                  int marginBegin)
{
    sigmaPoints.mean(distDesc.mean(), subDimension, marginBegin);
    sigmaPoints.covariance(distDesc.mean(), distDesc.covariance(), subDimension, marginBegin);
}

unsigned int UnscentedTransform::numberOfSigmaPoints(unsigned int dimension) const
{
    return 2 * dimension + 1;
}

double UnscentedTransform::lambdaScalar(unsigned int dimension)
{
    return alpha_ * alpha_ * (double(dimension) + kappa_) - double(dimension);
}

double UnscentedTransform::gammaFactor(unsigned int dimension)
{
    return std::sqrt((double(dimension) + lambdaScalar(dimension)));
}

double UnscentedTransform::alpha() const { return alpha_; }
double UnscentedTransform::beta() const { return beta_; }
double UnscentedTransform::kappa() const { return kappa_; }

void UnscentedTransform::alpha(double _alpha) { alpha_ = _alpha; }
void UnscentedTransform::beta(double _beta) { beta_ = _beta; }
void UnscentedTransform::kappa(double _kappa) { kappa_ = _kappa; }
