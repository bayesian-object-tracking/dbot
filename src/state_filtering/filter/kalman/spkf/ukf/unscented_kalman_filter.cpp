
#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include <Eigen/Eigen>

#include <state_filtering/filters/deterministic/spkf/ukf/ukf_distribution_descriptor.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/unscented_kalman_filter.hpp>

using namespace distributions;

/* ============================================================================================== */
/* == SpkfInternals interface implementations =================================================== */
/* ============================================================================================== */

void UkfInternals::onBeginUpdate(DistributionDescriptor& updatedState,
                                 DistributionDescriptor& observationDesc)
{
    sigmaPointTransform()->backward(observationDesc.sigmaPoints(),
                                    observationDesc);

    applyObservationUncertainty(observationDesc);
}

void UkfInternals::update(DistributionDescriptor& predictedStateDesc,
                          DistributionDescriptor& observationDesc,
                          const Observer::ObservationVector& observation,
                          DistributionDescriptor& updatedStateDesc)
{
    //DistributionDescriptor::Ptr observationDesc = createDistributionDescriptor();
    CovarianceMatrix measureStateCovariance;
    DynamicMatrix kalmanGain;

    computeCrossCovariance(predictedStateDesc.sigmaPoints(),
                           observationDesc.sigmaPoints(),
                           predictedStateDesc.mean(),
                           observationDesc.mean(),
                           measureStateCovariance);

    computeKalmanGain(observationDesc.covariance(),
                                   measureStateCovariance,
                                   kalmanGain);

    update(predictedStateDesc, observationDesc, kalmanGain, observation, updatedStateDesc);
}

void UkfInternals::onFinalizeUpdate(DistributionDescriptor& observationDesc,
                                    DistributionDescriptor& updatedState)
{

}

/* ============================================================================================== */
/* == Canonical UKF specifics implementation ==================================================== */
/* ============================================================================================== */

void UkfInternals::applyObservationUncertainty(DistributionDescriptor& observationDesc)
{
    // apply noise additive noise on observation model with additive noise
    switch (Observer_->modelType())
    {
    case distr::Linear:
    case distr::NonLinearWithAdditiveNoise:
    {        
        CovarianceMatrix& covariance = observationDesc.covariance();

        ROS_ASSERT(covariance.rows() == Observer_->noiseCovariance(observationDesc.mean()).rows());

        ROS_ASSERT(covariance.rows() == covariance.cols());

        if (Observer_->noiseCovariance(observationDesc.mean()).cols() == 1)
        {
            covariance += Observer_->noiseCovariance(observationDesc.mean()).asDiagonal();
        }
        else
        {
            covariance += Observer_->noiseCovariance(observationDesc.mean());
        }
    }break;

    case distr::NonLinearNonAdditiveNoise:
    default:
        // noise already included
        break;
    }
}

void UkfInternals::computeCrossCovariance(const SigmaPointMatrix& predictedStateSigmaPoints,
                                    const SigmaPointMatrix& observationSigmaPoints,
                                    const DynamicVector& stateMean,
                                    const ObservationVector& observationMean,
                                    CovarianceMatrix& measureStateCovariance)
{
    measureStateCovariance.setZero(stateMean.rows(), observationMean.rows());

    ProcessModel::StateType zeroMeanState;
    Observer::ObservationVector zeroMeanObservation;

    for (int i = 0; i < predictedStateSigmaPoints.cols(); i++)
    {
        zeroMeanState = predictedStateSigmaPoints.col(i) - stateMean;
        zeroMeanObservation = observationSigmaPoints.col(i) - observationMean;

        measureStateCovariance +=
              observationSigmaPoints.covWeight(i) * zeroMeanState * zeroMeanObservation.transpose();
    }
}

void UkfInternals::computeKalmanGain(const CovarianceMatrix& observationCovariance,
                                     const CovarianceMatrix& measureStateCovariance,
                                     DynamicMatrix& kalmanGain)
{
    kalmanGain = measureStateCovariance * observationCovariance.inverse();
}

void UkfInternals::update(const DistributionDescriptor& predictedStateDesc,
                          const DistributionDescriptor& observationDesc,
                          const DynamicMatrix & kalmanGain,
                          const Observer::ObservationVector& observation,
                          DistributionDescriptor& updatedStateDesc)
{
    updatedStateDesc.mean(
                predictedStateDesc.mean() +
                kalmanGain * (observation - observationDesc.mean()));

    updatedStateDesc.covariance(
                predictedStateDesc.covariance() -
                kalmanGain * observationDesc.covariance() * kalmanGain.transpose());

    /*
     * Joseph form
     * /
    Eigen::Matrix<double, DYNAMIC, DYNAMIC> joseph;
    Eigen::Matrix<double, DYNAMIC, DYNAMIC> H;
    Eigen::Matrix<double, DYNAMIC, DYNAMIC> I;
    H =  predictedState.covariance().inverse() * measureStateCovariance;
    I = Eigen::Matrix<double, DYNAMIC, DYNAMIC>::Identity(kalmanGain.rows(), H.cols());
    joseph = I - kalmanGain * H;
    updatedCovariance = joseph * predictedState.covariance() * joseph.transpose()
            + kalmanGain * Observer()->noiseCovariance() * kalmanGain.transpose();
    */
}
