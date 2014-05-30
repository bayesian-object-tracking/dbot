
#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include <Eigen/Eigen>

#include "filters/spkf/ukf/ukf_distribution_descriptor.hpp"
#include "filters/spkf/ukf/unscented_kalman_filter.hpp"

using namespace filter;

/* ============================================================================================== */
/* == SpkfInternals interface implementations =================================================== */
/* ============================================================================================== */

void UkfInternals::onBeginUpdate(DistributionDescriptor& updatedState,
                                 DistributionDescriptor& measurementDesc)
{
    sigmaPointTransform()->backward(measurementDesc.sigmaPoints(),
                                    measurementDesc);

    applyMeasurementUncertainty(measurementDesc);
}

void UkfInternals::update(DistributionDescriptor& predictedStateDesc,
                          DistributionDescriptor& measurementDesc,
                          const MeasurementModel::MeasurementVector& measurement,
                          DistributionDescriptor& updatedStateDesc)
{
    //DistributionDescriptor::Ptr measurementDesc = createDistributionDescriptor();
    CovarianceMatrix measureStateCovariance;
    DynamicMatrix kalmanGain;

    computeCrossCovariance(predictedStateDesc.sigmaPoints(),
                           measurementDesc.sigmaPoints(),
                           predictedStateDesc.mean(),
                           measurementDesc.mean(),
                           measureStateCovariance);

    computeKalmanGain(measurementDesc.covariance(),
                                   measureStateCovariance,
                                   kalmanGain);

    update(predictedStateDesc, measurementDesc, kalmanGain, measurement, updatedStateDesc);
}

void UkfInternals::onFinalizeUpdate(DistributionDescriptor& measurementDesc,
                                    DistributionDescriptor& updatedState)
{

}

/* ============================================================================================== */
/* == Canonical UKF specifics implementation ==================================================== */
/* ============================================================================================== */

void UkfInternals::applyMeasurementUncertainty(DistributionDescriptor& measurementDesc)
{
    // apply noise additive noise on measurement model with additive noise
    switch (measurementModel_->modelType())
    {
    case distr::Linear:
    case distr::NonLinearWithAdditiveNoise:
    {        
        CovarianceMatrix& covariance = measurementDesc.covariance();

        ROS_ASSERT(covariance.rows() == measurementModel_->noiseCovariance(measurementDesc.mean()).rows());

        ROS_ASSERT(covariance.rows() == covariance.cols());

        if (measurementModel_->noiseCovariance(measurementDesc.mean()).cols() == 1)
        {
            covariance += measurementModel_->noiseCovariance(measurementDesc.mean()).asDiagonal();
        }
        else
        {
            covariance += measurementModel_->noiseCovariance(measurementDesc.mean());
        }
    }break;

    case distr::NonLinearNonAdditiveNoise:
    default:
        // noise already included
        break;
    }
}

void UkfInternals::computeCrossCovariance(const SigmaPointMatrix& predictedStateSigmaPoints,
                                    const SigmaPointMatrix& measurementSigmaPoints,
                                    const DynamicVector& stateMean,
                                    const MeasurementVector& measurementMean,
                                    CovarianceMatrix& measureStateCovariance)
{
    measureStateCovariance.setZero(stateMean.rows(), measurementMean.rows());

    ProcessModel::StateVector zeroMeanState;
    MeasurementModel::MeasurementVector zeroMeanMeasurement;

    for (int i = 0; i < predictedStateSigmaPoints.cols(); i++)
    {
        zeroMeanState = predictedStateSigmaPoints.col(i) - stateMean;
        zeroMeanMeasurement = measurementSigmaPoints.col(i) - measurementMean;

        measureStateCovariance +=
              measurementSigmaPoints.covWeight(i) * zeroMeanState * zeroMeanMeasurement.transpose();
    }
}

void UkfInternals::computeKalmanGain(const CovarianceMatrix& measurementCovariance,
                                     const CovarianceMatrix& measureStateCovariance,
                                     DynamicMatrix& kalmanGain)
{
    kalmanGain = measureStateCovariance * measurementCovariance.inverse();
}

void UkfInternals::update(const DistributionDescriptor& predictedStateDesc,
                          const DistributionDescriptor& measurementDesc,
                          const DynamicMatrix & kalmanGain,
                          const MeasurementModel::MeasurementVector& measurement,
                          DistributionDescriptor& updatedStateDesc)
{
    updatedStateDesc.mean(
                predictedStateDesc.mean() +
                kalmanGain * (measurement - measurementDesc.mean()));

    updatedStateDesc.covariance(
                predictedStateDesc.covariance() -
                kalmanGain * measurementDesc.covariance() * kalmanGain.transpose());

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
            + kalmanGain * measurementModel()->noiseCovariance() * kalmanGain.transpose();
    */
}
