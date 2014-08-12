
#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include <Eigen/Eigen>

#include <state_filtering/filters/deterministic/spkf/ukf/ukf_distribution_descriptor.hpp>
#include <state_filtering/filters/deterministic/spkf/ukf/factorized_unscented_kalman_filter.hpp>

using namespace distributions;

/* ============================================================================================== */
/* == SpkfInternals interface implementations =================================================== */
/* ============================================================================================== */

void FactorizedUkfInternals::onBeginUpdate(DistributionDescriptor& updatedState,
                                  DistributionDescriptor& observationDesc)
{
    //ROS_ASSERT_MSG(Observer_->noiseCovariance().cols() == 1, "Currently this version works only with diagonal covariance matrix represented as a vector.");

    observationDesc.sigmaPoints().mean(observationDesc.mean(), observationDesc.dimension(), 0);
}

void FactorizedUkfInternals::update(DistributionDescriptor& predictedStateDesc,
                           DistributionDescriptor& observationDesc,
                           const Observer::ObservationVector& observation,
                           DistributionDescriptor& updatedStateDesc)
{    
    /*
    DynamicMatrix C;
    DynamicVector invR;
    DynamicMatrix zmX;
    DynamicMatrix zmY;
    DynamicVector invW;
    DynamicVector predictedState = predictedStateDesc.mean();
    DynamicVector predictedObservation = observationDesc.mean();
    */    

    predictedState = predictedStateDesc.mean();
    predictedObservation = observationDesc.mean();
    innovation = observation - predictedObservation;

    ROS_ASSERT(observation.rows() == predictedObservation.rows());


    predictedStateDesc.sigmaPoints().zeroMeanSigmaPointMatrix(predictedState, zmX);
    observationDesc.sigmaPoints().zeroMeanSigmaPointMatrix(predictedObservation, zmY);

    inverteDiagonalMatrix(Observer_->noiseCovariance(predictedObservation), invR);
    inverteDiagonalMatrix(predictedStateDesc.sigmaPoints().covarianceWeights(), invW);

    validationGate()->validate(innovation, invR, invR);

    computeMatrixC(zmY, invW, invR, C);  

    DynamicVector correction = zmX * C * zmY.transpose() * invR.asDiagonal() * innovation;  

    updatedStateDesc.mean() = predictedState + correction;
    updatedStateDesc.covariance() = zmX * C * zmX.transpose();
}

void FactorizedUkfInternals::onFinalizeUpdate(DistributionDescriptor& observationDesc,
                                     DistributionDescriptor& updatedState)
{
    /* nothing required here */
}

/* ============================================================================================== */
/* == Factorized form UKF specifics implementation ============================================== */
/* ============================================================================================== */
void FactorizedUkfInternals::inverteDiagonalMatrix(const DynamicVector& R, DynamicVector& invR)
{
    ROS_ASSERT(R.cols() == 1);

    invR.resize(R.rows(), 1);

    for (int i = 0; i < R.rows(); i++)
    {
        invR(i, 0) = 1./R(i, 0);
    }
}

void FactorizedUkfInternals::computeMatrixC(const DynamicMatrix& zmY,
                                            const DynamicVector& invW,
                                            const DynamicMatrix& invR,
                                            DynamicMatrix& innovationMatrix)
{
    innovationMatrix = DynamicMatrix::Identity(zmY.cols(), zmY.cols());
    innovationMatrix *= invW.asDiagonal();
    innovationMatrix += zmY.transpose() * invR.asDiagonal() * zmY;
    innovationMatrix = innovationMatrix.inverse();
}
