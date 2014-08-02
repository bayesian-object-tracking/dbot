
#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include <Eigen/Eigen>

#include <state_filtering/filter/kalman/spkf/ukf/ukf_distribution_descriptor.hpp>
#include <state_filtering/filter/kalman/spkf/ukf/unscented_kalman_filter_base.hpp>

#include <state_filtering/tools/macros.hpp>

//#include "distributions/gaussian_distribution.hpp>
#include <state_filtering/tools/helper_functions.hpp>

using namespace filter;
using namespace Eigen;

/* ============================================================================================== */
/* == SpkfInternals interface implementations =================================================== */
/* ============================================================================================== */

void UkfInternalsBase::onBeginPredict(const DistributionDescriptor &stateDesc,
                                      DistributionDescriptor& augmentedStateDesc)
{
    // Depending on the model type the state may has to be augmented with the process noise. This
    // would be required e.g. for a non-linear dynamic systems with non-additive noise.
    augmentFilteringState(stateDesc, augmentedStateDesc);

    INIT_PROFILING
    sigmaPointTransform()->forward(augmentedStateDesc, augmentedStateDesc.sigmaPoints());
    MEASURE(">>>>>>>>>>>>>> Sigma point sampling")
}

void UkfInternalsBase::process(DistributionDescriptor& currentStateDesc,
                                   const ProcessModel::PerturbationType &controlInput,
                                   const double deltaTime,
                                   DistributionDescriptor& predictedStateDesc)
{
    UkfDistributionDescriptor& ukfCurrentStateDesc =
            (UkfDistributionDescriptor&) currentStateDesc;

    int dimension = currentStateDesc.mean().rows();

    if (ukfCurrentStateDesc.segmentsDimensions().size() > 0)
    {
        dimension = ukfCurrentStateDesc.segmentsDimensions().at(0);
    }

    // prediction using the process model
    process(currentStateDesc.sigmaPoints(),
            controlInput,
            deltaTime,
            predictedStateDesc.sigmaPoints(),
            dimension);
}

void UkfInternalsBase::onFinalizePredict(const DistributionDescriptor& stateDesc,
                                         DistributionDescriptor& predictedStateDesc)
{
    sigmaPointTransform()->backward(predictedStateDesc.sigmaPoints(),
                                    predictedStateDesc,
                                    stateDesc.dimension());
}

DistributionDescriptor::Ptr UkfInternalsBase::createDistributionDescriptor()
{
    return DistributionDescriptor::Ptr(new UkfDistributionDescriptor());
}

/* ============================================================================================== */
/* == UKF commonalities specifics implementation ================================================ */
/* ============================================================================================== */

void UkfInternalsBase::augmentFilteringState(const DistributionDescriptor& stateDesc,
                                         DistributionDescriptor& augmentedStateDesc)
{
    int stateDimension = stateDesc.dimension();

    UkfDistributionDescriptor& ukfAugmentedStateDesc =
            (UkfDistributionDescriptor&) augmentedStateDesc;

    DynamicVector& augmentedState = ukfAugmentedStateDesc.mean();
    CovarianceMatrix& augmentedCovariance = ukfAugmentedStateDesc.covariance();

    /*
    // prepare state gaussian (now considering only the process model)
    switch (processModel_->model_type())
    {
    case distr::Linear:
    case distr::NonLinearWithAdditiveNoise:
        augmentedState = stateDesc.mean();
        augmentedCovariance = stateDesc.covariance();
        break;

    case distr::NonLinearNonAdditiveNoise:
        int augmentedDimension = stateDimension + processModel_->noise_dimension();

        // NOTE STDCOUT
        // std::cout << "augmentedDimension = " << augmentedDimension << std::endl;

        augmentedState = DynamicVector(augmentedDimension);
        augmentedCovariance = CovarianceMatrix(augmentedDimension, augmentedDimension);

        std::vector<int> segmentsDimensions;
        segmentsDimensions.push_back(stateDimension);
        segmentsDimensions.push_back(processModel_->noise_dimension());
        ukfAugmentedStateDesc.segmentsDimensions(segmentsDimensions);

        // x' = [x 0]^T
        augmentedState.segment(0, stateDimension) = stateDesc.mean();
        augmentedState.segment(stateDimension, processModel_->noise_dimension()).setZero();

        //       | P  0 |
        // P^a = | 0  Q |, with Q = I

        augmentedCovariance.setZero(augmentedDimension, augmentedDimension);

        augmentedCovariance.block(0,
                                  0,
                                  stateDimension,
                                  stateDimension) = stateDesc.covariance();

        augmentedCovariance.block(stateDimension,
                                  stateDimension,
                                  processModel_->noise_dimension(),
                                  processModel_->noise_dimension()).setIdentity();
    }
    */

    // TODO update. in the mean time assume it's NonLinearNonAdditiveNoise
    int augmentedDimension = stateDimension + processModel_->Dimension();

    // NOTE STDCOUT
    // std::cout << "augmentedDimension = " << augmentedDimension << std::endl;

    augmentedState = DynamicVector(augmentedDimension);
    augmentedCovariance = CovarianceMatrix(augmentedDimension, augmentedDimension);

    std::vector<int> segmentsDimensions;
    segmentsDimensions.push_back(stateDimension);
    segmentsDimensions.push_back(processModel_->Dimension());
    ukfAugmentedStateDesc.segmentsDimensions(segmentsDimensions);

    // x' = [x 0]^T
    augmentedState.segment(0, stateDimension) = stateDesc.mean();
    augmentedState.segment(stateDimension, processModel_->Dimension()).setZero();

    //       | P  0 |
    // P^a = | 0  Q |, with Q = I

    augmentedCovariance.setZero(augmentedDimension, augmentedDimension);

    augmentedCovariance.block(0,
                              0,
                              stateDimension,
                              stateDimension) = stateDesc.covariance();

    augmentedCovariance.block(stateDimension,
                              stateDimension,
                              processModel_->Dimension(),
                              processModel_->Dimension()).setIdentity();
}

void UkfInternalsBase::process(const SigmaPointMatrix& sigmaPoints,
                           const ProcessModel::PerturbationType &control,
                           const double deltaTime,
                           SigmaPointMatrix& processedSigmaPoints,
                           int stateDimension)
{
    int nSigmaPoints = sigmaPoints.cols();

    // adjust number of sigma points if needed
    if (processedSigmaPoints.cols() != nSigmaPoints)
    {
        processedSigmaPoints.resize(nSigmaPoints);
    }

    for (int i = 0; i < nSigmaPoints; i++)
    {
        // set conditionals which is the state part of the augmented state
        processModel_->Conditional(deltaTime,
                                    sigmaPoints.col(i).segment(0, stateDimension),
                                    control);

        // NOTE STDCOUT
        // std::cout << i << std::endl << sigmaPoints.col(i).transpose() << std::endl;

        // Only for non-additive noise process models. For additive process model noise this
        // is not needed
        ProcessModel::PerturbationType processNoise =
                sigmaPoints.col(i).segment(stateDimension, processModel_->Dimension());

        //processNoise = ProcessModel::NoiseVector::Zero(processModel_->noise_dimension(), processModel_->noise_dimension());

        processedSigmaPoints.point(i, processModel_->MapNormal(processNoise),
                                      sigmaPoints.meanWeight(i),
                                      sigmaPoints.covWeight(i));

/*
        distr::GaussianDistribution<3> linear_delta;
                    linear_delta.mean(Vector3d::Zero());
                    linear_delta.covariance(Matrix3d::Identity() * pow(0.001, 2));

        distr::GaussianDistribution<3> angular_delta;
        angular_delta.mean(Vector3d::Zero());
        angular_delta.covariance(Matrix3d::Identity() * pow(M_PI/1000.0, 2));

        DynamicVector processedState = sigmaPoints.col(i).segment(0, stateDimension);

        processedState.topRows(3) += linear_delta.MapFromGaussian(processNoise.topRows(3));
        processedState.middleRows(3, 4) = (hf::Delta2Quaternion(angular_delta.MapFromGaussian(processNoise.bottomRows(3))) * Eigen::Quaterniond(processedState.middleRows<4>(3))).coeffs();

        processedSigmaPoints.point(i, processedState,
                                      sigmaPoints.meanWeight(i),
                                      sigmaPoints.covWeight(i));
*/
        // NOTE STDCOUT
        // std::cout << processedSigmaPoints.col(i).transpose() << std::endl ;
    }
}

void UkfInternalsBase::predictMeasurement(const SigmaPointMatrix& predictedStateSigmaPoints,
                                      SigmaPointMatrix& measurementSigmaPoints)
{
    int nSigmaPoints = predictedStateSigmaPoints.cols();

    // adjust measurement sigma point list size if required
    if (measurementSigmaPoints.cols() != nSigmaPoints)
    {
        measurementSigmaPoints.resize(nSigmaPoints);
    }

    for (int i = 0; i < nSigmaPoints; i++)
    {
        measurementModel_->conditionals(predictedStateSigmaPoints.col(i));

        measurementSigmaPoints.point(i, measurementModel_->predict(),
                                        predictedStateSigmaPoints.meanWeight(i),
                                        predictedStateSigmaPoints.covWeight(i));
    }
}
