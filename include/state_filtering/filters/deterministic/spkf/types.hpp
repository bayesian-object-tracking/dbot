
#ifndef FILTERS_UKF_TYPES_HPP
#define FILTERS_UKF_TYPES_HPP

#include <Eigen/Dense>

#include <state_filtering/models/process/stationary_process_model.hpp>
#include <state_filtering/models/measurement/spkf/measurement_model.hpp>

#define DYNAMIC Eigen::Dynamic

namespace distributions
{
    typedef double ScalarType;
    typedef Eigen::Matrix<ScalarType, -1, 1> VectorType;
    typedef Eigen::Matrix<ScalarType, -1, -1> OperatorType;


    typedef StationaryProcess<ScalarType, VectorType, -1> ProcessModel;
    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;
    typedef distr::MeasurementModelBase<DYNAMIC, DYNAMIC, DYNAMIC>  MeasurementModel;

    typedef Eigen::VectorXd                     DynamicVector;
    typedef Eigen::MatrixXd                     DynamicMatrix;
    typedef Eigen::MatrixXd                     CovarianceMatrix;

    typedef DynamicVector                       AugmentedVector;
    typedef CovarianceMatrix                    AugmentedCovariance;

    typedef DynamicVector                       StateVector;
    typedef CovarianceMatrix                    StateCovariance;

    typedef MeasurementModel::MeasurementVector MeasurementVector;
    typedef CovarianceMatrix                    MeasurementCovariance;
}

#endif
