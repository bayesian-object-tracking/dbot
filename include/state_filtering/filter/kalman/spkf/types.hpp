
#ifndef FILTERS_UKF_TYPES_HPP
#define FILTERS_UKF_TYPES_HPP

#include <Eigen/Dense>

#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/observation_models/spkf/measurement_model.hpp>

#define DYNAMIC Eigen::Dynamic

namespace filter
{
    typedef StationaryProcessModel< > ProcessModel;
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
