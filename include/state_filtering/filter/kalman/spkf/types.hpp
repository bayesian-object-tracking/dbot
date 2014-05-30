
#ifndef FILTERS_UKF_TYPES_HPP
#define FILTERS_UKF_TYPES_HPP

#include <Eigen/Eigen>

#include "distributions/stationary_process_model.hpp"
#include "distributions/measurement_model.hpp"

namespace filter
{
    typedef distr::StationaryProcessModel<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC> ProcessModel;
    typedef distr::MeasurementModelBase<DYNAMIC, DYNAMIC, DYNAMIC>            MeasurementModel;

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
