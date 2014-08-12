
#ifndef FILTERS_UKF_TYPES_HPP
#define FILTERS_UKF_TYPES_HPP

#include <Eigen/Dense>

#include <state_filtering/models/processes/features/stationary_process.hpp>
#include <state_filtering/models/processes/implementations/brownian_object_motion.hpp>
#include <state_filtering/models/observers/spkf/measurement_model.hpp>

#define DYNAMIC Eigen::Dynamic

namespace distributions
{
    typedef double ScalarType;
    typedef Eigen::Matrix<ScalarType, -1, 1> VectorType;
    typedef Eigen::Matrix<ScalarType, -1, -1> OperatorType;
typedef Eigen::Matrix<ScalarType, -1, 1> InputType;



typedef distributions::BrownianObjectMotion<double, 5> ProcessModel; // TODO: this is shit!!!
    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;
    typedef distr::ObserverBase<DYNAMIC, DYNAMIC, DYNAMIC>  Observer;

    typedef Eigen::VectorXd                     DynamicVector;
    typedef Eigen::MatrixXd                     DynamicMatrix;
    typedef Eigen::MatrixXd                     CovarianceMatrix;

    typedef DynamicVector                       AugmentedVector;
    typedef CovarianceMatrix                    AugmentedCovariance;

    typedef DynamicVector                       StateVector;
    typedef CovarianceMatrix                    StateCovariance;

    typedef Observer::ObservationVector ObservationVector;
    typedef CovarianceMatrix                    ObservationCovariance;
}

#endif
