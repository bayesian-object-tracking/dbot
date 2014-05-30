#ifndef MEASUREMENT_MODEL_HPP_
#define MEASUREMENT_MODEL_HPP_

#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include <state_filtering/observation_models/spkf/model_types.hpp>
#include <state_filtering/tools/macros.hpp>

namespace distr
{
    template<int measurement_dim, int state_dim, int noise_dim>
    class MeasurementModelBase /* :
            public Distribution<measurement_dim,
                                state_dim,
                                measurement_dim, // not really needed !?
                                noise_dim> */
    {
    private:
        typedef MeasurementModelBase<measurement_dim, state_dim, noise_dim> this_type;

    public:
        typedef boost::shared_ptr<this_type > Ptr;

    public:
        static const int measurement_dim_ = measurement_dim;
        static const int state_dim_ = state_dim;
        static const int noise_dim_ = noise_dim;

        typedef Eigen::Matrix<double, this_type::measurement_dim_, 1> MeasurementVector;
        typedef Eigen::Matrix<double, this_type::noise_dim_, 1> NoiseVector;
        typedef Eigen::Matrix<double, this_type::state_dim_, 1> StateVector;
        typedef Eigen::MatrixXd NoiseCovariance;

    public:
        /**
         * @brief MeasurementModel default constructor
         *
         * @param [in] model_type Model type (linearity and noise behaviour)
         */
        explicit MeasurementModelBase(ModelType model_type = Linear): model_type(model_type) { }

        /**
         * @brief ~MeasurementModel Default destructor
         */
        virtual ~MeasurementModelBase() { }

        /**
         * @brief predict Predicts the measurement based on the current state.
         *
         * Based on the current state this predicts the measurement without incorporating any noise.
         * This is a deterministic process. That is, given the same state at different points in
         * time, the predicted measurement will remain the same.
         *
         * @return Predicted measurement vector
         */
        virtual MeasurementVector predict() = 0;

        /**
         * @brief predict Incorporates the given noise vector into the measurement prediciton.
         *
         * Maps the given noise vector int the measurement. The mapping may be linear or non-linear.
         *
         * @note The measurement model has to specify the characteristics of the noise. Eg. wheather
         * it's additive or not and what distribution it underlies.
         *
         * @param [in] randoms Noise vector
         *
         * @return Measurement vector including noise
         */
        virtual MeasurementVector predict(const NoiseVector& noise) = 0;

        /**
         * @brief sample Samples from the measurement model distribution including noise
         *
         * @return Measurement vector
         */
        virtual MeasurementVector sample() = 0;

        /**
         * @brief conditionals Sets the conditional events
         *
         * @param [in] state Model distribution conditional vector
         */
        virtual void conditionals(const StateVector& state) = 0;

        /**
         * @brief conditionals Returns the conditional event vector (state vector)
         *
         * @return StateVector vector
         */
        virtual StateVector conditionals() const = 0;

        /**
         * @brief model_type Returns the model type
         *
         * @return Model type id
         */
        virtual ModelType modelType(){ return model_type; }

        /**
         * @brief noiseCovariance Returns the noise covariance matrix
         *
         * @return Noise covariance matrix
         */
        virtual const NoiseCovariance& noiseCovariance(const MeasurementVector& sensor_measurement) = 0;


        virtual int measurement_dimension() const { return this_type::measurement_dim_; }
        virtual int noise_dimension() const { return this_type::noise_dim_; }
        virtual int conditional_dimension() const { return this_type::measurement_dim_; }

        /*
        virtual Eigen::Matrix<double, this_type::size_variables_, 1>
        MapFromGaussian(const Eigen::Matrix<double, this_type::size_randoms_, 1>& randoms) const
        {
            TO_BE_IMPLEMENTED
        }

        // special case where we just want to evaluate without any noise
        virtual Eigen::Matrix<double, this_type::size_variables_, 1> MapFromGaussian() const
        {
            TO_BE_IMPLEMENTED
        }

        virtual double Probability(const Eigen::Matrix<double, this_type::size_variables_, 1>& variables) const
        {
            TO_BE_IMPLEMENTED
        }

        virtual double LogProbability(const  Eigen::Matrix<double, this_type::size_variables_, 1>& variables) const
        {
            TO_BE_IMPLEMENTED
        }
        */

    protected:
        /**
         * @brief model_type Model type id
         */
        ModelType model_type;
    };
}

#endif
