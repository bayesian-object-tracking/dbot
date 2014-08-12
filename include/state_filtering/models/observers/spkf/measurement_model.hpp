#ifndef observer_HPP_
#define observer_HPP_

#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include <state_filtering/models/observers/spkf/model_types.hpp>
#include <state_filtering/utils/macros.hpp>

namespace distr
{
    template<int observation_dim, int state_dim, int noise_dim>
    class ObserverBase /* :
            public Distribution<measurement_dim,
                                state_dim,
                                observation_dim, // not really needed !?
                                noise_dim> */
    {
    private:
        typedef ObserverBase<observation_dim, state_dim, noise_dim> this_type;

    public:
        typedef boost::shared_ptr<this_type > Ptr;

    public:
        static const int observation_dim_ = observation_dim;
        static const int state_dim_ = state_dim;
        static const int noise_dim_ = noise_dim;

        typedef Eigen::Matrix<double, this_type::observation_dim_, 1> ObservationVector;
        typedef Eigen::Matrix<double, this_type::noise_dim_, 1> NoiseVector;
        typedef Eigen::Matrix<double, this_type::state_dim_, 1> StateVector;
        typedef Eigen::MatrixXd NoiseCovariance;

    public:
        /**
         * @brief Observer default constructor
         *
         * @param [in] model_type Model type (linearity and noise behaviour)
         */
        explicit ObserverBase(ModelType model_type = Linear): model_type(model_type) { }

        /**
         * @brief ~Observer Default destructor
         */
        virtual ~ObserverBase() { }

        /**
         * @brief predict Predicts the observation based on the current state.
         *
         * Based on the current state this predicts the observation without incorporating any noise.
         * This is a deterministic process. That is, given the same state at different points in
         * time, the predicted observation will remain the same.
         *
         * @return Predicted observation vector
         */
        virtual ObservationVector predict() = 0;

        /**
         * @brief predict Incorporates the given noise vector into the observation prediciton.
         *
         * Maps the given noise vector int the observation. The mapping may be linear or non-linear.
         *
         * @note The observation model has to specify the characteristics of the noise. Eg. wheather
         * it's additive or not and what distribution it underlies.
         *
         * @param [in] randoms Noise vector
         *
         * @return Observation vector including noise
         */
        virtual ObservationVector predict(const NoiseVector& noise) = 0;

        /**
         * @brief sample Samples from the observation model distribution including noise
         *
         * @return Observation vector
         */
        virtual ObservationVector sample() = 0;

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
        virtual const NoiseCovariance& noiseCovariance(const ObservationVector& sensor_measurement) = 0;


        virtual int observation_dimension() const { return this_type::observation_dim_; }
        virtual int noise_dimension() const { return this_type::noise_dim_; }
        virtual int conditional_dimension() const { return this_type::observation_dim_; }

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
