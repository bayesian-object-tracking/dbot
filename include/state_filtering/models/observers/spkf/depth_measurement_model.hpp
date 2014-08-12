
#ifndef DEPTH_observer_HPP_
#define DEPTH_observer_HPP_

#include <Eigen/Eigen>

//#include <opencv/cv.hpp>

#include <cmath>

#include <boost/shared_ptr.hpp>

#include <state_filtering/models/observers/spkf/measurement_model.hpp>
#include <state_filtering/distributions/gaussian/gaussian_distribution.hpp>
#include <state_filtering/utils/rigid_body_renderer.hpp>
#include <state_filtering/utils/macros.hpp>

namespace distr
{
    template<int observation_dim = -1, int state_dim = -1, int noise_dim = -1>
        class DepthObserver:
            public ObserverBase<measurement_dim, state_dim, noise_dim>
    {
        private:
            typedef ObserverBase<measurement_dim, state_dim, noise_dim>    BaseType;
            typedef DepthObserver<measurement_dim, state_dim, noise_dim>   ThisType;

        public:
            typedef boost::shared_ptr<ThisType > Ptr;

            typedef typename BaseType::ObservationVector ObservationVector;
            typedef typename BaseType::NoiseVector NoiseVector;
            typedef typename BaseType::StateVector StateVector;
            typedef typename BaseType::NoiseCovariance NoiseCovariance;

        public: /* Observer */
            /**
             * @brief DepthObserver Construct
             *
             * @param object_model Object model instance
             */
            explicit DepthObserver(obj_mod::TriangleObjectModel::Ptr object_model):
                BaseType(NonLinearWithAdditiveNoise),
                mean_depth(0.0),
                uncertain_variance(0.0),
                dirtyCov(true),
                n_predictions(0)
            {
                this->object_model = object_model;

                unit_gaussian.mean(Eigen::Matrix<double, 1, 1>::Zero());
                unit_gaussian.covariance(Eigen::Matrix<double, 1, 1>::Identity());
            }

            /**
             * @brief ~DepthObserver Destructor
             */
            virtual ~DepthObserver()
            {

            }

            /**
             * @see Observer::predict()
             */
            virtual ObservationVector predict()
            {
                ObservationVector predictedObservation;
                _predict(predictedObservation);

                return predictedObservation;
            }

            /**
             * @see Observer::sample(const NoiseVector&)
             *
             * NOTE: dimension might not be correct
             */
            virtual ObservationVector predict(const NoiseVector& randoms)
            {
                ObservationVector predictedObservation(ThisType::measurement_dim_);
                _predict(predictedObservation);

                return predictedObservation + randoms;
            }

            /**
             * @see Observer::sample()
             */
            virtual ObservationVector sample()
            {
                NoiseVector iso_sample(ThisType::noise_dim_);
                for (int i = 0; i < ThisType::noise_dim_; i++)
                {
                    iso_sample(i) = unit_gaussian.Sample()(0);
                }

                return predict(iso_sample);
            }

            /**
             * @see Observer::conditionals(const StateVector&)
             */
            virtual void conditionals(const StateVector& state)
            {
                this->state = state;
            }

            /**
             * @see Observer::conditionals()
             */
            virtual StateVector conditionals() const
            {
                return state;
            }

            /**
             * @see Observer::noiseCovariance()
             */
            virtual const NoiseCovariance& noiseCovariance(const ObservationVector& mean_measurement)
            {
                if (dirtyCov)
                {
                    /*
                    int dilation_size = 1;

                    // create mask, erode then dilate twice
                    cv::Mat mask = cv::Mat(n_rows, n_cols, CV_8UC1);
                    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                                         cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                                         cv::Point( dilation_size, dilation_size ) );

                    for (int i = 0; i < n_rows; i++)
                    {
                        for (int j = 0; j < n_cols; j++)
                        {
                            int ind = i * n_cols + j;

                            if (availableAndIntersectAll_(ind, 0) == n_predictions)
                            {
                                mask.at<unsigned char>(i, j) = 255;
                            }
                            else
                            {
                                mask.at<unsigned char>(i, j) = 0;
                            }
                        }
                    }

                    cv::dilate(mask, mask, element);
                    cv::erode(mask, mask, element);
                    cv::erode(mask, mask, element);
                    //cv::dilate(mask, mask, element);
                    //cv::imshow( "Erosion", mask );
                    //cv::waitKey(1);

                    for (int i = 0; i < n_rows; i++)
                    {
                        for (int j = 0; j < n_cols; j++)
                        {
                            int ind = i * n_cols + j;

                            if (mask.at<unsigned char>(i, j) == 255)
                            {
                                noise_covariance(ind, 0) = depth_noise_variance;
                            }
                            else if(availableAll_(ind, 0) == n_predictions)
                            {
                                noise_covariance(ind, 0) = uncertain_variance;
                            }
                            else
                            {
                                noise_covariance(ind, 0) = observation_NA_variance;
                            }
                        }
                    }
                    */


                    //double c = 1.0/std::sqrt(depth_noise_variance);
                    for (unsigned int i = 0; i < noise_covariance.rows(); i++)
                    {
                        if (availableAndIntersectAll_(i, 0) == n_predictions)
                        {
                            // nasty stuff or maybe not so much!
                            noise_covariance(i, 0) = depth_noise_variance;
                            /* only reject background depth
                            if (sensor_measurement_(i, 0) - mean_measurement(i, 0) > 0)
                            {
                                noise_covariance(i, 0) =
                                        (uncertain_variance - depth_noise_variance)
                                        * (1.0-((1.0+c)/(c + std::exp(std::pow(sensor_measurement_(i, 0) - mean_measurement(i, 0), 2.0)/(s_*s_)))))
                                        + depth_noise_variance;
                            }
                            else
                            {
                                noise_covariance(i, 0) = depth_noise_variance;
                            }
                            */
                            /*
                            noise_covariance(i, 0) =
                                    (uncertain_variance - depth_noise_variance)
                                    * (1.0-((1.0+c)/(c + std::exp(std::pow(sensor_measurement_(i, 0) - mean_measurement(i, 0), 2.0)/(s_*s_)))))
                                    + depth_noise_variance;
                            */
                        }
                        else if(availableAll_(i, 0) == n_predictions)
                        {
                            noise_covariance(i, 0) = uncertain_variance;
                        }
                        else
                        {
                            noise_covariance(i, 0) = observation_NA_variance;
                        }
                    }

                    dirtyCov = false;
                }

                return noise_covariance;
            }

            virtual int observation_dimension() const { return this->n_rows * this->n_cols; }
            virtual int noise_dimension() const { return observation_dimension(); }
            virtual int conditional_dimension() const { return state.rows(); }

        public: /* Model specifics */

            /**
             * @brief parameters sets depth observation model parameters.
             *
             * @param [in] camera_matrix     Camera projection matrix
             * @param [in] n_rows            Image y dimension
             * @param [in] n_cols            Image x dimension
             */
            virtual void parameters(const Eigen::Matrix3d camera_matrix,
                                    int n_rows,
                                    int n_cols,
                                    const std::vector<int>& availableIndices,
                                    double depth_noise_sigma,
                                    double mean_depth,
                                    double uncertain_sigma,
                                    double observation_NA_sigma,
                                    const ObservationVector& sensor_measurement,
                                    double c,
                                    double s)
            {
                this->camera_matrix = camera_matrix;
                this->n_rows = n_rows;
                this->n_cols = n_cols;
                this->availableIndices_ = availableIndices;

                this->depth_noise_variance = depth_noise_sigma * depth_noise_sigma;
                this->mean_depth = mean_depth;
                this->uncertain_variance = uncertain_sigma * uncertain_sigma;
                this->measurement_NA_variance = observation_NA_sigma * observation_NA_sigma;

                noise_covariance.resize(noise_dimension(), 1);
                noise_covariance.setOnes();
                noise_covariance *= (measurement_NA_variance-1);

                availableAll_.resize(noise_dimension(), 1);
                availableAll_.setZero();

                availableAndIntersectAll_.resize(noise_dimension(), 1);
                availableAndIntersectAll_.setZero();

                sensor_measurement_ = sensor_measurement;


                n_predictions = 0;

                c_ = c;
                s_ = s;
            }

            /**
             * @brief object_model Returns the object model used for observation prediction
             * @return
             */
            virtual obj_mod::TriangleObjectModel::Ptr objectModel()
            {
                return object_model;
            }

        private:
            /**
             * @brief predict Implementation of the observation prediction.
             *
             * @param [out] observation
             */
            virtual void _predict(ObservationVector& observation)
            {
                dirtyCov = true;

                std::vector<float> observationVector;

                object_model->state(state);
                object_model->Predict(camera_matrix,
                        n_rows,
                        n_cols,
                        observationVector);

                observation.resize(measurement_dimension(), 1);
                observation.setOnes();
                observation *= mean_depth;

                //availableAll_ += ObservationVector::Ones(measurement_dimension(), 1);
                //availableAndIntersectAll_ += ObservationVector::Ones(measurement_dimension(), 1);

                int curInd;
                for (unsigned int i = 0; i < availableIndices_.size(); i++)
                {
                    curInd = availableIndices_[i];

                    availableAll_(curInd, 0) += 1;

                    if (measurementVector[curInd] < std::numeric_limits<float>::max())
                    {
                        observation(curInd, 0) = observationVector[curInd];
                        availableAndIntersectAll_(curInd, 0) += 1;
                    }
                }

                n_predictions++;
            }

        private:
            obj_mod::RigidBodyRenderer::Ptr object_model;
            int n_rows;
            int n_cols;
            Eigen::Matrix3d camera_matrix;
            StateVector state;
            ObservationVector observation;
            filter::GaussianDistribution<double, 1> unit_gaussian;
            double depth_noise_variance;
            double mean_depth;
            double uncertain_variance;
            double observation_NA_variance;
            NoiseCovariance noise_covariance;
            std::vector<int> availableIndices_;

            ObservationVector availableAll_;
            ObservationVector availableAndIntersectAll_;
            ObservationVector sensor_measurement_;
            bool dirtyCov;

            double c_;
            double s_;

            int n_predictions;
    };
}

#endif
