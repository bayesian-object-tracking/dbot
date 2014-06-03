
#ifndef DEPTH_MEASUREMENT_MODEL_HPP_
#define DEPTH_MEASUREMENT_MODEL_HPP_

#include <Eigen/Eigen>

//#include <opencv/cv.hpp>

#include <cmath>

#include <boost/shared_ptr.hpp>

#include <state_filtering/observation_models/spkf/measurement_model.hpp>
#include <state_filtering/distribution/gaussian/gaussian_distribution.hpp>
#include <state_filtering/tools/rigid_body_renderer.hpp>
#include <state_filtering/tools/macros.hpp>

namespace distr
{
    template<int measurement_dim = -1, int state_dim = -1, int noise_dim = -1>
        class DepthMeasurementModel:
            public MeasurementModelBase<measurement_dim, state_dim, noise_dim>
    {
        private:
            typedef MeasurementModelBase<measurement_dim, state_dim, noise_dim>    BaseType;
            typedef DepthMeasurementModel<measurement_dim, state_dim, noise_dim>   ThisType;

        public:
            typedef boost::shared_ptr<ThisType > Ptr;

            typedef typename BaseType::MeasurementVector MeasurementVector;
            typedef typename BaseType::NoiseVector NoiseVector;
            typedef typename BaseType::StateVector StateVector;
            typedef typename BaseType::NoiseCovariance NoiseCovariance;

        public: /* MeasurementModel */
            /**
             * @brief DepthMeasurementModel Construct
             *
             * @param object_model Object model instance
             */
            explicit DepthMeasurementModel(obj_mod::TriangleObjectModel::Ptr object_model):
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
             * @brief ~DepthMeasurementModel Destructor
             */
            virtual ~DepthMeasurementModel()
            {

            }

            /**
             * @see MeasurementModel::predict()
             */
            virtual MeasurementVector predict()
            {
                MeasurementVector predictedMeasurement;
                _predict(predictedMeasurement);

                return predictedMeasurement;
            }

            /**
             * @see MeasurementModel::sample(const NoiseVector&)
             *
             * NOTE: dimension might not be correct
             */
            virtual MeasurementVector predict(const NoiseVector& randoms)
            {
                MeasurementVector predictedMeasurement(ThisType::measurement_dim_);
                _predict(predictedMeasurement);

                return predictedMeasurement + randoms;
            }

            /**
             * @see MeasurementModel::sample()
             */
            virtual MeasurementVector sample()
            {
                NoiseVector iso_sample(ThisType::noise_dim_);
                for (int i = 0; i < ThisType::noise_dim_; i++)
                {
                    iso_sample(i) = unit_gaussian.Sample()(0);
                }

                return predict(iso_sample);
            }

            /**
             * @see MeasurementModel::conditionals(const StateVector&)
             */
            virtual void conditionals(const StateVector& state)
            {
                this->state = state;
            }

            /**
             * @see MeasurementModel::conditionals()
             */
            virtual StateVector conditionals() const
            {
                return state;
            }

            /**
             * @see MeasurementModel::noiseCovariance()
             */
            virtual const NoiseCovariance& noiseCovariance(const MeasurementVector& mean_measurement)
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
                                noise_covariance(ind, 0) = measurement_NA_variance;
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
                            noise_covariance(i, 0) = measurement_NA_variance;
                        }
                    }

                    dirtyCov = false;
                }

                return noise_covariance;
            }

            virtual int measurement_dimension() const { return this->n_rows * this->n_cols; }
            virtual int noise_dimension() const { return measurement_dimension(); }
            virtual int conditional_dimension() const { return state.rows(); }

        public: /* Model specifics */

            /**
             * @brief parameters sets depth measurement model parameters.
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
                                    double measurement_NA_sigma,
                                    const MeasurementVector& sensor_measurement,
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
                this->measurement_NA_variance = measurement_NA_sigma * measurement_NA_sigma;

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
             * @brief object_model Returns the object model used for measurement prediction
             * @return
             */
            virtual obj_mod::TriangleObjectModel::Ptr objectModel()
            {
                return object_model;
            }

        private:
            /**
             * @brief predict Implementation of the measurement prediction.
             *
             * @param [out] measurement
             */
            virtual void _predict(MeasurementVector& measurement)
            {
                dirtyCov = true;

                std::vector<float> measurementVector;

                object_model->state(state);
                object_model->Predict(camera_matrix,
                        n_rows,
                        n_cols,
                        measurementVector);

                measurement.resize(measurement_dimension(), 1);
                measurement.setOnes();
                measurement *= mean_depth;

                //availableAll_ += MeasurementVector::Ones(measurement_dimension(), 1);
                //availableAndIntersectAll_ += MeasurementVector::Ones(measurement_dimension(), 1);

                int curInd;
                for (unsigned int i = 0; i < availableIndices_.size(); i++)
                {
                    curInd = availableIndices_[i];

                    availableAll_(curInd, 0) += 1;

                    if (measurementVector[curInd] < std::numeric_limits<float>::max())
                    {
                        measurement(curInd, 0) = measurementVector[curInd];
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
            MeasurementVector measurement;
            filter::GaussianDistribution<double, 1> unit_gaussian;
            double depth_noise_variance;
            double mean_depth;
            double uncertain_variance;
            double measurement_NA_variance;
            NoiseCovariance noise_covariance;
            std::vector<int> availableIndices_;

            MeasurementVector availableAll_;
            MeasurementVector availableAndIntersectAll_;
            MeasurementVector sensor_measurement_;
            bool dirtyCov;

            double c_;
            double s_;

            int n_predictions;
    };
}

#endif
