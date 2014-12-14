
#include <ros/package.h>

#include <ff/utils/profiling.hpp>
#include <ff/distributions/uniform_distribution.hpp>

#include <ff/utils/helper_functions.hpp>

#include <pose_tracking_interface/trackers/fukf_test_tracker.hpp>
#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/object_file_reader.hpp>

FukfTestTracker::FukfTestTracker():
        nh_("~"),
        last_measurement_time_(std::numeric_limits<Scalar>::quiet_NaN()),
        ip_(nh_)
{
    object_publisher_ = nh_.advertise<visualization_msgs::Marker>("object_model", 0);
}

void FukfTestTracker::Initialize(State_a initial_state,
                                 const sensor_msgs::Image& ros_image,
                                 Eigen::Matrix3d camera_matrix)
{
    boost::mutex::scoped_lock lock(mutex_);


    // read some parameters
    int evaluation_count;
    double max_kl_divergence;
    int max_sample_count;
    double initial_occlusion_prob;
    double p_occluded_visible;
    double p_occluded_occluded;
    double linear_acceleration_sigma;
    double angular_acceleration_sigma;
    double damping;    
    double occlusion_process_sigma;

    double tail_weight;
    double model_sigma;
    double sigma_factor;
    double half_life_depth;
    double max_depth;
    double min_depth;

    ri::ReadParameter("evaluation_count", evaluation_count, nh_);
    ri::ReadParameter("max_kl_divergence", max_kl_divergence, nh_);
    ri::ReadParameter("max_sample_count", max_sample_count, nh_);
    ri::ReadParameter("initial_occlusion_prob", initial_occlusion_prob, nh_);
    ri::ReadParameter("p_occluded_visible", p_occluded_visible, nh_);
    ri::ReadParameter("p_occluded_occluded", p_occluded_occluded, nh_);
    ri::ReadParameter("occlusion_process_sigma", occlusion_process_sigma, nh_);
    ri::ReadParameter("linear_acceleration_sigma", linear_acceleration_sigma, nh_);
    ri::ReadParameter("angular_acceleration_sigma", angular_acceleration_sigma, nh_);
    ri::ReadParameter("damping", damping, nh_);

    ri::ReadParameter("fukf_tail_weight", tail_weight, nh_);
    ri::ReadParameter("fukf_model_sigma", model_sigma, nh_);
    ri::ReadParameter("fukf_sigma_factor", sigma_factor, nh_);
    ri::ReadParameter("half_life_depth", half_life_depth, nh_);
    ri::ReadParameter("max_depth", max_depth, nh_);
    ri::ReadParameter("min_depth", min_depth, nh_);

    ri::ReadParameter("object_names", object_names_, nh_);
    ri::ReadParameter("downsampling_factor", downsampling_factor_, nh_);

    Eigen::MatrixXd linear_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * pow(double(linear_acceleration_sigma), 2);

    Eigen::MatrixXd angular_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * pow(double(angular_acceleration_sigma), 2);


    std::cout << "adapting camera_matrix" << std::endl;
    // convert camera matrix and image to desired format
    camera_matrix.topLeftCorner(2,3) /= double(downsampling_factor_);

    std::cout << "converting image" << std::endl;
    // convert to meters
    Eigen::MatrixXd image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_);

    std::cout << "reading parameter" << std::endl;

    std::cout << "loading objetc model" << std::endl;

    // load object mesh
    typedef std::vector<std::vector<Eigen::Vector3d> > ObjectVertices;
    typedef std::vector<std::vector<std::vector<int> > > ObjectTriangleIndecies;
    ObjectVertices object_vertices(object_names_.size());
    ObjectTriangleIndecies object_triangle_indices(object_names_.size());

    for(size_t i = 0; i < object_names_.size(); i++)
    {
        std::string object_model_path =
                ros::package::getPath("arm_object_models")
                + "/objects/" + object_names_[i]
                + "/" + object_names_[i] + "_downsampled" + ".obj";

        ObjectFileReader file_reader;
        file_reader.set_filename(object_model_path);
        file_reader.Read();

        object_vertices[i] = *file_reader.get_vertices();
        object_triangle_indices[i] = *file_reader.get_indices();

        std::cout << "Loaded " << object_names_[i] << " object file. !!!!!!!!!!!!!!!!" << std::endl;
    }

    std::cout << "creating state" << std::endl;


    boost::shared_ptr<State_a> rigid_bodies_state(
                new State_a(object_names_.size()));

    std::cout << "creating renderer" << std::endl;

    boost::shared_ptr<ff::RigidBodyRenderer> object_renderer(
                new ff::RigidBodyRenderer(
                    object_vertices,
                    object_triangle_indices,
                    rigid_bodies_state));

    std::cout << "creating observation model" << std::endl;

    boost::shared_ptr<ObservationModel >
            pixel_observation_model(
                new ObservationModel(
                    object_renderer,
                    camera_matrix,
                    image.rows(),
                    image.cols(),
                    tail_weight,
                    model_sigma,
                    sigma_factor,
                    half_life_depth,
                    max_depth,
                    min_depth));

    std::cout << "initialized observation omodel " << std::endl;

    boost::shared_ptr<ProcessModel_a> process_a(
                new ProcessModel_a(object_names_.size()));

    boost::shared_ptr<ProcessModel_b> process_b(
                new ProcessModel_b(
                    p_occluded_visible, p_occluded_occluded, occlusion_process_sigma));

    for(size_t i = 0; i < object_names_.size(); i++)
    {
        process_a->Parameters(i, object_renderer->object_center(i).cast<double>(),
                               damping,
                               linear_acceleration_covariance,
                               angular_acceleration_covariance);
    }

    std::cout << "initialized process model " << std::endl;   
    filter_ = boost::shared_ptr<FilterType>(
                new FilterType(process_a, process_b, pixel_observation_model));

    State_b b_i(1, 1);
    b_i(0, 0) = ff::hf::Logit(initial_occlusion_prob);
    state_distr.initialize(initial_state,
                           image.rows()* image.cols(),
                           b_i,
                           0.002,
                           occlusion_process_sigma*occlusion_process_sigma);
    std::cout << "initialized state and created filter" << std::endl;
}

void FukfTestTracker::Filter(const sensor_msgs::Image& ros_image)
{
    boost::mutex::scoped_lock lock(mutex_);

    if(std::isnan(last_measurement_time_))
    {
        last_measurement_time_ = ros_image.header.stamp.toSec();
    }

//    Scalar delta_time = ros_image.header.stamp.toSec() - last_measurement_time_;
    Scalar delta_time = 1.0/30.0;

    Eigen::MatrixXd image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_);

    Eigen::MatrixXd y(image.rows()*image.cols(), 1);
    for (size_t i = 0, k = 0; i < image.rows(); ++i)
    {
        for (int j = 0; j < image.cols(); ++j)
        {
            y(k++, 0) = image(i, j);
        }
    }

    rows_ = image.rows();
    cols_ = image.cols();

    INIT_PROFILING;
    filter_->Predict(state_distr, delta_time, state_distr);
    MEASURE("-----------------> filter_->predict");
    filter_->Update(state_distr, y, state_distr);
    MEASURE("<----------------- filter_->update");

    Eigen::MatrixXd image_vector;
    image_vector.resize(rows_*cols_, 1);

    // occlusion
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        image_vector(i, 0) = state_distr.joint_partitions[i].mean_b(0, 0);
    }
    ip_.publish(image_vector, "fukf/occlusion", rows_, cols_);

    // occlusion
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        image_vector(i, 0) = state_distr.joint_partitions[i].cov_bb(0, 0);
    }
    ip_.publish(image_vector, "fukf/occlusion_cov", rows_, cols_);

    // prediction
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        image_vector(i, 0) = state_distr.joint_partitions[i].mean_y(0, 0);
    }
    ip_.publish(image_vector, "fukf/prediction", rows_, cols_);

    // prediction_covariance
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        image_vector(i, 0) = state_distr.joint_partitions[i].cov_yy(0, 0);
    }
    ip_.publish(image_vector, "fukf/prediction_cov", rows_, cols_);

    // measurement
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        image_vector(i, 0) = (std::isnan(y(i, 0)) ? 0 : y(i, 0));
    }
    ip_.publish(y, "fukf/measurement", rows_, cols_);

    // innovation
    for (size_t i = 0; i < rows_*cols_; ++i)
    {
        if (!std::isnan(y(i, 0)))
        {
            image_vector(i, 0) = y(i, 0) - state_distr.joint_partitions[i].mean_y(0, 0);
        }
        else
        {
            image_vector(i, 0) = 0;
        }
    }
    ip_.publish(image_vector, "fukf/innovation", rows_, cols_);

    // visualize the mean state
    ff::FreeFloatingRigidBodiesState<> mean = state_distr.mean_a;
    for(size_t i = 0; i < object_names_.size(); i++)
    {
        std::string object_model_path = "package://arm_object_models/objects/" + object_names_[i] + "/" + object_names_[i] + ".obj";
        ri::PublishMarker(mean.homogeneous_matrix(i).cast<float>(),
                          ros_image.header, object_model_path, object_publisher_,
                          i, 0, 1, 0);
    }

    last_measurement_time_ = ros_image.header.stamp.toSec();
}
