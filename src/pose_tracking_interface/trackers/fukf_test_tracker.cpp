
#include <ros/package.h>

#include <fast_filtering/utils/profiling.hpp>
#include <fast_filtering/distributions/uniform_distribution.hpp>

#include <pose_tracking_interface/trackers/fukf_test_tracker.hpp>
#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/object_file_reader.hpp>

FukfTestTracker::FukfTestTracker():
        nh_("~"),
        last_measurement_time_(std::numeric_limits<Scalar>::quiet_NaN())
{
    object_publisher_ = nh_.advertise<visualization_msgs::Marker>("object_model", 0);
}

void FukfTestTracker::Initialize(
        std::vector<Eigen::VectorXd> initial_states,
        const sensor_msgs::Image& ros_image,
        Eigen::Matrix3d camera_matrix,
        bool state_is_partial)
{
//    boost::mutex::scoped_lock lock(mutex_);

//    // convert camera matrix and image to desired format
//    camera_matrix.topLeftCorner(2,3) /= double(downsampling_factor_);

//    // convert to meters
//    Observation image = ri::Ros2Eigen<Scalar>(ros_image, downsampling_factor_);

//    // read some parameters
//    int evaluation_count;
//    double max_kl_divergence;
//    int max_sample_count;
//    double initial_occlusion_prob;
//    double p_occluded_visible;
//    double p_occluded_occluded;
//    double linear_acceleration_sigma;
//    double angular_acceleration_sigma;
//    double damping;
//    double tail_weight;
//    double model_sigma;
//    double sigma_factor;

//    ri::ReadParameter("evaluation_count", evaluation_count, nh_);
//    ri::ReadParameter("max_kl_divergence", max_kl_divergence, nh_);
//    ri::ReadParameter("max_sample_count", max_sample_count, nh_);
//    ri::ReadParameter("initial_occlusion_prob", initial_occlusion_prob, nh_);
//    ri::ReadParameter("p_occluded_visible", p_occluded_visible, nh_);
//    ri::ReadParameter("p_occluded_occluded", p_occluded_occluded, nh_);
//    ri::ReadParameter("linear_acceleration_sigma", linear_acceleration_sigma, nh_);
//    ri::ReadParameter("angular_acceleration_sigma", angular_acceleration_sigma, nh_);
//    ri::ReadParameter("damping", damping, nh_);
//    ri::ReadParameter("tail_weight", tail_weight, nh_);
//    ri::ReadParameter("model_sigma", model_sigma, nh_);
//    ri::ReadParameter("sigma_factor", sigma_factor, nh_);

//    ri::ReadParameter("object_names", object_names_, nh_);
//    ri::ReadParameter("downsampling_factor", downsampling_factor_, nh_);


//    // load object mesh
//    typedef std::vector<std::vector<Eigen::Vector3d> > ObjectVertices;
//    typedef std::vector<std::vector<std::vector<int> > > ObjectTriangleIndecies;
//    ObjectVertices object_vertices(object_names_.size());
//    ObjectTriangleIndecies object_triangle_indices(object_names_.size());

//    for(size_t i = 0; i < object_names_.size(); i++)
//    {
//        std::string object_model_path =
//                ros::package::getPath("arm_object_models")
//                + "/objects/" + object_names_[i]
//                + "/" + object_names_[i] + "_downsampled" + ".obj";

//        ObjectFileReader file_reader;
//        file_reader.set_filename(object_model_path);
//        file_reader.Read();

//        object_vertices[i] = *file_reader.get_vertices();
//        object_triangle_indices[i] = *file_reader.get_indices();
//    }


//    boost::shared_ptr<State> rigid_bodies_state(new ff::FreeFloatingRigidBodiesState<>(object_names_.size()));
//    boost::shared_ptr<ff::RigidBodyRenderer> object_renderer(new ff::RigidBodyRenderer(
//                                                                      object_vertices,
//                                                                      object_triangle_indices,
//                                                                      rigid_bodies_state));






//    ff::ContinuousKinectPixelObservationModel<State> pixel_observation_model(
//                object_renderer,
//                camera_matrix,
//                image.rows(),
//                image.cols(),
//                0.01,
//                0.01,
//                0.001424,
//                1.0,
//                6.0,
//                0.0);

//    ff::ContinuousOcclusionProcessModel occlusion_process(p_occluded_visible,
//                                                          p_occluded_occluded,
//                                                          0.2);

//    boost::shared_ptr<ObservationModel> observation_model;

//    // cpu obseration model
//    boost::shared_ptr<ff::KinectPixelObservationModel> kinect_pixel_observation_model(
//                new ff::KinectPixelObservationModel(tail_weight, model_sigma, sigma_factor));
//    boost::shared_ptr<ff::OcclusionProcessModel> occlusion_process(
//                new ff::OcclusionProcessModel(p_occluded_visible, p_occluded_occluded));
//    observation_model = boost::shared_ptr<ObservationModelCPUType>(
//                new ObservationModelCPUType(camera_matrix,
//                                    image.rows(),
//                                    image.cols(),
//                                    initial_states.size(),
//                                    object_renderer,
//                                    kinect_pixel_observation_model,
//                                    occlusion_process,
//                                    initial_occlusion_prob));

//    std::cout << "initialized observation omodel " << std::endl;

//    // initialize process model ========================================================================================================================================================================================================================================================================================================================================================================================================================
//    Eigen::MatrixXd linear_acceleration_covariance = Eigen::MatrixXd::Identity(3, 3) * pow(double(linear_acceleration_sigma), 2);
//    Eigen::MatrixXd angular_acceleration_covariance = Eigen::MatrixXd::Identity(3, 3) * pow(double(angular_acceleration_sigma), 2);

//    boost::shared_ptr<ProcessModel> process(new ProcessModel(object_names_.size()));
//    for(size_t i = 0; i < object_names_.size(); i++)
//    {
//        process->Parameters(i, object_renderer->object_center(i).cast<double>(),
//                               damping,
//                               linear_acceleration_covariance,
//                               angular_acceleration_covariance);
//    }

//    std::cout << "initialized process model " << std::endl;
//    // initialize coordinate_filter ============================================================================================================================================================================================================================================================
//    filter_ = boost::shared_ptr<FilterType>
//            (new FilterType(process, observation_model, sampling_blocks, max_kl_divergence));

//    // for the initialization we do standard sampling
//    std::vector<std::vector<size_t> > dependent_sampling_blocks(1);
//    dependent_sampling_blocks[0].resize(object_names_.size()*6);
//    for(size_t i = 0; i < dependent_sampling_blocks[0].size(); i++)
//        dependent_sampling_blocks[0][i] = i;
//    filter_->SamplingBlocks(dependent_sampling_blocks);
//    if(state_is_partial)
//    {
//        // create the multi body initial samples ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//        ff::FreeFloatingRigidBodiesState<> default_state(object_names_.size());
//        for(size_t object_index = 0; object_index < object_names_.size(); object_index++)
//            default_state.position(object_index) = Eigen::Vector3d(0, 0, 1.5); // outside of image

//        std::vector<ff::FreeFloatingRigidBodiesState<> > multi_body_samples(initial_states.size());
//        for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
//            multi_body_samples[state_index] = default_state;

//        std::cout << "doing evaluations " << std::endl;
//        for(size_t body_index = 0; body_index < object_names_.size(); body_index++)
//        {
//            std::cout << "evalution of object " << object_names_[body_index] << std::endl;
//            for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
//            {
//                ff::FreeFloatingRigidBodiesState<> full_initial_state(multi_body_samples[state_index]);
//                full_initial_state[body_index] = initial_states[state_index];
//                multi_body_samples[state_index] = full_initial_state;
//            }
//            filter_->Samples(multi_body_samples);
//            filter_->Filter(image, 0.0, ProcessModel::Input::Zero(object_names_.size()*6));
//            filter_->Resample(multi_body_samples.size());

//            multi_body_samples = filter_->Samples();
//        }
//    }
//    else
//    {
//        std::vector<ff::FreeFloatingRigidBodiesState<> > multi_body_samples(initial_states.size());
//        for(size_t i = 0; i < multi_body_samples.size(); i++)
//            multi_body_samples[i] = initial_states[i];

//        filter_->Samples(multi_body_samples);
//        filter_->Filter(image, 0.0, ProcessModel::Input::Zero(object_names_.size()*6));
//   }
//    filter_->Resample(evaluation_count/sampling_blocks.size());
//    filter_->SamplingBlocks(sampling_blocks);
}

Eigen::VectorXd FukfTestTracker::Filter(const sensor_msgs::Image& ros_image)
{
//    boost::mutex::scoped_lock lock(mutex_);

//    if(std::isnan(last_measurement_time_))
//        last_measurement_time_ = ros_image.header.stamp.toSec();
//    Scalar delta_time = ros_image.header.stamp.toSec() - last_measurement_time_;

//    // convert image
//    Observation image = ri::Ros2Eigen<Scalar>(ros_image, downsampling_factor_); // convert to m

//    // filter
//    INIT_PROFILING;
//    filter_->Filter(image, delta_time, ProcessModel::Input::Zero(object_names_.size()*6));
//    MEASURE("-----------------> total time for filtering");


//    // visualize the mean state
//    ff::FreeFloatingRigidBodiesState<> mean = filter_->StateDistribution().Mean();
//    for(size_t i = 0; i < object_names_.size(); i++)
//    {
//        std::string object_model_path = "package://arm_object_models/objects/" + object_names_[i] + "/" + object_names_[i] + ".obj";
//        ri::PublishMarker(mean.homogeneous_matrix(i).cast<float>(),
//                          ros_image.header, object_model_path, object_publisher_,
//                          i, 1, 0, 0);
//    }

//    last_measurement_time_ = ros_image.header.stamp.toSec();
//    return filter_->StateDistribution().Mean();
}
