#ifndef POSE_TRACKING_MODELS_OBSERVATION_MODELS_KINECT_IMAGE_OBSERVATION_MODEL_GPU_HPP
#define POSE_TRACKING_MODELS_OBSERVATION_MODELS_KINECT_IMAGE_OBSERVATION_MODEL_GPU_HPP

#include <vector>
#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"
#include "Eigen/Core"

#include <fl/util/math/pose_vector.hpp>
#include <dbot/models/observation_models/rao_blackwell_observation_model.hpp>
#include <dbot/states/free_floating_rigid_bodies_state.hpp>

#include <dbot/models/observation_models/kinect_image_observation_model_gpu/object_rasterizer.hpp>
#include <dbot/models/observation_models/kinect_image_observation_model_gpu/cuda_filter.hpp>

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_gl_interop.h>

#include <dbot/utils/profiling.hpp>

namespace ff
{

// Forward declarations
template <typename State> class KinectImageObservationModelGPU;

namespace internal
{
/**
 * ImageObservationModelCPU distribution traits specialization
 * \internal
 */
template <typename State>
struct Traits<KinectImageObservationModelGPU<State> >
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef RBObservationModel<State, Observation> Base;

    typedef typename Eigen::Matrix<Scalar, 3, 3> CameraMatrix;

//  typedef sf::RigidBodySystem<-1>           State;
};
}

/**
 * \class ImageObservationModelGPU
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
template <typename State>
class KinectImageObservationModelGPU:
        public internal::Traits<KinectImageObservationModelGPU<State> >::Base
{
public:
    typedef internal::Traits<KinectImageObservationModelGPU<State> > Traits;

    typedef typename Traits::Scalar         Scalar;
    typedef typename Traits::Observation    Observation;
    typedef typename Traits::CameraMatrix   CameraMatrix;

    typedef typename Traits::Base::StateArray StateArray;
    typedef typename Traits::Base::RealArray RealArray;
    typedef typename Traits::Base::IntArray IntArray;

    typedef typename Eigen::Transform<fl::Real, 3, Eigen::Affine> Affine;



    // TODO: ALL THIS SHOULD SWITCH FROM USING VISIBILITY TO OCCLUSION
    KinectImageObservationModelGPU(const CameraMatrix& camera_matrix,
                    const size_t& n_rows,
                    const size_t& n_cols,
                    const size_t& max_sample_count,
                    const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
                    const std::vector<std::vector<std::vector<int> > > indices,
                    const std::string vertex_shader_path,
                    const std::string fragment_shader_path,
                    const Scalar& initial_occlusion_prob = 0.1d,
                    const double& delta_time = 0.033d,
                    const float p_occluded_visible = 0.1f,
                    const float p_occluded_occluded = 0.7f,
                    const float tail_weight = 0.01f,
                    const float model_sigma = 0.003f,
                    const float sigma_factor = 0.0014247f,
                    const float max_depth = 6.0f,
                    const float exponential_rate = -log(0.5f)):
            camera_matrix_(camera_matrix),
            n_rows_(n_rows),
            n_cols_(n_cols),
            max_sample_count_(max_sample_count),
            indices_(indices),
            initial_visibility_prob_(1 - initial_occlusion_prob),
            p_visible_visible_(1.0 - p_occluded_visible),
            p_visible_occluded_(1.0 - p_occluded_occluded),
            tail_weight_(tail_weight),
            model_sigma_(model_sigma),
            sigma_factor_(sigma_factor),
            max_depth_(max_depth),
            exponential_rate_(exponential_rate),
            n_poses_(max_sample_count),
            observations_set_(false),
            resource_registered_(false),
            nr_calls_set_observation_(0),
            observation_time_(0),
            Traits::Base(delta_time)
    {
        // set constants
        this->default_poses_.recount(vertices_double.size());
        this->default_poses_.setZero();

        // convert doubles to floats
        vertices_.resize(vertices_double.size());
        for(size_t object_index = 0; object_index < vertices_.size(); object_index++)
        {
            vertices_[object_index].resize(vertices_double[object_index].size());
            for(size_t vertex_index = 0; vertex_index < vertices_[object_index].size(); vertex_index++)
                vertices_[object_index][vertex_index] = vertices_double[object_index][vertex_index].cast<float>();
        }


        // check for incorrect path names
        if(!boost::filesystem::exists(vertex_shader_path))
        {
            std::cout << "vertex shader does not exist at: "
                 << vertex_shader_path << std::endl;
            exit(-1);
        }
        if(!boost::filesystem::exists(fragment_shader_path))
        {
            std::cout << "fragment_shader does not exist at: "
                 << fragment_shader_path << std::endl;
            exit(-1);
        }

        vertex_shader_path_ =  vertex_shader_path;
        fragment_shader_path_ = fragment_shader_path;


        // initialize opengl and cuda
        opengl_ = boost::shared_ptr<ObjectRasterizer>
                (new ObjectRasterizer(vertices_,
                                      indices_,
                                      vertex_shader_path_,
                                      fragment_shader_path_,
                                      camera_matrix_.cast<float>()));
        cuda_ = boost::shared_ptr<fil::CudaFilter> (new fil::CudaFilter());



        //opengl_->PrepareRender(camera_matrix_.cast<float>());


        opengl_->set_number_of_max_poses(max_sample_count_);
        n_poses_x_ = opengl_->get_n_poses_x();
        cuda_->set_number_of_max_poses(max_sample_count_, n_poses_x_);


        std:: cout << "set resolution in cuda..." << std::endl;

        opengl_->set_resolution(n_rows_, n_cols_);
        cuda_->set_resolution(n_rows_, n_cols_);

        register_resource();

        std:: cout << "set occlusions..." << std::endl;

        reset();

        float c = p_visible_visible_ - p_visible_occluded_;
        float log_c = log(c);

        std::vector<std::vector<float> > dummy_com_models;
        cuda_->init(dummy_com_models, 0.0f, 0.0f,
                    initial_visibility_prob_, c, log_c, p_visible_occluded_,
                    tail_weight_, model_sigma_, sigma_factor_, max_depth_, exponential_rate_);


        count_ = 0;


        visibility_probs_.resize(n_rows_ * n_cols_);
    }

    ~KinectImageObservationModelGPU() { }


    RealArray loglikes(const StateArray& deltas,
                                  IntArray& occlusion_indices,
                                  const bool& update_occlusions = false)
    {

        if(!observations_set_)
        {
            std:: cout << "GPU: observations not set" << std::endl;
            exit(-1);
        }

        n_poses_ = deltas.size();
        std::vector<float> flog_likelihoods (n_poses_, 0);
        set_number_of_poses(n_poses_);

        // transform occlusion indices from size_t to int
        std::vector<int> occlusion_indices_transformed (occlusion_indices.size(), 0);
        for (size_t i = 0; i < occlusion_indices.size(); i++)
        {
            occlusion_indices_transformed[i] = (int) occlusion_indices[i];
        }

        INIT_PROFILING;
        // copy occlusion indices to GPU
        cuda_->set_prev_sample_indices(occlusion_indices_transformed.data());
        MEASURE("gpu: setting occlusion indices");
        // convert to internal state format
        std::vector<std::vector<std::vector<float> > > poses(
                    n_poses_,
                    std::vector<std::vector<float> >(vertices_.size(),
                    std::vector<float>(7, 0)));

        MEASURE("gpu: creating state vectors");


        for(size_t i_state = 0; i_state < size_t(n_poses_); i_state++)
        {
            for(size_t i_obj = 0; i_obj < vertices_.size(); i_obj++)
            {
                auto pose_0 = this->default_poses_.component(i_obj);
                auto delta = deltas[i_state].component(i_obj);

                fl::PoseVector pose;
                pose.orientation() = delta.orientation() * pose_0.orientation();
                pose.position() = delta.position() + pose_0.position();

                poses[i_state][i_obj][0] = pose.orientation().quaternion().w();
                poses[i_state][i_obj][1] = pose.orientation().quaternion().x();
                poses[i_state][i_obj][2] = pose.orientation().quaternion().y();
                poses[i_state][i_obj][3] = pose.orientation().quaternion().z();
                poses[i_state][i_obj][4] = pose.position()[0];
                poses[i_state][i_obj][5] = pose.position()[1];
                poses[i_state][i_obj][6] = pose.position()[2];
            }
        }

        MEASURE("gpu: converting state format");



        opengl_->render(poses);

        MEASURE("gpu: rendering");


        cudaGraphicsMapResources(1, &combined_texture_resource_, 0);
        cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);
        cuda_->set_texture_array(texture_array_);
        cuda_->map_texture();
        MEASURE("gpu: mapping texture");

        cuda_->compare_multiple(update_occlusions, flog_likelihoods);
        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);

        MEASURE("gpu: computing likelihoods");


        if(update_occlusions)
        {
            for(size_t i_state = 0; i_state < occlusion_indices.size(); i_state++)
                occlusion_indices[i_state] = i_state;

            MEASURE("gpu: updating occlusions");
        }


        // convert
        RealArray log_likelihoods(flog_likelihoods.size());
        for(size_t i = 0; i < flog_likelihoods.size(); i++)
            log_likelihoods[i] = flog_likelihoods[i];

        return log_likelihoods;
    }


    void set_observation(const Observation& image){
        std::vector<float> std_measurement(image.size());

        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                std_measurement[row*image.cols() + col] = image(row, col);

        set_observation(std_measurement, this->delta_time_);
    }

    virtual void reset()
    {
        set_occlusions();
        observation_time_ = 0;
    }


    // TODO: this image should be in a different format BOTH OF THEM!!
    const std::vector<float> get_occlusions(size_t index) const
    {
        std::vector<float> visibility_probs = cuda_->get_visibility_probabilities((int) index);
        return visibility_probs;
    }

    void get_range_image(std::vector<std::vector<int> > &intersect_indices,
                    std::vector<std::vector<float> > &depth)
    {
        opengl_->get_depth_values(intersect_indices, depth);
    }

private:
    // TODO: this function should disappear, BOTH OF THEM
    void set_observation(const std::vector<float>& observations, const Scalar& delta_time)
    {
        observation_time_ += delta_time;

        cuda_->set_observations(observations.data(), observation_time_);
        observations_set_ = true;

    }

    void set_occlusions(const float& visibility_prob = -1)
    {
        float default_visibility_probability = visibility_prob;
        if (visibility_prob == -1) default_visibility_probability = initial_visibility_prob_;

        std::vector<float> visibility_probabilities (n_rows_ * n_cols_ * n_poses_, default_visibility_probability);
        cuda_->set_visibility_probabilities(visibility_probabilities.data());
        // TODO set update times if you want to use them

    }

    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;

    void set_number_of_poses(int n_poses){

        n_poses_ = n_poses;
        opengl_->set_number_of_poses(n_poses_);
        n_poses_x_ = opengl_->get_n_poses_x();
        cuda_->set_number_of_poses(n_poses_, n_poses_x_);

    }

    void check_cuda_error(const char *msg)
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
            fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
            exit(EXIT_FAILURE);
        }
    }


    void unregister_resource()
    {
        if (resource_registered_) {
            cudaGraphicsUnregisterResource(combined_texture_resource_);
            check_cuda_error("cudaGraphicsUnregisterResource");
            resource_registered_ = false;
        }
    }

    void register_resource()
    {
        if (!resource_registered_) {
            combined_texture_opengl_ = opengl_->get_combined_texture();
            cudaGraphicsGLRegisterImage(&combined_texture_resource_, combined_texture_opengl_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
            check_cuda_error("cudaGraphicsGLRegisterImage)");
            resource_registered_ = true;
        }
    }


    boost::shared_ptr<ObjectRasterizer> opengl_;
    boost::shared_ptr<fil::CudaFilter> cuda_;

    // arrays for timings
    std::vector<std::vector<double> > cpu_times_;
    std::vector<std::vector<float> > cuda_times_;

    // data
    std::vector<std::vector<Eigen::Vector3f> > vertices_;
    std::vector<std::vector<std::vector<int> > > indices_;
    std::vector<float> visibility_probs_;

    // constants for likelihood evaluation
    float p_visible_visible_;
    float p_visible_occluded_;
    float tail_weight_;
    float model_sigma_;
    float sigma_factor_;
    float max_depth_;
    float exponential_rate_;

    std::string vertex_shader_path_;
    std::string fragment_shader_path_;


    double start_time_;

    int n_poses_;
    int n_poses_x_;

    int count_;

    // booleans to ensure correct usage of function calls
    bool observations_set_, resource_registered_;
    int nr_calls_set_observation_;

    // Shared resource between OpenGL and CUDA
    GLuint combined_texture_opengl_;
    cudaGraphicsResource* combined_texture_resource_;
    cudaArray_t texture_array_;

    double observation_time_;

    // used for time observations
    static const int TIME_MEASUREMENTS_COUNT = 8;
    static const int COUNT = 500;
    enum time_measurement {SEND_OBSERVATIONS, RENDER, MAP_RESOURCE, GET_MAPPED_ARRAY, SET_TEXTURE_ARRAY,
                           MAP_TEXTURE, COMPUTE_LIKELIHOODS, UNMAP_RESOURCE};

};

}
#endif
