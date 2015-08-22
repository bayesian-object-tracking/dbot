#ifndef POSE_TRACKING_MODELS_OBSERVATION_MODELS_KINECT_IMAGE_OBSERVATION_MODEL_GPU_HACK_HPP
#define POSE_TRACKING_MODELS_OBSERVATION_MODELS_KINECT_IMAGE_OBSERVATION_MODEL_GPU_HACK_HPP

#include <vector>
#include "boost/shared_ptr.hpp"
#include "Eigen/Core"


#include <dbot/models/observation_models/rao_blackwell_observation_model.hpp>
#include <dbot/states/free_floating_rigid_bodies_state.hpp>

#include <dbot/models/observation_models/kinect_image_observation_model_gpu/object_rasterizer.hpp>
#include <dbot/models/observation_models/kinect_image_observation_model_gpu/cuda_filter.hpp>

#include <omp.h>


#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_gl_interop.h>

#include <dbot/utils/profiling.hpp>

namespace ff
{

// Forward declarations
template <typename State> class KinectImageObservationModelGPUHack;

namespace internal
{
/**
 * ImageObservationModelCPU distribution traits specialization
 * \internal
 */
template <typename State>
struct Traits<KinectImageObservationModelGPUHack<State> >
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef RaoBlackwellObservationModel<State, Observation> Base;

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
class KinectImageObservationModelGPUHack:
        public internal::Traits<KinectImageObservationModelGPUHack<State> >::Base
{
public:
    typedef internal::Traits<KinectImageObservationModelGPUHack<State> > Traits;

    typedef typename Traits::Scalar         Scalar;
    typedef typename Traits::Observation    Observation;
    typedef typename Traits::CameraMatrix   CameraMatrix;

    typedef typename Traits::Base::StateArray StateArray;
    typedef typename Traits::Base::RealArray RealArray;
    typedef typename Traits::Base::IntArray IntArray;


    // TODO: ALL THIS SHOULD SWITCH FROM USING VISIBILITY TO OCCLUSION
    KinectImageObservationModelGPUHack(const CameraMatrix& camera_matrix,
                     const size_t& n_rows,
                     const size_t& n_cols,
                     const size_t& max_sample_count,
                     const Scalar& initial_occlusion_prob,
                     const double& delta_time):
            camera_matrix_(camera_matrix),
            n_rows_(n_rows),
            n_cols_(n_cols),
            initial_visibility_prob_(1 - initial_occlusion_prob),
            max_sample_count_(max_sample_count),
            n_poses_(max_sample_count),
            constants_set_(false),
            initialized_(false),
            observations_set_(false),
            resource_registered_(false),
            nr_calls_set_observation_(0),
            observation_time_(0),
            Traits::Base(delta_time)
    {
        visibility_probs_.resize(n_rows_ * n_cols_);
    }

    ~KinectImageObservationModelGPUHack() { }

    // TODO: DO WE NEED TWO DIFFERENT FUNCTIONS FOR THIS??
    void Initialize(const KinematicsFromURDF& kinematics)
    {
        if (constants_set_)
        {


            // TODO: this is a disgusting hack!!!
            std::cout << "warning: copying kinematics in gpu observation model, this is a hack!!!" << std::endl;
            std::cout << "resizing " << std::endl;
            kinematics_.resize(30);

            for(size_t i = 0; i < 30; i++)
            {
                std::cout << "iteration " << i << std::endl;
                kinematics_[i] = kinematics;
            }
            std::cout << "done copying kinematics" << std::endl;











            opengl_ = boost::shared_ptr<ObjectRasterizer>
                    (new ObjectRasterizer(vertices_,
                                          indices_,
                                          vertex_shader_,
                                          fragment_shader_));
            cuda_ = boost::shared_ptr<fil::CudaFilter> (new fil::CudaFilter());

            initialized_ = true;


            opengl_->PrepareRender(camera_matrix_.cast<float>());


            opengl_->set_number_of_max_poses(max_sample_count_);
            n_poses_x_ = opengl_->get_n_poses_x();
            cuda_->set_number_of_max_poses(max_sample_count_, n_poses_x_);


            std:: cout << "set resolution in cuda..." << std::endl;

            opengl_->set_resolution(n_rows_, n_cols_);
            cuda_->set_resolution(n_rows_, n_cols_);

            RegisterResource();

            std:: cout << "set occlusions..." << std::endl;

    //        set_occlusions();
            Reset();

            float c = p_visible_visible_ - p_visible_occluded_;
            float log_c = log(c);

            std::vector<std::vector<float> > dummy_com_models;
            cuda_->Init(dummy_com_models, 0.0f, 0.0f,
                        initial_visibility_prob_, c, log_c, p_visible_occluded_,
                        tail_weight_, model_sigma_, sigma_factor_, max_depth_, exponential_rate_);


            count_ = 0;

        } else {
            std:: cout << "WARNING: GPUImageObservationModel::Initialize() was not executed, because GPUImageObservationModel::set_constants() has not been called previously." << std::endl;
        }
    }

    void Constants(const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
                   const std::vector<std::vector<std::vector<int> > > indices,
                   const float p_occluded_visible,
                   const float p_occluded_occluded,
                   const float tail_weight,
                   const float model_sigma,
                   const float sigma_factor,
                   const float max_depth,
                   const float exponential_rate,
                   const std::string vertex_shader,
                   const std::string fragment_shader)
    {

        vertex_shader_ = vertex_shader;
        fragment_shader_ = fragment_shader;


        // since you love doubles i changed the argument type of the vertices to double and convert it here :)
        vertices_.resize(vertices_double.size());
        for(size_t object_index = 0; object_index < vertices_.size(); object_index++)
        {
            vertices_[object_index].resize(vertices_double[object_index].size());
            for(size_t vertex_index = 0; vertex_index < vertices_[object_index].size(); vertex_index++)
                vertices_[object_index][vertex_index] = vertices_double[object_index][vertex_index].cast<float>();
        }


        indices_ = indices;
        p_visible_visible_ = 1.0 - p_occluded_visible;
        p_visible_occluded_ = 1.0 - p_occluded_occluded;
        tail_weight_ = tail_weight;
        model_sigma_ = model_sigma;
        sigma_factor_ = sigma_factor;
        max_depth_ = max_depth;
        exponential_rate_ = exponential_rate;


        constants_set_ = true;
    }

    RealArray Loglikes(const StateArray& states,
                                  IntArray& occlusion_indices,
                                  const bool& update_occlusions = false)
    {
        if (!initialized_)

        {
            std:: cout << "GPU: not initialized" << std::endl;
            exit(-1);
        }
        else if(!observations_set_)
        {
            std:: cout << "GPU: observations not set" << std::endl;
            exit(-1);
        }

        n_poses_ = states.size();
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
        std::vector<std::vector<std::vector<float> > > states_internal_format(
                    n_poses_,
                    std::vector<std::vector<float> >(states[0].count(),
                    std::vector<float>(7, 0)));

        MEASURE("gpu: creating state vectors");


// TODO: this is a hack!!!
#pragma omp parallel for
        for(size_t state_index = 0; state_index < size_t(n_poses_); state_index++)
        {

            int tid = omp_get_thread_num();

            kinematics_[tid].InitKDLData(states[state_index]);

            for(size_t body_index = 0; body_index < states[state_index].count(); body_index++)
            {
                const Eigen::Quaternion<Scalar>& quaternion = kinematics_[tid].GetLinkOrientation(body_index);
                states_internal_format[state_index][body_index][0] = quaternion.w();
                states_internal_format[state_index][body_index][1] = quaternion.x();
                states_internal_format[state_index][body_index][2] = quaternion.y();
                states_internal_format[state_index][body_index][3] = quaternion.z();
                const Eigen::Matrix<Scalar, 3, 1>& position =  kinematics_[tid].GetLinkPosition(body_index);
                states_internal_format[state_index][body_index][4] = position[0];
                states_internal_format[state_index][body_index][5] = position[1];
                states_internal_format[state_index][body_index][6] = position[2];
            }
        }

        MEASURE("gpu: converting state format");



        opengl_->Render(states_internal_format);

        MEASURE("gpu: rendering");


        cudaGraphicsMapResources(1, &combined_texture_resource_, 0);
        cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);
        cuda_->set_texture_array(texture_array_);
        cuda_->MapTexture();
        MEASURE("gpu: mapping texture");

        cuda_->CompareMultiple(update_occlusions, flog_likelihoods);
        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);

        MEASURE("gpu: computing likelihoods");


        if(update_occlusions)
        {
            for(size_t state_index = 0; state_index < occlusion_indices.size(); state_index++)
                occlusion_indices[state_index] = state_index;

            MEASURE("gpu: updating occlusions");
        }


        // convert
        RealArray log_likelihoods(flog_likelihoods.size());
        for(size_t i = 0; i < flog_likelihoods.size(); i++)
            log_likelihoods[i] = flog_likelihoods[i];

        return log_likelihoods;
    }


    void SetObservation(const Observation& image){
        std::vector<float> std_measurement(image.size());

        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                std_measurement[row*image.cols() + col] = image(row, col);

        SetObservation(std_measurement, this->delta_time_);
    }

    virtual void Reset()
    {
        Occlusions();
        observation_time_ = 0;
    }


    // TODO: this image should be in a different format BOTH OF THEM!!
    const std::vector<float> Occlusions(size_t index) const
    {
        std::vector<float> visibility_probs = cuda_->get_visibility_probabilities((int) index);
        return visibility_probs;
    }

    void RangeImage(std::vector<std::vector<int> > &intersect_indices,
                    std::vector<std::vector<float> > &depth)
    {
        opengl_->get_depth_values(intersect_indices, depth);
    }

private:
    // TODO: this function should disappear, BOTH OF THEM
    void SetObservation(const std::vector<float>& observations, const Scalar& delta_time)
    {
        observation_time_ += delta_time;
        if (initialized_)
        {
            cuda_->set_observations(observations.data(), observation_time_);
            observations_set_ = true;
        }
    }

    void Occlusions(const float& visibility_prob = -1)
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
        if (initialized_) {
            n_poses_ = n_poses;
            opengl_->set_number_of_poses(n_poses_);
            n_poses_x_ = opengl_->get_n_poses_x();
            cuda_->set_number_of_poses(n_poses_, n_poses_x_);
        } else {
            std:: cout << "WARNING: GPUImageObservationModel::set_number_of_poses() was not executed, because GPUImageObservationModel::Initialize() has not been called previously." << std::endl;
        }
    }

    void checkCUDAError(const char *msg)
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
            fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
            exit(EXIT_FAILURE);
        }
    }


    void UnregisterResource()
    {
        if (resource_registered_) {
            cudaGraphicsUnregisterResource(combined_texture_resource_);
            checkCUDAError("cudaGraphicsUnregisterResource");
            resource_registered_ = false;
        }
    }

    void RegisterResource()
    {
        if (!resource_registered_) {
            combined_texture_opengl_ = opengl_->get_combined_texture();
            cudaGraphicsGLRegisterImage(&combined_texture_resource_, combined_texture_opengl_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
            checkCUDAError("cudaGraphicsGLRegisterImage)");
            resource_registered_ = true;
        }
    }


    std::vector<KinematicsFromURDF> kinematics_;

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


    std::string vertex_shader_;
    std::string fragment_shader_;

    double start_time_;

    int n_poses_;
    int n_poses_x_;

    int count_;

    // booleans to ensure correct usage of function calls
    bool constants_set_, initialized_, observations_set_, resource_registered_;
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
