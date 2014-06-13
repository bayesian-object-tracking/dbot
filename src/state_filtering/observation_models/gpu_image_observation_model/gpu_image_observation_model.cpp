
#include <state_filtering/observation_models/gpu_image_observation_model/gpu_image_observation_model.hpp>
#include <state_filtering/observation_models/gpu_image_observation_model/object_rasterizer.hpp>
#include <state_filtering/observation_models/gpu_image_observation_model/cuda_filter.hpp>

#include <state_filtering/tools/helper_functions.hpp>
#include <state_filtering/tools/macros.hpp>

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_gl_interop.h>


using namespace std;
using namespace Eigen;
using namespace obs_mod;


GPUImageObservationModel::GPUImageObservationModel(
        const Eigen::Matrix3d& camera_matrix,
        const size_t& n_rows,
        const size_t& n_cols,
        const size_t& max_sample_count,
        const float& initial_visibility_prob,
        const boost::shared_ptr<RigidBodySystem<-1> >& rigid_body_system):
    ImageObservationModel(camera_matrix, n_rows, n_cols, initial_visibility_prob, max_sample_count, rigid_body_system),

    n_poses_(max_sample_count),
    constants_set_(false),
    initialized_(false),
    observations_set_(false),
    resource_registered_(false),
    nr_calls_set_observation_(0)
{
    cout << "resize visprobs" << endl;
    visibility_probs_.resize(n_rows_ * n_cols_);
    cout << "successfully resized visprobs" << endl;
}


GPUImageObservationModel::~GPUImageObservationModel() { }



void GPUImageObservationModel::Initialize() {
    if (constants_set_) {
        opengl_ = boost::shared_ptr<ObjectRasterizer> (new ObjectRasterizer(vertices_, indices_));
        cuda_ = boost::shared_ptr<fil::CudaFilter> (new fil::CudaFilter());

        initialized_ = true;


        opengl_->PrepareRender(camera_matrix_.cast<float>());


        opengl_->set_number_of_max_poses(max_sample_count_);
        n_poses_x_ = opengl_->get_n_poses_x();
        cuda_->set_number_of_max_poses(max_sample_count_, n_poses_x_);


        cout << "set resolution in cuda..." << endl;

        opengl_->set_resolution(n_rows_, n_cols_);
        cuda_->set_resolution(n_rows_, n_cols_);

        RegisterResource();

        cout << "set occlusions..." << endl;

        set_occlusions();

        float c = p_visible_visible_ - p_visible_occluded_;
        float log_c = log(c);

        vector<vector<float> > dummy_com_models;
        cuda_->Init(dummy_com_models, 0.0f, 0.0f,
                    initial_visibility_prob_, c, log_c, p_visible_occluded_,
                    tail_weight_, model_sigma_, sigma_factor_, max_depth_, exponential_rate_);


        count_ = 0;

    } else {
        cout << "WARNING: GPUImageObservationModel::Initialize() was not executed, because GPUImageObservationModel::set_constants() has not been called previously." << endl;
    }
}


std::vector<float> GPUImageObservationModel::Evaluate(
        const std::vector<Eigen::VectorXd>& states,
        std::vector<size_t>& occlusion_indices,
        const bool& update_occlusions)
{
    n_poses_ = states.size();
    vector<float> log_likelihoods (n_poses_, 0);

    if (initialized_ && observations_set_) {


#ifdef PROFILING_ACTIVE
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        float milliseconds;
        double start;
        double stop;
        vector<float> cuda_times;
        vector<double> cpu_times;
        cudaEventRecord(start_event);
        start = hf::get_wall_time();
#endif

//        cout << "setting number of poses to " << n_poses_ << endl;

        set_number_of_poses(n_poses_);

        // transform occlusion indices from size_t to int
        vector<int> occlusion_indices_transformed (occlusion_indices.size(), 0);
        for (size_t i = 0; i < occlusion_indices.size(); i++) {
            occlusion_indices_transformed[i] = (int) occlusion_indices[i];
        }

        // copy occlusion indices to GPU
        cuda_->set_prev_sample_indices(occlusion_indices_transformed.data());

        // transform state format
        vector<vector<vector<float> > > states_transformed (n_poses_,
                                                            vector<vector<float> >(rigid_body_system_->bodies_size(),
                                                            vector<float>(7, 0)));
        for(size_t state_index = 0; state_index < size_t(n_poses_); state_index++)
        {
            *rigid_body_system_ = states[state_index];
            for(size_t body_index = 0; body_index < rigid_body_system_->bodies_size(); body_index++)
            {
                states_transformed[state_index][body_index][0] = rigid_body_system_->quaternion(body_index).w();
                states_transformed[state_index][body_index][1] = rigid_body_system_->quaternion(body_index).x();
                states_transformed[state_index][body_index][2] = rigid_body_system_->quaternion(body_index).y();
                states_transformed[state_index][body_index][3] = rigid_body_system_->quaternion(body_index).z();
                states_transformed[state_index][body_index][4] = rigid_body_system_->position(body_index)[0];
                states_transformed[state_index][body_index][5] = rigid_body_system_->position(body_index)[1];
                states_transformed[state_index][body_index][6] = rigid_body_system_->position(body_index)[2];
            }
        }

#ifdef PROFILING_ACTIVE
        stop = hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = hf::get_wall_time();
#endif

        opengl_->Render(states_transformed);

#ifdef PROFILING_ACTIVE
        stop = hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = hf::get_wall_time();
#endif


        cudaGraphicsMapResources(1, &combined_texture_resource_, 0);
        cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);
        cuda_->set_texture_array(texture_array_);
        cuda_->MapTexture();


#ifdef PROFILING_ACTIVE
        stop = hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = hf::get_wall_time();
#endif


        cuda_->CompareMultiple(update_occlusions, log_likelihoods);
        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);

        if(update_occlusions) {
            for(size_t state_index = 0; state_index < occlusion_indices.size(); state_index++)
                occlusion_indices[state_index] = state_index;
        }


#ifdef PROFILING_ACTIVE
        stop = hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);

        cpu_times_.push_back(cpu_times);
        cuda_times_.push_back(cuda_times);
        count_++;

        if (count_ == COUNT) {
            string names[TIME_MEASUREMENTS_COUNT];
            names[SEND_INDICES] = "send indices";
            names[RENDER] = "render poses";
            names[MAP_RESOURCE] = "map resource";
            names[COMPUTE_LIKELIHOODS] = "compute likelihoods";

            float final_cpu_times[TIME_MEASUREMENTS_COUNT] = {0};
            float final_cuda_times[TIME_MEASUREMENTS_COUNT] = {0};

            for (int i = 0; i < count_; i++) {
                for (int j = 0; j < TIME_MEASUREMENTS_COUNT; j++) {
                    final_cpu_times[j] += cpu_times_[i][j];
                    final_cuda_times[j] += cuda_times_[i][j];
                }
            }

            float total_time_cuda = 0;
            float total_time_cpu = 0;
            cout << "EvaluateMultiple() runtimes: " << endl;
            for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {
                cout << names[i] << ": \t(GPU) " << final_cuda_times[i] * 1e3 / count_<< "\t(CPU) " << final_cpu_times[i] * 1e6 / count_<< endl;
                total_time_cuda += final_cuda_times[i] * 1e3 / count_;
                total_time_cpu += final_cpu_times[i] * 1e6 / count_;
            }
            cout << "TOTAL: " << "\t(GPU) " << total_time_cuda << "\t(CPU) " << total_time_cpu << endl;
        }
#endif

    } else {
        cout << "WARNING: GPUImageObservationModel::EvaluateMultiple() was not executed, because GPUImageObservationModel::Initialize() or GPUImageObservationModel::set_observations() has not been called previously." << endl;
    }




    return log_likelihoods;
}




std::vector<float> GPUImageObservationModel::Evaluate_test(
        const std::vector<Eigen::VectorXd>& states,
        std::vector<size_t>& occlusion_indices,
        const bool& update_occlusions,
        vector<vector<int> > intersect_indices,
        vector<vector<float> > predictions) {
    vector<float> bla;
    return bla;
}



// ===================================================================================== //
// ====================================  SETTERS ======================================= //
// ===================================================================================== //


void GPUImageObservationModel::set_constants(
        const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
        const std::vector<std::vector<std::vector<int> > > indices,
        const float p_visible_visible,
        const float p_visible_occluded,
        const float tail_weight,
        const float model_sigma,
        const float sigma_factor,
        const float max_depth,
        const float exponential_rate) {


    // since you love doubles i changed the argument type of the vertices to double and convert it here :)
    vertices_.resize(vertices_double.size());
    for(size_t object_index = 0; object_index < vertices_.size(); object_index++)
    {
        vertices_[object_index].resize(vertices_double[object_index].size());
        for(size_t vertex_index = 0; vertex_index < vertices_[object_index].size(); vertex_index++)
            vertices_[object_index][vertex_index] = vertices_double[object_index][vertex_index].cast<float>();
    }


    indices_ = indices;
    p_visible_visible_ = p_visible_visible;
    p_visible_occluded_ = p_visible_occluded;
    tail_weight_ = tail_weight;
    model_sigma_ = model_sigma;
    sigma_factor_ = sigma_factor;
    max_depth_ = max_depth;
    exponential_rate_ = exponential_rate;


    constants_set_ = true;
}


void GPUImageObservationModel::set_number_of_poses(int n_poses) {
    if (initialized_) {
        n_poses_ = n_poses;
        opengl_->set_number_of_poses(n_poses_);
        n_poses_x_ = opengl_->get_n_poses_x();
        cuda_->set_number_of_poses(n_poses_, n_poses_x_);
    } else {
        cout << "WARNING: GPUImageObservationModel::set_number_of_poses() was not executed, because GPUImageObservationModel::Initialize() has not been called previously." << endl;
    }
}

void GPUImageObservationModel::set_occlusions(const float& visibility_prob)
{
    float default_visibility_probability = visibility_prob;
    if (visibility_prob == -1) default_visibility_probability = initial_visibility_prob_;

    vector<float> visibility_probabilities (n_rows_ * n_cols_ * n_poses_, default_visibility_probability);
    cuda_->set_visibility_probabilities(visibility_probabilities.data());
    // TODO set update times if you want to use them

}


void GPUImageObservationModel::measurement(const std::vector<float>& observations, const double& time_since_start)
{
    if (initialized_) {
        cuda_->set_observations(observations.data(), time_since_start);
        observations_set_ = true;
    }
}



// ===================================================================================== //
// ====================================  GETTERS ======================================= //
// ===================================================================================== //


const std::vector<float> GPUImageObservationModel::get_occlusions(size_t index) const
{
    vector<float> visibility_probs = cuda_->get_visibility_probabilities((int) index);
    return visibility_probs;
}


void GPUImageObservationModel::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                                std::vector<std::vector<float> > &depth)
{
    opengl_->get_depth_values(intersect_indices, depth);
}




// ===================================================================================== //
// ============================== OPENGL INTEROP STUFF ================================= //
// ===================================================================================== //


void GPUImageObservationModel::UnregisterResource() {
    if (resource_registered_) {
        cudaGraphicsUnregisterResource(combined_texture_resource_);
        checkCUDAError("cudaGraphicsUnregisterResource");
        resource_registered_ = false;
    }
}


void GPUImageObservationModel::RegisterResource() {
    if (!resource_registered_) {
        combined_texture_opengl_ = opengl_->get_combined_texture();
        cudaGraphicsGLRegisterImage(&combined_texture_resource_, combined_texture_opengl_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        checkCUDAError("cudaGraphicsGLRegisterImage)");
        resource_registered_ = true;
    }
}

// ===================================================================================== //
// ================================ HELPER FUNCTIONS =================================== //
// ===================================================================================== //


void GPUImageObservationModel::checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

