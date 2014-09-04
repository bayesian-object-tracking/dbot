/** @author Claudia Pfreundt */

//#define PROFILING_ACTIVE

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <boost/timer.hpp>

#include <pose_tracking/models/observers/image_observer_gpu/cuda_opengl_filter.hpp>
#include <pose_tracking/models/observers/image_observer_gpu/object_rasterizer.hpp>
#include <pose_tracking/models/observers/image_observer_gpu/cuda_filter.hpp>


#include <cuda_gl_interop.h>
#include <fast_filtering/utils/helper_functions.hpp>

using namespace std;
using namespace Eigen;
using namespace fil;


CudaOpenglFilter::CudaOpenglFilter() :
    constants_set_(false),
    resolution_set_(false),
    states_set_(false),
    initialized_(false),
    observations_set_(false),
    resource_registered_(false)
{
}

void CudaOpenglFilter::Initialize() {
    if (constants_set_) {
        opengl_ = boost::shared_ptr<ObjectRasterizer> (new ObjectRasterizer(vertices_, indices_));
        cuda_ = boost::shared_ptr<CudaFilter> (new CudaFilter());        

        initialized_ = true;
        set_number_of_poses(n_init_poses_);

//        combined_texture_opengl_ = opengl_->get_combined_texture();




        float c = p_visible_visible_ - p_visible_occluded_;
        float log_c = log(c);

        cuda_->Init(com_models_, angle_sigma_, trans_sigma_,
                    p_visible_init_, c, log_c, p_visible_occluded_,
                    tail_weight_, model_sigma_, sigma_factor_, max_depth_, exponential_rate_);


        log_likelihoods_.resize(n_init_poses_);
        count_ = 0;

    } else {
        cout << "WARNING: CudaOpenglFilter::Initialize() was not executed, because CudaOpenglFilter::set_constants() has not been called previously." << endl;
    }
}







void CudaOpenglFilter::Propagate(const float &current_time, vector<vector<float> > &states) {
    if (initialized_ && states_set_) {
        cuda_->Propagate(current_time, states);
    } else {
        cout << "WARNING: CudaOpenglFilter::Propagate() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}



void CudaOpenglFilter::Evaluate(
        const vector<vector<vector<float> > > &states,
        const vector<float> &observations,
        vector<float> &log_likelihoods)
{
    if (initialized_) {
        cuda_->set_observations(observations.data());
        opengl_->Render(states);

        // map the resources in CUDA
        cudaGraphicsMapResources(1, &combined_texture_resource_, 0);
        cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);

        cuda_->set_texture_array(texture_array_);
        cuda_->MapTexture();
        float observation_time = 0;

        cuda_->Compare(observation_time, true, log_likelihoods);
        log_likelihoods_ = log_likelihoods;

        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);

        // change to regular number of poses
        set_number_of_poses(n_regular_poses_);

        // TODO don't remember what this is good for? Should it not stay the same size until after resampling?
        log_likelihoods_.resize(n_poses_);
    } else {
        cout << "WARNING: CudaOpenglFilter::Evaluate() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}



void CudaOpenglFilter::Evaluate(
            const vector<vector<vector<float> > > &states,
            const vector<float> &observations,
            const float &observation_time,
            vector<float> &log_likelihoods)
{
    if (initialized_) {

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
        start = sf::hf::get_wall_time();
#endif

        cuda_->set_observations(observations.data());

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif


        opengl_->Render(states);

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

//         map the resources in CUDA
        cudaGraphicsMapResources(1, &combined_texture_resource_, 0);

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

        cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

        cuda_->set_texture_array(texture_array_);

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

        cuda_->MapTexture();

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

        cuda_->Compare(observation_time, false, log_likelihoods);
        log_likelihoods_ = log_likelihoods;

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
        cpu_times.push_back(stop - start);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        cuda_times.push_back(milliseconds);
        cudaEventRecord(start_event);
        start = sf::hf::get_wall_time();
#endif

        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);

#ifdef PROFILING_ACTIVE
        stop = sf::hf::get_wall_time();
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
            names[SEND_OBSERVATIONS] = "send observations";
            names[RENDER] = "render poses";
            names[MAP_RESOURCE] = "map resource";
            names[GET_MAPPED_ARRAY] = "get mapped array";
            names[SET_TEXTURE_ARRAY] = "set texture array";
            names[MAP_TEXTURE] = "map texture";
            names[COMPUTE_LIKELIHOODS] = "compute likelihoods";
            names[UNMAP_RESOURCE] = "unmap resource";

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
            for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {
                cout << names[i] << ": \t(GPU) " << final_cuda_times[i] * 1e3 / count_<< "\t(CPU) " << final_cpu_times[i] * 1e6 / count_<< endl;
                total_time_cuda += final_cuda_times[i] * 1e3 / count_;
                total_time_cpu += final_cpu_times[i] * 1e6 / count_;
            }
            cout << "TOTAL: " << "\t(GPU) " << total_time_cuda << "\t(CPU) " << total_time_cpu << endl;
        }
#endif

    } else {
        cout << "WARNING: CudaOpenglFilter::Evaluate() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}



// only use if log_likelihoods were assigned after Evalute() step
vector<int> CudaOpenglFilter::Resample()
{
    vector<int> resampling_indices;
    sf::hf::DiscreteSampler sampler(log_likelihoods_);

    for (int i = 0; i < n_poses_; i++) {
        resampling_indices.push_back(sampler.Sample());
    }

    if (initialized_ && states_set_) {
        cuda_->Resample(resampling_indices);
    } else {
        cout << "WARNING: CudaOpenglFilter::Resample() was not executed, because CudaOpenglFilter::Initialize() or ::set_states() has not been called previously." << endl;
    }

    return resampling_indices;
}




// ===================================================================================== //
// ====================================  SETTERS ======================================= //
// ===================================================================================== //


void CudaOpenglFilter::set_observations(const vector<float> &observations) {
    if (initialized_) {
        cuda_->set_observations(observations.data());
        observations_set_ = true;
        // TODO observations_set_ = false in some function.. evaluate or resample
    }
}

void CudaOpenglFilter::set_constants(const std::vector<std::vector<Eigen::Vector3f> > vertices,
                                     const std::vector<std::vector<std::vector<int> > > indices,
                                     const int n_init_poses,
                                     const int n_poses,
                                     const float angle_sigma,
                                     const float trans_sigma,
                                     const std::vector<std::vector<float> > com_models,
                                     const Eigen::Matrix3f camera_matrix,
                                     const float p_visible_init,
                                     const float p_visible_visible,
                                     const float p_visible_occluded,
                                     const float tail_weight,
                                     const float model_sigma,
                                     const float sigma_factor,
                                     const float max_depth,
                                     const float exponential_rate) {
    vertices_ = vertices;
    indices_ = indices;
    n_init_poses_ = n_init_poses;
    n_regular_poses_ = n_poses;
    angle_sigma_ = angle_sigma;
    trans_sigma_ = trans_sigma;
    com_models_ = com_models;
    camera_matrix_ = camera_matrix;
    p_visible_init_ = p_visible_init;
    p_visible_visible_ = p_visible_visible;
    p_visible_occluded_ = p_visible_occluded;
    tail_weight_ = tail_weight;
    model_sigma_ = model_sigma;
    sigma_factor_ = sigma_factor;
    max_depth_ = max_depth;
    exponential_rate_ = exponential_rate;


    constants_set_ = true;
}

void CudaOpenglFilter::set_number_of_poses(int n_poses) {
    if (initialized_) {
        // resource needs to be unregistered so that OpenGL can reallocate the texture with a different size
        UnregisterResource();

        n_poses_ = n_poses;
        opengl_->set_number_of_max_poses(n_poses_);
        opengl_->PrepareRender(camera_matrix_);
        n_poses_x_ = opengl_->get_n_poses_x();

        cuda_->set_number_of_max_poses(n_poses_, n_poses_x_);


        RegisterResource();
    } else {
        cout << "WARNING: CudaOpenglFilter::set_number_of_poses() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}

void CudaOpenglFilter::set_resolution(int n_cols, int n_rows) {
    if (initialized_) {
        n_cols_ = n_cols;
        n_rows_ = n_rows;
        // resource needs to be unregistered so that OpenGL can reallocate the texture with a different size
        UnregisterResource();

        opengl_->set_resolution(n_rows_, n_cols_);
        cuda_->set_resolution(n_rows_, n_cols_);

        resolution_set_ = true;

        RegisterResource();
    } else {
        cout << "WARNING: CudaOpenglFilter::set_resolution() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}



void CudaOpenglFilter::set_states(vector<vector<float> > states, int seed) {
    if (initialized_) {
        cuda_->set_states(states, seed);
        states_set_ = true;
    } else {
        cout << "WARNING: CudaOpenglFilter::set_states() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}





void CudaOpenglFilter::UnregisterResource() {
    if (resource_registered_) {
        cudaGraphicsUnregisterResource(combined_texture_resource_);
        checkCUDAError("cudaGraphicsUnregisterResource");
        resource_registered_ = false;
    }
}


void CudaOpenglFilter::RegisterResource() {
    if (!resource_registered_) {
        combined_texture_opengl_ = opengl_->get_combined_texture();
        cudaGraphicsGLRegisterImage(&combined_texture_resource_, combined_texture_opengl_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        checkCUDAError("cudaGraphicsGLRegisterImage)");
        resource_registered_ = true;
    }
}






void CudaOpenglFilter::checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}



CudaOpenglFilter::~CudaOpenglFilter() {
//    delete combined_texture_resource_;
}

void CudaOpenglFilter::destroy_context() {
    cuda_->destroy_context();
}






















