#ifndef CUDA_FILTER_HPP
#define CUDA_FILTER_HPP

// #include "state_filter.hpp"
#include "boost/shared_ptr.hpp"
#include <curand_kernel.h>
#include "GL/glut.h"

#include <vector>

namespace fil {

class CudaFilter
{
public:
    CudaFilter();
    ~CudaFilter();

    void Init(std::vector<std::vector<float> > com_models, float angle_sigma, float trans_sigma,
              float p_visible_init, float c, float log_c, float p_visible_occluded,
              float tail_weight, float model_sigma, float sigma_factor, float max_depth, float exponential_rate);

    // filter functions
    void Propagate(const float &current_time, std::vector<std::vector<float> > &states);
    void PropagateMultiple(const float &current_time, std::vector<std::vector<std::vector<float> > > &states);
    void Compare(float observation_time, bool constant_occlusion, std::vector<float> &log_likelihoods);
    void CompareMultiple(bool update, std::vector<float> &log_likelihoods);
    void Resample(std::vector<int> resampling_indices);
    void ResampleMultiple(std::vector<int> resampling_indices);


    // setters
    void set_states(std::vector<std::vector<float> > &states, int seed);
    void set_states_multiple(int n_objects, int n_features, int seed);
    void set_observations(const float* observations, const float observation_time);
    void set_observations(const float* observations);
    void set_visibility_probabilities(const float* visibility_probabilities);
    void set_prev_sample_indices(const int* prev_sample_indices);
    void set_resolution(int n_rows, int n_cols);
    void set_number_of_max_poses(int n_poses, int n_poses_x);
    void set_number_of_poses(int n_poses, int n_poses_x);
    void set_texture_array(cudaArray_t texture_array);

    // getters
    std::vector<float> get_visibility_probabilities(int state_id);
    std::vector<std::vector<float> > get_visibility_probabilities();

    void MapTexture();
    void destroy_context();

private:
    // resolution values if not specified
    static const int WINDOW_WIDTH = 80;
    static const int WINDOW_HEIGHT = 60;

    // time measurement
    static const int COUNT = 500;
    int count_;
    double compare_kernel_time_;
    double copy_likelihoods_time_;

    // device pointers to arrays stored in global memory on the GPU
    float *d_states_;
    float *d_states_copy_;
    float *d_visibility_probs_;
    float *d_visibility_probs_copy_;
    float *d_observations_;
    float *d_log_likelihoods_;
    int *d_prev_sample_indices_;
    int *d_resampling_indices_;
    float *d_test_array_;
    curandStateMRG32k3a *d_mrg_states_;
    
    // for OpenGL interop
    cudaArray_t d_texture_array_;

    // resolution
    int n_cols_;
    int n_rows_;

    // maximum number of poses and their arrangement in the OpenGL texture
    int n_max_poses_;
    int n_max_poses_x_;
    int n_max_poses_y_;

    // number of poses and their arrangement in the OpenGL texture
    int n_poses_;
    int n_poses_x_;
    int n_poses_y_;

    // number of features in a state vector
    int n_features_;

    // block and grid arrangement of the CUDA kernels
    int n_threads_, n_blocks_;

    // system properties
    int warp_size_;
    int n_mps_;

    // visibility prob default
    float visibility_prob_default_;

    // time values to compute the time deltas when calling propagate() or evaluate()
    float occlusion_time_;
    float observation_time_;

//    float delta_time_;
    float last_propagation_time_;

    // booleans to describe the state of the cuda filter, to avoid wrong usage of the class
    bool n_poses_set_;


    void set_default_kernel_config();

    // helper functions
    template <typename T> void allocate(T * &pointer, size_t size, char* name);
    void coutArray(float* array, int size, char* name);
    void coutArray(std::vector<float> array, char* name);
    void checkCUDAError(const char *msg);

};

}

#endif // CUDAFILTER_HPP
