#ifndef POSE_TRACKING_MODELS_OBSERVATION_MODELS_CUDA_FILTER_HPP
#define POSE_TRACKING_MODELS_OBSERVATION_MODELS_CUDA_FILTER_HPP

#include "boost/shared_ptr.hpp"
#include <curand_kernel.h>
#include "GL/glut.h"

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

namespace fil
{
class CudaFilter
{
   public:
    CudaFilter();
    ~CudaFilter();

    void init(std::vector<std::vector<float> > com_models,
              float angle_sigma,
              float trans_sigma,
              float p_visible_init,
              float c,
              float log_c,
              float p_visible_occluded,
              float tail_weight,
              float model_sigma,
              float sigma_factor,
              float max_depth,
              float exponential_rate);

    // filter functions
    void propagate(const float& current_time,
                   std::vector<std::vector<float> >& states);  // not used
    void propagate_multiple(
        const float& current_time,
        std::vector<std::vector<std::vector<float> > >& states);  // not used
    void compare(float observation_time,
                 bool constant_occlusion,
                 std::vector<float>& log_likelihoods);  // not used
    void compare_multiple(bool update, std::vector<float>& log_likelihoods);
    void resample(std::vector<int> resampling_indices);           // not used
    void resample_multiple(std::vector<int> resampling_indices);  // not used

    // setters
    void set_states(std::vector<std::vector<float> >& states,
                    int seed);  // not needed if propagation not on GPU
    void set_states_multiple(int n_objects,
                             int n_features,
                             int seed);  // not needed if propagation not on GPU
    void set_observations(const float* observations,
                          const float observation_time);
    void set_observations(const float* observations);  // not used outside, can
                                                       // be integrated into
                                                       // above
    void set_visibility_probabilities(const float* visibility_probabilities);
    void set_prev_sample_indices(const int* prev_sample_indices);
    void set_resolution(const int n_rows,
                        const int n_cols,
                        int& nr_poses,
                        int& nr_poses_per_row,
                        int& nr_poses_per_column);
    void allocate_memory_for_max_poses(int& allocated_poses,
                                       int& allocated_poses_per_row,
                                       int& allocated_poses_per_column);
    void set_number_of_poses(int& nr_poses,
                             int& nr_poses_per_row,
                             int& nr_poses_per_column);
    void set_texture_array(cudaArray_t texture_array);

    // getters
    std::vector<float> get_visibility_probabilities(int state_id);
    std::vector<std::vector<float> > get_visibility_probabilities();  // returns
                                                                      // all of
                                                                      // them.
                                                                      // Ask
                                                                      // Manuel
                                                                      // if they
                                                                      // could
                                                                      // need
                                                                      // that.

    void map_texture();
    void destroy_context();

   private:
    // resolution values if not specified
    static const int WINDOW_WIDTH = 80;
    static const int WINDOW_HEIGHT = 60;
    static const int DEFAULT_NR_THREADS = 128;

    // time observation
    static const int COUNT = 500;
    int count_;
    double compare_kernel_time_;
    double copy_likelihoods_time_;

    // device pointers to arrays stored in global memory on the GPU
    float* d_states_;       // not needed if propagation not on GPU
    float* d_states_copy_;  // not needed if propagation not on GPU
    float* d_visibility_probs_;
    float* d_visibility_probs_copy_;
    float* d_observations_;
    float* d_log_likelihoods_;
    int* d_prev_sample_indices_;
    int* d_resampling_indices_;  // not needed if resampling not on GPU
    curandStateMRG32k3a* d_mrg_states_;

    // for OpenGL interop
    cudaArray_t d_texture_array_;

    // resolution
    int n_cols_;
    int n_rows_;

    // maximum number of poses and their arrangement in the OpenGL texture
    int nr_max_poses_;
    int nr_max_poses_per_row_;
    int nr_max_poses_per_column_;

    // number of poses and their arrangement in the OpenGL texture
    int nr_poses_;
    int nr_poses_per_row_;
    int nr_poses_per_column_;

    // number of features in a state vector
    int n_features_;

    // block and grid arrangement of the CUDA kernels
    int nr_threads_, n_blocks_;
    dim3 grid_dimension_;

    // system properties
    int warp_size_;
    int n_mps_;

    // visibility prob default
    float visibility_prob_default_;

    // time values to compute the time deltas when calling propagate() or
    // evaluate()
    float occlusion_time_;
    float observation_time_;

    // float delta_time_;
    float last_propagation_time_;  // not needed if propagation not on GPU

    // booleans to describe the state of the cuda filter, to avoid wrong usage
    // of the class
    bool n_poses_set_;

    // CUDA device properties
    cudaDeviceProp cuda_device_properties_;

    void set_default_kernel_config(int& nr_poses_,
                                   int& nr_poses_per_row,
                                   int& nr_poses_per_column,
                                   bool& nr_poses_changed);

    // helper functions
    template <typename T>
    void allocate(T*& pointer, size_t size, std::string name);
    void check_cuda_error(const char* msg);
};
}

#endif  // CUDAFILTER_HPP
