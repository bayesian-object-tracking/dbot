/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file cuda_filter.hpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date November 2015
 */

#pragma once

#include <curand_kernel.h>
#include <vector>

namespace fil
{
/**
 * \brief This class provides a parallel implementation of the weighting step on
 *        the GPU.
 *
 * After initializing the class and setting the execution parameters like the
 * number of poses and the resolution, you can weigh poses with the
 * weigh_poses() function.
 *
 * Make sure to
 *  always render the poses first with opengl, then map the texture into CUDA,
 *  update the observation image with set_observations() and update the
 * occlusion indices (after resampling)
 *  before you call the weigh_poses() function.
 */
class CudaFilter
{
public:
    /**
     * \brief Constructor which takes the resolution of the camera image
     *
     * \param [in] nr_rows
     *     The number of rows in each camera image
     * \param [in] nr_cols
     *     The number of columns in each camera image
     */
    CudaFilter(const int nr_rows = 60, const int nr_cols = 80);

    /**
     * \brief Destructor which frees the memory used on the GPU
     */
    ~CudaFilter();

    /**
     * \brief This function has to be called once in the beginning, before
     *        calling the weigh_poses() function.
     *
     * Copies constants to GPU memory and initializes some memory-related values
     *
     * \param [in] initial_occlusion_prob
     *     The initial probability for each pixel of being occluded, meaning
     *     that something occludes the object in this pixel
     * \param [in] p_occluded_occluded
     *     The probability of a pixel staying occluded from one frame to the
     *     next
     * \param [in] p_occluded_visible
     *     The probability of a pixel changing from being occluded to being
     *     visible from one frame to the next
     * \param [in] tail_weight
     *     The probability of a faulty measurement
     * \param [in] model_sigma
     *     The uncertainty in the 3D model of the object
     * \param [in] sigma_factor
     *     The standard deviation of the measurement noise at a distance of 1m
     *     to the camera
     * \param [in] max_depth
     *     Maximum value which can be measured by the depth sensor
     * \param [in] exponential_rate
     *     The rate of the exponential distribution that models the probability
     *     of a measurement coming from an unknown object
     */
    void init(const float initial_occlusion_prob,
              const float p_occluded_occluded,
              const float p_occluded_visible,
              const float tail_weight,
              const float model_sigma,
              const float sigma_factor,
              const float max_depth,
              const float exponential_rate);

    /**
     * \brief Weights the different poses that were previously rendered with
     *        OpenGL
     *
     * \param [in] update_occlusions
     *     Update whether or not to update the occlusion probabilities during
     *     this weighting
     * \param [out] log_likelihoods
     *     The computed likelihoods for each pose
     */
    void weigh_poses(const bool update_occlusions,
                     std::vector<float>& log_likelihoods);

    // setters

    /**
     * Sets the number of threads used for the CUDA weighting kernel to the
     * desired number.  A default of 128 is used if nothing is specified here.
     *
     * \param [in] nr_threads
     *     The desired number of threads
     */
    void set_nr_threads(const int nr_threads);

    /// copies the observation image from the camera to the GPU for comparison
    /**
     * \param [in] observations a pointer to the observation values
     * \param [in] observation_time the time at which this observation was
     * captured
     */
    void set_observations(const float* observations,
                          const float observation_time);

    /// sets the indices to the occlusion array for every state
    /**
     * \param [in] occlusion_indices [state_nr] = {index}. For each state, this
     * gives the index
     * into the occlusion array.
     */
    void set_occlusion_indices(const int* occlusion_indices);

    /// sets the resolution for the images to be compared
    /** This function might downgrade the number of poses or change the
     * arrangement of the
     *  poses in the grid due to the resolution change.
     * \param [in] n_rows the number of rows in an image
     * \param [in] n_cols the number of columns in an image
     * \param [out] nr_poses the number of poses that will be weighted
     * \param [out] nr_poses_per_row the number of poses per row that will be
     * weighted
     * \param [out] nr_poses_per_column the number of poses per column that will
     * be weighted
     * \param [in] adapt_to_constraints whether to automatically adapt to GPU
     * constraints or quit the program if constraints are not met
     */
    void set_resolution(const int n_rows,
                        const int n_cols,
                        int& nr_poses,
                        int& nr_poses_per_row,
                        int& nr_poses_per_column,
                        bool adapt_to_constraints = false);

    /// sets the occlusion probabilities for all pixels for all states
    /**
    * \param [in] occlusion_probabilities a 1D-array of occlusion probabilities
    * which should contain
    * nr_rows * nr_cols * nr_poses values.
    */
    void set_occlusion_probabilities(const float* occlusion_probabilities);

    /// maps the texture array to an actual texture reference
    /**
     * \param [in] texture_array the cudaArray retrieved from OpenGL
     */
    void map_texture_to_texture_array(const cudaArray_t texture_array);

    /// allocates the maximum amount of memory that will ever be needed by CUDA
    /// during runtime
    /**
     * \param [in][out] allocated_poses the maximum number of poses that will
     * ever be evaluated in one weighting step.
     * This number might be lowered if GPU contraints do now allow this number
     * of poses.
     * \param [out] allocated_poses_per_row the maximum number of poses per row
     * \param [out] allocated_poses_per_column the maximum number of poses per
     * column
     * \param [in] adapt_to_constraints whether to automatically adapt to GPU
     * constraints or quit the program if constraints are not met
     */
    void allocate_memory_for_max_poses(int& allocated_poses,
                                       int& allocated_poses_per_row,
                                       int& allocated_poses_per_column,
                                       bool adapt_to_constraints = false);

    /// sets the number of poses to be weighted in the next weighting step
    /**
     * \param [in][out] nr_poses the desired number of poses. Might be changed
     * due to GPU constraints.
     * \param [out] nr_poses_per_row the number of poses per row
     * \param [out] nr_poses_per_column the number of poses per column
     * \param [in] adapt_to_constraints whether to automatically adapt to GPU
     * constraints or quit the program if constraints are not met
     */
    void set_number_of_poses(int& nr_poses,
                             int& nr_poses_per_row,
                             int& nr_poses_per_column,
                             bool adapt_to_constraints = false);

    // getters
    /// gets the maximum number of threads that can be handled with this GPU
    /**
     * \return the maximum number of threads that can be scheduled per block on
     * the GPU
     */
    int get_max_nr_threads();

    /// gets the warp size of this GPU
    /**
     * \return the warp size = the number of threads that are executed
     * concurrently on a CUDA streaming multiprocessor
     */
    int get_warp_size();

    /// gets the occlusion probabilities of a particular state
    /**
     * \param [in] state_id the index into the state array
     * \return a 1D array containing the occlusion probability for each pixel
     */
    std::vector<float> get_occlusion_probabilities(int state_id);

private:
    static const int DEFAULT_NR_THREADS = 128;

    // device pointers to arrays stored in global memory on the GPU
    float* d_occlusion_probs_;
    float* d_occlusion_probs_copy_;
    float* d_observations_;
    float* d_log_likelihoods_;
    int* d_occlusion_indices_;  // this contains, for each pose, the index into
                                // the occlusion probabilities array, which
                                // contains the occlusion probabilities for that
                                // particular pose.

    // for OpenGL interop
    cudaArray_t d_texture_array_;

    // resolution
    int nr_cols_;
    int nr_rows_;

    // maximum number of poses and their arrangement in the OpenGL texture
    int nr_max_poses_;
    int nr_max_poses_per_row_;
    int nr_max_poses_per_column_;

    // actual number of poses and their arrangement in the OpenGL texture
    // (current frame)
    int nr_poses_;
    int nr_poses_per_row_;
    int nr_poses_per_column_;

    // block and grid arrangement of the CUDA kernels
    int nr_threads_, n_blocks_;
    dim3 grid_dimension_;

    // occlusion probability default value
    float occlusion_prob_default_;

    // time values to compute the time deltas when calling the weighting
    // function
    float occlusion_time_;
    float observation_time_;

    // CUDA device properties
    cudaDeviceProp cuda_device_properties_;
    int warp_size_;
    int n_mps_;

    // bool to ensure correct usage of public functions
    bool observations_set_, occlusion_indices_set_,
        occlusion_probabilities_set_, memory_allocated_;
    bool number_of_poses_set_, constants_initialized_;

    void set_default_kernel_config(int& nr_poses_,
                                   int& nr_poses_per_row,
                                   int& nr_poses_per_column,
                                   bool& nr_poses_changed,
                                   bool adapt_to_constraints);
    // helper functions
    template <typename T>
    void allocate(T*& pointer, size_t size);
    void check_cuda_error(const char* msg);
};
}
