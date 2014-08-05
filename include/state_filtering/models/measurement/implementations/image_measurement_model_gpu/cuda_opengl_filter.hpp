#ifndef CUDA_OPENGL_FILTER_HPP
#define CUDA_OPENGL_FILTER_HPP

#include <vector>
#include <Eigen/Dense>
#include "boost/shared_ptr.hpp"

#include <state_filtering/models/measurement/implementations/image_measurement_model_gpu/object_rasterizer.hpp>
#include <state_filtering/models/measurement/implementations/image_measurement_model_gpu/cuda_filter.hpp>

namespace fil {

/// A particle filter that uses the GPU through OpenGL and CUDA.
/** It provides a propagate, evaluate and resample function and a couple of functions for initializing
 *  values and copying them to the GPU.
  */
class CudaOpenglFilter
{
public:
    /// constructor which does not do much, except for initializing some boolean values
    CudaOpenglFilter();

    /// destructor which outputs the average time used for rasterization and comparison
    ~CudaOpenglFilter();

    /// creates cuda and opengl object and initializes some values
    /** necessary previous call: set_constants(..)
     *  should be called once, after set_constants(..) and before anything else
      */
    void Initialize();



    /// propagates the states that already reside on the GPU and returns them in the states vector
    /** necessary previous calls: Initialize(), set_states(..)
     * @param[in]   current_time    a number that will be subtracted from the previously passed
     *                              number to build the delta_time needed for propagation
     * @param[out]  states          the states vector that is returned from this function, containing
     *                              the new (propagated) set of states
     */
    void Propagate(const float &current_time, std::vector<std::vector<float> > &states);



    /// evaluates the different states that are passed by rendering them and comparing them to the observations
    /** necessary previous call: Initialize()
     *  This Evaluate(..)(1) function is for evaluating the initial states, hence it does not need a current
     *  observation time (it is 0). It does not use occlusion. All visibility probabilites have the same
     *  (default) value.
     * @param states    [state_nr][feature_nr (0-6, 7-13, .. for respective object)] = feature value
     * @param observations      [pixel] = depth observed at that pixel
     * @param[out] log_likelihoods   [state_nr] = log_likelihood calculated and returned by this function
     */
    void Evaluate(
            const std::vector<std::vector<std::vector<float> > > &states,
            const std::vector<float> &observations,
            std::vector<float> &log_likelihoods);



    /// evaluates the different states that are passed by rendering them and comparing them to the observations
    /** necessary previous call: Initialize(), Evaluate(..)(the other one)
     *  This Evaluate(..)(2) function is for evaluating the regular states every filter round.
     * @param states    [state_nr][feature_nr (0-6, 7-13, .. for respective object)] = feature value
     * @param observations      [pixel] = depth observed at that pixel
     * @param observation_time  current time that will be subtracted from the previous time, to find out
     *                          how much time passed and calculate the occlusion probability accordingly
     * @param[out] log_likelihoods   [state_nr] = log_likelihood calculated and returned by this function
     */
    void Evaluate(
                const std::vector<std::vector<std::vector<float> > > &states,
                const std::vector<float> &observations,
                const float &observation_time,
                std::vector<float> &log_likelihoods);



    /// resamples the states that already reside on the GPU.
    /** necessary previous calls: Initialize(), set_states(..)
     *  Depending on their likelihood, the states are being resampled and their
     *  data (features, visibility probabilities) are shifted accordingly on the GPU.
     * @return  [old_state_nr] = new_state_nr
     */
    std::vector<int> Resample();

    /// sends the specified observations to the GPU
    /** necessary previous call: Initialize()
      * This function has to be called previously to calling Evaluate(), since the observations
      * are needed for the likelihood computation that takes place in the Evaluate() function.
      * @param observations      [pixel] = depth observed at that pixel
      */
    void set_observations(const std::vector<float> &observations);




    /// sets all the constants needed for the filtering
    /** should be called once, as soon as the caller knows the values for all constants
     * @param vertices  [object_nr][vertex_nr] = vertex of that object
     * @param indices   [object_nr][triangle_nr][vertex_index (0|1|2)] = index of this vertex
     *                  in its vertices list
     * @param n_init_poses  number of initial poses, used to locate the object
     * @param n_poses       number of poses, used in each filter round
     * @param angle_sigma   used for propagation of the states
     * @param trans_sigma   used for propagation of the states
     * @param com_models    [object_nr][dimension (0|1|2)] = center of mass model of that object
     * @param camera_matrix     matrix containing the intrinsic parameters of the camera
     * @param p_visible_init    initial probability for a pixel to be visible
     * @param p_visible_visible     probability that a pixel is visible if it was visible before
     * @param p_visible_occluded    probability that a pixel is visible if it was occluded before
     * @param tail_weight           used for probability calculation in the comparison step
     * @param model_sigma           used for probability calculation in the comparison step
     * @param sigma_factor          used for probability calculation in the comparison step
     * @param max_depth             used for probability calculation in the comparison step
     * @param exponential_rate      used for probability calculation in the comparison step
     */
    void set_constants(const std::vector<std::vector<Eigen::Vector3f> > vertices,
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
                       const float exponential_rate);

    /// sets the resolution
    /** necessary previous call: Initialize()
     *  should be called once after initialization to change the resolution if it is not
     *  the default of 60x80.
     * @param n_cols    number of columns (width)
     * @param n_rows    number of rows (height)
     */
    void set_resolution(int n_cols, int n_rows);

    /// initializes the states for the regular filter rounds
    /** necessary previous call: Initialize()
     *  needs to be called before any call to Propagate() or Resample(), because they
     *  change the status of the states on the GPU.
     * @param states    [state_nr][feature_nr (0-6, 7-13, .. for respective object)] = feature value
     * @param seed      a number used as seed for a random number generator
     */
    void set_states(std::vector<std::vector<float> > states, int seed);

    void set_number_of_poses(int n_poses);

    /// only needed for CUDA debugging. ignore.
    void destroy_context();


protected:

    void checkCUDAError(const char *msg);
    void UnregisterResource();
    void RegisterResource();

    boost::shared_ptr<ObjectRasterizer> opengl_;
    boost::shared_ptr<fil::CudaFilter> cuda_;

    // resolution
    int n_cols_, n_rows_;

    // arrays for timings
    std::vector<std::vector<double> > cpu_times_;
    std::vector<std::vector<float> > cuda_times_;

    // constants
    std::vector<std::vector<Eigen::Vector3f> > vertices_;
    std::vector<std::vector<std::vector<int> > > indices_;
    std::vector<float> log_likelihoods_;
    std::vector<std::vector<float> > com_models_;
    Eigen::Matrix3f camera_matrix_;
    float angle_sigma_, trans_sigma_;
    float p_visible_init_;
    float p_visible_visible_;
    float p_visible_occluded_;
    float tail_weight_;
    float model_sigma_;
    float sigma_factor_;
    float max_depth_;
    float exponential_rate_;


    int n_init_poses_, n_regular_poses_;
    int n_poses_;
    int n_renders_;
    int n_poses_x_;

    int count_;

    // booleans to ensure correct usage of function calls
    bool constants_set_, resolution_set_, states_set_, initialized_, observations_set_, resource_registered_;

    // Shared resource between OpenGL and CUDA
    GLuint combined_texture_opengl_;
    cudaGraphicsResource* combined_texture_resource_;
    cudaArray_t texture_array_;

private:

    static const int TIME_MEASUREMENTS_COUNT = 8;
    static const int COUNT = 500;

    enum time_measurement {SEND_OBSERVATIONS, RENDER, MAP_RESOURCE, GET_MAPPED_ARRAY, SET_TEXTURE_ARRAY,
                           MAP_TEXTURE, COMPUTE_LIKELIHOODS, UNMAP_RESOURCE};


};

}

#endif // CUDA_OPENGL_FILTER_HPP
