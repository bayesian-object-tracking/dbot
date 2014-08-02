#ifndef CUDA_OPENGL_MULTIPLE_FILTER_HPP
#define CUDA_OPENGL_MULTIPLE_FILTER_HPP

#include <state_filtering/models/measurement/gpu_image_observation_model/cuda_opengl_filter.hpp>
namespace fil {

class CudaOpenglMultipleFilter : public fil::CudaOpenglFilter
{
public:

    void PropagateMultiple(const float &current_time, std::vector<std::vector<std::vector<float> > > &states);

    void EvaluateMultiple(const std::vector<std::vector<std::vector<float> > > &states,
                          const std::vector<int> prev_sample_indices,
                          const float &observation_time,
                          bool update,
                          std::vector<float> &log_likelihoods);

    void InitialEvaluateMultiple(int n_objects,
                                 const std::vector<std::vector<std::vector<float> > > &states,
                                 const std::vector<float> &observations,
                                 std::vector<std::vector<float> > &log_likelihoods);

    /// resamples the visibility probabilities that reside on the GPU according to the indices given.
    /** necessary previous call: Initialize()
      * @param resampling_indices   [new_state_nr] = old_state_nr
      */
    void ResampleMultiple(std::vector<int> resampling_indices);

    void set_states_multiple(int n_objects, int n_features, int seed);

private:

    static const int TIME_MEASUREMENTS_COUNT = 4;
    static const int COUNT = 500;

    enum time_measurement {SEND_INDICES, RENDER, MAP_RESOURCE, COMPUTE_LIKELIHOODS};

};
}

#endif // CUDA_OPENGL_MULTIPLE_FILTER_HPP
