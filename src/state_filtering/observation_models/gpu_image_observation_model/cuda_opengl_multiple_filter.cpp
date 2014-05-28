#define PROFILING_ACTIVE

#include "cuda_opengl_multiple_filter.hpp"
#include "iostream"
#include "helper_functions.hpp"

using namespace fil;
using namespace std;


void CudaOpenglMultipleFilter::PropagateMultiple(const float &current_time, vector<vector<vector<float> > > &states) {
    if (initialized_) {
        cuda_->PropagateMultiple(current_time, states);
    } else {
        cout << "WARNING: CudaOpenglFilter::Propagate() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}


// REMEMBER TO SET n_poses_ RIGHT WHEN YOU EVALUATE MORE THAN N_PARTICLES
void CudaOpenglMultipleFilter::EvaluateMultiple(const vector<vector<vector<float> > > &states,
                                                const vector<int> prev_sample_indices,
                                                const float &observation_time,
                                                bool update,
                                                vector<float> &log_likelihoods) {
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
        cuda_->set_prev_sample_indices(prev_sample_indices.data());


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

        opengl_->Render(states);


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

//        std::vector<std::vector<int> > intersect_indices;
//        std::vector<std::vector<float> > depth;
//        opengl_->RenderCombinedFast(states,
//                                    intersect_indices,
//                                    depth);

//        for (int i = 0; i < intersect_indices.size(); i++) {
//            cout << "particle: " << i << endl;
//            for (int j = 0; j < intersect_indices[i].size(); j++) {
//                cout << "index: " << intersect_indices[i][j] << ", depth: " << depth[i][j] << endl;
//            }
//        }

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
        cuda_->Compare(observation_time, false, log_likelihoods);
//        cuda_->CompareMultiple(observation_time, false, update, log_likelihoods);
        log_likelihoods_ = log_likelihoods;
        cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);


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
        cout << "WARNING: CudaOpenglFilter::EvaluateMultiple() was not executed, because CudaOpenglFilter::Initialize() or CudaOpenglFilter::set_observations() has not been called previously." << endl;
    }
}



void CudaOpenglMultipleFilter::InitialEvaluateMultiple(
        int n_objects,
        const vector<vector<vector<float> > > &states,
        const vector<float> &observations,
        vector<vector<float> > &log_likelihoods)
{
    if (initialized_) {
        cuda_->set_observations(observations.data());
        for (int i = 0; i < n_objects; i++) {
            vector<int> number (1, i);
            opengl_->set_objects(number);
            opengl_->Render(states);

            // map the resources in CUDA
            cudaGraphicsMapResources(1, &combined_texture_resource_, 0);
            cudaGraphicsSubResourceGetMappedArray(&texture_array_, combined_texture_resource_, 0, 0);

            cuda_->set_texture_array(texture_array_);
            cuda_->MapTexture();
            float observation_time = 0;

            vector<float> tmp_log_likelihoods (n_poses_, 0);
            cuda_->Compare(observation_time, true, tmp_log_likelihoods);
            log_likelihoods.push_back(tmp_log_likelihoods);
//            log_likelihoods_ = log_likelihoods;

            cudaGraphicsUnmapResources(1, &combined_texture_resource_, 0);
        }
        vector<int> numbers;
        for (int i = 0; i < n_objects; i++) numbers.push_back(i);
        opengl_->set_objects(numbers);

        // change to regular number of poses
        set_number_of_poses(n_regular_poses_);

        // TODO don't remember what this is good for?
        log_likelihoods_.resize(n_poses_);
    } else {
        cout << "WARNING: CudaOpenglFilter::Evaluate() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}


void CudaOpenglMultipleFilter::ResampleMultiple(vector<int> resampling_indices) {
    if (initialized_) {
        cuda_->ResampleMultiple(resampling_indices);
    } else {
        cout << "WARNING: CudaOpenglFilter::Resample() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}

void CudaOpenglMultipleFilter::set_states_multiple(int n_objects, int n_features, int seed) {
    if (initialized_) {
        cuda_->set_states_multiple(n_objects, n_features, seed);
        states_set_ = true;
    } else {
        cout << "WARNING: CudaOpenglFilter::set_states() was not executed, because CudaOpenglFilter::Initialize() has not been called previously." << endl;
    }
}
