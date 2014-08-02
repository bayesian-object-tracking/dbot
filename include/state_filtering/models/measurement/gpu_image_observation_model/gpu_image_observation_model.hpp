#ifndef GPU_IMAGE_OBSERVATION_MODEL_
#define GPU_IMAGE_OBSERVATION_MODEL_

#include <vector>
#include "boost/shared_ptr.hpp"
#include "Eigen/Core"


#include <state_filtering/models/measurement/image_observation_model.hpp>
#include <state_filtering/models/measurement/gpu_image_observation_model/object_rasterizer.hpp>
#include <state_filtering/models/measurement/gpu_image_observation_model/cuda_filter.hpp>


namespace obs_mod
{
class GPUImageObservationModel: public ImageObservationModel
{
public:

    GPUImageObservationModel(const Eigen::Matrix3d& camera_matrix,
            const size_t& n_rows,
            const size_t& n_cols,
            const size_t& max_sample_count,
            const float& initial_visibility_prob,
            const boost::shared_ptr<RigidBodySystem<-1> > &rigid_body_system);

	~GPUImageObservationModel();

    void Initialize();

    std::vector<float> Evaluate(
			const std::vector<Eigen::VectorXd >& states,
			std::vector<size_t>& occlusion_indices,
			const bool& update_occlusions = false);

    std::vector<float> Evaluate_test(
            const std::vector<Eigen::VectorXd>& states,
            std::vector<size_t>& occlusion_indices,
            const bool& update_occlusions,
            std::vector<std::vector<int> > intersect_indices = 0,
            std::vector<std::vector<float> > predictions = 0);

	// set and get functions =============================================================================================================================================================================================================================================================================================
    const std::vector<float> get_occlusions(size_t index) const;
	void set_occlusions(const float& visibility_prob = -1);
    void measurement(const std::vector<float>& observations, const double& time_since_start);


    void set_constants(const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
                       const std::vector<std::vector<std::vector<int> > > indices,
                       const float p_visible_visible,
                       const float p_visible_occluded,
                       const float tail_weight,
                       const float model_sigma,
                       const float sigma_factor,
                       const float max_depth,
                       const float exponential_rate);

    // [pose index][0] = intersect index, [pose_index][1] = depth value
    void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                          std::vector<std::vector<float> > &depth);


private:


    void set_number_of_poses(int n_poses);
    void checkCUDAError(const char *msg);
    void UnregisterResource();
    void RegisterResource();

    boost::shared_ptr<ObjectRasterizer> opengl_;
    boost::shared_ptr<fil::CudaFilter> cuda_;

    // resolution
//    int n_cols_;
//    int n_rows_;

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

    // used for time measurements
    static const int TIME_MEASUREMENTS_COUNT = 8;
    static const int COUNT = 500;
    enum time_measurement {SEND_OBSERVATIONS, RENDER, MAP_RESOURCE, GET_MAPPED_ARRAY, SET_TEXTURE_ARRAY,
                           MAP_TEXTURE, COMPUTE_LIKELIHOODS, UNMAP_RESOURCE};

};

}
#endif
