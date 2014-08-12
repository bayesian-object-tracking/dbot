#ifndef MODELS_OBSERVERS_IMPLEMENTATIONS_IMAGE_observer_GPU_HPP
#define MODELS_OBSERVERS_IMPLEMENTATIONS_IMAGE_observer_GPU_HPP

#include <vector>
#include "boost/shared_ptr.hpp"
#include "Eigen/Core"


#include <state_filtering/models/observers/features/rao_blackwell_observer.hpp>
#include <state_filtering/models/observers/implementations/image_observer_gpu/object_rasterizer.hpp>
#include <state_filtering/models/observers/implementations/image_observer_gpu/cuda_filter.hpp>

#include <state_filtering/states/floating_body_system.hpp>



namespace distributions
{


struct ImageObserverGPUTypes
{
    typedef double                              ScalarType;
    typedef RigidBodySystem<-1>                 StateType;
    typedef Eigen::Matrix<ScalarType, -1, -1>   ObservationType;
    typedef size_t                              IndexType;


    typedef RaoBlackwellObserver<ScalarType, StateType, ObservationType, IndexType>
                                            RaoBlackwellObserverType;
};


class ImageObserverGPU: public ImageObserverGPUTypes::RaoBlackwellObserverType
{
public:
    typedef typename ImageObserverGPUTypes::ScalarType      ScalarType;
    typedef typename ImageObserverGPUTypes::StateType       StateType;
    typedef typename ImageObserverGPUTypes::ObservationType ObservationType;
    typedef typename ImageObserverGPUTypes::IndexType       IndexType;

    typedef typename Eigen::Matrix<ScalarType, 3, 3> CameraMatrixType;

    ImageObserverGPU(const CameraMatrixType& camera_matrix,
                             const IndexType& n_rows,
                             const IndexType& n_cols,
                             const IndexType& max_sample_count,
                             const ScalarType& initial_visibility_prob);

    ~ImageObserverGPU();

    // TODO: DO WE NEED TWO DIFFERENT FUNCTIONS FOR THIS??
    void Initialize();
    void Constants(const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
                       const std::vector<std::vector<std::vector<int> > > indices,
                       const float p_visible_visible,
                       const float p_visible_occluded,
                       const float tail_weight,
                       const float model_sigma,
                       const float sigma_factor,
                       const float max_depth,
                       const float exponential_rate);

    std::vector<ScalarType> Loglikes(const std::vector<const StateType*>& states,
                                     std::vector<IndexType>& occlusion_indices,
                                     const bool& update_occlusions = false);

    void Observation(const ObservationType& image, const double& delta_time);

    virtual void Reset();

    // TODO: this image should be in a different format BOTH OF THEM!!
    const std::vector<float> Occlusions(size_t index) const;
    void RangeImage(std::vector<std::vector<int> > &intersect_indices,
                    std::vector<std::vector<float> > &depth);
private:
    // TODO: this function should disappear, BOTH OF THEM
    void Observation(const std::vector<float>& observations, const ScalarType& delta_time);
    void Occlusions(const float& visibility_prob = -1);

    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;

    void set_number_of_poses(int n_poses);
    void checkCUDAError(const char *msg);
    void UnregisterResource();
    void RegisterResource();

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
