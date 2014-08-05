#ifndef GPU_IMAGE_OBSERVATION_MODEL_
#define GPU_IMAGE_OBSERVATION_MODEL_

#include <vector>
#include "boost/shared_ptr.hpp"
#include "Eigen/Core"


#include <state_filtering/models/measurement/image_observation_model.hpp>
#include <state_filtering/models/measurement/gpu_image_observation_model/object_rasterizer.hpp>
#include <state_filtering/models/measurement/gpu_image_observation_model/cuda_filter.hpp>

#include <state_filtering/states/floating_body_system.hpp>



namespace distributions
{


struct ImageMeasurementModelGPUTypes
{
    typedef double                              ScalarType;
    typedef FloatingBodySystem<-1>              VectorType;
    typedef Eigen::Matrix<ScalarType, -1, -1>   MeasurementType;
    typedef unsigned                            IndexType;


    typedef RaoBlackwellMeasurementModel<ScalarType, VectorType, MeasurementType, IndexType>
                                            RaoBlackwellMeasurementModelType;
};


class ImageMeasurementModelGPU: public ImageMeasurementModelGPUTypes::RaoBlackwellMeasurementModelType
{
public:
    typedef typename ImageMeasurementModelGPUTypes::ScalarType ScalarType;
    typedef typename ImageMeasurementModelGPUTypes::VectorType VectorType;
    typedef typename ImageMeasurementModelGPUTypes::MeasurementType MeasurementType;
    typedef typename ImageMeasurementModelGPUTypes::IndexType IndexType;

    typedef typename Eigen::Matrix<ScalarType, 3, 3> CameraMatrixType;



    ImageMeasurementModelGPU(const CameraMatrixType& camera_matrix,
            const IndexType& n_rows,
            const IndexType& n_cols,
            const IndexType& max_sample_count,
            const ScalarType& initial_visibility_prob,
            const boost::shared_ptr<RigidBodySystem<-1> > &rigid_body_system);

    ~ImageMeasurementModelGPU();

    void Initialize();

    std::vector<float> Loglikes(const std::vector<Eigen::VectorXd >& states,
                                std::vector<size_t>& occlusion_indices,
                                const bool& update_occlusions = false);

    // set and get functions
    const std::vector<float> Occlusions(size_t index) const;

    // TODO: this should take the occlusion probability, not visibility
    void Occlusions(const float& visibility_prob = -1);

    void Measurement(const MeasurementType& image, const double& delta_time);



    // TODO: this image should be in a different format
    void RangeImage(std::vector<std::vector<int> > &intersect_indices,
                    std::vector<std::vector<float> > &depth);

    virtual void Reset();

    void Constants(const std::vector<std::vector<Eigen::Vector3d> > vertices_double,
                       const std::vector<std::vector<std::vector<int> > > indices,
                       const float p_visible_visible,
                       const float p_visible_occluded,
                       const float tail_weight,
                       const float model_sigma,
                       const float sigma_factor,
                       const float max_depth,
                       const float exponential_rate);

private:
    // TODO: this function should disappear
    void Measurement(const std::vector<float>& observations, const double& delta_time);



    const Eigen::Matrix3d camera_matrix_;
    const size_t n_rows_;
    const size_t n_cols_;
    const float initial_visibility_prob_;
    const size_t max_sample_count_;
    const boost::shared_ptr<RigidBodySystem<-1> > rigid_body_system_;


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

    double observation_time_;




//    size_t state_size();

//    size_t measurement_rows();
//    size_t measurement_cols();


    // used for time measurements
    static const int TIME_MEASUREMENTS_COUNT = 8;
    static const int COUNT = 500;
    enum time_measurement {SEND_OBSERVATIONS, RENDER, MAP_RESOURCE, GET_MAPPED_ARRAY, SET_TEXTURE_ARRAY,
                           MAP_TEXTURE, COMPUTE_LIKELIHOODS, UNMAP_RESOURCE};

};

}
#endif
