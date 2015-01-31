


#include <fstream>
#include <ctime>
#include <memory>
#include <unordered_map>

#include <std_msgs/Header.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>

#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/model/observation/factorized_iid_observation_model.hpp>

#include <pose_tracking/utils/rigid_body_renderer.hpp>
#include <pose_tracking/states/free_floating_rigid_bodies_state.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/object_file_reader.hpp>
#include <pose_tracking_interface/utils/image_publisher.hpp>

#include <pose_tracking/models/process_models/brownian_object_motion_model.hpp>


#include <fl/util/profiling.hpp>



template <typename Vector> class VectorHash;

template<> class VectorHash<Eigen::MatrixXd>
{
public:
    std::size_t operator()(Eigen::MatrixXd const& s) const
    {
        /* primes */
        static constexpr int p1 = 15487457;
        static constexpr int p2 = 24092821;
        static constexpr int p3 = 73856093;
        static constexpr int p4 = 19349663;
        static constexpr int p5 = 83492791;
        static constexpr int p6 = 17353159;

        /* map size */
        static constexpr int n = 1200;

        /* precision */
        static constexpr int c = 1000000;

        return  ((int(s(0, 0)*c) * p1) ^
                 (int(s(1, 0)*c) * p2) ^
                 (int(s(2, 0)*c) * p3) ^
                 (int(s(3, 0)*c) * p4) ^
                 (int(s(4, 0)*c) * p5) ^
                 (int(s(5, 0)*c) * p6) % n);
    }
};



/* ############################## */
/* # Observation Model          # */
/* ############################## */
namespace fl
{
template <
    typename State,
    typename Scalar,
    int ResRows,
    int ResCols
>
class DepthObservationModel;

template <
    typename State_,
    typename Scalar,
    int ResRows,
    int ResCols
>
struct Traits<
           DepthObservationModel<State_, Scalar, ResRows, ResCols>
       >
{
    enum
    {
        PixelObsrvDim = 1,
        PixelStateDim = 1
    };

    // [y  y^2] kernel space?
    typedef Eigen::Matrix<Scalar, PixelObsrvDim, 1> PixelObsrv;

    // [h_i(x) h_i(x)^2] rendered pixel
    typedef Eigen::Matrix<Scalar, PixelStateDim, 1> PixelState;

    // local linear gaussian observation model
    typedef LinearGaussianObservationModel<
                PixelObsrv,
                PixelState
            > PixelObsrvModel;

    typedef typename Traits<PixelObsrvModel>::SecondMoment PixelCov;
    typedef typename Traits<PixelObsrvModel>::SensorMatrix PixelSensorMatrix;

    // Holistic observation model
    typedef FactorizedIIDObservationModel<
                PixelObsrvModel,
                FactorSize<ResRows, ResCols>::Size
            > CameraObservationModel;

    typedef State_ State;
    typedef typename Traits<CameraObservationModel>::State StateInternal;
    typedef typename Traits<CameraObservationModel>::Observation Observation;
    typedef typename Traits<CameraObservationModel>::Noise Noise;

    typedef ANObservationModelInterface<
                Observation,
                State_,
                Noise
            > ObservationModelBase;
};


template <
    typename State,
    typename Scalar,
    int ResRows,
    int ResCols
>
class DepthObservationModel
    : public Traits<
                DepthObservationModel<State, Scalar, ResRows, ResCols>
             >
{
public:
    typedef DepthObservationModel<State, Scalar, ResRows, ResCols> This;

    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::Noise Noise;

    enum
    {
        PixelObsrvDim =  Traits<This>::PixelObsrvDim,
        PixelStateDim =  Traits<This>::PixelStateDim
    };
    typedef typename Traits<This>::PixelObsrvModel PixelObsrvModel;
    typedef typename Traits<This>::PixelCov PixelCov;
    typedef typename Traits<This>::PixelSensorMatrix PixelSensorMatrix;

    typedef typename Traits<This>::StateInternal StateInternal;
    typedef typename Traits<This>::CameraObservationModel CameraObservationModel;

public:
    DepthObservationModel(std::shared_ptr<fl::RigidBodyRenderer> renderer,
                          Scalar camera_sigma,
                          Scalar model_sigma,
                          size_t state_dimension = DimensionOf<State>(),
                          int res_rows = ResRows,
                          int res_cols = ResCols)
        : camera_obsrv_model_(
              std::make_shared<PixelObsrvModel>(
                  PixelCov::Identity(PixelObsrvDim, PixelObsrvDim)
                  * ((camera_sigma*camera_sigma) + (model_sigma*model_sigma))),
              (res_rows*res_cols)),
          model_sigma_(model_sigma),
          camera_sigma_(camera_sigma),
          renderer_(renderer),
          state_dimension_(state_dimension)
    {
        assert(res_rows > 0);
        assert(res_cols > 0);
        assert(state_dimension_ > 0);

        depth_rendering_.resize(res_rows * res_cols);
    }


    ~DepthObservationModel() { }

    virtual Observation predict_observation(const State& state,
                                            double delta_time)
    {
        Eigen::MatrixXd pose = state.topRows(6);

        if (predictions_cache_.find(pose) == predictions_cache_.end())
        {
            map(state, predictions_cache_[pose]);
        }

        state_internal_ = predictions_cache_[pose];

//        return camera_obsrv_model_.predict_observation(
//                    state_internal_,
//                    Noise::Zero(noise_dimension(), 1),
//                    delta_time);

        return state_internal_;
    }

    virtual size_t observation_dimension() const
    {
        return camera_obsrv_model_.observation_dimension();
    }

    virtual size_t state_dimension() const
    {
        return state_dimension_;
    }

    virtual size_t noise_dimension() const
    {
        return camera_obsrv_model_.noise_dimension();
    }

    virtual Noise noise_covariance_vector() const
    {
        return Noise::Ones(noise_dimension(), 1) *
               (model_sigma_ * model_sigma_ + camera_sigma_ * camera_sigma_);
    }

    virtual void clear_cache()
    {
        predictions_cache_.clear();
    }

public:
    /** \cond INTERNAL */
    void map(const State& state, StateInternal& state_internal)
    {
        renderer_->state(state);
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, state_internal);
    }

    void convert(const std::vector<float>& depth,
                 StateInternal& state_internal)
    {
        const int pixel_count = depth.size();
        state_internal.resize(pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            state_internal(i, 0) = (std::isinf(depth[i]) ? 7 : depth[i]);
        }
    }

    /** \endcond */

protected:
    CameraObservationModel camera_obsrv_model_;
    Scalar model_sigma_;
    Scalar camera_sigma_;
    std::shared_ptr<fl::RigidBodyRenderer> renderer_;
    std::vector<float> depth_rendering_;
    StateInternal state_internal_;
    size_t state_dimension_;

    std::unordered_map<Eigen::MatrixXd,
                       StateInternal,
                       VectorHash<Eigen::MatrixXd>> predictions_cache_;
};

}


template <typename State>
class VirtualObject
{
public:
    VirtualObject(ros::NodeHandle& nh)
        : renderer(create_object_renderer(nh)),
          object_publisher(
              nh.advertise<visualization_msgs::Marker>("object_model", 0)),
          state(1)
    {
        ri::ReadParameter("downsampling", downsampling, nh);
        ri::ReadParameter("pose_x", pose_x, nh);
        ri::ReadParameter("pose_y", pose_y, nh);
        ri::ReadParameter("pose_z", pose_z, nh);
        ri::ReadParameter("pose_alpha", pose_alpha, nh);
        ri::ReadParameter("pose_beta", pose_beta, nh);
        ri::ReadParameter("pose_gamma", pose_gamma, nh);
        ri::ReadParameter("pose_alpha_v", pose_alpha_v, nh);
        ri::ReadParameter("pose_beta_v", pose_beta_v, nh);
        ri::ReadParameter("pose_gamma_v", pose_gamma_v, nh);

        header.frame_id = "/SIM_CAM";

        res_rows = 480 / downsampling;
        res_cols = 640 / downsampling;

        state.pose()(0) = pose_x;
        state.pose()(1) = pose_y;
        state.pose()(2) = pose_z;

        m_c =
            Eigen::AngleAxisd(pose_alpha * 2 * M_PI, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(pose_beta * 2 * M_PI, Eigen::Vector3d::UnitZ());
        shift = 0.;

        camera_matrix.setZero();
        camera_matrix(0, 0) = 580.0 / downsampling; // fx
        camera_matrix(1, 1) = 580.0 / downsampling; // fy
        camera_matrix(2, 2) = 1.0;
        camera_matrix(0, 2) = 320 / downsampling;   // cx
        camera_matrix(1, 2) = 240 / downsampling;   // cy

        renderer->parameters(camera_matrix, res_rows, res_cols);

        std::cout << "Resolution: " <<
                     res_cols << "x" << res_rows <<
                     " (" << res_cols*res_rows << " pixels)" << std::endl;

        animate(); // set orientation
    }

    void animate()
    {
        m = m_c * Eigen::AngleAxisd(
                    (pose_gamma + std::cos(shift)/16.)  * 2 * M_PI,
                    Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q(m);
        state.quaternion(q);

        shift += pose_gamma_v;
        if (shift > 2*M_PI) shift -= 2*M_PI;
    }

    void publish_marker(const State& state,
                        int id=1,
                        float r=1.f, float g=0.f, float b=0.f)
    {
        header.stamp = ros::Time::now();
        ri::PublishMarker(
                    state.homogeneous_matrix(0).template cast<float>(),
                    header,
                    object_model_uri,
                    object_publisher,
                    id, r, g, b);
    }

    void render(std::vector<float>& depth)
    {
        renderer->state(state);
        renderer->Render(depth);
    }

    int image_size()
    {
        return res_rows * res_cols;
    }

public:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */
    std::shared_ptr<fl::RigidBodyRenderer>
    create_object_renderer(ros::NodeHandle& nh)
    {
        ri::ReadParameter("object_package", object_package, nh);
        ri::ReadParameter("object_model", object_model, nh);

        object_model_path = ros::package::getPath(object_package) + object_model;
        object_model_uri = "package://" + object_package + object_model;

        std::cout << "Opening object file " << object_model_path << std::endl;
        std::vector<Eigen::Vector3d> object_vertices;
        std::vector<std::vector<int>> object_triangle_indices;
        ObjectFileReader file_reader;
        file_reader.set_filename(object_model_path);
        file_reader.Read();
        object_vertices = *file_reader.get_vertices();
        object_triangle_indices = *file_reader.get_indices();

        boost::shared_ptr<fl::FreeFloatingRigidBodiesState<>> state(
                new fl::FreeFloatingRigidBodiesState<>(1));

        std::shared_ptr<fl::RigidBodyRenderer> object_renderer(
                new fl::RigidBodyRenderer(
                    {object_vertices},
                    {object_triangle_indices},
                    state
                )
        );

        return object_renderer;
    }

public:
    State state;
    std::string object_package;
    std::string object_model;
    std::string object_model_path;
    std::string object_model_uri;
    double downsampling;
    int res_rows;
    int res_cols;
    std::shared_ptr<fl::RigidBodyRenderer> renderer;
    ros::Publisher object_publisher;
    std_msgs::Header header;

protected:
    Eigen::Matrix3d m;
    Eigen::Matrix3d m_c;
    Eigen::Matrix3d camera_matrix;
    double shift;

    // parameters
    double pose_x;
    double pose_y;
    double pose_z;
    double pose_alpha;
    double pose_beta;
    double pose_gamma;
    double pose_alpha_v;
    double pose_beta_v;
    double pose_gamma_v;
};




class DevTestTracker
{
public:
    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef fl::BrownianObjectMotionModel<
                fl::FreeFloatingRigidBodiesState<>
            >  ProcessModel;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    // Holistic observation model
    typedef fl::DepthObservationModel<
                typename fl::Traits<ProcessModel>::State,
                typename fl::Traits<ProcessModel>::Scalar,
                Eigen::Dynamic,
                Eigen::Dynamic
            > ObsrvModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObsrvModel,
                fl::UnscentedTransform,
                fl::AdditiveObservationNoise
            > FilterAlgo;

    typedef fl::FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::State State;
    typedef typename Filter::Input Input;
    typedef typename Filter::Observation Obsrv;
    typedef typename Obsrv::Scalar Scalar;
    typedef typename Filter::StateDistribution StateDistribution;

    DevTestTracker(ros::NodeHandle& nh, VirtualObject<State>& object)
        : object(object),
          process_model_(create_process_model(nh)),
          obsrv_model_(create_obsrv_model(nh, process_model_)),
          filter_(create_filter(nh, process_model_, obsrv_model_)),
          state_distr_(filter_->process_model()->state_dimension()),
          zero_input_(Input::Zero(filter_->process_model()->input_dimension(), 1)),
          y(filter_->observation_model()->observation_dimension(), 1)
    {
        state_distr_.mean(object.state);
        state_distr_.covariance(state_distr_.covariance() * 0.0001);

        ri::ReadParameter("inv_sigma", filter_->inv_sigma, nh);
        ri::ReadParameter("threshold", filter_->threshold, nh);
    }

    void filter(std::vector<float> y_vec)
    {
        const int y_vec_size = y_vec.size();

        Scalar y_i;
        for (int i = 0; i < y_vec_size; ++i)
        {
            y_i = y_vec[i];
            y(i, 0) = (std::isinf(y_i) ? 7 : y_i);
        }

        filter_->predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_->update(y, state_distr_, state_distr_);

        obsrv_model_->clear_cache();
    }

public:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    /**
     * \return Process model instance
     */
    std::shared_ptr<ProcessModel> create_process_model(ros::NodeHandle& nh)
    {
        double linear_acceleration_sigma;
        double angular_acceleration_sigma;
        double damping;

        ri::ReadParameter("linear_acceleration_sigma",
                          linear_acceleration_sigma, nh);
        ri::ReadParameter("angular_acceleration_sigma",
                          angular_acceleration_sigma, nh);
        ri::ReadParameter("damping",
                          damping, nh);

        auto model = std::make_shared<ProcessModel>(1 /* one object */);

        Eigen::MatrixXd linear_acceleration_covariance =
                Eigen::MatrixXd::Identity(3, 3)
                * std::pow(double(linear_acceleration_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
                Eigen::MatrixXd::Identity(3, 3)
                * std::pow(double(angular_acceleration_sigma), 2);

        model->parameters(0,
                          object.renderer->object_center(0).cast<double>(),
                          damping,
                          linear_acceleration_covariance,
                          angular_acceleration_covariance);

        return model;
    }

    /**
     * \return Observation model instance
     */
    std::shared_ptr<ObsrvModel> create_obsrv_model(
            ros::NodeHandle& nh,
            std::shared_ptr<ProcessModel> process_model)
    {
        double model_sigma;
        double camera_sigma;
        ri::ReadParameter("model_sigma", model_sigma, nh);
        ri::ReadParameter("camera_sigma", camera_sigma, nh);

        return std::make_shared<ObsrvModel>(object.renderer,
                                            camera_sigma,
                                            model_sigma,
                                            process_model->state_dimension(),
                                            object.res_rows,
                                            object.res_cols);
    }

    /**
     * \return Filter instance
     */
    Filter::Ptr create_filter(
            ros::NodeHandle& nh,
            const std::shared_ptr<ProcessModel>& process_model,
            const std::shared_ptr<ObsrvModel>& obsrv_model)
    {
        return std::make_shared<FilterAlgo>(
                    process_model,
                    obsrv_model,
                    std::make_shared<fl::UnscentedTransform>());
    }

public:
    VirtualObject<State>& object;
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
    Filter::Ptr filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;
};


int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "dev_test_tracker");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    VirtualObject<DevTestTracker::State> object(nh);
    DevTestTracker tracker(nh, object);

    /* ############################## */
    /* # Images                     # */
    /* ############################## */
    fl::ImagePublisher ip(nh);
    std::vector<float> depth;
    Eigen::MatrixXd image_vector(object.res_rows * object.res_cols, 1);

    std::cout << ">> initial state " << std::endl;
    std::cout << tracker.state_distr_.mean().transpose() << std::endl;
    while(ros::ok())
    {
//        std::cout <<  "==========================================" << std::endl;
//        std::cout << tracker.state_distr_.mean().transpose() << std::endl;
//        std::cout <<  "--------------------" << std::endl;
//        std::cout << tracker.state_distr_.covariance() << std::endl;

        /* ############################## */
        /* # Animate & Render           # */
        /* ############################## */
        object.animate();
        object.render(depth);

        /* ############################## */
        /* # Filter                     # */
        /* ############################## */
        INIT_PROFILING
        tracker.filter(depth);
        MEASURE("filter")

        /* ############################## */
        /* # Visualize                  # */
        /* ############################## */
        object.publish_marker(object.state, 1, 1, 0, 0);
        object.publish_marker(tracker.state_distr_.mean(), 2, 0, 1, 0);

//        image_vector.setZero();
//        for (size_t i = 0; i < depth.size(); ++i)
//        {
//            image_vector(i, 0) = (std::isinf(depth[i]) ? 0 : depth[i]);
//        }
//        ip.publish(image_vector, "/dev_test_tracker/observation",
//                   object.res_rows,
//                   object.res_cols);

//        ip.publish(tracker.filter_->innovation, "/dev_test_tracker/innovation",
//                   object.res_rows,
//                   object.res_cols);

//        ip.publish(tracker.filter_->prediction, "/dev_test_tracker/prediction",
//                   object.res_rows,
//                   object.res_cols);

//        ip.publish(tracker.filter_->invR, "/dev_test_tracker/inv_covariance",
//                   object.res_rows,
//                   object.res_cols);

        ros::spinOnce();

    }

    return 0;
}
