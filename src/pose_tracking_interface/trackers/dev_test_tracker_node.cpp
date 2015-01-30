#include <fstream>
#include <ctime>
#include <memory>

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





/* ############################## */
/* # Observation Model          # */
/* ############################## */
namespace fl
{
template <
    typename State,
    typename Scalar,
    int ResHeight,
    int ResWidth
>
class DepthObservationModel;

template <
    typename State,
    typename Scalar,
    int ResHeight,
    int ResWidth
>
struct Traits<
           DepthObservationModel<State, Scalar, ResHeight, ResWidth>
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
                FactorSize<ResHeight, ResWidth>::Size
            > CameraObservationModel;

    typedef typename Traits<CameraObservationModel>::State StateInternal;
    typedef typename Traits<CameraObservationModel>::Observation Observation;
    typedef typename Traits<CameraObservationModel>::Noise Noise;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};


template <
    typename State,
    typename Scalar,
    int ResHeight,
    int ResWidth
>
class DepthObservationModel
    : public Traits<
                DepthObservationModel<State, Scalar, ResHeight, ResWidth>
             >
{
public:
    typedef DepthObservationModel<State, Scalar, ResHeight, ResWidth> This;

    typedef typename Traits<This>::State State;
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
                          int res_height = ResHeight,
                          int res_width = ResWidth)
        : camera_obsrv_model_(
              std::make_shared<PixelObsrvModel>(
                  PixelCov::Identity(PixelObsrvDim, PixelObsrvDim)
                  * ((camera_sigma*camera_sigma) + (model_sigma*model_sigma))),
              (res_height*res_width)),
          model_sigma_(model_sigma),
          renderer_(renderer),
          state_dimension_(state_dimension)
    {
        assert(res_height > 0);
        assert(res_width > 0);
        assert(state_dimension_ > 0);

        depth_rendering_.resize(res_height * res_width);
    }


    ~DepthObservationModel() { }

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        renderer_->state(state);
        renderer_->Render(depth_rendering_);

        map(state, state_internal_);

        return camera_obsrv_model_->predict_observation(
                    state_internal_,
                    noise,
                    delta_time);
    }

    virtual size_t observation_dimension() const
    {
        return camera_obsrv_model_->observation_dimension();
    }

    virtual size_t state_dimension() const
    {
        return state_dimension_;
    }

    virtual size_t noise_dimension() const
    {
        return camera_obsrv_model_->noise_dimension();
    }

public:
    /** \cond INTERNAL */
    void map(const State& state, StateInternal& state_internal)
    {
        renderer_->state(state);
        renderer_->Render(depth_rendering_);

        convert(depth_rendering_, state_internal);
    }

    void convert(const std::vector<float>& depth_rendering,
                 StateInternal& state_internal)
    {
        const int pixel_count = depth_rendering.size();
        state_internal.resize(pixel_count, 1);

        for (int i = 0; i < pixel_count; ++i)
        {
            state_internal(i, 0) = state_internal[i];
        }
    }

    /** \endcond */

protected:
    CameraObservationModel camera_obsrv_model_;
    Scalar model_sigma_;
    std::shared_ptr<fl::RigidBodyRenderer> renderer_;
    std::vector<float> depth_rendering_;
    StateInternal state_internal_;
    size_t state_dimension_;
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

    void publish_marker(State& state,
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
                    state,
                    ));

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
    enum
    {
        LocalObsrvDim = 1,
        LocalObsrvStateDim = 1
    };

    // [y  y^2] kernel space?
    typedef Eigen::Matrix<double, LocalObsrvDim, 1> LocalObsrv;

    // [h_i(x) h_i(x)^2] rendered pixel
    typedef Eigen::Matrix<double, LocalObsrvStateDim, 1> LocalObsrvState;

    // local observation model
    typedef fl::LinearGaussianObservationModel<
                LocalObsrv,
                LocalObsrvState
            > PixelObsrvModel;

    // Holistic observation model
    typedef fl::FactorizedIIDObservationModel<
                PixelObsrvModel,
                Eigen::Dynamic
            > ObsrvModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObsrvModel,
                fl::UnscentedTransform
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
          filter_(create_filter(nh)),
          state_distr_(filter_->process_model()->state_dimension()),
          zero_input_(Input::Zero(filter_->process_model()->input_dimension(), 1)),
          y(filter_->observation_model()->observation_dimension(), 1)
    {
        state_distr_.mean(object.state);
    }

    void initialize(State initial_state,
                    const sensor_msgs::Image& ros_image,
                    Eigen::Matrix3d camera_matrix)
    {

    }

    void filter(std::vector<float> y_vec)
    {
        const int y_vec_size = y_vec.size();

        Scalar y_i;
        for (int i = 0; i < y_vec_size; ++i)
        {
            y_i = y_vec[i];
            y(i, 0) = y_i;
            //y(y_vec_size + i, 0) = y_i * y_i;
        }

        filter_->predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_->update(y, state_distr_, state_distr_);
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
    std::shared_ptr<ObsrvModel> create_obsrv_model(ros::NodeHandle& nh)
    {
        double camera_sigma;
        ri::ReadParameter("camera_sigma", camera_sigma, nh);

        typedef typename fl::Traits<PixelObsrvModel>::SecondMoment LocalCov;
        typedef typename fl::Traits<PixelObsrvModel>::SensorMatrix SensorMatrix;

        auto local_model =
                std::make_shared<PixelObsrvModel>(
                    LocalCov::Identity(LocalObsrvDim, LocalObsrvDim)
                    * (camera_sigma * camera_sigma));

        auto H = local_model->H();
        H.setIdentity();

        return std::make_shared<ObsrvModel>(local_model, object.image_size());
    }

    /**
     * \return Filter instance
     */
    Filter::Ptr create_filter(ros::NodeHandle& nh)
    {
        return std::make_shared<FilterAlgo>(
                    create_process_model(nh),
                    create_obsrv_model(nh),
                    std::make_shared<fl::UnscentedTransform>());
    }

public:
    VirtualObject<State>& object;
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

    while(ros::ok())
    {
        /* ############################## */
        /* # Animate & Render           # */
        /* ############################## */
        object.animate();
        object.render(depth);

        /* ############################## */
        /* # Filter                     # */
        /* ############################## */
        tracker.filter(depth);

        /* ############################## */
        /* # Visualize                  # */
        /* ############################## */
        object.publish_marker(object.state, 1, 1, 0, 0);

        image_vector.setZero();
        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = (std::isinf(depth[i]) ? 0 : depth[i]);
        }
        ip.publish(image_vector, "/dev_test_tracker/observation",
                   object.res_rows,
                   object.res_cols);

        ros::Duration(1./30.).sleep();
    }

    return 0;
}
