/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <memory>

#include <Eigen/Dense>

#include <std_msgs/Header.h>
#include <ros/ros.h>
#include <ros/package.h>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/object_file_reader.hpp>
#include <pose_tracking/states/free_floating_rigid_bodies_state.hpp>
#include <pose_tracking/utils/rigid_body_renderer.hpp>


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
