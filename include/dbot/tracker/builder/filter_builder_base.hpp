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
 * \file filter_builder_hpp.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <type_traits>

#include <Eigen/Dense>
#include <osr/pose_vector.hpp>
#include <osr/pose_velocity_vector.hpp>

#include <fl/util/types.hpp>
#include <fl/model/process/linear_state_transition_model.hpp>

#include <dbot_ros/utils/ros_interface.hpp>



namespace rmsgf
{

struct Args
{
    Args(ros::NodeHandle& nh)
        : nh_(nh)
    { }

    template<typename Parameter>
    void get(const std::string& path, Parameter& parameter)
    {
        XmlRpc::XmlRpcValue ros_parameter;
        nh_.getParam(path, ros_parameter);
        parameter = Parameter(ros_parameter);
    }

    ros::NodeHandle& nh_;
};


/**
 * Declaration of the filter builder
 */
template <
    template <typename ...> class FilterClass,
    template <typename...> class TailModelClass,
    typename Quadrature,
    typename StateType
>
struct FilterBuilder;

/**
 * Declaration of the filter builder base
 */
template <
    template <typename...> class Filter,
    typename Quadrature,
    typename StateType
>
struct FilterBuilderBase
{
    /* ---------------------------------------------------------------------- */
    /* - Basic Types                                                        - */
    /* ---------------------------------------------------------------------- */
    typedef StateType      State;
    typedef fl::Vector1d   Input;

    /* ---------------------------------------------------------------------- */
    /* - State Transition Model                                             - */
    /* ---------------------------------------------------------------------- */
    typedef fl::LinearStateTransitionModel<State, Input> LinearStateModel;

    struct Parameter
    {
        Parameter(Args& args)
        {
            args.get("/rmsgf/linear_sigma",    linear_sigma);
            args.get("/rmsgf/angular_sigma",   angular_sigma);
            args.get("/rmsgf/velocity_factor", velocity_factor);
            args.get("obsrv_bg_depth",         obsrv_bg_depth);
            args.get("obsrv_fg_noise_std",     obsrv_fg_noise_std);
            args.get("obsrv_bg_noise_std",     obsrv_bg_noise_std);
            args.get("obsrv_body_tail_weight", obsrv_body_tail_weight);
            args.get("uniform_tail_min",       uniform_tail_min);
            args.get("uniform_tail_max",       uniform_tail_max);
            args.get("ut_alpha",               ut_alpha);
            args.get("downsampling",           downsampling);
            args.get("resX",                   resX);
            args.get("resY",                   resY);

            sensors = (resX / downsampling) * (resY / downsampling);
        }

        int sensors;

        int downsampling;
        fl::Real linear_sigma;
        fl::Real angular_sigma;
        fl::Real velocity_factor;
        fl::Real obsrv_bg_depth;
        fl::Real obsrv_fg_noise_std;
        fl::Real obsrv_bg_noise_std;
        fl::Real obsrv_body_tail_weight;
        fl::Real uniform_tail_min;
        fl::Real uniform_tail_max;
        fl::Real ut_alpha;
        int resX;
        int resY;

        void print()
        {
            PF(sensors);
            PF(linear_sigma);
            PF(angular_sigma);
            PF(velocity_factor);
            PF(obsrv_bg_depth);
            PF(obsrv_fg_noise_std);
            PF(obsrv_bg_noise_std);
            PF(obsrv_body_tail_weight);
            PF(uniform_tail_min);
            PF(uniform_tail_max);
            PF(ut_alpha);
            PF(downsampling);
            PF(resX);
            PF(resY);
        }
    };


    /**
     * \brief Creates an instance of the linear state transition model
     */
    template <typename T> typename std::enable_if<
        std::is_same<T, osr::PoseVector>::value,
        LinearStateModel
    >::type
    create_state_transtion_model(const Parameter& param, T t )
    {
        auto state_transition_model = LinearStateModel();

        auto A = state_transition_model.create_dynamics_matrix();
        auto B = state_transition_model.create_input_matrix();
        auto Q = state_transition_model.create_noise_matrix();

        A.setIdentity();
        B.setZero();
        Q.topLeftCorner(3,3) *= param.linear_sigma;
        Q.bottomRightCorner(3,3) *=  param.angular_sigma;

        state_transition_model.dynamics_matrix(A);
        state_transition_model.input_matrix(B);
        state_transition_model.noise_matrix(Q);

        PV(state_transition_model.noise_matrix());
        PV(state_transition_model.dynamics_matrix());

        return state_transition_model;
    }

    /**
     * \brief Creates an instance of the linear state transition model
     */
    template <typename T> typename std::enable_if<
        std::is_same<T, osr::PoseVelocityVector>::value,
        LinearStateModel
    >::type
    create_state_transtion_model(const Parameter& param, T t )
    {
        auto state_transition_model = LinearStateModel();

        auto A = state_transition_model.create_dynamics_matrix();
        auto B = state_transition_model.create_input_matrix();
        auto Q = state_transition_model.create_noise_matrix();

        A.setIdentity();
        A.topRightCorner(6,6).setIdentity();
        A.rightCols(6) *= param.velocity_factor;

        B.setZero();

//        Q.setIdentity();
//        Q.block(0, 0, 3, 3) *= param.linear_sigma;
//        Q.block(3, 3, 3, 3) *= param.angular_sigma;
//        Q.block(6, 6, 3, 3) *= param.state_pos_vel_noise_std;
//        Q.block(9, 9, 3, 3) *= param.state_rot_vel_noise_std;


        Q.setZero();
        Q.block(6,0,3,3) = Eigen::Matrix3d::Identity() * param.linear_sigma;
        Q.block(9,3,3,3) = Eigen::Matrix3d::Identity() * param.angular_sigma;
        Q.topRows(6) = Q.bottomRows(6);

        state_transition_model.dynamics_matrix(A);
        state_transition_model.input_matrix(B);
        state_transition_model.noise_matrix(Q);

        PV(state_transition_model.dynamics_matrix());
        PV(state_transition_model.noise_matrix());

        return state_transition_model;
    }
};

}
