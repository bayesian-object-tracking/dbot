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
 * \file object_tracker.hpp
 * \date December 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <memory>
#include <mutex>

#include <dbot/object_model.hpp>

#include <osr/pose_vector.hpp>
#include <osr/composed_vector.hpp>
#include <osr/free_floating_rigid_bodies_state.hpp>

namespace dbot
{

/**
 * \brief Abstract ObjectTracker context
 */
class ObjectTracker
{
public:
    typedef osr::FreeFloatingRigidBodiesState<> State;
    typedef Eigen::Matrix<fl::Real, Eigen::Dynamic, 1> Obsrv;
    typedef Eigen::Matrix<fl::Real, Eigen::Dynamic, 1> Noise;
    typedef Eigen::Matrix<fl::Real, Eigen::Dynamic, 1> Input;

public:
    /**
     * \brief Creates the tracker
     *
     * \param filter
     *     Rbc particle filter instance
     * \param object_model
     *     Object model instance
     * \param update_rate
     *     Moving average update rate
     */
    ObjectTracker(const std::shared_ptr<ObjectModel>& object_model,
                  double update_rate,
                  bool center_object_frame);

    virtual ~ObjectTracker() { }

    /**
     * \brief Hook function which is called during tracking
     * \return Current belief state
     */
    virtual State on_track(const Obsrv& image) = 0;

    /**
     * \brief Hook function which is called during initialization
     * \return Initial belief state
     */
    virtual State on_initialize(const std::vector<State>& initial_states) = 0;

    /**
     * \brief perform a single filter step
     *
     * \param image
     *     Current observation image
     */
    virtual State track(const Obsrv& image);

    /**
     * \brief Initializes the particle filter with the given initial states and
     *     the number of evaluations
     * @param initial_states
     * @param evaluation_count
     */
    virtual void initialize(const std::vector<State>& initial_states);

    /**
     * \brief Transforms the given state or pose in the model coordinate system
     *        to the center coordinate system
     * \param state
     *          Object pose in the model coordinate system
     */
    State to_center_coordinate_system(const State& state);

    /**
     * \brief Transforms the given state or pose in the center coordinate system
     *        to the model coordinate system
     * \param state
     *          Object pose in the center coordinate system
     */
    State to_model_coordinate_system(const State& state);

    /**
     * \brief Updates the moving average with the new state using the specified
     *        update rate. The update rate is the weight on the new state. That
     *        is the new moving average is (1-update_rate) * moving_average +
     *        update_rate * new_state
     * \param moving_average
     *          Last moveing average state
     * @param new_state
     *          New incoming state
     * @param update_rate
     *          Moving average update rate. The update rate is the weight on the
     *          new state.
     */
    void move_average(const State& new_State,
                      State& moving_average,
                      double update_rate);

    /**
     * \brief Shorthand for a zero input vector
     */
    Input zero_input() const;

protected:
    std::shared_ptr<ObjectModel> object_model_;
    State moving_average_;
    double update_rate_;
    bool center_object_frame_;
    std::mutex mutex_;
};
}
