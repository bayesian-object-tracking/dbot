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

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
 * Probabilistic Object Tracking using a Range Camera
 * IEEE Intl Conf on Intelligent Robots and Systems, 2013
 * http://arxiv.org/abs/1505.00241
 *
 */

/**
 * \file rb_observation_model_builder.cpp
 * \date January 2016
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <osr/free_floating_rigid_bodies_state.hpp>

#include <dbot/builder/rb_observation_model_builder.h>
#include <dbot/builder/rb_observation_model_builder.hpp>


namespace dbot
{
template class RbObservationModelBuilder<osr::FreeFloatingRigidBodiesState<>>;
}
