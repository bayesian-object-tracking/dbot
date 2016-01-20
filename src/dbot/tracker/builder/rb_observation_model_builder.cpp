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
 * \file rb_observation_model_builder.cpp
 * \date January 2016
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <osr/free_floating_rigid_bodies_state.hpp>

#include <dbot/tracker/builder/rb_observation_model_builder.h>
#include <dbot/tracker/builder/rb_observation_model_builder.hpp>


namespace dbot
{
template class RbObservationModelBuilder<osr::FreeFloatingRigidBodiesState<>>;
}
