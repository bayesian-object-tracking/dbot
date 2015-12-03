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
 * @date 05/25/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 *  University of Southern California
 */

#pragma once

#include <Eigen/Dense>

namespace dbot
{

namespace internal
{
/**
 * \internal
 * Generic distribution trait template
 */
template <typename T> struct Traits { };

//struct Empty { };
typedef Eigen::Matrix<double, 0, 0> Empty;
}

using internal::Traits;

}
