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
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <cstdlib>
#include <Eigen/Dense>

#include <osr/pose_vector.hpp>
#include <osr/pose_velocity_vector.hpp>
#include <osr/free_floating_rigid_bodies_state.hpp>


namespace osr
{
template <typename Vector>
class PoseHash;

template <>
class PoseHash<PoseVector>
{
public:
    std::size_t operator()(const PoseVector& s) const
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

        return ((int(s(0, 0) * c) * p1) ^ (int(s(1, 0) * c) * p2) ^
                (int(s(2, 0) * c) * p3) ^ (int(s(3, 0) * c) * p4) ^
                (int(s(4, 0) * c) * p5) ^ (int(s(5, 0) * c) * p6) % n);
    }
};

template <>
class PoseHash<PoseVelocityVector>
{
public:
    std::size_t operator()(const PoseVelocityVector& s) const
    {
        auto h = PoseHash<PoseVector>();

        std::size_t hash = h(s.pose());

        return hash;
    }
};
template <>
class PoseHash<FreeFloatingRigidBodiesState<>>
{
public:
    std::size_t operator()(const FreeFloatingRigidBodiesState<>& s) const
    {
        auto h = PoseHash<PoseVector>();

        std::size_t hash = 0;
        for (int i = 0; i < s.count(); ++i)
        {
            hash += h(s.component(i).pose());
        }

        return hash;
    }
};
}
