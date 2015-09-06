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

#pragma once


#include <cstdlib>
#include <Eigen/Dense>

#include <fl/util/types.hpp>
#include <fl/util/math/pose_vector.hpp>

template <typename Vector> class PoseHash;

template <> class PoseHash<fl::PoseVector>
{
public:
    std::size_t operator()(const fl::PoseVector & s) const
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

        return  ((int(s(0, 0) * c) * p1) ^
                 (int(s(1, 0) * c) * p2) ^
                 (int(s(2, 0) * c) * p3) ^
                 (int(s(3, 0) * c) * p4) ^
                 (int(s(4, 0) * c) * p5) ^
                 (int(s(5, 0) * c) * p6) % n);
    }
};
