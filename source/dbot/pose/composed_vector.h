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
 * \file composed_vector.h
 * \date August 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

namespace dbot
{
template <typename Block, typename Vector>
class ComposedVector : public Vector
{
public:
    // constructor and destructor **********************************************
    ComposedVector() {}
    template <typename T>
    ComposedVector(const Eigen::MatrixBase<T>& vector)
        : Vector(vector)
    {
    }

    virtual ~ComposedVector() noexcept {}
    // operators ***************************************************************
    template <typename T>
    void operator=(const Eigen::MatrixBase<T>& vector)
    {
        *((Vector*)(this)) = vector;
    }

    // accessors ***************************************************************
    const Block component(int index) const
    {
        return Block(*this, index * Block::SizeAtCompileTime);
    }
    int count() const { return this->size() / Block::SizeAtCompileTime; }
    // mutators ****************************************************************
    Block component(int index)
    {
        return Block(*this, index * Block::SizeAtCompileTime);
    }
    void recount(int new_count)
    {
        return this->resize(new_count * Block::SizeAtCompileTime);
    }
};
}
