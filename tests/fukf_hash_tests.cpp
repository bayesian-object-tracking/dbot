

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <pose_tracking/utils/hash_mapping.hpp>

size_t dimension = 51;
size_t iterations = 30*640*480/64;

TEST(HashingTests, eigenmatrix_find)
{
    size_t keys = 0;
    std::vector<Eigen::MatrixXd> points;
    boost::unordered_map<Eigen::MatrixXd, Eigen::MatrixXd> hash_map;

    for (size_t i = 0; i < dimension; ++i)
    {
        points.push_back(Eigen::MatrixXd::Random(10,1));
    }

    for (size_t i = 0; i < iterations; ++i)
    {
        for (size_t j = 0; j < dimension; ++j)
        {
            if(hash_map.find(points[j]) == hash_map.end())
            {
                keys++;
                hash_map[points[j]] = Eigen::MatrixXd::Random(1000,1);
            }
        }
    }

    std::cout << "find::iterations " << iterations*dimension << " with " << keys  << " different keys" << std::endl;
}

TEST(HashingTests, eigenmatrix_hash)
{
    size_t keys = 0;
    std::vector<Eigen::MatrixXd> points;
    boost::unordered_map<Eigen::MatrixXd, bool> hash_map;

    for (size_t i = 0; i < dimension; ++i)
    {
        points.push_back(Eigen::MatrixXd::Random(10,1));
    }

    size_t seed;

    for (size_t i = 0; i < iterations; ++i)
    {
        for (size_t j = 0; j < dimension; ++j)
        {
            keys++;
            seed = Eigen::hash_value(points[j]);
        }
    }

    std::cout << "find::iterations " << iterations*dimension << " with " << keys  << " different keys" << std::endl;
}

Eigen::MatrixXd expensive_function()
{
    return Eigen::MatrixXd::Random(100, 1);
}

TEST(IndexLookup, lookup_only)
{
    size_t keys = 0;
    std::vector<Eigen::MatrixXd> points;
    std::vector<std::pair<bool, Eigen::MatrixXd> > lookup;

    lookup.resize(dimension, {false, Eigen::MatrixXd()});


    for (size_t i = 0; i < dimension; ++i)
    {
        points.push_back(Eigen::MatrixXd::Random(10,1));
    }

    for (size_t i = 0; i < iterations; ++i)
    {
        for (size_t j = 0; j < dimension; ++j)
        {
            if (!lookup[j].first)
            {
                keys++;

                lookup[j].first = true;
                lookup[j].second = expensive_function();
            }
        }
    }

    std::cout << "lookup::iterations " << iterations*dimension << " with " << keys  << " different keys" << std::endl;
}


//TEST(IndexLookup, indexed_cache)
//{
//    ff::IndexedCache<Eigen::MatrixXd, Eigen::MatrixXd> cache;
//    cache.reserve(dimension);

//    std::vector<Eigen::MatrixXd> points;
//    for (size_t i = 0; i < dimension; ++i)
//    {
//        points.push_back(Eigen::MatrixXd::Random(10,1));
//    }

//    points[10] = points[1];
//    points[22] = points[3];
//    points[30] = points[3];
//    points[31] = points[3];
//    points[32] = points[3];
//    points[34] = points[3];

//    Eigen::MatrixXd temp;

//    for (size_t i = 0; i < iterations; ++i)
//    {
//        for (size_t j = 0; j < dimension; ++j)
//        {
////            if (cache.hit(points[j], j))
////            {
////                temp = cache.data(j);
////            }
////            else
////            {
////                std::cout << "CREATINGGGGGGGGGGGGGGGGGGGg" << std::endl;

////                cache.data(j) = expensive_function();
////                cache.update(points[j], j);
////            }

//            if (i == 0)
//            {
//                cache.data(j) = expensive_function();
//                cache.update(points[j], j);
//                temp = cache.data(j);
//            }
//            else
//            {
//                temp = cache.data(j);
//            }
//        }
//    }
//}

