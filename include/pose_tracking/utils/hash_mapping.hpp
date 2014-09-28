/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
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
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <iostream>

#include <Eigen/Dense>

#include <vector>

#include <boost/unordered_map.hpp>

namespace Eigen
{

/**
 * Hashes an Eigen::MatrixXd
 *
 * @param matrix    Hash source
 *
 * @return hash seed
 */
std::size_t hash_value(Eigen::MatrixXd const& matrix);

}


//namespace ff
//{

//template <typename Key, typename Data>
//class IndexedCache
//{
//public:
//    IndexedCache()
//    {

//    }

//    void reserve(size_t size)
//    {
//        data_.clear();
//        data_.resize(size, {false, Data()});
//    }

//    void clear()
//    {
//        for (auto& datum: data_)
//        {
//            datum.first = false;
//        }
//    }

//    bool hit(const Key& key, const size_t index)
//    {
//        if (!data_[index].first)
//        {
//            std::cout << "MISSSSSSSSSSSSSSSSSSS" << std::endl;
//            if (key_map_.find(key) != key_map_.end())
//            {
//                std::cout << "FOUNDDDDDDDDDDDDDd" << std::endl;
//                data_[index].first = true;
//                data_[index].second = data_[key_map_[key]].second;
//            }
//            else
//            {
//                return false;
//            }
//        }

//        return true;
//    }

//    void update(const Key& key, const size_t index)
//    {
//        data_[index].first = true;
//        key_map_[key] = index;
//    }

//    Data& data(const size_t index)
//    {
//        return data_[index].second;
//    }

//protected:
//    std::vector<std::pair<bool, Data> > data_;
//    boost::unordered_map<Key, size_t> key_map_;
//};

//}



