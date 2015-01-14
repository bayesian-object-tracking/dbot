/*************************************************************************
This software allows for filtering in high-dimensional observation and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/

#ifndef POSE_TRACKING__UTILS_HELPER_FUNCTIONS_HPP
#define POSE_TRACKING__UTILS_HELPER_FUNCTIONS_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <cmath>

#include <ctime>
#include <fstream>



#include <boost/lexical_cast.hpp>
#include <boost/random/lagged_fibonacci.hpp>

#include <fl/util/random_seed.hpp>



// TODO: THIS HAS TO BE CLEANED, POSSIBLY SPLIT INTO SEVERAL FILES

namespace fl
{

namespace hf
{

template <typename T>  void PrintVector(std::vector<T> v)
{
    for(size_t i = 0; i < v.size(); i++)
        std::cout << "(" << i << ": " << v[i] << ") ";
    std::cout << std::endl;
}
template <typename T> void PrintVector(std::vector<std::vector<T>> v)
{
    for(size_t i = 0; i < v.size(); i++)
    {
        std::cout << i <<  " --------------------------------" << std::endl;
        PrintVector(v[i]);
    }
}

template <typename T> void PrintVector(std::vector<std::vector<std::vector<T>> > v)
{
    for(size_t i = 0; i < v.size(); i++)
    {
        std::cout << i <<  " ================================" << std::endl;
        PrintVector(v[i]);
    }
}

// application specific

// depth image functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// converts from cartesian to image space
template <typename T> Eigen::Matrix<T, 2, 1>
CartCoord2ImageCoord(const Eigen::Matrix<T, 3, 1>& cart,
                     const Eigen::Matrix<T, 3, 3>& camera_matrix)
{
    return (camera_matrix * cart/cart(2)).topRows(2);
}

// converts from cartesian space to image index, (row, col)
template <typename T> Eigen::Matrix<int, 2, 1>
CartCoord2ImageIndex(const Eigen::Matrix<T, 3, 1>& cart,
                     const Eigen::Matrix<T, 3, 3>& camera_matrix)
{
    Eigen::Matrix<T, 2, 1> image = CartCoord2ImageCoord(cart, camera_matrix);
    Eigen::Matrix<int, 2, 1> image_index;
    image_index(0) = floor(image(1)+0.5);
    image_index(1) = floor(image(0)+0.5);

    return image_index;
}

// converts from image coordinates and depth (z value) to cartesian coordinates
template <typename T> Eigen::Matrix<T, 3, 1>
ImageCoord2CartCoord(const Eigen::Matrix<T, 2, 1>& image,
                     const T& depth,
                     const Eigen::Matrix<T, 3, 3>& camera_matrix_inverse)
{
    Eigen::Matrix<T, 3, 1> image_augmented;
    image_augmented.topRows(2) = image;
    image_augmented(2) = 1;

    return depth * camera_matrix_inverse * image_augmented;
}

// converts from image index (row, col) and depth (z value) to cartesian coordinates
template <typename T> Eigen::Matrix<T, 3, 1>
ImageIndex2CartCoord(const Eigen::Matrix<int, 2, 1>& image_index,
                     const T& depth,
                     const Eigen::Matrix<T, 3, 3>& camera_matrix_inverse)
{
    Eigen::Matrix<T, 2, 1> image;
    image(0) = image_index(1);
    image(1) = image_index(0);
    return ImageCoord2CartCoord(image, depth, camera_matrix_inverse);
}


template <typename T> Eigen::Matrix<Eigen::Matrix<T, 3, 1> , -1, -1>
Image2Points(const Eigen::Matrix<T, -1, -1>& image,
             const Eigen::Matrix<T, 3, 3>& camera_matrix)
{
    Eigen::Matrix<T, 3, 3> camera_matrix_inverse = camera_matrix.inverse();

    Eigen::Matrix<Eigen::Matrix<T, 3, 1> , -1, -1> points(image.rows(), image.cols());
    for(size_t row = 0; row < points.rows(); row++)
        for(size_t col = 0; col < points.cols(); col++)
            points(row, col) = ImageIndex2CartCoord(Eigen::Vector2i(row, col),
                                                    image(row, col),
                                                    camera_matrix_inverse);

    return points;
}



// actually not used anywhere
template <typename T> struct ValueIndex
{
    T value;
    int index;

    bool operator < (const ValueIndex& str) const
    {
        return (value < str.value);
    }
};
template <typename T> std::vector<int> SortAscend(const std::vector<T> &values)
{
    std::vector<int> indices(values.size());

    std::vector<ValueIndex<T>> values_indices(values.size());
    for(int i = 0; i < int(values.size()); i++)
    {
        values_indices[i].index = i;
        values_indices[i].value = values[i];
    }

    std::sort(values_indices.begin(), values_indices.end());

    for(int i = 0; i < int(indices.size()); i++)
        indices[i] = values_indices[i].index;

    return indices;
}

template <typename T> std::vector<int> SortDescend(const std::vector<T> &values)
{
    std::vector<int> ascend_indices = SortAscend(values);
    std::vector<int> descend_indices(ascend_indices.size());

    for(int i = 0; i < int(ascend_indices.size()); i++)
        descend_indices[i] = ascend_indices[ascend_indices.size()-1-i];

    return descend_indices;
}


template <typename T> void Sort(std::vector<T> &ref_values, bool order = 0)
{
    std::vector<T> temp_ref_values = ref_values;
    std::vector<int> indices;
    if(order == 0)
        indices = SortAscend(temp_ref_values);
    else
        indices = SortDescend(temp_ref_values);


    for(int i = 0; i < int(indices.size()); i++)
        ref_values[i] = temp_ref_values[indices[i]];
}


template <typename T0, typename T1> void SortAscend(std::vector<T0> &ref_values, std::vector<T1> &values)
{
    std::vector<T0> temp_ref_values = ref_values;
    std::vector<T1> temp_values = values;

    std::vector<int> indices = SortAscend(temp_ref_values);

    for(int i = 0; i < int(indices.size()); i++)
    {
        ref_values[i] = temp_ref_values[indices[i]];
        values[i] = temp_values[indices[i]];
    }
}

template <typename T0, typename T1> void SortDescend(std::vector<T0> &ref_values, std::vector<T1> &values)
{
    std::vector<T0> temp_ref_values = ref_values;
    std::vector<T1> temp_values = values;

    std::vector<int> indices = SortDescend(temp_ref_values);

    for(int i = 0; i < int(indices.size()); i++)
    {
        ref_values[i] = temp_ref_values[indices[i]];
        values[i] = temp_values[indices[i]];
    }
}

template <typename T0, typename T1, typename T2> void SortAscend(std::vector<T0> &ref_values,
        std::vector<T1> &values1, std::vector<T2> &values2)
{
    std::vector<T0> temp_ref_values = ref_values;
    std::vector<T1> temp_values1 = values1;
    std::vector<T2> temp_values2 = values2;

    std::vector<int> indices = SortAscend(temp_ref_values);

    for(int i = 0; i < int(indices.size()); i++)
    {
        ref_values[i] = temp_ref_values[indices[i]];
        values1[i] = temp_values1[indices[i]];
        values2[i] = temp_values2[indices[i]];
    }
}

template <typename T0, typename T1, typename T2> void SortDescend(std::vector<T0> &ref_values,
        std::vector<T1> &values1, std::vector<T2> &values2)
{
    std::vector<T0> temp_ref_values = ref_values;
    std::vector<T1> temp_values1 = values1;
    std::vector<T2> temp_values2 = values2;

    std::vector<int> indices = SortDescend(temp_ref_values);

    for(int i = 0; i < int(indices.size()); i++)
    {
        ref_values[i] = temp_ref_values[indices[i]];
        values1[i] = temp_values1[indices[i]];
        values2[i] = temp_values2[indices[i]];
    }
}

// not used! remove?
class DiscreteSampler
{
public:
    template <typename T> DiscreteSampler(std::vector<T> log_prob)
    {
        fibo_.seed(RANDOM_SEED);

        // compute the prob and normalize them
        sorted_indices_ = fl::hf::SortDescend(log_prob);
        double max = log_prob[sorted_indices_[0]];
        for(int i = 0; i < int(log_prob.size()); i++)
            log_prob[i] -= max;

        std::vector<double> prob(log_prob.size());
        double sum = 0;
        for(int i = 0; i < int(log_prob.size()); i++)
        {
            prob[i] = exp(log_prob[i]);
            sum += prob[i];
        }
        for(int i = 0; i < int(prob.size()); i++)
            prob[i] /= sum;

        // compute the cumulative probability
        cumulative_prob_.resize(log_prob.size());
        cumulative_prob_[0] = prob[sorted_indices_[0]];
        for(int i = 1; i < int(log_prob.size()); i++)
            cumulative_prob_[i] = cumulative_prob_[i-1] + prob[sorted_indices_[i]];
    }

    ~DiscreteSampler() {}

    int Sample()
    {
        double uniform_sample = fibo_();
        int sample_index = 0;
        while(uniform_sample > cumulative_prob_[sample_index]) sample_index++;

        return sorted_indices_[sample_index];
    }



    int MapStandardGaussian(double gaussian_sample) const
    {
        // map from a gaussian to a uniform distribution
        double uniform_sample = 0.5 *
                (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        int sample_index = 0;
        while(uniform_sample > cumulative_prob_[sample_index]) sample_index++; //TODO: THIS COULD BE DONE IN LOG TIME

        return sorted_indices_[sample_index];
    }

private:
    boost::lagged_fibonacci607 fibo_;
    std::vector<int> sorted_indices_;
    std::vector<double> cumulative_prob_;
};




}

}

#endif
