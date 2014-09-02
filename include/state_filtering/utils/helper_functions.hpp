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

#ifndef STATE_FILTERING_UTILS_HELPER_FUNCTIONS_HPP_
#define STATE_FILTERING_UTILS_HELPER_FUNCTIONS_HPP_

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

#include <boost/lexical_cast.hpp>
#include <boost/random/lagged_fibonacci.hpp>

#include <state_filtering/utils/macros.hpp>


namespace hf
{

inline double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        std::cout << "WARNING: gettimeofday() Error" << std::endl;
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


template <typename T> void Compare(const std::vector<T> vector_1,
                              const std::vector<T> vector_2,
                              char* name_1,
                              char* name_2) {
    if (vector_1.size() == vector_2.size()) {
        T difference = 0;
        for (uint i = 0; i < vector_1.size(); i++) {
            std::cout << name_1 << "[" << i << "]: " << vector_1[i] << ", "
                      << name_2 << "[" << i << "]: " << vector_2[i] << ", "
                      << "difference: " << std::abs(vector_1[i] - vector_2[i]) << std::endl;
            difference += std::abs(vector_1[i] - vector_2[i]);
        }
        std::cout << "average difference between " << name_1 << " and " << name_2 << " : " << difference / vector_1.size() << std::endl;

    } else {
        std::cout << "WARNING: The two vectors to be compared have a different amount of elements. " << std::endl
                  << name_1 << ": " << vector_1.size() << ", " << name_2 << ": " << vector_2.size() << std::endl
                  << "Comparison aborted." << std::endl;
    }
}

template <typename T> void AverageDifference(const std::vector<T> vector_1,
                                             const std::vector<T> vector_2,
                                             char* name_1,
                                             char* name_2) {
    if (vector_1.size() == vector_2.size()) {
        T difference = 0;
        for (uint i = 0; i < vector_1.size(); i++) {
            difference += std::abs(vector_1[i] - vector_2[i]);
        }
        std::cout << "average difference between " << name_1 << " and " << name_2 << " : " << difference / vector_1.size() << std::endl;

    } else {
        std::cout << "WARNING: The two vectors to be compared have a different amount of elements. " << std::endl
                  << name_1 << ": " << vector_1.size() << ", " << name_2 << ": " << vector_2.size() << std::endl
                  << "Comparison aborted." << std::endl;
    }
}

template <typename T> void Output(const T value,
                                  char* name) {
    std::cout << name << ": " << value << std::endl;
}


template <typename T>
int IndexNextReal(
		const std::vector<T>& data,
		const int& current_index = -1,
		const int& step_size = 1)
{
	int index_next_real = current_index + step_size;
	while(
			index_next_real < int(data.size()) &&
			index_next_real >= 0 &&
			!std::isfinite(data[index_next_real])) index_next_real+=step_size;

	return index_next_real;
}

//  various useful functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class Structurer2D
{
public:
	Structurer2D(){}

	~Structurer2D() {}

	template <typename T> std::vector<T> Flatten(const std::vector<std::vector<T> >& deep_structure)
	{
		sizes_.resize(deep_structure.size());
		size_t global_size = 0;
		for(size_t  i = 0; i < deep_structure.size(); i++)
		{
			sizes_[i] = deep_structure[i].size();
			global_size += deep_structure[i].size();
		}

		std::vector<T> flat_structure(global_size);
		size_t global_index = 0;
		for(size_t  i = 0; i < deep_structure.size(); i++)
			for(size_t j = 0; j < deep_structure[i].size(); j++)
				flat_structure[global_index++] = deep_structure[i][j];

		return flat_structure;
	}

	template <typename T> std::vector<std::vector<T> > Deepen(const std::vector<T>& flat_structure)
	{
		std::vector<std::vector<T> > deep_structure(sizes_.size());

		size_t global_index = 0;
		for(size_t i = 0; i < deep_structure.size(); i++)
		{
			deep_structure[i].resize(sizes_[i]);
			for(size_t j = 0; j < deep_structure[i].size(); j++)
				deep_structure[i][j] = flat_structure[global_index++];
		}
		return deep_structure;
	}

private:
	std::vector<size_t> sizes_;
};



class Structurer3D
{
public:
	Structurer3D(){}

	~Structurer3D() {}

	template <typename T> std::vector<T> Flatten(const std::vector<std::vector<std::vector<T> > >& deep_structure)
	{
		size_t global_size = 0;
		sizes_.resize(deep_structure.size());
		for(size_t i = 0; i < deep_structure.size(); i++)
		{
			sizes_[i].resize(deep_structure[i].size());
			for(size_t  j = 0; j < deep_structure[i].size(); j++)
			{
				sizes_[i][j] = deep_structure[i][j].size();
				global_size += deep_structure[i][j].size();
			}
		}

		std::vector<T> flat_structure(global_size);
		size_t global_index = 0;
		for(size_t  i = 0; i < deep_structure.size(); i++)
			for(size_t j = 0; j < deep_structure[i].size(); j++)
				for(size_t k = 0; k < deep_structure[i][j].size(); k++)
					flat_structure[global_index++] = deep_structure[i][j][k];

		return flat_structure;
	}

	template <typename T> std::vector<std::vector<std::vector<T> > > Deepen(const std::vector<T>& flat_structure)
	{
		size_t global_index = 0;
		std::vector<std::vector<std::vector<T> > > deep_structure(sizes_.size());
		for(size_t i = 0; i < deep_structure.size(); i++)
		{
			deep_structure[i].resize(sizes_[i].size());
			for(size_t j = 0; j < deep_structure[i].size(); j++)
			{
				deep_structure[i][j].resize(sizes_[i][j]);
				for(size_t k = 0; k < deep_structure[i][j].size(); k++)
					deep_structure[i][j][k] = flat_structure[global_index++];
			}
		}
		return deep_structure;
	}

private:
	std::vector<std::vector<size_t> > sizes_;
};







// this function will interpolat the vector wherever it is NAN or INF
template <typename T>
void LinearlyInterpolate(std::vector<T>& data)
{
	std::vector<int> limits;
	limits.push_back(0);
	limits.push_back(data.size()-1);
	std::vector<int> step_direction;
	step_direction.push_back(1);
	step_direction.push_back(-1);

	// extrapolate
	for(int i = 0; i < 2; i++)
	{
		int index_first_real = IndexNextReal(data, limits[i] - step_direction[i], step_direction[i]);
		int index_next_real = IndexNextReal(data, index_first_real, step_direction[i]);
		if(index_next_real >= int(data.size()) || index_next_real < 0)
			return;

		double slope =
				double(data[index_next_real] - data[index_first_real]) /
				double(index_next_real - index_first_real);

		for(int j = limits[i]; j != index_first_real; j += step_direction[i])
			data[j] = data[index_first_real] + (j - index_first_real) * slope;
	}

	// interpolate
	int index_current_real = IndexNextReal(data, limits[0] - step_direction[0], step_direction[0]);
	int index_next_real = IndexNextReal(data, index_current_real, step_direction[0]);

	while(index_next_real < int(data.size()))
	{
		double slope =
				double(data[index_next_real] - data[index_current_real]) /
				double(index_next_real - index_current_real);

		for(int i = index_current_real + 1; i < index_next_real; i++)
			data[i] =  data[index_current_real] + (i - index_current_real) * slope;

		index_current_real = index_next_real;
		index_next_real = IndexNextReal(data, index_next_real, step_direction[0]);
	}
}

template <typename T>  void PrintVector(std::vector<T> v)
{
	for(size_t i = 0; i < v.size(); i++)
		std::cout << "(" << i << ": " << v[i] << ") ";
	std::cout << std::endl;
}
template <typename T> void PrintVector(std::vector<std::vector<T> > v)
{
	for(size_t i = 0; i < v.size(); i++)
	{
		std::cout << i <<  " --------------------------------" << std::endl;
		PrintVector(v[i]);
	}
}

template <typename T> void PrintVector(std::vector<std::vector<std::vector<T> > > v)
{
	for(size_t i = 0; i < v.size(); i++)
	{
		std::cout << i <<  " ================================" << std::endl;
		PrintVector(v[i]);
	}
}


template <typename T>
void Swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template <typename T>
Eigen::Matrix<T, -1, 1> Std2Eigen(const std::vector<T> &std)
{
	Eigen::Matrix<T, -1, 1> eigen(std.size());
	for(size_t i = 0; i < std.size(); i++)
		eigen(i) = std[i];
	return eigen;
}

template <typename Derived>
std::string Eigen2Mathematica(const Eigen::MatrixBase<Derived>& eigen, const std::string &name = "")
{
	std::string mathematica;
	if(!name.empty())
		mathematica += name + " = ";

	mathematica += "{";
	for(int row = 0; row < eigen.rows(); row++)
	{
		mathematica += "{";
		for(int col = 0; col < eigen.cols(); col++)
		{
			mathematica += boost::lexical_cast<std::string>(eigen(row, col));
			if(col != eigen.cols() -1)
				mathematica += ", ";
		}
		mathematica += "}";
		if(row != eigen.rows()-1)
			mathematica += ", ";
	}
	mathematica += "}";
	if(!name.empty())
		mathematica += ";";

	return mathematica;
}

// returns the first index where the two vectors differ, if there is no difference then -1 is returned.
template <typename T, int n_rows, int n_cols> int
DifferenceAt(
		const std::vector<Eigen::Matrix<T, n_rows, n_cols> >& a,
		const std::vector<Eigen::Matrix<T, n_rows, n_cols> >& b,
		const T& epsilon)
{
	if(a.size() < b.size())
		return a.size();
	else if(b.size() < a.size())
		return b.size();

	for(size_t i = 0; i < a.size(); i++)
		for(size_t row = 0; row < n_rows; row++)
			for(size_t col = 0; col < n_cols; col++)
			{
				if(isnan(a[i](row, col)) && isnan(b[i](row, col)))
					continue;
				else if(isnan(a[i](row, col)) || isnan(b[i](row, col)))
					return i;
				else if( (a[i](row, col) - b[i](row, col) > epsilon) || (b[i](row, col) - a[i](row, col) > epsilon) )
					return i;
			}

	return -1;
}

// returns the first index where the two vectors differ, if there is no difference then -1 is returned.
template <typename T> int
DifferenceAt(
		const std::vector<T>& a,
		const std::vector<T>& b,
		const T& epsilon)
{
	if(a.size() < b.size())
		return a.size();
	else if(b.size() < a.size())
		return b.size();

	for(size_t i = 0; i < a.size(); i++)
	{
		if(isnan(a[i]) && isnan(b[i]))
			continue;
		else if(isnan(a[i]) || isnan(b[i]))
			return i;
		else if( (a[i] - b[i] > epsilon) || (b[i] - a[i] > epsilon) )
			return i;
	}

	return -1;
}

template <typename T> std::vector<T> ReverseOrder(std::vector<T> ordered)
{
	std::vector<T> reversed(ordered.size());
	for(size_t i = 0; i < ordered.size(); i++)
		reversed[i] = ordered[ordered.size() - 1 - i];
}

template <typename T> std::vector<T> Count(T from, T to, T increment = 1)
{
	std::vector<T> count(size_t((to - from)/increment));
	count[0] = from;
	for(size_t i = 1; i < count.size(); i++)
		count[i] = count[i-1] + increment;
}

template <typename T> struct ValuesIndex
{
	std::vector<T> values;
	int index;

	bool operator < (const ValuesIndex& right_side) const
	{
		for(size_t i = 0; i < values.size(); i++)
		{
			if(values[i] < right_side.values[i])
				return true;
			if(values[i] > right_side.values[i])
				return false;
		}
		return false;
	}


	bool operator != (const ValuesIndex& right_side) const
	{
		for(size_t i = 0; i < values.size(); i++)
			if(values[i] != right_side.values[i])
				return true;

		return false;
	}
};

template <typename T> void
SortAndCollapse(
		std::vector<std::vector<T> >& values,
		std::vector<size_t>& multiplicities)
{
	std::vector<ValuesIndex<T> > values_indices(values.size());
	for(size_t i = 0; i < values.size(); i++)
	{
		values_indices[i].index = i;
		values_indices[i].values = values[i];
	}

	std::sort(values_indices.begin(), values_indices.end());

	std::vector<std::vector<T> > temp_values = values;
	multiplicities = std::vector<size_t>(values.size(), 0);
	size_t distinct_index = 0;
	values[0] = temp_values[values_indices[0].index];
	for(size_t i = 0; i < values.size(); i++)
	{
		if(i > 0 && temp_values[values_indices[i-1].index] != temp_values[values_indices[i].index])
			values[++distinct_index] = temp_values[values_indices[i].index];
		multiplicities[distinct_index]++;
	}
	values.resize(distinct_index+1);
	multiplicities.resize(distinct_index+1);
}










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

	std::vector<ValueIndex<T> > values_indices(values.size());
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

template <typename T> int BoundIndex(const std::vector<T> &values, bool bound_type) // bound type 1 for max and 0 for min
{
	int BoundIndex = 0;
	T bound_value = bound_type ? -std::numeric_limits<T>::max() : std::numeric_limits<T>::max();

	for(int i = 0; i < int(values.size()); i++)
		if(bound_type ? (values[i] > bound_value) : (values[i] < bound_value) )
		{
			BoundIndex = i;
			bound_value = values[i];
		}

	return BoundIndex;
}

template <typename T> T bound_value(const std::vector<T> &values, bool bound_type) // bound type 1 for max and 0 for min
{
	return values[BoundIndex(values, bound_type)];
}

template <typename Tin, typename Tout>
std::vector<Tout> Apply(const std::vector<Tin> &input, Tout(*f)(Tin))
{
	std::vector<Tout> output(input.size());
	for(size_t i = 0; i < output.size(); i++)
		output[i] = (*f)(input[i]);

	return output;
}

template <typename T> std::vector<T>
SetSum(const std::vector<T> &input, T sum)
{
	T old_sum = 0;
	for(size_t i = 0; i < input.size(); i++)
		old_sum += input[i];
	T factor = sum/old_sum;

	std::vector<T> output(input.size());
	for(size_t i = 0; i < input.size(); i++)
		output[i] = factor*input[i];

	return output;
}

// geometry functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <typename T>
Eigen::Quaternion<typename T::Scalar> Delta2Quaternion(const Eigen::MatrixBase<T>& delta)
{
	typename T::Scalar angle = delta.norm();
	Eigen::Matrix<typename T::Scalar, 3, 1> axis = delta.normalized();
	Eigen::Quaternion<typename T::Scalar> q;
	if(std::isfinite(axis.norm()))
		q = Eigen::AngleAxisd(angle, axis);
	else
		q = Eigen::Quaterniond::Identity();

	return q;
}

// this function finds the intersection between two lines. a is a point on line and d_a is the direction
// vector of the line. if lines do not cross, this will return the point which minimizes the squared
// distance between the two lines. if they are parallel, then inf will be returned.
template <typename T1, typename T2, typename T3, typename T4>
Eigen::Matrix<typename T1::Scalar, 3, 1> Intersection(
		const Eigen::MatrixBase<T1>& a,
		const Eigen::MatrixBase<T2>& d_a_in,
		const Eigen::MatrixBase<T3>& b,
		const Eigen::MatrixBase<T4>& d_b_in)
{
	const Eigen::Matrix<typename T1::Scalar, 3, 1> d_a = d_a_in.normalized();
	const Eigen::Matrix<typename T1::Scalar, 3, 1> d_b = d_b_in.normalized();

	double c_a = d_a.dot((b-a) - d_b.dot(b-a)*d_b)/
			(1.-pow(d_b.dot(d_a),2));
	double c_b = d_b.dot((a-b) - d_a.dot(a-b)*d_a)/
			(1.-pow(d_a.dot(d_b),2));

	return((b + c_b*d_b)+(a + c_a*d_a))/2.;
}

template <typename T1, typename T2>
typename T1::Scalar Angle(
		const Eigen::MatrixBase<T1>& a,
		const Eigen::MatrixBase<T2>& b)
{
	return atan2(a.cross(b).norm(), a.dot(b));
}

template <typename T>
Eigen::Matrix<typename T::Scalar, 3, 1> OrthogonalUnitVector(const Eigen::MatrixBase<T>& v)
{
	Eigen::Matrix<typename T::Scalar, 3, 1> n =
		(v.cross(Eigen::Matrix<typename T::Scalar, 3, 1>(1.0, 0, 0))).normalized();
	if(!std::isfinite(n.norm()))
		n = (v.cross(Eigen::Matrix<typename T::Scalar, 3, 1>(0, 1.0, 0))).normalized();

	return n;
}

// returns the angle and axis which rotate a onto b
template <typename T1, typename T2>
Eigen::AngleAxis<typename T1::Scalar> AngleAxis(
		const Eigen::MatrixBase<T1>& a,
		const Eigen::MatrixBase<T2>& b)
{
	Eigen::Matrix<typename T1::Scalar, 3, 1> axis = a.cross(b);
	typename T1::Scalar axis_norm = axis.norm();
	axis /= axis_norm;

	// if the axis is inf then the angle is either 0 or pi, so the axis has to be just some orthogonal vector
	if(!std::isfinite(axis.norm()))
		axis = OrthogonalUnitVector(a);

	typename T1::Scalar angle = atan2(axis_norm, a.dot(b));

	return Eigen::AngleAxis<typename T1::Scalar>(angle, axis);
}

// the input quaternion has the order xyzw
inline Eigen::Matrix<double, 4, 3> QuaternionMatrix(const Eigen::Matrix<double, 4, 1>& q_xyzw)
{
	Eigen::Matrix<double, 4, 3> Q;
	Q <<	q_xyzw(3), q_xyzw(2), -q_xyzw(1),
			-q_xyzw(2), q_xyzw(3), q_xyzw(0),
			q_xyzw(1), -q_xyzw(0), q_xyzw(3),
			-q_xyzw(0), -q_xyzw(1), -q_xyzw(2);
	return 0.5*Q;
}

template <typename T> class TransformationSequence
{
public:
	TransformationSequence(
			const Eigen::Matrix<T, 3, 3> &R = Eigen::Matrix<T, 3, 3>::Identity(),
			const Eigen::Matrix<T, 3, 1> &t = Eigen::Matrix<T, 3, 1>::Zero()): R_(R), t_(t) {}

	~TransformationSequence(){}

	void PreRotate(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &c = Eigen::Matrix<T, 3, 1>::Zero())
	{
		t_ = t_ - R_*R*c + R_*c;
		R_ = R_*R;
	}
	void PreTranslate(const Eigen::Matrix<T, 3, 1> &t)
	{
		t_ = t_ + R_*t;
	}
	void PostRotate(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &c = Eigen::Matrix<T, 3, 1>::Zero())
	{
		t_ = R*t_ - R*c + c;
		R_ = R*R_;
	}
	void PostTranslate(const Eigen::Matrix<T, 3, 1> &t)
	{
		t_ = t_ + t;
	}

	void get(Eigen::Matrix<T, 3, 3> &R, Eigen::Matrix<T, 3, 1> &t)
	{
		R = R_;
		t = t_;
	}

	void NormalizeQuat()
	{
		Eigen::Quaternion<T> q; q = R_;
		q.normalize();
		R_ = q.toRotationMatrix();
	}

private:
	Eigen::Matrix<T, 3, 3> R_;
	Eigen::Matrix<T, 3, 1> t_;

};

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




template <typename T> std::vector<Eigen::Matrix<T, 3, 1> >
DepthImage2CartVectors(const std::vector<T>& image,
                       const size_t& n_rows, const size_t& n_cols,
                       const Eigen::Matrix<T, 3, 3>& camera_matrix)
{
	Eigen::Matrix<T, 3, 3> camera_matrix_inverse = camera_matrix.inverse();

	std::vector<Eigen::Matrix<T, 3, 1> > vectors(n_rows*n_cols);
	for(size_t row = 0; row < n_rows; row++)
		for(size_t col = 0; col < n_cols; col++)
			vectors[row*n_cols + col] =
					ImageIndex2CartCoord(
							Eigen::Vector2i(row, col),
							image[row*n_cols + col],
							camera_matrix_inverse);

	return vectors;
}

template <typename T> std::vector<T>
CartVectors2DepthImage(
		const std::vector<Eigen::Matrix<T, 3, 1> >& vectors,
		const int& n_rows, const int& n_cols,
		const Eigen::Matrix<T, 3, 3>& camera_matrix)
{
	std::vector<T> image(n_rows*n_cols, NAN);

	for(size_t i = 0; i < vectors.size(); i++)
	{
		if( isnan(vectors[i](0)) || isnan(vectors[i](1)) || isnan(vectors[i](2)) )
			continue;

		Eigen::Matrix<int, 2, 1> index = CartCoord2ImageIndex(vectors[i], camera_matrix);

		if(index(0) < 0 || index(0) >= n_rows || index(1) < 0 || index(1) >= n_cols)
		{
			std::cout << "ERROR: in CartVectors2DepthImage vector comes to lie outside of image " << std::endl;
			exit(-1);
		}

		image[index(0)*n_cols + index(1)] = vectors[i](2);
	}

	return image;
}



// numerically stable implementation of log(sum(exp(xi))) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
inline double log_sum_exp(double a, double b)
{
	if(a>b) return a + log(1+exp(b-a));
	else return b + log(1+exp(a-b));
}
inline double log_sum_exp(double a, double b, double c)
{
	if(a>b && a>c) return a + log(1+exp(b-a)+exp(c-a));
	if(b>c && b>a) return b + log(1+exp(a-b)+exp(c-b));
	else return c + log(1+exp(a-c)+exp(b-c));
}
inline double log_sum_exp(double a, double b, double c, double d)
{
	if(a>b && a>c && a>d) return a + log(1+exp(b-a)+exp(c-a)+exp(d-a));
	if(b>a && b>c && b>d) return b + log(1+exp(a-b)+exp(c-b)+exp(d-b));
	if(c>a && c>b && c>d) return c + log(1+exp(a-c)+exp(b-c)+exp(d-c));
	else return d + log(1+exp(a-d)+exp(b-d)+exp(c-d));
}

template<typename T> T log_sum_exp(std::vector<T> exponents)
{
	T max_exponent = -std::numeric_limits<double>::max();
	for(size_t i = 0; i < exponents.size(); i++)
		if(exponents[i] > max_exponent) max_exponent = exponents[i];

	for(size_t i = 0; i < exponents.size(); i++)
		exponents[i] -= max_exponent;

	T sum = 0;
	for(size_t i = 0; i < exponents.size(); i++)
		sum += exp(exponents[i]);

	return max_exponent + log(sum);
}


template <typename T> class LogSumExp
{
public:
	LogSumExp(){}
	~LogSumExp(){}

	void add_exponent(T exponent)
	{
		exponents_.push_back(exponent);
	}

	T Compute()
	{
		T max_exponent = -std::numeric_limits<T>::max();
		for(int i = 0; i < int(exponents_.size()); i++)
			if(exponents_[i] > max_exponent) max_exponent = exponents_[i];

		for(int i = 0; i < int(exponents_.size()); i++)
			exponents_[i] -= max_exponent;

		T sum = 0;
		for(int i = 0; i < int(exponents_.size()); i++)
			sum += exp(exponents_[i]);

		return max_exponent + log(sum);
	}
private:
	std::vector<T> exponents_;
};

// sampling class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class DiscreteSampler
{
public:
	template <typename T> DiscreteSampler(std::vector<T> log_likelihoods)
	{
		fibo_.seed(RANDOM_SEED);

		// compute the likelihoods and normalize them ------------------------------------------------------------------------------
		sorted_indices_ = hf::SortDescend(log_likelihoods);
		double max = log_likelihoods[sorted_indices_[0]];
		for(int i = 0; i < int(log_likelihoods.size()); i++)
			log_likelihoods[i] -= max;

		std::vector<double> likelihoods(log_likelihoods.size());
		double sum = 0;
		for(int i = 0; i < int(log_likelihoods.size()); i++)
		{
			likelihoods[i] = exp(log_likelihoods[i]);
			sum += likelihoods[i];
		}
		for(int i = 0; i < int(likelihoods.size()); i++)
			likelihoods[i] /= sum;

		// compute the cumulative likelihood ------------------------------------------------------------------
		cum_likelihoods_.resize(log_likelihoods.size());
		cum_likelihoods_[0] = likelihoods[sorted_indices_[0]];
		for(int i = 1; i < int(log_likelihoods.size()); i++)
			cum_likelihoods_[i] = cum_likelihoods_[i-1] + likelihoods[sorted_indices_[i]];
	}

	~DiscreteSampler() {}

    int Sample()
	{
		double rand_number = fibo_();
		int sample_index = 0;
		while(rand_number > cum_likelihoods_[sample_index]) sample_index++;

		return sorted_indices_[sample_index];
	}

private:
	boost::lagged_fibonacci607 fibo_;
	std::vector<int> sorted_indices_;
	std::vector<double> cum_likelihoods_;
};
























// todo this stuff should be removed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <typename T> void
Vector2TranslAndQuat(const std::vector<T> &v, Eigen::Matrix<T, 3, 1> &t, Eigen::Quaternion<T> &q)
{
	Eigen::Quaternion<T> quat();
	q.w() = v[0]; q.x() = v[1]; q.y() = v[2]; q.z() = v[3];
	t(0) = v[4]; t(1) = v[5]; t(2) = v[6];
}

template <typename T> void
TranslAndQuat2Vector(const Eigen::Quaternion<T> &q, const Eigen::Matrix<T, 3, 1> &t, std::vector<T> &v)
{
	v.resize(7);

	v[0] = q.w();  v[1] = q.x(); v[2] = q.y(); v[3] = q.z();
	v[4] = t(0); v[5] = t(1); v[6] = t(2);
}

template <typename T> void
Vector2TranslAndRot(const std::vector<T> &v, Eigen::Matrix<T, 3, 3> &R, Eigen::Matrix<T, 3, 1> &t)
{
	Eigen::Quaternion<T> q;

	Vector2QuatAndTransl(v, q, t);
	R = q;
}

template <typename T> void
TranslAndRot2Vector(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &t, std::vector<T> &v)
{
	const Eigen::Quaternion<T> q(R);

	QuatAndTransl2Vector(q, t, v);
}

inline Eigen::VectorXd OldState2NewState(const std::vector<float>& old_state)
{
	Eigen::VectorXd new_state(old_state.size() * 2 - 1); // the new state also contains the velocity
	new_state(0) = old_state[4];
	new_state(1) = old_state[5];
	new_state(2) = old_state[6];

	new_state(3) = old_state[1];
	new_state(4) = old_state[2];
	new_state(5) = old_state[3];
	new_state(6) = old_state[0];

//	new_state.topRows(3) +=  Eigen::Quaterniond(new_state.middleRows<4>(3))._transformVector(center);
	new_state.middleRows(7, 6) = Eigen::VectorXd::Zero(6);

	// we fill in the joint angles
	for(size_t i = 7; i < old_state.size(); i++)
		new_state(i + 6) = old_state[i];
	// we fill in the joint velocities
	for(size_t i = old_state.size() + 6; i < size_t(new_state.rows()); i++)
		new_state(i) = 0;

	return new_state;
}

inline std::vector<float> NewState2OldState(const Eigen::VectorXd& new_state)
{
	std::vector<float> old_state((new_state.rows()+1)/2);

	old_state[4] = new_state(0);
	old_state[5] = new_state(1);
	old_state[6] = new_state(2);

	old_state[1] = new_state(3);
	old_state[2] = new_state(4);
	old_state[3] = new_state(5);
	old_state[0] = new_state(6);

	for(size_t i = 7; i < old_state.size(); i++)
		old_state[i] = new_state(i+6);

	return old_state;
}

template <typename T> void
Vector2QuatAndTransl(const std::vector<T> &v, Eigen::Quaternion<T> &q, Eigen::Matrix<T, 3, 1> &t)
{
	q.w() = v[0]; q.x() = v[1]; q.y() = v[2]; q.z() = v[3];
	t(0) = v[4]; t(1) = v[5]; t(2) = v[6];
}

template <typename T> void
QuatAndTransl2Vector(const Eigen::Quaternion<T> &q, const Eigen::Matrix<T, 3, 1> &t, std::vector<T> &v)
{
	v.resize(7);

	v[0] = q.w();  v[1] = q.x(); v[2] = q.y(); v[3] = q.z();
	v[4] = t(0); v[5] = t(1); v[6] = t(2);
}

template <typename T> void
Vector2RotAndTransl(const std::vector<T> &v, Eigen::Matrix<T, 3, 3> &R, Eigen::Matrix<T, 3, 1> &t)
{
	Eigen::Quaternion<T> q;

	Vector2QuatAndTransl(v, q, t);
	R = q;
}

template <typename T> void
RotAndTransl2Vector(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &t, std::vector<T> &v)
{
	const Eigen::Quaternion<T> q(R);

	QuatAndTransl2Vector(q, t, v);
}

template <typename T> void
Vector2Hom(const std::vector<T> &v, Eigen::Matrix<T, 4, 4> &H)
{
	Eigen::Matrix<T, 3, 3> R;
	Eigen::Matrix<T, 3, 1> t;
	Vector2RotAndTransl(v, R, t);

	H.topLeftCorner(3,3) = R;
	H.topRightCorner(3,1) = t;
	H.row(3) = Eigen::Matrix<T, 1, 4> (0,0,0,1);
}

template <typename T> void
Hom2Vector(const Eigen::Matrix<T, 4, 4> &H, std::vector<T> &v)
{
	const Eigen::Matrix<T, 3, 3> R(H.topLeftCorner(3,3));
	const Eigen::Matrix<T, 3, 1> t(H.topRightCorner(3,1));

	RotAndTransl2Vector(R, t, v);
}

template <typename T> void
Vector2Affine(const std::vector<T> &v, Eigen::Transform<T,3,Eigen::Affine> &A)
{
	Eigen::Matrix<T, 4, 4> H;
	Vector2Hom(v, H);
	A = H;
}

template <typename T> void
Affine2Vector(const Eigen::Transform<T,3,Eigen::Affine> &A, std::vector<T> &v)
{
	Eigen::Matrix<T, 4, 4> H;
	H = A.matrix();
	Hom2Vector(H, v);
}




}

#endif
