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

#ifndef UTILS_MACROS_HPP
#define UTILS_MACROS_HPP

#include <sys/time.h>
#include <iostream>
#include <time.h>
#include <boost/current_function.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace macros
{

#define GET_TIME(time) {struct timeval profiling_time; gettimeofday(&profiling_time, NULL);\
    time = (profiling_time.tv_sec * 1000000u + profiling_time.tv_usec) /1000000.;}
#ifdef PROFILING_ON
    #define PRINT(object) std::cout << object;

	#define INIT_PROFILING struct timeval profiling_start_time, profiling_end_time; gettimeofday(&profiling_start_time, NULL);
	#define RESET gettimeofday(&profiling_start_time, NULL);
	#define MEASURE(text) 	gettimeofday(&profiling_end_time, NULL); std::cout << "time for " << text << " " \
	<< ((profiling_end_time.tv_sec - profiling_start_time.tv_sec) * 1000000u + profiling_end_time.tv_usec - profiling_start_time.tv_usec) /1000000. \
    << " s" << std::endl; gettimeofday(&profiling_start_time, NULL);
#else
    #define PRINT(object)
	#define INIT_PROFILING
	#define RESET
	#define MEASURE(text)
#endif

#define PRINT_VARIABLE(variable) std::cout << "--------------------------------------" << std::endl << \
#variable  << ": " << std::endl << variable << std::endl \
<< "--------------------------------------" << std::endl;

#define TO_BE_IMPLEMENTED  std::cout << "The function " << BOOST_CURRENT_FUNCTION << \
	" has not been implemented yet." << std::endl; exit(-1);

#ifndef USE_UNTESTED_FUNCTIONS
#define TO_BE_TESTED  std::cout << "The function " << BOOST_CURRENT_FUNCTION << \
	" has not been tested yet." << std::endl; exit(-1);
#else
#define TO_BE_TESTED
#endif


#define RANDOM_SEED (unsigned)time(0)//1



#define SF_DISABLE_IF_DYNAMIC_SIZE(VariableType) \
    if (macros::Invalidate<VariableType::SizeAtCompileTime != Eigen::Dynamic> \
        ::YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_DISTRIBUTION) { }

#define SF_DISABLE_IF_FIXED_SIZE(VariableType) \
    if (macros::Invalidate<VariableType::SizeAtCompileTime == Eigen::Dynamic> \
        ::YOU_CALLED_A_DYNAMIC_SIZE_METHOD_ON_A_FIXED_SIZE_DISTRIBUTION) { }

template <bool Condition> struct Invalidate { };

template <>
struct Invalidate<true>
{
    enum
    {
        YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_DISTRIBUTION,
        YOU_CALLED_A_DYNAMIC_SIZE_METHOD_ON_A_FIXED_SIZE_DISTRIBUTION
    };
};


/**
 * This variadic macro performs a compile time derivation assertion. The first
 * argument is the derived type which is being tested whether it implements a
 * base type. The base time is given as the second argument list.
 *
 * __VA_ARGS__ was used as a second parameter to enable passing template
 * specialization to the macro.
 *
 * Note: The macro requires <em>derived_type</em> to be a single worded type. In
 *       case of a template specialization, please use a typedef.
 */
#define SF_REQUIRE_INTERFACE(derived_type, ...)\
    BOOST_STATIC_ASSERT_MSG(( \
            boost::is_base_of<__VA_ARGS__, derived_type>::value), \
            #derived_type " must implement " #__VA_ARGS__ " interface.");
}
#endif
