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

#ifndef FAST_FILTERING_UTILS_PROFILING_HPP
#define FAST_FILTERING_UTILS_PROFILING_HPP

#include <sys/time.h>
#include <iostream>


// profiling macros
#define GET_TIME(time) {struct timeval profiling_time; gettimeofday(&profiling_time, NULL);\
    time = (profiling_time.tv_sec * 1000000u + profiling_time.tv_usec) /1000000.;}
#ifdef PROFILING_ON
    #define PRINT(object) std::cout << object;

    #define INIT_PROFILING struct timeval profiling_start_time, profiling_end_time;\
                                    gettimeofday(&profiling_start_time, NULL);
    #define RESET gettimeofday(&profiling_start_time, NULL);
    #define MEASURE(text)\
            gettimeofday(&profiling_end_time, NULL);\
            std::cout << "time for " << text << " " \
              << std::setprecision(9) << std::fixed\
              << ((profiling_end_time.tv_sec - profiling_start_time.tv_sec) * 1000000u\
                 + profiling_end_time.tv_usec - profiling_start_time.tv_usec) /1000000. \
              << " s" << std::endl; gettimeofday(&profiling_start_time, NULL);
#else
    #define PRINT(object)
    #define INIT_PROFILING
    #define RESET
    #define MEASURE(text)
#endif


#endif
