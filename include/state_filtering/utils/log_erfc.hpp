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

#ifndef STATE_FILTERING_UTILS_LOG_ERF_HPP_
#define STATE_FILTERING_UTILS_LOG_ERF_HPP_

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/policies/error_handling.hpp>

#include <iostream>

// like the erfc function from boost except that it returns the log, therefore for large z we do not encounter num problems
double log_erfc(double z)
{
   double result;
   double z_abs = z > 0 ? z : -z;

   if(z_abs < 0.5)  // We're going to calculate erf
   {
      if(z_abs < 1e-10)
      {
         if(z_abs == 0) { result = double(0); }
         else { result = static_cast<double>(z_abs * 1.125f + z_abs * 0.003379167095512573896158903121545171688L); }
      }
      else
      {
         // Maximum Deviation Found:                     1.561e-17
         // Expected Error Term:                         1.561e-17
         // Maximum Relative Change in Control Points:   1.155e-04
         // Max Error found at double precision =        2.961182e-17
         static const double Y = 1.044948577880859375f;
         static const double P[] = {
            0.0834305892146531832907L,
            -0.338165134459360935041L,
            -0.0509990735146777432841L,
            -0.00772758345802133288487L,
            -0.000322780120964605683831L,
         };
         static const double Q[] = {
            1L,
            0.455004033050794024546L,
            0.0875222600142252549554L,
            0.00858571925074406212772L,
            0.000370900071787748000569L,
         };
         double zz = z_abs * z_abs;
         result = z_abs * (Y + boost::math::tools::evaluate_polynomial(P, zz) / boost::math::tools::evaluate_polynomial(Q, zz));
      }

      if(z < 0)
    	  result = log(1+result);
      else
    	  result = log(1-result);
   }

   else // We'll be calculating erfc
   {
      if(z_abs < 1.5f)
      {
         // Maximum Deviation Found:                     3.702e-17
         // Expected Error Term:                         3.702e-17
         // Maximum Relative Change in Control Points:   2.845e-04
         // Max Error found at double precision =        4.841816e-17
         static const double Y = 0.405935764312744140625f;
         static const double P[] = {
            -0.098090592216281240205L,
            0.178114665841120341155L,
            0.191003695796775433986L,
            0.0888900368967884466578L,
            0.0195049001251218801359L,
            0.00180424538297014223957L,
         };
         static const double Q[] = {
            1L,
            1.84759070983002217845L,
            1.42628004845511324508L,
            0.578052804889902404909L,
            0.12385097467900864233L,
            0.0113385233577001411017L,
            0.337511472483094676155e-5L,
         };
         result = Y + boost::math::tools::evaluate_polynomial(P, z_abs - 0.5) / boost::math::tools::evaluate_polynomial(Q, z_abs - 0.5);
//         result *= exp(-z_abs * z_abs) / z_abs;
      }
      else if(z_abs < 2.5f)
      {
         // Max Error found at double precision =        6.599585e-18
         // Maximum Deviation Found:                     3.909e-18
         // Expected Error Term:                         3.909e-18
         // Maximum Relative Change in Control Points:   9.886e-05
         static const double Y = 0.50672817230224609375f;
         static const double P[] = {
            -0.0243500476207698441272L,
            0.0386540375035707201728L,
            0.04394818964209516296L,
            0.0175679436311802092299L,
            0.00323962406290842133584L,
            0.000235839115596880717416L,
         };
         static const double Q[] = {
            1L,
            1.53991494948552447182L,
            0.982403709157920235114L,
            0.325732924782444448493L,
            0.0563921837420478160373L,
            0.00410369723978904575884L,
         };
         result = Y + boost::math::tools::evaluate_polynomial(P, z_abs - 1.5) / boost::math::tools::evaluate_polynomial(Q, z_abs - 1.5);
//         result *= exp(-z_abs * z_abs) / z_abs;
      }
      else if(z_abs < 4.5f)
      {
         // Maximum Deviation Found:                     1.512e-17
         // Expected Error Term:                         1.512e-17
         // Maximum Relative Change in Control Points:   2.222e-04
         // Max Error found at double precision =        2.062515e-17
         static const double Y = 0.5405750274658203125f;
         static const double P[] = {
            0.00295276716530971662634L,
            0.0137384425896355332126L,
            0.00840807615555585383007L,
            0.00212825620914618649141L,
            0.000250269961544794627958L,
            0.113212406648847561139e-4L,
         };
         static const double Q[] = {
            1L,
            1.04217814166938418171L,
            0.442597659481563127003L,
            0.0958492726301061423444L,
            0.0105982906484876531489L,
            0.000479411269521714493907L,
         };
         result = Y + boost::math::tools::evaluate_polynomial(P, z_abs - 3.5) / boost::math::tools::evaluate_polynomial(Q, z_abs - 3.5);
//         result *= exp(-z_abs * z_abs) / z_abs;
      }
      else
      {
         // Max Error found at double precision =        2.997958e-17
         // Maximum Deviation Found:                     2.860e-17
         // Expected Error Term:                         2.859e-17
         // Maximum Relative Change in Control Points:   1.357e-05
         static const double Y = 0.5579090118408203125f;
         static const double P[] = {
            0.00628057170626964891937L,
            0.0175389834052493308818L,
            -0.212652252872804219852L,
            -0.687717681153649930619L,
            -2.5518551727311523996L,
            -3.22729451764143718517L,
            -2.8175401114513378771L,
         };
         static const double Q[] = {
            1L,
            2.79257750980575282228L,
            11.0567237927800161565L,
            15.930646027911794143L,
            22.9367376522880577224L,
            13.5064170191802889145L,
            5.48409182238641741584L,
         };
         result = Y + boost::math::tools::evaluate_polynomial(P, 1 / z_abs) / boost::math::tools::evaluate_polynomial(Q, 1 / z_abs);
//         result *= exp(-z_abs * z_abs) / z_abs;
      }

      if(z<0)
      {
    	  result *= exp(-z_abs * z_abs) / z_abs;
    	  result = log(2-result);
      }
      else
          result = -z_abs * z_abs - log(z_abs) + log(result);


   }
   return result;
}

#endif

