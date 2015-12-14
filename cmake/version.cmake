##
## This is part of the Bayesian Object Tracking (bot),
## (https://github.com/bayesian-object-tracking)
##
## Copyright (c) 2015 Max Planck Society,
## 				 Autonomous Motion Department,
## 			     Institute for Intelligent Systems
##
## This Source Code Form is subject to the terms of the GNU General Public
## License License (GNU GPL). A copy of the license can be found in the LICENSE
## file distributed with this source code.
##

##
## Date November 2015
## Author Jan Issac (jan.issac@gmail.com)
##

find_package(Git)

# todo: fix old git version issue
execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE PROJECT_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
