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

############################
# Info gen. functions      #
############################
# I'm sure there is a better way of doing this...
execute_process(
    COMMAND bash -c
        "v=`ps -o stat= -p $PPID`
         [[ $v == *+* ]] || [[ $v == *s* ]] && echo YES || echo NO"
    OUTPUT_VARIABLE ISATTY
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(ISATTY STREQUAL "YES")
    string(ASCII 27 Esc)
    set(COLOR_BORDER "${Esc}[35m")
    set(COLOR_HEADER "${Esc}[34m")
    set(COLOR_BOLD   "${Esc}[1m")
    set(COLOR_CLEAR  "${Esc}[m")
else(ISATTY STREQUAL "YES")
    set(COLOR_BORDER "")
    set(COLOR_HEADER "")
    set(COLOR_BOLD   "")
    set(COLOR_CLEAR  "")
endif(ISATTY STREQUAL "YES")

function(info_begin)
  message(STATUS "${COLOR_BORDER}=================================================${COLOR_CLEAR}")
endfunction(info_begin)

function(info_end)
  message(STATUS "${COLOR_BORDER}=====${COLOR_CLEAR}")
endfunction(info_end)

function(info_project project_name project_version)
  message(STATUS "${COLOR_BORDER}== ${COLOR_CLEAR} ${COLOR_HEADER}${project_name}${COLOR_CLEAR}")
  message(STATUS "${COLOR_BORDER}== ${COLOR_CLEAR} Version: ${COLOR_BOLD}${project_version}${COLOR_CLEAR}")
endfunction(info_project)

function(info_header list_header)
  message(STATUS "${COLOR_BORDER}== ${COLOR_CLEAR} ")
  message(STATUS "${COLOR_BORDER}== ${COLOR_CLEAR} ${COLOR_BOLD}${list_header}")
endfunction(info_header)

function(info_item item_name item_value)
  message(STATUS "${COLOR_BORDER}== ${COLOR_CLEAR} - ${item_name}:${COLOR_BOLD} ${item_value}")
endfunction(info_item)
