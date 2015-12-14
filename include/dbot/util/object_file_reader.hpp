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

#pragma once

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <list>

#include <Eigen/Core>

namespace dbot
{

class ObjectFileReader
{
public:
	ObjectFileReader();
	~ObjectFileReader(){}

	void set_filename(std::string filename);
	void Read();
	void Process(float max_side_length);

    std::shared_ptr<std::vector<Eigen::Vector3d> > get_vertices();
    std::shared_ptr<std::vector<std::vector<int> > > get_indices();


    std::shared_ptr<std::vector<Eigen::Vector3d> > get_centers();
    std::shared_ptr<std::vector<float> > get_areas();

private:
	std::string filename_;
    std::shared_ptr<std::vector<Eigen::Vector3d> > vertices_;
    std::shared_ptr<std::vector<std::vector<int> > > indices_;

    std::shared_ptr<std::vector<Eigen::Vector3d> > centers_;
    std::shared_ptr<std::vector<float> > areas_;
};

}
