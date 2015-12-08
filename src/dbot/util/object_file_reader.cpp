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


#include <dbot/util/object_file_reader.hpp>

#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;
using namespace Eigen;

namespace dbot
{

ObjectFileReader::ObjectFileReader()
: vertices_(new vector<Vector3d>)
, indices_(new vector<vector<int> >)
, centers_(new vector<Vector3d>)
, areas_(new vector<float>)
{
}


void ObjectFileReader::set_filename(string filename) {filename_ = filename;}


void ObjectFileReader::Read()
{
	indices_->clear();
	vertices_->clear();

	ifstream file;
	file.open(filename_.c_str());

	if(!file.is_open())
	{
		cout << "ERROR: the wavefront file " << filename_ << " could not be opened." << endl;
		exit(-1);
	}

	string line;

	float x_min = std::numeric_limits<float>::max();
	float x_max = -std::numeric_limits<float>::max();

	while(!file.eof())
	{
		string data_type;
		getline(file, line, '\n');

		stringstream line_stream(line);
		line_stream >> data_type;

		if(data_type == "v")
		{
			Vector3d point;

			line_stream >> point(0);
			line_stream >> point(1);
			line_stream >> point(2);
			vertices_->push_back(point);
		}
		else if(data_type == "f")
		{
			vector<int> triangle(3);

			char trash[100];

			line_stream >> triangle[0];
			line_stream.getline(trash, 100, ' ');
			line_stream >> triangle[1];
			line_stream.getline(trash, 100, ' ');
			line_stream >> triangle[2];

			// substract one because indices in object files start with 1 and we start with 0
			triangle[0]--; triangle[1]--; triangle[2]--;
			indices_->push_back(triangle);
		}
	}
	file.close();

	// todo this is a bit hacky: we check if extension in x is larger than 10, if it is we assume that the unit is mm
	if(x_max - x_min > 10)
	{
		for(unsigned int i = 0; i < vertices_->size(); i++)
		{
			(*vertices_)[i] = (*vertices_)[i]/1000.;
		}
	}
}


void ObjectFileReader::Process(float max_side_length) // todo: has to be tested since we switched from Vector3i to vector<int>
{
	// subdivide triangles until no triangle has a side longer than max_side_length ----------------------------------------
	vector<vector<int> >::iterator triangle_indices = indices_->begin();

	int index = 0;
	while(triangle_indices != indices_->end())
	{
		vector<Vector3d> triangle;
		for(unsigned int corner_index = 0; corner_index < 3; corner_index++)
			triangle.push_back((*vertices_)[(*triangle_indices)[corner_index]]);

		// find longest side of triangle ----------------------------------------------------------
		float AB = -1.;
		int A = -1 , B = -1 , C = -1;
		for(unsigned int i = 0; i < 3; i++)
		{
			float length = (triangle[i] - triangle[(i+1)%3]).norm();

			if( length > AB )
			{
				AB = length;
				A = i;
				B = (i+1)%3;
				C = (i+2)%3;
			}
		}

		assert( A!= -1 && B!=-1 && C!=-1 && AB!=-1);
		// if the longest side is too long we split the triangle -----------------------------------
		if(AB > max_side_length)
		{
			vertices_->push_back( (triangle[A] + triangle[B]) / 2.);
			float center_AB = vertices_->size()-1;

			vector<int> triangle_indices_A(3);
			triangle_indices_A[0] = (*triangle_indices)[A];
			triangle_indices_A[1] = center_AB;
			triangle_indices_A[2] = (*triangle_indices)[C];


			vector<int> triangle_indices_B(3);
			triangle_indices_B[0] = (*triangle_indices)[B];
			triangle_indices_B[1] = center_AB;
			triangle_indices_B[2] = (*triangle_indices)[C];

			indices_->push_back(triangle_indices_A);
			indices_->push_back(triangle_indices_B);
			indices_->erase(triangle_indices++);
		}
		else {triangle_indices++; index++;}
	}

	// now we compute the area and the center of each triangle ---------------------------------------------------------------
	for(vector<vector<int> >::iterator indices = indices_->begin(); indices != indices_->end(); indices++)
	{
		Vector3d A,B,C;
		float a,b,c, s;

		A = (*vertices_)[(*indices)[0]];
		B = (*vertices_)[(*indices)[1]];
		C = (*vertices_)[(*indices)[2]];
		a = (B-A).norm();
		b = (C-B).norm();
		c = (A-C).norm();
		s = (a+b+c)/2.;

		centers_->push_back(1./3.*A + 1./3.*B + 1./3.*C);
		areas_->push_back(sqrt(s*(s-a)*(s-b)*(s-c)));
	}
}

std::shared_ptr<std::vector<Eigen::Vector3d> > ObjectFileReader::get_vertices() {return vertices_;}
std::shared_ptr<std::vector<std::vector<int> > > ObjectFileReader::get_indices() {return indices_;}

std::shared_ptr<std::vector<Eigen::Vector3d> > ObjectFileReader::get_centers() {return centers_;}
std::shared_ptr<std::vector<float> > ObjectFileReader::get_areas() {return areas_;}

}
