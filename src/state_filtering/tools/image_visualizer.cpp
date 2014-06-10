/*************************************************************************
This software allows for filtering in high-dimensional measurement and
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


#include <state_filtering/tools/image_visualizer.hpp>
#include <state_filtering/tools/helper_functions.hpp>

#include <cv.h>
#include <highgui.h>
#include <limits>


namespace vis
{
using namespace Eigen;
using namespace std;


ImageVisualizer::ImageVisualizer(
		const int &n_rows, const int &n_cols)
		:n_rows_(n_rows), n_cols_(n_cols)
{
    image_ = cvCreateImage(cvSize(n_cols_, n_rows_),IPL_DEPTH_8U,3);

	for(int row = 0; row < n_rows_; row++)
		for(int col = 0; col < n_cols_; col++)
		{
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 0] = 0;
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 1] = 0;
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 2] = 0;
		}
}

ImageVisualizer::~ImageVisualizer() {cvReleaseImage((IplImage**)(&image_));}


void ImageVisualizer::set_image(
        const Eigen::MatrixXd &image,
        const float &min_value,
        const float &max_value,
         const bool &invert_image)
{
    std::vector<float> std_image(image.rows()*image.cols());
    for(size_t row = 0; row < image.rows(); row++)
        for(size_t col = 0; col < image.cols(); col++)
        {
            std_image[row*image.cols()+col] = image(row, col);
        }

    set_image(std_image, min_value, max_value, invert_image);
}



void ImageVisualizer::set_image(
        const std::vector<float> &image,
        const float &min_value,
        const float &max_value,
         const bool &invert_image)
{
	std::vector<float> display_image(image.size());
    for(size_t i = 0; i < image.size(); i++)
    {
        if(invert_image)
            display_image[i] = 1/image[i];
        else
            display_image[i] = image[i];
    }

	// find min and max of image -------------------------------------------------------------------------------------------------
    float max, min;

    if(min_value == 0 && max_value == 0)
    {
        max = -numeric_limits<float>::max();
        min = numeric_limits<float>::max();
        for(size_t i = 0; i < image.size(); i++)
        {
            min = display_image[i] < min ? display_image[i] : min;
            max = display_image[i] > max ? display_image[i] : max;
        }
    }
    else
    {
        max = max_value;
        min = min_value;
    }

	for(size_t i = 0; i < display_image.size(); i++)
		display_image[i] = (display_image[i] - min) / (max-min) * 255.;


	// fill values from vector into image --------------------------------------------------------------------------------------------
	for(int row = 0; row < n_rows_; row++)
		for(int col = 0; col < n_cols_; col++)
		{
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 0] = display_image[row*n_cols_ + col];
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 1] = display_image[row*n_cols_ + col];
            ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 2] = display_image[row*n_cols_ + col];
		}
}

void ImageVisualizer::add_points(
		const std::vector<Eigen::Vector3f> &points,
		const Eigen::Matrix3f &camera_matrix,
		const Eigen::Matrix3f &R,
		const Eigen::Vector3f &t,
		const std::vector<float> &colors)
{
	std::vector<int> point_indices(points.size());

	std::vector<float> new_colors;
	if(colors.size() != 0)
		new_colors = colors;
	else
		new_colors.resize(points.size());

	for(size_t i = 0; i < points.size(); i++)
	{
		Vector3f point = R * points[i] + t;
		int row, col;
		Cart2Index(point, camera_matrix, row, col);
		point_indices[i] = row*n_cols_ + col;

		if(colors.size() == 0)
			new_colors[i] = point(2);
	}

	add_points(point_indices, new_colors);
}


void ImageVisualizer::add_points(
        const Eigen::Matrix<Eigen::Vector3d, -1, -1> &points,
        const Eigen::Matrix3d &camera_matrix,
        const Eigen::Matrix3d &R,
        const Eigen::Vector3d &t,
        const std::vector<float> &colors)
{
    std::vector<int> point_indices(points.cols() * points.rows());

    std::vector<float> new_colors;
    if(colors.size() != 0)
        new_colors = colors;
    else
        new_colors.resize(points.cols() * points.rows());

    for(size_t row = 0; row < points.rows(); row++)
        for(size_t col = 0; col < points.cols(); col++)
        {
            Vector3d point = R * points(row, col) + t;
            Vector2i index = hf::CartCoord2ImageIndex(point, camera_matrix);

            if(!(index(0) > 479 || index(1) > 639 || index(0) < 0 || index(1) < 0))
            {
                point_indices[row * points.cols() + col] = index(0)*n_cols_ + index(1);

                if(colors.size() == 0)
                    new_colors[row * points.cols() + col] = point(2);
            }
        }

    add_points(point_indices, new_colors);
}



void ImageVisualizer::add_points(
		const std::vector<int> &point_indices,
		const std::vector<float> &colors)
{
	// if no color has been given we set it to some value -----------------------------
	vector<float> new_colors;

	if(colors.size() != 0)
		new_colors = colors;
	else
		new_colors = vector<float>(point_indices.size(), 1);

	// renormalize colors -----------------------------
	float max = -numeric_limits<float>::max();
	float min = numeric_limits<float>::max();
	for(int i = 0; i < int(colors.size()); i++)
	{
		min = colors[i] < min ? colors[i] : min;
		max = colors[i] > max ? colors[i] : max;
	}
	if(min == max) min = 0;

	for(int i = 0; i < int(point_indices.size()); i++)
	{
		int row = point_indices[i]/n_cols_;
		int col = point_indices[i]%n_cols_;

        ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 2] = (new_colors[i]-min)/(max-min) * 255.;
        ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 1] = 0;
        ((uchar *)(((IplImage*)(image_))->imageData + row*((IplImage*)(image_))->widthStep))[col*3 + 0] = (1 - (new_colors[i]-min)/(max-min)) * 255.;
	}
}

char ImageVisualizer::show_image(
		const std::string &window_name,
		const int &window_width, const int &window_height,
		const int &delay) const
{
	cvNamedWindow(window_name.c_str(), CV_WINDOW_NORMAL);
    cvShowImage(window_name.c_str(), ((IplImage*)(image_)));
	cvResizeWindow(window_name.c_str(), window_width, window_height);
	return cvWaitKey(delay);
}

void ImageVisualizer::Cart2Index(
		const Eigen::Vector3f &cart,
		const Eigen::Matrix3f &camera_matrix,
		int &row, int &col) const
{
	Vector3f index_temp = camera_matrix * cart/cart(2);

	row = floor(float(index_temp(1)+0.5f));
	col = floor(float(index_temp(0)+0.5f));
}



}

