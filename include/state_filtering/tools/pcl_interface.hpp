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


#ifndef POSE_FILTERING_PCL_INTERFACE_HPP_
#define POSE_FILTERING_PCL_INTERFACE_HPP_

#include <Eigen/Core>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/PointIndices.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/segment_differences.h>

#include <boost/assert.hpp>

#include <state_filtering/tools/helper_functions.hpp>
// #include <state_filtering/tools/cloud_visualizer.hpp>
#include <state_filtering/tools/macros.hpp>
#include <state_filtering/system_states/full_rigid_body_system.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>

namespace pi
{
template<typename PointT, typename MatrixT>
void Pcl2Eigen(
		const pcl::PointCloud<PointT> &point_cloud,
		std::vector<Eigen::Matrix<MatrixT, 3, 1> > &vector,
		size_t &n_rows, size_t &n_cols)
{
	n_cols = point_cloud.width;
	n_rows = point_cloud.height;
	vector.resize (n_rows * n_cols);

	for (size_t i = 0; i < point_cloud.points.size (); ++i)
	{
		vector[i](0) = point_cloud.points[i].x;
		vector[i](1) = point_cloud.points[i].y;
		vector[i](2) = point_cloud.points[i].z;
	}
}


template<typename T>
void Ros2Std(
        const sensor_msgs::PointCloud2& ros_cloud,
        const size_t& n_downsampling,
        vector<T>& observations,
        size_t& n_rows,
        size_t& n_cols)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg (ros_cloud, *point_cloud);
    n_rows = point_cloud->height/n_downsampling; n_cols = point_cloud->width/n_downsampling;
    observations.resize(n_rows*n_cols);
    for(size_t row = 0; row < n_rows; row++)
        for(size_t col = 0; col < n_cols; col++)
            observations[row*n_cols + col] = point_cloud->at(col*n_downsampling, row*n_downsampling).z;
}





template<typename PointT, typename MatrixT>
std::vector<Eigen::Matrix<MatrixT, 3, 1> > Pcl2Eigen(const pcl::PointCloud<PointT> &point_cloud)
{
	std::vector<Eigen::Matrix<MatrixT, 3, 1> >  vector;
	size_t n_rows, n_cols;
	Pcl2Eigen(point_cloud, vector, n_rows, n_cols);

	return vector;
}


template<typename PointT, typename MatrixT>
void Eigen2Pcl(
		const std::vector<Eigen::Matrix<MatrixT, 3, 1> > &vector,
		const size_t &n_rows, const size_t &n_cols,
		pcl::PointCloud<PointT> &point_cloud)
{
	point_cloud.width    = n_cols;
	point_cloud.height   = n_rows;
	point_cloud.is_dense = (n_cols == 1);
	point_cloud.points.resize (n_rows * n_cols);

	for (size_t i = 0; i < point_cloud.points.size (); ++i)
	{
		point_cloud.points[i].x = vector[i](0);
		point_cloud.points[i].y = vector[i](1);
		point_cloud.points[i].z = vector[i](2);
	}
}
template<typename PointT, typename MatrixT>
pcl::PointCloud<PointT> Eigen2Pcl(
		const std::vector<Eigen::Matrix<MatrixT, 3, 1> > &vector,
		const size_t &n_rows = 0, const size_t &n_cols = 1)
		{
	size_t true_n_rows;
	if(n_rows == 0)
		true_n_rows = vector.size();
	else
		true_n_rows = n_rows;

	pcl::PointCloud<PointT> point_cloud;
	Eigen2Pcl(vector, true_n_rows, n_cols, point_cloud);
	return point_cloud;
		}


template<typename PointT, typename MatrixT>
void PointsOnPlane(
		const pcl::PointCloud<PointT>& input_cloud,
		pcl::PointCloud<PointT>& output_cloud,
		Eigen::Matrix<MatrixT, 4, 1>& table_plane,
		const bool& keep_organized = false,
		const float& z_min = 0.3,  const float& z_max = 2.0,
		const float& y_min = -1.0, const float& y_max = 1.0,
		const float& x_min = -1.0, const float& x_max = 1.0,
		const float& min_table_dist = 0.01, const float& max_table_dist = 0.4,
		const float& grid_size = 0.02)
{
	bool is_empty = true;
	for(size_t i = 0; i < input_cloud.points.size(); i++)
		if(isfinite(input_cloud.points[i].getVector3fMap().norm()))
		{
			is_empty = false;
			break;
		}
	if(is_empty)
	{
		output_cloud = input_cloud;
		table_plane = Eigen::Matrix<MatrixT, 4, 1>::Zero();
		return;
	}

	boost::shared_ptr<pcl::PointCloud<PointT> > temp_cloud(new pcl::PointCloud<PointT>);
	*temp_cloud = input_cloud;

	boost::shared_ptr<pcl::PointCloud<PointT> > input_cloud_ptr(new pcl::PointCloud<PointT>);
	*input_cloud_ptr = input_cloud;

	pcl::PointCloud<PointT> downsampled;
	pcl::VoxelGrid<PointT> grid_;
	grid_.setLeafSize (grid_size, grid_size, grid_size);

	grid_.setFilterFieldName ("z");
	grid_.setFilterLimits (z_min, z_max);
	grid_.setInputCloud (temp_cloud);
	grid_.filter (downsampled);
	*temp_cloud = downsampled;

	grid_.setFilterFieldName ("y");
	grid_.setFilterLimits (y_min, y_max);
	grid_.setInputCloud (temp_cloud);
	grid_.filter (downsampled);
	*temp_cloud = downsampled;

	grid_.setFilterFieldName ("x");
	grid_.setFilterLimits (x_min, x_max);
	grid_.setInputCloud (temp_cloud);
	grid_.filter (downsampled);
	*temp_cloud = downsampled;


	// Estimate normals ----------------------------------------
	boost::shared_ptr<pcl::search::KdTree<PointT> > normals_tree_ (new pcl::search::KdTree<PointT> ());
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<PointT, pcl::Normal> n3d_;
	n3d_.setKSearch (10);
	n3d_.setSearchMethod (normals_tree_);
	n3d_.setInputCloud (temp_cloud);
	n3d_.compute (*normals);

	// Perform planar segmentation -------------------------------------------------------------------
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
	pcl::PointIndices::Ptr table_inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr table_coefficients(new pcl::ModelCoefficients);

	seg.setDistanceThreshold (0.01);//0.005
	seg.setMaxIterations (10000);
	seg.setNormalDistanceWeight (0.1);
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setProbability (0.99);
	seg.setInputCloud (temp_cloud);
	seg.setInputNormals (normals);
	seg.segment (*table_inliers, *table_coefficients);
	table_plane(0) = table_coefficients->values[0];
	table_plane(1) = table_coefficients->values[1];
	table_plane(2) = table_coefficients->values[2];
	table_plane(3) = table_coefficients->values[3];

	// Project the table inliers on the table -----------------------------------------
	boost::shared_ptr<pcl::PointCloud<PointT> > table_projected(new pcl::PointCloud<PointT>);
	pcl::ProjectInliers<PointT> proj_;
	proj_.setModelType (pcl::SACMODEL_PLANE);
	proj_.setInputCloud (temp_cloud);
	const pcl::PointIndices::ConstPtr table_inliers_const(table_inliers); // just to make eclipse shut up
	proj_.setIndices (table_inliers_const);
	proj_.setModelCoefficients (table_coefficients);
	proj_.filter (*table_projected);

	// Estimate the convex hull ---------------------------------------------------------
	boost::shared_ptr<pcl::PointCloud<PointT> > table_hull(new pcl::PointCloud<PointT>);
	pcl::ConvexHull<PointT> hull_;

	hull_.setInputCloud (table_projected);
	hull_.setDimension(2);
	hull_.reconstruct (*table_hull);

	// Get the objects on top of the table ======================================
	pcl::PointIndices::Ptr cloud_object_indices(new pcl::PointIndices);
	pcl::ExtractPolygonalPrismData<PointT> prism_;
	prism_.setHeightLimits (min_table_dist, max_table_dist);
	prism_.setInputCloud (input_cloud_ptr);
	prism_.setInputPlanarHull (table_hull);
	prism_.segment (*cloud_object_indices);

	std::vector<int> indices = cloud_object_indices->indices;

	if(keep_organized)
	{
		output_cloud.header = input_cloud.header;
		output_cloud.is_dense = false;
		output_cloud.height = input_cloud.height;
		output_cloud.width = input_cloud.width;
		output_cloud.points.resize(input_cloud.points.size());
		for(size_t i = 0; i < output_cloud.points.size(); i++)
		{
			output_cloud.points[i].x = NAN;
			output_cloud.points[i].y = NAN;
			output_cloud.points[i].z = NAN;
		}
		for(size_t i = 0; i < indices.size(); i++)
			output_cloud.points[indices[i]] = input_cloud.points[indices[i]];
	}
	else
	{
		output_cloud.header = input_cloud.header;
		output_cloud.is_dense = true;
		output_cloud.height = indices.size();
		output_cloud.width = 1;
		output_cloud.points.resize(indices.size());
		for(unsigned int i = 0; i < indices.size() ; i++)
			output_cloud.points[i] = input_cloud.points[indices[i]];
	}
}
template<typename MatrixT>
void PointsOnPlane(
		const std::vector<Eigen::Matrix<MatrixT, 3, 1> >& input_points,
		const size_t &input_rows, const size_t &input_cols,
		std::vector<Eigen::Matrix<MatrixT, 3, 1> >& output_points,
		size_t& output_rows, size_t& output_cols,
		Eigen::Matrix<MatrixT, 4, 1>& table_plane,
		const bool& keep_organized = false,
		const float& z_min = 0.3,  const float& z_max = 1.5,
		const float& y_min = -0.4, const float& y_max = 0.4,
		const float& x_min = -0.3, const float& x_max = 0.3,
		const float& min_table_dist = 0.01, const float& max_table_dist = 0.3,
		const float& grid_size = 0.02)
{
	pcl::PointCloud<pcl::PointXYZ> input_cloud;
	pi::Eigen2Pcl(input_points, input_rows, input_cols, input_cloud);


	pcl::PointCloud<pcl::PointXYZ> output_cloud;
	PointsOnPlane(input_cloud, output_cloud, table_plane,
			keep_organized, z_min, z_max, y_min, y_max, x_min, x_max, min_table_dist, max_table_dist, grid_size);

	pi::Pcl2Eigen(output_cloud, output_points, output_rows, output_cols);
}


template <typename PointT> void Cluster(
		const pcl::PointCloud<PointT>& input_point_cloud,
		std::vector<pcl::PointCloud<PointT> >& clusters,
		const float& cluster_delta_ = 0.01,
		const size_t& min_cluster_size_ = 200,
		const float& erosion_pixel_radius = 2)
{
	if(input_point_cloud.is_dense)
	{
		std::cout << "for clustering point cloud has to be organized" << std::endl;
		exit(-1);
	}

	clusters.clear();
	pcl::PointCloud<PointT> point_cloud = input_point_cloud;

	// create erosion operator ----------------------------------------------------------------------------------------------------------------------------------------
	std::vector<Eigen::Vector2i> erosion_operator;
	for(int col = -int(erosion_pixel_radius); col <= int(erosion_pixel_radius); col++)
		for(int row = -int(erosion_pixel_radius); row <= int(erosion_pixel_radius); row++)
			if(sqrt(float(col*col+row*row))<=erosion_pixel_radius)
				erosion_operator.push_back(Eigen::Vector2i(col,row));

	float max_depth=0.01;
	std::vector<Eigen::Vector2i> bad_indices;
	for(unsigned int col = 0; col < point_cloud.width; col++)
		for(unsigned int row = 0; row < point_cloud.height; row++)
		{
			if(isnan(point_cloud.at(col,row).x)) continue;

			for(int col_ = int(col-1); col_<= int(col+1); col_++)
				for(int row_ = int(row-1); row_<= int(row+1); row_++)
				{
					if(col_<0 || col_ >= (int) point_cloud.width || row_<0 || row_ >= (int) point_cloud.height)
					{
						bad_indices.push_back(Eigen::Vector2i(col,row));
						goto end;
					}

					Eigen::Vector3f point = point_cloud.at(col_, row_).getVector3fMap();
					Eigen::Vector3f center = point_cloud.at(col, row).getVector3fMap();
					if((point-center).norm() > max_depth || isnan(point_cloud.at(col_, row_).x))
					{
						bad_indices.push_back(Eigen::Vector2i(col,row));
						goto end;
					}
				}
			end:;
		}

	boost::shared_ptr<pcl::PointCloud<PointT> > eroded_cloud(new pcl::PointCloud<PointT>);
	*eroded_cloud = point_cloud;
	for(size_t i = 0; i < bad_indices.size(); i++)
	{
		for(size_t j = 0; j < erosion_operator.size(); j++)
		{
			Eigen::Vector2i index = bad_indices[i] + erosion_operator[j];
			int col_ = index(0);
			int row_ = index(1);

			if(col_<0 || col_ >= (int) eroded_cloud->width ||
					row_<0 || row_ >= (int) eroded_cloud->height ||
					isnan(eroded_cloud->at(col_, row_).x))
				continue;

			eroded_cloud->at(col_, row_).getVector3fMap() = Eigen::Vector3f(NAN, NAN, NAN);
		}
	}

	//clustering -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int window = cluster_delta_/0.002;

	for(unsigned int col = 0; col < point_cloud.width; col++)
		for(unsigned int row = 0; row < point_cloud.height; row++)
		{
			if(isnan(point_cloud.at(col,row).x)) continue;

			pcl::PointCloud<PointT> cluster;
			cluster.header = point_cloud.header;
			cluster.width = 0;
			cluster.height = 1;
			cluster.is_dense = true;
			cluster.points.clear();


			std::vector<Eigen::Vector2i> indices;
			std::vector<Eigen::Vector3f> centers;

			indices.push_back(Eigen::Vector2i(col,row));
			centers.push_back(point_cloud.at(col,row).getVector3fMap());

			while(indices.size())
			{
				int center_col = indices[indices.size()-1](0);
				int center_row = indices[indices.size()-1](1);
				indices.pop_back();

				Eigen::Vector3f center = centers[centers.size()-1];
				centers.pop_back();

				for(int window_col = center_col-window; window_col < center_col + window; window_col++)
					for(int window_row = center_row-window; window_row < center_row + window; window_row++)
					{
						if(window_col<0 || window_col >= (int) point_cloud.width || window_row<0 ||
								window_row >= (int) point_cloud.height || isnan(point_cloud.at(window_col, window_row).x) ) continue;

						Eigen::Vector3f point = point_cloud.at(window_col, window_row).getVector3fMap();

						if((center-point).norm() <= cluster_delta_)
						{
							if(!isnan(eroded_cloud->at(window_col, window_row).x))
							{
								cluster.points.push_back(point_cloud.at(window_col, window_row));
								cluster.width++;
							}

							indices.push_back(Eigen::Vector2i(window_col, window_row));
							centers.push_back(point_cloud.at(window_col, window_row).getVector3fMap());
							point_cloud.at(window_col, window_row).getVector3fMap() = Eigen::Vector3f(NAN, NAN, NAN);
						}

					}
			}
			if(cluster.size() >= min_cluster_size_)
				clusters.push_back(cluster);
		}

	// sort them from largest to smallest
	bool is_sorted = false;
	while(is_sorted == false)
	{
		is_sorted = true;
		for(int i = 0; i < int(clusters.size())-1; i++)
		{
			if(clusters[i].size() < clusters[i+1].size())
			{
				pcl::PointCloud<PointT> temp = clusters[i];
				clusters[i] = clusters[i+1];
				clusters[i+1] = temp;
				is_sorted = false;
			}
		}
	}
}
template <typename MatrixT> void Cluster(
		const std::vector<Eigen::Matrix<MatrixT, 3, 1> > &input_points,
		const size_t& input_rows,
		const size_t& input_cols,
		std::vector<std::vector<Eigen::Matrix<MatrixT, 3, 1> > > &output_points,
		const float& cluster_delta = 0.01,
		const size_t& min_cluster_size = 200,
		const float& erosion_pixel_radius = 2)
{
	pcl::PointCloud<pcl::PointXYZ> input_cloud;
	pi::Eigen2Pcl(input_points, input_rows, input_cols, input_cloud);

	std::vector<pcl::PointCloud<pcl::PointXYZ> > output_cloud;
	Cluster(input_cloud, output_cloud,
			cluster_delta, min_cluster_size, erosion_pixel_radius);

	output_points.resize(output_cloud.size());
	for(size_t i = 0; i < output_cloud.size(); i++)
		output_points[i] = pi::Pcl2Eigen<pcl::PointXYZ, MatrixT>(output_cloud[i]);
}


template <typename PointT> void FindCylinder(
		const pcl::PointCloud<PointT> &input_point_cloud,
		pcl::PointCloud<PointT> &inliers,
		pcl::PointCloud<PointT> &outliers,
		Eigen::Matrix<float, 7, 1> &coefficients,
		const float r_min,
		const float r_max)
{
	boost::shared_ptr<pcl::PointCloud<PointT> > point_cloud_ptr(new pcl::PointCloud<PointT>);
	*point_cloud_ptr = input_point_cloud;

	// Estimate point normals
	boost::shared_ptr<pcl::search::KdTree<PointT> > tree(new pcl::search::KdTree<PointT> ());
	boost::shared_ptr<pcl::PointCloud<pcl::Normal> > normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	ne.setSearchMethod (tree);
	ne.setInputCloud (point_cloud_ptr);
	ne.setKSearch (50);
	ne.compute(*normals);

	// Create the segmentation object for cylinder segmentation and set all the parameters
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_LINE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setNormalDistanceWeight (0.1);
	seg.setMaxIterations (100000);
	seg.setDistanceThreshold (0.01);//0.05
	seg.setRadiusLimits (r_min, r_max);
	seg.setInputCloud (point_cloud_ptr);
	seg.setInputNormals (normals);

	// Obtain the cylinder inliers and coefficients
	boost::shared_ptr<pcl::ModelCoefficients> coefficients_cylinder(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
	seg.segment (*inliers_cylinder, *coefficients_cylinder);

	// Write the cylinder inliers to disk
	pcl::ExtractIndices<PointT> extract;

	extract.setInputCloud (point_cloud_ptr);
	extract.setIndices (inliers_cylinder);
	extract.setNegative (false);
	extract.filter (inliers);


	boost::shared_ptr<pcl::PointCloud<PointT> > inliers_ptr(new pcl::PointCloud<PointT>);
	*inliers_ptr = inliers;

	pcl::SegmentDifferences<PointT> diff;
	diff.setSearchMethod(tree);
	diff.setInputCloud( point_cloud_ptr);
	diff.setTargetCloud(inliers_ptr);
	diff.setDistanceThreshold(0.02*0.02);
	diff.segment(outliers);


	for(size_t i = 0; i < 7; i++)
		coefficients(i) = coefficients_cylinder->values[i];
	coefficients.middleRows(3,3).normalize();


	Eigen::Vector3f p = coefficients.topRows(3);
	Eigen::Vector3f e = coefficients.middleRows(3,3);

	// find the beginning and the end of the cylinders
	float min = std::numeric_limits<float>::max();
	float max = -std::numeric_limits<float>::max();
	for(size_t i = 0; i < inliers.size(); i++)
	{
		Eigen::Vector3f point = inliers.points[i].getVector3fMap();

		min = min < e.dot(point - p) ? min : e.dot(point - p);
		max = max > e.dot(point - p) ? max : e.dot(point - p);
	}

	coefficients.topRows(3) = p + min*e;
	coefficients.middleRows(3,3) = p + max*e - coefficients.topRows(3);
}



// this function creates some samples around clusters on a plane. it assumes
// that when the object is standing on the table, the origin coincides with the
// table plane and z points upwards
template<typename Scalar> std::vector<FullRigidBodySystem<-1>::State>
SampleTableClusters(std::vector<Eigen::Matrix<Scalar,3,1> > points,
                    size_t n_rows, size_t n_cols,
                    size_t sample_count)
{
    typedef Eigen::Matrix<Scalar,3,1> Vector;
    typedef Eigen::Matrix<Scalar,3,3> Matrix;
    typedef Eigen::Matrix<Scalar,4,1> Plane;

    typedef FullRigidBodySystem<-1> BodySystem;

    vector<BodySystem::State> states;

    // find points on table and cluster them
    std::vector<Vector> table_points;
    size_t table_rows, table_cols;
    Plane table_plane;
    pi::PointsOnPlane(points, n_rows, n_cols, table_points, table_rows, table_cols, table_plane, true);
    if(table_plane.topRows(3).dot(Eigen::Vector3d(0,1,0)) > 0)
        table_plane = - table_plane;
    table_plane /= table_plane.topRows(3).norm();
    Vector table_normal = table_plane.topRows(3);
    std::vector<std::vector<Vector> > clusters;
    pi::Cluster(table_points, table_rows, table_cols, clusters);
    if(clusters.size() == 0)
        return states;

    // we create samples around the clusters on the table
    // create gaussian for sampling
    Scalar standard_deviation_translation = 0.03;
    Scalar standard_deviation_rotation = 100.0;
    filter::GaussianDistribution<Scalar, 1> unit_gaussian;
    unit_gaussian.setNormal();

    for(size_t cluster_index = 0; cluster_index < clusters.size(); cluster_index++)
    {
        Vector com(0,0,0);
        for(size_t i = 0; i < clusters[cluster_index].size(); i++)
            com += clusters[cluster_index][i];
        com /= Scalar(clusters[cluster_index].size());

        Vector t_mean = com - (com.dot(table_normal)+table_plane(3))*table_normal; // project center of mass in table plane
        Vector table_vector_a = table_normal.cross(Vector(1,1,1)).normalized(); // vector along table plane
        Vector table_vector_b = table_normal.cross(table_vector_a); // second vector along table plane
        Matrix R_mean; R_mean.col(0) = table_vector_a; R_mean.col(1) = table_vector_b; R_mean.col(2) = table_normal;

        // sample around mean
        for(size_t i = 0; i < size_t(sample_count)/clusters.size(); i++)
        {
            BodySystem body_system(1);
            body_system.position() =
                    t_mean +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_a +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_b;
            body_system.quaternion(Eigen::Quaterniond(
                                     Eigen::AngleAxisd(standard_deviation_rotation * unit_gaussian.sample()(0), table_normal) * R_mean));

            states.push_back(body_system);
        }
    }

    return states;
}














}








#endif
