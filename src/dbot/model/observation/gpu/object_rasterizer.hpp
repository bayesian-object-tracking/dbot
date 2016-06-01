/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file object_resterizer.hpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date November 2015
 */

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "GL/glew.h"

#include <dbot/model/observation/gpu/shader_provider.hpp>
#include <memory>

 /**
 * \brief renders the objects using openGL rasterization.
 * The objects that should be rendered have to be passed in the constructor and can then be rendered
 * in different poses with the render() function. The resulting depth values are stored in a texture
 * whose values can be obtained with get_depth_values(). Alternatively, get_framebuffer_texture() returns
 * the ID of the texture for mapping it into CUDA.
 */
class ObjectRasterizer
{
public:
    /**
     * \brief constructor which takes the vertices and indices that describe the objects as input. The paths to the
     * shader files and the instrinsic camera matrix also have to be passed here.
     * \param [in]  vertices [object_nr][vertex_nr] = {x, y, z}. This list should contain, for each object,
     * a list of 3-dimensional vectors that specify the corners of the triangles of the object mesh.
     * \param [in]  indices [object_nr][index_nr][0 - 2] = {index}. This list should contain the indices
     * that index the vertices list and tell us which vertices to connect to a triangle (every group of 3).
     * For each object, the indices should be in the range of [0, nr_vertices - 1].
     * \param [in]  vertex_shader_path path to the vertex shader
     * \param [in]  fragment_shader_path path to the fragment shader
     * \param [in]  camera_matrix matrix of the intrinsic parameters of the camera
     * \param [in]  nr_rows the number of rows in one sensor image (vertical resolution)
     * \param [in]  nr_cols the number of columns in one sensor image (horizontal resolution)
     * \param [in]  near_plane everything closer than the near plane will not be rendered. This should
     * be similar to the minimal distance up to which the sensor can see objects.
     * \param [in]  far_plane everything further away than the far plane will not be rendered. This should
     * be similar to the maximum distance up to which the sensor can see objects.
     */
    ObjectRasterizer(const std::vector<std::vector<Eigen::Vector3f> > vertices,
                     const std::vector<std::vector<std::vector<int> > > indices,
                     const std::shared_ptr<dbot::ShaderProvider>& shader_provider,
                     const Eigen::Matrix3f camera_matrix,
                     const int nr_rows,
                     const int nr_cols,
                     const float near_plane = 0.4,
                     const float far_plane = 4);

    /** destructor which deletes the buffers and programs used by openGL */
    ~ObjectRasterizer();


    /**
     * \brief render the objects in all given states and return the depth for all pixels of each rendered object.
     * This function renders all poses (of all objects) into one large texture. Reading back the depth values
     * is a relatively slow process, so this function should mainly be used for debugging. If you are using
     * CUDA to further process the depth values, please use the other render() function.
     * \param [in]  states [pose_nr][object_nr][0 - 6] = {qw, qx, qy, qz, tx, ty, tz}. This should contain the quaternion
     * and the translation for each object per pose.
     * \param [out] depth_values [pose_nr][0 - nr_pixels] = {depth value of that pixel}
     */
    void render(const std::vector<std::vector<Eigen::Matrix4f> > states,
                std::vector<std::vector<float> >& depth_values);

    /**
     * \brief render the objects in all given states into a texture that can then be accessed by CUDA.
     * This function renders all poses (of all objects) into one large texture, which can then be mapped into the CUDA
     * context. To get the ID of the texture, call get_texture_ID().
     * \param [in]  states [pose_nr][object_nr][0 - 6] = {qw, qx, qy, qz, tx, ty, tz}. This should contain the quaternion
     * and the translation for each object per pose.
     */
    void render(const std::vector<std::vector<Eigen::Matrix4f> > states);

    /**
     * \brief sets the objects that should be rendered.
     * This function only needs to be called if any objects initially passed in the constructor should be ignored when rendering.
     * \param [in]  object_numbers [0 - nr_objects] = {object_nr}. This list should contain the indices of all objects that
     * should be rendered when calling render(). For example, [0,1,4,5] will only render objects 0,1,4 and 5 (whose vertices
     * were passed in the constructor).
     */
    void set_objects(std::vector<int> object_numbers);

    /**
     * \brief set a new resolution.
     * \param [in]  nr_rows the height of the image
     * \param [in]  nr_cols the width of the image
     */
    void set_resolution(const int nr_rows, const int nr_cols);

    /**
     * \brief allocates memory on the GPU.
     * Use this function to allocate memory for the maximum number of poses that you will need throughout the filtering.
     * \param[in] nr_poses number of poses for which space should be allocated.
     * \param [in] nr_poses_per_row the number of poses that will be rendered per row of the texture
     * \param [in] nr_poses_per_col the number of poses that will be rendered per column of the texture
     */
    void allocate_textures_for_max_poses(int nr_poses,
                                         int nr_poses_per_row,
                                         int nr_poses_per_col);


    /**
     * \brief returns the OpenGL framebuffer texture ID, which is needed for CUDA interoperation.
     * Use this function to retrieve the texture ID and pass it to the cudaGraphicsGLRegisterImage call.
     * \return The texture ID
     */
    GLuint get_framebuffer_texture();

    /**
     * \brief returns the rendered depth values of all poses.
     * This function should only be used for debugging. It will be extremely slow.
     * \return [pose_nr][0 - nr_pixels] = {depth value of that pixel}
     */
    std::vector<std::vector<float> > get_depth_values(int nr_poses);

    /**
     * \brief returns the constant and per-pose memory needs that OpenGL will have (in bytes)
     * \param [in] nr_rows the vertical resolution per pose rendering
     * \param [in] nr_cols the horizontal resolution per pose rendering
     * \param [out] constant_need the amount of memory that OpenGL will need,
     * independently of the number of poses (in bytes)
     * \param [out] per_pose_need the amount of memory that OpenGL will need
     * per pose (in bytes)
     */
    void get_memory_need_parameters(int nr_rows, int nr_cols,
                                    int& constant_need, int& per_pose_need);

    /**
     * \brief returns the maximum texture size that OpenGL supports with this GPU
     * \return The maximum texture size that can be allocated
     */
    int get_max_texture_size();

private:
    // GPU constraints
    GLint max_texture_size_;

    // values initialized in constructor. May be changed by user with set_resolution().
    int nr_rows_;
    int nr_cols_;

    // values initialized in constructor. Cannot be changed afterwards.
    float near_plane_;
    float far_plane_;

    // number of poses in the current render call
    int nr_poses_;

    // maximum number of poses that can be rendered in one call
    int max_nr_poses_;
    int max_nr_poses_per_row_;
    int max_nr_poses_per_column_;

    // needed for OpenGL time measurement
    static const int NR_SUBROUTINES_TO_MEASURE = 4;
    GLuint time_query_[NR_SUBROUTINES_TO_MEASURE];
    enum subroutines_to_measure { ATTACH_TEXTURE, CLEAR_SCREEN, RENDER, DETACH_TEXTURE};
    std::vector<std::string> strings_for_subroutines;
    std::vector<double> time_measurement_;
    int nr_calls_;
    bool initial_run_;  // the first run should not count

    // lists of all vertices and indices of all objects
    std::vector<float> vertices_list_;
    std::vector<uint> indices_list_;
    std::vector<int> indices_per_object_;
    std::vector<int> start_position_;

    // contains a list of object indices which should be rendered
    std::vector<int> object_numbers_;

    // matrices to transform vertices into image space
    Eigen::Matrix4f projection_matrix_;
    Eigen::Matrix4f view_matrix_;

    // shader program ID and matrix uniform IDs to pass variables to them
    GLuint shader_ID_;
    GLuint model_view_matrix_ID_;    // ID to which we pass the modelview matrix
    GLuint projection_matrix_ID_;    // ID to which we pass the projection matrix

    // VAO, VBO and element arrays are needed to store the object meshes
    GLuint vertex_array_;   // The vertex array contains the vertex and index buffers
    GLuint vertex_buffer_;    // contains the vertices of the object meshes passed in the constructor
    GLuint index_buffer_;     // contains the indices of the object meshes passed in the constructor

    // PBO for copying results to CPU for debugging
    GLuint result_buffer_;

    // custom framebuffer and its textures for depth (for z-testing) and color (which also represents depth in our case)
    GLuint framebuffer_;
    GLuint framebuffer_texture_for_all_poses_;
    GLuint texture_for_z_testing;



    // ====================== PRIVATE FUNCTIONS ====================== //

    void reallocate_buffers();

    // set up view- and projection-matrix
    void setup_view_matrix();
    void setup_projection_matrix(const Eigen::Matrix3f camera_matrix);
    Eigen::Matrix4f get_projection_matrix(float n, float f, float l, float r, float t, float b);

    // functions for time measurement
    void store_time_measurements();
    std::string get_text_for_enum( int enumVal );

    // functions for error checking
    void check_GL_errors(const char *label);
    bool check_framebuffer_status();
};
