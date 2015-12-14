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
 * \file object_rasterizer.cpp
 * \author Claudia Pfreundt <claudilein@gmail.com>
 * \date November 2015
 */

#include <GL/glew.h>
#include <GL/glx.h>
#include <Eigen/Geometry>
#include <dbot/util/helper_functions.hpp>
#include <dbot/model/observation/gpu/shader.hpp>
#include <dbot/model/observation/gpu/object_rasterizer.hpp>

using namespace std;
using namespace Eigen;

ObjectRasterizer::ObjectRasterizer(const std::vector<std::vector<Eigen::Vector3f> > vertices,
                                   const std::vector<std::vector<std::vector<int> > > indices,
                                   const std::string vertex_shader_path,
                                   const std::string fragment_shader_path,
                                   const Eigen::Matrix3f camera_matrix,
                                   const float near_plane,
                                   const float far_plane,
                                   const int nr_rows,
                                   const int nr_cols) :
    near_plane_(near_plane),
    far_plane_(far_plane),
    nr_rows_(nr_rows),
    nr_cols_(nr_cols),
    vertex_shader_path_(vertex_shader_path),
    fragment_shader_path_(fragment_shader_path)
{

    // ========== CREATE WINDOWLESS OPENGL CONTEXT =========== //


    typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
    typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
    static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
    static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = 0;

    static int visual_attribs[] = {
       None
    };
    int context_attribs[] = {
           GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
           GLX_CONTEXT_MINOR_VERSION_ARB, 2,
           None
    };


    Display* dpy;
    int fbcount = 0;
    GLXFBConfig* fbc = NULL;
    GLXContext ctx;
    GLXPbuffer pbuf;


    /* open display */
    if ( ! (dpy = XOpenDisplay(0)) ){
           fprintf(stderr, "Failed to open display\n");
           exit(1);
    }

    /* get framebuffer configs, any is usable (might want to add proper attribs) */
    if ( !(fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), visual_attribs, &fbcount) ) ){
           fprintf(stderr, "Failed to get FBConfig\n");
           exit(1);
    }

    /* get the required extensions */
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB");
    glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent");
    if ( !(glXCreateContextAttribsARB && glXMakeContextCurrentARB) ){
           fprintf(stderr, "missing support for GLX_ARB_create_context\n");
           XFree(fbc);
           exit(1);
    }


    /* create a context using glXCreateContextAttribsARB */
    if ( !( ctx = glXCreateContextAttribsARB(dpy, fbc[0], 0, True, context_attribs)) ){
           fprintf(stderr, "Failed to create opengl context\n");
           XFree(fbc);
           exit(1);
    }


    /* create temporary pbuffer */
    int pbuffer_attribs[] = {
           GLX_PBUFFER_WIDTH, nr_cols_,
           GLX_PBUFFER_HEIGHT, nr_rows_,
           None
    };
    pbuf = glXCreatePbuffer(dpy, fbc[0], pbuffer_attribs);

    XFree(fbc);
    XSync(dpy, False);

    /* try to make it the current context */
    if ( !glXMakeContextCurrent(dpy, pbuf, pbuf, ctx) ){
           /* some drivers do not support context without default framebuffer, so fallback on
            * using the default window.
            */
           if ( !glXMakeContextCurrent(dpy, DefaultRootWindow(dpy), DefaultRootWindow(dpy), ctx) ){
                   fprintf(stderr, "failed to make current\n");
                   exit(1);
           }
    }

    /* try it out */
    printf("vendor: %s\n", (const char*)glGetString(GL_VENDOR));

    check_GL_errors("init windowsless context");

    // Initialize GLEW

    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }
    check_GL_errors("GLEW init");




    // ======================== SET OPENGL OPTIONS ======================== //


    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // ensures that all default values in a texture are = 1
    glClearColor(0.0, 0.0, 0.0, 0.0);

    // Disable color writes for other colors than RED
    // -> we use the RED channel of a color texture to save the depth values
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);

    // get GPU constraint values
    GLint max_texture_size;
    GLint max_renderbuffer_size;

    /* The values specify the max texture size in every dimension, i.e. 2048
     * would mean we can have a texture of 2048 x 2048. */
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_renderbuffer_size);

    max_texture_size_ = min(max_texture_size, max_renderbuffer_size);



    // ========== CHANGE VERTICES AND INDICES FORMATS SO THAT OPENGL CAN READ THEM =========== //


    std::vector<int> vertices_per_object;

    for (size_t i = 0; i < vertices.size(); i++) {        // each i equals one object
        object_numbers_.push_back(i);
        vertices_per_object.push_back(vertices[i].size());
        for (size_t j = 0; j < vertices[i].size(); j++) { // each j equals one vertex in that object
            for (int k = 0; k < vertices[i][j].size(); k++) {  // each k equals one dimension of that vertex
                vertices_list_.push_back(vertices[i][j][k]);
            }
        }
    }

    start_position_.push_back(0);
    int vertex_count = 0;
    for (size_t i = 0; i < indices.size(); i++) {         // each i equals one object
        indices_per_object_.push_back(indices[i].size() * 3);
        start_position_.push_back(start_position_[i] + indices_per_object_[i]);
        for (size_t j = 0; j < indices[i].size(); j++) {         // each j equals one triangle in that object
            for (size_t k = 0; k < indices[i][j].size(); k++) {   // each k equals one index for a vertex in that triangle
                indices_list_.push_back(indices[i][j][k] + vertex_count);
            }
        }
        vertex_count += vertices_per_object[i];
    }


    // ==================== CREATE AND FILL VAO, VBO & element array ==================== //

    // creating a vertex array object (VAO)
    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);

    // creating a vertex buffer object (VBO) and filling it with vertices
    glGenBuffers(1, &vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, vertices_list_.size() * sizeof(float), &vertices_list_[0], GL_STATIC_DRAW);

    // create and fill index buffer with indices_
    glGenBuffers(1, &index_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_list_.size() * sizeof(uint), &indices_list_[0], GL_STATIC_DRAW);


    // ============== TELL OPENGL WHERE TO LOOK FOR VERTICES ============== //

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );


    // ======================= CREATE PBO ======================= //

    // create PBO that will be used for copying the depth values to the CPU, if requested
    glGenBuffers(1, &result_buffer_);


    // ======================= CREATE FRAMEBUFFER OBJECT AND ITS TEXTURES ======================= //

    // create a framebuffer object
    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

    // create color texture that will contain the depth values after the rendering
    glGenTextures(1, &framebuffer_texture_for_all_poses_);
    glBindTexture(GL_TEXTURE_2D, framebuffer_texture_for_all_poses_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // create a renderbuffer to store depth info for z-testing
    glGenRenderbuffers(1, &texture_for_z_testing);


    // ================= COMPILE SHADERS AND GET HANDLES ================= //

    // Create and compile our GLSL program from the shaders
    vector<const char *> shader_list;
    shader_list.clear();
    shader_list.push_back(vertex_shader_path_.c_str());
    shader_list.push_back(fragment_shader_path_.c_str());
    shader_ID_ = LoadShaders(shader_list);


    // Set up handles for uniforms
    model_view_matrix_ID_ = glGetUniformLocation(shader_ID_, "MV");
    projection_matrix_ID_ = glGetUniformLocation(shader_ID_, "P");

    /* The view matrix is constant throughout this class since we are not changing the camera position.
       If you are looking to pass a different camera matrix for each render call, move this function
       into the render call and change it to accept the camera position and rotation as parameter
       and calculate the respective matrix. */
    setup_view_matrix();

    check_GL_errors("OpenGL initialization");


    // ========== SETUP PROJECTION MATRIX AND SHADER TO USE =========== //

    setup_projection_matrix(camera_matrix);
    glUseProgram(shader_ID_);
    check_GL_errors("setup projection matrix");


    // ========== INITIALIZE & SET DEFAULTS FOR MEASURING EXECUTION TIMES =========== //

#ifdef PROFILING_ACTIVE
    // generate query objects needed for timing OpenGL commands
    glGenQueries(NR_SUBROUTINES_TO_MEASURE, time_query_);

    nr_calls_ = 0;
    time_measurement_ = vector<double> (NR_SUBROUTINES_TO_MEASURE, 0);
    initial_run_ = true;

    strings_for_subroutines.push_back("ATTACH_TEXTURE");
    strings_for_subroutines.push_back("CLEAR_SCREEN");
    strings_for_subroutines.push_back("RENDER");
    strings_for_subroutines.push_back("DETACH_TEXTURE");

    check_GL_errors("Generating time queries");
#endif
}



void ObjectRasterizer::render(const std::vector<std::vector<Eigen::Matrix4f> > states,
                              std::vector<std::vector<float> > depth_values) {
    render(states);
    depth_values = get_depth_values();
}


void ObjectRasterizer::render(const std::vector<std::vector<Eigen::Matrix4f> > states) {


#ifdef PROFILING_ACTIVE
    glBeginQuery(GL_TIME_ELAPSED, time_query_[ATTACH_TEXTURE]);
    nr_calls_++;
#endif

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           framebuffer_texture_for_all_poses_,     // 4. tex ID
                           0);
#ifdef DEBUG
    check_GL_errors("attaching texture to framebuffer");
#endif
#ifdef PROFILING_ACTIVE
    glFinish();
    glEndQuery(GL_TIME_ELAPSED);
    glBeginQuery(GL_TIME_ELAPSED, time_query_[CLEAR_SCREEN]);
#endif

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

#ifdef DEBUG
    check_GL_errors("clearing framebuffer");
#endif
#ifdef PROFILING_ACTIVE
    glFinish();
    glEndQuery(GL_TIME_ELAPSED);
    glBeginQuery(GL_TIME_ELAPSED, time_query_[RENDER]);
#endif

    glUniformMatrix4fv(projection_matrix_ID_, 1, GL_FALSE, projection_matrix_.data());

    Matrix4f model_view_matrix;

    for (int i = 0; i < nr_poses_per_column_ -1; i++) {
        for (int j = 0; j < nr_poses_per_row_; j++) {

            glViewport(j * nr_cols_, (nr_poses_per_column_ - 1 - i) * nr_rows_, nr_cols_, nr_rows_);
            #ifdef DEBUG
                check_GL_errors("setting the viewport");
            #endif
            for (size_t k = 0; k < object_numbers_.size(); k++) {
                int index = object_numbers_[k];

                model_view_matrix = view_matrix_ * states[nr_poses_per_row_ * i + j][index];
                glUniformMatrix4fv(model_view_matrix_ID_, 1, GL_FALSE, model_view_matrix.data());

                glDrawElements(GL_TRIANGLES, indices_per_object_[index], GL_UNSIGNED_INT, (void*) (start_position_[index] * sizeof(uint)));
                #ifdef DEBUG
                    check_GL_errors("render call");
                #endif
            }
        }
    }

    // render last row of poses
    for (int j = 0; j < nr_poses_ - (nr_poses_per_row_ * (nr_poses_per_column_ - 1)); j++) {

        glViewport(j * nr_cols_, 0, nr_cols_, nr_rows_);
        #ifdef DEBUG
            check_GL_errors("setting the viewport");
        #endif

        for (size_t k = 0; k < object_numbers_.size(); k++) {
            int index = object_numbers_[k];

            model_view_matrix = view_matrix_ * states[nr_poses_per_row_ * (nr_poses_per_column_ - 1) + j][index];
            glUniformMatrix4fv(model_view_matrix_ID_, 1, GL_FALSE, model_view_matrix.data());

            glDrawElements(GL_TRIANGLES, indices_per_object_[index], GL_UNSIGNED_INT, (void*) (start_position_[index] * sizeof(uint)));
            #ifdef DEBUG
                check_GL_errors("render call");
            #endif
        }
    }


#ifdef PROFILING_ACTIVE
    glFinish();
    glEndQuery(GL_TIME_ELAPSED);
    glBeginQuery(GL_TIME_ELAPSED, time_query_[DETACH_TEXTURE]);
#endif

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           0,     // 4. tex ID
                           0);

#ifdef DEBUG
    check_GL_errors("detaching texture from framebuffer");
#endif

#ifdef PROFILING_ACTIVE
    glFinish();
    glEndQuery(GL_TIME_ELAPSED);
    store_time_measurements();
#endif


}


void ObjectRasterizer::set_objects(vector<int> object_numbers) {
    object_numbers_ = object_numbers;
}


void ObjectRasterizer::set_resolution(const int n_rows, const int n_cols,
                                      int& nr_poses, int& nr_poses_per_row, int& nr_poses_per_column, const bool adapt_to_constraints) {
        nr_rows_ = n_rows;
        nr_cols_ = n_cols;

        // reallocate textures
        allocate_textures_for_max_poses(nr_poses, nr_poses_per_row, nr_poses_per_column, adapt_to_constraints);
}

void ObjectRasterizer::allocate_textures_for_max_poses(int& allocated_poses,
                                                       int& allocated_poses_per_row,
                                                       int& allocated_poses_per_column,
                                                       const bool adapt_to_constraints) {
    int max_poses_per_row = floor(max_texture_size_ / nr_cols_);
    int max_poses_per_column = floor(max_texture_size_ / nr_rows_);

    allocated_poses_per_row = min(max_poses_per_row, allocated_poses);
    allocated_poses_per_column = min(max_poses_per_column, (int) ceil(allocated_poses / (float) allocated_poses_per_row));

    if (allocated_poses > allocated_poses_per_row * allocated_poses_per_column) {
        if (adapt_to_constraints) {
            std::cout << "WARNING (OPENGL): The space for the number of maximum poses you requested (" << allocated_poses << ") cannot be allocated. "
                      << "The limit is OpenGL texture size: " << max_texture_size_ << ". Current resolution is (" << nr_cols_ << ", "
                      << nr_rows_ << ") , which means a maximum of (" << max_poses_per_row << ", " << max_poses_per_column << ") poses. "
                      << "As a result, space for " << allocated_poses_per_row * allocated_poses_per_column << " poses will be allocated "
                      << "in the form of (" << allocated_poses_per_row << ", " << allocated_poses_per_column << ")." << std::endl;
        } else {
            std::cout << "ERROR (OPENGL): The number of poses you requested cannot be rendered. The limit is the maximum OpenGL texture size: "
                      << max_texture_size_ << " x " << max_texture_size_ << ". You requested a resolution of " << nr_cols_  << " x " << nr_rows_
                      << " and " << allocated_poses << " poses." << std::endl;
            exit(-1);
        }
    }

    allocated_poses = allocated_poses_per_row * allocated_poses_per_column;

    nr_max_poses_ = allocated_poses;
    nr_poses_ = allocated_poses;
    nr_max_poses_per_row_ = allocated_poses_per_row;
    nr_poses_per_row_ = allocated_poses_per_row;
    nr_max_poses_per_column_ = allocated_poses_per_column;
    nr_poses_per_column_ = allocated_poses_per_column;

    reallocate_buffers();
}

void ObjectRasterizer::set_number_of_poses(int& nr_poses, int& nr_poses_per_row, int& nr_poses_per_column, const bool adapt_to_constraints) {

    if (nr_poses > nr_max_poses_) {
        if (adapt_to_constraints) {
            std::cout << "WARNING (OPENGL): You tried to evaluate more poses (" << nr_poses << ") than specified by max_poses (" << nr_max_poses_ << ")."
                      << "The number of poses was automatically reduced to " << nr_max_poses_ << "." << std::endl;
            nr_poses = nr_max_poses_;
        } else {
            cout << "ERROR (OPENGL): You tried to evaluate more poses (" << nr_poses << ") than specified by max_poses (" << nr_max_poses_ << ")" << endl;
            exit(-1);
        }
    }

    nr_poses_per_row = min(nr_max_poses_per_row_, nr_poses);
    nr_poses_per_column = min(nr_max_poses_per_column_, (int) ceil(nr_poses / (float) nr_poses_per_row));

    nr_poses_ = nr_poses;
    nr_poses_per_row_ = nr_poses_per_row;
    nr_poses_per_column_ = nr_poses_per_column;
}


GLuint ObjectRasterizer::get_framebuffer_texture() {
    return framebuffer_texture_for_all_poses_;
}



vector<vector<float> > ObjectRasterizer::get_depth_values() {

    // ===================== ATTACH TEXTURE TO FRAMEBUFFER ================ //

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           framebuffer_texture_for_all_poses_,     // 4. tex ID
                           0);

    // ===================== TRANSFER DEPTH VALUES FROM GPU TO CPU == SLOW!!! ================ //

    glBindBuffer(GL_PIXEL_PACK_BUFFER, result_buffer_);
    glBindTexture(GL_TEXTURE_2D, framebuffer_texture_for_all_poses_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, 0);

    GLfloat *pixel_depth = (GLfloat*) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);


    if (pixel_depth != (GLfloat*) NULL) {
        vector<float> depth_image(nr_rows_ * nr_cols_ * nr_poses_per_row_ * nr_poses_per_column_, numeric_limits<float>::max());

        int pixels_per_row_texture = nr_max_poses_per_row_ * nr_cols_;
        int pixels_per_row_extracted = nr_poses_per_row_ * nr_cols_;
        int highest_pixel_per_col = (nr_poses_per_column_ * nr_rows_) - 1;

        // reading OpenGL texture into an array on the CPU (inverted rows)
        for (int row = 0; row < nr_poses_per_column_ * nr_rows_; row++) {
            int inverted_row = highest_pixel_per_col - row;

            for (int col = 0; col < nr_poses_per_row_ * nr_cols_; col++) {
                depth_image[row * pixels_per_row_extracted  + col] = pixel_depth[inverted_row * pixels_per_row_texture + col];
            }
        }

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

        // subdividing this array into an array per pose
        vector<vector<float> > depth_image_per_pose (nr_poses_, vector<float> (nr_rows_ * nr_cols_, 0));

        for (int pose_y = 0; pose_y < nr_poses_per_column_; pose_y++) {
            for (int pose_x = 0; pose_x < nr_poses_per_row_ && pose_y * nr_poses_per_row_ + pose_x < nr_poses_; pose_x++) {
                for (int i = 0; i < nr_rows_ * nr_cols_; i++) {

                    depth_image_per_pose[pose_y * nr_poses_per_row_ + pose_x][i] = depth_image[(pose_y * nr_rows_ + (i / nr_cols_)) * pixels_per_row_extracted + pose_x * nr_cols_ + (i % nr_cols_)];
                }
            }
        }

        return depth_image_per_pose;

    } else {
        cout << "WARNING: Could not map Pixel Pack Buffer." << endl;
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // ===================== DETACH TEXTURE FROM FRAMEBUFFER ================ //

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           0,     // 4. tex ID
                           0);

    #ifdef DEBUG
        check_GL_errors("copying depth values to CPU");
    #endif
}



// ================================================================= //
// ================================================================= //
// ======================= PRIVATE FUNCTIONS ======================= //
// ================================================================= //
// ================================================================= //



void ObjectRasterizer::reallocate_buffers() {

    // ======================= REALLOCATE PBO ======================= //

    glBindBuffer(GL_PIXEL_PACK_BUFFER, result_buffer_);
    // the NULL means this buffer is uninitialized, since I only want to copy values back to the CPU that will be written by the GPU
    glBufferData(GL_PIXEL_PACK_BUFFER, nr_max_poses_per_row_* nr_cols_ * nr_max_poses_per_column_ * nr_rows_ *  sizeof(GLfloat), NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // ======================= DETACH TEXTURES FROM FRAMEBUFFER ======================= //

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,      // 1. fbo target: GL_FRAMEBUFFER
                              GL_DEPTH_ATTACHMENT, // 2. attachment point
                              GL_RENDERBUFFER,     // 3. rbo target: GL_RENDERBUFFER
                              0);     // 4. rbo ID
    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           0,     // 4. tex ID
                           0);

    // ======================= REALLOCATE FRAMEBUFFER TEXTURES ======================= //

    glBindTexture(GL_TEXTURE_2D, framebuffer_texture_for_all_poses_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nr_max_poses_per_row_ * nr_cols_, nr_max_poses_per_column_ * nr_rows_, 0, GL_RED, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, texture_for_z_testing);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, nr_max_poses_per_row_* nr_cols_, nr_max_poses_per_column_ * nr_rows_);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // ======================= ATTACH NEW TEXTURES TO FRAMEBUFFER ======================= //

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,      // 1. fbo target: GL_FRAMEBUFFER
                              GL_DEPTH_ATTACHMENT, // 2. attachment point
                              GL_RENDERBUFFER,     // 3. rbo target: GL_RENDERBUFFER
                              texture_for_z_testing);     // 4. rbo ID
    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           framebuffer_texture_for_all_poses_,     // 4. tex ID
                           0);


    check_framebuffer_status();
    GLenum color_buffers[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, color_buffers);
}


void ObjectRasterizer::setup_view_matrix() {
    // =========================== VIEW MATRIX =========================== //

    // OpenGL camera is rotated 180 degrees around the X-Axis compared to the Kinect camera
    view_matrix_ = Matrix4f::Identity();
    Transform<float, 3, Affine, ColMajor> view_matrix_transform;
    view_matrix_transform = AngleAxisf(M_PI, Vector3f::UnitX());
    view_matrix_ = view_matrix_transform.matrix();
}

void ObjectRasterizer::setup_projection_matrix(const Eigen::Matrix3f camera_matrix) {

    // ==================== CONFIGURE IMAGE PARAMETERS ==================== //

    Eigen::Matrix3f camera_matrix_inverse = camera_matrix.inverse();

    Vector3f boundaries_min = camera_matrix_inverse * Vector3f(-0.5, -0.5, 1);
    Vector3f boundaries_max = camera_matrix_inverse * Vector3f(float(nr_cols_)-0.5, float(nr_rows_)-0.5, 1);

    float near = near_plane_;
    float far = far_plane_;
    float left = near * boundaries_min(0);
    float right = near * boundaries_max(0);
    float top = -near * boundaries_min(1);
    float bottom = -near * boundaries_max(1);

    // ======================== PROJECTION MATRIX ======================== //

    // Projection Matrix takes into account our view frustum and projects the (3D)-scene onto a 2D image
    projection_matrix_ = get_projection_matrix(near, far, left, right, top, bottom);

}


Matrix4f ObjectRasterizer::get_projection_matrix(float n, float f, float l, float r, float t, float b) {
    Matrix4f projection_matrix = Matrix4f::Zero();
    projection_matrix(0,0) = 2 * n / (r - l);
    projection_matrix(0,2) = (r + l) / (r - l);
    projection_matrix(1,1) = 2 * n / (t - b);
    projection_matrix(1,2) = (t + b) / (t - b);
    projection_matrix(2,2) = -(f + n) / (f - n);
    projection_matrix(2,3) = - 2 * f * n / (f - n);
    projection_matrix(3,2) = -1;

    return projection_matrix;
}

void ObjectRasterizer::store_time_measurements() {
#ifdef PROFILING_ACTIVE

    // retrieve times from OpenGL and store them
    for (int i = 0; i < NR_SUBROUTINES_TO_MEASURE; i++) {
        GLint available = 0;
        double time_elapsed_s;
        unsigned int time_elapsed_ns;

        while (!available) {
            glGetQueryObjectiv(time_query_[i], GL_QUERY_RESULT_AVAILABLE, &available);
        }
        glGetQueryObjectuiv(time_query_[i], GL_QUERY_RESULT, &time_elapsed_ns);
        time_elapsed_s = time_elapsed_ns / (double) 1e9;
        time_measurement_[i] += time_elapsed_s;

    }

    // the first run should not count. Reset all the counters.
    if (initial_run_) {
        initial_run_ = false;
        for (int i = 0; i < NR_SUBROUTINES_TO_MEASURE; i++) {
            time_measurement_[i] = 0;
        }

        nr_calls_ = 0;
    }
#endif
}


string ObjectRasterizer::get_text_for_enum( int enumVal ) {
    return strings_for_subroutines[enumVal];
}


void ObjectRasterizer::check_GL_errors(const char *label) {
    GLenum errCode;
    const GLubyte *errStr;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errStr = gluErrorString(errCode);
        printf("OpenGL ERROR: ");
        printf("%s", (char*)errStr);
        printf("(Label: ");
        printf("%s", label);
        printf(")\n.");
    }
}

bool ObjectRasterizer::check_framebuffer_status() {
    GLenum status;
    status=(GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            printf("Framebuffer incomplete,incomplete attachment\n");
            return false;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            printf("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            printf("Framebuffer incomplete,missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            printf("Framebuffer incomplete,attached images must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
             printf("Framebuffer incomplete,attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            printf("Framebuffer incomplete,missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            printf("Framebuffer incomplete,missing read buffer\n");
            return false;
    }
    return false;
}


ObjectRasterizer::~ObjectRasterizer()
{

#ifdef PROFILING_ACTIVE

    if (nr_calls_ != 0) {
        cout << endl << "Time measurements for the different steps of the rendering process averaged over " << nr_calls_ << " render calls:" << endl << endl;

        double total_time_per_render = 0;

        for (int i = 0; i < NR_SUBROUTINES_TO_MEASURE; i++) {
            total_time_per_render += time_measurement_[i];
        }
        total_time_per_render /= nr_calls_;

        for (int i = 0; i < NR_SUBROUTINES_TO_MEASURE; i++) {
            double time_per_subroutine = time_measurement_[i] / nr_calls_;

                cout << get_text_for_enum(i) << ":     "
                     << "\t " << time_per_subroutine << " s \t " << setprecision(1)
                     << time_per_subroutine * 100 / total_time_per_render << " %" << setprecision(9) << endl;
        }

        cout << "TOTAL TIME PER RENDER CALL : " << total_time_per_render << endl << endl;
    } else {
        cout << "The render() function was never called, so there are no time measurements of it available." << endl;
    }

    glDeleteQueries(NR_SUBROUTINES_TO_MEASURE, time_query_);
#endif


    glDisableVertexAttribArray(0);
    glDeleteVertexArrays(1, &vertex_array_);

    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteBuffers(1, &index_buffer_);
    glDeleteBuffers(1, &result_buffer_);

    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteTextures(1, &framebuffer_texture_for_all_poses_);
    glDeleteRenderbuffers(1, &texture_for_z_testing);

    glDeleteProgram(shader_ID_);

}

