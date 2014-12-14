/** @author Claudia Pfreundt */

//#define PROFILING_ACTIVE

#include <pose_tracking/models/observation_models/kinect_image_observation_model_gpu/object_rasterizer.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <iostream>
#include <GL/glx.h>
#include <Eigen/Geometry>
#include <ros/package.h>


#include <pose_tracking/models/observation_models/kinect_image_observation_model_gpu/shader.hpp>

#include <ff/utils/helper_functions.hpp>

using namespace std;
using namespace Eigen;


ObjectRasterizer::ObjectRasterizer()
{
}

ObjectRasterizer::ObjectRasterizer(const std::vector<std::vector<Eigen::Vector3f> > vertices,
                                   const std::vector<std::vector<std::vector<int> > > indices,
                                   const std::string vertex_shader_path,
                                   const std::string fragment_shader_path) :
    n_rows_(WINDOW_HEIGHT),
    n_cols_(WINDOW_WIDTH),
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
           GLX_PBUFFER_WIDTH, n_cols_,
           GLX_PBUFFER_HEIGHT, n_rows_,
           None
    };
    pbuf = glXCreatePbuffer(dpy, fbc[0], pbuffer_attribs);


    XFree(fbc);
    XSync(dpy, False);

    /* try to make it the current context */
    if ( !glXMakeContextCurrent(dpy, pbuf, pbuf, ctx) ){
           /* some drivers does not support context without default framebuffer, so fallback on
            * using the default window.
            */
           if ( !glXMakeContextCurrent(dpy, DefaultRootWindow(dpy), DefaultRootWindow(dpy), ctx) ){
                   fprintf(stderr, "failed to make current\n");
                   exit(1);
           }
    }

    /* try it out */
    printf("vendor: %s\n", (const char*)glGetString(GL_VENDOR));

    checkGLErrors("init windowsless context");

    // Initialize GLEW

    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }
    checkGLErrors("GLEW init");







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
    GLint max_viewport_dims;
    GLint max_renderbuffer_size;

    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, &max_viewport_dims);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_renderbuffer_size);

    /* TODO this only checks for the first dimension and then assumes the size will be
     * the same for the other dimension. It is most likely to be so, but maybe this
     * should be changed to avoid errors in the future. */
    max_dimension_ = min(max_viewport_dims, max_renderbuffer_size);




    // ========== CHANGE VERTICES AND INDICES FORMATS SO THAT OPENGL CAN READ THEM =========== //

    for (size_t i = 0; i < vertices.size(); i++) {        // each i equals one object
        object_numbers_.push_back(i);
        // vertices_per_object_.push_back(vertices[i].size() * 3); ?? * 3 equals floats per object, how does gldrawelements index?
        vertices_per_object_.push_back(vertices[i].size());
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
        vertex_count += vertices_per_object_[i];
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

//    // Index buffer
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);




    // ======================= CREATE PBO ======================= //

    // create PBO that will be used for copying the depth values to the CPU, if requested
    glGenBuffers(1, &combined_result_buffer_);


    // ======================= CREATE FRAMEBUFFER OBJECT AND ITS TEXTURES ======================= //

    // create a framebuffer object
    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

    // create color texture that will contain the depth values after the render
    glGenTextures(1, &combined_texture_);
    glBindTexture(GL_TEXTURE_2D, combined_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // create a renderbuffer to store depth info for z-testing
    glGenRenderbuffers(1, &combined_depth_texture_);


    // ================= COMPILE SHADERS AND GET HANDLES ================= //

    // Create and compile our GLSL program from the shaders
    vector<const char *> shader_list;
    shader_list.clear();
    shader_list.push_back(vertex_shader_path_.c_str());
    shader_list.push_back(fragment_shader_path_.c_str());
    shader_ID_ = LoadShaders(shader_list);


    // Set up handles for uniforms in transformation saving shaders
    model_view_matrix_ID_ = glGetUniformLocation(shader_ID_, "MV");
    projection_matrix_ID_ = glGetUniformLocation(shader_ID_, "P");

    /* The view matrix is constant throughout this class since we are not changing the camera position.
       If you are looking to pass a different camera matrix for each render call, move this function
       into that render call and change it to accept the camera position and rotation as parameter
       and calculate the respective matrix. */
    SetupViewMatrix();

    checkGLErrors("OpenGL initialization");



    // ========== INITIALIZE & SET DEFAULTS FOR MEASURING EXECUTION TIMES =========== //


    // generate query objects needed for timing OpenGL commands
    glGenQueries(TIME_MEASUREMENTS_COUNT, time_query_);

    vector<int> zero_calls (TIME_MEASUREMENTS_COUNT, 0);
    vector<double> times (TIME_MEASUREMENTS_COUNT, 0);
    for (int i = 0; i < RENDER_TYPE_COUNT; i++) {
        calls_aggregate_.push_back(vector<int>(zero_calls));
        cpu_times_aggregate_.push_back(vector<double>(times));
        gpu_times_aggregate_.push_back(vector<double>(times));
        poses_rendered_.push_back(0);
    }

    for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {
        cpu_times_[i] = -1;
    }
    for (int i = 0; i < RENDER_TYPE_COUNT; i++) {
        initial_[i] = true;
    }

    enum_strings_.push_back("SEND_MATRICES");
    enum_strings_.push_back("CLEAR_SCREEN");
    enum_strings_.push_back("RENDER");
    enum_strings_.push_back("FILL_PBO");
    enum_strings_.push_back("MAP_PBO");
    enum_strings_.push_back("GET_DEPTH_VALUES");
    enum_strings_.push_back("SEND_CONSTANTS");
    enum_strings_.push_back("SEND_TEXTURES");
    enum_strings_.push_back("SUM_LIKELIHOODS");
    enum_strings_.push_back("UPDATE_TEXTURES");
    enum_strings_.push_back("ATTACH_TEXTURE");
    enum_strings_.push_back("ALLOCATE");
    enum_strings_.push_back("SET_VIEWPORT");
    enum_strings_.push_back("SEND_MODEL_MATRIX");
    enum_strings_.push_back("DETACH_TEXTURE");
    enum_strings_.push_back("GL_FINISH");
}










void ObjectRasterizer::PrepareRender(const Eigen::Matrix3f camera_matrix) {
    SetupProjectionMatrix(camera_matrix);
    glUseProgram(shader_ID_);
    checkGLErrors("prepare RenderCombinedFast");
}









void ObjectRasterizer::Render(const std::vector<std::vector<std::vector<float> > > states,
                                          std::vector<std::vector<int> > &intersect_indices,
                                          std::vector<std::vector<float> > &depth) {
    Render(states);
    get_depth_values(intersect_indices, depth);
}



void ObjectRasterizer::Render(const std::vector<std::vector<std::vector<float> > > states) {

#ifdef PROFILING_ACTIVE
    combined_fast_set_viewport_queries_.resize(n_poses_);
    combined_fast_send_model_matrix_queries_.resize(n_poses_);
    combined_fast_render_queries_.resize(n_poses_);
    glGenQueries(n_poses_, &combined_fast_set_viewport_queries_[0]);
    glGenQueries(n_poses_, &combined_fast_send_model_matrix_queries_[0]);
    glGenQueries(n_poses_, &combined_fast_render_queries_[0]);
    glBeginQuery(GL_TIME_ELAPSED, time_query_[ATTACH_TEXTURE]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][ATTACH_TEXTURE]++;
#endif

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           combined_texture_,     // 4. tex ID
                           0);

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[ATTACH_TEXTURE] = stop_time_ - start_time_;

    glBeginQuery(GL_TIME_ELAPSED, time_query_[CLEAR_SCREEN]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][CLEAR_SCREEN]++;
#endif

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[CLEAR_SCREEN] = stop_time_ - start_time_;

    glBeginQuery(GL_TIME_ELAPSED, time_query_[SEND_MATRICES]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][SEND_MATRICES]++;
#endif

    glUniformMatrix4fv(projection_matrix_ID_, 1, GL_FALSE, projection_matrix_.data());

    Matrix4f model_view_matrix;

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[SEND_MATRICES] = stop_time_ - start_time_;
#endif


    for (int i = 0; i < n_poses_y_ -1; i++) {
        for (int j = 0; j < n_poses_x_; j++) {

    #ifdef PROFILING_ACTIVE
        glBeginQuery(GL_TIME_ELAPSED, combined_fast_set_viewport_queries_[i * n_poses_x_ + j]);
//        glBeginQuery(GL_TIME_ELAPSED, time_query_[SET_VIEWPORT]);
        start_time_ = sf::hf::get_wall_time();
        calls_aggregate_[COMBINED_FAST][SET_VIEWPORT]++;
    #endif

            glViewport(j * n_cols_, (n_poses_y_ - 1 - i) * n_rows_, n_cols_, n_rows_);

    #ifdef PROFILING_ACTIVE

        glEndQuery(GL_TIME_ELAPSED);
        stop_time_ = sf::hf::get_wall_time();
        cpu_times_[SET_VIEWPORT] = stop_time_ - start_time_;
    #endif

            for (size_t k = 0; k < object_numbers_.size(); k++) {
                int index = object_numbers_[k];

        #ifdef PROFILING_ACTIVE
            glBeginQuery(GL_TIME_ELAPSED, combined_fast_send_model_matrix_queries_[i * n_poses_x_ + j]);
//            glBeginQuery(GL_TIME_ELAPSED, time_query_[SEND_MODEL_MATRIX]);
            start_time_ = sf::hf::get_wall_time();
            calls_aggregate_[COMBINED_FAST][SEND_MODEL_MATRIX]++;
        #endif

                model_view_matrix = view_matrix_ * GetModelMatrix(states[n_poses_x_ * i + j][index]);
                glUniformMatrix4fv(model_view_matrix_ID_, 1, GL_FALSE, model_view_matrix.data());

        #ifdef PROFILING_ACTIVE
            glEndQuery(GL_TIME_ELAPSED);
            stop_time_ = sf::hf::get_wall_time();
            cpu_times_[SEND_MODEL_MATRIX] = stop_time_ - start_time_;
            glBeginQuery(GL_TIME_ELAPSED, combined_fast_render_queries_[i * n_poses_x_ + j]);
//            glBeginQuery(GL_TIME_ELAPSED, time_query_[RENDER]);
            start_time_ = sf::hf::get_wall_time();
            calls_aggregate_[COMBINED_FAST][RENDER]++;
        #endif

            glDrawElements(GL_TRIANGLES, indices_per_object_[index], GL_UNSIGNED_INT, (void*) (start_position_[index] * sizeof(uint)));

        #ifdef PROFILING_ACTIVE

            glEndQuery(GL_TIME_ELAPSED);
            stop_time_ = sf::hf::get_wall_time();
            cpu_times_[RENDER] = stop_time_ - start_time_;
        #endif
            }
        }
    }

    for (int j = 0; j < n_poses_ - (n_poses_x_ * (n_poses_y_ - 1)); j++) {

#ifdef PROFILING_ACTIVE
    glBeginQuery(GL_TIME_ELAPSED, combined_fast_set_viewport_queries_[(n_poses_x_ * (n_poses_y_ - 1)) + j]);
//        glBeginQuery(GL_TIME_ELAPSED, time_query_[SET_VIEWPORT]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][SET_VIEWPORT]++;
#endif

        glViewport(j * n_cols_, 0, n_cols_, n_rows_);

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[SET_VIEWPORT] = stop_time_ - start_time_;
#endif

        for (size_t k = 0; k < object_numbers_.size(); k++) {
            int index = object_numbers_[k];
#ifdef PROFILING_ACTIVE
    glBeginQuery(GL_TIME_ELAPSED, combined_fast_send_model_matrix_queries_[(n_poses_x_ * (n_poses_y_ - 1)) + j]);
//            glBeginQuery(GL_TIME_ELAPSED, time_query_[SEND_MODEL_MATRIX]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][SEND_MODEL_MATRIX]++;
#endif
            model_view_matrix = view_matrix_ * GetModelMatrix(states[n_poses_x_ * (n_poses_y_ - 1) + j][index]);
            glUniformMatrix4fv(model_view_matrix_ID_, 1, GL_FALSE, model_view_matrix.data());

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[SEND_MODEL_MATRIX] = stop_time_ - start_time_;
    glBeginQuery(GL_TIME_ELAPSED, combined_fast_render_queries_[(n_poses_x_ * (n_poses_y_ - 1)) + j]);
//            glBeginQuery(GL_TIME_ELAPSED, time_query_[RENDER]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][RENDER]++;
#endif

            glDrawElements(GL_TRIANGLES, indices_per_object_[index], GL_UNSIGNED_INT, (void*) (start_position_[index] * sizeof(uint)));

#ifdef PROFILING_ACTIVE

    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[RENDER] = stop_time_ - start_time_;
#endif

        }
    }



#ifdef PROFILING_ACTIVE
    glBeginQuery(GL_TIME_ELAPSED, time_query_[DETACH_TEXTURE]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][DETACH_TEXTURE]++;
#endif

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           0,     // 4. tex ID
                           0);

#ifdef PROFILING_ACTIVE
    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[DETACH_TEXTURE] = stop_time_ - start_time_;
    glBeginQuery(GL_TIME_ELAPSED, time_query_[GL_FINISH]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[COMBINED_FAST][GL_FINISH]++;
#endif

    /* TODO should be unnecessary when texture is previously detached from framebuffer..
     * would like to find evidence that this detaching really introduces a synchronization though*/
    glFinish();

#ifdef PROFILING_ACTIVE
    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[GL_FINISH] = stop_time_ - start_time_;
    poses_rendered_[COMBINED_FAST] += n_poses_ * object_numbers_.size();
    map<int, int> factors;
    factors[SET_VIEWPORT] = n_poses_;
    factors[SEND_MODEL_MATRIX] = n_poses_ * object_numbers_.size();
    factors[RENDER] = n_poses_ * object_numbers_.size();
    DisplayTimeObservations(factors, COMBINED_FAST);
#endif

}




void ObjectRasterizer::ReadDepth(vector<vector<int> > &intersect_indices,
                                 vector<vector<float> > &depth,
                                 GLuint pixel_buffer_object,
                                 GLuint framebuffer_texture,
                                 render_type calling_function) {



#ifdef PROFILING_ACTIVE
    glBeginQuery(GL_TIME_ELAPSED, time_query_[FILL_PBO]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[calling_function][FILL_PBO]++;
#endif

    // ===================== COPY VALUES FROM GPU TO CPU ===================== //

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pixel_buffer_object);
    glBindTexture(GL_TEXTURE_2D, framebuffer_texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, 0);

    GLfloat *pixel_depth = (GLfloat*) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

#ifdef PROFILING_ACTIVE
    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[FILL_PBO] = stop_time_ - start_time_;
    glBeginQuery(GL_TIME_ELAPSED, time_query_[GET_DEPTH_VALUES]);
    start_time_ = sf::hf::get_wall_time();
    calls_aggregate_[calling_function][GET_DEPTH_VALUES]++;
#endif


    if (pixel_depth != (GLfloat*) NULL) {
        vector<float> depth_image(n_rows_ * n_cols_ * n_poses_x_ * n_poses_y_, numeric_limits<float>::max());

        int pixels_per_row_texture = n_max_poses_x_ * n_cols_;
        int pixels_per_row_extracted = n_poses_x_ * n_cols_;
        int highest_pixel_per_col = (n_poses_y_ * n_rows_) - 1;

        // reading OpenGL texture into an array on the CPU (inverted rows)
        for (int row = 0; row < n_poses_y_ * n_rows_; row++) {
            int inverted_row = highest_pixel_per_col - row;

            for (int col = 0; col < n_poses_x_ * n_cols_; col++) {
                depth_image[row * pixels_per_row_extracted  + col] = pixel_depth[inverted_row * pixels_per_row_texture + col];
            }
        }


        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

        // subdividing this array into an array per pose
        vector<vector<float> > depth_image_per_pose (n_poses_, vector<float> (n_rows_ * n_cols_, 0));

        for (int pose_y = 0; pose_y < n_poses_y_; pose_y++) {
            for (int pose_x = 0; pose_x < n_poses_x_ && pose_y * n_poses_x_ + pose_x < n_poses_; pose_x++) {
                for (int i = 0; i < n_rows_ * n_cols_; i++) {

                    depth_image_per_pose[pose_y * n_poses_x_ + pose_x][i] = depth_image[(pose_y * n_rows_ + (i / n_cols_)) * pixels_per_row_extracted + pose_x * n_cols_ + (i % n_cols_)];
                }
            }
        }

        // filling the respective values per pose into their indices and depth vectors
        vector<int> tmp_indices;
        vector<float> tmp_depth;

        for (int state = 0; state < n_poses_; state++) {
            for (int i = 0; i < n_rows_ * n_cols_; i++) {
                if (depth_image_per_pose[state][i] != 0) {
                    tmp_indices.push_back(i);
                    tmp_depth.push_back(depth_image_per_pose[state][i]);
                }
            }
            intersect_indices.push_back(tmp_indices);
            depth.push_back(tmp_depth);
            tmp_indices.resize(0);
            tmp_depth.resize(0);
        }




    } else {
        cout << "WARNING: Could not map Pixel Pack Buffer." << endl;
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

#ifdef PROFILING_ACTIVE
    glEndQuery(GL_TIME_ELAPSED);
    stop_time_ = sf::hf::get_wall_time();
    cpu_times_[GET_DEPTH_VALUES] = stop_time_ - start_time_;
#endif
}





void ObjectRasterizer::set_objects(vector<int> object_numbers) {
    // TODO does it copy the vector or set a reference?
    object_numbers_ = object_numbers;
}

void ObjectRasterizer::set_number_of_max_poses(int n_poses) {
//    if (n_poses_ != n_poses) {
        n_max_poses_ = n_poses;
        n_poses_ = n_poses;

//        // TODO max_dimension[0], [1], at the moment they are identical
        float sqrt_poses = sqrt(n_poses_);
        // TODO this can be done smarter. I want to only increment sqrt_poses, if it is not an int, i.e. 10.344 instead of 10)
        if (sqrt_poses * sqrt_poses != n_poses_) sqrt_poses = (int) ceil(sqrt_poses);
        // TODO max_dimension[0], [1], at the moment they are identical
        n_max_poses_x_ = min((int) sqrt_poses, (max_dimension_ / n_cols_));
        int y_poses = n_max_poses_ / n_max_poses_x_;
        if (y_poses * n_max_poses_x_ < n_max_poses_) y_poses++;
        n_max_poses_y_ = min(y_poses, (max_dimension_ / n_rows_));

        n_poses_x_ = n_max_poses_x_;
        n_poses_y_ = n_max_poses_y_;

        ReallocateBuffers();
//    }
}

void ObjectRasterizer::set_number_of_poses(int n_poses) {
    if (n_poses <= n_max_poses_) {
        n_poses_ = n_poses;


//        // TODO max_dimension[0], [1], at the moment they are identical
        float sqrt_poses = sqrt(n_poses_);
        // TODO this can be done smarter. I want to only increment sqrt_poses, if it is not an int, i.e. 10.344 instead of 10)
        if (sqrt_poses * sqrt_poses != n_poses_) sqrt_poses = (int) ceil(sqrt_poses);
        // TODO max_dimension[0], [1], at the moment they are identical
        n_poses_x_ = min((int) sqrt_poses, (max_dimension_ / n_cols_));
        int y_poses = n_poses_ / n_poses_x_;
        if (y_poses * n_poses_x_ < n_poses_) y_poses++;
        n_poses_y_ = min(y_poses, (max_dimension_ / n_rows_));

        if (n_poses_x_ > n_max_poses_x_ || n_poses_y_ > n_max_poses_y_) {
            cout << "WARNING: You tried to evaluate more poses in a row or in a column than was allocated in the beginning."
                 << endl << "Check set_number_of_poses() functions in object_rasterizer.cpp" << endl;
        }

    } else {
        cout << "WARNING: You tried to evaluate more poses than specified by max_poses" << endl;
    }
}

void ObjectRasterizer::set_resolution(const int n_rows,
                                     const int n_cols) {
//    if (n_rows_ != n_rows || n_cols_ != n_cols) {
        n_rows_ = n_rows;
        n_cols_ = n_cols;

        // recalculate n_poses_x_ and n_poses_y_ depending on the resolution
        set_number_of_max_poses(n_max_poses_);


//    }
}

void ObjectRasterizer::ReallocateBuffers() {

    // ======================= REALLOCATE PBO ======================= //

    glBindBuffer(GL_PIXEL_PACK_BUFFER, combined_result_buffer_);
    // TODO PERFORMANCE: try GL_STREAM_READ instead of GL_DYNAMIC_READ
    // the NULL means this buffer is uninitialized, since I only want to copy values back to the CPU that will be written by the GPU
    glBufferData(GL_PIXEL_PACK_BUFFER, n_max_poses_x_* n_cols_ * n_max_poses_y_ * n_rows_ *  sizeof(GLfloat), NULL, GL_STREAM_READ);
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

    cout << "reallocating combined texture..." << endl;

    glBindTexture(GL_TEXTURE_2D, combined_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, n_max_poses_x_ * n_cols_, n_max_poses_y_ * n_rows_, 0, GL_RED, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    cout << "reallocating combined depth texture..." << endl;

    glBindRenderbuffer(GL_RENDERBUFFER, combined_depth_texture_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, n_max_poses_x_* n_cols_, n_max_poses_y_ * n_rows_);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // ======================= ATTACH NEW TEXTURES TO FRAMEBUFFER ======================= //

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,      // 1. fbo target: GL_FRAMEBUFFER
                              GL_DEPTH_ATTACHMENT, // 2. attachment point
                              GL_RENDERBUFFER,     // 3. rbo target: GL_RENDERBUFFER
                              combined_depth_texture_);     // 4. rbo ID
    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           combined_texture_,     // 4. tex ID
                           0);


    checkFramebufferStatus();
    GLenum color_buffers[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, color_buffers);
}






Matrix4f ObjectRasterizer::GetModelMatrix(const vector<float> state) {
    // Model matrix equals the state of the object. Position and Quaternion just have to be expressed as a matrix.
    // note: state = (q.w, q.x, q.y, q.z, v.x, v.y, v.z)
    Matrix4f model_matrix = Matrix4f::Identity();
    Transform<float, 3, Affine, ColMajor> model_matrix_transform;
    model_matrix_transform = Translation3f(state[4], state[5], state[6]);
    model_matrix = model_matrix_transform.matrix();

    Quaternionf qRotation = Quaternionf(state[0], state[1], state[2], state[3]);
    model_matrix.topLeftCorner(3, 3) = qRotation.toRotationMatrix();

    return model_matrix;
}


Matrix4f ObjectRasterizer::GetProjectionMatrix(float n, float f, float l, float r, float t, float b) {
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

void ObjectRasterizer::SetupProjectionMatrix(const Eigen::Matrix3f camera_matrix) {

    // ==================== CONFIGURE IMAGE PARAMETERS ==================== //

    Eigen::Matrix3f camera_matrix_inverse = camera_matrix.inverse();

    Vector3f boundaries_min = camera_matrix_inverse * Vector3f(-0.5, -0.5, 1);
    Vector3f boundaries_max = camera_matrix_inverse * Vector3f(float(n_cols_)-0.5, float(n_rows_)-0.5, 1);

    float near = NEAR_PLANE;
    float far = FAR_PLANE;
    float left = near * boundaries_min(0);
    float right = near * boundaries_max(0);
    float top = -near * boundaries_min(1);
    float bottom = -near * boundaries_max(1);

    // ======================== PROJECTION MATRIX ======================== //

    // Projection Matrix takes into account our view frustum and projects the (3D)-scene onto a 2D image
    projection_matrix_ = GetProjectionMatrix(near, far, left, right, top, bottom);

}

void ObjectRasterizer::SetupViewMatrix() {
    // =========================== VIEW MATRIX =========================== //

    // OpenGL camera is rotated 180 degrees around the X-Axis compared to the Kinect camera
    view_matrix_ = Matrix4f::Identity();
    Transform<float, 3, Affine, ColMajor> view_matrix_transform;
    view_matrix_transform = AngleAxisf(M_PI, Vector3f::UnitX());
    view_matrix_ = view_matrix_transform.matrix();
}

void ObjectRasterizer::DisplayTimeObservations(map<int, int> factors, render_type calling_function) {

#ifdef PROFILING_ACTIVE

    string function;
    switch (calling_function) {
    case COMBINED_FAST: function = "COMBINED_FAST"; break;
    default: function = "default";
    }


    for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {

        if (cpu_times_[i] != -1) {
            GLint available = 0;
            double time_elapsed_s;

            if (calling_function == COMBINED_FAST) {
                if (i == SET_VIEWPORT) {
                    for (int j = 0; j < combined_fast_set_viewport_queries_.size(); j++) {
                        available = 0;
                        while (!available) {
                            glGetQueryObjectiv(combined_fast_set_viewport_queries_[j], GL_QUERY_RESULT_AVAILABLE, &available);
                        }
                        glGetQueryObjectuiv(combined_fast_set_viewport_queries_[j], GL_QUERY_RESULT, &time_elapsed_ns_);
                        time_elapsed_s = time_elapsed_ns_ / (double) 1e9;

                        gpu_times_aggregate_[calling_function][i] += time_elapsed_s;
                    }
                } else if (i == SEND_MODEL_MATRIX) {
                    for (int j = 0; j < combined_fast_send_model_matrix_queries_.size(); j++) {
                        available = 0;
                        while (!available) {
                            glGetQueryObjectiv(combined_fast_send_model_matrix_queries_[j], GL_QUERY_RESULT_AVAILABLE, &available);
                        }
                        glGetQueryObjectuiv(combined_fast_send_model_matrix_queries_[j], GL_QUERY_RESULT, &time_elapsed_ns_);
                        time_elapsed_s = time_elapsed_ns_ / (double) 1e9;

                        gpu_times_aggregate_[calling_function][i] += time_elapsed_s;
                    }
                } else if (i == RENDER) {
                    for (int j = 0; j < combined_fast_render_queries_.size(); j++) {
                        available = 0;
                        while (!available) {
                            glGetQueryObjectiv(combined_fast_render_queries_[j], GL_QUERY_RESULT_AVAILABLE, &available);
                        }
                        glGetQueryObjectuiv(combined_fast_render_queries_[j], GL_QUERY_RESULT, &time_elapsed_ns_);
                        time_elapsed_s = time_elapsed_ns_ / (double) 1e9;

                        gpu_times_aggregate_[calling_function][i] += time_elapsed_s;
                    }
                } else {
                    available = 0;
                    while (!available) {
                        glGetQueryObjectiv(time_query_[i], GL_QUERY_RESULT_AVAILABLE, &available);
                    }
                    glGetQueryObjectuiv(time_query_[i], GL_QUERY_RESULT, &time_elapsed_ns_);
                    time_elapsed_s = time_elapsed_ns_ / (double) 1e9;
                    gpu_times_aggregate_[calling_function][i] += time_elapsed_s;
                }
            } else {
                available = 0;
                while (!available) {
                    glGetQueryObjectiv(time_query_[i], GL_QUERY_RESULT_AVAILABLE, &available);
                }
                glGetQueryObjectuiv(time_query_[i], GL_QUERY_RESULT, &time_elapsed_ns_);
                time_elapsed_s = time_elapsed_ns_ / (double) 1e9;
                gpu_times_aggregate_[calling_function][i] += time_elapsed_s;
            }

            cpu_times_aggregate_[calling_function][i] += cpu_times_[i];

            cpu_times_[i] = -1;
        }
    }

    if (poses_rendered_[calling_function] == n_poses_ && initial_[calling_function]) {
        cout << "reset after initial poses" << endl;
        initial_[calling_function] = false;
        for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {
            if (calls_aggregate_[calling_function][i] != 0) {
                cpu_times_aggregate_[calling_function][i] = 0;
                gpu_times_aggregate_[calling_function][i] = 0;
            }
            calls_aggregate_[calling_function][i] = 0;
        }

        poses_rendered_[calling_function] = 0;
    }

    const int ROUNDS = 500;
    if (poses_rendered_[calling_function] >= n_poses_ * ROUNDS) {

        cout << endl << "Time observations for " << function << " averaged over " << poses_rendered_[calling_function] << " poses:" << endl << endl;


        double total_time = 0;

        for (int i = 0; i < TIME_MEASUREMENTS_COUNT; i++) {
            if (calls_aggregate_[calling_function][i] != 0) {

                total_time += max(gpu_times_aggregate_[calling_function][i], cpu_times_aggregate_[calling_function][i]);

                int n_calls = calls_aggregate_[calling_function][i];
                cout << getTextForEnum(i) << ":    "
                     << "\t (GPU) " << gpu_times_aggregate_[calling_function][i] / n_calls
                     << "\t (CPU) " << cpu_times_aggregate_[calling_function][i] / n_calls
                     << "\t (x" << n_calls /* / ROUNDS */ << ") " << max(gpu_times_aggregate_[calling_function][i], cpu_times_aggregate_[calling_function][i]) /* / ROUNDS*/
                     << "\t TOTAL_TIME: " << total_time /* / ROUNDS*/ << ", calls_aggregate: " << n_calls << endl;
                cpu_times_aggregate_[calling_function][i] = 0;
                gpu_times_aggregate_[calling_function][i] = 0;
            }

            calls_aggregate_[calling_function][i] = 0;
        }
        cout << "TOTAL_TIME: \t" << total_time / poses_rendered_[calling_function] << "\t(x" << poses_rendered_[calling_function] << ") " << total_time / ROUNDS << endl << endl;

        poses_rendered_[calling_function] = 0;
    }

#endif

}

GLuint ObjectRasterizer::get_combined_texture() {
    return combined_texture_;
}

int ObjectRasterizer::get_n_poses_x() {
    return n_poses_x_;
}



void ObjectRasterizer::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                        std::vector<std::vector<float> > &depth) {

    // ===================== ATTACH TEXTURE TO FRAMEBUFFER ================ //

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           combined_texture_,     // 4. tex ID
                           0);

    // ===================== TRANSFERS DEPTH VALUES TO CPU == SLOW!!! ================ //

    ReadDepth(intersect_indices, depth, combined_result_buffer_, combined_texture_, COMBINED_FAST);

//    vector<int> intersect_indices_all;
//    vector<float> depth_all;

//    int width = n_cols_ * n_poses_x_;
//    int index, index_x, index_y, pose_x, pose_y, local_index, pose;

//    vector<int> intersect_indices_tmp[n_poses_];
//    vector<float> depth_tmp[n_poses_];

////    int n_cols_old = n_cols_;
////    int n_rows_old = n_rows_;
////    n_cols_ = n_poses_x_ * n_cols_;
////    n_rows_ = n_poses_y_ * n_rows_;
//    ReadDepth(intersect_indices_all, depth_all, combined_result_buffer_, combined_texture_, COMBINED_FAST);
////    n_cols_ = n_cols_old;
////    n_rows_ = n_rows_old;

//    cout << "intersect_indices_all size: " << intersect_indices_all.size() << endl;

//    for (size_t i = 0; i < intersect_indices_all.size(); i++) {
//        index = intersect_indices_all[i];
//        index_x = index % width;
//        index_y = index / width;
//        pose_x = index_x / n_cols_;
//        pose_y = index_y / n_rows_;

//        local_index = (index_x % n_cols_) + (index_y % n_rows_) * n_cols_;
//        pose = pose_y * n_poses_x_ + pose_x;

//        intersect_indices_tmp[pose].push_back(local_index);
//        depth_tmp[pose].push_back(depth_all[i]);

//    }

//    for (int i = 0; i < n_poses_; i++) {
//        intersect_indices.push_back(intersect_indices_tmp[i]);
//        depth.push_back(depth_tmp[i]);
//        cout << "rasterizer indices: [" << i << "]: " << intersect_indices_tmp[i].size() << endl;
//    }


    // ===================== DETACH TEXTURE FROM FRAMEBUFFER ================ //

    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           0,     // 4. tex ID
                           0);

}


string ObjectRasterizer::getTextForEnum( int enumVal ) {
    return enum_strings_[enumVal];
}


void ObjectRasterizer::checkGLErrors(const char *label) {
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

bool ObjectRasterizer::checkFramebufferStatus() {
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
    cout << "clean up OpenGL.." << endl;

    glDisableVertexAttribArray(0);
    glDeleteVertexArrays(1, &vertex_array_);

    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteBuffers(1, &index_buffer_);
    glDeleteBuffers(1, &combined_result_buffer_);

    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteTextures(1, &combined_texture_);
    glDeleteRenderbuffers(1, &combined_depth_texture_);

    glDeleteProgram(shader_ID_);
    glDeleteQueries(21, time_query_);
#ifdef PROFILING_ON
    glDeleteQueries(n_poses_, &combined_fast_set_viewport_queries_[0]);
    glGenQueries(n_poses_, &combined_fast_send_model_matrix_queries_[0]);
    glGenQueries(n_poses_, &combined_fast_render_queries_[0]);
#endif

}


