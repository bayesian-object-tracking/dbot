#ifndef OBJECT_RASTERIZER_HPP
#define OBJECT_RASTERIZER_HPP

#include <vector>
#include <Eigen/Dense>
#include "GL/glew.h"
#include <map>


/// renders the object using openGL rasterization
/** The objects that should be rendered have to be passed in the constructor and can then be rendered
  * in different poses with the Render() function. The resulting depth values are stored in a texture
  * whose values can be obtained with get_depth_values(). Alternatively, get_combined_texture() returns
  * the ID of the texture for mapping it into the CUDA context.
  */
class ObjectRasterizer
{
public:
    /// constructor which takes the vertices and indices that describe the objects as input
    /**
      * @param[in] vertices [object_nr][vertex_nr] = {x, y, z}. This list should contain 3-dimensional
      * vectors that specify the corners of the triangles the object meshes consists of.
      * @param[in] indices [object_nr][index_nr][0 - 2] = {index}. This list should contain the indices
      * that index the vertices list and tell us which vertices to connect to a triangle (every group of 3).
      * For each object, the indices should be in the range of [0, nr_vertices - 1].
      */
    ObjectRasterizer(const std::vector<std::vector<Eigen::Vector3f> >vertices,
                     const std::vector<std::vector<std::vector<int> > >indices);

    /// constructor with no arguments, should not be used
    ObjectRasterizer();

    /// destructor which deletes the buffers and programs used by openGL
    ~ObjectRasterizer();


    /// prepare openGL for a function call to Render()
    /** Has to be called once before calling Render() (which can then be
      * called frequently). It sets the shaders to the correct program and computes
      * the perspective projection matrix from the camera parameters. The near/far planes
      * are set by the constants NEAR_PLANE and FAR_PLANE in the header.
      * @param[in] camera_matrix matrix of the intrinsic parameters of the camera
      */
    void PrepareRender(const Eigen::Matrix3f camera_matrix);



    /// render the objects in all given states and return the depth for all relevant pixels of each rendered object
    /** This function renders all poses (of all objects) into one large texture. Reading back the depth values
      * is a relatively slow process, so this function should mainly be used for debugging. If you are using
      * CUDA to further process the depth values, please use the other Render() function.
      * @param[in] states [pose_nr][object_nr][0 - 6] = {qw, qx, qy, qz, tx, ty, tz}. This should contain the quaternion
      * and the translation for each object per pose.
      * @param[in,out] intersect_indices [pose_nr][0 - nr_relevant_pixels] = {pixel_nr}. This list should be empty when passed
      * to the function. Afterwards, it will contain the pixel number of all pixels that were rendered to, per pose.
      * @param[in,out] depth [pose_nr][0 - nr_relevant_pixels] = {depth_value}. This list should be empty when passed to the function.
      * Afterwards, it will contain the depth value of all pixels that were rendered to, per pose.
      */
    void Render(const std::vector<std::vector<std::vector<float> > > states,
                            std::vector<std::vector<int> > &intersect_indices,
                            std::vector<std::vector<float> > &depth);


    /// render the objects in all given states into a texture that can then be accessed by CUDA
    /** This function renders all poses (of all objects) into one large texture, which can then be mapped into the CUDA
      * context. To get the ID of the texture, call get_texture_ID().
      * @param[in] states [pose_nr][object_nr][0 - 6] = {qw, qx, qy, qz, tx, ty, tz}. This should contain the quaternion
      * and the translation for each object per pose.
      */
    void Render(const std::vector<std::vector<std::vector<float> > > states);

    /// sets the objects that should be rendered.
    /** This function only needs to be called if any objects initially passed in the constructor should be left out when rendering.
      * @param[in] object_numbers [0 - nr_objects] = {object_nr}. This list should contain the indices of all objects that
      * should be rendered when calling Render().
      */
    void set_objects(std::vector<int> object_numbers);

    /// set a new resolution
    /** This function reallocates the framebuffer textures.
      * @param[in] n_rows the height of the image
      * @param[in] n_cols the width of the image
      */
    void set_resolution(const int n_rows,
                       const int n_cols);

    void set_number_of_max_poses(int n_poses);

    void set_number_of_poses(int n_poses);


    GLuint get_combined_texture();
    int get_n_poses_x();
    int get_n_renders();
    void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                          std::vector<std::vector<float> > &depth);

private:
    static const float NEAR_PLANE = 0.4f; // Kinect does not see anything closer than 0.3 meters
    static const float FAR_PLANE = 4.0f; // Kinect does not see anything further away than 7 meters
    static const int WINDOW_WIDTH = 80;  // default values if not specified
    static const int WINDOW_HEIGHT = 60; // default values if not specified
    static const int TIME_MEASUREMENTS_COUNT = 16;
    static const int RENDER_TYPE_COUNT = 1;
    enum time_measurement { SEND_MATRICES, CLEAR_SCREEN, RENDER, FILL_PBO, MAP_PBO, GET_DEPTH_VALUES,
                       SEND_CONSTANTS, SEND_TEXTURES, SUM_LIKELIHOODS, UPDATE_TEXTURES,
                       ATTACH_TEXTURE, ALLOCATE, SET_VIEWPORT, SEND_MODEL_MATRIX, DETACH_TEXTURE,
                       GL_FINISH};
    enum render_type { COMBINED_FAST };
    std::vector<std::string> enum_strings_;
    std::vector<std::vector<double> > cpu_times_aggregate_;
    std::vector<std::vector<double> > gpu_times_aggregate_;
    std::vector<int> poses_rendered_;
    std::vector<std::vector<int> > calls_aggregate_;
    std::vector<GLuint> combined_fast_set_viewport_queries_;
    std::vector<GLuint> combined_fast_send_model_matrix_queries_;
    std::vector<GLuint> combined_fast_render_queries_;

    bool initial_[RENDER_TYPE_COUNT];

    // GPU constraints

    GLint max_dimension_;

    // values initialized to WINDOW_WIDTH, WINDOW_HEIGHT in constructor. May be changed by user with set_resolution().
    int n_rows_;
    int n_cols_;

    // number of poses to render
    int n_poses_;
    int n_poses_x_;
    int n_poses_y_;

    // number of maximum poses
    int n_max_poses_;
    int n_max_poses_x_;
    int n_max_poses_y_;

    // needed for openGL time measurement
    GLuint time_query_[TIME_MEASUREMENTS_COUNT];
    unsigned int time_elapsed_ns_;

    // needed for cpu time measurement
    double start_time_;
    double stop_time_;
    double cpu_times_[TIME_MEASUREMENTS_COUNT];


    // the paths to the respective shaders
    std::string vertex_shader_path_;
    std::string fragment_shader_path_;

    std::vector<uint> indices_list_;
    // a list of all vertices of all objects
    std::vector<float> vertices_list_;
    std::vector<int> vertices_per_object_;
    std::vector<int> indices_per_object_;
    std::vector<int> start_position_;
    // contains a list of object indices which should be rendered
    std::vector<int> object_numbers_;

    // matrices to transform vertices into image space
    Eigen::Matrix4f projection_matrix_;
    Eigen::Matrix4f view_matrix_;

    // shader program ID and uniform IDs to pass variables to it
    GLuint shader_ID_;
    GLuint model_view_matrix_ID_;    // ID to which we pass the modelview matrix
    GLuint projection_matrix_ID_;    // ID to which we pass the projection matrix

    // VAO, VBO and element array are needed to store the object meshes
    GLuint vertex_array_;   // The vertex array contains the vertex and index buffers
    GLuint vertex_buffer_;    // contains the vertices of the object meshes passed in the constructor
    GLuint index_buffer_;     //  contains the indices of the object meshes passed in the constructor

    // PBO for copying results to CPU
    GLuint combined_result_buffer_;

    // custom framebuffer and its textures for color and depth
    GLuint framebuffer_;
    GLuint combined_texture_;
    GLuint combined_depth_texture_;



    // ====================== PRIVATE FUNCTIONS ====================== //


    std::string getTextForEnum( int enumVal );
    void checkGLErrors(const char *label);
    bool checkFramebufferStatus();
    void ReadDepth(std::vector<std::vector<int> > &intersect_indices,
                   std::vector<std::vector<float> > &depth,
                   GLuint pixel_buffer_object,
                   GLuint framebuffer_texture,
                   render_type calling_function);

    Eigen::Matrix4f GetModelMatrix(const std::vector<float> state);
    Eigen::Matrix4f GetProjectionMatrix(float n, float f, float l, float r, float t, float b);

    void DisplayTimeMeasurements(std::map<int, int> factors, render_type calling_function);
    void ReallocateBuffers();
    void SetupViewMatrix();
    void SetupProjectionMatrix(const Eigen::Matrix3f camera_matrix);

};

#endif // OBJECT_RASTERIZER_HPP
