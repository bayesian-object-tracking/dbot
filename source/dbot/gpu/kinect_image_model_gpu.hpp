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

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
 * Probabilistic Object Tracking using a Range Camera
 * IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013
 * http://arxiv.org/abs/1505.00241
 *
 */

/**
 * \file kinect_image_model_gpu.hpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date November 2015
 */

#pragma once

#define PROFILING_ACTIVE
//#define OPTIMIZE_NR_THREADS

#include <vector>
#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"
#include "Eigen/Core"

#include <dbot/pose/pose_vector.hpp>
#include <dbot/model/rao_blackwell_sensor.hpp>
#include <dbot/pose/free_floating_rigid_bodies_state.hpp>

#include <dbot/traits.hpp>
#include <dbot/gpu/object_rasterizer.hpp>
#include <dbot/gpu/cuda_likelihood_evaluator.hpp>
#include <dbot/gpu/buffer_configuration.hpp>

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_gl_interop.h>

#include <dbot/helper_functions.hpp>
#include <fl/util/profiling.hpp>


namespace dbot
{
// Forward declarations
template <typename State>
class KinectImageModelGPU;

namespace internal
{
/**
 * ImageSensorGPU distribution traits specialization
 * \internal
 */
template <typename State>
struct Traits<KinectImageModelGPU<State>>
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef RbSensor<State> Base;

    typedef typename Eigen::Matrix<Scalar, 3, 3> CameraMatrix;
};
}



/**
 * \class ImageSensorGPU
 *
 * \ingroup distributions
 * \ingroup sensors
 */
template <typename State>
class KinectImageModelGPU
    : public internal::Traits<KinectImageModelGPU<State>>::Base
{
public:
    typedef internal::Traits<KinectImageModelGPU<State>> Traits;

    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::Observation Observation;
    typedef typename Traits::CameraMatrix CameraMatrix;

    typedef typename Traits::Base::StateArray StateArray;
    typedef typename Traits::Base::RealArray RealArray;
    typedef typename Traits::Base::IntArray IntArray;

    typedef typename Eigen::Transform<fl::Real, 3, Eigen::Affine> Affine;

    /**
     * \brief constructor which takes relevant constants and initializes the graphics
     * pipeline with them
     *
     * \param [in] camera_matrix
     * 		matrix of the intrinsic parameters of the camera
     * \param [in] nr_rows
     * 		the number of rows in one sensor image (vertical resolution)
     * \param [in] nr_cols
     * 		the number of columns in one sensor image (horizontal
     * 		resolution)
     * \param [in] max_sample_count
     * 		the maximum number of poses that will be rendered per object
     *      in one frame.
     * This is needed to allocate the necessary memory on the GPU.
     * \param [in] vertices
     * 		[object_nr][vertex_nr] = {x, y, z}. This list should
     * 		contain, for each object, a list of 3-dimensional vectors
     * 		that specify the corners of the triangles of the object mesh
     * \param [in] indices [object_nr][index_nr][0 - 2] = {index}. This list
     * 		should contain the indices
     * 		that index the vertices list and tell us which vertices to connect
     * to
     * a
     * 		triangle (every group of 3).
     * 		For each object, the indices should be in the range of [0,
     * nr_vertices
     * - 1].
     * \param [in] vertex_shader_path path to the vertex shader
     * \param [in] fragment_shader_path path to the fragment shader
     * \param [in] initial_occlusion_prob the initial probability for each pixel
     * of being occluded, meaning
     * that something occludes the object in this pixel
     * \param [in] delta_time the time between two frames in seconds. This
     * should
     * correspond to the rate at which the
     * sensor provides new images of the scene.
     * \param [in] p_occluded_visible the probability of a pixel changing from
     * being occluded to being visible
     * from one frame to the next
     * \param [in] p_occluded_occluded the probability of a pixel staying
     * occluded from one frame to the next
     * \param [in] tail_weight the probability of a faulty measurement
     * \param [in] model_sigma the uncertainty in the 3D model of the object
     * \param [in] sigma_factor the standard deviation of the measurement noise
     * at a distance of 1m to the camera
     * \param [in] max_depth maximum value which can be measured by the depth
     * sensor
     * \param [in] exponential_rate the rate of the exponential distribution
     * that
     * models the probability of a measurement coming from an unknown object
     */
    KinectImageModelGPU(
        const CameraMatrix& camera_matrix,
        const size_t& nr_rows,
        const size_t& nr_cols,
        const size_t& max_sample_count,
        const std::vector<std::vector<Eigen::Vector3d>> vertices_double,
        const std::vector<std::vector<std::vector<int>>> indices,
        const std::shared_ptr<ShaderProvider>& shader_provider,
        const bool adapt_to_constraints = false,
        const bool optimize_nr_threads = false,
        const Scalar& initial_occlusion_prob = 0.1d,
        const double& delta_time = 0.033d,
        const float p_occluded_visible = 0.1f,
        const float p_occluded_occluded = 0.7f,
        const float tail_weight = 0.01f,
        const float model_sigma = 0.003f,
        const float sigma_factor = 0.0014247f,
        const float max_depth = 6.0f,
        const float exponential_rate = -log(0.5f))
        : camera_matrix_(camera_matrix),
          nr_rows_(nr_rows),
          nr_cols_(nr_cols),
          nr_max_poses_(max_sample_count),
          indices_(indices),
          optimize_nr_threads_(optimize_nr_threads),
          initial_occlusion_prob_(initial_occlusion_prob),
          tail_weight_(tail_weight),
          model_sigma_(model_sigma),
          sigma_factor_(sigma_factor),
          max_depth_(max_depth),
          exponential_rate_(exponential_rate),
          nr_poses_(max_sample_count),
          observations_set_(false),
          resource_registered_(false),
          observation_time_(0),
          Traits::Base(delta_time)
    {

        // set constants
        this->default_poses_.recount(vertices_double.size());
        this->default_poses_.setZero();

        // convert doubles to floats
        vertices_.resize(vertices_double.size());
        for (size_t object_index = 0; object_index < vertices_.size();
             object_index++)
        {
            vertices_[object_index].resize(
                vertices_double[object_index].size());
            for (size_t vertex_index = 0;
                 vertex_index < vertices_[object_index].size();
                 vertex_index++)
                vertices_[object_index][vertex_index] =
                    vertices_double[object_index][vertex_index].cast<float>();
        }

        // initialize opengl and cuda
        opengl_ = boost::shared_ptr<ObjectRasterizer>(
            new ObjectRasterizer(vertices_,
                                 indices_,
                                 shader_provider,
                                 camera_matrix_.cast<float>(),
                                 nr_rows_,
                                 nr_cols_,
                                 0.4,
                                 4));


        cuda_ = boost::shared_ptr<CudaEvaluator>(new CudaEvaluator(nr_rows_, nr_cols_));

        cuda_->init(initial_occlusion_prob_,
                    p_occluded_occluded,
                    p_occluded_visible,
                    tail_weight_,
                    model_sigma_,
                    sigma_factor_,
                    max_depth_,
                    exponential_rate_);

        bufferConfig_ = boost::shared_ptr<BufferConfiguration>(
                            new BufferConfiguration(opengl_, cuda_,
                                nr_max_poses_, nr_rows_, nr_cols_));

        bufferConfig_->set_adapt_to_constraints(adapt_to_constraints);

        // allocates memory and sets the dimensions of how many poses will be
        // rendered per row and per column in the texture
        int tmp_max_nr_poses;
        if (bufferConfig_->allocate_memory(nr_max_poses_, tmp_max_nr_poses)) {
                nr_max_poses_ = tmp_max_nr_poses;
        } else {
            exit(-1);
        }


        // sets the resolution and rearranges the pose grid accordingly
        if (bufferConfig_->set_resolution(nr_rows_, nr_cols_, tmp_max_nr_poses)) {
            nr_max_poses_ = tmp_max_nr_poses;
        } else {
            exit(-1);
        }

        register_resource();

        reset();



        count_ = 0;

        occlusion_probs_.resize(nr_rows_ * nr_cols_);

#ifdef PROFILING_ACTIVE
        for (int i = 0; i < NR_SUBTASKS_TO_MEASURE; i++)
        {
            time_[i] = 0;
        }
        strings_for_subtasks_[SET_OCCLUSION_INDICES] =
            "Setting occlusion indices";
        strings_for_subtasks_[CONVERTING_STATE_FORMAT] =
            "Converting state format";
        strings_for_subtasks_[RENDERING] = "Rendering step";
        strings_for_subtasks_[MAPPING] =
            "Mapping the texture from OpenGL to CUDA";
        strings_for_subtasks_[WEIGHTING] = "Weighting step";
        strings_for_subtasks_[UNMAPPING] =
            "Unmapping the texture and reconverting the likelihoods";
#endif

        optimize_nr_threads_ = false;

#ifdef OPTIMIZE_NR_THREADS
        set_optimization_of_thread_nr(true);
#endif
        if (optimize_nr_threads_)
        {
            max_nr_threads_ = cuda_->get_max_nr_threads();
            warp_size_ = cuda_->get_warp_size();
            nr_threads_ = warp_size_;
            best_time_ = std::numeric_limits<double>::infinity();
            best_nr_threads_ = nr_threads_;
            average_time_ = 0;
        }

        optimization_runs_ = 0;
    }





    /**
     * \brief computes the loglikelihoods for the given states
     * Make sure the observation image was set previously, as it is used for
     * comparison.
     * \param [in] deltas the states which should be evaluated
     * \param [in][out] occlusion_indices for each state, this should contain
     * the index into the occlusion array where the corresponding occlusion
     * probabilities per pixel are stored
     * \param [in] update_occlusions whether or not the occlusions should be
     * updated in this evaluation step
     * \return the loglikelihoods for the given states
     */
    RealArray loglikes(const StateArray& deltas,
                       IntArray& occlusion_indices,
                       const bool& update_occlusions = false)
    {
#ifdef PROFILING_ACTIVE
        if (!optimize_nr_threads_ && optimization_runs_ != count_)
        {
            time_before_ = dbot::hf::get_wall_time();
        }
#endif

        if (!observations_set_)
        {
            std::cout << "GPU: observations not set" << std::endl;
            exit(-1);
        }

        nr_poses_ = deltas.size();
        std::vector<float> flog_likelihoods(nr_poses_, 0);

        int tmp_nr_poses;
        if (bufferConfig_->set_nr_of_poses(nr_poses_, tmp_nr_poses)) {
            nr_poses_ = tmp_nr_poses;
        } else {
            exit(-1);
        }


        // transform occlusion indices from size_t to int
        std::vector<int> occlusion_indices_transformed(occlusion_indices.size(),
                                                       0);
        for (size_t i = 0; i < occlusion_indices.size(); i++)
        {
            occlusion_indices_transformed[i] = (int)occlusion_indices[i];
        }

        // copy occlusion indices to GPU
        cuda_->set_occlusion_indices(occlusion_indices_transformed.data(),
                                     occlusion_indices.size());

#ifdef PROFILING_ACTIVE
        store_time(SET_OCCLUSION_INDICES);
#endif

        int nr_objects = vertices_.size();

        std::vector<std::vector<Eigen::Matrix4f>> poses(
            nr_poses_, std::vector<Eigen::Matrix4f>(nr_objects));

        for (size_t i_state = 0; i_state < size_t(nr_poses_); i_state++)
        {
            for (size_t i_obj = 0; i_obj < nr_objects; i_obj++)
            {
                auto pose_0 = this->default_poses_.component(i_obj);
                auto delta = deltas[i_state].component(i_obj);

                osr::PoseVector pose;

                /// \todo: this should be done through the the apply_delta
                /// function
                pose.position() = pose_0.orientation().rotation_matrix()
                        * delta.position() + pose_0.position();
                pose.orientation() = pose_0.orientation() * delta.orientation();

                poses[i_state][i_obj] = pose.homogeneous().cast<float>();
            }
        }

#ifdef PROFILING_ACTIVE
        store_time(CONVERTING_STATE_FORMAT);
#endif

        opengl_->render(poses);


#ifdef PROFILING_ACTIVE
        store_time(RENDERING);
#endif

        cudaGraphicsMapResources(1, &texture_resource_, 0);
        cudaGraphicsSubResourceGetMappedArray(
            &texture_array_, texture_resource_, 0, 0);
        cuda_->map_texture_to_texture_array(texture_array_);

#ifdef PROFILING_ACTIVE
        store_time(MAPPING);
#endif

        if (optimize_nr_threads_)
        {
            if (nr_threads_ <= max_nr_threads_)
            {
                before_weighting_ = dbot::hf::get_wall_time();

                int tmp_nr_threads;
                if (bufferConfig_->set_number_of_threads(nr_threads_, tmp_nr_threads)) {
                    nr_threads_ = tmp_nr_threads;
                } else {
                    exit(-1);
                }

                optimization_runs_++;
            }
            else
            {
                nr_threads_ = best_nr_threads_;
                optimize_nr_threads_ = false;
                std::cout << std::endl
                          << "Best #threads: " << nr_threads_ << std::endl
                          << std::endl;
            }
        }

        cuda_->weigh_poses(update_occlusions, flog_likelihoods);

        if (optimize_nr_threads_)
        {
            if (nr_threads_ <= max_nr_threads_)
            {
                after_weighting_ = dbot::hf::get_wall_time();
                double time = after_weighting_ - before_weighting_;
                average_time_ += time;

                if (count_ % NR_ROUNDS_PER_SETTING_ ==
                    NR_ROUNDS_PER_SETTING_ - 1)
                {
                    average_time_ /= NR_ROUNDS_PER_SETTING_;
                    if (average_time_ < best_time_)
                    {
                        best_time_ = average_time_;
                        best_nr_threads_ = nr_threads_;
                    }

                    nr_threads_ += warp_size_;
                    average_time_ = 0;
                }
            }
        }

#ifdef PROFILING_ACTIVE
        store_time(WEIGHTING);
#endif

        cudaGraphicsUnmapResources(1, &texture_resource_, 0);

        if (update_occlusions)
        {
            for (size_t i_state = 0; i_state < occlusion_indices.size();
                 i_state++)
                occlusion_indices[i_state] = i_state;
        }

        // convert
        RealArray log_likelihoods(flog_likelihoods.size());
        for (size_t i = 0; i < flog_likelihoods.size(); i++)
            log_likelihoods[i] = flog_likelihoods[i];

#ifdef PROFILING_ACTIVE
        store_time(UNMAPPING);
#endif

        count_++;
        return log_likelihoods;
    }

    /**
     * \brief Sets the observation image that should be used for comparison in the
     * next evaluation step
     *
     * \param [in] image the image obtained from the camera
     */
    void set_observation(const Observation& image)
    {
        std::vector<float> std_measurement(image.size());

        for (int i = 0; i < image.size(); ++i)
        {
            std_measurement[i] = image(i);
        }

        observation_time_ += this->delta_time_;

        cuda_->set_observations(std_measurement.data(), observation_time_);
        observations_set_ = true;
    }

    /** \brief Resets the occlusion probabilities and observation time */
    virtual void reset()
    {
        float default_occlusion_probability = initial_occlusion_prob_;
        int array_size = nr_rows_ * nr_cols_ * nr_max_poses_;

        std::vector<float> occlusion_probabilities(
            array_size, default_occlusion_probability);

        cuda_->set_occlusion_probabilities(occlusion_probabilities.data(),
                                           array_size);
//        cuda_->set_occlusion_probabilities(NULL);

        observation_time_ = 0;
    }

    /** activates automatic optimization of the number of threads */
    void set_optimization_of_thread_nr(bool shouldOptimize)
    {
        optimize_nr_threads_ = shouldOptimize;
    }


    /**
     * \brief Returns the occlusion probabilities for each pixel for a given state
     *
     * \param [in] index the index into the state array of the state you are
     * interested in
     * \return an Eigen Matrix containing the occlusion probabilities for each
     * pixel for the given state
     */
    const Eigen::Map<Observation> get_occlusions(size_t index) const
    {
        std::vector<float> occlusion_probs =
            cuda_->get_occlusion_probabilities((int)index);

        // convert values to doubles and put them into an Eigen Matrix
        std::vector<double> occlusion_probs_double(occlusion_probs.begin(),
                                                   occlusion_probs.end());
        Eigen::Map<Observation> occlusion_probs_matrix(
            &occlusion_probs_double[0], nr_rows_, nr_cols_);

        return occlusion_probs_matrix;
    }

    /**
     * \brief Returns the depth values of the rendered states
     *
     * \return an Eigen Matrix containing the depth values per pixel, stored in
     * a 1D array denoting the respective pose
      */
    std::vector<Eigen::Map<Observation>> get_range_image()
    {
        std::vector<std::vector<float>> depth_values_raw =
            opengl_->get_depth_values(nr_poses_);

        // convert depth values to doubles
        std::vector<std::vector<double>> depth_values_raw_double;
        for (int i = 0; i < depth_values_raw.size(); i++)
        {
            depth_values_raw_double.push_back(std::vector<double>(
                depth_values_raw[i].begin(), depth_values_raw[i].end()));
        }

        // map values into an Eigen Matrix
        std::vector<Eigen::Map<Observation>> depth_values;

        for (int i = 0; i < depth_values_raw.size(); i++)
        {
            Eigen::Map<Observation> tmp(
                &depth_values_raw_double[i][0], nr_rows_, nr_cols_);
            depth_values.push_back(tmp);
        }

        return depth_values;
    }




    /** \brief The destructor. If profiling is activated, the time measurements are
     * processed and printed out here */
    virtual ~KinectImageModelGPU() noexcept
    {

        unregister_resource();

#ifdef PROFILING_ACTIVE

        count_ -= optimization_runs_;
        if (count_ > 0)
        {
            std::cout << std::endl
                      << "Time measurements for the different steps of the "
                         "evaluation process averaged over "
                      << count_ << " evaluation calls:" << std::endl
                      << std::endl;

            double total_time_per_evaluation = 0;
            for (int i = 0; i < NR_SUBTASKS_TO_MEASURE; i++)
            {
                total_time_per_evaluation += time_[i];
            }
            total_time_per_evaluation /= count_;

            for (int i = 0; i < NR_SUBTASKS_TO_MEASURE; i++)
            {
                std::cout << strings_for_subtasks_[i] << ": \t"
                          << time_[i] / count_ << " s \t "
                          << std::setprecision(1)
                          << time_[i] / count_ * 100 / total_time_per_evaluation
                          << " %" << std::setprecision(9) << std::endl;
            }
            std::cout << "Total time per evaluation call: "
                      << total_time_per_evaluation << std::endl;

            if (optimization_runs_ > 0)
            {
                std::cout << std::endl
                          << "The best number of threads for this setup was "
                             "estimated to be "
                          << nr_threads_ << std::endl;
            }
        }
        else
        {
            std::cout << "No measurement for the different steps of the "
                         "evaluation was taken. "
                      << "Most likely, you need to deactivate the optimization "
                         "for #threads or"
                      << " let the program run for a longer period of time."
                      << std::endl;

            if (optimize_nr_threads_ == true)
            {
                std::cout << std::endl
                          << "The best #threads could not be computed, because "
                             "the runtime of the program was too short."
                          << " Please allow the program to run a bit longer to "
                             "find the best #threads."
                          << std::endl;
            }
        }
#endif
    }

private:
    const Eigen::Matrix3d camera_matrix_;
    int nr_rows_;
    int nr_cols_;
    int nr_max_poses_;

    void store_time(int task)
    {
        if (!optimize_nr_threads_ && optimization_runs_ != count_)
        {
            time_after_ = dbot::hf::get_wall_time();
            time_[task] += time_after_ - time_before_;
            time_before_ = time_after_;
        }
    }

    void check_cuda_error(const char* msg)
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(
                stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void register_resource()
    {
        if (!resource_registered_)
        {
            opengl_texture_ = opengl_->get_framebuffer_texture();
            cudaGraphicsGLRegisterImage(&texture_resource_,
                                        opengl_texture_,
                                        GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsReadOnly);
            check_cuda_error("cudaGraphicsGLRegisterImage)");
            resource_registered_ = true;
        }
    }

    void unregister_resource()
    {
        if (resource_registered_)
        {
            cudaGraphicsUnregisterResource(texture_resource_);
            check_cuda_error("cudaGraphicsUnregisterResource");
            resource_registered_ = false;
        }
    }

    // OpenGL handle and input
    boost::shared_ptr<ObjectRasterizer> opengl_;
    std::vector<std::vector<Eigen::Vector3f>> vertices_;
    std::vector<std::vector<std::vector<int>>> indices_;
    std::string vertex_shader_path_;
    std::string fragment_shader_path_;

    // CUDA handle
    boost::shared_ptr<CudaEvaluator> cuda_;

    // Buffer configuration handle
    boost::shared_ptr<BufferConfiguration> bufferConfig_;

    // constants for likelihood evaluation in CUDA kernel
    double observation_time_;
    float initial_occlusion_prob_;
    float tail_weight_;
    float model_sigma_;
    float sigma_factor_;
    float max_depth_;
    float exponential_rate_;
    std::vector<float> occlusion_probs_;

    // amount of poses and pose distribution in the OpenGL texture
    int nr_poses_;
    int nr_poses_per_row_;
    int nr_poses_per_column_;

    // Shared resource between OpenGL and CUDA
    GLuint opengl_texture_;
    cudaGraphicsResource* texture_resource_;
    cudaArray_t texture_array_;

    // booleans to ensure correct usage of function calls
    bool observations_set_, resource_registered_;

    // used for time observations
    static const int NR_SUBTASKS_TO_MEASURE = 6;
    enum subtasks_to_measure
    {
        SET_OCCLUSION_INDICES,
        CONVERTING_STATE_FORMAT,
        RENDERING,
        MAPPING,
        WEIGHTING,
        UNMAPPING
    };
    double time_[NR_SUBTASKS_TO_MEASURE];
    std::string strings_for_subtasks_[NR_SUBTASKS_TO_MEASURE];
    double time_before_, time_after_;
    int count_;

    // variables for the optimization runs
    int nr_threads_;
    int max_nr_threads_;
    int warp_size_;
    double best_time_;
    double before_weighting_;
    double after_weighting_;
    int best_nr_threads_;
    double average_time_;
    bool stop_optimizing_;
    static const int NR_ROUNDS_PER_SETTING_ = 30;
    int optimization_runs_;

    // optional flag for optimizing the #threads
    bool optimize_nr_threads_;
};
}
