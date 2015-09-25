/** @author Claudia Pfreundt */

#define CHECK_ERRORS
//#define PROFILING_ACTIVE
//#define DEBUG_ON

#define VECTOR_DIM 3
#define MATRIX_DIM 9

#include <dbot/models/observation_models/kinect_image_observation_model_gpu/cuda_filter.hpp>
#include <GL/glut.h>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include <cuda.h>
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
#include <math.h>
#include <math_constants.h>


using namespace std;

namespace fil
{

// ====================== CUDA CONSTANT VALUES ======================= //

// used in propagate
// 1000 denotes the maximum number of objects
__constant__ float3 g_rot_center[1000];
// sigmas.x == angle_sigma, sigmas.y == trans_sigma
__constant__ float2 g_sigmas;

// used in propagateOcclusion
__constant__ float g_p_visible_occluded;
__constant__ float g_c;
__constant__ float g_log_c;

// used in prob
__constant__ float g_tail_weight;
__constant__ float g_model_sigma;
__constant__ float g_sigma_factor;
__constant__ float g_max_depth;
__constant__ float g_exponential_rate;

// used in compare
__constant__ float g_p_visible_init;

// texture for OpenGL interop
texture<float, cudaTextureType2D, cudaReadModeElementType> texture_reference;





// ************************************************************************************** //
// ************************************************************************************** //
// ================================== CUDA KERNELS ====================================== //
// ************************************************************************************** //
// ************************************************************************************** //

// ============================================================================================= //
// ====================== DEVICE kernels - to be called by other kernels ======================= //
// ============================================================================================= //

// ====================== MATRIX MANIPULATION FUNCTIONS ======================= //

__device__ void multiply_matrices(float *A, float *B, float *C) {
    float sum = 0;
    for (int i = 0; i < VECTOR_DIM; i++) {        // iterate through rows
        for (int j = 0; j < VECTOR_DIM; j++) {    // iterate through cols
            for (int k = 0; k < VECTOR_DIM; k++) {
                sum += A[i * VECTOR_DIM + k] * B[k * VECTOR_DIM + j];
            }
            C[i * VECTOR_DIM + j] = sum;
            sum = 0;
        }
    }
}

__device__ float3 multiply_matrix_with_vector(float* M, float3 v) {
    float result[3];
    float v_copy[3];
    v_copy[0] = v.x; v_copy[1] = v.y; v_copy[2] = v.z;
    float sum = 0;

    for (int i = 0; i < VECTOR_DIM; i++) {
        for (int j = 0; j < VECTOR_DIM; j++) {
            sum += M[i * VECTOR_DIM + j] * v_copy[j];
        }
        result[i] = sum;
        sum = 0;
    }

    return make_float3(result[0], result[1], result[2]);
}

/* axis is defined as follows: 0 = x, 1 = y, 2 = z */
__device__ void create_rotation_matrix(const float angle, const int axis, float *R) {
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);

    if (axis == 0) {
        R[0] = 1;   R[1] = 0;           R[2] = 0;
        R[3] = 0;   R[4] = cos_angle;   R[5] = -sin_angle;
        R[6] = 0;   R[7] = sin_angle;   R[8] = cos_angle;
    } else if (axis == 1) {
        R[0] = cos_angle;   R[1] = 0;   R[2] = sin_angle;
        R[3] = 0;           R[4] = 1;   R[5] = 0;
        R[6] = -sin_angle;  R[7] = 0;   R[8] = cos_angle;
    } else if (axis == 2) {
        R[0] = cos_angle;   R[1] = -sin_angle;  R[2] = 0;
        R[3] = sin_angle;   R[4] = cos_angle;   R[5] = 0;
        R[6] = 0;           R[7] = 0;           R[8] = 1;
    }
}

__device__ void transpose_matrix(float *A, float *T) {
    T[0] = A[0];
    T[1] = A[3];
    T[2] = A[6];
    T[3] = A[1];
    T[4] = A[4];
    T[5] = A[7];
    T[6] = A[2];
    T[7] = A[5];
    T[8] = A[8];
}

// ====================== VECTOR MANIPULATION FUNCTIONS ======================= //

__device__ float4 normalize(const float4 v) {
    float4 v_n = v;
    const float n = 1.0f/sqrt(v_n.x*v_n.x+v_n.y*v_n.y+v_n.z*v_n.z+v_n.w*v_n.w);
    v_n.x *= n;
    v_n.y *= n;
    v_n.z *= n;
    v_n.w *= n;

    return v_n;
}

__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__device__ float3 negate(const float3 &a) {
    return make_float3(-a.x, -a.y, -a.z);
}

// ======================= QUATERNION CONVERSIONS AND MANIPULATION FUNCTIONS ======================= //

__device__ void quaternion_to_matrix(const float4 q_in, float *Q) {
    float4 q = normalize(q_in);
    Q[0] = 1.0f - 2.0f*q.y*q.y - 2.0f*q.z*q.z;  Q[1] = 2.0f*q.x*q.y - 2.0f*q.z*q.w;         Q[2] = 2.0f*q.x*q.z + 2.0f*q.y*q.w;
    Q[3] = 2.0f*q.x*q.y + 2.0f*q.z*q.w;         Q[4] = 1.0f - 2.0f*q.x*q.x - 2.0f*q.z*q.z;  Q[5] = 2.0f*q.y*q.z - 2.0f*q.x*q.w;
    Q[6] = 2.0f*q.x*q.z - 2.0f*q.y*q.w;         Q[7] = 2.0f*q.y*q.z + 2.0f*q.x*q.w;         Q[8] = 1.0f - 2.0f*q.x*q.x - 2.0f*q.y*q.y;
}

__device__ float4 matrix_to_quaternion(float *Q) {
    float4 q;

    q.w = sqrtf( fmaxf( 0, 1 + Q[0] + Q[4] + Q[8] ) ) / 2;
    q.x = sqrtf( fmaxf( 0, 1 + Q[0] - Q[4] - Q[8] ) ) / 2;
    q.y = sqrtf( fmaxf( 0, 1 - Q[0] + Q[4] - Q[8] ) ) / 2;
    q.z = sqrtf( fmaxf( 0, 1 - Q[0] - Q[4] + Q[8] ) ) / 2;
    if (( q.x * ( Q[7] - Q[5] ) ) < 0) {
        q.x = -q.x;
    }
    if (( q.y * ( Q[2] - Q[6] ) ) < 0) {
        q.y = -q.y;
    }
    if (( q.z * ( Q[3] - Q[1] ) ) < 0) {
        q.z = -q.z;
    }

    return q;
}

__device__ float4 multiply_quaternions(float4 q1, float4 q2) {
    float w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    float x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    float y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    float z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);

    return make_float4(x, y, z, w);
}



// ======================= helper functions for compare (observation model)  ======================= //

__device__ float propagate_occlusion(float initial_p_source, float time) {
    if (isnan(time)) {
        return initial_p_source;
    }
    float pow_c_time = exp(time*g_log_c);
    return (float) pow_c_time*initial_p_source + g_p_visible_occluded*(pow_c_time-1.)/(g_c-1.);
}



__device__ float prob(float observation, float prediction, bool visible)
{
    // todo: if the prediction is infinite, the prob should not depend on visibility. it does not matter
    // for the algorithm right now, but it should be changed

    float sigma = g_model_sigma + g_sigma_factor*observation*observation;
    if(visible)
    {
        if(isinf(prediction)) // if the prediction is infinite we return the limit
            return g_tail_weight/g_max_depth;
        else
            return g_tail_weight/g_max_depth
                    + (1 - g_tail_weight)*expf(-(powf(prediction-observation,2)/(2*sigma*sigma)))
                    / (sqrtf(2*M_PI) *sigma);
    }
    else
    {
        if(isinf(prediction)) // if the prediction is infinite we return the limit
            return g_tail_weight/g_max_depth +
                    (1-g_tail_weight)*g_exponential_rate*
                    expf(0.5*g_exponential_rate*(-2*observation + g_exponential_rate*sigma*sigma));

        else
            return g_tail_weight/g_max_depth +
                    (1-g_tail_weight)*g_exponential_rate*
                    expf(0.5*g_exponential_rate*(2*prediction-2*observation + g_exponential_rate*sigma*sigma))
        *(1+erff((prediction-observation+g_exponential_rate*sigma*sigma)/(sqrtf(2)*sigma)))
        /(2*(expf(prediction*g_exponential_rate)-1));
    }
}






// ============================================================================================= //
// ========================= GLOBAL kernels - to be called by CPU code ========================= //
// ============================================================================================= //



__global__ void setup_number_generators_kernel(int current_time, curandStateMRG32k3a *mrg_state, int n_poses)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_poses) {
        /* Each thread gets same seed, a different sequence number, no offset */
        curand_init(current_time, id, 0, &mrg_state[id]);
    }
}


__global__ void propagate_kernel(float *states, int n_states, int states_size, float delta_time, curandStateMRG32k3a *mrg_state)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_states) {

        /* Copy sigmas from constant memory into local register */
        float2 local_sigmas = g_sigmas;

        /* Copy mrg_state from global memory into local register */
        curandStateMRG32k3a local_mrg_state = mrg_state[id];        

        for (int i = 0; i < states_size / 7; i++) {

            int states_index = id * states_size + i * 7;

            /* Copy rot_center from constant memory into local register */
            float3 local_rot_center = g_rot_center[i];

            /* TODO coalesce accesses? Does it do it automatically or do I manually have to store them as
            * float3 and float4 values? */
            /* quaternion stored as (w,x,y,z), but make_float4 takes (x,y,z,w) */
            float4 q_init_vector = make_float4(states[states_index + 1], states[states_index + 2], states[states_index + 3], states[states_index]);
            float3 t_init = make_float3(states[states_index + 4], states[states_index + 5], states[states_index + 6]);

            float angle_x, angle_y, angle_z;
            float trans_x, trans_y, trans_z;

            angle_x = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            angle_y = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            angle_z = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            trans_x = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;
            trans_y = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;
            trans_z = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;


            float q_rand_matrix[MATRIX_DIM];
            float q_init_matrix[MATRIX_DIM];

            float rot_matrix_x[MATRIX_DIM];
            float rot_matrix_y[MATRIX_DIM];
            float rot_matrix_z[MATRIX_DIM];

            float tmp_matrix[MATRIX_DIM];


            float3 t_rand = make_float3(trans_x, trans_y, trans_z);

            create_rotation_matrix(angle_x, 0, rot_matrix_x);
            create_rotation_matrix(angle_y, 1, rot_matrix_y);
            create_rotation_matrix(angle_z, 2, rot_matrix_z);

            multiply_matrices(rot_matrix_y, rot_matrix_z, tmp_matrix);
            multiply_matrices(rot_matrix_x, tmp_matrix, q_rand_matrix);

            float4 q_rand_vector = matrix_to_quaternion(q_rand_matrix);

            quaternion_to_matrix(q_init_vector, q_init_matrix);

            float3 t = negate(multiply_matrix_with_vector(q_init_matrix, multiply_matrix_with_vector(q_rand_matrix, local_rot_center)))
                   + multiply_matrix_with_vector(q_init_matrix, local_rot_center)
                   + t_init
                   + t_rand;

            float4 q = multiply_quaternions(q_init_vector, q_rand_vector);
            q = normalize(q);

            /* write state back into global memory */
            states[states_index] = q.w;
            states[states_index + 1] = q.x;
            states[states_index + 2] = q.y;
            states[states_index + 3] = q.z;

            states[states_index + 4] = t.x;
            states[states_index + 5] = t.y;
            states[states_index + 6] = t.z;
        }

        /* Copy mrg state back to global memory */
        mrg_state[id] = local_mrg_state;
    }
}






__global__ void propagate_multiple_kernel(float *states, int n_states, int n_objects, int states_size, float delta_time, curandStateMRG32k3a *mrg_state)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_states) {

        /* Copy sigmas from constant memory into local register */
        float2 local_sigmas = g_sigmas;

        /* Copy mrg_state from global memory into local register */
        curandStateMRG32k3a local_mrg_state = mrg_state[id];

        for (int i = 0; i < n_objects; i++) {

            int states_index = id * n_objects * states_size + i * states_size;

            /* Copy rot_center from constant memory into local register */
            float3 local_rot_center = g_rot_center[i];

            /* TODO coalesce accesses? Does it do it automatically or do I manually have to store them as
            * float3 and float4 values? */
            /* quaternion stored as (w,x,y,z), but make_float4 takes (x,y,z,w) */
            float4 q_init_vector = make_float4(states[states_index + 1], states[states_index + 2], states[states_index + 3], states[states_index]);
            float3 t_init = make_float3(states[states_index + 4], states[states_index + 5], states[states_index + 6]);

            float angle_x, angle_y, angle_z;
            float trans_x, trans_y, trans_z;

            // WARNING: same random number for all states, because mrg_state is the same..
            angle_x = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            angle_y = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            angle_z = curand_normal(&local_mrg_state) * delta_time * local_sigmas.x;
            trans_x = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;
            trans_y = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;
            trans_z = curand_normal(&local_mrg_state) * delta_time * local_sigmas.y;


            float q_rand_matrix[MATRIX_DIM];
            float q_init_matrix[MATRIX_DIM];

            float rot_matrix_x[MATRIX_DIM];
            float rot_matrix_y[MATRIX_DIM];
            float rot_matrix_z[MATRIX_DIM];

            float tmp_matrix[MATRIX_DIM];


            float3 t_rand = make_float3(trans_x, trans_y, trans_z);

            create_rotation_matrix(angle_x, 0, rot_matrix_x);
            create_rotation_matrix(angle_y, 1, rot_matrix_y);
            create_rotation_matrix(angle_z, 2, rot_matrix_z);

            multiply_matrices(rot_matrix_y, rot_matrix_z, tmp_matrix);
            multiply_matrices(rot_matrix_x, tmp_matrix, q_rand_matrix);

            float4 q_rand_vector = matrix_to_quaternion(q_rand_matrix);

            quaternion_to_matrix(q_init_vector, q_init_matrix);

            float3 t = negate(multiply_matrix_with_vector(q_init_matrix, multiply_matrix_with_vector(q_rand_matrix, local_rot_center)))
                   + multiply_matrix_with_vector(q_init_matrix, local_rot_center)
                   + t_init
                   + t_rand;

            float4 q = multiply_quaternions(q_init_vector, q_rand_vector);
            q = normalize(q);

            /* write state back into global memory */
            states[states_index] = q.w;
            states[states_index + 1] = q.x;
            states[states_index + 2] = q.y;
            states[states_index + 3] = q.z;

            states[states_index + 4] = t.x;
            states[states_index + 5] = t.y;
            states[states_index + 6] = t.z;
        }

        /* Copy mrg state back to global memory */
        mrg_state[id] = local_mrg_state;
    }
}












__global__ void compare_kernel(float *observations, float* visibility_probs, int n_pixels_per_pose,
                        bool constant_occlusion, float *d_log_likelihoods, float delta_time, int n_poses, int n_rows, int n_cols) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    if (block_id < n_poses) {

        int pixel_nr = threadIdx.x;
//        int pixel_nr = threadIdx.x * ceilf(n_pixels_per_pose / blockDim.x);
        int global_index = block_id * n_pixels_per_pose + pixel_nr;

        // OpenGL contructs the texture so that the left lower edge is (0,0), but our observations texture
        // has its (0,0) in the upper left corner, so we need to reverse the reads from the OpenGL texture.
        float texture_array_index_x = blockIdx.x * n_cols + pixel_nr % n_cols;
        float texture_array_index_y = gridDim.y * n_rows - (blockIdx.y * n_rows + pixel_nr / n_cols + 1);

        float depth;
        float observed_depth;
        float visibility_prob = g_p_visible_init;
        float local_sum_of_likelihoods = 0;
        float p_obsIpred_vis, p_obsIpred_occl, p_obsIinf;

        // TODO: uninitialized?
        __shared__ float log_likelihoods;

        if (threadIdx.x == 0) {
            log_likelihoods = 0;
        }

        __syncthreads();

        while (pixel_nr < n_pixels_per_pose ) { //&& pixel_nr < (threadIdx.x + 1) * ceilf(n_pixels_per_pose / blockDim.x)) {

            depth = tex2D(texture_reference, texture_array_index_x, texture_array_index_y);
            observed_depth = observations[pixel_nr];

            // TODO either this, or only write the values back for pixels with depth value == 1.
            // Could save some data transfer time, but will cost more execution time, since all
            // the threads in one warp will have to wait for the else-branch to finish
            if (!constant_occlusion) {
                visibility_prob = propagate_occlusion(visibility_probs[global_index], delta_time);
                visibility_probs[global_index] = visibility_prob;
            }
//            if (!constant_occlusion) {
//                visibility_prob = propagateOcclusion(visibility_probs[global_index], delta_time);
//            }

            //TODO slow: 4800 threads have to go through this whole if instruction
            if (depth != 0 && !isnan(observed_depth)) {

                // prob of observation given prediction, knowing that the object is visible
                p_obsIpred_vis = prob(observed_depth, depth, true) * visibility_prob;
                // prob of observation given prediction, knowing that the object is occluded
                p_obsIpred_occl = prob(observed_depth, depth, false) * (1-visibility_prob);
                // prob of observation given no intersection
                p_obsIinf = prob(observed_depth, CUDART_INF_F, false);

                local_sum_of_likelihoods += logf((p_obsIpred_vis + p_obsIpred_occl)/p_obsIinf);

                if(!constant_occlusion) { // we check if we are tracking the visibilities
                    // we update the visibility (occlusion) with the observations
                    visibility_probs[global_index] = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
                }
//                if (!constant_occlusion) {
//                    visibility_prob = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
//                }
            }

//            if (!constant_occlusion) {
//                visibility_probs[global_index] = visibility_prob;
//            }

            pixel_nr += blockDim.x;
//            pixel_nr += 1;
            global_index = block_id * n_pixels_per_pose + pixel_nr;
            texture_array_index_x = blockIdx.x * n_cols + pixel_nr % n_cols;
            texture_array_index_y = gridDim.y * n_rows - (blockIdx.y * n_rows + pixel_nr / n_cols + 1);
        }

        // TODO: will execute blockDim.x sequential writes to log_likelihoods
        // instead could do a manual reduction after syncthreads
        atomicAdd(&log_likelihoods, local_sum_of_likelihoods);

        __syncthreads();

        if (threadIdx.x == 0) {
            d_log_likelihoods[block_id] = log_likelihoods;
        }
    } else {
        __syncthreads();
    }

}





__global__ void compare_multiple_kernel(float *observations, float* old_visibility_probs, float* new_visibility_probs, int* occlusion_image_indices, int nr_pixels,
                                 float *d_log_likelihoods, float delta_time, int n_poses, int n_rows, int n_cols, bool update) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    if (block_id < n_poses) {

        int pixel_nr = threadIdx.x;

        // OpenGL contructs the texture so that the left lower edge is (0,0), but our observations texture
        // has its (0,0) in the upper left corner, so we need to reverse the reads from the OpenGL texture.
        float texture_array_index_x = blockIdx.x * n_cols + pixel_nr % n_cols;

        float texture_array_index_y = gridDim.y * n_rows - 1 - (blockIdx.y * n_rows + pixel_nr / n_cols);

        float depth;
        float observed_depth;
        float visibility_prob = g_p_visible_init;
        float local_sum_of_likelihoods = 0;
        float p_obsIpred_vis, p_obsIpred_occl, p_obsIinf;

        __shared__ float log_likelihoods;
        __shared__ int occlusion_image_index;

        if (threadIdx.x == 0) {
            log_likelihoods = 0;
            occlusion_image_index = occlusion_image_indices[block_id];
        }

        __syncthreads();

        float* visibility_probs = old_visibility_probs;
        int occlusion_pixel_index= occlusion_image_index * nr_pixels + pixel_nr;

        if (update) {
            // copy / duplicate visibility probabilities from the old particles
            int index_from_visibility = occlusion_image_indices[block_id] * nr_pixels;
            int index_to_visibility = block_id * nr_pixels;

            while (pixel_nr < nr_pixels) {
                new_visibility_probs[index_to_visibility + pixel_nr] = old_visibility_probs[index_from_visibility + pixel_nr];
                pixel_nr += blockDim.x;
            }

            // change visibility prob array to the new one and change the global index
            visibility_probs = new_visibility_probs;
            // reset pixel_nr for following loop
            pixel_nr = threadIdx.x;

            occlusion_pixel_index= block_id * nr_pixels + pixel_nr;
        }


        while (pixel_nr < nr_pixels ) {

            depth = tex2D(texture_reference, texture_array_index_x, texture_array_index_y);
            observed_depth = observations[pixel_nr];

            // TODO either this, or only write the values back for pixels with depth value == 1.
            // Could save some data transfer time, but will cost more execution time, since all
            // the threads in one warp will have to wait for the else-branch to finish

            visibility_prob = propagate_occlusion(visibility_probs[occlusion_pixel_index], delta_time);
            if (update) visibility_probs[occlusion_pixel_index] = visibility_prob;


            if (depth != 0 && !isnan(observed_depth)) {


                // prob of observation given prediction, knowing that the object is visible
                p_obsIpred_vis = prob(observed_depth, depth, true) * visibility_prob;
                // prob of observation given prediction, knowing that the object is occluded
                p_obsIpred_occl = prob(observed_depth, depth, false) * (1-visibility_prob);
                // prob of observation given no intersection
                p_obsIinf = prob(observed_depth, CUDART_INF_F, false);

                local_sum_of_likelihoods += logf((p_obsIpred_vis + p_obsIpred_occl)/p_obsIinf);


                if(update) {
                    // we update the visibility (occlusion) probability with the observations
                    visibility_probs[occlusion_pixel_index] = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
                }
            }

            pixel_nr += blockDim.x;
            occlusion_pixel_index += blockDim.x;
            texture_array_index_x = blockIdx.x * n_cols + pixel_nr % n_cols;
            texture_array_index_y = gridDim.y * n_rows - (blockIdx.y * n_rows + pixel_nr / n_cols + 1);
        }

        // TODO: will execute blockDim.x sequential writes to log_likelihoods
        // instead could do a manual reduction after syncthreads
        atomicAdd(&log_likelihoods, local_sum_of_likelihoods);

        __syncthreads();

        if (threadIdx.x == 0) {
            d_log_likelihoods[block_id] = log_likelihoods;
        }
    } else {
        __syncthreads();
    }

}







__global__ void resample_kernel(float *visibility_probs,
                         float *visibility_probs_copy,
                         float *states,
                         float *states_copy,
                         int *resampling_indices,
                         int nr_pixels,
                         int nr_features) {

    int pixel_nr = threadIdx.x;
    int feature_nr = threadIdx.x;
    int index_from_visibility = resampling_indices[blockIdx.x] * nr_pixels;
    int index_to_visibility = blockIdx.x * nr_pixels;
    int index_from_states = resampling_indices[blockIdx.x] * nr_features;
    int index_to_states = blockIdx.x * nr_features;


    while (pixel_nr < nr_pixels) {
        visibility_probs_copy[index_to_visibility + pixel_nr] = visibility_probs[index_from_visibility + pixel_nr];
        pixel_nr += blockDim.x;
    }
    while (feature_nr < nr_features) {
        states_copy[index_to_states + feature_nr] = states[index_from_states + feature_nr];
        feature_nr += blockDim.x;
    }
}


__global__ void resample_multiple_kernel(float *visibility_probs,
                                  float *visibility_probs_copy,
                                  int *resampling_indices,
                                  int nr_pixels) {

    int pixel_nr = threadIdx.x;
    int index_from_visibility = resampling_indices[blockIdx.x] * nr_pixels;
    int index_to_visibility = blockIdx.x * nr_pixels;

    while (pixel_nr < nr_pixels) {
        visibility_probs_copy[index_to_visibility + pixel_nr] = visibility_probs[index_from_visibility + pixel_nr];
        pixel_nr += blockDim.x;
    }
}



// ************************************************************************************** //
// ************************************************************************************** //
// ========================== CUDA_FILTER MEMBER FUNCTIONS ============================== //
// ************************************************************************************** //
// ************************************************************************************** //


CudaFilter::CudaFilter() :
    n_cols_(WINDOW_WIDTH),
    n_rows_(WINDOW_HEIGHT),
    n_poses_set_(false)
{

    cudaDeviceProp  props;
    int device_number;

    memset( &props, 0, sizeof( cudaDeviceProp ) );
    props.major = 2;
    props.minor = 0;
    cudaChooseDevice( &device_number, &props );
    check_cuda_error("No device with compute capability > 2.0 found");

    /* tell CUDA which device we will be using for graphic interop
     * requires that the CUDA device be specified by
     * cudaGLSetGLDevice() before any other runtime calls. */

    cudaGLSetGLDevice( device_number );
    check_cuda_error("cudaGLsetGLDevice");

    cudaGetDeviceProperties(&props, device_number);     // we will run the program only on one graphics card, the first one we can find = 0
    warp_size_ = props.warpSize;            // equals 32 for all current graphics cards, but might change in the future
    n_mps_ = props.multiProcessorCount;

    cuda_device_properties_ = props;

    cout << "Your device has the following properties: " << endl
         << "CUDA Version: " << props.major << "." << props.minor << endl
         << "Number of multiprocessors: " << n_mps_ << endl
         << "Warp size: " << warp_size_ << endl;

    /* each multiprocessor has various KB of memory (64 KB for the GTX 560 Ti 448) which can be subdivided
     * into L1 cache or shared memory. If you don't need a lot of shared memory set this to prefer L1. */
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);


    d_states_ = NULL;
    d_states_copy_ = NULL;
    d_visibility_probs_ = NULL;
    d_visibility_probs_copy_ = NULL;
    d_observations_ = NULL;
    d_log_likelihoods_ = NULL;
    d_mrg_states_ = NULL;
    d_resampling_indices_ = NULL;
    d_prev_sample_indices_ = NULL;

}

void CudaFilter::init(vector<vector<float> > com_models, float angle_sigma, float trans_sigma,
                      float p_visible_init, float c, float log_c, float p_visible_occluded,
                      float tail_weight, float model_sigma, float sigma_factor, float max_depth, float exponential_rate) {

    occlusion_time_ = 0;
    last_propagation_time_ = 0;
    count_ = 0;
    compare_kernel_time_ = 0;
    copy_likelihoods_time_ = 0;
    visibility_prob_default_ = p_visible_init;

    float2 local_sigmas = make_float2(angle_sigma, trans_sigma);

    allocate(d_observations_, n_cols_ * n_rows_ * sizeof(float), "d_observations");
//    allocate(d_log_likelihoods_, sizeof(float) * n_poses_, "d_log_likelihoods");
    // TODO don't allocate here!! only when setting resolution!
//    allocate(d_prev_sample_indices_, sizeof(int) * n_poses_, "d_prev_sample_indices");
//    allocate(d_resampling_indices_, sizeof(int) * n_poses_, "d_resampling_indices");

    cudaMemset(d_log_likelihoods_, 0, sizeof(float) * nr_poses_);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemset d_log_likelihoods");
    #endif

    cudaMemcpyToSymbol(g_sigmas, &local_sigmas, sizeof(float2), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol local_sigmas -> sigmas");
    #endif

    vector<float3> com_models_raw;
    for (int i = 0; i < com_models.size(); i++) {
        com_models_raw.push_back(make_float3(com_models[i][0], com_models[i][1], com_models[i][2]));
    }

    cudaMemcpyToSymbol(g_rot_center, com_models_raw.data(), com_models_raw.size() * sizeof(float3), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol com_model -> rot_center");
    #endif

    cudaMemcpyToSymbol(g_p_visible_init, &p_visible_init, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol p_visible_init -> g_p_visible_init");
    #endif

    cudaMemcpyToSymbol(g_c, &c, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol c -> g_c");
    #endif

    cudaMemcpyToSymbol(g_log_c, &log_c, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol log_c -> g_log_c");
    #endif

    cudaMemcpyToSymbol(g_p_visible_occluded, &p_visible_occluded, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol p_visible_occluded -> g_p_visible_occluded");
    #endif

    cudaMemcpyToSymbol(g_tail_weight, &tail_weight, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol tail_weight -> g_tail_weight");
    #endif

    cudaMemcpyToSymbol(g_model_sigma, &model_sigma, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol model_sigma -> g_model_sigma");
    #endif

    cudaMemcpyToSymbol(g_sigma_factor, &sigma_factor, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol sigma_factor -> g_sigma_factor");
    #endif

    cudaMemcpyToSymbol(g_max_depth, &max_depth, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol max_depth -> g_max_depth");
    #endif

    cudaMemcpyToSymbol(g_exponential_rate, &exponential_rate, sizeof(float), 0, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpyToSymbol exponential_rate -> g_exponential_rate");
    #endif
}


void CudaFilter::propagate(const float &current_time, vector<vector<float> > &states)
{


    float delta_time = current_time - last_propagation_time_;
    last_propagation_time_ = current_time;


    propagate_kernel <<< n_blocks_, nr_threads_ >>> (d_states_, nr_poses_, n_features_, delta_time, d_mrg_states_);
    #ifdef CHECK_ERRORS
        check_cuda_error("propagate kernel call");
    #endif



    // TODO necessary? Remove for performance?
    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize propagate");
    #endif



    float *states_raw = (float*) malloc(nr_poses_ * n_features_ * sizeof(float));
    cudaMemcpy(states_raw, d_states_, nr_poses_ * n_features_ * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy d_states -> states");
    #endif


    for (int i = 0; i < nr_poses_; i++) {
        for (int j = 0; j < n_features_; j++) {
            states[i][j] = states_raw[i * n_features_ + j];
        }
    }
}




void CudaFilter::propagate_multiple(const float &current_time, vector<vector<vector<float> > > &states)
{

    float delta_time = current_time - last_propagation_time_;
    last_propagation_time_ = current_time;

    int n_objects = states[0].size();

    float *states_raw = (float*) malloc(nr_poses_ * n_objects * n_features_ * sizeof(float));
    for (int i = 0; i < nr_poses_; i++) {
        for (int j = 0; j < n_objects; j++) {
            for (int k = 0; k < n_features_; k++) {
                states_raw[(i * n_objects *n_features_) + j * n_features_ + k] = states[i][j][k];
            }
        }
    }


    cudaMemcpy(d_states_, states_raw, nr_poses_ * n_objects * n_features_ * sizeof(float), cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy states -> d_states");
    #endif


    propagate_multiple_kernel <<< n_blocks_, nr_threads_ >>> (d_states_, nr_poses_, n_objects, n_features_, delta_time, d_mrg_states_);
    #ifdef CHECK_ERRORS
        check_cuda_error("propagate kernel call");
    #endif



    // TODO necessary? Remove for performance?
    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize propagate");
    #endif



    cudaMemcpy(states_raw, d_states_, nr_poses_ * n_objects * n_features_ * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy d_states -> states");
    #endif


    for (int i = 0; i < nr_poses_; i++) {
        for (int j = 0; j < n_objects; j++) {
            for (int k = 0; k < n_features_; k++) {
                states[i][j][k] = states_raw[(i * n_objects *n_features_) + j * n_features_ + k];
            }
        }
    }

    free(states_raw);
}




void CudaFilter::compare(float observation_time, bool constant_occlusion, vector<float> &log_likelihoods) {

#ifdef PROFILING_ACTIVE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
#endif

    dim3 gridDim = dim3(nr_poses_per_row_, nr_poses_per_column_);

    // update observation time
    float delta_time = observation_time - occlusion_time_;
    occlusion_time_ = observation_time;



#ifdef PROFILING_ACTIVE
    cudaEventRecord(start);
#endif

    compare_kernel <<< gridDim, 128 >>> (d_observations_, d_visibility_probs_, n_cols_ * n_rows_,
            constant_occlusion, d_log_likelihoods_, delta_time, nr_poses_, n_rows_, n_cols_);
    #ifdef CHECK_ERRORS
        check_cuda_error("compare kernel call");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize compare");
    #endif

#ifdef PROFILING_ACTIVE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);    
    count_++;
    compare_kernel_time_ += milliseconds;
    if (count_ == COUNT) {
        cout << "compare kernel: " << compare_kernel_time_ * 1e3 / count_ << " us" << endl;
    }
    cudaEventRecord(start);
#endif

    cudaMemcpy(&log_likelihoods[0], d_log_likelihoods_, nr_poses_ * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy d_log_likelihoods -> log_likelihoods");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize compare");
    #endif

#ifdef PROFILING_ACTIVE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    copy_likelihoods_time_ += milliseconds;
    if (count_ == COUNT) {
        cout << "copy likelihoods: " << copy_likelihoods_time_ * 1e3 / count_ << " us" << endl;
    }
#endif
}



void CudaFilter::compare_multiple(bool update, vector<float> &log_likelihoods) {

#ifdef PROFILING_ACTIVE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    cudaEventRecord(start);
#endif

    double delta_time = observation_time_ - occlusion_time_;
    if(update) occlusion_time_ = observation_time_;
//    cout << "delta time: " << delta_time << endl;


    compare_multiple_kernel <<< grid_dimension_, nr_threads_ >>> (d_observations_, d_visibility_probs_, d_visibility_probs_copy_, d_prev_sample_indices_, n_cols_ * n_rows_,
                                           d_log_likelihoods_, delta_time, nr_poses_, n_rows_, n_cols_, update);
    #ifdef CHECK_ERRORS
        check_cuda_error("compare kernel call");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize compare_multiple");
    #endif



    // switch to new / copied visibility probabilities
    if (update) {
        float *tmp_pointer;
        tmp_pointer = d_visibility_probs_;
        d_visibility_probs_ = d_visibility_probs_copy_;
        d_visibility_probs_copy_ = tmp_pointer;
    }

#ifdef PROFILING_ACTIVE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    count_++;
    compare_kernel_time_ += milliseconds;
    if (count_ == COUNT) {
        cout << "compare kernel: " << compare_kernel_time_ * 1e3 / count_ << " us" << endl;
    }
    cudaEventRecord(start);
#endif

    cudaMemcpy(&log_likelihoods[0], d_log_likelihoods_, nr_poses_ * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy d_log_likelihoods -> log_likelihoods");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize compare");
    #endif

#ifdef PROFILING_ACTIVE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    copy_likelihoods_time_ += milliseconds;
    if (count_ == COUNT) {
        cout << "copy likelihoods: " << copy_likelihoods_time_ * 1e3 / count_ << " us" << endl;
    }
#endif
}







void CudaFilter::resample(vector<int> resampling_indices) {

//    cout << "resample <<< " << n_poses_ << ", " << 128 << " >>>" << endl;

    cudaMemcpy(d_resampling_indices_, &resampling_indices[0], sizeof(int) * nr_poses_, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy resampling_indices -> d_resampling_indices_");
    #endif

//        int min = 100;
//        int max = -1;
//        for (int i = 0; i < resampling_indices.size(); i++) {
//            int value = resampling_indices[i];
//            if (value > max) max = value;
//            if (value < min) min = value;
//        }
//        cout << "resample min: " << min << ", max: " << max << endl;


    int nr_pixels = n_rows_ * n_cols_;

    resample_kernel <<< nr_poses_, 128 >>> (d_visibility_probs_, d_visibility_probs_copy_,
                                    d_states_, d_states_copy_,
                                    d_resampling_indices_, nr_pixels, n_features_);
    #ifdef CHECK_ERRORS
        check_cuda_error("resample kernel call");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize resample");
    #endif


    // switch the visibility probs pointers, so that the next Compare() call will access the resampled
    // visibility probs. Same for the states.
    float *tmp_pointer;
    tmp_pointer = d_visibility_probs_;
    d_visibility_probs_ = d_visibility_probs_copy_;
    d_visibility_probs_copy_ = tmp_pointer;
    tmp_pointer = d_states_;
    d_states_ = d_states_copy_;
    d_states_copy_ = tmp_pointer;

}




void CudaFilter::resample_multiple(vector<int> resampling_indices) {

    cudaMemcpy(d_resampling_indices_, &resampling_indices[0], sizeof(int) * nr_poses_, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy resampling_indices -> d_resampling_indices_");
    #endif

    int nr_pixels = n_rows_ * n_cols_;

    resample_multiple_kernel <<< nr_poses_, 128 >>> (d_visibility_probs_, d_visibility_probs_copy_,
                                             d_resampling_indices_, nr_pixels);
    #ifdef CHECK_ERRORS
        check_cuda_error("resample kernel call");
    #endif

    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize resample");
    #endif


    // switch the visibility probs pointers, so that the next Compare() call will access the resampled
    // visibility probs.
    float *tmp_pointer;
    tmp_pointer = d_visibility_probs_;
    d_visibility_probs_ = d_visibility_probs_copy_;
    d_visibility_probs_copy_ = tmp_pointer;
}





// ===================================================================================== //
// =============================== CUDA FILTER SETTERS ================================= //
// ===================================================================================== //

void CudaFilter::set_states(std::vector<std::vector<float> > &states, int seed)
{
    if (n_poses_set_) {
        // copy states into linear array
        /* TODO maybe padding can speed up the memory accesses later from the kernel, since
         * right now, each MP needs 7 values out of d_states_. 8 would be a much better number. */
        n_features_ = states[0].size();

        int states_size = nr_poses_ * n_features_ * sizeof(float);
        float *states_raw = (float*) malloc(states_size);

        for (size_t i = 0; i < nr_poses_; i++) {
            for (size_t j = 0; j < n_features_; j++) {
                states_raw[i * n_features_ + j] = states[i][j];
            }
        }

        allocate(d_states_, states_size, "d_states");
        allocate(d_states_copy_, states_size, "d_states_copy");     // placeholder for resampling purposes

        cudaMemcpy(d_states_, states_raw, states_size, cudaMemcpyHostToDevice);
        #ifdef CHECK_ERRORS
            check_cuda_error("cudaMemcpy states_raw -> d_states_");
        #endif

        free(states_raw);

        // setup random number generators for each thread to be used in the propagate kernel
        allocate(d_mrg_states_, nr_poses_ * sizeof(curandStateMRG32k3a), "d_mrg_states");

        setup_number_generators_kernel <<< n_blocks_, nr_threads_ >>> (seed, d_mrg_states_, nr_poses_);

        cudaDeviceSynchronize();
    } else {
        cout << "WARNING: set_states() was not executed, because n_poses_ has not been set previously";
        exit(-1);
    }
}






void CudaFilter::set_states_multiple(int n_objects, int n_features, int seed)
{
    if (n_poses_set_) {
        n_features_ = n_features;

        int states_size = nr_poses_ * n_objects * n_features_ * sizeof(float);
        allocate(d_states_, states_size, "d_states");


        // setup random number generators for each thread to be used in the propagate kernel
        allocate(d_mrg_states_, nr_poses_ * sizeof(curandStateMRG32k3a), "d_mrg_states");

        setup_number_generators_kernel <<< n_blocks_, nr_threads_ >>> (seed, d_mrg_states_, nr_poses_);

        cudaDeviceSynchronize();
    } else {
        cout << "WARNING: set_states_multiple() was not executed, because n_poses_ has not been set previously";
        exit(-1);
    }
}







void CudaFilter::set_observations(const float* observations, const float observation_time) {

//    delta_time_ = observation_time - last_observation_time_;
    observation_time_ = observation_time;
//    cout << "delta time: " << delta_time_ << ", last_observation_time: " << occlusion_time_ << endl;
    set_observations(observations);
}

void CudaFilter::set_observations(const float* observations) {
    cudaMemcpy(d_observations_, observations, n_cols_ * n_rows_ * sizeof(float), cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy observations -> d_observations_");
    #endif
    cudaDeviceSynchronize();
}


void CudaFilter::set_prev_sample_indices(const int* prev_sample_indices) {
    cudaMemcpy(d_prev_sample_indices_, prev_sample_indices, nr_poses_ * sizeof(int), cudaMemcpyHostToDevice);
//    cout << "when setting prev_sample_indices: n_poses: " << n_poses_ << ", max poses: " << n_max_poses_ << endl;
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy prev_sample_indices -> d_prev_sample_indices_");
    #endif
    cudaDeviceSynchronize();
}


void CudaFilter::set_resolution(const int n_rows, const int n_cols, int& nr_poses, int& nr_poses_per_row, int& nr_poses_per_column) {
    n_rows_ = n_rows;
    n_cols_ = n_cols;

    // reallocate buffers
    allocate(d_observations_, n_cols_ * n_rows_ * sizeof(float), "d_observations");    
    allocate_memory_for_max_poses(nr_poses, nr_poses_per_row, nr_poses_per_column);
}


void CudaFilter::set_visibility_probabilities(const float* visibility_probabilities) {
    cudaMemcpy(d_visibility_probs_, visibility_probabilities, n_rows_ * n_cols_ * nr_poses_ * sizeof(float), cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy visibility_probabilities -> d_visibility_probs_");
    #endif
}




void CudaFilter::allocate_memory_for_max_poses(int& allocated_poses,
                                               int& allocated_poses_per_row,
                                               int& allocated_poses_per_column) {

    // check limitation by global memory
    size_t size_of_log_likelihoods = sizeof(float) * allocated_poses;
    size_t size_of_resampling_indices = sizeof(int) * allocated_poses;
    size_t size_of_prev_sample_indices = sizeof(int) * allocated_poses;
    size_t size_of_visibility_probs = n_rows_ * n_cols_ * allocated_poses * sizeof(float);
    size_t size_of_opengl_textures = size_of_visibility_probs * 2;
    size_t size_of_observations = n_cols_ * n_rows_ * sizeof(float);

    size_t total_size = size_of_log_likelihoods + size_of_resampling_indices + size_of_prev_sample_indices
                      + size_of_visibility_probs * 2 + size_of_opengl_textures + size_of_observations;

    if (total_size > cuda_device_properties_.totalGlobalMem) {

        std::cout << "The space (" << total_size << " B) for the number of maximum poses you requested (" << allocated_poses << ") cannot be allocated. "
                  << "The limit is global memory size (" << cuda_device_properties_.totalGlobalMem
                  << " B) retrieved from CUDA properties." << std::endl;

        size_t size_depending_on_nr_poses = (sizeof(float) + sizeof(int) * 2 + n_rows_ * n_cols_ * sizeof(float) * 4);
        allocated_poses = min(allocated_poses, (int) floor((cuda_device_properties_.totalGlobalMem - size_of_observations) / size_depending_on_nr_poses));
        allocated_poses_per_column = ceil(allocated_poses / allocated_poses_per_row);

        std::cout << "Instead, space (" << allocated_poses * size_depending_on_nr_poses + size_of_observations << " B) for " << allocated_poses << " poses was allocated. " << std::endl;
    }


    // check limitation by texture size
    if (cuda_device_properties_.maxTexture2D[0] <= allocated_poses_per_row * n_cols_) {

        std::cout << "The max poses you requested (" << allocated_poses << ") could not be allocated." << std::endl;

        allocated_poses_per_row = cuda_device_properties_.maxTexture2D[0] / n_cols_;
        allocated_poses_per_column = ceil(allocated_poses / allocated_poses_per_row);

        if (cuda_device_properties_.maxTexture2D[1] <= allocated_poses_per_column * n_rows_) {
            allocated_poses_per_column = cuda_device_properties_.maxTexture2D[1] / n_rows_;
        }

        allocated_poses = min(allocated_poses, allocated_poses_per_row * allocated_poses_per_column);

        std::cout << "The limit is max texture size (" << cuda_device_properties_.maxTexture2D[0]
                  << ", " << cuda_device_properties_.maxTexture2D[1] << ") retrieved from CUDA properties. "
                  << "Number of poses was reduced to (" << allocated_poses_per_row << ", "
                  << allocated_poses_per_column << "), a total of " << allocated_poses << std::endl;
    }

    nr_max_poses_ = allocated_poses;
    nr_max_poses_per_row_ = allocated_poses_per_row;
    nr_max_poses_per_column_ = allocated_poses_per_column;


/*
    nr_max_poses_ = n_poses;
    nr_max_poses_per_row_ = n_poses_x;

    // determine n_max_poses_y_
    nr_max_poses_per_column_ = nr_max_poses_ / nr_max_poses_per_row_;
    if (n_poses % nr_max_poses_per_row_ != 0) nr_max_poses_per_column_++;

    n_poses_ = nr_max_poses_;
    n_poses_x_ = nr_max_poses_per_row_;
    n_poses_y_ = nr_max_poses_per_column_;

*/
    n_poses_set_ = true;

    bool nr_poses_changed = false;
    set_default_kernel_config(nr_max_poses_, nr_max_poses_per_row_, nr_max_poses_per_column_, nr_poses_changed);

    allocated_poses = nr_max_poses_;
    allocated_poses_per_row = nr_max_poses_per_row_;
    allocated_poses_per_column = nr_max_poses_per_column_;

    nr_poses_ = nr_max_poses_;
    nr_poses_per_row_ = nr_max_poses_per_row_;
    nr_poses_per_column_ = nr_max_poses_per_column_;

    if (nr_poses_changed) {
        size_of_log_likelihoods = sizeof(float) * nr_max_poses_;
        size_of_resampling_indices = sizeof(int) * nr_max_poses_;
        size_of_prev_sample_indices = sizeof(int) * nr_max_poses_;
        size_of_visibility_probs = n_rows_ * n_cols_ * nr_max_poses_ * sizeof(float);
    }


    // reallocate arrays
    allocate(d_log_likelihoods_, size_of_log_likelihoods, "d_log_likelihoods");
    allocate(d_resampling_indices_, size_of_resampling_indices, "d_resampling_indices");
    allocate(d_prev_sample_indices_, size_of_prev_sample_indices, "d_prev_sample_indices");
    allocate(d_visibility_probs_, size_of_visibility_probs, "d_visibility_probs");
    allocate(d_visibility_probs_copy_, size_of_visibility_probs, "d_visibility_probs_copy");

    // TODO maybe delete after set_visibility_probabilities is properly in use
    vector<float> initial_visibility_probs (n_rows_ * n_cols_ * nr_max_poses_, visibility_prob_default_);
    cudaMemcpy(d_visibility_probs_, &initial_visibility_probs[0], size_of_visibility_probs, cudaMemcpyHostToDevice);
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaMemcpy visibility_prob_default_ -> d_visibility_probs_");
    #endif


    cudaDeviceSynchronize();
    #ifdef CHECK_ERRORS
        check_cuda_error("cudaDeviceSynchronize allocate_memory_for_max_poses");
    #endif


}


void CudaFilter::set_number_of_poses(int& nr_poses, int& nr_poses_per_row, int& nr_poses_per_column) {
    if (nr_poses <= nr_max_poses_) {

        if (nr_max_poses_per_row_ < nr_poses_per_row) {
            nr_poses_per_row = nr_max_poses_per_row_;
            nr_poses_per_column = ceil(nr_poses / nr_poses_per_row);
            if (nr_max_poses_per_column_ < nr_poses_per_column) {
                nr_poses_per_column = nr_max_poses_per_column_;
            }

            std::cout << "Number of poses was reduced to (" << nr_poses_per_row << ", "
                      << nr_poses_per_column << ") because of the maximum number of poses set in the beginning." << std::endl;
        }

        nr_poses = min(nr_poses, nr_poses_per_row * nr_poses_per_column);

        nr_poses_ = nr_poses;
        nr_poses_per_row_ = nr_poses_per_row;
        nr_poses_per_column_ = nr_poses_per_column;

        /*


        nr_poses_ = nr_poses;
        nr_poses_per_row_ = nr_poses_per_row;

        // determine n_max_poses_y_
        nr_poses_per_column_ = nr_poses_ / nr_poses_per_row_;
        if (nr_poses % nr_poses_per_row_ != 0) nr_poses_per_column_++;

        if (nr_poses_per_row_ > nr_max_poses_per_row_ || nr_poses_per_column_ > nr_max_poses_per_column_) {
            cout << "WARNING: You tried to evaluate more poses in a row or in a column than was allocated in the beginning."
                 << endl << "Check set_number_of_poses() functions in object_rasterizer.cpp" << endl;
        }*/

        bool nr_poses_changed = false;
        set_default_kernel_config(nr_poses_, nr_poses_per_row_, nr_poses_per_column_, nr_poses_changed);

        nr_poses = nr_poses_;
        nr_poses_per_row = nr_poses_per_row_;
        nr_poses_per_column = nr_poses_per_column_;

    } else {
        cout << "ERROR (Cuda): You tried to evaluate more poses (" << nr_poses << ") than specified by max_poses (" << nr_max_poses_ << ")" << endl;
        exit(-1);
    }
}



void CudaFilter::set_default_kernel_config(int& nr_poses, int& nr_poses_per_row, int& nr_poses_per_column,
                                           bool& nr_poses_changed) {
    nr_threads_ = min(DEFAULT_NR_THREADS, cuda_device_properties_.maxThreadsDim[0]);

    // check for grid dimension limitations
    if (cuda_device_properties_.maxGridSize[0] < nr_poses_per_row) {
        nr_poses_per_row = cuda_device_properties_.maxGridSize[0];
        nr_poses_per_column = ceil(nr_poses / nr_poses_per_row);
        if (cuda_device_properties_.maxGridSize[1] < nr_poses_per_column) {
            nr_poses_per_column = cuda_device_properties_.maxGridSize[1];
        }

        nr_poses = min(nr_poses, nr_poses_per_row * nr_poses_per_column);

        nr_poses_changed = true;

        std::cout << "Number of poses was reduced to (" << nr_poses_per_row << ", "
                  << nr_poses_per_column << ") because of the maximum grid size ("
                  << cuda_device_properties_.maxGridSize[0] << ", " << cuda_device_properties_.maxGridSize[1]
                  << ") retrieved from CUDA properties." << std::endl;
    }


    grid_dimension_ = dim3(nr_poses_per_row, nr_poses_per_column);


    /*

    // determine n_threads_ and n_blocks_
    // n_threads_ should lie between 32 (warp_size) and 128 and all microprocessors should be busy
    nr_threads_ = ((nr_poses_ / n_mps_) / warp_size_) * warp_size_;
    if (nr_threads_ == 0) nr_threads_ = warp_size_;
    if (nr_threads_ > 4 * warp_size_) nr_threads_ = 4 * warp_size_;

    n_blocks_ = nr_poses_ / nr_threads_;
    if (n_blocks_ % nr_poses_ != 0) n_blocks_++;

    */
}


void CudaFilter::set_texture_array(cudaArray_t texture_array) {
    d_texture_array_ = texture_array;
}


// ===================================================================================== //
// =============================== CUDA FILTER GETTERS ================================= //
// ===================================================================================== //




vector<float> CudaFilter::get_visibility_probabilities(int state_id) {
//    cout << "n_rows_: " << n_rows_ << ", n_cols_: " << n_cols_ << endl;
    float* visibility_probabilities = (float*) malloc(n_rows_ * n_cols_ * sizeof(float));
    int offset = state_id * n_rows_ * n_cols_;
    cudaMemcpy(visibility_probabilities, d_visibility_probs_ + offset, n_rows_ * n_cols_ * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef CHECK_ERRORS
    check_cuda_error("cudaMemcpy d_visibility_probabilities -> visibility_probabilities");
#endif
    vector<float> visibility_probabilities_vector;
    for (int i = 0; i < n_rows_ * n_cols_; i++) {
        visibility_probabilities_vector.push_back(visibility_probabilities[i]);
    }
    free(visibility_probabilities);
    return visibility_probabilities_vector;
}



vector<vector<float> > CudaFilter::get_visibility_probabilities() {
    float* visibility_probabilities = (float*) malloc(nr_poses_ * n_rows_ * n_cols_ * sizeof(float));
    cudaMemcpy(visibility_probabilities, d_visibility_probs_, nr_poses_ * n_rows_ * n_cols_ * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef CHECK_ERRORS
    check_cuda_error("cudaMemcpy d_visibility_probabilities -> visibility_probabilities");
#endif
    vector<vector<float> > visibility_probabilities_vector;
    vector<float> tmp_vector (n_rows_ * n_cols_);
    for (int i = 0; i < nr_poses_; i++) {
        for (int j = 0; j < n_rows_ * n_cols_; j++) {
            tmp_vector[j] = visibility_probabilities[i * n_rows_ * n_cols_ + j];
        }
        visibility_probabilities_vector.push_back(tmp_vector);
    }
    return visibility_probabilities_vector;
}



// ===================================================================================== //
// ========================== CUDA FILTER HELPER FUNCTIONS ============================= //
// ===================================================================================== //




template <typename T> void CudaFilter::allocate(T * &pointer, size_t size, string name) {
#ifdef CHECK_ERRORS
    size_t free_space_before, free_space_after, total_space;
    cuMemGetInfo(&free_space_before, &total_space);
#endif
    cudaFree(pointer);
    cudaMalloc((void **) &pointer, size);
#ifdef CHECK_ERRORS
    cuMemGetInfo(&free_space_after, &total_space);
    std::cout << "memory to allocate for " << name << ": " << size / 1e6 << " MB; free space: " << free_space_before / 1e6
         << "MB; --> allocated " << (free_space_before - free_space_after) / 1e6 << " MB, free space left: " << free_space_after / 1e6 << " MB" << std::endl;
    check_cuda_error("cudaMalloc failed");
#endif
}



void CudaFilter::map_texture() {
    cudaBindTextureToArray(texture_reference, d_texture_array_);
    check_cuda_error("cudaBindTextureToArray");
}




void CudaFilter::check_cuda_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

CudaFilter::~CudaFilter() {
    cudaFree(d_states_);
    cudaFree(d_states_copy_);
    cudaFree(d_visibility_probs_);
    cudaFree(d_visibility_probs_copy_);
    cudaFree(d_observations_);
    cudaFree(d_log_likelihoods_);
    cudaFree(d_mrg_states_);
    cudaFree(d_resampling_indices_);

}

void CudaFilter::destroy_context() {
    cudaFree(d_states_);
    cudaFree(d_states_copy_);
    cudaFree(d_visibility_probs_);
    cudaFree(d_visibility_probs_copy_);
    cudaFree(d_observations_);
    cudaFree(d_log_likelihoods_);
    cudaFree(d_mrg_states_);
    cudaFree(d_resampling_indices_);
    cudaDeviceReset();
}

}

