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

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>

#include <dbot/pose/rigid_bodies_state.h>

namespace dbot
{
class RigidBodyRenderer
{
public:
    typedef std::shared_ptr<RigidBodyRenderer> Ptr;
    typedef dbot::RigidBodiesState<Eigen::Dynamic> State;
    typedef Eigen::Vector3d Vector;
    typedef Eigen::Matrix3d Matrix;
    typedef typename Eigen::Transform<double, 3, Eigen::Affine> Affine;

    RigidBodyRenderer(
        const std::vector<std::vector<Eigen::Vector3d>>& vertices,
        const std::vector<std::vector<std::vector<int>>>& indices);

    RigidBodyRenderer(const std::vector<std::vector<Eigen::Vector3d>>& vertices,
                      const std::vector<std::vector<std::vector<int>>>& indices,
                      Matrix camera_matrix,
                      int n_rows,
                      int n_cols);

    virtual ~RigidBodyRenderer();

    void Render(Matrix camera_matrix,
                int n_rows,
                int n_cols,
                std::vector<int>& intersect_indices,
                std::vector<float>& depth) const;

    void Render(Matrix camera_matrix,
                int n_rows,
                int n_cols,
                std::vector<float>& depth_image) const;

    void Render(std::vector<float>& depth_image) const;

    template <typename RigidbodyState>
    void Render(const RigidbodyState& state, std::vector<float>& depth_vector)

    {
        std::vector<Eigen::Matrix3d> rotations(state.count_parts());
        std::vector<Eigen::Vector3d> translations(state.count_parts());
        for (size_t i = 0; i < rotations.size(); i++)
        {
            rotations[i] = state.component(i).orientation().rotation_matrix();
            translations[i] = state.component(i).position();
        }

        set_poses(rotations, translations);

        Render(camera_matrix_, n_rows_, n_cols_, depth_vector);
    }

    template <typename RigidbodyState>
    void Render(const RigidbodyState& state,
                Eigen::VectorXd& depth_image,
                double bad_value = std::numeric_limits<double>::infinity())
    {
        std::vector<float> depth_vector;
        Render(state, depth_vector);

        depth_image.resize(depth_vector.size(), 1);
        for (int i = 0; i < depth_vector.size(); ++i)
        {
            float value = depth_vector[i];
            if (std::isfinite(value))
            {
                depth_image(i, 0) = double(value);
            }
            else
            {
                depth_image(i, 0) = bad_value;
            }
        }
    }

    template <typename RigidbodyState>
    void Render(const RigidbodyState& state,
                Eigen::VectorXf& depth_image,
                float bad_value = std::numeric_limits<float>::infinity())
    {
        std::vector<float> depth_vector;
        Render(state, depth_vector);

        depth_image.resize(depth_vector.size());
        for (int i = 0; i < depth_vector.size(); ++i)
        {
            float value = depth_vector[i];
            if (std::isfinite(value))
            {
                depth_image[i] = float(value);
            }
            else
            {
                depth_image[i] = std::numeric_limits<float>::infinity();
            }
        }
    }

    std::vector<std::vector<Vector>> vertices() const;

    virtual void set_poses(const std::vector<Matrix>& rotations,
                           const std::vector<Vector>& translations);

    virtual void set_poses(const std::vector<Affine>& poses);

    void parameters(Matrix camera_matrix, int n_rows, int n_cols);

private:
    /**
     * Because c++0x on gcc.4.6 does not implement delegating constructors
     */
    void init();

    // protected:
public:
    Matrix camera_matrix_;
    int n_rows_;
    int n_cols_;

    // triangles
    std::vector<std::vector<Vector>> vertices_;
    std::vector<std::vector<Vector>> normals_;
    std::vector<std::vector<std::vector<int>>> indices_;

    // state
    std::vector<Matrix> R_;
    std::vector<Vector> t_;

    // cached center of mass
    std::vector<Vector> coms_;
    std::vector<float> com_weights_;
};
}
