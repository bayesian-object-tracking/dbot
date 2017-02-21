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
 * \file object_file_reader.h
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <list>

#include <Eigen/Core>

#include <fl/exception/exception.hpp>

namespace dbot
{

class CannotOpenWavefrontFileException:
        public fl::Exception
{
public:
    /**
     * Creates an WrongSizeException with a customized message
     */
    CannotOpenWavefrontFileException(std::string msg)
        : Exception()
    {
        info("File", msg);
    }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept
    {
        return "dbot::CannotOpenWavefrontFileException";
    }

};

class ObjectFileReader
{
public:
	ObjectFileReader();
	~ObjectFileReader(){}

	void set_filename(std::string filename);
	void Read();
	void Process(float max_side_length);

    std::shared_ptr<std::vector<Eigen::Vector3d> > get_vertices();
    std::shared_ptr<std::vector<std::vector<int> > > get_indices();


    std::shared_ptr<std::vector<Eigen::Vector3d> > get_centers();
    std::shared_ptr<std::vector<float> > get_areas();

private:
	std::string filename_;
    std::shared_ptr<std::vector<Eigen::Vector3d> > vertices_;
    std::shared_ptr<std::vector<std::vector<int> > > indices_;

    std::shared_ptr<std::vector<Eigen::Vector3d> > centers_;
    std::shared_ptr<std::vector<float> > areas_;
};

}
