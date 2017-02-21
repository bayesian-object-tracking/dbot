/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file object_resource_identifier.cpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date November 2015
 */

#include <dbot/object_resource_identifier.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>

namespace dbot
{
ObjectResourceIdentifier::ObjectResourceIdentifier()
{
}

ObjectResourceIdentifier::ObjectResourceIdentifier(
    const std::string& package_path_,
    const std::string& path_,
    const std::vector<std::string>& meshes_)
{
    package_path(package_path_);
    directory(path_);
    meshes(meshes_);
}

ObjectResourceIdentifier::~ObjectResourceIdentifier()
{
}

const std::string& ObjectResourceIdentifier::package() const
{
    return package_;
}

const std::string& ObjectResourceIdentifier::package_path() const
{
    return package_path_;
}

const std::string& ObjectResourceIdentifier::directory() const
{
    return directory_;
}

std::string ObjectResourceIdentifier::mesh_uri(size_t i) const
{
    assert(i < meshes_.size());

    boost::filesystem::path p(package());
    p /= directory_;
    p /= meshes_[i];

    return "package://" + p.string();
}

std::string ObjectResourceIdentifier::mesh_path(size_t i) const
{
    assert(i < meshes_.size());

    boost::filesystem::path p(package_path_);
    p /= directory_;
    p /= meshes_[i];

    return p.string();
}

int ObjectResourceIdentifier::count_meshes() const
{
    return meshes_.size();
}

const std::vector<std::string>& ObjectResourceIdentifier::meshes() const
{
    return meshes_;
}

std::string ObjectResourceIdentifier::mesh(int i) const
{
    return meshes_[i];
}

std::string ObjectResourceIdentifier::mesh_without_extension(int i) const
{
    return boost::filesystem::change_extension(mesh(i), "").string();
}

void ObjectResourceIdentifier::package_path(const std::string& package_path_)
{
    std::string pp = package_path_;

    boost::trim(pp);

    if (pp.empty() || pp.compare("/") == 0)
    {
        throw MissingPackageNameInPathException();
    }

    // should be made mode portable!
    if (!boost::starts_with(pp, "/"))
    {
        throw InvalidPackagePathException();
    }

    this->package_path_ = pp;
    this->package_ = boost::filesystem::path(pp).filename().string();
}

void ObjectResourceIdentifier::directory(const std::string& directory_)
{
    this->directory_ = directory_;
}

void ObjectResourceIdentifier::meshes(const std::vector<std::string>& meshes_)
{
    this->meshes_ = meshes_;
}

void ObjectResourceIdentifier::mesh(const std::string& mesh_)
{
   this->meshes_.clear();
   this->meshes_.push_back(mesh_);
}

}
