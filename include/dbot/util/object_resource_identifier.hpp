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
 * \file object_resource_identifier.hpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date November 2015
 */

#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <exception>

namespace dbot
{

/**
 * \brief Represents an exception thrown if the package path is invalid, e.g.
 *        empty or missing leading '/'
 */
class InvalidPackagePathException : public std::exception
{
public:
    const char* what() const noexcept
    {
        return "Package path must be of the form '/path/to/packagename'";
    }
};

/**
 * \brief Represents an exception thrown if no package name has been specified
 *        a missing package name is triggered when the path is either empty
 *        or consists only of '/'
 */
class MissingPackageNameInPathException : public InvalidPackagePathException
{
};

class ObjectResourceIdentifier
{
public:
    /**
     * \brief Creates an empty ObjectResourceIdentifier
     */
    ObjectResourceIdentifier();

    /**
     * \brief Creates an ObjectResourceIdentifier
     */
    ObjectResourceIdentifier(const std::string& package_path_,
                             const std::string& directory_,
                             const std::vector<std::string>& meshes_);

    /**
     * \brief Default virtual destructor
     */
    virtual ~ObjectResourceIdentifier();

public:
    /**
     * \brief Returns the mesh URI of the i-th mesh file
     *        ("package://<package>/<path>/<meshes[i]>")
     * \param [in]  i  Index of the requested mesh file
     */
    std::string mesh_uri(size_t i) const;

    /**
     * \brief Returns the absolute path of the i-th mesh file
     *        ("<package_path>/<path>/<meshes[i]>")
     * \param [in]  i  Index of the requested mesh file
     */
    std::string mesh_path(size_t i) const;

    /**
     * \brief package name if using catkin (ros) packages.
     *
     * This is used to generate a URI of the form
     *   "package://<PACKAGE>/<path>/<meshes[i]>"
     */
    const std::string& package() const;

    /**
     * \brief Top level package path which contains the objects (note: this path
     *        includes the package name)
     */
    const std::string& package_path() const;

    /**
     * \brief Path within the package pointing to the mesh files
     *
     *   "package://<package>/<PATH>/<meshes[i]>"
     */
    const std::string& directory() const;

    /**
     * \brief Returns the number of mesh files
     */
    int count_meshes() const;

    /**
     * \brief Returns the list of mesh filenames
     */
    const std::vector<std::string>& meshes() const;

    /**
     * \brief Returns the i-th mesh file name
     */
    std::string mesh(int i) const;

    /**
     * \brief Returns the i-th mesh file name without file type extension
     */
    std::string mesh_without_extension(int i) const;

public:
    /**
     * \brief Sets the package top level path. This must include the package
     * name. The package name is therefore deduced from this package path
     */
    void package_path(const std::string& package_path_);

//    /**
//     * \brief Sets the package name when the package name differs from the
//     *        package top-level directory
//     */
//    void package(const std::string& package_name);

    /**
     * \brief Sets the relative path within the package pointing to the mesh
     * file
     *        location
     */
    void directory(const std::string& directory_);

    /**
     * \brief Sets mesh filenames
     */
    void meshes(const std::vector<std::string>& meshes_);

private:
    /**
     * \brief package name if using catkin (ros) packages.
     *
     * This is used to generate a URI of the form
     *   "package://<PACKAGE>/<path>/<meshes[i]>"
     */
    std::string package_;

    /**
     * \brief Top level package path which contains the objects.
     */
    std::string package_path_;

    /**
     * \brief Path within the package pointing to the mesh files
     *
     *   "package://<package>/<PATH>/<meshes[i]>"
     */
    std::string directory_;

    /**
     * \brief Mesh file names
     *
     *
     *   "package://<package>/<path>/<MESHES[i]>"
     */
    std::vector<std::string> meshes_;
};
}
