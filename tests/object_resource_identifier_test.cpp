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

#include <gtest/gtest.h>

#include <dbot/util/object_resource_identifier.hpp>

TEST(ObjectResourceIndentifierTests, package_long_path)
{
    EXPECT_EQ(dbot::ObjectResourceIdentifier("/path/to/workspace/mypackage",
                                             "/path/within/package/to/object",
                                             {"myobject.obj"})
                  .package()
                  .compare("mypackage"),
              0);

    EXPECT_EQ(dbot::ObjectResourceIdentifier(" /path/to/workspace/mypack age ",
                                             "/path/within/package/to/object",
                                             {"myobject.obj"})
                  .package()
                  .compare("mypack age"),
              0);
}

TEST(ObjectResourceIndentifierTests, package_name_only)
{
    EXPECT_EQ(
        dbot::ObjectResourceIdentifier(
            "/mypackage", "/path/within/package/to/object", {"myobject.obj"})
            .package()
            .compare("mypackage"),
        0);

    EXPECT_EQ(
        dbot::ObjectResourceIdentifier(
            " /mypackage ", "/path/within/package/to/object", {"myobject.obj"})
            .package()
            .compare("mypackage"),
        0);

    EXPECT_EQ(dbot::ObjectResourceIdentifier("/path/to/pkg/mypackage ",
                                             "/path/within/package/to/object",
                                             {"myobject.obj"})
                  .package()
                  .compare("mypackage"),
              0);
}

TEST(ObjectResourceIndentifierTests, package_name_missing)
{
    EXPECT_THROW(dbot::ObjectResourceIdentifier(
                     "/", "/path/within/package/to/object", {"myobject.obj"}),
                 dbot::MissingPackageNameInPathException);

    EXPECT_THROW(dbot::ObjectResourceIdentifier(
                     " / ", "/path/within/package/to/object", {"myobject.obj"}),
                 dbot::MissingPackageNameInPathException);

    EXPECT_THROW(dbot::ObjectResourceIdentifier(
                     " ", "/path/within/package/to/object", {"myobject.obj"}),
                 dbot::MissingPackageNameInPathException);

    EXPECT_THROW(dbot::ObjectResourceIdentifier(
                     "", "/path/within/package/to/object", {"myobject.obj"}),
                 dbot::MissingPackageNameInPathException);
}

TEST(ObjectResourceIndentifierTests, package_missing_leading_slash)
{
    EXPECT_THROW(
        dbot::ObjectResourceIdentifier(
            "mypackage", "/path/within/package/to/object", {"myobject.obj"}),
        dbot::InvalidPackagePathException);

    EXPECT_THROW(
        dbot::ObjectResourceIdentifier(
            " mypackage ", "/path/within/package/to/object", {"myobject.obj"}),
        dbot::InvalidPackagePathException);
}

TEST(ObjectResourceIndentifierTests, mesh_count)
{
    EXPECT_EQ(
        dbot::ObjectResourceIdentifier("/mypackage",
                                       "/path/within/package/to/object",
                                       {"myobject.obj", "myotherobject.obj"})
            .count_meshes(),
        2);
}

TEST(ObjectResourceIndentifierTests, mesh_uri)
{
    EXPECT_EQ(
        dbot::ObjectResourceIdentifier(
            "/mypackage", "/path/within/package/to/object", {"myobject.obj"})
            .mesh_uri(0)
            .compare("package://mypackage/path/within/package/to/object/"
                     "myobject.obj"),
        0);
    EXPECT_EQ(dbot::ObjectResourceIdentifier("/path/to/pkg/mypackage",
                                             "/path/within/package/to/object",
                                             {"myobject.obj"})
                  .mesh_uri(0)
                  .compare("package://mypackage/path/within/package/to/object/"
                           "myobject.obj"),
              0);
}

TEST(ObjectResourceIndentifierTests, mesh_path)
{
    EXPECT_EQ(
        dbot::ObjectResourceIdentifier(
            "/mypackage", "/path/within/package/to/object", {"myobject.obj"})
            .mesh_path(0)
            .compare("/mypackage/path/within/package/to/object/"
                     "myobject.obj"),
        0);

    EXPECT_EQ(
        dbot::ObjectResourceIdentifier("/path/to/pkg/mypackage",
                                       "/path/within/package/to/object",
                                       {"myobject.obj"})
            .mesh_path(0)
            .compare("/path/to/pkg/mypackage/path/within/package/to/object/"
                     "myobject.obj"),
        0);
}
