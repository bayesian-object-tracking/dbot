#ifndef FAST_FILTERING_UTILS_DISTRIBUTION_TEST_HPP
#define FAST_FILTERING_UTILS_DISTRIBUTION_TEST_HPP

#include <iostream>

#include <ff/utils/helper_functions.hpp>

#include <boost/filesystem.hpp>
#include <limits>


namespace fl
{

template<typename Distribution>
void TestDistribution(Distribution distribution, size_t sample_count = 100000, size_t evaluation_count = 100000)
{
    // for now we assume that it is just a univariate distribution
    std::vector<double> samples;
    for(size_t i = 0; i < sample_count; i++)
        samples.push_back(distribution.Sample());

    std::vector<double> evaluation_inputs(evaluation_count, std::numeric_limits<double>::quiet_NaN());
    double min = hf::bound_value(samples, false);
    double max = hf::bound_value(samples, true);
    evaluation_inputs[0] = min - (max - min)*0.1;
    evaluation_inputs.back() = max + (max - min)*0.1;
    hf::LinearlyInterpolate(evaluation_inputs);

    std::vector<double> evaluation_outputs(evaluation_count, 0);
    for(size_t i = 0; i < evaluation_inputs.size(); i++)
        evaluation_outputs[i] = distribution.Probability(evaluation_inputs[i]);

    boost::filesystem::path path = "/tmp";
    path /= "distribution_test";
    boost::filesystem::create_directory(path);
    std::ofstream file;

    double mean = 0;
    double variance = 0;

    for (auto& sample: samples)
    {
        mean += sample;
    }
    mean /= double(samples.size());

    for (auto& sample: samples)
    {
        variance += (sample-mean) * (sample-mean);
    }
    variance /= double(samples.size()-1);

    // write samples to file
    file.open((path / "gaussian.txt").c_str(), std::ios::out);
    file << mean << " " << variance << std::endl;
    file.close();

    // write samples to file
    file.open((path / "samples.txt").c_str(), std::ios::out);
    if(file.is_open())
    {
        for(size_t i = 0; i < samples.size(); i++)
            file << samples[i] << std::endl;
        file.close();
    }

    // write evaluation to file
    file.open((path / "evaluation.txt").c_str(), std::ios::out);
    if(file.is_open())
    {
        for(size_t i = 0; i < evaluation_inputs.size(); i++)
        {
            file << evaluation_inputs[i] << " ";
            file << evaluation_outputs[i] << std::endl;
        }
        file.close();
    }
}




template<typename Distribution>
void TestDistributionSampling(Distribution distribution, size_t sample_count = 100000)
{
    // for now we assume that it is just a univariate distribution
    std::vector<double> samples;
    for(size_t i = 0; i < sample_count; i++)
        samples.push_back(distribution.Sample()(0,0));

    std::vector<double> evaluation_inputs(sample_count, std::numeric_limits<double>::quiet_NaN());
    double min = hf::bound_value(samples, false);
    double max = hf::bound_value(samples, true);
    evaluation_inputs[0] = min - (max - min)*0.1;
    evaluation_inputs.back() = max + (max - min)*0.1;
    hf::LinearlyInterpolate(evaluation_inputs);

    std::vector<double> evaluation_outputs(sample_count, 0);

    boost::filesystem::path path = "/tmp";
    path /= "distribution_test";
    boost::filesystem::create_directory(path);
    std::ofstream file;


    // write samples to file
    file.open((path / "samples.txt").c_str(), std::ios::out);
    if(file.is_open())
    {
        for(size_t i = 0; i < samples.size(); i++)
            file << samples[i] << std::endl;
        file.close();
    }

    // write evaluation to file
    file.open((path / "evaluation.txt").c_str(), std::ios::out);
    if(file.is_open())
    {
        for(size_t i = 0; i < evaluation_inputs.size(); i++)
        {
            file << evaluation_inputs[i] << " ";
            file << evaluation_outputs[i] << std::endl;
        }
        file.close();
    }
}




}


#endif
