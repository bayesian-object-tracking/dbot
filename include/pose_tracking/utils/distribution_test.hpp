#ifndef FAST_FILTERING_UTILS_DISTRIBUTION_TEST_HPP
#define FAST_FILTERING_UTILS_DISTRIBUTION_TEST_HPP

#include <iostream>

#include <boost/filesystem.hpp>
#include <limits>

#include <algorithm>

#include <fl/util/discrete_distribution.hpp>

namespace fl
{

template <typename T>
int IndexNextReal(
        const std::vector<T>& data,
        const int& current_index = -1,
        const int& step_size = 1)
{
    int index_next_real = current_index + step_size;
    while(
            index_next_real < int(data.size()) &&
            index_next_real >= 0 &&
            !std::isfinite(data[index_next_real])) index_next_real+=step_size;

    return index_next_real;
}

// this function will interpolat the vector wherever it is NAN or INF
template <typename T>
void LinearlyInterpolate(std::vector<T>& data)
{
    std::vector<int> limits;
    limits.push_back(0);
    limits.push_back(data.size()-1);
    std::vector<int> step_direction;
    step_direction.push_back(1);
    step_direction.push_back(-1);

    // extrapolate
    for(int i = 0; i < 2; i++)
    {
        int index_first_real = IndexNextReal(data, limits[i] - step_direction[i], step_direction[i]);
        int index_next_real = IndexNextReal(data, index_first_real, step_direction[i]);
        if(index_next_real >= int(data.size()) || index_next_real < 0)
            return;

        double slope =
                double(data[index_next_real] - data[index_first_real]) /
                double(index_next_real - index_first_real);

        for(int j = limits[i]; j != index_first_real; j += step_direction[i])
            data[j] = data[index_first_real] + (j - index_first_real) * slope;
    }

    // interpolate
    int index_current_real = IndexNextReal(data, limits[0] - step_direction[0], step_direction[0]);
    int index_next_real = IndexNextReal(data, index_current_real, step_direction[0]);

    while(index_next_real < int(data.size()))
    {
        double slope =
                double(data[index_next_real] - data[index_current_real]) /
                double(index_next_real - index_current_real);

        for(int i = index_current_real + 1; i < index_next_real; i++)
            data[i] =  data[index_current_real] + (i - index_current_real) * slope;

        index_current_real = index_next_real;
        index_next_real = IndexNextReal(data, index_next_real, step_direction[0]);
    }
}

template<typename Distribution>
void TestDistribution(Distribution distribution, size_t sample_count = 100000, size_t evaluation_count = 100000)
{
    // for now we assume that it is just a univariate distribution
    std::vector<double> samples;
    for(size_t i = 0; i < sample_count; i++)
        samples.push_back(distribution.sample());

    std::vector<double> evaluation_inputs(evaluation_count, std::numeric_limits<double>::quiet_NaN());

    double min = *(std::min_element(samples.begin(), samples.end()));
    double max = *(std::max_element(samples.begin(), samples.end()));
    evaluation_inputs[0] = min - (max - min)*0.1;
    evaluation_inputs.back() = max + (max - min)*0.1;
    LinearlyInterpolate(evaluation_inputs);

    std::vector<double> evaluation_outputs(evaluation_count, 0);
    for(size_t i = 0; i < evaluation_inputs.size(); i++)
        evaluation_outputs[i] = distribution.probability(evaluation_inputs[i]);

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
        samples.push_back(distribution.sample()(0,0));

    std::vector<double> evaluation_inputs(sample_count, std::numeric_limits<double>::quiet_NaN());
    double min = *(std::min_element(samples.begin(), samples.end()));
    double max = *(std::max_element(samples.begin(), samples.end()));
    evaluation_inputs[0] = min - (max - min)*0.1;
    evaluation_inputs.back() = max + (max - min)*0.1;
    LinearlyInterpolate(evaluation_inputs);

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
