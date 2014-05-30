
#define ROS_ASSERT_ENABLED
#include <ros/assert.h>

#include "filters/spkf/sigma_point_matrix.hpp"

using namespace filter;

// state less implementation

SigmaPointMatrix::SigmaPointMatrix(size_type numberOfPoints)
{
    if (numberOfPoints > 0)
    {
        meanWeights_.resize(numberOfPoints);
        covWeights_.resize(numberOfPoints);
    }

    MatrixType::resize(1, 1);
}

SigmaPointMatrix::~SigmaPointMatrix()
{

}

double SigmaPointMatrix::meanWeight(int index) const
{
    return meanWeights_[index];
}

double SigmaPointMatrix::covWeight(int index)  const
{
    return covWeights_[index];
}

void SigmaPointMatrix::resize(int numberOfPoints, int dimension)
{
    MatrixType::resize(dimension, numberOfPoints);

    meanWeights_.resize(numberOfPoints);
    covWeights_.resize(numberOfPoints);
}

void SigmaPointMatrix::point(int index,
                             const DynamicVector& point,
                             double meanWeight,
                             double covWeight)
{
    if (col(index).rows() != point.rows())
    {
        ROS_ASSERT_MSG(index == 0, "Only the first column can induce a change in dimension.");

        MatrixType::resize(point.rows(), Eigen::NoChange);
    }

    col(index) = point;
    meanWeights_[index] = meanWeight;
    covWeights_[index] = covWeight;
}

void  SigmaPointMatrix::mean(DynamicVector& _mean, int subDimension, int marginBegin) const
{
    SigmaPointMatrix::size_type numberOfPoints = cols();

    subDimension = subDimension == 0 ? col(0).rows() : subDimension;

    // adjust mean dimension if needed
    if (_mean.rows() != subDimension)
    {
        _mean.resize(subDimension, 1);
    }

    _mean.setZero();

    for (SigmaPointMatrix::size_type i = 0; i < numberOfPoints; i++)
    {
        _mean += meanWeights_[i] * col(0).segment(marginBegin, subDimension);
    }
}

void SigmaPointMatrix::covariance(DynamicMatrix& cov, int subDimension, int marginBegin) const
{
    MatrixType wzmMatrix;
    weightedZeroMeanSigmaPointMatrix(wzmMatrix, subDimension, marginBegin);

    cov = wzmMatrix * wzmMatrix.transpose();
}

void SigmaPointMatrix::covariance(const DynamicVector& _mean,
                                  DynamicMatrix& cov,
                                  int subDimension,
                                  int marginBegin) const
{
    MatrixType zmMatrix;
    zeroMeanSigmaPointMatrix(_mean, zmMatrix, subDimension, marginBegin);

    // TODO STDCOUT
    //std::cout << "wzmMatrix" << std::endl << wzmMatrix << std::endl;

    cov = zmMatrix * covWeights_.asDiagonal() * zmMatrix.transpose();
}

void SigmaPointMatrix::weightedZeroMeanSigmaPointMatrix(SigmaPointMatrix::MatrixType& wzmMatrix,
                                                        int subDimension,
                                                        int marginBegin) const
{
    DynamicVector _mean;
    mean(_mean, subDimension, marginBegin);

    weightedZeroMeanSigmaPointMatrix(_mean, wzmMatrix, subDimension, marginBegin);
}

void SigmaPointMatrix::weightedZeroMeanSigmaPointMatrix(const DynamicVector &_mean,
                                                        SigmaPointMatrix::MatrixType &wzmMatrix,
                                                        int subDimension,
                                                        int marginBegin) const
{
    SigmaPointMatrix::size_type numberOfPoints = cols();

    subDimension = subDimension == 0 ? col(0).rows() : subDimension;

    wzmMatrix.resize(subDimension, numberOfPoints);
    for (SigmaPointMatrix::size_type i = 0; i < numberOfPoints; i++)
    {
        wzmMatrix.col(i) = col(i).segment(marginBegin, subDimension) - _mean;
        wzmMatrix.col(i) *= std::sqrt(covWeight(i));
    }
}

void SigmaPointMatrix::zeroMeanSigmaPointMatrix(SigmaPointMatrix::MatrixType& zmMatrix,
                                                int subDimension,
                                                int marginBegin) const
{
    DynamicVector _mean;
    mean(_mean, subDimension, marginBegin);

    zeroMeanSigmaPointMatrix(_mean, zmMatrix, subDimension, marginBegin);
}

void SigmaPointMatrix::zeroMeanSigmaPointMatrix(const DynamicVector &_mean,
                                                SigmaPointMatrix::MatrixType &zmMatrix,
                                                int subDimension,
                                                int marginBegin) const
{
    SigmaPointMatrix::size_type numberOfPoints = cols();

    subDimension = subDimension == 0 ? col(0).rows() : subDimension;

    zmMatrix.resize(subDimension, numberOfPoints);
    for (SigmaPointMatrix::size_type i = 0; i < numberOfPoints; i++)
    {
        zmMatrix.col(i) = col(i).segment(marginBegin, subDimension) - _mean;
    }
}

const DynamicVector& SigmaPointMatrix::meanWeights() const
{
    return meanWeights_;
}

const DynamicVector& SigmaPointMatrix::covarianceWeights() const
{
    return covWeights_;
}



/* efficient implementation
SigmaPointMatrix::SigmaPointMatrix(size_type numberOfPoints):
    dirty_(true)

{
    if (numberOfPoints > 0)
    {
        meanWeights_.resize(numberOfPoints);
        covWeights_.resize(numberOfPoints);
    }

    oldSubDimension = -1;
    oldMarginBegin = -1;

    MatrixType::resize(1, 1);
}

SigmaPointMatrix::~SigmaPointMatrix()
{

}

double SigmaPointMatrix::meanWeight(int index) const
{
    return meanWeights_[index];
}

double SigmaPointMatrix::covWeight(int index)  const
{
    return covWeights_[index];
}

void SigmaPointMatrix::resize(int numberOfPoints, int dimension)
{
    MatrixType::resize(dimension, numberOfPoints);

    meanWeights_.resize(numberOfPoints);
    covWeights_.resize(numberOfPoints);

    dirty_ = true;
}

void SigmaPointMatrix::point(int index,
                             const DynamicVector& point,
                             double meanWeight,
                             double covWeight)
{
    if (col(index).rows() != point.rows())
    {
        ROS_ASSERT_MSG(index == 0, "Only the first column can induce a change in dimension.");

        MatrixType::resize(point.rows(), Eigen::NoChange);
    }

    col(index) = point;
    meanWeights_[index] = meanWeight;
    covWeights_[index] = covWeight;

    dirty_ = true;
}

const DynamicVector& SigmaPointMatrix::mean(int subDimension, int marginBegin)
{
    if (!dirty_ &&
        oldSubDimension == subDimension &&
        oldMarginBegin == marginBegin)
    {
        return mean_;
    }

    SigmaPointMatrix::size_type numberOfPoints = cols();

    subDimension = subDimension == 0 ? col(0).rows() : subDimension;

    // adjust mean dimension if needed
    if (mean_.rows() != subDimension)
    {
        mean_.resize(subDimension, 1);
    }

    mean_.setZero();

    for (SigmaPointMatrix::size_type i = 0; i < numberOfPoints; i++)
    {
        mean_ += meanWeights_[i] * col(0).segment(marginBegin, subDimension);
    }    

    // pre compute the factor matrix
    weightedZeroMeanSigmaPointMatrix(subDimension, marginBegin);

    dirty_ = false;
    oldSubDimension = subDimension;
    oldMarginBegin = marginBegin;    

    return mean_;
}

DynamicMatrix SigmaPointMatrix::covariance()
{
    return wzmMatrix_ * wzmMatrix_.transpose();
}

const SigmaPointMatrix::MatrixType& SigmaPointMatrix::weightedZeroMeanSigmaPointMatrix() const
{
    return wzmMatrix_;
}

void SigmaPointMatrix::weightedZeroMeanSigmaPointMatrix(int subDimension, int marginBegin)
{
    SigmaPointMatrix::size_type numberOfPoints = cols();

    subDimension = subDimension == 0 ? col(0).rows() : subDimension;

    wzmMatrix_.resize(subDimension, numberOfPoints);

    DynamicVector zeroMeanPoint;
    for (SigmaPointMatrix::size_type i = 0; i < numberOfPoints; i++)
    {
        zeroMeanPoint = col(i).segment(marginBegin, subDimension) - mean_;

        wzmMatrix_.col(i) = std::sqrt(covWeight(i)) * zeroMeanPoint;
    }
}

bool SigmaPointMatrix::hasBeenModified()
{
    return dirty_;
}
*/
