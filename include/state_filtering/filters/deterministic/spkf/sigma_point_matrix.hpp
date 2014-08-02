
#ifndef FILTERS_SPKF_SIGMA_POINT_MATRIX_HPP
#define FILTERS_SPKF_SIGMA_POINT_MATRIX_HPP

#include <Eigen/Eigen>

#include <vector>

#include <state_filtering/filters/deterministic/spkf/types.hpp>
#include <state_filtering/filters/deterministic/spkf/distribution_descriptor.hpp>

namespace distributions
{
    class SigmaPointMatrix:
            public Eigen::MatrixXd
    {
    public:
        typedef std::vector<double>::size_type size_type;
        typedef Eigen::MatrixXd MatrixType;

    public:
        /**
         * @brief SigmaPointMatrix Constructs a matrix instance
         *
         * @param numberOfPoints
         */
        explicit SigmaPointMatrix(size_type numberOfPoints = 0);

        /**
         * @brief ~SigmaPointMatrix Customizable destructor
         */
        virtual ~SigmaPointMatrix();

        /**
         * @brief meanWeight Mean weight of the sigma point at the specified index
         *
         * @param index      Column index number
         *
         * @return           Mean weight of the reuquested sigma point
         */
        virtual double meanWeight(int index) const;

        /**
         * @brief meanWeight Covariance weight of the sigma point at the specified index
         *
         * @param index      Column index number
         *
         * @return           Covariance weight of the reuquested sigma point
         */
        virtual double covWeight(int index) const;

        /**
         * @brief resize Resizes the matrix and its weight vectors
         *
         * @param dimension         Rows of the matrix
         * @param numberOfPoints    Columns of the matrix
         */
        virtual void resize(int numberOfPoints, int dimension = 1);

        /**
         * @brief point Sets the sigma point and its related weights
         *
         * @param [in] index        Sigma point index
         * @param [in] point        sigma point column vector
         * @param [in] meanWeight   mean related weight
         * @param [in] covWeight    covariance matrix related weight
         */
        virtual void point(int index,
                           const DynamicVector& point,
                           double meanWeight,
                           double covWeight);

        /**
         * @brief mean Computes the weighted mean of the sigma points
         *
         * @param [out] mean    Mean of the sigma points
         */
        virtual void mean(DynamicVector& _mean, int subDimension, int marginBegin) const;

        /**
         * @brief asWeightedZeroMean Returns the weighted zero mean matrix
         */
        virtual void weightedZeroMeanSigmaPointMatrix(SigmaPointMatrix::MatrixType& wzmMatrix,
                                                      int subDimension = 0,
                                                      int marginBegin = 0) const;

        /**
         * @brief asWeightedZeroMean Returns the weighted zero mean matrix
         */
        virtual void weightedZeroMeanSigmaPointMatrix(const DynamicVector& _mean,
                                                      SigmaPointMatrix::MatrixType& wzmMatrix,
                                                      int subDimension = 0,
                                                      int marginBegin = 0) const;

        /**
         * @brief mean Computes the weighted covariance matrix of the sigma points
         *
         * @param [out] covariance    Covariance of the sigma points
         */
        virtual void covariance(DynamicMatrix& cov,
                                int subDimension = 0,
                                int marginBegin = 0) const;

        /**
         * @brief mean Computes the weighted covariance matrix of the sigma points
         *
         * @param [out] covariance    Covariance of the sigma points
         */
        virtual void covariance(const DynamicVector& _mean,
                                DynamicMatrix& cov,
                                int subDimension = 0,
                                int marginBegin = 0) const;

        /**
         * @brief zeroMeanSigmaPointMatrix Computes the zero mean sigma point matrix
         */
        virtual void zeroMeanSigmaPointMatrix(const DynamicVector& _mean,
                                              SigmaPointMatrix::MatrixType& zmMatrix,
                                              int subDimension = 0,
                                              int marginBegin = 0) const;

        /**
         * @brief zeroMeanSigmaPointMatrix Computes the zero mean sigma point matrix
         */
        virtual void zeroMeanSigmaPointMatrix(SigmaPointMatrix::MatrixType& zmMatrix,
                                              int subDimension = 0,
                                              int marginBegin = 0) const;

        /**
         * @brief meanWeights Returns the mean weight vector
         *
         * @return mean weight vector
         */
        virtual const DynamicVector& meanWeights() const;

        /**
         * @brief covarianceWeights Returns the covariance weight vector
         *
         * @return covariance weight vetcor
         */
        virtual const DynamicVector& covarianceWeights() const;

    protected:
        /**
         * @brief meanWeights Mean weight vector
         */
        DynamicVector meanWeights_;

        /**
         * @brief covWeights Covariance weight vector
         */
        DynamicVector covWeights_;
    };
}

#endif
