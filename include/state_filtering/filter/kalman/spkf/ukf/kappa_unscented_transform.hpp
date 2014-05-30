
#ifndef FILTERS_SPKF_UKF_KAPPA_UNSCENTED_TRANFROM_HPP
#define FILTERS_SPKF_UKF_KAPPA_UNSCENTED_TRANFROM_HPP

#include <cmath>

#include <boost/shared_ptr.hpp>

#include <Eigen/Eigen>

#include "filters/spkf/types.hpp"
#include "filters/spkf/sigma_point_transform.hpp"

namespace filter
{
    /**
     * The UnscentedTransform sigma point sampler using 2n + 1 sigma points.
     */
    class KappaUnscentedTransform:
            public SigmaPointTransform
    {
    public:
        typedef boost::shared_ptr<KappaUnscentedTransform> Ptr;

    public:
        /**
         * @brief UnscentedTransform
         *
         * @param alpha     Sigma points scaling parameter alpha
         * @param beta      Sigma points scaling parameter beta (for the first point)
         * @param kappa     Sigma points scaling parameter kappa
         */
        KappaUnscentedTransform(double _kappa);

        /**
         * @brief ~UnscentedTransform Customizable destructor
         */
        virtual ~KappaUnscentedTransform();

        /**
         * @brief forward Samples sigma points from the given gaussian distribution
         *
         * @param [in]  distDesc        Input distribution descriptor
         * @param [out] sigmaPoints     Sampled sigma points
         *
         * Samples 2n+1 sigma points from the given gaussian distribution defined by the mean vector
         * and its covariance matrix. n is the dimension of the state
         */
        virtual void forward(const DistributionDescriptor &distDesc,
                             SigmaPointMatrix& sigmaPoints);

        /**
         * @brief Calculates the gaussian distribution out of the given points
         *
         * Uses the 2n+1 sigma points to determine the mean and the covariance of the gaussian
         *
         * @param [in]  sigmaPoints  Given sigma points
         * @param [out] distDesc     Reconstructed gaussian distribution description
         */
        virtual void backward(const SigmaPointMatrix& sigmaPoints,
                              DistributionDescriptor& distDesc,
                              int subDimension = 0,
                              int marginBegin = 0);

        /**
         * @brief numberOfSigmaPoints Determines the number of required sigma points for the UT.
         *
         * Returns the number of sigma points the unscented transform operates on, which is 2n + 1.
         * Here n is the dimension of the state vector.
         *
         * @param dimension State dimension
         *
         * @return Number of required sigma points
         */
        virtual unsigned int numberOfSigmaPoints(unsigned int dimension) const;

    public:
        double gammaFactor(unsigned int dimension);
        double kappa() const;
        void kappa(double _kappa);

    private:
        double kappa_;
    };
}

#endif
