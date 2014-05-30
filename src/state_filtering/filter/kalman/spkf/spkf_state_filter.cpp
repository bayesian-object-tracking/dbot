
#include <Eigen/Eigen>

#include "filters/spkf/spkf_state_filter.hpp"

using namespace filter;

SpkfStateFilter::SpkfStateFilter(SpkfInternals::Ptr _internals):
    internals_(_internals)
{
    augmentedStateDesc = internals()->createDistributionDescriptor();
    measurementDesc = internals()->createDistributionDescriptor();    
}

SpkfStateFilter::~SpkfStateFilter()
{
}

void SpkfStateFilter::predict(const DistributionDescriptor& stateDesc,
                              const ProcessModel::ControlVector& control,
                              double deltaTime,
                              DistributionDescriptor& predictedStateDesc)
{
    internals()->onBeginPredict(stateDesc, *augmentedStateDesc);

    internals()->process(*augmentedStateDesc,
                         control,
                         deltaTime,
                         predictedStateDesc);

    internals()->onFinalizePredict(stateDesc, predictedStateDesc);
}

void SpkfStateFilter::update(const MeasurementModel::MeasurementVector& measurement,
                             DistributionDescriptor& predictedState,
                             DistributionDescriptor& updatedState)
{
    internals()->predictMeasurement(predictedState.sigmaPoints(),
                                    measurementDesc->sigmaPoints());

    internals()->onBeginUpdate(predictedState, *measurementDesc);

    internals()->update(predictedState, *measurementDesc, measurement, updatedState);

    internals()->onFinalizeUpdate(*measurementDesc, updatedState);
}

MeasurementModel::Ptr SpkfStateFilter::measurementModel()
{
    return internals()->measurementModel();
}

ProcessModel::Ptr SpkfStateFilter::processModel()
{
    return internals()->processModel();
}

SigmaPointTransform::Ptr SpkfStateFilter::sigmaPointTransform()
{
    return internals()->sigmaPointTransform();
}

void SpkfStateFilter::measurementModel(const MeasurementModel::Ptr& measurementModel_)
{
    internals()->measurementModel(measurementModel_);
}

void SpkfStateFilter::processModel(const ProcessModel::Ptr& processModel_)
{
    internals()->processModel(processModel_);
}

void SpkfStateFilter::sigmaPointTransform(SigmaPointTransform::Ptr _sigmaPointTransform)
{
    internals()->sigmaPointTransform(_sigmaPointTransform);
}

void SpkfStateFilter::validationGate(ValidationGate::Ptr validationGate)
{
    internals()->validationGate(validationGate);
}

ValidationGate::Ptr SpkfStateFilter::validationGate()
{
    return internals()->validationGate();
}

SpkfInternals::Ptr SpkfStateFilter::internals()
{
    return internals_;
}

const DistributionDescriptor& SpkfStateFilter::augmentedStateDescriptor() const
{
    return *augmentedStateDesc;
}

const DistributionDescriptor& SpkfStateFilter::measurementDescriptor() const
{
    return *measurementDesc;
}
