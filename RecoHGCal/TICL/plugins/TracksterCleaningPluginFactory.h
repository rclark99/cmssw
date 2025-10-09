#ifndef RecoHGCal_TICL_TracksterCleaningPluginFactory_h
#define RecoHGCal_TICL_TracksterCleaningPluginFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/interface/TracksterCleaningAlgoBase.h"

typedef edmplugin::PluginFactory<ticl::TracksterCleaningAlgoBase*(const edm::ParameterSet&, edm::ConsumesCollector)>
    TracksterCleaningPluginFactory;

#endif