// Author: Riley Clark - riley.coltrane.clark@cern.ch
// Date: 10/2025

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "RecoHGCal/TICL/interface/TracksterCleaningAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TracksterCleaningPluginFactory.h"
#include "RecoHGCal/TICL/plugins/TracksterCleaningByBeta.h"

namespace ticl {

class TracksterCleaningProducer : public edm::stream::EDProducer<> {
public:
  explicit TracksterCleaningProducer(const edm::ParameterSet& ps);
  ~TracksterCleaningProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  // inputs
  edm::EDGetTokenT<std::vector<Trackster>> link_token_;
  edm::EDGetTokenT<std::vector<Trackster>> tracksters_token_;
  edm::EDGetTokenT<std::vector<std::vector<unsigned int>>> map_token_;

  // algo
  std::unique_ptr<ticl::TracksterCleaningAlgoBase> cleaningAlgo_;
  int algoVerbosity_{0};

  std::string labelLinkedOut_;
  std::string labelMapOut_;
  std::string labelWeightsOut_;
};

TracksterCleaningProducer::TracksterCleaningProducer(const edm::ParameterSet& ps) {
  // Input tags
  const auto linkedTag = ps.getParameter<edm::InputTag>("linkedTracksters");
  const auto clue3dTag = ps.getParameter<edm::InputTag>("clue3DTracksters");
  const auto mapTag    = ps.getParameter<edm::InputTag>("clue3DInLinkedIndices");

  link_token_ = consumes<std::vector<Trackster>>(linkedTag);
  tracksters_token_ = consumes<std::vector<Trackster>>(clue3dTag);
  map_token_    = consumes<std::vector<std::vector<unsigned int>>>(mapTag);

  algoVerbosity_ = ps.getParameter<int>("algo_verbosity");

  // Outputs (instance names configurable)
  labelLinkedOut_  = ps.getParameter<std::string>("labelLinkedOut");   // e.g. "cleaned"
  labelMapOut_     = ps.getParameter<std::string>("labelMapOut");      // e.g. "cleanedToCLUE3D"
  labelWeightsOut_ = ps.getParameter<std::string>("labelWeightsOut");  // e.g. "weights"

  // Cleaning plugin
  const auto& cleanerPSet = ps.getParameter<edm::ParameterSet>("cleaner");
  const auto pluginName   = cleanerPSet.getParameter<std::string>("type");
  cleaningAlgo_ = TracksterCleaningPluginFactory::get()->create(pluginName, cleanerPSet, consumesCollector());

  // products
  produces<std::vector<Trackster>>(labelLinkedOut_);
  produces<std::vector<std::vector<unsigned int>>>(labelMapOut_);
  produces<std::vector<std::vector<float>>>(labelWeightsOut_);
}

void TracksterCleaningProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  auto const& linked = ev.get(link_token_);
  auto const& clue3d = ev.get(tracksters_token_);
  auto const& mapIn  = ev.get(map_token_);

  auto outLinked  = std::make_unique<std::vector<Trackster>>();
  auto outMap     = std::make_unique<std::vector<std::vector<unsigned int>>>();
  auto outWeights = std::make_unique<std::vector<std::vector<float>>>();

  ticl::TracksterCleaningAlgoBase::Inputs in(ev, es, linked, clue3d, mapIn);
  cleaningAlgo_->cleanTracksters(in, *outLinked, *outMap, *outWeights);

  ev.put(std::move(outLinked),  labelLinkedOut_);
  ev.put(std::move(outMap),     labelMapOut_);
  ev.put(std::move(outWeights), labelWeightsOut_);
}

void TracksterCleaningProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Inputs: default to outputs of TracksterLinksProducer + CLUE3D collection
  desc.add<edm::InputTag>("linkedTracksters",        edm::InputTag("tracksterLinksProducer")); // tracksters after linking
  desc.add<edm::InputTag>("clue3DTracksters",        edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.add<edm::InputTag>("clue3DInLinkedIndices",
                          edm::InputTag("tracksterLinksProducer", "linkedTracksterIdToInputTracksterId"));

  desc.add<int>("algo_verbosity", 0);

  edm::ParameterSetDescription cleanerDesc;
  cleanerDesc.addNode(edm::PluginDescription<TracksterCleaningPluginFactory>("type", "Beta", true));
  desc.add<edm::ParameterSetDescription>("cleaner", cleanerDesc);

  desc.add<std::string>("labelLinkedOut",  "cleaned");
  desc.add<std::string>("labelMapOut",     "cleanedToCLUE3D");
  desc.add<std::string>("labelWeightsOut", "weights");

  descriptions.add("ticlTracksterCleaningProducer", desc);
}

}  // namespace ticl

DEFINE_FWK_MODULE(ticl::TracksterCleaningProducer);
