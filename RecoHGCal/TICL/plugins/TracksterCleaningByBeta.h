#ifndef RecoHGCal_TICL_TracksterCleaningByBeta_H
#define RecoHGCal_TICL_TracksterCleaningByBeta_H

#include <string>
#include "RecoHGCal/TICL/interface/TracksterCleaningAlgoBase.h"

namespace ticl {

class TracksterCleaningByBeta final : public TracksterCleaningAlgoBase {
public:
  TracksterCleaningByBeta(const edm::ParameterSet& ps, edm::ConsumesCollector);

  void cleanTracksters(const Inputs& input,
                       std::vector<ticl::Trackster>& outTracksters,
                       std::vector<std::vector<unsigned int>>& outMap,
                       std::vector<std::vector<float>>& outWeights) const override;

  void initialize() override {}
  void setEvent(edm::Event&, edm::EventSetup const&) override {}

  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    TracksterCleaningAlgoBase::fillPSetDescription(desc);
    desc.add<double>("betaContamMin", 1.12); // 90% PU identification efficiency, 17% signal fake rate
    desc.add<double>("R0", 0.10);
    desc.add<bool>("useRawEnergy", true);
    desc.add<double>("epsE", 1e-6);
    desc.add<double>("epsDR", 1e-6);
    desc.add<std::string>("reweightMode", "drop");           // "drop" or "weight"
    desc.add<bool>("emitDroppedAsStandalone", false);
    desc.add<double>("zAbsCut", 25.0);  // [cm]
    desc.add<double>("tAbsCut", 0.15);  // [ns]
    desc.add<double>("sigmaZ", 12.5);
    desc.add<double>("sigmaT", 0.08);
    desc.add<double>("sigmaDR", 0.08);
    desc.add<double>("zPower", 1.5);
    desc.add<double>("tPower", 0.5);
    desc.add<double>("drPower", 0.5);
  }

private:
  // β = log( Ek * sum_ij ΔR_ij * Θ(ΔR_ij ≤ R0) )
  double betaContamMin_, R0_, epsE_, epsDR_;
  bool   useRawEnergy_, emitDroppedAsStandalone_;

  // pruning / weighting
  std::string mode_;
  double zAbsCut_, tAbsCut_;
  double sigmaZ_, sigmaT_, sigmaDR_;
  double zPower_, tPower_, drPower_;

  // helpers
  inline float linkEnergy_(const ticl::Trackster& trackster) const {
    return useRawEnergy_ ? trackster.raw_energy() : trackster.regressed_energy();
  }
  inline void setLinkRawEnergy_(ticl::Trackster& trackster, float e) const {
    trackster.setRawEnergy(e);
  }
};

}  // namespace ticl

#endif
