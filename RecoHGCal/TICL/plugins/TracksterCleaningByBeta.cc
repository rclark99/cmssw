// user include files
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "RecoHGCal/TICL/plugins/TracksterCleaningByBeta.h"
#include "RecoHGCal/TICL/interface/TracksterCleaningAlgoBase.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"

namespace {
    constexpr double c_cm_per_ns = 29.9792458;
    constexpr double inv_c_cm_per_ns = 1.0 / c_cm_per_ns;
}

using namespace ticl;

TracksterCleaningByBeta::TracksterCleaningByBeta(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : TracksterCleaningAlgoBase(conf, iC),
      betaContamMin_(conf.getParameter<double>("betaContamMin")),
      R0_(conf.getParameter<double>("R0")),
      epsE_(conf.getParameter<double>("epsE")),
      epsDR_(conf.getParameter<double>("epsDR")),
      useRawEnergy_(conf.getParameter<bool>("useRawEnergy")),
      emitDroppedAsStandalone_(conf.getParameter<bool>("emitDroppedAsStandalone")),
      mode_(conf.getParameter<std::string>("reweightMode")),
      zAbsCut_(conf.getParameter<double>("zAbsCut")),
      tAbsCut_(conf.getParameter<double>("tAbsCut")),
      sigmaZ_(conf.getParameter<double>("sigmaZ")),
      sigmaT_(conf.getParameter<double>("sigmaT")),
      sigmaDR_(conf.getParameter<double>("sigmaDR")),
      zPower_(conf.getParameter<double>("zPower")),
      tPower_(conf.getParameter<double>("tPower")),
      drPower_(conf.getParameter<double>("drPower")) 
      {}

void TracksterCleaningByBeta::cleanTracksters(const Inputs& in,
                                              std::vector<ticl::Trackster>& outTracksters,
                                              std::vector<std::vector<unsigned int>>& outMap,
                                              std::vector<std::vector<float>>& outWeights) const {
  const size_t nL = in.linked.size();
  outTracksters.clear(); outMap.clear(); outWeights.clear();
  outTracksters.reserve(nL);
  outMap.reserve(nL * (emitDroppedAsStandalone_ ? 2 : 1));
  outWeights.reserve(outMap.capacity());

  for (size_t L = 0; L < nL; ++L) {
    const auto& link    = in.linked[L];
    const auto& members = in.map[L];
    if (members.size() < 2) {  // nothing to clean
      outTracksters.push_back(link);
      outMap.push_back(members);
      outWeights.emplace_back();
      continue;
    }
    const auto& bcL = link.barycenter();
    const double LL = std::sqrt(double(bcL.x())*bcL.x() +
                                double(bcL.y())*bcL.y() +
                                double(bcL.z())*bcL.z());
    const double tLcorr = double(link.time()) - LL * inv_c_cm_per_ns;
    const double zL = bcL.z();
    const float etaL = bcL.eta();
    const float phiL = bcL.phi();

    // β = log( E_link * Σ ΔR(≤R0) )
    const double Ek = std::max(epsE_, static_cast<double>(linkEnergy_(link)));
    double sumDR = 0.0;
    const size_t nm = members.size();
    if (nm >= 2) {
        for (size_t a = 0; a < nm; ++a) {
            const auto& ta = in.clue3d[members[a]];
            const float etaA = ta.barycenter().eta();
            const float phiA = ta.barycenter().phi();

            for (size_t b = a + 1; b < nm; ++b) {             // avoid double counting & self-pairs
                const auto& tb = in.clue3d[members[b]];
                const double dR = reco::deltaR(etaA, phiA, tb.barycenter().eta(), tb.barycenter().phi());
                if (dR <= R0_) sumDR += dR;
            }
        }
    }
    const double beta = std::log(Ek * std::max(epsDR_, sumDR));

    ticl::Trackster cleaned = link;
    std::vector<unsigned int> kept, dropped;
    std::vector<float> keptW, droppedW;
    kept.reserve(members.size());
    keptW.reserve(members.size());

    if (beta < betaContamMin_) {
      kept = members;
      outTracksters.push_back(std::move(cleaned));
      outMap.push_back(std::move(kept));
      outWeights.emplace_back();
      continue;
    }

    for (auto idx : members) {
      const auto& t = in.clue3d[idx];
      const auto& bcCLUE = t.barycenter();
      const double dz = bcCLUE.z() - zL;
      const double LCLUE = std::sqrt(double(bcCLUE.x())*bcCLUE.x() +
                                     double(bcCLUE.y())*bcCLUE.y() +
                                     double(bcCLUE.z())*bcCLUE.z());
      const double tCLUEcorr = double(t.time()) - LCLUE * inv_c_cm_per_ns;
      const double dt = tCLUEcorr - tLcorr;
      const double dR = reco::deltaR(etaL, phiL, t.barycenter().eta(), t.barycenter().phi());

      const bool passZ = std::abs(dz) <= zAbsCut_;
      const bool passT = (tAbsCut_ <= 0.) ? true : (std::abs(dt) <= tAbsCut_);

      if (mode_ == "drop") {
        (passZ && passT) ? kept.push_back(idx) : dropped.push_back(idx);
      } else {
        const double wz = std::exp(-0.5 * (dz*dz)/(sigmaZ_*sigmaZ_));
        const double wt = std::exp(-0.5 * (dt*dt)/(sigmaT_*sigmaT_));
        const double wr = std::exp(-0.5 * (dR*dR)/(sigmaDR_*sigmaDR_));
        const float  w  = std::pow(wz, zPower_) * std::pow(wt, tPower_) * std::pow(wr, drPower_);
        if (w > 1e-3f) { kept.push_back(idx); keptW.push_back(w); }
        else           { dropped.push_back(idx); droppedW.push_back(0.f); }
      }
    }

    // recompute energy from kept CLUE3D tracksters
    float eNew = 0.f;
    if (mode_ == "drop") {
      for (auto idx : kept) eNew += in.clue3d[idx].raw_energy();
    } else {
      for (size_t i = 0; i < kept.size(); ++i)
        eNew += keptW[i] * in.clue3d[kept[i]].raw_energy();
    }
    setLinkRawEnergy_(cleaned, eNew);

    outTracksters.push_back(std::move(cleaned));
    outMap.push_back(std::move(kept));
    outWeights.push_back(std::move(keptW));

    // optionally emit dropped trackster as its own link
    if (emitDroppedAsStandalone_ && !dropped.empty()) {
      ticl::Trackster droppedLink = link;
      float eDrop = 0.f;
      if (mode_ == "drop") {
        for (auto idx : dropped) eDrop += in.clue3d[idx].raw_energy();
      } else {
        for (size_t i = 0; i < dropped.size(); ++i)
          eDrop += droppedW[i] * in.clue3d[dropped[i]].raw_energy();
      }
      setLinkRawEnergy_(droppedLink, eDrop);

      outTracksters.push_back(std::move(droppedLink));
      outMap.push_back(std::move(dropped));
      outWeights.push_back(std::move(droppedW));
    }
  }
}
