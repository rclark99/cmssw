#include <vector>
#include <cmath>
#include <algorithm>

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
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoHGCal/TICL/plugins/TracksterCleaningByBeta.h"
#include "RecoHGCal/TICL/interface/TracksterCleaningAlgoBase.h"

namespace {
  constexpr double c_cm_per_ns     = 29.9792458;
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
      weightMode_(conf.getParameter<bool>("weightMode")),
      zAbsCut_(conf.getParameter<double>("zAbsCut")),
      tAbsCut_(conf.getParameter<double>("tAbsCut")),
      sigmaZ_(conf.getParameter<double>("sigmaZ")),
      sigmaT_(conf.getParameter<double>("sigmaT")),
      sigmaDR_(conf.getParameter<double>("sigmaDR")),
      zPower_(conf.getParameter<double>("zPower")),
      tPower_(conf.getParameter<double>("tPower")),
      drPower_(conf.getParameter<double>("drPower")),
      wmin_(conf.getParameter<double>("wmin"))
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

    const auto& bcL = link.barycenter();
    const double LL = std::sqrt(bcL.x()*bcL.x() + bcL.y()*bcL.y() + bcL.z()*bcL.z());
    const double tLcorr = link.time() - LL * inv_c_cm_per_ns;
    const double zL  = bcL.z();
    const double etaL = bcL.eta();
    const double phiL = bcL.phi();

    // beta = log( E_link * sumΔR )
    const double Ek = std::max(epsE_, linkEnergy_(link));
    double sumDR = 0.0;
    const size_t nm = members.size();
    if (nm >= 2) {
      for (size_t a = 0; a + 1 < nm; ++a) {
        const auto& ta = in.clue3d[members[a]];
        const double etaA = ta.barycenter().eta();
        const double phiA = ta.barycenter().phi();
        for (size_t b = a + 1; b < nm; ++b) {
          const auto& tb = in.clue3d[members[b]];
          const double dR = reco::deltaR(etaA, phiA, tb.barycenter().eta(), tb.barycenter().phi());
          if (dR <= R0_) sumDR += dR;
        }
      }
    }
    const double beta = std::log(Ek * std::max(epsDR_, sumDR));

    if (beta < betaContamMin_) {
      std::vector<unsigned int> kept = members;
      kept.shrink_to_fit();
      std::vector<float> keptW; keptW.shrink_to_fit();
      ticl::Trackster cleaned = link;

      outTracksters.emplace_back(std::move(cleaned));
      outMap.emplace_back(std::move(kept));
      outWeights.emplace_back(std::move(keptW));
      continue;
    }

    ticl::Trackster cleaned = link;
    std::vector<unsigned int> kept, dropped;
    std::vector<float> keptW, droppedW;
    kept.reserve(members.size());
    keptW.reserve(members.size());

    for (const auto idx : members) {
      const auto& t  = in.clue3d[idx];
      const auto& bc = t.barycenter();

      const double dz = bc.z() - zL;

      const double Lm = std::sqrt(bc.x()*bc.x() + bc.y()*bc.y() + bc.z()*bc.z());
      const double tmCorr = t.time() - Lm * inv_c_cm_per_ns;
      const double dt = tmCorr - tLcorr;

      const double dR = reco::deltaR(etaL, phiL, bc.eta(), bc.phi());

      const bool passZ = std::abs(dz) <= zAbsCut_;
      const bool passT = (tAbsCut_ <= 0.0) ? true : (std::abs(dt) <= tAbsCut_);

      const double sZ = std::max(1e-12, sigmaZ_);
      const double sT = std::max(1e-12, sigmaT_);
      const double sR = std::max(1e-12, sigmaDR_);

      double wz = std::exp(-0.5 * (dz*dz)/(sZ*sZ));
      double wt = std::exp(-0.5 * (dt*dt)/(sT*sT));
      double wr = std::exp(-0.5 * (dR*dR)/(sR*sR));

      if (!passZ) wz = 0.0;
      if (!passT) wt = 0.0;

      const double w = std::pow(wz, zPower_) * std::pow(wt, tPower_) * std::pow(wr, drPower_);

      if (w >= wmin_) {
        kept.push_back(idx);
        if (weightMode_) keptW.push_back(static_cast<float>(w));
      } else {
        dropped.push_back(idx);
        if (weightMode_) droppedW.push_back(0.0f);
      }
    }

    // compute energy of cleaned link
    double eNew = 0.0;
    if (!weightMode_) {
      for (auto idx : kept) eNew += in.clue3d[idx].raw_energy();      
    } else {
      for (size_t i = 0; i < kept.size(); ++i)
        eNew += keptW[i] * in.clue3d[kept[i]].raw_energy();
    }
    setLinkRawEnergy_(cleaned, eNew);

    if (!weightMode_) keptW.clear();
    kept.shrink_to_fit();
    keptW.shrink_to_fit();

    outTracksters.emplace_back(std::move(cleaned));
    outMap.emplace_back(std::move(kept));
    outWeights.emplace_back(std::move(keptW));

    if (emitDroppedAsStandalone_ && !dropped.empty()) {
      ticl::Trackster droppedLink = link;

      double eDrop = 0.0;
      if (!weightMode_) {
        for (auto idx : dropped) eDrop += in.clue3d[idx].raw_energy();
      } else {
        for (size_t i = 0; i < dropped.size(); ++i)
          eDrop += droppedW[i] * in.clue3d[dropped[i]].raw_energy();
      }
      setLinkRawEnergy_(droppedLink, eDrop);

      if (!weightMode_) droppedW.clear();
      dropped.shrink_to_fit();
      droppedW.shrink_to_fit();

      outTracksters.emplace_back(std::move(droppedLink));
      outMap.emplace_back(std::move(dropped));
      outWeights.emplace_back(std::move(droppedW));
    }
  }
}
