import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.tracksterCleaningProducer_cfi import tracksterCleaningProducer as _tracksterCleaningProducer

hltTiclTracksterCleaning = _tracksterCleaningProducer.clone(
    linkedTracksters      = cms.InputTag("hltTiclTracksterLinks"),
    clue3DTracksters      = cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
    clue3DInLinkedIndices = cms.InputTag("hltTiclTracksterLinks","linkedTracksterIdToInputTracksterId"),

    labelLinkedOut        = cms.string(""),
    labelMapOut           = cms.string("linkedTracksterIdToInputTracksterId"),
    cleaner = cms.PSet(
        type = cms.string("Beta"),
        algo_verbosity = cms.int32(0),
        betaContamMin  = cms.double(1.12),
        R0             = cms.double(0.1),
        useRawEnergy   = cms.bool(True),
        epsE           = cms.double(1e-6),
        epsDR          = cms.double(1e-6),
        weightMode     = cms.bool(True),
        emitDroppedAsStandalone = cms.bool(False),
        zAbsCut = cms.double(25.0),
        tAbsCut = cms.double(0.15),
        sigmaZ  = cms.double(12.5),
        sigmaT  = cms.double(0.08),
        sigmaDR = cms.double(0.08),
        zPower  = cms.double(1.5),
        tPower  = cms.double(0.5),
        drPower = cms.double(0.5),
        wmin    = cms.double(1e-3),
    ),
)
