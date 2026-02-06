#!/usr/bin/env python3
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*smallest subnormal.*",
    category=UserWarning,
    module="numpy\\.core\\.getlimits"
)

import sys
import yaml
import awkward as ak
import numpy as np

from utils import *

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mode   = cfg.get("mode", "eff")
    outdir = cfg.get("outdir", "images")

    infile = cfg["input"]["file"]
    tree   = cfg["input"].get("tree", "Events")

    truth_cfg   = cfg.get("truth", {}) or {}
    eta_max     = float(truth_cfg.get("eta_max", 2.5))
    dr_resolved = float(truth_cfg.get("dr_resolved", 0.4))
    dr_merged   = float(truth_cfg.get("dr_merged", 0.8))
    dr_paired   = float(truth_cfg.get("dr_paired", 0.4))

    branches = [
        "GenPart_pt","GenPart_eta","GenPart_phi","GenPart_pdgId","GenPart_statusFlags","GenPart_genPartIdxMother",
        "Jet_pt","Jet_eta","Jet_phi", "nJet",
        "FatJet_pt","FatJet_eta","FatJet_phi",
        "nPAIReDJets","PAIReDJets_bb_score","PAIReDJets_ll_score","PAIReDJets_idx_jet1","PAIReDJets_idx_jet2", "PAIReDJets_n_bparton",
    ]

    events = load_arrays(infile, tree=tree, branches=branches)

    genH_pt, b_eta, b_phi, final_mask = has_gen_Hbb(events, eta_max=eta_max)
    events = {k: v[final_mask] for k, v in events.items()}
    events["genH_pt"] = genH_pt

    resolved_truth = match_resolved(events, b_eta, b_phi, dr=dr_resolved, jet_pt_min=0.)
    merged_truth   = match_merged(events, b_eta, b_phi, dr=dr_merged, jet_pt_min=0.)
    paired_truth   = match_paired(events, b_eta, b_phi, dr=dr_paired, jet_pt_min=0.)

    bb, ll = events["PAIReDJets_bb_score"], events["PAIReDJets_ll_score"]
    nPAIReDJets = events["nPAIReDJets"]
    n_bparton = events["PAIReDJets_n_bparton"]

    nJets = events["nJet"]
    n_b_event = n_bpartons_per_event(events)

    env = build_env(events)
    env.update({
        "resolved_truth": resolved_truth,
        "merged_truth": merged_truth,
        "paired_truth": paired_truth,
        "PAIReDJets_BBvsLL_score": probBBvsLL(bb, ll),
        "nPairedJets": nPAIReDJets,
        "n_bparton": n_bparton,
        "nJet": nJets,
        "n_bparton_event": n_b_event,
    })

    style = cfg.get("style", {}) or {}

    if mode in ("hist", "check_events"):
        if mode == "hist":
            matched = paired_truth_per_candidate(events, b_eta, b_phi, dr=dr_paired)
            env["paired_is_hbb"] = matched

        if mode == "check_events":
            matched = paired_truth_per_candidate(events, b_eta, b_phi, dr=dr_paired)

            scores = events["PAIReDJets_bb_score"]
            idx1   = events["PAIReDJets_idx_jet1"]
            idx2   = events["PAIReDJets_idx_jet2"]
            n_bparton = events["PAIReDJets_n_bparton"]

            jpt  = events["Jet_pt"]
            jeta = events["Jet_eta"]
            jphi = events["Jet_phi"]

            pt1, eta1, phi1 = jpt[idx1],  jeta[idx1], jphi[idx1]
            pt2, eta2, phi2 = jpt[idx2],  jeta[idx2], jphi[idx2]

            dR_cand  = delta_r(eta1, phi1, eta2, phi2)
            mjj_cand = get_mjj(pt1, pt2, eta1, eta2, phi1, phi2)

            has_true = ak.any(matched, axis=1)
            has_best = (ak.num(scores, axis=1) > 0)

            scores_true = ak.where(matched, scores, -np.inf)
            idx_true    = ak.argmax(scores_true, axis=1)
            score_true  = ak.max(scores_true, axis=1)

            idx_best    = ak.argmax(scores, axis=1)
            score_best  = ak.max(scores, axis=1)

            mjj_true = ak.firsts(mjj_cand[ak.local_index(mjj_cand) == idx_true[:, None]])
            dR_true  = ak.firsts(dR_cand[ak.local_index(dR_cand) == idx_true[:, None]])
            nb_true = ak.firsts(n_bparton[ak.local_index(n_bparton) == idx_true[:, None]])

            mjj_best = ak.firsts(mjj_cand[ak.local_index(mjj_cand) == idx_best[:, None]])
            dR_best  = ak.firsts(dR_cand[ak.local_index(dR_cand) == idx_best[:, None]])
            nb_best = ak.firsts(n_bparton[ak.local_index(n_bparton) == idx_best[:, None]])

            mjj_true = ak.mask(mjj_true, has_true)
            dR_true  = ak.mask(dR_true,  has_true)
            nb_true  = ak.mask(nb_true,  has_true)

            mjj_best = ak.mask(mjj_best, has_best)
            dR_best  = ak.mask(dR_best,  has_best)
            nb_best  = ak.mask(nb_best,  has_best)

            best_is_true = ak.firsts(matched[ak.local_index(matched) == idx_best[:, None]])
            best_is_true = ak.mask(best_is_true, has_best)

            cand_idx = ak.local_index(scores)  # shape: (nevents, ncands)

            best_jet1_pt  = ak.firsts(pt1[cand_idx == idx_best[:, None]])
            best_jet1_eta = ak.firsts(eta1[cand_idx == idx_best[:, None]])
            best_jet1_phi = ak.firsts(phi1[cand_idx == idx_best[:, None]])

            best_jet2_pt  = ak.firsts(pt2[cand_idx == idx_best[:, None]])
            best_jet2_eta = ak.firsts(eta2[cand_idx == idx_best[:, None]])
            best_jet2_phi = ak.firsts(phi2[cand_idx == idx_best[:, None]])

            # mask out events with no candidates
            best_jet1_pt  = ak.mask(best_jet1_pt,  has_best)
            best_jet2_pt  = ak.mask(best_jet2_pt,  has_best)
            best_jet1_eta = ak.mask(best_jet1_eta, has_best)
            best_jet2_eta = ak.mask(best_jet2_eta, has_best)
            best_jet1_phi = ak.mask(best_jet1_phi, has_best)
            best_jet2_phi = ak.mask(best_jet2_phi, has_best)

            # Higgs truth-matched PAIReD jet does NOT have the highest Prob(bb) score in event
            flag_event = has_true & has_best & (~ak.fill_none(best_is_true, False)) & (score_best > score_true)

            env.update({
                "has_true": has_true,
                "has_best": has_best,
                "flag_event": flag_event,
                "mjj_true": mjj_true,
                "dR_true": dR_true,
                "mjj_best": mjj_best,
                "dR_best": dR_best,
                "score_true": score_true,
                "score_best": score_best,
                "best_is_true": best_is_true,
                "nb_best": nb_best,
                "nb_true": nb_true,
                "best_jet1_pt": best_jet1_pt,
                "best_jet2_pt": best_jet2_pt,
                "best_jet1_eta": best_jet1_eta,
                "best_jet2_eta": best_jet2_eta,
                "best_jet1_phi": best_jet1_phi,
                "best_jet2_phi": best_jet2_phi,
                "nJets": nJets,
            })

        hcfg = cfg["hist"]
        if isinstance(hcfg, list):
            for hc in hcfg:
                out = plot_hist(hc, env, outdir=outdir, style=style)
                print(f"Wrote {out}")
        else:
            out = plot_hist(hcfg, env, outdir=outdir, style=style)
            print(f"Wrote {out}")
        return

    base_mask = ak.ones_like(events["genH_pt"], dtype=bool)
    for expr in cfg.get("selections", {}).get("base_event", []):
        base_mask = base_mask & eval_expr(expr, env)

    bkg_vals = events["genH_pt"][base_mask]
    b = cfg["binning"]
    if "edges" in b:
        edges = np.asarray(b["edges"], dtype=float)
    else:
        xmin  = float(b["xmin"])
        xmax  = float(b["xmax"])
        nbins = int(b["nbins"])
        if b.get("log", False):
            if xmin <= 0 or xmax <= 0:
                raise ValueError(f"log binning requested but xmin/xmax not positive: xmin={xmin}, xmax={xmax}")
            edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
        else:
            edges = np.linspace(xmin, xmax, nbins + 1)

    den = binned_counts(events["genH_pt"], edges, mask=base_mask)
    min_events = int(cfg.get("plot", {}).get("min_events", 10))
    valid = den >= min_events

    # choose x-range to last bin with # > 10
    idx_valid = np.where(valid)[0]
    if idx_valid.size > 0:
        xmax_plot = edges[idx_valid.max() + 1]
    else:
        idx_nonzero = np.where(den > 0)[0]
        xmax_plot = edges[idx_nonzero.max() + 1] if idx_nonzero.size > 0 else edges[-1]


    series = []
    for name, cat in cfg["categories"].items():
        m = base_mask
        for expr in cat.get("mask", []):
            m = m & eval_expr(expr, env)
        num = binned_counts(events["genH_pt"], edges, mask=m)
        eff, err = binom_eff(num, den)
        eff[~valid] = np.nan
        err[~valid] = np.nan
        series.append({"label": cat.get("label", name), "eff": eff, "err": err, "style": cat.get("style", {})})

    p = cfg["plot"]
    out = plot_efficiency(
        edges=edges,
        series=series,
        xlabel=p.get("xlabel", "pT(H) [GeV]"),
        ylabel=p.get("ylabel", "Reconstruction efficiency"),
        ylim=tuple(p.get("ylim", [0.0, 1.05])),
        out=p.get("out", "eff.png"),
        outdir=outdir,
        logx=bool(p.get("logx", False)),
        normalize=bool(p.get("normalize", True)),
        style=p,
        text=p.get("text", []),
        bkg_vals=bkg_vals,
        bkg_style=p.get("background", {}),
    )

    plots2d = cfg.get("plots2d", []) or []
    for p2 in plots2d:
        kind = p2.get("kind", "hist2d")
        if kind == "hist2d":
            out = plot2d_hist(p2, env, outdir=outdir, style=style)
        elif kind == "profile":
            out = plot2d_profile(p2, env, outdir=outdir, style=style)
        else:
            raise ValueError(f"Unknown 2D plot kind: {kind}")
    print(f"Wrote {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: plot.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
