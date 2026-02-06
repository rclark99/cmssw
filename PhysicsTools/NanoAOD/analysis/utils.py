import os
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep

def load_arrays(filename, tree="Events", branches=None):
    with uproot.open(filename) as f:
        t = f[tree]
        if branches is None:
            branches = t.keys()
        return t.arrays(branches, how=dict)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _to_numpy_flat(x):
    x = ak.flatten(x, axis=None)
    x = ak.to_numpy(ak.fill_none(x, np.nan))
    return x[np.isfinite(x)]

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2*np.pi) - np.pi

def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt(delta_r2(eta1, phi1, eta2, phi2))

def get_mjj(pt1, pt2, eta1, eta2, phi1, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    m2 = 2.0 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi))
    return np.sqrt(np.maximum(m2, 0.0))

def probBBvsLL(probBB, probLL):
    return probBB/(probBB + probLL)

def n_bpartons_per_event(events):
    pid = events["GenPart_pdgId"]
    flags  = events["GenPart_statusFlags"]

    is_lastCopy_b = (abs(pid) == 5) & ((flags & (1 << 13)) != 0)
    
    return ak.sum(is_lastCopy_b, axis=1)

def has_gen_Hbb(events, eta_max=2.5):
    pid    = events["GenPart_pdgId"]
    flags  = events["GenPart_statusFlags"]
    mother = events["GenPart_genPartIdxMother"]
    pt     = events["GenPart_pt"]
    eta    = events["GenPart_eta"]
    phi    = events["GenPart_phi"]

    is_h = (pid == 25) & ((flags & (1 << 13)) != 0)
    h_idx = ak.firsts(ak.local_index(pid)[is_h])
    genH_pt_all = ak.firsts(pt[is_h])

    has_h = ~ak.is_none(h_idx) & ~ak.is_none(genH_pt_all)

    is_b = (abs(pid) == 5) & (mother == h_idx[:, None])
    b_eta_all = eta[is_b]
    b_phi_all = phi[is_b]

    has_2b = ak.num(b_eta_all) >= 2
    b_eta_2 = b_eta_all[has_2b][:, :2]
    b_phi_2 = b_phi_all[has_2b][:, :2]

    acceptance = ak.all(abs(b_eta_2) < eta_max, axis=1)

    final_mask = ak.zeros_like(has_h, dtype=bool)
    final_mask = ak.where(has_h, has_2b & acceptance, False)

    genH_pt = genH_pt_all[final_mask]
    b_eta = b_eta_2[acceptance]
    b_phi = b_phi_2[acceptance]

    return genH_pt, b_eta, b_phi, final_mask

# def calc_n_bPartons(events):



def match_resolved(events, b_eta, b_phi, dr=0.4, jet_pt_min=30.0, jet_eta_max=2.5):
    jet_pt  = events["Jet_pt"]
    jet_eta = events["Jet_eta"]
    jet_phi = events["Jet_phi"]

    good = (jet_pt > jet_pt_min) & (abs(jet_eta) < jet_eta_max)

    dr2 = delta_r2(b_eta[:, :, None], b_phi[:, :, None], jet_eta[:, None, :], jet_phi[:, None, :])
    dr2 = ak.where(good[:, None, :], dr2, 1e9)

    jmin = ak.argmin(dr2, axis=2)
    dmin = ak.min(dr2, axis=2)

    pass_dr = dmin < (dr*dr)
    distinct = jmin[:, 0] != jmin[:, 1]
    return ak.all(pass_dr, axis=1) & distinct

def match_merged(events, b_eta, b_phi, dr=0.8, jet_pt_min=170.0, jet_eta_max=2.5):
    jet_pt  = events["FatJet_pt"]
    jet_eta = events["FatJet_eta"]
    jet_phi = events["FatJet_phi"]

    good = (jet_pt > jet_pt_min) & (abs(jet_eta) < jet_eta_max)

    dr2 = delta_r2(b_eta[:, :, None], b_phi[:, :, None], jet_eta[:, None, :], jet_phi[:, None, :])
    both_in = (dr2 < (dr*dr)) & good[:, None, :]
    per_fat = ak.all(both_in, axis=1)
    return ak.any(per_fat, axis=1)

def paired_truth_per_candidate(events, b_eta, b_phi, dr=0.4, jet_pt_min=30.0, jet_eta_max=2.5):
    idx1 = events["PAIReDJets_idx_jet1"]
    idx2 = events["PAIReDJets_idx_jet2"]

    Jet_pt  = events["Jet_pt"]
    Jet_eta = events["Jet_eta"]
    Jet_phi = events["Jet_phi"]

    j1_pt  = Jet_pt[idx1]
    j1_eta = Jet_eta[idx1]
    j1_phi = Jet_phi[idx1]

    j2_pt  = Jet_pt[idx2]
    j2_eta = Jet_eta[idx2]
    j2_phi = Jet_phi[idx2]

    good1 = (j1_pt > jet_pt_min) & (abs(j1_eta) < jet_eta_max)
    good2 = (j2_pt > jet_pt_min) & (abs(j2_eta) < jet_eta_max)

    dr2_b1_j1 = delta_r2(b_eta[:, 0], b_phi[:, 0], j1_eta, j1_phi)
    dr2_b2_j2 = delta_r2(b_eta[:, 1], b_phi[:, 1], j2_eta, j2_phi)
    dr2_b1_j2 = delta_r2(b_eta[:, 0], b_phi[:, 0], j2_eta, j2_phi)
    dr2_b2_j1 = delta_r2(b_eta[:, 1], b_phi[:, 1], j1_eta, j1_phi)

    m1 = good1 & good2 & (dr2_b1_j1 < dr*dr) & (dr2_b2_j2 < dr*dr)
    m2 = good1 & good2 & (dr2_b1_j2 < dr*dr) & (dr2_b2_j1 < dr*dr)
    return (m1 | m2)

def match_paired(events, b_eta, b_phi, dr=0.4, jet_pt_min=30.0, jet_eta_max=2.5):
    return ak.any(paired_truth_per_candidate(events, b_eta, b_phi, dr, jet_pt_min, jet_eta_max), axis=1)

def count(x):
    return ak.sum(x, axis=1)

def vmax(x):
    return ak.max(x, axis=1, initial=-np.inf)

def binned_counts(x, edges, mask=None):
    if mask is not None:
        x = x[mask]
    x = ak.flatten(x, axis=None)
    x = ak.to_numpy(ak.fill_none(x, np.nan))
    x = x[np.isfinite(x)]
    return np.histogram(x, bins=np.asarray(edges, dtype=float))[0]

def binom_eff(num, den):
    den = np.asarray(den)
    num = np.asarray(num)
    eff = np.full_like(den, np.nan, dtype=float)
    err = np.full_like(den, np.nan, dtype=float)
    m = den > 0
    eff[m] = num[m] / den[m]
    err[m] = np.sqrt(eff[m] * (1.0 - eff[m]) / den[m])
    return eff, err

def build_env(events):
    env = {k: v for k, v in events.items()}
    env.update({"abs": np.abs, "count": count, "max": vmax, "np": np, "ak": ak})
    return env

def eval_expr(expr, env):
    return eval(expr, {"__builtins__": {}}, env)

def plot_efficiency(
    edges, series, xlabel, ylabel, ylim, out, outdir="images",
    logx=False, style=None, text=None,
    bkg_vals=None, bkg_style=None, normalize=False,
    xlim=None,):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, out)

    style = style or {}
    bkg_style = bkg_style or {}

    plt.style.use(hep.style.CMS)
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        raise ValueError("Need at least two bin edges.")
    if logx and np.any(edges <= 0):
        raise ValueError("log-x plotting requested but some bin edges are <= 0.")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # ax  = LEFT  axis  -> efficiency
    # ax2 = RIGHT axis  -> background events
    ax2 = ax.twinx()

    nbins = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])

    # --- LEFT axis: efficiency ---
    ax.set_ylabel(ylabel, fontsize=style.get("labelsize", 18))
    for s in series:
        st = s.get("style", {})
        color = st.get("color", None) or ax._get_lines.get_next_color()

        eff = np.asarray(s["eff"], dtype=float)[:nbins]
        err = np.asarray(s["err"], dtype=float)[:nbins]
        good = np.isfinite(eff) & np.isfinite(err)

        y = np.full(nbins + 1, np.nan, dtype=float)
        y[:-1] = eff
        y[-1]  = eff[-1]

        ax.step(
            edges, y,
            where="post",
            linewidth=st.get("linewidth", 4.0),
            linestyle=st.get("linestyle", "-"),
            color=color,
            label=s.get("label", ""),
        )
        ax.errorbar(
            centers[good], eff[good], yerr=err[good],
            fmt=st.get("marker", "o"),
            linestyle="None",
            color=color,
            markersize=st.get("markersize", 7),
            capsize=2.5,
        )

    ax.set_ylim(*ylim)

    # --- RIGHT axis: Higgs pT shape (events) ---
    show_bkg = bool(bkg_style.get("show", False))
    if show_bkg and (bkg_vals is not None):
        color = bkg_style.get("axis_color", "tab:blue")

        # style RIGHT axis (ticks/label) for events
        ax2.spines['right'].set_color(color)
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)

        x = ak.to_numpy(ak.fill_none(bkg_vals, np.nan))
        x = x[np.isfinite(x)]
        ax2.hist(
            x,
            bins=edges,
            density=bool(bkg_style.get("normalize", False)),
            histtype=bkg_style.get("histtype", "stepfilled"),
            alpha=bkg_style.get("alpha", 0.25),
            linewidth=bkg_style.get("linewidth", 0.0),
            color=color,
            label=bkg_style.get("label", None),  # optional
        )

        if normalize:
            ax2.set_ylabel(bkg_style.get("ylabel", "a.u."), fontsize=style.get("labelsize", 18))
        else:
            ax2.set_ylabel(bkg_style.get("ylabel", "Events"), fontsize=style.get("labelsize", 18))
    else:
        # keep right axis quiet if no bkg
        ax2.set_ylabel(bkg_style.get("ylabel", ""), fontsize=style.get("labelsize", 18))

    # text (use LEFT axis coords)
    for t in (text or []):
        ax.text(
            float(t.get("x", 0.05)),
            float(t.get("y", 0.85)),
            t.get("s", ""),
            transform=ax.transAxes,
            fontsize=int(t.get("fontsize", 18)),
            ha=t.get("ha", "left"),
            va=t.get("va", "top"),
        )

    ax.set_xlabel(xlabel, fontsize=style.get("labelsize", 18))
    ax.tick_params(axis="both", which="both", labelsize=style.get("ticksize", 14))
    ax2.tick_params(axis="y", which="both", labelsize=style.get("ticksize", 14))

    if xlim is None:
        xlim = (edges[0], edges[-1])
    ax.set_xlim(*xlim)
    if logx:
        ax.set_xscale("log")
        ax2.set_xscale("log")

    # CMS label (stick it to LEFT axis so it follows the main plot)
    hep.cms.label(
        ax=ax,
        label=style.get("cms_label", ""),
        data=bool(style.get("data", False)),
        com=float(style.get("com", 13.6)),
        loc=0,
    )

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=style.get("legendsize", 14))

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)
    return outpath



def plot_hist(cfg, env, outdir="images", style=None):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, cfg.get("out", "hist.png"))

    style = style or {}
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ymax = 0.0

    bcfg = cfg["bins"]
    if "edges" in bcfg:
        bins = np.asarray(bcfg["edges"], dtype=float)
    else:
        bins = np.linspace(float(bcfg["xmin"]), float(bcfg["xmax"]), int(bcfg["nbins"]) + 1)

    density = bool(cfg.get("normalize", True))

    base_var = cfg.get("variable", None)
    base_vals = eval_expr(base_var, env) if base_var is not None else None

    for sel in cfg["selections"]:
        st = sel.get("style", {})
        color = st.get("color", None)
        var_expr = sel.get("variable", base_var)
        if var_expr is None:
            raise ValueError("plot_hist needs either cfg['variable'] or each selection['variable'].")

        vals = base_vals if (var_expr == base_var and base_vals is not None) else eval_expr(var_expr, env)

        m = ak.ones_like(vals, dtype=bool)
        for expr in sel.get("mask", []):
            mm = eval_expr(expr, env)
            if hasattr(mm, "ndim") and mm.ndim == 1 and hasattr(vals, "ndim") and vals.ndim > 1:
                mm = mm[:, None]
            m = m & mm

        x = vals[m]
        x = ak.flatten(x, axis=None)
        x = ak.to_numpy(ak.fill_none(x, np.nan))
        x = x[np.isfinite(x)]

        counts, _, _ = ax.hist(
            x, bins=bins,
            histtype=st.get("histtype", "step"),
            linewidth=st.get("linewidth", 3.0),
            linestyle=st.get("linestyle", "-"),
            label=sel.get("label", ""),
            color=color,
            density=density,
        )
    
        ymax = max(ymax, float(np.max(counts)) if len(counts) else 0.0)

    ax.set_xlabel(cfg.get("xlabel", ""), fontsize=style.get("labelsize", 18))
    ax.set_ylabel(cfg.get("ylabel", "Density" if density else "Events"),
                  fontsize=style.get("labelsize", 18))
    ax.tick_params(axis="both", which="both", labelsize=style.get("ticksize", 14))
    if ymax > 0:
        ax.set_ylim(0.0, 1.4 * ymax)

    hep.cms.label(
    ax=ax,
    label=style.get("cms_label", ""),
    data=bool(style.get("data", False)),
    com=float(style.get("com", 13.6)),
    loc=0,
    )

    for t in cfg.get("text", []) or []:
        ax.text(
            float(t.get("x", 0.05)),
            float(t.get("y", 0.85)),
            t.get("s", ""),
            transform=ax.transAxes,
            fontsize=int(t.get("fontsize", 18)),
            ha=t.get("ha", "left"),
            va=t.get("va", "top"),
        )

    leg = cfg.get("legend", {}) or {}
    ax.legend(
        frameon=False,
        loc=leg.get("loc", "upper right"),
        fontsize=leg.get("fontsize", style.get("legendsize", 14)),
        ncol=leg.get("ncol", 1),
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)
    return outpath

def plot2d_hist(cfg, env, outdir="images", style=None):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, cfg.get("out", "hist2d.png"))
    style = style or {}

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # bins
    bx = cfg["xbins"]
    by = cfg["ybins"]
    xedges = np.asarray(bx.get("edges", np.linspace(bx["xmin"], bx["xmax"], bx["nbins"] + 1)), float)
    yedges = np.asarray(by.get("edges", np.linspace(by["xmin"], by["ymax"], by["nbins"] + 1)), float)

    xexpr = cfg["x"]
    yexpr = cfg["y"]

    # mask
    m = None
    for expr in cfg.get("mask", []) or []:
        mm = eval_expr(expr, env)
        m = mm if m is None else (m & mm)

    x = eval_expr(xexpr, env)
    y = eval_expr(yexpr, env)
    if m is not None:
        x = x[m]
        y = y[m]

    x = _to_numpy_flat(x)
    y = _to_numpy_flat(y)

    weights = None
    if "weight" in cfg and cfg["weight"] is not None:
        w = eval_expr(cfg["weight"], env)
        if m is not None:
            w = w[m]
        weights = _to_numpy_flat(w)

    # 2D hist
    H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges], weights=weights)
    if bool(cfg.get("normalize", False)):
        s = H.sum()
        if s > 0:
            H = H / s

    # draw
    im = ax.pcolormesh(xe, ye, H.T, shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cfg.get("clabel", "Events"), fontsize=style.get("labelsize", 18))
    cbar.ax.tick_params(labelsize=style.get("ticksize", 14))

    ax.set_xlabel(cfg.get("xlabel", xexpr), fontsize=style.get("labelsize", 18))
    ax.set_ylabel(cfg.get("ylabel", yexpr), fontsize=style.get("labelsize", 18))
    ax.tick_params(axis="both", which="both", labelsize=style.get("ticksize", 14))

    hep.cms.label(
        ax=ax,
        label=style.get("cms_label", ""),
        data=bool(style.get("data", False)),
        com=float(style.get("com", 13.6)),
        loc=0,
    )

    for t in cfg.get("text", []) or []:
        ax.text(
            float(t.get("x", 0.05)),
            float(t.get("y", 0.85)),
            t.get("s", ""),
            transform=ax.transAxes,
            fontsize=int(t.get("fontsize", 18)),
            ha=t.get("ha", "left"),
            va=t.get("va", "top"),
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)
    return outpath


def plot2d_profile(cfg, env, outdir="images", style=None):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, cfg.get("out", "profile.png"))
    style = style or {}

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    bx = cfg["xbins"]
    xedges = np.asarray(bx.get("edges", np.linspace(bx["xmin"], bx["xmax"], bx["nbins"] + 1)), float)

    xexpr = cfg["x"]
    yexpr = cfg["y"]

    m = None
    for expr in cfg.get("mask", []) or []:
        mm = eval_expr(expr, env)
        m = mm if m is None else (m & mm)

    x = eval_expr(xexpr, env)
    y = eval_expr(yexpr, env)
    if m is not None:
        x = x[m]
        y = y[m]

    x = _to_numpy_flat(x)
    y = _to_numpy_flat(y)

    # bin
    inds = np.digitize(x, xedges) - 1
    nb = len(xedges) - 1

    means = np.full(nb, np.nan)
    errs  = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=int)

    for i in range(nb):
        sel = inds == i
        yy = y[sel]
        counts[i] = yy.size
        if yy.size > 0:
            means[i] = np.mean(yy)
            # standard error on mean:
            errs[i] = np.std(yy, ddof=1) / np.sqrt(yy.size) if yy.size > 1 else 0.0

    centers = 0.5 * (xedges[:-1] + xedges[1:])
    good = np.isfinite(means)

    ax.errorbar(
        centers[good], means[good], yerr=errs[good],
        fmt=cfg.get("marker", "o"),
        linestyle="None",
        capsize=2.5,
    )

    ax.set_xlabel(cfg.get("xlabel", xexpr), fontsize=style.get("labelsize", 18))
    ax.set_ylabel(cfg.get("ylabel", f"<{yexpr}>"), fontsize=style.get("labelsize", 18))
    ax.tick_params(axis="both", which="both", labelsize=style.get("ticksize", 14))

    hep.cms.label(
        ax=ax,
        label=style.get("cms_label", ""),
        data=bool(style.get("data", False)),
        com=float(style.get("com", 13.6)),
        loc=0,
    )

    for t in cfg.get("text", []) or []:
        ax.text(
            float(t.get("x", 0.05)),
            float(t.get("y", 0.85)),
            t.get("s", ""),
            transform=ax.transAxes,
            fontsize=int(t.get("fontsize", 18)),
            ha=t.get("ha", "left"),
            va=t.get("va", "top"),
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)
    return outpath
