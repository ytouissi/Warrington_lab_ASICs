import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control
from pathlib import Path
import os, json, sys, gc
import string
import matplotlib.font_manager as fm

INPUT_FILE  = "./SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
CONFIG_FILE = "./config.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, CONFIG_FILE), "r") as _f:
    _cfg = json.load(_f)

GENE_LIST  = _cfg["gene_list"]
SEX_FILTER = _cfg.get("sex_filter", "Both")

DONOR_COL = "Donor ID"
COG_COL   = "Cognitive Status"
SEX_COL   = "Sex"
PMI_COL   = "PMI"
AGE_COL   = "Age at Death"
APOE_COL  = "APOE Genotype"
ADNC_COL  = "Overall AD neuropathological Change"
CPS_COL   = "Continuous Pseudo-progression Score"

def sanitize(name):
    for ch in r'/\*?"<>|':
        name = name.replace(ch, "_")
    return name

OUT_DIR = os.path.join(SCRIPT_DIR, "PanelB_Pseudobulk_Strip_results")
os.makedirs(OUT_DIR, exist_ok=True)

WITHIN_GAP  = 0.85 
RNG = np.random.default_rng(42)

DEMENTIA_COLOR    = "#C62828"
NO_DEMENTIA_COLOR = "#1565C0"

def _darken(hex_color, factor=0.45):
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

def _qstr(q, mode='dec', bold=False):
    if pd.isna(q):
        return "p=NA"
    if mode == 'sci':
        if q < 0.001:
            exp  = int(np.floor(np.log10(abs(q))))
            coef = q / (10 ** exp)
            if bold:
                return f"$\\bf{{p={coef:.2f}\\!\\times\\!10^{{{exp}}}}}$"
            return f"$p={coef:.2f}\\!\\times\\!10^{{{exp}}}$"
        else:
            if bold:
                return f"$\\bf{{p={q:.4f}}}$"
            return f"p={q:.4f}"
    else:
        if q < 0.0001:
            res = "p<0.0001"
        else:
            res = f"p={q:.4f}"
        if bold:
            return f"$\\bf{{{res}}}$"
        return res

def build_pseudobulk():
    fpath = os.path.join(SCRIPT_DIR, INPUT_FILE)
    if not Path(fpath).exists():
        sys.exit(f"ERROR Input file not found {fpath}")

    print(f"\nOpening file (backed) ALL cells ...")
    adata = ad.read_h5ad(fpath, backed="r")
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    obs = adata.obs.copy()

    results = {}

    for gene in GENE_LIST:
        print(f"\n  Gene {gene}")

        gu    = gene.upper()
        match = adata.var.index.str.upper() == gu
        if not match.any():
            for col in ["gene_symbol", "gene_name", "gene_ids", "feature_name"]:
                if col in adata.var.columns:
                    match = adata.var[col].astype(str).str.upper() == gu
                    if match.any():
                        break
        if not match.any():
            print(f"    [WARNING] '{gene}' not found skipping.")
            results[gene] = None
            continue

        gene_idx = int(np.where(match)[0][0])

        gene_counts = None
        for ln in ["raw", "UMIs", "counts", "spliced"]:
            if ln in adata.layers:
                raw         = adata.layers[ln][:, gene_idx]
                gene_counts = raw.toarray().flatten() if hasattr(raw, "toarray") else np.asarray(raw).flatten()
                break
        if gene_counts is None:
            raw         = adata.X[:, gene_idx]
            gene_counts = raw.toarray().flatten() if hasattr(raw, "toarray") else np.asarray(raw).flatten()

        df = pd.DataFrame({"donor": obs[DONOR_COL].values, "gene_raw": gene_counts})
        found_umi = False
        for try_col in ["Number of UMIs", "n_counts", "nCount_RNA"]:
            if try_col in obs.columns:
                df["total_umis"] = pd.to_numeric(obs[try_col].values, errors="coerce")
                found_umi = True
                break
        if not found_umi:
            CHUNK  = 500
            totals = np.zeros(adata.n_obs, dtype=np.float64)
            for start in range(0, adata.n_obs, CHUNK):
                x_chunk  = adata.X[start:start + CHUNK, :]
                row_sums = np.array(x_chunk.sum(axis=1)).flatten() if hasattr(x_chunk, "toarray") else x_chunk.sum(axis=1)
                totals[start:start + CHUNK] = row_sums
            df["total_umis"] = totals

        for col, label in [(COG_COL, "cog_status"), (SEX_COL, "sex"),
                           (PMI_COL, "pmi"), (AGE_COL, "age"), (APOE_COL, "apoe"),
                           (ADNC_COL, "adnc"), (CPS_COL, "cps")]:
            if col in obs.columns:
                df[label] = obs[col].values

        df = df.dropna(subset=["donor", "total_umis"])
        df = df[df["total_umis"] > 0]

        agg = df.groupby("donor", observed=True).agg(
            gene_sum       =("gene_raw",   "sum"),
            total_umis_sum =("total_umis", "sum"),
            n_cells        =("gene_raw",   "count"),
            n_expressing   =("gene_raw",   lambda x: (x > 0).sum()),
        ).reset_index()

        if "sex" in df.columns and SEX_FILTER.lower() != "both":
            sex_map = df.groupby("donor", observed=True)["sex"].first().reset_index()
            agg     = agg.merge(sex_map, on="donor", how="left")
            agg     = agg[agg["sex"].astype(str).str.lower() == SEX_FILTER.lower()]

        agg["up10k"]    = (agg["gene_sum"] / agg["total_umis_sum"]) * 1e4
        agg["ln_up10k"] = np.log(agg["up10k"] + 1)

        for label in ["cog_status", "sex", "pmi", "age", "apoe", "adnc", "cps"]:
            if label in df.columns and label not in agg.columns:
                cov = df.groupby("donor", observed=True)[label].first().reset_index()
                agg = agg.merge(cov, on="donor", how="left")

        if "cog_status" in agg.columns:
            cog   = agg["cog_status"].astype(str).str.strip()
            dem   = [v for v in cog.unique() if "dementia" in v.lower() and "no" not in v.lower()]
            nodem = [v for v in cog.unique() if "no" in v.lower() and "dementia" in v.lower()]
            agg["is_dementia"]    = cog.isin(dem).astype(int)
            agg["is_no_dementia"] = cog.isin(nodem).astype(int)

        met = "ln_up10k"
        gd  = agg.loc[agg.get("is_dementia",    pd.Series(dtype=int)) == 1, met].dropna().values
        gn  = agg.loc[agg.get("is_no_dementia", pd.Series(dtype=int)) == 1, met].dropna().values
        raw_p, test_name = np.nan, "N/A"
        if len(gd) >= 3 and len(gn) >= 3:
            norm_ok = True
            for vals in [gd, gn]:
                if len(vals) >= 8:
                    _, sp = stats.shapiro(vals)
                    if sp < 0.05:
                        norm_ok = False
                else:
                    norm_ok = False
            _, lev_p = stats.levene(gd, gn, center="median") if (len(gd) >= 2 and len(gn) >= 2) else (0, 0.5)
            var_ok = lev_p >= 0.05
            if norm_ok and var_ok:
                _, raw_p = stats.ttest_ind(gd, gn, equal_var=True)
                test_name = "Student's t"
            elif norm_ok and not var_ok:
                _, raw_p = stats.ttest_ind(gd, gn, equal_var=False)
                test_name = "Welch's t"
            else:
                _, raw_p = stats.mannwhitneyu(gd, gn, alternative="two-sided")
                test_name = "MWU"

        print(f"    n_donors Dem={len(gd)}, No-Dem={len(gn)}  |  test={test_name}  raw_p={raw_p:.4f}" if not np.isnan(raw_p) else f"    n_donors Dem={len(gd)}, No-Dem={len(gn)}  |  test=N/A")
        results[gene] = {"pb": agg, "raw_p": raw_p, "test": test_name}

        del df, gene_counts
        gc.collect()

    try:
        adata.file.close()
    except Exception:
        pass
    del adata, obs
    gc.collect()
    print("\nFile closed.")
    return results

def apply_bh_fdr(results):
    keys, pvals = [], []
    for gene, entry in results.items():
        if entry is not None and not pd.isna(entry["raw_p"]):
            keys.append(gene)
            pvals.append(entry["raw_p"])

    if not pvals:
        return

    pvals_arr = np.array(pvals)
    try:
        qvals = false_discovery_control(pvals_arr, method="bh")
    except AttributeError:
        n    = len(pvals_arr)
        rank = np.argsort(pvals_arr)
        qvals = np.empty(n)
        qvals[rank] = pvals_arr[rank] * n / (np.arange(n) + 1)
        qvals = np.minimum.accumulate(qvals[::-1])[::-1]
        qvals = np.minimum(qvals, 1.0)

    print(f"\nBH-FDR applied across {len(pvals)} genes")
    for gene, q, p in zip(keys, qvals, pvals_arr):
        results[gene]["adj_q"] = q
        sig = " *" if q < 0.05 else ""
        print(f"  {gene:12s}  raw_p={p:.4f}  q={q:.4f}{sig}")

    for gene, entry in results.items():
        if entry is not None and "adj_q" not in entry:
            entry["adj_q"] = np.nan

def _render_plot_panel(cur_ax, item, gene, letter_label=None, p_format='dec'):
    half = WITHIN_GAP / 2.0
    xn   = -half
    xd   =  half

    if letter_label:
        cur_ax.text(
            -0.14, 1.08, letter_label,
            transform=cur_ax.transAxes,
            fontsize=20, fontweight="bold",
            va="top", ha="right"
        )

    if item is None:
        cur_ax.set_visible(False)
        return

    v_nd, v_d, adj_q, test = item

    vals_all = np.concatenate([v_nd, v_d]) if (len(v_nd) > 0 and len(v_d) > 0) else (v_nd if len(v_nd) > 0 else v_d)
    if len(vals_all) == 0:
        cur_ax.set_visible(False)
        return

    lo, hi    = np.min(vals_all), np.max(vals_all)
    yrange    = hi - lo if hi > lo else max(abs(hi) * 0.15, 0.15)
    
    step      = yrange * 0.20
    y_bracket = hi + step * 1.2
    y_bracket_text = y_bracket + step * 0.2
    
    cur_ax.set_ylim(bottom=lo - yrange * 0.10, top=y_bracket + step * 2.8)

    cur_ax.set_ylabel(f"{gene} Expression\nln(UP10K+1)", fontsize=18, fontweight="bold", labelpad=8)
    cur_ax.set_xlabel("Dementia Status", fontsize=18, fontweight="bold", labelpad=10)

    for xpos, vals, color, mark in [(xn, v_nd, NO_DEMENTIA_COLOR, "o"), (xd, v_d, DEMENTIA_COLOR, "^")]:
        if len(vals) == 0:
            continue
        bp = cur_ax.boxplot(
            vals,
            positions=[xpos],
            widths=WITHIN_GAP * 0.35,
            patch_artist=True,
            medianprops=dict(color=_darken(color, 0.45), linewidth=2.8, solid_capstyle="butt", zorder=7),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="", linewidth=0),
            manage_ticks=False,
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor(color)
        bp["boxes"][0].set_linewidth(1.2)

        jit = RNG.normal(0, 0.040, len(vals))
        cur_ax.scatter(
            np.full(len(vals), xpos) + jit, vals,
            c=color, marker=mark,
            edgecolors="black", linewidths=0.4,
            s=28, zorder=4, alpha=0.88,
        )

    if len(v_nd) > 0 and len(v_d) > 0:
        is_sig = (not pd.isna(adj_q)) and (adj_q < 0.05)
        cur_ax.plot(
            [xn, xn, xd, xd],
            [y_bracket - step * 0.15, y_bracket, y_bracket, y_bracket - step * 0.15],
            color="black", linewidth=2.0 if is_sig else 0.9, clip_on=False,
        )
        cur_ax.text(
            0.0, y_bracket_text,
            _qstr(adj_q, mode=p_format, bold=False),
            ha="center", va="bottom", fontsize=11,
        )

    cur_ax.set_xticks([xn, xd])
    cur_ax.set_xticklabels(["\u2212Dem", "+Dem"], fontsize=15, fontweight="bold")
    cur_ax.set_xlim(xn - WITHIN_GAP * 0.75, xd + WITHIN_GAP * 0.75)
    cur_ax.tick_params(axis="x", labelsize=15)
    cur_ax.tick_params(axis="y", labelsize=12)
    cur_ax.set_title(gene, fontsize=16, fontweight="bold", fontstyle="italic", pad=10)
    cur_ax.spines["top"].set_visible(False)
    cur_ax.spines["right"].set_visible(False)

def draw_strip_horizontal(results, base_filename, p_format='dec'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_genes = len(GENE_LIST)
    met     = "ln_up10k"

    gene_data = []
    for gene in GENE_LIST:
        entry = results.get(gene)
        if entry is None:
            gene_data.append(None)
            continue
        pb    = entry["pb"]
        adj_q = entry.get("adj_q", np.nan)
        test  = entry.get("test", "N/A")
        if "is_dementia" not in pb.columns:
            gene_data.append(None)
            continue
        v_nd = pb.loc[pb["is_no_dementia"] == 1, met].dropna().values
        v_d  = pb.loc[pb["is_dementia"]    == 1, met].dropna().values
        gene_data.append((v_nd, v_d, adj_q, test))

    panel_w  = 2.8
    pad_w    = 1.10
    margin_l = 1.00
    margin_r = 0.40
    margin_t = 0.18
    margin_b = 0.25

    fig_w = margin_l + n_genes * panel_w + (n_genes - 1) * pad_w + margin_r
    fig_h = 6.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_parent = fig.add_axes([0, 0, 1, 1])
    ax_parent.set_axis_off()

    inset_h = 1.0 - margin_t - margin_b
    letters = list(string.ascii_uppercase)

    axes_list = []
    for gi in range(n_genes):
        left = (margin_l + gi * (panel_w + pad_w)) / fig_w
        w    = panel_w / fig_w
        ax_i = fig.add_axes([left, margin_b, w, inset_h])
        axes_list.append(ax_i)

    for gi, (gene, item) in enumerate(zip(GENE_LIST, gene_data)):
        cur_ax = axes_list[gi]
        letter_label = letters[gi] if gi < len(letters) else f"({gi+1})"
        _render_plot_panel(cur_ax, item, gene, letter_label=letter_label, p_format=p_format)

    out_png  = os.path.join(OUT_DIR, f"{base_filename}.png")
    out_tiff = os.path.join(OUT_DIR, f"{base_filename}.tiff")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tiff, dpi=300, bbox_inches="tight", facecolor="white", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_tiff}")

def draw_strip_vertical(results, base_filename, p_format='dec'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_genes = len(GENE_LIST)
    met     = "ln_up10k"

    gene_data = []
    for gene in GENE_LIST:
        entry = results.get(gene)
        if entry is None:
            gene_data.append(None)
            continue
        pb    = entry["pb"]
        adj_q = entry.get("adj_q", np.nan)
        test  = entry.get("test", "N/A")
        if "is_dementia" not in pb.columns:
            gene_data.append(None)
            continue
        v_nd = pb.loc[pb["is_no_dementia"] == 1, met].dropna().values
        v_d  = pb.loc[pb["is_dementia"]    == 1, met].dropna().values
        gene_data.append((v_nd, v_d, adj_q, test))

    panel_w = 2.8
    margin_l = 1.00
    margin_r = 0.40
    fig_w = margin_l + panel_w + margin_r
    fig_h = 5.2 * n_genes

    fig, axes = plt.subplots(
        nrows=n_genes,
        ncols=1,
        figsize=(fig_w, fig_h),
        squeeze=False
    )
    plt.subplots_adjust(hspace=0.45)

    letters = list(string.ascii_uppercase)

    for gi, (gene, item) in enumerate(zip(GENE_LIST, gene_data)):
        cur_ax = axes[gi, 0]
        letter_label = letters[gi] if gi < len(letters) else f"({gi+1})"
        _render_plot_panel(cur_ax, item, gene, letter_label=letter_label, p_format=p_format)

    out_png  = os.path.join(OUT_DIR, f"{base_filename}.png")
    out_tiff = os.path.join(OUT_DIR, f"{base_filename}.tiff")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tiff, dpi=300, bbox_inches="tight", facecolor="white", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_tiff}")

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys_treb   = [p for p in fm.findSystemFonts() if "trebuc" in p.lower()]
    local_treb = [os.path.join(SCRIPT_DIR, f) for f in os.listdir(SCRIPT_DIR)
                  if "trebuc" in f.lower() and f.lower().endswith(".ttf")]
    treb = local_treb + sys_treb
    if treb:
        for tp in treb:
            try: fm.fontManager.addfont(tp)
            except Exception: pass
        fname = fm.FontProperties(fname=treb[0]).get_name()
    else:
        fname = "DejaVu Sans"

    plt.rcParams.update({
        "font.family":       fname,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "text.usetex":       False,
    })

    print("PSEUDOBULK STRIP PLOT")
    print(f"  Genes {GENE_LIST}")
    print(f"  Sex filter {SEX_FILTER}")
    print(f"  Tests {len(GENE_LIST)} total (BH-FDR)")

    results = build_pseudobulk()

    apply_bh_fdr(results)

    print("\nBuilding figures ...")
    gene_label = "_".join(sanitize(g) for g in GENE_LIST)

    draw_strip_horizontal(
        results,
        base_filename=f"Pseudobulk_Strip_Horizontal_{gene_label}_dec",
        p_format='dec'
    )

    draw_strip_vertical(
        results,
        base_filename=f"Pseudobulk_Strip_Vertical_{gene_label}_dec",
        p_format='dec'
    )

    print("\nDone.")

if __name__ == "__main__":
    main()