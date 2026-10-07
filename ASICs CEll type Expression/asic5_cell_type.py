import warnings
import os
import json
import sys
import gc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.stats import false_discovery_control
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

warnings.filterwarnings("ignore")

INPUT_FILE = "SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
CONFIG_FILE = "config.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, CONFIG_FILE), "r") as _f:
    _cfg = json.load(_f)

_raw_cell_type = _cfg["cell_type"]
CELL_TYPE_LIST = _raw_cell_type if isinstance(_raw_cell_type, list) else [_raw_cell_type]
GENE_LIST = _cfg["gene_list"]
SUBCLASS_COL = _cfg["subclass_col"]
SEX_FILTER = _cfg.get("sex_filter", "Both")

DONOR_COL = "Donor ID"
ADNC_COL = "Overall AD neuropathological Change"
COG_COL = "Cognitive Status"
SEX_COL = "Sex"
PMI_COL = "PMI"
AGE_COL = "Age at Death"
APOE_COL = "APOE Genotype"
CPS_COL = "Continuous Pseudo-progression Score"

GENES_PER_FIGURE = 6
GRAPH_GENES = list(GENE_LIST)

def sanitize(name):
    for ch in r'/\*?"<>|':
        name = name.replace(ch, "_")
    return name

OUT_DIR = os.path.join(SCRIPT_DIR, "PanelB_Grid_results")
os.makedirs(OUT_DIR, exist_ok=True)

_PALETTE = [
    {"color": "#000000", "marker": "o"},
    {"color": "#E91E8C", "marker": "^"},
    {"color": "#00897B", "marker": "s"},
    {"color": "#7B1FA2", "marker": "v"},
    {"color": "#1565C0", "marker": "D"},
    {"color": "#F57F17", "marker": "P"},
    {"color": "#558B2F", "marker": "*"},
    {"color": "#C62828", "marker": "X"},
]

def _style(idx):
    return _PALETTE[idx % len(_PALETTE)]

def _qstr(q, mode='dec', bold=False):
    if pd.isna(q):
        return "p=NA"
    if mode == 'sci':
        if q < 0.001:
            exp = int(np.floor(np.log10(abs(q))))
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

def _qstar(q, mode='dec'):
    if pd.isna(q) or q >= 0.05:
        return "ns"
    return _qstr(q, mode)

def process_cell_type(cell_type):
    fpath = os.path.join(SCRIPT_DIR, INPUT_FILE)
    if not Path(fpath).exists():
        sys.exit(f"ERROR Input file not found {fpath}")

    adata = ad.read_h5ad(fpath, backed="r")
    m = adata.obs[SUBCLASS_COL].astype(str).isin([cell_type])
    if m.sum() == 0:
        adata.file.close()
        del adata
        gc.collect()
        return {}

    cell_indices = np.where(m)[0]
    obs_sub = adata.obs.iloc[cell_indices].copy()
    results = {}

    for gene in GENE_LIST:
        gu = gene.upper()
        match = adata.var.index.str.upper() == gu
        if not match.any():
            for col in ["gene_symbol", "gene_name", "gene_ids", "feature_name"]:
                if col in adata.var.columns:
                    match = adata.var[col].astype(str).str.upper() == gu
                    if match.any():
                        break
        if not match.any():
            results[gene] = None
            continue

        gene_idx = int(np.where(match)[0][0])
        gene_counts = None
        for ln in ["raw", "UMIs", "counts", "spliced"]:
            if ln in adata.layers:
                raw = adata.layers[ln][cell_indices, gene_idx]
                gene_counts = raw.toarray().flatten() if hasattr(raw, "toarray") else np.asarray(raw).flatten()
                break
        if gene_counts is None:
            raw = adata.X[cell_indices, gene_idx]
            gene_counts = raw.toarray().flatten() if hasattr(raw, "toarray") else np.asarray(raw).flatten()

        df = pd.DataFrame({"donor": obs_sub[DONOR_COL].values, "gene_raw": gene_counts})
        found_umi = False
        for try_col in ["Number of UMIs", "n_counts", "nCount_RNA"]:
            if try_col in obs_sub.columns:
                df["total_umis"] = pd.to_numeric(obs_sub[try_col].values, errors="coerce")
                found_umi = True
                break
        if not found_umi:
            CHUNK = 500
            totals = np.zeros(len(cell_indices), dtype=np.float64)
            for start in range(0, len(cell_indices), CHUNK):
                chunk_idx = cell_indices[start:start + CHUNK]
                x_chunk = adata.X[chunk_idx, :]
                row_sums = np.array(x_chunk.sum(axis=1)).flatten() if hasattr(x_chunk, "toarray") else x_chunk.sum(axis=1)
                totals[start:start + CHUNK] = row_sums
            df["total_umis"] = totals

        for col, label in [(COG_COL, "cog_status"), (SEX_COL, "sex"),
                           (PMI_COL, "pmi"), (AGE_COL, "age"), (APOE_COL, "apoe"),
                           (ADNC_COL, "adnc"), (CPS_COL, "cps")]:
            if col in obs_sub.columns:
                df[label] = obs_sub[col].values

        df = df.dropna(subset=["donor", "total_umis"])
        df = df[df["total_umis"] > 0]

        agg = df.groupby("donor", observed=True).agg(
            gene_sum=("gene_raw", "sum"),
            total_umis_sum=("total_umis", "sum"),
            n_cells=("gene_raw", "count"),
            n_expressing=("gene_raw", lambda x: (x > 0).sum()),
        ).reset_index()

        if "sex" in df.columns and SEX_FILTER.lower() != "both":
            sex_map = df.groupby("donor", observed=True)["sex"].first().reset_index()
            agg = agg.merge(sex_map, on="donor", how="left")
            agg = agg[agg["sex"].astype(str).str.lower() == SEX_FILTER.lower()]

        agg["up10k"] = (agg["gene_sum"] / agg["total_umis_sum"]) * 1e4
        agg["ln_up10k"] = np.log(agg["up10k"] + 1)

        for label in ["cog_status", "sex", "pmi", "age", "apoe", "adnc", "cps"]:
            if label in df.columns and label not in agg.columns:
                cov = df.groupby("donor", observed=True)[label].first().reset_index()
                agg = agg.merge(cov, on="donor", how="left")

        if "cog_status" in agg.columns:
            cog = agg["cog_status"].astype(str).str.strip()
            dem = [v for v in cog.unique() if "dementia" in v.lower() and "no" not in v.lower()]
            nodem = [v for v in cog.unique() if "no" in v.lower() and "dementia" in v.lower()]
            agg["is_dementia"] = cog.isin(dem).astype(int)
            agg["is_no_dementia"] = cog.isin(nodem).astype(int)

        met = "ln_up10k"
        gd = agg.loc[agg.get("is_dementia", pd.Series(dtype=int)) == 1, met].dropna().values
        gn = agg.loc[agg.get("is_no_dementia", pd.Series(dtype=int)) == 1, met].dropna().values
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
            _, lev_p = stats.levene(gd, gn, center="median") if (len(gd)>=2 and len(gn)>=2) else (0, 0.5)
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

        results[gene] = {"pb": agg, "raw_p": raw_p, "test": test_name}
        del df, gene_counts
        gc.collect()

    try:
        adata.file.close()
    except Exception:
        pass
    del adata, obs_sub, cell_indices
    gc.collect()
    return results

def apply_bh_fdr(all_data):
    keys, pvals = [], []
    for gene in GENE_LIST:
        for col_idx in range(len(CELL_TYPE_LIST)):
            entry = all_data[col_idx].get(gene)
            if entry is not None and not pd.isna(entry["raw_p"]):
                keys.append((gene, col_idx))
                pvals.append(entry["raw_p"])

    if not pvals:
        return

    pvals_arr = np.array(pvals)

    try:
        qvals = false_discovery_control(pvals_arr, method="bh")
    except AttributeError:
        n = len(pvals_arr)
        rank = np.argsort(pvals_arr)
        qvals = np.empty(n)
        qvals[rank] = pvals_arr[rank] * n / (np.arange(n) + 1)
        qvals = np.minimum.accumulate(qvals[::-1])[::-1]
        qvals = np.minimum(qvals, 1.0)

    for (gene, col_idx), q, p in zip(keys, qvals, pvals_arr):
        all_data[col_idx][gene]["adj_q"] = q

    for gene in GENE_LIST:
        for col_idx in range(len(CELL_TYPE_LIST)):
            entry = all_data[col_idx].get(gene)
            if entry is not None and "adj_q" not in entry:
                entry["adj_q"] = np.nan

def export_results_csv(all_data):
    met = "ln_up10k"
    rows = []

    for gene in GENE_LIST:
        for col_idx, cell_type in enumerate(CELL_TYPE_LIST):
            entry = all_data[col_idx].get(gene)
            if entry is None:
                rows.append({
                    "gene": gene, "cell_type": cell_type,
                    "test": "N/A",
                    "n_nodem": np.nan, "n_dem": np.nan,
                    "mean_nodem": np.nan, "sd_nodem": np.nan, "median_nodem": np.nan,
                    "mean_dem": np.nan, "sd_dem": np.nan, "median_dem": np.nan,
                    "mean_diff (dem - nodem)": np.nan,
                    "raw_p": np.nan, "adj_q_global_BH": np.nan,
                    "significant (q<0.05)": "N/A",
                })
                continue

            pb = entry["pb"]
            raw_p = entry.get("raw_p", np.nan)
            adj_q = entry.get("adj_q", np.nan)
            test = entry.get("test", "N/A")

            v_nd = pb.loc[pb.get("is_no_dementia", pd.Series(dtype=int)) == 1, met].dropna().values
            v_d = pb.loc[pb.get("is_dementia", pd.Series(dtype=int)) == 1, met].dropna().values

            def _stats(arr):
                if len(arr) == 0:
                    return np.nan, np.nan, np.nan
                return float(np.mean(arr)), float(np.std(arr, ddof=1)), float(np.median(arr))

            mn_nd, sd_nd, med_nd = _stats(v_nd)
            mn_d, sd_d, med_d = _stats(v_d)
            diff = (mn_d - mn_nd) if not (np.isnan(mn_d) or np.isnan(mn_nd)) else np.nan
            sig = "Yes" if (not pd.isna(adj_q) and adj_q < 0.05) else "No"

            rows.append({
                "gene": gene,
                "cell_type": cell_type,
                "test": test,
                "n_nodem": len(v_nd),
                "n_dem": len(v_d),
                "mean_nodem": round(mn_nd, 6) if not np.isnan(mn_nd) else np.nan,
                "sd_nodem": round(sd_nd, 6) if not np.isnan(sd_nd) else np.nan,
                "median_nodem": round(med_nd, 6) if not np.isnan(med_nd) else np.nan,
                "mean_dem": round(mn_d, 6) if not np.isnan(mn_d) else np.nan,
                "sd_dem": round(sd_d, 6) if not np.isnan(sd_d) else np.nan,
                "median_dem": round(med_d, 6) if not np.isnan(med_d) else np.nan,
                "mean_diff (dem - nodem)": round(diff, 6) if not np.isnan(diff) else np.nan,
                "raw_p": raw_p,
                "adj_q_global_BH": adj_q,
                "significant (q<0.05)": sig,
            })

    df_out = pd.DataFrame(rows)
    cell_label = "_".join(sanitize(ct) for ct in CELL_TYPE_LIST)
    csv_path = os.path.join(OUT_DIR, f"PanelB_Grid_{cell_label}_all_results.csv")
    df_out.to_csv(csv_path, index=False)

def _darken(hex_color, factor=0.45):
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

RNG = np.random.default_rng(42)

def draw_gene_panel(ax, gene, all_data, p_format):
    met = "ln_up10k"
    n_ct = len(CELL_TYPE_LIST)
    
    within_gap = 0.90
    between_gap = 2.10
    centres = np.arange(n_ct) * between_gap
    half = within_gap / 2.0
    x_nodem = centres - half
    x_dem = centres + half
    
    pair_data = []
    global_ymax = -np.inf
    global_yrange = 0.0

    for ct_idx, cell_type in enumerate(CELL_TYPE_LIST):
        sty = _style(ct_idx)
        entry = all_data[ct_idx].get(gene)
        if entry is None:
            pair_data.append(None)
            continue
        pb = entry["pb"]
        adj_q = entry.get("adj_q", np.nan)
        if "is_dementia" not in pb.columns or "is_no_dementia" not in pb.columns:
            pair_data.append(None)
            continue
        v_nd = pb.loc[pb["is_no_dementia"] == 1, met].dropna().values
        v_d = pb.loc[pb["is_dementia"] == 1, met].dropna().values
        pair_data.append((ct_idx, sty["color"], sty["marker"], v_nd, v_d, adj_q))
        if len(v_nd) > 0 and len(v_d) > 0:
            pair_ymax = max(np.max(v_nd), np.max(v_d))
            pair_ymin = min(np.min(v_nd), np.min(v_d))
            pair_yrange = pair_ymax - pair_ymin if pair_ymax > pair_ymin else max(abs(pair_ymax) * 0.15, 0.15)
            if pair_ymax > global_ymax:
                global_ymax = pair_ymax
                global_yrange = pair_yrange

    step = global_yrange * 0.18 if global_yrange > 0 else 0.15
    y_bracket = global_ymax + step * 1.0 if global_ymax > -np.inf else 1.0

    for item in pair_data:
        if item is None:
            continue
        ct_idx, color, mark, v_nd, v_d, adj_q = item
        for xpos, vals in [(x_nodem[ct_idx], v_nd), (x_dem[ct_idx], v_d)]:
            if len(vals) == 0:
                continue
            median_color = _darken(color, factor=0.45)
            bp = ax.boxplot(
                vals,
                positions=[xpos],
                widths=within_gap * 0.40,
                patch_artist=True,
                medianprops=dict(color=median_color, linewidth=3.0, solid_capstyle="butt", zorder=7),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker="", linewidth=0),
                manage_ticks=False,
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(color)
            bp["boxes"][0].set_linewidth(1.2)
            jit = RNG.normal(0, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jit, vals,
                c=color, marker=mark,
                edgecolors="black", linewidths=0.4,
                s=30, zorder=4, alpha=0.88,
            )
        if len(v_nd) > 0 and len(v_d) > 0:
            is_sig = (not pd.isna(adj_q)) and (adj_q < 0.05)
            ax.plot(
                [x_nodem[ct_idx], x_nodem[ct_idx], x_dem[ct_idx], x_dem[ct_idx]],
                [y_bracket - step * 0.15, y_bracket, y_bracket, y_bracket - step * 0.15],
                color="black", linewidth=2.2 if is_sig else 1.0, clip_on=False,
            )
            p_label = _qstr(adj_q, mode=p_format, bold=False)
            ax.text(
                centres[ct_idx], y_bracket + step * 0.2,
                p_label,
                ha="center", va="bottom",
                fontsize=13,
            )

    tick_positions = []
    tick_labels = []
    for ct_idx in range(n_ct):
        tick_positions.extend([x_nodem[ct_idx], x_dem[ct_idx]])
        tick_labels.extend(["\u2212Dem", "+Dem"])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_xlim(centres[0] - between_gap * 0.55, centres[-1] + between_gap * 0.55)
    ax.set_ylim(top=y_bracket + step * 2.5)
    current_ylim = ax.get_ylim()
    y_span = current_ylim[1] - current_ylim[0]
    ax.set_ylim(bottom=current_ylim[0] - y_span * 0.05)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=13, pad=8)
    ax.set_xlabel("Dementia Status", fontsize=15, labelpad=18, fontweight="bold")
    ax.set_ylabel(f"{gene} Expression \nln(UP10K+1)", fontsize=18, labelpad=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"$\\it{{{gene}}}$", fontsize=20, pad=18, loc="center")

    legend_handles = [
        Patch(facecolor=_style(i)["color"], edgecolor="black", linewidth=0.8, label=ct)
        for i, ct in enumerate(CELL_TYPE_LIST)
    ]
    legend_font = FontProperties(size=14, weight="bold")
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=n_ct,
        prop=legend_font,
        frameon=False,
        handlelength=1.4,
        handleheight=1.2,
    )

def build_figure(gene_batch, batch_idx, all_data, n_cells, fname, plt, p_format):
    N_COLS = 3
    n_genes_b = len(gene_batch)
    n_cols = min(n_genes_b, N_COLS)
    n_rows = int(np.ceil(n_genes_b / n_cols)) if n_cols > 0 else 1
    
    base_w = 1.35
    subplot_w = 2.2 + n_cells * base_w
    subplot_h = 5.8

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * subplot_w, n_rows * subplot_h),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.70, wspace=0.35, top=0.90)

    for local_idx, gene in enumerate(gene_batch):
        row = local_idx // n_cols
        col = local_idx % n_cols
        ax = axes[row][col]
        draw_gene_panel(ax, gene, all_data, p_format)

    for extra in range(n_genes_b, n_rows * n_cols):
        row = extra // n_cols
        col = extra % n_cols
        axes[row][col].set_visible(False)

    cell_label = "_".join(sanitize(ct) for ct in CELL_TYPE_LIST)
    gene_label = "_".join(sanitize(g) for g in gene_batch)
    out_path = os.path.join(OUT_DIR, f"PanelB_Grid_{cell_label}_fig{batch_idx}_{gene_label}_{p_format}.png")
    out_path_tiff = os.path.join(OUT_DIR, f"PanelB_Grid_{cell_label}_fig{batch_idx}_{gene_label}_{p_format}.tiff")
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = 0.15
    extra_top = 0.08
    padded_bbox = Bbox.from_extents(
        tight_bbox.x0 - pad, tight_bbox.y0 - pad,
        tight_bbox.x1 + pad, tight_bbox.y1 + pad + extra_top,
    )
    fig.savefig(out_path, dpi=300, bbox_inches=padded_bbox, facecolor="white")
    fig.savefig(out_path_tiff, dpi=300, bbox_inches=padded_bbox, facecolor="white", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def main():
    sys_treb = [p for p in fm.findSystemFonts() if "trebuc" in p.lower()]
    local_treb = [os.path.join(SCRIPT_DIR, f) for f in os.listdir(SCRIPT_DIR) if "trebuc" in f.lower() and f.lower().endswith(".ttf")]
    treb = local_treb + sys_treb
    if treb:
        for tp in treb:
            try: fm.fontManager.addfont(tp)
            except Exception: pass
        fname = fm.FontProperties(fname=treb[0]).get_name()
    else:
        fname = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": fname,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.usetex": False,
    })

    n_cells = len(CELL_TYPE_LIST)
    graph_set = {g.upper() for g in GRAPH_GENES}
    GRAPH_GENE_LIST = [g for g in GENE_LIST if g.upper() in graph_set]
    n_graph_genes = len(GRAPH_GENE_LIST)
    n_figs = int(np.ceil(n_graph_genes / GENES_PER_FIGURE)) if n_graph_genes else 0

    all_data = []
    for col_idx, cell_type in enumerate(CELL_TYPE_LIST):
        pb_dict = process_cell_type(cell_type)
        all_data.append(pb_dict)
        gc.collect()

    apply_bh_fdr(all_data)
    export_results_csv(all_data)

    for fig_idx in range(n_figs):
        start = fig_idx * GENES_PER_FIGURE
        end = min(start + GENES_PER_FIGURE, n_graph_genes)
        gene_batch = GRAPH_GENE_LIST[start:end]
        build_figure(gene_batch, fig_idx + 1, all_data, n_cells, fname, plt, 'dec')

if __name__ == "__main__":
    main()