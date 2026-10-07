import warnings
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from patsy import dmatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Random effects covariance is singular")
warnings.filterwarnings("ignore", message="The MLE may be on the boundary of the parameter space")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "Gene_Region_Grid_results")
os.makedirs(OUT_DIR, exist_ok=True)

GENES_PER_FIGURE = 3
N_COLS = 1
MAX_ROWS = 3

REGION_ORDER = ["HIP", "PCx", "TCx", "FWM"]
GROUP_ORDER = ["Control", "AD"]
REGION_LABELS = {
    "HIP": "Hippocampus",
    "PCx": "Parietal Cortex",
    "TCx": "Temporal Cortex",
    "FWM": "Forebrain White Matter",
}
REGION_LABELS_SHORT = {
    "HIP": "Hippocampus",
    "PCx": "Parietal Cx",
    "TCx": "Temporal Cx",
    "FWM": "Forebrain WM",
}

_PALETTE = [
    {"color": "#1F77B4", "marker": "o"},
    {"color": "#FF7F0E", "marker": "^"},
    {"color": "#2CA02C", "marker": "s"},
    {"color": "#D62728", "marker": "v"},
]

def _darken(hex_color, factor=0.45):
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

_sys_treb = [p for p in fm.findSystemFonts() if "trebuc" in p.lower()]
_local_treb = [os.path.join(SCRIPT_DIR, f) for f in os.listdir(SCRIPT_DIR)
               if "trebuc" in f.lower() and f.lower().endswith(".ttf")]
_treb = _local_treb + _sys_treb
if _treb:
    for _tp in _treb:
        try: fm.fontManager.addfont(_tp)
        except Exception: pass
    FONT = fm.FontProperties(fname=_treb[0]).get_name()
else:
    FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.usetex": False,
})

RNG = np.random.default_rng(42)

def discover_gene_dirs():
    genes = {}
    for name in sorted(os.listdir(SCRIPT_DIR)):
        d = os.path.join(SCRIPT_DIR, name)
        if not os.path.isdir(d):
            continue
        needed = ["Expression.csv", "Columns.csv", "DonorInformation.csv"]
        if all(os.path.exists(os.path.join(d, f)) for f in needed):
            genes[name] = d
    return genes

def build_merged(gene, gene_dir):
    cols = pd.read_csv(os.path.join(gene_dir, "Columns.csv"))
    donor = pd.read_csv(os.path.join(gene_dir, "DonorInformation.csv"))
    expr = pd.read_csv(os.path.join(gene_dir, "Expression.csv"), header=None)
    vals = expr.iloc[0, 1:].astype(float).values
    if len(vals) != len(cols):
        raise ValueError(f"{gene} expression columns length mismatch")
    cols = cols.copy()
    cols["expression"] = vals
    merged = cols.merge(
        donor[["donor_id", "act_demented", "ever_tbi_w_loc", "dsm_iv_clinical_diagnosis"]],
        on="donor_id",
        how="inner",
    )
    
    merged = merged[merged["ever_tbi_w_loc"] == "N"]
    
    is_ad = merged["dsm_iv_clinical_diagnosis"] == "Alzheimer's Disease Type"
    is_ctrl = (merged["act_demented"] == "No Dementia") & (merged["dsm_iv_clinical_diagnosis"] == "No Dementia")
    
    merged = merged[is_ad | is_ctrl].copy()
    merged["group"] = np.where(is_ad[merged.index], "AD", "Control")
    return merged

def load_gene_data(merged):
    data = {}
    for region in REGION_ORDER:
        sub = merged[merged["structure_abbreviation"] == region]
        ad = sub[sub["group"] == "AD"]["expression"].dropna().values.astype(float)
        ctrl = sub[sub["group"] == "Control"]["expression"].dropna().values.astype(float)
        data[region] = {"dem": ad, "nodem": ctrl}
    return data

def build_long(gene, merged):
    long = merged[["donor_id", "group", "structure_abbreviation", "expression"]].copy()
    long.columns = ["donor_id", "group", "region", "value"]
    long = long.dropna(subset=["value"])
    long["donor_id"] = long["donor_id"].astype(str)
    long["region"] = pd.Categorical(long["region"], categories=REGION_ORDER, ordered=True)
    long["group"] = pd.Categorical(long["group"], categories=GROUP_ORDER, ordered=True)
    return long

def sidak_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(pvals)
    adj = np.full(len(pvals), np.nan)
    m = mask.sum()
    if m == 0:
        return adj
    adj[mask] = np.minimum(1.0 - (1.0 - pvals[mask]) ** m, 1.0)
    return adj

def fit_one_gene(gene, dat):
    if dat is None or dat.empty:
        return None, "No data parsed"
    counts = dat.groupby(["region", "group"], observed=True).size().unstack(fill_value=0)
    present_regions = [r for r in REGION_ORDER if r in dat["region"].cat.categories and (dat["region"] == r).any()]
    fit = None
    model_name = None
    for method in ["lbfgs", "powell", "nm"]:
        try:
            model = smf.mixedlm("value ~ C(group) * C(region)", data=dat, groups=dat["donor_id"])
            fit = model.fit(reml=False, method=method, maxiter=500, disp=False)
            model_name = f"MixedLM ({method})"
            break
        except Exception:
            continue
    if fit is None:
        try:
            model = smf.ols("value ~ C(group) * C(region) + C(donor_id)", data=dat)
            fit = model.fit()
            model_name = "OLS_subject_fixed_effect"
        except Exception as e:
            return None, f"All models failed {e}"
    design_info = fit.model.data.design_info
    cov = fit.cov_params()
    df_resid = getattr(fit, "df_resid", np.inf)
    rows, raw_ps = [], []
    for region in present_regions:
        ctrl_df = pd.DataFrame({"group": ["Control"], "region": [region], "donor_id": [dat["donor_id"].iloc[0]]})
        ad_df = pd.DataFrame({"group": ["AD"], "region": [region], "donor_id": [dat["donor_id"].iloc[0]]})
        ctrl_pred = float(fit.predict(ctrl_df)[0])
        ad_pred = float(fit.predict(ad_df)[0])
        ls_diff = ctrl_pred - ad_pred
        ctrl_dm = dmatrix(design_info, ctrl_df, return_type="dataframe")
        ad_dm = dmatrix(design_info, ad_df, return_type="dataframe")
        contrast = (ctrl_dm.values - ad_dm.values).flatten()
        cov_use = cov.values
        if cov_use.shape[0] != len(contrast):
            k = min(cov_use.shape[0], len(contrast))
            contrast = contrast[:k]
            cov_use = cov_use[:k, :k]
        se = float(np.sqrt(max(contrast @ cov_use @ contrast, 0)))
        if se == 0 or np.isnan(se):
            raw_p = ci_low = ci_high = t_val = np.nan
        else:
            t_val = ls_diff / se
            if np.isinf(df_resid):
                raw_p = float(2 * stats.norm.sf(abs(t_val)))
                q = stats.norm.ppf(0.975)
            else:
                raw_p = float(2 * stats.t.sf(abs(t_val), df=df_resid))
                q = stats.t.ppf(0.975, df=df_resid)
            ci_low, ci_high = ls_diff - q * se, ls_diff + q * se
        raw_ps.append(raw_p)
        rows.append({
            "gene": gene, "region": region,
            "n_control": int(counts.loc[region, "Control"]) if "Control" in counts.columns else 0,
            "n_ad": int(counts.loc[region, "AD"]) if "AD" in counts.columns else 0,
            "ls_mean_control": ctrl_pred, "ls_mean_ad": ad_pred, "ls_diff": ls_diff,
            "se": se, "t": t_val, "ci_low": ci_low, "ci_high": ci_high,
            "model": model_name, "raw_p": raw_p,
        })
    for row, p in zip(rows, sidak_adjust(raw_ps)):
        row["sidak_p_within_gene"] = p
    return rows, None

def run_mixedmodel_stats(genes, long_dfs):
    all_rows, errors = [], []
    for gene in genes:
        rows, err = fit_one_gene(gene, long_dfs[gene])
        if err:
            errors.append({"gene": gene, "error": err})
            continue
        all_rows.extend(rows)
    raw_p_list = [r["raw_p"] for r in all_rows]
    valid_idx = [i for i, p in enumerate(raw_p_list) if not pd.isna(p)]
    bh_q = [np.nan] * len(all_rows)
    if valid_idx:
        pvals_arr = np.array([raw_p_list[i] for i in valid_idx])
        try:
            qvals = false_discovery_control(pvals_arr, method="bh")
        except AttributeError:
            n = len(pvals_arr)
            rank = np.argsort(pvals_arr)
            qvals = np.empty(n)
            qvals[rank] = pvals_arr[rank] * n / (np.arange(n) + 1)
            qvals = np.minimum.accumulate(qvals[::-1])[::-1]
            qvals = np.minimum(qvals, 1.0)
        for i, q in zip(valid_idx, qvals):
            bh_q[i] = q
    for row, q in zip(all_rows, bh_q):
        row["bh_q_global"] = q
    adj_qs = {(r["gene"], r["region"]): r["bh_q_global"] for r in all_rows}
    raw_ps = {(r["gene"], r["region"]): r["raw_p"] for r in all_rows}
    return adj_qs, raw_ps, all_rows, errors

def _fmt_p(q):
    if pd.isna(q):
        return "p=NA"
    if q < 0.001:
        exp = int(np.floor(np.log10(abs(q))))
        coef = q / (10 ** exp)
        return f"$p={coef:.2f}\\!\\times\\!10^{{{exp}}}$"
    return f"p={q:.4f}"

WITHIN_GAP = 0.70
BETWEEN_GAP = 1.50

def draw_panel(ax, gene, regions, data, adj_qs, panel_letter):
    n = len(regions)
    ctrs = np.arange(n, dtype=float) * BETWEEN_GAP
    half = WITHIN_GAP / 2.0
    x_nd = ctrs - half
    x_d = ctrs + half
    all_vals = []
    for region in regions:
        all_vals.extend(data[gene][region]["dem"].tolist())
        all_vals.extend(data[gene][region]["nodem"].tolist())
    global_max = max(all_vals) if all_vals else 1.0
    step = max(global_max * 0.10, 0.15)
    y_bracket = global_max + step
    for ri, region in enumerate(regions):
        color = _PALETTE[ri]["color"]
        mark = _PALETTE[ri]["marker"]
        med_c = _darken(color)
        v_nd = data[gene][region]["nodem"]
        v_d = data[gene][region]["dem"]
        adj_q = adj_qs.get((gene, region), np.nan)
        for xpos, vals in [(x_nd[ri], v_nd), (x_d[ri], v_d)]:
            if len(vals) == 0:
                continue
            bp = ax.boxplot(
                vals,
                positions=[xpos],
                widths=WITHIN_GAP * 0.38,
                patch_artist=True,
                medianprops=dict(color=med_c, linewidth=3.2, solid_capstyle="butt", zorder=7),
                whiskerprops=dict(color=color, linewidth=1.4),
                capprops=dict(color=color, linewidth=1.4),
                flierprops=dict(marker="", linewidth=0),
                manage_ticks=False,
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(color)
            bp["boxes"][0].set_linewidth(1.4)
            jit = RNG.normal(0, 0.040, len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jit, vals,
                c=color, marker=mark,
                edgecolors="black", linewidths=0.5,
                s=40, zorder=4, alpha=0.88,
            )
        is_sig = (not pd.isna(adj_q)) and adj_q < 0.05
        ax.plot(
            [x_nd[ri], x_nd[ri], x_d[ri], x_d[ri]],
            [y_bracket - step * 0.15, y_bracket, y_bracket, y_bracket - step * 0.15],
            color="black", linewidth=2.4 if is_sig else 1.2, clip_on=False,
        )
        ax.text(ctrs[ri], y_bracket + step * 0.15, _fmt_p(adj_q), ha="center", va="bottom", fontsize=15)
        ax.text(ctrs[ri], y_bracket + step * 1.15, REGION_LABELS_SHORT[region], ha="center", va="bottom", fontsize=16, fontweight="bold", color="black")
    tick_pos = [x for ri in range(n) for x in (x_nd[ri], x_d[ri])]
    tick_lbl = [lbl for _ in range(n) for lbl in ["−Dem", "+Dem"]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=18, fontweight="normal")
    ax.set_xlim(x_nd[0] - BETWEEN_GAP * 0.40, x_d[-1] + BETWEEN_GAP * 0.40)
    ax.set_ylim(bottom=0, top=y_bracket + step * 2.8)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Dementia Status", fontsize=19, labelpad=14, fontweight="bold")
    ax.set_ylabel("mRNA (Log₂ Intensity)", fontsize=20, labelpad=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(gene, fontsize=21, fontweight="bold", fontstyle="italic", pad=14)
    ax.text(-0.09, 1.12, panel_letter, transform=ax.transAxes, fontsize=30, fontweight="black", va="top", ha="left")

def build_figures(genes, data, adj_qs):
    n_genes = len(genes)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    subplot_w = 9.0
    subplot_h = 5.5
    
    fig, axes = plt.subplots(
        n_genes, 1,
        figsize=(subplot_w, n_genes * subplot_h),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.52, top=0.96, bottom=0.04, left=0.15, right=0.95)
    
    for gi, gene in enumerate(genes):
        draw_panel(axes[gi][0], gene, REGION_ORDER, data, adj_qs, letters[gi])
        
    out_path = os.path.join(OUT_DIR, "Gene_Region_Grid_vertical.png")
    out_path_tiff = os.path.join(OUT_DIR, "Gene_Region_Grid_vertical.tiff")
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = 0.1
    from matplotlib.transforms import Bbox
    padded_bbox = Bbox.from_extents(
        tight_bbox.x0 - pad, tight_bbox.y0 - pad,
        tight_bbox.x1 + pad, tight_bbox.y1 + pad,
    )
    fig.savefig(out_path, dpi=300, bbox_inches=padded_bbox, facecolor="white")
    fig.savefig(out_path_tiff, dpi=300, bbox_inches=padded_bbox, facecolor="white", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {out_path} and {out_path_tiff} ({n_genes} genes)")

def export_results_csv(all_rows, errors):
    df_out = pd.DataFrame(all_rows)
    if not df_out.empty:
        df_out["significant (BH q<0.05)"] = np.where(
            df_out["bh_q_global"].notna() & (df_out["bh_q_global"] < 0.05), "Yes", "No"
        )
        df_out = df_out.rename(columns={
            "n_control": "N (No Dementia)", "n_ad": "N (AD)",
            "ls_mean_control": "LS Mean (No Dementia)", "ls_mean_ad": "LS Mean (AD)",
            "ls_diff": "LS Mean Diff (No Dementia - AD)", "se": "SE (Diff)", "t": "t statistic",
            "ci_low": "95% CI Low", "ci_high": "95% CI High", "model": "Model",
            "raw_p": "p (raw)", "sidak_p_within_gene": "p (Sidak within gene)",
            "bh_q_global": "p (BH global)",
        })
    csv_path = os.path.join(OUT_DIR, "Gene_Region_Grid_all_results.csv")
    try:
        df_out.to_csv(csv_path, index=False)
        print(f"CSV saved {csv_path}")
    except PermissionError:
        pass
    if errors:
        err_path = os.path.join(OUT_DIR, "Gene_Region_Grid_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors CSV saved {err_path}")

def main():
    gene_dirs = discover_gene_dirs()
    genes = ["ASIC1", "ASIC2", "ASIC5"]
    genes = [g for g in genes if g in gene_dirs]
    merged = {gene: build_merged(gene, gene_dirs[gene]) for gene in genes}
    data = {gene: load_gene_data(merged[gene]) for gene in genes}
    long_dfs = {gene: build_long(gene, merged[gene]) for gene in genes}
    adj_qs, raw_ps, all_rows, errors = run_mixedmodel_stats(genes, long_dfs)
    export_results_csv(all_rows, errors)
    build_figures(genes, data, adj_qs)

if __name__ == "__main__":
    main()