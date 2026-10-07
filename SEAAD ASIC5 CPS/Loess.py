#[cite: 6]
#!/usr/bin/env python3
"""
SEA-AD Memory-Efficient Grid LOESS Trajectory + Spearman Statistics
===============================================================================
Memory-efficient & multiprocessed version.
Outputs a side-by-side grid of fully boxed LOESS trajectories for each cell type.
"""

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import issparse
from pathlib import Path
import sys, os, json
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ======================================================================
# CONFIGURATION
# ======================================================================
INPUT_FILE  = "SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
CONFIG_FILE = "config.json"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, CONFIG_FILE), "r") as _f:
    _cfg = json.load(_f)

_ct_raw = _cfg.get("cell_types", _cfg.get("cell_type", []))
CELL_TYPES       = [_ct_raw] if isinstance(_ct_raw, str) else list(_ct_raw)

SUBCLASS_COL     = _cfg["subclass_col"]
SUPERTYPE_COL    = _cfg["supertype_col"]
SUPERTYPE_FILTER = [s.strip() for s in _cfg.get("supertype_filter", []) if s.strip()]
GENE_LIST        = _cfg["gene_list"]
SEX_FILTER       = _cfg.get("sex_filter", "Both")
AGE_COL          = _cfg.get("age_col", None)

SUPERTYPE_MODE = len(SUPERTYPE_FILTER) > 0

DONOR_COL = "Donor ID"
CPS_COL   = "Continuous Pseudo-progression Score"
SEX_COL   = "Sex"

N_BOOT       = 1000
LOESS_FRAC   = 0.8
LOESS_SPAN   = 0.8
GRID_POINTS  = 200
N_WORKERS    = max(1, mp.cpu_count() - 1)

# ======================================================================
# FONT
# ======================================================================
def _get_font():
    local  = [os.path.join(SCRIPT_DIR, f) for f in os.listdir(SCRIPT_DIR)
               if "trebuc" in f.lower() and f.endswith(".ttf")]
    system = [p for p in fm.findSystemFonts() if "trebuc" in p.lower()]
    fonts  = local + system
    if fonts:
        try:
            fm.fontManager.addfont(fonts[0])
            return fm.FontProperties(fname=fonts[0]).get_name()
        except Exception:
            pass
    return "DejaVu Sans"

FONT_NAME = _get_font()

# ======================================================================
# 1. MEMORY-EFFICIENT LOAD
# ======================================================================
def load_data():
    fpath = os.path.join(SCRIPT_DIR, INPUT_FILE)
    if not Path(fpath).exists():
        sys.exit(f"ERROR: File not found: {fpath}")
    print(f"[1] Loading (backed): {fpath}")
    adata = ad.read_h5ad(fpath, backed="r")
    print(f"    Total cells: {adata.n_obs:,}")

    obs_sub_col = adata.obs[SUBCLASS_COL].astype(str)
    if SUPERTYPE_MODE:
        ct = CELL_TYPES[0]
        cell_mask = obs_sub_col.str.contains(ct, case=False).values
    else:
        pattern = "|".join(CELL_TYPES)
        cell_mask = obs_sub_col.str.contains(pattern, case=False).values

    n_kept = int(cell_mask.sum())
    if n_kept == 0:
        sys.exit(f"ERROR: No cells matching filter in '{SUBCLASS_COL}'.")
    print(f"    Cells matching filter: {n_kept:,}")

    var_names_upper = adata.var_names.str.upper()
    gene_indices = {}
    for gene in GENE_LIST:
        gu = gene.upper()
        match = np.where(var_names_upper == gu)[0]
        if len(match) == 0:
            for col in ["gene_symbol", "gene_name", "gene_ids", "feature_name"]:
                if col in adata.var.columns:
                    match = np.where(adata.var[col].astype(str).str.upper() == gu)[0]
                    if len(match) > 0:
                        break
        if len(match) > 0:
            gene_indices[gene] = int(match[0])
        else:
            print(f"    [WARNING] Gene '{gene}' not found — will be skipped.")

    if not gene_indices:
        sys.exit("ERROR: None of the requested genes were found.")

    print(f"    Extracting {len(gene_indices)} gene columns for {n_kept:,} cells...")
    cell_idx = np.where(cell_mask)[0]

    layer_name = None
    for ln in ["raw", "UMIs", "counts", "spliced"]:
        if ln in adata.layers:
            layer_name = ln
            break

    CHUNK = 50000
    gene_col_idx = sorted(gene_indices.values())
    gene_col_arr = np.array(gene_col_idx)
    n_cells = len(cell_idx)
    expr_matrix = np.zeros((n_cells, len(gene_col_idx)), dtype=np.float32)

    for start in range(0, n_cells, CHUNK):
        end = min(start + CHUNK, n_cells)
        rows = cell_idx[start:end]
        if layer_name:
            chunk = adata.layers[layer_name][rows]
        else:
            chunk = adata.X[rows]
        if issparse(chunk):
            chunk = chunk.toarray()
        else:
            chunk = np.asarray(chunk)
        expr_matrix[start:end, :] = chunk[:, gene_col_arr]

    idx_to_pos = {idx: pos for pos, idx in enumerate(gene_col_idx)}
    gene_expr = {}
    for gene, gidx in gene_indices.items():
        gene_expr[gene] = expr_matrix[:, idx_to_pos[gidx]]

    needed_cols = [SUBCLASS_COL, DONOR_COL]
    for c in [SUPERTYPE_COL, CPS_COL, SEX_COL, AGE_COL,
              "Number of UMIs", "n_counts", "nCount_RNA"]:
        if c and c in adata.obs.columns:
            needed_cols.append(c)
    needed_cols = list(dict.fromkeys(needed_cols))

    obs_df = adata.obs.iloc[cell_idx][needed_cols].copy()
    obs_df = obs_df.reset_index(drop=True)

    lib_col = None
    for try_col in ["Number of UMIs", "n_counts", "nCount_RNA"]:
        if try_col in obs_df.columns:
            lib_col = try_col
            break

    if lib_col:
        obs_df["_lib"] = pd.to_numeric(obs_df[lib_col], errors="coerce")
    else:
        print("    Computing library sizes from X (chunked)...")
        lib_sizes = np.zeros(n_cells, dtype=np.float64)
        for start in range(0, n_cells, CHUNK):
            end = min(start + CHUNK, n_cells)
            rows = cell_idx[start:end]
            chunk = adata.X[rows]
            if issparse(chunk):
                lib_sizes[start:end] = np.array(chunk.sum(axis=1)).flatten()
            else:
                lib_sizes[start:end] = np.asarray(chunk).sum(axis=1)
        obs_df["_lib"] = lib_sizes

    adata.file.close()
    del adata

    print(f"    Data extraction complete. Shape: {obs_df.shape}")
    return obs_df, gene_expr

# ======================================================================
# 2. PSEUDOBULK
# ======================================================================
def build_pseudobulk(obs_df, gene_counts_arr, line_label, is_supertype, sex_value=None):
    df = obs_df.copy()
    df["_raw"] = gene_counts_arr

    if is_supertype:
        mask = df[SUPERTYPE_COL].astype(str) == line_label
    else:
        mask = df[SUBCLASS_COL].astype(str).str.contains(line_label, case=False)
    df = df[mask]

    if len(df) == 0:
        return None

    if sex_value and SEX_COL in df.columns:
        df = df[df[SEX_COL].astype(str).str.lower() == sex_value.lower()]

    df = df.dropna(subset=[DONOR_COL, "_lib"])
    df = df[df["_lib"] > 0]

    if len(df) == 0:
        return None

    agg = df.groupby(DONOR_COL, observed=True).agg(
        sum_raw=("_raw", "sum"),
        sum_lib=("_lib", "sum"),
    ).reset_index()

    if CPS_COL in df.columns:
        first = df.groupby(DONOR_COL, observed=True)[CPS_COL].first().reset_index()
        first.columns = [DONOR_COL, "cps"]
        agg = agg.merge(first, on=DONOR_COL, how="left")

    if SEX_COL in df.columns:
        first_sex = df.groupby(DONOR_COL, observed=True)[SEX_COL].first().reset_index()
        first_sex.columns = [DONOR_COL, "sex"]
        agg = agg.merge(first_sex, on=DONOR_COL, how="left")

    if len(agg) == 0:
        return None

    agg["up10k"]    = (agg["sum_raw"] / agg["sum_lib"]) * 1e4
    agg["ln_up10k"] = np.log(agg["up10k"] + 1)
    agg["cps"]      = pd.to_numeric(agg.get("cps"), errors="coerce")

    return agg

# ======================================================================
# 3. PARALLELIZED LOESS BOOTSTRAP
# ======================================================================
def _loess_worker(args):
    x, y, indices_batch, span = args
    results = []
    grid = np.linspace(0, 1, GRID_POINTS)
    for idx in indices_batch:
        xs, ys = x[idx], y[idx]
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        try:
            fit = sm.nonparametric.lowess(ys, xs, frac=span, it=3, return_sorted=False)
            f = interpolate.interp1d(xs, fit, kind="linear", fill_value="extrapolate")
            results.append(f(grid))
        except Exception:
            continue
    return results

def bootstrap_loess_parallel(x, y, n_boot=N_BOOT, frac=LOESS_FRAC, span=LOESS_SPAN, n_workers=N_WORKERS):
    x, y = np.array(x), np.array(y)
    n_sub = int(len(x) * frac)
    rng = np.random.default_rng(42)

    all_indices = [rng.choice(len(x), n_sub, replace=False) for _ in range(n_boot)]
    actual_workers = min(n_workers, n_boot)
    batches = np.array_split(all_indices, actual_workers)
    worker_args = [(x, y, batch, span) for batch in batches]

    with mp.Pool(actual_workers) as pool:
        batch_results = pool.map(_loess_worker, worker_args)

    fits = []
    for br in batch_results:
        fits.extend(br)

    if not fits:
        grid = np.linspace(0, 1, GRID_POINTS)
        return grid, np.zeros(GRID_POINTS), np.zeros(GRID_POINTS)

    fits = np.array(fits)
    grid = np.linspace(0, 1, GRID_POINTS)
    mean_y = np.nanmean(fits, axis=0)
    se_y = np.nanstd(fits, axis=0, ddof=1)
    return grid, mean_y, se_y

# ======================================================================
# 4. PROCESS ONE LINE 
# ======================================================================
def process_one_line(args):
    obs_df, gene_arr, line_label, is_supertype, sex_value, gene_name = args
    display = f"{line_label} — {sex_value}" if sex_value else line_label

    pb = build_pseudobulk(obs_df, gene_arr, line_label, is_supertype, sex_value=sex_value)
    if pb is None:
        print(f"    [WARNING] No data for '{display}' — skipping.")
        return None, []

    df_clean = pb.dropna(subset=["cps", "ln_up10k"])
    n_donors = len(df_clean)
    print(f"    {display}: {n_donors} donors")

    sp_rows = []
    if n_donors >= 5:
        rho, pval = stats.spearmanr(df_clean["cps"], df_clean["ln_up10k"])
        sp_rows.append({
            "gene": gene_name, "label": display,
            "rho": float(rho), "pval": float(pval),
            "padj": np.nan, "n_donors": n_donors,
        })
    else:
        sp_rows.append({
            "gene": gene_name, "label": display,
            "rho": np.nan, "pval": np.nan,
            "padj": np.nan, "n_donors": n_donors,
        })

    if n_donors < 10:
        print(f"    [WARNING] Only {n_donors} donors for '{display}' — skipping LOESS.")
        return None, sp_rows

    print(f"    Running LOESS for '{display}' ({N_BOOT} bootstraps, {N_WORKERS} workers)...")
    grid, mean_y, se_y = bootstrap_loess_parallel(
        df_clean["cps"].values, df_clean["ln_up10k"].values
    )

    loess_dict = {
        "label":    display,
        "grid":     grid,
        "mean_y":   mean_y,
        "se_y":     se_y,
        "n_donors": n_donors,
    }
    return loess_dict, sp_rows

# ======================================================================
# 5. FDR CORRECTION
# ======================================================================
def apply_global_fdr(all_spearman_rows):
    valid_idx = [i for i, r in enumerate(all_spearman_rows) if np.isfinite(r["pval"])]
    valid_pvals = [all_spearman_rows[i]["pval"] for i in valid_idx]
    if valid_pvals:
        padj_vals = multipletests(valid_pvals, method="fdr_bh")[1]
        for j, i in enumerate(valid_idx):
            all_spearman_rows[i]["padj"] = float(padj_vals[j])
    return all_spearman_rows

# ======================================================================
# 6. PLOT - GRID LAYOUT WITH TEXT BOX
# ======================================================================
_PALETTE = [
    "#C0392B", "#E8735A", "#F0A500", "#2471A3", "#1E8449",
    "#D4820A", "#6C3483", "#117A65", "#784212", "#1A5276",
    "#2E86C1", "#A93226", "#1D8348", "#B7950B", "#6D4C41",
    "#4A235A", "#0E6655", "#B03A2E", "#2980B9", "#27AE60",
    "#D35400", "#7D6608", "#717D7E"
]

def plot_trajectories(loess_data, gene, out_dir, spearman_rows=None):
    stats_lookup = {}
    if spearman_rows:
        for r in spearman_rows:
            stats_lookup[r["label"]] = (r.get("rho", np.nan), r.get("padj", np.nan))
    
    plt.rcParams.update({
        "font.family":        FONT_NAME,
        "font.size":          12,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     12,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.linewidth":     1.0,  
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })
    
    n_panels = len(loess_data)
    if n_panels == 0:
        return
        
    fig, axes = plt.subplots(1, n_panels, figsize=(1.5 * n_panels, 3.8), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    all_y = []
    for d in loess_data:
        all_y.extend(d["mean_y"] + d["se_y"])
        all_y.extend(d["mean_y"] - d["se_y"])
    y_min, y_max = min(all_y), max(all_y) if all_y else (0, 1)
    y_range = y_max - y_min
    y_bottom = y_min - 0.05 * y_range
    y_top = y_max + 0.35 * y_range 
    
    for i, (d, ax) in enumerate(zip(loess_data, axes)):
        col   = _PALETTE[i % len(_PALETTE)]
        grid  = d["grid"]
        my    = d["mean_y"]
        se    = d["se_y"]
        label = d["label"]
        
        ax.fill_between(grid, my - se, my + se,
                        color=col, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(grid, my, color=col, linewidth=1.8,
                solid_capstyle="round", zorder=3)
        
        ax.set_title(label, pad=10, color="#111111")
        
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(y_bottom, y_top)
        
        if i == 0:
            ax.set_ylabel(f"{gene} Expression\nln(UP10K + 1)", labelpad=6)
            ax.tick_params(axis='y', which='both', left=True, labelleft=True)
        else:
            # Hide the y-axis ticks (dents) for panels after the first
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        rho, padj = stats_lookup.get(label, (np.nan, np.nan))
        if np.isfinite(rho) and np.isfinite(padj):
            if padj < 0.0001:
                p_str = "p < 0.0001"
            else:
                p_str = f"p = {padj:.4f}"
            rho_str = f"ρ = {rho:+.2f}"
            textstr = f"{rho_str}\n{p_str}"
        else:
            textstr = "n/a"
            
        props = dict(boxstyle='square,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5, linestyle='--')
        
        # Padded inward (x = 0.90) to prevent overlap with the right border
        ax.text(0.90, 0.93, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', multialignment='left', bbox=props, zorder=5)
        
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
            spine.set_visible(True)
    # Single centered x-axis label for all subplots
    fig.supxlabel("AD Pathology Progression (CPS)", fontsize=12, y=0.02)
                 
    plt.tight_layout(pad=0.5, w_pad=0.2, rect=[0, 0.04, 1, 0.98])
    
    out_path = os.path.join(out_dir, f"{gene}_SEA-AD_Trajectory.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor="white", edgecolor="none")
    plt.close()
    print(f"    Plot saved: {out_path}")

# ======================================================================
# 7. EXCEL
# ======================================================================
def save_excel(all_spearman_rows, gene, out_dir):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        df = pd.DataFrame(all_spearman_rows)
        csv_path = os.path.join(out_dir, f"{gene}_spearman.csv")
        df.to_csv(csv_path, index=False)
        print(f"    CSV fallback saved: {csv_path}")
        return

    wb = Workbook()
    wb.remove(wb.active)

    header_fill = PatternFill("solid", fgColor="4472C4")
    header_font = Font(name="Trebuchet MS", bold=True, size=11, color="FFFFFF")
    data_font   = Font(name="Trebuchet MS", size=10)
    row_border  = Border(bottom=Side(style="thin", color="D9D9D9"))

    df_all         = pd.DataFrame(all_spearman_rows)
    labels_ordered = list(dict.fromkeys(r["label"] for r in all_spearman_rows))
    COL_ORDER      = ["label", "gene", "rho", "pval", "padj", "n_donors"]

    for lbl in labels_ordered:
        sub = df_all[df_all["label"] == lbl][COL_ORDER].copy()
        safe_lbl = lbl.translate(str.maketrans("/\\?*[]:", "-------"))
        sheet_name = safe_lbl[:31]
        ws = wb.create_sheet(title=sheet_name)

        for ci, col_name in enumerate(COL_ORDER, 1):
            cell = ws.cell(row=1, column=ci, value=col_name)
            cell.font      = header_font
            cell.fill      = header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        for ri, (_, row) in enumerate(sub.iterrows(), 2):
            for ci, col_name in enumerate(COL_ORDER, 1):
                val  = row[col_name]
                cell = ws.cell(row=ri, column=ci)
                if isinstance(val, float) and np.isnan(val):
                    cell.value = None
                elif isinstance(val, (np.floating, float)):
                    cell.value         = float(val)
                    cell.number_format = "0.0000"
                elif isinstance(val, (np.integer, int)):
                    cell.value = int(val)
                else:
                    cell.value = str(val) if pd.notna(val) else None
                cell.font   = data_font
                cell.border = row_border

        for ci, col_name in enumerate(COL_ORDER, 1):
            max_len = max(
                len(col_name),
                sub[col_name].astype(str).str.len().max() if len(sub) > 0 else 0,
            )
            ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 3, 28)

        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes    = "A2"

    xlsx_path = os.path.join(out_dir, f"{gene}_spearman.xlsx")
    wb.save(xlsx_path)
    print(f"    Excel saved: {xlsx_path}")

# ======================================================================
# MAIN
# ======================================================================
def main():
    mode_str = "SUPERTYPE" if SUPERTYPE_MODE else "SUBCLASS"
    print("=" * 65)
    print(f"SEA-AD  LOESS + SPEARMAN  [{mode_str} MODE]")
    if SUPERTYPE_MODE:
        print(f"Subclass       : {CELL_TYPES[0]}")
        print(f"Supertypes     : {SUPERTYPE_FILTER}")
    else:
        print(f"Subclasses     : {CELL_TYPES}")
    print(f"Sex filter     : {SEX_FILTER}")
    print(f"Workers        : {N_WORKERS}")
    print(f"FDR            : BH across all {mode_str.lower()}s")
    print(f"Genes          : {GENE_LIST}")
    print("=" * 65)

    obs_df, gene_expr = load_data()

    if SUPERTYPE_MODE:
        if SUPERTYPE_COL not in obs_df.columns:
            sys.exit(f"ERROR: Column '{SUPERTYPE_COL}' not found.")
        available = sorted(obs_df[SUPERTYPE_COL].astype(str).unique())
        lines = [s for s in SUPERTYPE_FILTER if s in available]
        missing = [s for s in SUPERTYPE_FILTER if s not in available]
        if missing:
            print(f"    [WARNING] Supertypes not found: {missing}")
        if not lines:
            sys.exit("ERROR: None of the requested supertypes were found.")
    else:
        lines = CELL_TYPES

    if SEX_FILTER.lower() == "split":
        sex_groups = ["Male", "Female"]
    elif SEX_FILTER.lower() == "both":
        sex_groups = [None]
    else:
        sex_groups = [SEX_FILTER]

    for gene in GENE_LIST:
        print(f"\n{'─' * 60}\nGene: {gene}\n{'─' * 60}")

        if gene not in gene_expr:
            print(f"    [WARNING] '{gene}' was not found during loading — skipping.")
            continue

        gene_arr = gene_expr[gene]
        out_dir = os.path.join(SCRIPT_DIR, f"{gene}_results")
        os.makedirs(out_dir, exist_ok=True)

        tasks = []
        for line_label in lines:
            for sex_val in sex_groups:
                tasks.append((
                    obs_df, gene_arr, line_label,
                    SUPERTYPE_MODE, sex_val, gene,
                ))

        loess_data   = []
        all_spearman = []

        for task in tasks:
            loess_dict, sp_rows = process_one_line(task)
            if loess_dict is not None:
                loess_data.append(loess_dict)
            all_spearman.extend(sp_rows)

        if not loess_data:
            print(f"    [WARNING] No lines with enough donors for {gene} — skipping plot.")
            continue

        all_spearman = apply_global_fdr(all_spearman)

        for r in all_spearman:
            rho_s  = f"{r['rho']:+.4f}" if np.isfinite(r["rho"])  else "n/a"
            p_s    = f"{r['pval']:.3e}"  if np.isfinite(r["pval"]) else "n/a"
            padj_s = f"{r['padj']:.3e}"  if np.isfinite(r.get("padj", np.nan)) else "n/a"
            sig    = "  ← FDR < 0.05" if np.isfinite(r.get("padj", np.nan)) and r["padj"] < 0.05 else ""
            print(f"    Spearman [{r['label']:25s}] rho={rho_s}  p={p_s}  padj={padj_s}  n={r['n_donors']}{sig}")

        print(f"\n  Plotting {len(loess_data)} trajectory lines...")
        plot_trajectories(loess_data, gene, out_dir, spearman_rows=all_spearman)

        print(f"  Saving Excel workbook...")
        save_excel(all_spearman, gene, out_dir)

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)

if __name__ == "__main__":
    main()