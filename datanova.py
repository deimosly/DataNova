# datanova.py
"""DataNova: Gaia DR3 stellar classification (Streamlit)

Streamlit app that can:
- Fetch Gaia DR3 samples (vari_classifier_result + gaia_source + astrophysical_parameters)
- Load local CSVs
- Optionally enrich with gaiadr3.vari_summary and/or SOS period tables (rrlyrae/cepheid/lpv)
- Preprocess (dereddening, RUWE quality filter, abs mag from parallax or Bailer-Jones distances)
- Compute derived features (BC placeholder, luminosity proxy)
- Train a RandomForest classifier with optional SMOTE/ADASYN (if imblearn available)
- Show correlation analysis and publication-style HR diagrams
- Export filtered/original CSVs + metadata

Notes:
- Streamlit reruns the script on every interaction. This app uses caching + forms + session_state
  to avoid unnecessary refetching/retraining and to keep the UI responsive.
"""

from __future__ import annotations

# =========================
# Imports & dependency checks
# =========================
import datetime as dt
import glob
import io
import zipfile
import pickle
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import importlib
import importlib.metadata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer

# Optional extras
IMBLEARN_AVAILABLE = importlib.util.find_spec("imblearn") is not None
OPTUNA_AVAILABLE = importlib.util.find_spec("optuna") is not None

if IMBLEARN_AVAILABLE:
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

# Optional Gaia query library
try:
    from astroquery.gaia import Gaia  # type: ignore
    GAIA_AVAILABLE = True
except Exception:
    Gaia = None
    GAIA_AVAILABLE = False

REQUIRED_MODULES = ["pandas", "numpy", "matplotlib", "streamlit", "seaborn", "sklearn"]
_missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
if _missing:
    # Must be called before any other Streamlit calls in this run
    st.set_page_config(layout="wide", page_title="DataNova")
    st.error(f"Missing required Python modules: {', '.join(_missing)}.\n\nInstall them and restart the app.")
    st.stop()

# Streamlit page config (must happen early)
st.set_page_config(layout="wide", page_title="DataNova")

# Reproducibility
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)

# =========================
# Utilities
# =========================
def safe_json_default(o):
    """json.dump default handler for numpy and datetimes."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (dt.datetime,)):
        return o.isoformat() + "Z"
    return str(o)

def convert_fig_to_bytes(fig: plt.Figure, dpi: int = 250) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes (for Streamlit download buttons)."""
    return df.to_csv(index=False).encode("utf-8")

def make_zip_bytes(file_paths: List[Path], arc_prefix: str = "") -> bytes:
    """Create an in-memory zip bundle from a list of file paths."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            try:
                p = Path(p)
                if not p.exists() or not p.is_file():
                    continue
                arcname = f"{arc_prefix}{p.name}" if arc_prefix else p.name
                zf.write(str(p), arcname=arcname)
            except Exception:
                continue
    buf.seek(0)
    return buf.read()

def extract_estimator(model):
    """Return the underlying classifier estimator from a (imblearn/sklearn) Pipeline if possible."""
    try:
        if hasattr(model, "named_steps"):
            return model.named_steps.get("clf", model)
        return model
    except Exception:
        return model

@st.cache_data(show_spinner=False)
def compute_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value report sorted by percent missing."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "missing_pct", "missing_n", "dtype"])
    miss_frac = df.isna().mean()
    miss_n = df.isna().sum()
    dtypes = df.dtypes.astype(str)
    out = pd.DataFrame({
        "column": miss_frac.index,
        "missing_pct": (miss_frac.values * 100.0),
        "missing_n": miss_n.values,
        "dtype": dtypes.values
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return out

def robust_clip(series: pd.Series, lo=0.5, hi=99.5) -> pd.Series:
    """Clip extreme outliers for plotting without mutating original data."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    a, b = np.percentile(s, [lo, hi])
    return s.clip(a, b)

def get_env_info() -> Dict:
    pkgs = {}
    for dist in ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "streamlit", "astroquery", "imblearn", "optuna"]:
        try:
            pkgs[dist] = importlib.metadata.version(dist)
        except Exception:
            pkgs[dist] = None

    git_sha = None
    try:
        if Path(".git").exists():
            git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git_sha = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "git_sha": git_sha,
        "gaia_available": GAIA_AVAILABLE,
        "imblearn_available": IMBLEARN_AVAILABLE,
        "optuna_available": OPTUNA_AVAILABLE,
    }

def is_secondary_artifact(path: Path) -> bool:
    name = path.name.lower()
    bad_tokens = ["correlation", "feature_scores", "filtered", "original", "metadata", "hr_diagrams", "confusion_matrix"]
    return any(tok in name for tok in bad_tokens)

def find_latest_sample(folder: Path) -> Optional[Path]:
    """Return newest gaia_sample*.csv in folder, skipping secondary artifacts."""
    pattern = str(folder / "gaia_sample*.csv")
    candidates = [Path(p) for p in glob.glob(pattern) if not is_secondary_artifact(Path(p))]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def list_datasets(folder: Path) -> List[Path]:
    """List candidate datasets (gaia_sample*.csv and local_*.csv) excluding artifacts."""
    out: List[Path] = []
    for pat in ["gaia_sample*.csv", "local_*.csv"]:
        for p in glob.glob(str(folder / pat)):
            pp = Path(p)
            if not is_secondary_artifact(pp):
                out.append(pp)
    out = sorted(out, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return out

def set_current_csv(path: Optional[Path]):
    st.session_state["current_csv_path"] = str(path) if path else None
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1

def get_current_csv_path() -> Optional[Path]:
    p = st.session_state.get("current_csv_path")
    return Path(p) if p else None

def dataset_id(path: Optional[Path]) -> str:
    return str(path.resolve()) if path else "__none__"

def append_research_log(output_folder: Path, title: str, payload: Dict):
    """Append a single, structured entry to research_log.md (only when called)."""
    log_file = output_folder / "research_log.md"
    if not log_file.exists():
        log_file.write_text("# DataNova Research Log\n\n", encoding="utf-8")
    ts = dt.datetime.utcnow().isoformat() + "Z"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"## {title} — {ts}\n\n")
        f.write("```json\n")
        f.write(json.dumps(payload, indent=2, ensure_ascii=False, default=safe_json_default))
        f.write("\n```\n\n")

# =========================
# Cached IO / computations
# =========================
@st.cache_data(show_spinner=False)
def read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    # mtime is included so edits/overwrites invalidate cache
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def read_bailer_jones_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def compute_corr_cached(df_num: pd.DataFrame) -> pd.DataFrame:
    return df_num.corr(method="pearson")

# =========================
# Gaia fetch helpers
# =========================
def gaia_launch_job(query: str):
    if not GAIA_AVAILABLE or Gaia is None:
        raise RuntimeError("astroquery.gaia is not available")
    return Gaia.launch_job(query)

def fetch_gaia_sample(sample_size: int) -> Tuple[pd.DataFrame, str]:
    query = f"""
SELECT TOP {int(sample_size)}
    v.source_id,
    g.ra, g.dec,
    g.phot_g_mean_mag,
    g.phot_bp_mean_mag,
    g.phot_rp_mean_mag,
    g.parallax, g.parallax_error,
    g.teff_gspphot,
    ap.lum_flame,
    ap.radius_flame,
    ap.ag_gspphot,
    ap.ebpminrp_gspphot,
    v.best_class_name,
    g.ruwe
FROM gaiadr3.vari_classifier_result AS v
JOIN gaiadr3.gaia_source AS g
    ON v.source_id = g.source_id
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON v.source_id = ap.source_id
WHERE v.best_class_name IS NOT NULL
""".strip()
    job = gaia_launch_job(query)
    df = job.get_results().to_pandas()
    return df, query

def _chunked(iterable: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def fetch_vari_summary(source_ids: List[int], chunk_size: int = 2000) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for chunk in _chunked(source_ids, chunk_size):
        id_list_str = ",".join(str(int(x)) for x in chunk)
        q = f"SELECT * FROM gaiadr3.vari_summary WHERE source_id IN ({id_list_str})"
        job = gaia_launch_job(q)
        parts.append(job.get_results().to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def fetch_sos_table(table: str, source_ids: List[int], chunk_size: int = 2000) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for chunk in _chunked(source_ids, chunk_size):
        id_list_str = ",".join(str(int(x)) for x in chunk)
        q = f"SELECT * FROM gaiadr3.{table} WHERE source_id IN ({id_list_str})"
        job = gaia_launch_job(q)
        parts.append(job.get_results().to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# =========================
# Preprocessing / features
# =========================
def bolometric_correction_placeholder(teff: float) -> float:
    """Placeholder BC curve. Replace with a calibrated BC table/polynomial for publication."""
    if pd.isna(teff):
        return 0.0
    x = (float(teff) - 5800.0) / 1000.0
    return -0.08 - 0.35 * x + 0.06 * x * x

def preprocess_dataset(
    df_raw: pd.DataFrame,
    *,
    apply_deredden: bool,
    apply_ruwe: bool,
    ruwe_threshold: float,
    dist_method: str,
    bj_df: Optional[pd.DataFrame],
    parallax_snr_cut: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[int, str], List[str], bool]:
    """Return df_clean + class mapping + feature candidates + deredden_applied flag."""
    df = df_raw.copy()

    # Ensure key columns exist
    for c in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "parallax", "parallax_error", "teff_gspphot"]:
        if c not in df.columns:
            df[c] = np.nan

    # Unify astrophysical parameter names to internal keys
    ap_map = {
        "lum_val": "lum_flame",
        "radius_val": "radius_flame",
        "a_g_val": "ag_gspphot",
        "e_bp_min_rp_val": "ebpminrp_gspphot",
    }
    for internal, gaia_col in ap_map.items():
        if gaia_col in df.columns:
            df[internal] = df[gaia_col]
        elif internal not in df.columns:
            df[internal] = np.nan

    # Color index (raw)
    df["bp_rp_raw"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

    # Dereddening (optional)
    dered_applied = False
    if apply_deredden:
        # phot_g_mean_mag_corr = G - A_G
        if "a_g_val" in df.columns and df["a_g_val"].notna().any():
            df["phot_g_mean_mag_corr"] = df["phot_g_mean_mag"] - df["a_g_val"]
            dered_applied = True
        elif "ag_gspphot" in df.columns and df["ag_gspphot"].notna().any():
            df["phot_g_mean_mag_corr"] = df["phot_g_mean_mag"] - df["ag_gspphot"]
            dered_applied = True
        else:
            df["phot_g_mean_mag_corr"] = df["phot_g_mean_mag"]

        # bp_rp = (BP-RP) - E(BP-RP)
        if "e_bp_min_rp_val" in df.columns and df["e_bp_min_rp_val"].notna().any():
            df["bp_rp"] = df["bp_rp_raw"] - df["e_bp_min_rp_val"]
            dered_applied = True
        elif "ebpminrp_gspphot" in df.columns and df["ebpminrp_gspphot"].notna().any():
            df["bp_rp"] = df["bp_rp_raw"] - df["ebpminrp_gspphot"]
            dered_applied = True
        else:
            df["bp_rp"] = df["bp_rp_raw"]
    else:
        df["phot_g_mean_mag_corr"] = df["phot_g_mean_mag"]
        df["bp_rp"] = df["bp_rp_raw"]

    # Absolute magnitude: prefer Bailer-Jones if provided & selected, else parallax inversion
    df["abs_mag_g"] = np.nan
    df["abs_mag_g_sigma"] = np.nan

    if dist_method.startswith("bailerjones") and bj_df is not None and not bj_df.empty:
        dist_col_candidates = ["r_est", "r_med_geo", "r_med_photogeo", "r_med", "r"]
        dist_col = next((c for c in dist_col_candidates if c in bj_df.columns), None)
        if "source_id" in bj_df.columns and dist_col:
            bj_small = bj_df[["source_id", dist_col]].rename(columns={dist_col: "bj_distance_pc"}).copy()
            df = df.merge(bj_small, how="left", on="source_id")
            mask_bj = (
                df["bj_distance_pc"].notna()
                & (df["bj_distance_pc"] > 0)
                & df["phot_g_mean_mag_corr"].notna()
            )
            df.loc[mask_bj, "abs_mag_g"] = df.loc[mask_bj, "phot_g_mean_mag_corr"] - 5 * np.log10(df.loc[mask_bj, "bj_distance_pc"]) + 5
        # else: silently fall back to parallax

    # Parallax inversion fill (vectorized)
    plx = df["parallax"].to_numpy(dtype=float, copy=False)
    plx_err = df["parallax_error"].to_numpy(dtype=float, copy=False)
    Gcorr = df["phot_g_mean_mag_corr"].to_numpy(dtype=float, copy=False)

    mask_par = (
        np.isfinite(plx) & (plx > 0)
        & np.isfinite(Gcorr)
        & np.isfinite(plx_err) & (plx_err > 0)
        & ((plx / plx_err) >= float(parallax_snr_cut))
    )
    # compute only where abs_mag_g is still NaN
    abs_mag = np.full_like(plx, np.nan, dtype=float)
    sigma_M = np.full_like(plx, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_pc = 1000.0 / plx
        abs_mag[mask_par] = Gcorr[mask_par] - 5.0 * np.log10(d_pc[mask_par]) + 5.0
        sigma_M[mask_par] = (5.0 / np.log(10.0)) * (plx_err[mask_par] / plx[mask_par])

    need_fill = df["abs_mag_g"].isna().to_numpy()
    fill_mask = need_fill & mask_par
    df.loc[fill_mask, "abs_mag_g"] = abs_mag[fill_mask]
    df.loc[fill_mask, "abs_mag_g_sigma"] = sigma_M[fill_mask]

    # Drop rows missing essentials
    essential_cols = ["bp_rp", "abs_mag_g", "teff_gspphot"]
    df_clean = df.dropna(subset=essential_cols).copy()

    # Encode classes
    if "best_class_name" in df_clean.columns:
        df_clean["class_label"] = df_clean["best_class_name"].astype("category").cat.codes
        class_mapping = dict(enumerate(df_clean["best_class_name"].astype("category").cat.categories))
    else:
        df_clean["class_label"] = pd.Series(dtype=int)
        class_mapping = {}

    # RUWE filtering (optional)
    ruwe_col_candidates = ["ruwe", "renormalised_unit_weight_error"]
    ruwe_col = next((c for c in ruwe_col_candidates if c in df_clean.columns), None)
    if apply_ruwe and ruwe_col:
        df_clean = df_clean[df_clean[ruwe_col].fillna(9999.0) < float(ruwe_threshold)].copy()

    # Derived features (BC placeholder + luminosity proxy)
    M_bol_sun = 4.74
    df_clean["BC"] = df_clean["teff_gspphot"].apply(bolometric_correction_placeholder)
    df_clean["M_bol"] = df_clean["abs_mag_g"] + df_clean["BC"]
    df_clean["lum_ratio"] = 10 ** ((M_bol_sun - df_clean["M_bol"]) / 2.5)
    df_clean["teff_lum_ratio"] = df_clean["teff_gspphot"] * df_clean["lum_ratio"]

    # Unify a few AP columns if present
    for src in ["lum_flame", "lum_val", "luminosity"]:
        if src in df_clean.columns:
            df_clean["lum_val"] = df_clean[src]
            break
    for src in ["radius_flame", "radius_gspphot", "radius_val"]:
        if src in df_clean.columns:
            df_clean["radius_val"] = df_clean[src]
            break

    if "a_g_val" not in df_clean.columns:
        df_clean["a_g_val"] = np.nan
    if "e_bp_min_rp_val" not in df_clean.columns:
        df_clean["e_bp_min_rp_val"] = np.nan

    # Candidate feature set
    feature_candidates = [
        "bp_rp", "abs_mag_g", "teff_gspphot",
        "lum_val", "radius_val", "a_g_val", "e_bp_min_rp_val",
        "M_bol", "lum_ratio", "teff_lum_ratio",
    ]
    # Variability summary canonical fields (if present)
    for v in ["var_median_mag_g_fov", "var_median_mag_bp", "var_median_mag_rp", "var_amplitude_g", "var_n_obs_g"]:
        if v in df_clean.columns:
            feature_candidates.append(v)
    # SOS periods (if present)
    for c in df_clean.columns:
        if c.startswith("period_from_"):
            feature_candidates.append(c)

    return df_clean, class_mapping, feature_candidates, dered_applied

# =========================
# Plotting
# =========================
# =========================
# Researcher-friendly plots (optional, lightweight)
# =========================
def plot_class_distribution(df: pd.DataFrame, class_col: str = "best_class_name", top_n: int = 25):
    if df is None or df.empty or class_col not in df.columns:
        return None
    vc = df[class_col].astype(str).value_counts().head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(vc) + 1.5)))
    ax.barh(vc.index, vc.values)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {min(top_n, len(vc))} classes by count")
    plt.tight_layout()
    return fig

def plot_cmd_density(df: pd.DataFrame, x_col: str = "bp_rp", y_col: str = "abs_mag_g", gridsize: int = 220):
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    x = robust_clip(df[x_col], 0.5, 99.5)
    y = robust_clip(df[y_col], 0.5, 99.5)
    if x.empty or y.empty:
        return None
    # align lengths (clip returns dropna)
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(d["x"], d["y"], gridsize=gridsize, mincnt=1)
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP (mag)")
    ax.set_ylabel("$M_G$ (mag)")
    ax.set_title("CMD density (hexbin)")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Counts")
    plt.tight_layout()
    return fig

def plot_sky_map(df: pd.DataFrame, ra_col: str = "ra", dec_col: str = "dec", max_points: int = 25000):
    if df is None or df.empty or ra_col not in df.columns or dec_col not in df.columns:
        return None
    d = df[[ra_col, dec_col]].copy()
    d[ra_col] = pd.to_numeric(d[ra_col], errors="coerce")
    d[dec_col] = pd.to_numeric(d[dec_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return None
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=42)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(d[ra_col], d[dec_col], s=2, alpha=0.35, edgecolors="none")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Sky distribution (sampled)")
    plt.tight_layout()
    return fig

def plot_diagnostic_hist(df: pd.DataFrame, series: pd.Series, title: str, xlabel: str):
    s = robust_clip(series, 0.5, 99.5)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(s.values, bins=60)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

def plot_feature_importance_bar(estimator, feature_names: List[str], top_n: int = 25):
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None
    imp = np.array(estimator.feature_importances_, dtype=float)
    if imp.size != len(feature_names):
        # best-effort alignment
        n = min(len(feature_names), imp.size)
        imp = imp[:n]
        feature_names = feature_names[:n]
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(idx) + 1.5)))
    ax.barh([feature_names[i] for i in idx], imp[idx])
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {min(top_n, len(idx))} feature importances")
    plt.tight_layout()
    return fig

def plot_hr_diagrams(
    df_plot: pd.DataFrame,
    class_mapping: Dict[int, str],
    *,
    clf_pipeline,
    trained_feature_cols: Optional[List[str]],
) -> plt.Figure:
    """Create a 3-panel, publication-style HR diagram figure."""
    # Keep only finite coords
    x_vals = df_plot["bp_rp"].replace([np.inf, -np.inf], np.nan).dropna()
    y_vals = df_plot["abs_mag_g"].replace([np.inf, -np.inf], np.nan).dropna()

    if x_vals.empty or y_vals.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Not enough data for HR diagram", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Robust limits
    x_min, x_max = np.percentile(x_vals, [1, 99])
    y_min, y_max = np.percentile(y_vals, [1, 99])
    dx = (x_max - x_min) * 0.06 if x_max != x_min else 0.5
    dy = (y_max - y_min) * 0.06 if y_max != y_min else 0.5

    # Prepare class colors
    unique_codes = sorted(df_plot["class_label"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {code: cmap(i % 20) for i, code in enumerate(unique_codes)}

    # Physics colormap: radius if available else lum if available
    phys_val = None
    phys_label = None
    if "radius_val" in df_plot.columns and df_plot["radius_val"].notna().any():
        phys_val = np.log10(df_plot["radius_val"].replace(0, np.nan))
        phys_label = r"log$_{10}$(R / R$_\odot$)"
    elif "lum_val" in df_plot.columns and df_plot["lum_val"].notna().any():
        phys_val = np.log10(df_plot["lum_val"].replace(0, np.nan))
        phys_label = r"log$_{10}$(L / L$_\odot$)"

    # Classifier confidence
    class_conf = None
    if clf_pipeline is not None and hasattr(clf_pipeline, "predict_proba") and trained_feature_cols:
        # Ensure all trained columns exist
        Xp = df_plot.copy()
        for c in trained_feature_cols:
            if c not in Xp.columns:
                Xp[c] = np.nan
        try:
            probs = clf_pipeline.predict_proba(Xp[trained_feature_cols])
            class_conf = np.max(probs, axis=1)
        except Exception:
            class_conf = None

    # Publication-ish defaults
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 1.0, 0.95]}
    )
    ax_class, ax_phys, ax_conf = axes

    # ---- Panel 1: class-coded ----
    for code in unique_codes:
        sub = df_plot[df_plot["class_label"] == code]
        if sub.empty:
            continue
        ax_class.scatter(
            sub["bp_rp"], sub["abs_mag_g"],
            s=10, alpha=0.55,
            color=color_map.get(code),
            edgecolors="none",
            label=class_mapping.get(int(code), f"class_{int(code)}"),
            rasterized=True,
        )
    # density contours (subtle)
    try:
        sns.kdeplot(
            x=df_plot["bp_rp"],
            y=df_plot["abs_mag_g"],
            ax=ax_class,
            levels=8,
            color="black",
            linewidths=0.8,
            alpha=0.25,
        )
    except Exception:
        pass

    ax_class.set_title("HR diagram — colored by class")
    ax_class.set_xlabel("BP − RP (mag)")
    ax_class.set_ylabel(r"$M_G$ (mag)")
    # compact legend (outside)
    if unique_codes:
        ax_class.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    # ---- Panel 2: physics-coded ----
    if phys_val is not None and phys_label is not None:
        sc = ax_phys.scatter(
            df_plot["bp_rp"], df_plot["abs_mag_g"],
            c=phys_val,
            s=10, alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax_phys, fraction=0.046, pad=0.03)
        cbar.set_label(phys_label)
    else:
        ax_phys.scatter(df_plot["bp_rp"], df_plot["abs_mag_g"], s=10, alpha=0.6, edgecolors="none", rasterized=True)

    ax_phys.set_title("HR diagram — colored by radius / luminosity")
    ax_phys.set_xlabel("BP − RP (mag)")

    # ---- Panel 3: classifier confidence ----
    if class_conf is not None and np.isfinite(class_conf).any():
        conf = np.asarray(class_conf, dtype=float)
        conf = np.clip(conf, 0.0, 1.0)
        # size scale
        rng = float(np.nanmax(conf) - np.nanmin(conf))
        sizes = 18.0 + (conf - np.nanmin(conf)) / rng * 150.0 if rng > 0 else np.full_like(conf, 60.0)
        sc2 = ax_conf.scatter(
            df_plot["bp_rp"], df_plot["abs_mag_g"],
            s=sizes,
            c=conf,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )
        cbar2 = fig.colorbar(sc2, ax=ax_conf, fraction=0.046, pad=0.03)
        cbar2.set_label("Max class probability")
        ax_conf.set_title("HR diagram — classifier confidence")
    else:
        ax_conf.text(0.5, 0.5, "Train classifier to show confidence", ha="center", va="center", transform=ax_conf.transAxes)
        ax_conf.set_title("HR diagram — classifier confidence")

    ax_conf.set_xlabel("BP − RP (mag)")

    # ---- Shared axis formatting (HR convention: brighter = up) ----
    for a in axes:
        a.set_xlim(x_min - dx, x_max + dx)
        # Set limits then invert so smaller magnitudes are at the top
        a.set_ylim(y_min - dy, y_max + dy)
        a.invert_yaxis()
        a.grid(False)

    fig.tight_layout()
    return fig

# =========================
# App UI
# =========================
st.title("DataNova: Gaia Stellar Classification (reproducible)")
st.markdown(
    "Fetch Gaia DR3 samples, optionally enrich with variability/SOS tables, train a RandomForest classifier, "
    "inspect HR diagrams, and export reproducible datasets + metadata."
)

# Session defaults
st.session_state.setdefault("env_info", get_env_info())
st.session_state.setdefault("fetch_timestamp", None)
st.session_state.setdefault("last_query", None)
st.session_state.setdefault("gaia_release", "gaiadr3")
st.session_state.setdefault("sample_size", None)
st.session_state.setdefault("current_csv_path", None)
st.session_state.setdefault("dataset_version", 0)
st.session_state.setdefault("trained_model_present", False)
st.session_state.setdefault("trained_model", None)
st.session_state.setdefault("trained_feature_cols", None)
st.session_state.setdefault("last_eval", None)

# =========================
# Controls (sidebar)
# =========================
with st.sidebar:
    st.header("Dataset")
    default_folder = Path.cwd() / "data"
    output_folder = Path(st.text_input("Data/artifacts folder", value=str(st.session_state.get("output_folder", default_folder))))
    output_folder.mkdir(parents=True, exist_ok=True)
    if st.session_state.get("output_folder") != str(output_folder):
        st.session_state["output_folder"] = str(output_folder)

    datasets = list_datasets(output_folder)
    ds_labels = [f"{p.name}  (modified {dt.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" for p in datasets]
    current_path = get_current_csv_path()

    # Auto-select latest if none
    if current_path is None:
        latest = find_latest_sample(output_folder)
        if latest is not None:
            set_current_csv(latest)
            current_path = latest

    selected_idx = 0
    if current_path is not None:
        try:
            selected_idx = datasets.index(current_path)
        except Exception:
            selected_idx = 0

    if datasets:
        chosen = st.selectbox("Choose dataset", options=list(range(len(datasets))), format_func=lambda i: ds_labels[i], index=selected_idx)
        chosen_path = datasets[chosen]
        if current_path != chosen_path:
            set_current_csv(chosen_path)
            st.rerun()
    else:
        st.caption("No datasets found yet. Fetch from Gaia or upload a CSV.")

    st.divider()
    st.header("Fetch")
    sample_size = st.number_input("Sample size", min_value=1000, max_value=50000, value=20000, step=1000)
    if st.button("Fetch Gaia sample", use_container_width=True):
        if not GAIA_AVAILABLE:
            st.error("astroquery.gaia is not available. Install it to fetch from Gaia.")
        else:
            with st.spinner("Querying Gaia DR3..."):
                df_fetched, query = fetch_gaia_sample(int(sample_size))
                # Determine next sample index
                base_name = "gaia_sample"
                extension = ".csv"
                sample_num = 0
                while (output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}").exists():
                    sample_num += 1
                output_file = output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}"
                df_fetched.to_csv(output_file, index=False)
                st.session_state["fetch_timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
                st.session_state["last_query"] = query
                st.session_state["gaia_release"] = "gaiadr3"
                st.session_state["sample_size"] = int(sample_size)
                set_current_csv(output_file)
                append_research_log(output_folder, "Fetch", {"rows": int(len(df_fetched)), "file": str(output_file), "query": query})
                st.success(f"Saved {len(df_fetched)} rows to {output_file.name}")
                st.rerun()

    st.divider()
    st.header("Physics / enrichment")
    fetch_variability = st.checkbox("Enrich with gaiadr3.vari_summary", value=False)
    fetch_sos_periods = st.checkbox("Enrich with SOS period tables", value=False)

    st.caption("Tip: enrichment may take a while; results are saved to an *_enriched.csv so it won't refetch on reruns.")

    st.divider()
    st.header("Preprocessing")
    apply_ruwe = st.checkbox("Apply RUWE filter (if available)", value=True)
    ruwe_threshold = st.number_input("RUWE threshold", value=1.40, format="%.2f")
    apply_deredden = st.checkbox("Apply Gaia AP dereddening (if available)", value=True)

    dist_method = st.selectbox("Distance method", options=["inverse (default)", "bailerjones (local CSV)"], index=0)
    bj_path = ""
    if dist_method.startswith("bailerjones"):
        bj_path = st.text_input("Bailer-Jones CSV path (source_id + r_est / r_med_*)", value="")

    st.divider()
    st.header("Training")
    balancing_method = st.selectbox(
        "Balancing during training",
        options=[
            "none (class_weight only)",
            "SMOTE" if IMBLEARN_AVAILABLE else "SMOTE (requires imblearn)",
            "ADASYN" if IMBLEARN_AVAILABLE else "ADASYN (requires imblearn)",
        ],
        index=1 if IMBLEARN_AVAILABLE else 0,
    )
    use_full_features = st.checkbox("Use expanded feature set", value=True)
    min_samples_per_class = st.number_input("Min samples per class", min_value=2, max_value=200, value=15, step=1)

# =========================
# Load dataset (main area)
# =========================
st.subheader("Load data")
uploaded_file = st.file_uploader("Upload a local CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df_up = pd.read_csv(uploaded_file)
        local_name = Path(uploaded_file.name).name
        saved_path = output_folder / f"local_{local_name}"
        df_up.to_csv(saved_path, index=False)
        st.session_state["fetch_timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
        st.session_state["last_query"] = "local_upload"
        set_current_csv(saved_path)
        append_research_log(output_folder, "Upload", {"rows": int(len(df_up)), "file": str(saved_path)})
        st.success(f"Uploaded and saved: {saved_path.name} (rows: {len(df_up)})")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

current_csv = get_current_csv_path()
if current_csv is None or not current_csv.exists():
    st.warning("No dataset selected/available yet.")
    st.stop()

df_raw = read_csv_cached(str(current_csv), current_csv.stat().st_mtime)
st.info(f"Loaded `{current_csv.name}` — rows: {len(df_raw):,}  columns: {len(df_raw.columns)}")

# =========================
# Optional enrichment (persist to *_enriched.csv)
# =========================
def has_vari_summary_cols(df_: pd.DataFrame) -> bool:
    return any(c.startswith("var_") for c in df_.columns)

def has_sos_cols(df_: pd.DataFrame) -> bool:
    return any(c.startswith("period_from_") for c in df_.columns)

enrich_requested = (fetch_variability or fetch_sos_periods)
if enrich_requested:
    if not GAIA_AVAILABLE:
        st.warning("Enrichment requested, but astroquery.gaia is not available.")
    else:
        # Only enrich Gaia-fetched datasets by default (skip local uploads unless user insists).
        if current_csv.name.lower().startswith("local_"):
            st.info("Skipping enrichment for locally uploaded data (to avoid unexpected long Gaia queries).")

        else:
            needs_vari = fetch_variability and (not has_vari_summary_cols(df_raw))
            needs_sos = fetch_sos_periods and (not has_sos_cols(df_raw))

            if needs_vari or needs_sos:
                with st.spinner("Enriching dataset from Gaia tables..."):
                    ids = df_raw.get("source_id", pd.Series(dtype=float)).dropna()
                    ids = ids.astype("int64", errors="ignore").unique().tolist()
                    ids = [int(x) for x in ids if np.isfinite(x)]

                    if not ids:
                        st.warning("No source_id column/values — cannot enrich.")
                    else:
                        df_enriched = df_raw.copy()
                        payload = {"base": current_csv.name, "n_ids": len(ids), "vari": bool(needs_vari), "sos": bool(needs_sos)}

                        if needs_vari:
                            try:
                                vs = fetch_vari_summary(ids, chunk_size=2000)
                                if not vs.empty:
                                    # map candidate columns to canonical names
                                    def pick_col(df_local: pd.DataFrame, candidates: List[str]) -> Optional[str]:
                                        for c in candidates:
                                            if c in df_local.columns:
                                                return c
                                        return None

                                    amplitude_candidates = ["range_mag_g_fov", "trimmed_range_mag_g_fov", "range_mag_g_fov_trim"]
                                    median_g_candidates = ["median_mag_g_fov", "median_mag_g"]
                                    median_bp_candidates = ["median_mag_bp"]
                                    median_rp_candidates = ["median_mag_rp"]
                                    nobs_g_candidates = ["num_selected_g_fov", "n_selected_g_fov", "num_selected_g_fov_for_var"]

                                    map_cols = {
                                        "var_median_mag_g_fov": pick_col(vs, median_g_candidates),
                                        "var_median_mag_bp": pick_col(vs, median_bp_candidates),
                                        "var_median_mag_rp": pick_col(vs, median_rp_candidates),
                                        "var_amplitude_g": pick_col(vs, amplitude_candidates),
                                        "var_n_obs_g": pick_col(vs, nobs_g_candidates),
                                    }
                                    keep_cols = ["source_id"] + [v for v in map_cols.values() if v is not None]
                                    vs_trim = vs[keep_cols].copy()
                                    rename_map = {v: k for k, v in map_cols.items() if v is not None}
                                    vs_trim = vs_trim.rename(columns=rename_map).groupby("source_id").first().reset_index()
                                    df_enriched = df_enriched.merge(vs_trim, how="left", on="source_id")
                                    payload["vari_rows"] = int(len(vs))
                                    payload["vari_merged"] = int(len(vs_trim))
                                else:
                                    payload["vari_rows"] = 0
                            except Exception as e:
                                st.warning(f"vari_summary enrichment failed: {e}")
                                payload["vari_error"] = str(e)

                        if needs_sos:
                            sos_tables = {
                                "vari_rrlyrae": ["pf", "p1_o", "period"],
                                "vari_cepheid": ["pf", "p1_o", "period", "period_f"],
                                "vari_long_period_variable": ["frequency", "period"],
                            }
                            for table, candidates in sos_tables.items():
                                try:
                                    sos = fetch_sos_table(table, ids, chunk_size=2000)
                                    if sos.empty:
                                        continue
                                    period_col = None
                                    for cand in candidates:
                                        if cand in sos.columns and sos[cand].notna().any():
                                            period_col = cand
                                            break
                                    if period_col is None:
                                        continue
                                    sos2 = sos.copy()
                                    if period_col == "frequency":
                                        with np.errstate(divide="ignore", invalid="ignore"):
                                            sos2["period_calculated"] = 1.0 / sos2["frequency"].replace(0, np.nan)
                                        period_col = "period_calculated"
                                    sos2 = sos2.loc[sos2[period_col].notna() & (sos2[period_col] > 0), ["source_id", period_col]]
                                    if sos2.empty:
                                        continue
                                    sos_trim = sos2.groupby("source_id").first().reset_index().rename(columns={period_col: f"period_from_{table}"})
                                    df_enriched = df_enriched.merge(sos_trim, how="left", on="source_id")
                                    payload[f"sos_{table}_rows"] = int(len(sos))
                                    payload[f"sos_{table}_merged"] = int(len(sos_trim))
                                except Exception as e:
                                    st.warning(f"SOS enrichment failed for {table}: {e}")
                                    payload[f"sos_{table}_error"] = str(e)

                        # Persist and switch current dataset
                        stem = current_csv.stem
                        enriched_path = output_folder / f"{stem}_enriched.csv"
                        df_enriched.to_csv(enriched_path, index=False)
                        set_current_csv(enriched_path)
                        append_research_log(output_folder, "Enrich", payload)
                        st.success(f"Enrichment complete. Saved: {enriched_path.name}")
                        st.rerun()
            else:
                st.info("Enrichment already present in this dataset — skipping.")

# =========================
# Preprocess
# =========================
bj_df = None
if dist_method.startswith("bailerjones") and bj_path:
    try:
        bj_p = Path(bj_path)
        if bj_p.exists():
            bj_df = read_bailer_jones_cached(str(bj_p), bj_p.stat().st_mtime)
        else:
            st.warning("Bailer-Jones path does not exist; falling back to parallax inversion.")
            bj_df = None
    except Exception as e:
        st.warning(f"Failed to read Bailer-Jones CSV: {e}")
        bj_df = None

df_clean, class_mapping, feature_candidates, dered_applied = preprocess_dataset(
    df_raw,
    apply_deredden=apply_deredden,
    apply_ruwe=apply_ruwe,
    ruwe_threshold=float(ruwe_threshold),
    dist_method=dist_method,
    bj_df=bj_df,
    parallax_snr_cut=5.0,
)

st.caption(
    f"After preprocessing: {len(df_clean):,} rows. "
    f"Dereddening applied: {'yes' if dered_applied else 'no'}. "
    f"(Streamlit reruns on interactions; heavy steps are cached and enrichment is persisted.)"
)

with st.expander("Show raw class counts"):
    if "best_class_name" in df_clean.columns:
        st.write(df_clean["best_class_name"].value_counts().head(30))
    else:
        st.write("No best_class_name column.")

# Build training feature matrix
X_full = df_clean[[c for c in feature_candidates if c in df_clean.columns]].copy()
# Keep numeric only for training
for c in X_full.columns:
    X_full[c] = pd.to_numeric(X_full[c], errors="coerce")

# Minimal feature set
features_min = [c for c in ["bp_rp", "abs_mag_g", "teff_gspphot"] if c in df_clean.columns]
X_min = df_clean[features_min].copy()

y = df_clean["class_label"] if "class_label" in df_clean.columns else pd.Series(dtype=int)
# =========================
# Dataset overview & QA (optional; does not change your data)
# Streamlit reruns the script on most interactions; heavy computations are cached where possible.
# =========================
st.subheader("Dataset overview & QA (research helpers)")
with st.expander("Open dataset overview & quality checks", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows (clean)", f"{len(df_clean):,}")
    with c2:
        n_cls = int(df_clean["best_class_name"].nunique()) if "best_class_name" in df_clean.columns else 0
        st.metric("Classes", n_cls)
    with c3:
        dup = int(df_clean["source_id"].duplicated().sum()) if "source_id" in df_clean.columns else 0
        st.metric("Duplicate source_id", dup)
    with c4:
        st.metric("Dereddened", "Yes" if dered_applied else "No")

    miss = compute_missingness_table(df_clean)
    st.markdown("##### Missingness report")
    st.dataframe(miss, use_container_width=True, height=260)
    st.download_button(
        "Download missingness report (CSV)",
        data=df_to_csv_bytes(miss),
        file_name=f"{(current_csv.stem if current_csv else 'dataset')}_missingness.csv",
        mime="text/csv",
    )

    st.markdown("##### Quick diagnostic plots")
    tab_a, tab_b, tab_c, tab_d = st.tabs(["Class distribution", "CMD density", "Sky map", "Key distributions"])
    with tab_a:
        fig_cd = plot_class_distribution(df_clean)
        if fig_cd is None:
            st.info("No class column available to plot.")
        else:
            st.pyplot(fig_cd)
            st.download_button("Download class distribution (PNG)", data=convert_fig_to_bytes(fig_cd), file_name="class_distribution.png", mime="image/png")

    with tab_b:
        fig_cmd = plot_cmd_density(df_clean)
        if fig_cmd is None:
            st.info("Not enough data for CMD density plot.")
        else:
            st.pyplot(fig_cmd)
            st.download_button("Download CMD density (PNG)", data=convert_fig_to_bytes(fig_cmd), file_name="cmd_density.png", mime="image/png")

    with tab_c:
        fig_sky = plot_sky_map(df_clean)
        if fig_sky is None:
            st.info("RA/Dec columns not found (or insufficient data).")
        else:
            st.pyplot(fig_sky)
            st.download_button("Download sky map (PNG)", data=convert_fig_to_bytes(fig_sky), file_name="sky_map.png", mime="image/png")

    with tab_d:
        # compute a few common diagnostics on the fly
        if "parallax" in df_clean.columns and "parallax_error" in df_clean.columns:
            plx = pd.to_numeric(df_clean["parallax"], errors="coerce")
            plxerr = pd.to_numeric(df_clean["parallax_error"], errors="coerce").replace(0, np.nan)
            snr = plx / plxerr
            fig_snr = plot_diagnostic_hist(df_clean, snr, "Parallax SNR (clipped)", "parallax / parallax_error")
            if fig_snr is not None:
                st.pyplot(fig_snr)
        if "bp_rp" in df_clean.columns:
            fig_col = plot_diagnostic_hist(df_clean, df_clean["bp_rp"], "Color (BP-RP) distribution (clipped)", "BP - RP (mag)")
            if fig_col is not None:
                st.pyplot(fig_col)
        if "abs_mag_g" in df_clean.columns:
            fig_mg = plot_diagnostic_hist(df_clean, df_clean["abs_mag_g"], "$M_G$ distribution (clipped)", "$M_G$ (mag)")
            if fig_mg is not None:
                st.pyplot(fig_mg)
        if "teff_gspphot" in df_clean.columns:
            fig_teff = plot_diagnostic_hist(df_clean, df_clean["teff_gspphot"], "Effective temperature distribution (clipped)", "T_eff (K)")
            if fig_teff is not None:
                st.pyplot(fig_teff)


# =========================
# Correlation analysis
# =========================
st.subheader("Correlation analysis")
numeric_df = df_clean.select_dtypes(include=[np.number]).copy()
if "class_label" in numeric_df.columns:
    numeric_features = numeric_df.drop(columns=["class_label"])
else:
    numeric_features = numeric_df

if numeric_features.shape[0] < 2 or numeric_features.shape[1] < 2:
    st.info("Not enough numeric data to compute correlations.")
else:
    corr = compute_corr_cached(numeric_features)

    save_corr = st.button("Save correlation artifacts (CSV + JSON)", key=f"save_corr_{current_csv.stem}")
    corr_file = output_folder / f"{current_csv.stem}_correlation_matrix.csv"

    n = corr.shape[0]
    show_annot = n <= 20  # annotated heatmap gets unreadable/heavy beyond this
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(min(14, 0.5*n + 4), min(10, 0.4*n + 3)))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        annot=show_annot,
        fmt=".2f",
        annot_kws={"size": 7} if show_annot else None,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"shrink": 0.7},
        ax=ax_corr,
    )
    ax_corr.set_title("Pearson correlation matrix (lower triangle)")
    ax_corr.tick_params(axis="x", labelrotation=45)
    ax_corr.tick_params(axis="y", rotation=0)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    st.download_button("Download correlation heatmap (PNG)", data=convert_fig_to_bytes(fig_corr), file_name="correlation_matrix.png", mime="image/png")

    if save_corr and ("class_label" not in df_clean.columns or df_clean["class_label"].nunique() < 2):
        try:
            corr.to_csv(corr_file)
            append_research_log(output_folder, "Correlation artifacts", {"corr_csv": str(corr_file)})
            st.success(f"Saved: {corr_file.name}")
        except Exception as e:
            st.warning(f"Could not save correlation CSV: {e}")

    # Feature→class scoring (optional)
    if "class_label" in df_clean.columns and df_clean["class_label"].nunique() >= 2:
        X_scores = numeric_features.copy()
        # median-impute for scoring
        X_scores = X_scores.fillna(X_scores.median(numeric_only=True))
        y_labels = df_clean["class_label"].values
        try:
            f_vals, p_vals = f_classif(X_scores, y_labels)
        except Exception:
            f_vals = np.full(X_scores.shape[1], np.nan)
            p_vals = np.full(X_scores.shape[1], np.nan)
        try:
            mi_vals = mutual_info_classif(X_scores, y_labels, discrete_features=False, random_state=DEFAULT_SEED)
        except Exception:
            mi_vals = np.full(X_scores.shape[1], np.nan)

        def eta_squared(feature_series: pd.Series, labels: np.ndarray) -> float:
            data = pd.DataFrame({"x": feature_series, "g": labels}).dropna()
            if data.empty:
                return float("nan")
            grand_mean = data["x"].mean()
            ss_between = (data.groupby("g").size() * (data.groupby("g")["x"].mean() - grand_mean) ** 2).sum()
            ss_total = ((data["x"] - grand_mean) ** 2).sum()
            return float(ss_between / ss_total) if ss_total > 0 else float("nan")

        eta2 = [eta_squared(X_scores[col], y_labels) for col in X_scores.columns]

        scores_df = pd.DataFrame(
            {"feature": X_scores.columns, "f_score": f_vals, "f_pvalue": p_vals, "mutual_info": mi_vals, "eta_squared": eta2}
        ).set_index("feature")
        st.markdown("**Top features by F-score / MI / eta²**")
        cols = st.columns(3)
        with cols[0]:
            st.dataframe(scores_df[["f_score"]].sort_values("f_score", ascending=False).head(10))
        with cols[1]:
            st.dataframe(scores_df[["mutual_info"]].sort_values("mutual_info", ascending=False).head(10))
        with cols[2]:
            st.dataframe(scores_df[["eta_squared"]].sort_values("eta_squared", ascending=False).head(10))

        if save_corr:
            try:
                corr.to_csv(corr_file)
            except Exception:
                pass
            scores_file = output_folder / f"{current_csv.stem}_feature_scores.json"
            with open(scores_file, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset": current_csv.name,
                        "computed_at": dt.datetime.utcnow().isoformat() + "Z",
                        "n_samples": int(len(df_clean)),
                        "correlation_csv": str(corr_file),
                        "feature_scores": scores_df.to_dict(orient="index"),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                    default=safe_json_default,
                )
            append_research_log(output_folder, "Correlation artifacts", {"corr_csv": str(corr_file), "scores_json": str(scores_file)})
            st.success(f"Saved: {corr_file.name} and {scores_file.name}")

# =========================
# Training (button-driven; persisted in session_state)
# =========================
st.subheader("Train classifier")

if y.empty or y.nunique() < 2:
    st.warning("Not enough class labels to train a classifier.")
    clf = None
    trained_feature_cols = None
else:
    if use_full_features and not X_full.empty:
        X_train_base = X_full
    else:
        X_train_base = X_min

    # Ensure numeric dtypes
    X_train_base = X_train_base.apply(pd.to_numeric, errors="coerce")

    train_clicked = st.button("Train classifier now", key="train_btn")

    clf = st.session_state.get("trained_model") if st.session_state.get("trained_model_present") else None
    trained_feature_cols = st.session_state.get("trained_feature_cols")

    if train_clicked:
        with st.spinner("Training model..."):
            try:
                counts = y.value_counts()
                valid_classes = counts[counts >= int(min_samples_per_class)].index
                dropped = counts[counts < int(min_samples_per_class)].index.tolist()
                if dropped:
                    st.warning(f"Dropping classes with < {int(min_samples_per_class)} samples: {dropped}")

                Xv = X_train_base[y.isin(valid_classes)].copy()
                yv = y[y.isin(valid_classes)].copy()

                if yv.nunique() < 2 or len(yv) < (2 * int(min_samples_per_class)):
                    st.warning("Not enough data after class filtering.")
                else:
                    n_classes = int(yv.nunique())
                    n_samples = int(len(yv))
                    test_size = max(n_classes, int(0.2 * n_samples))
                    if test_size >= n_samples:
                        test_size = max(1, n_samples - 1)
                    stratify_param = yv if (test_size >= n_classes) else None

                    X_train, X_test, y_train, y_test = train_test_split(
                        Xv, yv, test_size=test_size, stratify=stratify_param, random_state=DEFAULT_SEED
                    )

                    base_clf = RandomForestClassifier(
                        n_estimators=250,
                        random_state=DEFAULT_SEED,
                        class_weight="balanced",
                        n_jobs=-1,
                    )

                    # Build pipeline: impute -> (sampler) -> clf
                    if IMBLEARN_AVAILABLE and balancing_method in {"SMOTE", "ADASYN"}:
                        min_train_class = int(y_train.value_counts().min())
                        # keep k_neighbors safely below smallest class size
                        safe_k = max(1, min(5, max(1, min_train_class - 1)))
                        sampler = SMOTE(random_state=DEFAULT_SEED, k_neighbors=safe_k) if balancing_method == "SMOTE" else ADASYN(random_state=DEFAULT_SEED, n_neighbors=safe_k)

                        pipeline = ImbPipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("sampler", sampler),
                                ("clf", base_clf),
                            ]
                        )
                    else:
                        pipeline = SkPipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("clf", base_clf),
                            ]
                        )

                    param_grid = {
                        "clf__n_estimators": [150, 250],
                        "clf__max_depth": [None, 12],
                        "clf__min_samples_split": [2, 5],
                        "clf__min_samples_leaf": [1, 2],
                    }
                    cv = 3 if len(X_train) < 200 else 5

                    gs = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        cv=cv,
                        scoring="f1_weighted",
                        n_jobs=-1,
                        error_score="raise",
                    )
                    gs.fit(X_train, y_train)
                    best = gs.best_estimator_

                    # Evaluate
                    y_pred = best.predict(X_test)
                    unique_classes = sorted(set(y_test) | set(y_pred))
                    target_names = [class_mapping.get(int(i), f"class_{int(i)}") for i in unique_classes]

                    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
                    macro_f1 = f1_score(y_test, y_pred, average="macro")
                    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
                    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

                    st.session_state["trained_model_present"] = True
                    st.session_state["trained_model"] = best
                    st.session_state["trained_feature_cols"] = list(Xv.columns)
                    st.session_state["last_eval"] = {
                        "macro_f1": float(macro_f1),
                        "weighted_f1": float(weighted_f1),
                        "report": report,
                        "confusion_matrix": cm.tolist(),
                        "labels": unique_classes,
                        "target_names": target_names,
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                    }

                    append_research_log(output_folder, "Train", {
                        "dataset": current_csv.name,
                        "features": list(Xv.columns),
                        "best_params": gs.best_params_,
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "macro_f1": float(macro_f1),
                        "weighted_f1": float(weighted_f1),
                    })

                    st.success(f"Model trained. Macro F1: {macro_f1:.3f}  Weighted F1: {weighted_f1:.3f}")
                    st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    # show evaluation if present
    if st.session_state.get("trained_model_present") and st.session_state.get("last_eval") is not None:
        ev = st.session_state["last_eval"]
        st.caption(f"Restored trained model. Train/test: {ev.get('n_train')}/{ev.get('n_test')}  Macro F1: {ev.get('macro_f1'):.3f}")
        col_rep, col_cm = st.columns([1.2, 1.0])
        with col_rep:
            rep_df = pd.DataFrame(ev["report"]).transpose()
            st.markdown("##### Classification report")
            st.dataframe(rep_df.style.format("{:.2f}", subset=[c for c in ["precision", "recall", "f1-score"] if c in rep_df.columns]))
        with col_cm:
            st.markdown("##### Confusion matrix")
            cm = np.array(ev["confusion_matrix"], dtype=int)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ev["target_names"], yticklabels=ev["target_names"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            st.download_button("Download confusion matrix (PNG)", data=convert_fig_to_bytes(fig_cm), file_name="confusion_matrix.png", mime="image/png")

    clf = st.session_state.get("trained_model") if st.session_state.get("trained_model_present") else None
    trained_feature_cols = st.session_state.get("trained_feature_cols")

# =========================
# HR Diagrams
# =========================
# =========================
# Researcher tools: interpretability + prediction exports (only after training)
# =========================
if clf is not None and st.session_state.get("trained_model_present", False):
    st.subheader("Model interpretability & prediction export")
    with st.expander("Open interpretability / prediction tools", expanded=False):
        stem = current_csv.stem if current_csv else "dataset"
        est = extract_estimator(clf)

        # --- Feature importance
        if trained_feature_cols and hasattr(est, "feature_importances_"):
            fig_fi = plot_feature_importance_bar(est, trained_feature_cols, top_n=25)
            if fig_fi is not None:
                st.pyplot(fig_fi)
                st.download_button("Download feature importances (PNG)", data=convert_fig_to_bytes(fig_fi), file_name=f"{stem}_feature_importance.png", mime="image/png")
        else:
            st.info("Feature importances not available for the current model.")

        # --- Permutation importance (optional; can be slow)
        st.markdown("##### Permutation importance (test set)")
        perm_cols = st.columns([1,1,2])
        with perm_cols[0]:
            perm_repeats = st.number_input("Repeats", min_value=2, max_value=30, value=8, step=1)
        with perm_cols[1]:
            perm_max_rows = st.number_input("Max test rows", min_value=200, max_value=10000, value=3000, step=200)
        with perm_cols[2]:
            st.caption("This can take time for many features. Uses the stored test split (if still compatible with current settings).")

        if st.button("Compute permutation importance", key="perm_importance_btn"):
            idx_info = st.session_state.get("train_test_index")
            if not idx_info or idx_info.get("dataset_tag") != dataset_tag:
                st.warning("Test split not available (or settings changed since training). Retrain, then retry.")
            else:
                test_idx = idx_info.get("test", [])
                if not test_idx:
                    st.warning("No stored test indices found.")
                else:
                    # rebuild X_test/y_test from current df_clean (best-effort)
                    X_all = pd.DataFrame({c: df_clean.get(c, np.nan) for c in trained_feature_cols})
                    y_all = df_clean["class_label"] if "class_label" in df_clean.columns else None
                    X_test = X_all.loc[X_all.index.intersection(test_idx)]
                    y_test = y_all.loc[y_all.index.intersection(test_idx)] if y_all is not None else None
                    if y_test is None or X_test.empty or y_test.empty:
                        st.warning("Could not reconstruct test split from current dataset view. Retrain and retry.")
                    else:
                        # sample rows for speed
                        if len(X_test) > int(perm_max_rows):
                            X_test = X_test.sample(n=int(perm_max_rows), random_state=42)
                            y_test = y_test.loc[X_test.index]
                        try:
                            with st.spinner("Computing permutation importance..."):
                                perm = permutation_importance(
                                    clf, X_test, y_test,
                                    n_repeats=int(perm_repeats),
                                    random_state=42,
                                    n_jobs=-1,
                                    scoring="f1_weighted"
                                )
                            imp_mean = perm.importances_mean
                            order = np.argsort(imp_mean)[-25:]
                            fig_pi, ax_pi = plt.subplots(figsize=(8, max(4, 0.25*len(order)+1.5)))
                            ax_pi.barh([trained_feature_cols[i] for i in order], imp_mean[order])
                            ax_pi.set_xlabel("Permutation importance (Δ score)")
                            ax_pi.set_title("Top permutation importances (weighted F1)")
                            plt.tight_layout()
                            st.pyplot(fig_pi)
                            st.download_button("Download permutation importances (PNG)", data=convert_fig_to_bytes(fig_pi), file_name=f"{stem}_permutation_importance.png", mime="image/png")
                        except Exception as e:
                            st.error(f"Permutation importance failed: {e}")

        st.markdown("##### Export predictions (full or filtered dataset)")
        pred_scope = st.selectbox("Predict on:", options=["clean dataset"], index=0)
        top_k = st.number_input("Top-K classes to include", min_value=1, max_value=5, value=3, step=1)

        if st.button("Generate prediction table", key="gen_pred_table_btn"):
            X_pred = pd.DataFrame({c: df_clean.get(c, np.nan) for c in trained_feature_cols})
            if X_pred.empty:
                st.warning("No rows available for prediction with the current selection.")
            else:
                try:
                    y_hat = clf.predict(X_pred)
                    out = pd.DataFrame(index=X_pred.index)
                    if "source_id" in df_clean.columns:
                        out["source_id"] = df_clean.loc[out.index, "source_id"].values
                    # map to class names where possible
                    model_classes = getattr(clf, "classes_", None)
                    if model_classes is None and hasattr(extract_estimator(clf), "classes_"):
                        model_classes = extract_estimator(clf).classes_
                    out["pred_code"] = y_hat
                    out["pred_class"] = [class_mapping.get(int(c), str(c)) for c in y_hat]

                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X_pred)
                        conf = proba.max(axis=1)
                        out["confidence"] = conf
                        # top-k
                        cls_list = list(getattr(clf, "classes_", range(proba.shape[1])))
                        top_idx = np.argsort(proba, axis=1)[:, ::-1][:, : int(top_k)]
                        for rank in range(int(top_k)):
                            codes = [cls_list[i] for i in top_idx[:, rank]]
                            out[f"top{rank+1}_class"] = [class_mapping.get(int(c), str(c)) for c in codes]
                            out[f"top{rank+1}_prob"] = [float(proba[r, i]) for r, i in enumerate(top_idx[:, rank])]
                    st.session_state["last_prediction_table"] = out.reset_index(drop=True)
                    st.success(f"Predictions generated: {len(out):,} rows")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        pred_table = st.session_state.get("last_prediction_table")
        if isinstance(pred_table, pd.DataFrame) and not pred_table.empty:
            st.dataframe(pred_table.head(200), use_container_width=True, height=260)
            st.download_button(
                "Download predictions (CSV)",
                data=df_to_csv_bytes(pred_table),
                file_name=f"{stem}_predictions.csv",
                mime="text/csv",
            )

        st.markdown("##### Download trained model (pickle)")
        try:
            model_bytes = pickle.dumps(clf)
            st.download_button(
                "Download model file (.pkl)",
                data=model_bytes,
                file_name=f"{stem}_model.pkl",
                mime="application/octet-stream",
                help="Pickle is Python-specific. Only unpickle in a trusted environment."
            )
        except Exception as e:
            st.info(f"Model download unavailable: {e}")

        st.markdown("##### Reproducibility bundle (zip)")
        st.caption("Bundles common artifacts created in this run. Large datasets may make the zip big.")
        if st.button("Build reproducibility zip", key="build_zip_btn"):
            stem = current_csv.stem if current_csv else "dataset"
            candidates = []
            # dataset itself
            if current_csv and current_csv.exists():
                candidates.append(current_csv)
            # common exported artifacts (if they exist already)
            for suffix in [
                "_original.csv", "_filtered.csv", "_metadata.json",
                "_feature_scores.json", "_correlation_matrix.csv"
            ]:
                p = output_folder / f"{stem}{suffix}"
                if p.exists():
                    candidates.append(p)
            # images commonly produced
            for img in [
                output_folder / f"{stem}_hr_diagrams.png",
                output_folder / "correlation_matrix.png",
                output_folder / "confusion_matrix.png",
                output_folder / "hr_diagrams.png",
            ]:
                if img.exists():
                    candidates.append(img)
            # research log (if present)
            log_path = output_folder / "research_log.md"
            if log_path.exists():
                candidates.append(log_path)

            zbytes = make_zip_bytes(candidates, arc_prefix=f"{stem}_")
            st.session_state["repro_zip"] = zbytes
            st.success(f"Zip built with {len(candidates)} file(s).")

        zbytes = st.session_state.get("repro_zip")
        if isinstance(zbytes, (bytes, bytearray)) and len(zbytes) > 0:
            stem = current_csv.stem if current_csv else "dataset"
            st.download_button(
                "Download reproducibility bundle (ZIP)",
                data=zbytes,
                file_name=f"{stem}_repro_bundle.zip",
                mime="application/zip",
            )

st.subheader("HR Diagrams")

if {"bp_rp", "abs_mag_g"}.issubset(df_clean.columns):
    fig_hr = plot_hr_diagrams(df_clean, class_mapping, clf_pipeline=clf, trained_feature_cols=trained_feature_cols)
    st.pyplot(fig_hr)
    st.download_button("Download HR diagrams (PNG)", data=convert_fig_to_bytes(fig_hr, dpi=300), file_name="hr_diagrams.png", mime="image/png")
else:
    st.warning("Missing bp_rp or abs_mag_g — cannot draw HR diagrams.")

# =========================
# Filtering & export (use a form to avoid reruns on each slider change)
# =========================
st.subheader("Filter stars and export")
st.caption("Inputs below are in a form, so changing them will not rerun the whole app until you click Apply/Export.")

available_classes = list(df_clean["best_class_name"].dropna().unique()) if "best_class_name" in df_clean.columns else []
min_default = float(df_clean["abs_mag_g"].min()) if not df_clean.empty else 0.0
max_default = float(df_clean["abs_mag_g"].max()) if not df_clean.empty else 0.0

def filter_stars(df_: pd.DataFrame, star_class: Optional[str], min_mag: float, max_mag: float) -> pd.DataFrame:
    out = df_.copy()
    if star_class:
        if "best_class_name" in out.columns:
            out = out[out["best_class_name"] == star_class]
    out = out[out["abs_mag_g"] >= float(min_mag)]
    out = out[out["abs_mag_g"] <= float(max_mag)]
    cols = [c for c in ["source_id", "best_class_name", "bp_rp", "abs_mag_g", "teff_gspphot"] if c in out.columns]
    return out[cols] if cols else out

with st.form(key=f"filter_form_{st.session_state.get('dataset_version', 0)}"):
    c1, c2, c3 = st.columns(3)
    with c1:
        star_class = st.selectbox("Class", options=[None] + available_classes, index=0)
    with c2:
        min_mag = st.number_input("Min M_G", value=min_default)
    with c3:
        max_mag = st.number_input("Max M_G", value=max_default)
        include_predictions = st.checkbox("Include model predictions in exported CSVs (if a model is trained)", value=False)
        st.caption("Note: Streamlit can rerun the script on export/apply — that’s normal.")

    apply_btn = st.form_submit_button("Apply filters")
    export_btn = st.form_submit_button("Export CSVs & metadata")

# Compute filtered view (either after apply or always show last-used; we compute each run cheaply)
filtered_df = filter_stars(df_clean, star_class, float(min_mag), float(max_mag))
st.caption(f"Filtered stars: {len(filtered_df):,}")
with st.expander("Preview filtered table"):
    st.dataframe(filtered_df.reset_index(drop=True).head(200))

if export_btn:
    stem = current_csv.stem
    original_csv_path = output_folder / f"{stem}_original.csv"
    filtered_csv_path = output_folder / f"{stem}_filtered.csv"
    metadata_file = output_folder / f"{stem}_metadata.json"

    # build export tables (optionally with model predictions)
    df_export = df_clean.copy()
    filtered_export = filtered_df.copy()
    if include_predictions and st.session_state.get('trained_model_present', False):
        clf_export = st.session_state.get('trained_model')
        feat_cols = st.session_state.get('trained_feature_cols', [])
        if clf_export is not None and feat_cols:
            try:
                X_pred = pd.DataFrame({c: df_export.get(c, np.nan) for c in feat_cols})
                y_hat = clf_export.predict(X_pred)
                df_export['pred_code'] = y_hat
                df_export['pred_class'] = [class_mapping.get(int(c), str(c)) for c in y_hat]
                pred_cols = ['pred_code', 'pred_class']
                if hasattr(clf_export, 'predict_proba'):
                    proba = clf_export.predict_proba(X_pred)
                    df_export['confidence'] = proba.max(axis=1)
                    pred_cols.append('confidence')
                # attach predictions to the filtered export (preserve filtered columns)
                try:
                    filtered_export = filtered_export.join(df_export[pred_cols], how='left')
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Could not add predictions to export: {e}")

    df_export.to_csv(original_csv_path, index=False)
    filtered_export.to_csv(filtered_csv_path, index=False)

    metadata = {
        "dataset": current_csv.name,
        "exported_at": dt.datetime.utcnow().isoformat() + "Z",
        "exported_original": str(original_csv_path),
        "exported_filtered": str(filtered_csv_path),
        "include_predictions": bool(include_predictions),
        "filters": {"class": star_class, "min_mag": float(min_mag), "max_mag": float(max_mag)},
        "preprocessing": {
            "apply_ruwe": bool(apply_ruwe),
            "ruwe_threshold": float(ruwe_threshold),
            "apply_deredden": bool(apply_deredden),
            "distance_method": dist_method,
            "parallax_snr_cut": 5.0,
        },
        "training": {
            "trained_model_present": bool(st.session_state.get("trained_model_present")),
            "trained_feature_cols": st.session_state.get("trained_feature_cols"),
            "last_eval": st.session_state.get("last_eval"),
        },
        "env": st.session_state.get("env_info", {}),
        "query": st.session_state.get("last_query"),
        "fetch_timestamp": st.session_state.get("fetch_timestamp"),
        "gaia_release": st.session_state.get("gaia_release"),
        "sample_size_requested": st.session_state.get("sample_size"),
    }
    with open(metadata_file, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=safe_json_default)

    append_research_log(output_folder, "Export", {"original": str(original_csv_path), "filtered": str(filtered_csv_path), "metadata": str(metadata_file)})

    st.success(
        "CSVs exported:\n\n"
        f"Original: {original_csv_path}\n"
        f"Filtered: {filtered_csv_path}\n"
        f"Metadata: {metadata_file}"
    )

# Footer: environment info
with st.expander("Environment / reproducibility info"):
    st.json(st.session_state.get("env_info", {}))
