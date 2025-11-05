# %% DEPENDENCY CHECK
import importlib
import sys
required_packages = ["pandas", "numpy", "matplotlib", "sklearn", "streamlit", "astroquery", "seaborn"]
missing = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
if missing:
    import streamlit as st
    st.error(
        f"The following required packages are missing:\n{', '.join(missing)}\n\n"
        "Please install them and restart the app."
    )
    sys.exit("Dependencies missing. Exiting.")

# %% IMPORTS
from pathlib import Path
import json
import datetime as dt
import io
import numpy as np
import random
random.seed(42)
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.feature_selection import f_classif, mutual_info_classif

import streamlit as st
import glob
import os

# Optional Gaia query
try:
    from astroquery.gaia import Gaia
    GAIA_AVAILABLE = True
except Exception:
    GAIA_AVAILABLE = False

# =========================
# Helpers for persistence
# =========================
def find_latest_sample(folder: Path) -> Path | None:
    """Return the newest gaia_sample*.csv in folder, or None if none exist."""
    pattern = str(folder / "gaia_sample*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    newest = max(candidates, key=os.path.getmtime)
    return Path(newest)

def set_current_csv(path: Path | None):
    """Persist the current CSV path in session_state and bump dataset_version."""
    st.session_state["current_csv_path"] = str(path) if path else None
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1

def get_current_csv_path() -> Path | None:
    p = st.session_state.get("current_csv_path")
    return Path(p) if p else None

def safe_json_default(o):
    """Helper for json.dump default to convert numpy and datetimes."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (dt.datetime,)):
        return o.isoformat() + "Z"
    return str(o)

# =========================
# STREAMLIT GUI
# =========================
st.title("DataNova: Gaia Stellar Classification")
st.markdown("""
Fetch Gaia star samples, train a RandomForest classifier, inspect HR diagrams, run correlation analysis, and export reproducible datasets + metadata.
""")

# session-state defaults (prevents NameError on first run)
st.session_state.setdefault("fetch_timestamp", None)
st.session_state.setdefault("last_query", None)
st.session_state.setdefault("gaia_release", "gaiadr3")
st.session_state.setdefault("sample_size", None)
st.session_state.setdefault("current_csv_path", None)
st.session_state.setdefault("dataset_version", 0)

# --- User selects sample size
sample_size = st.number_input(
    "Number of stars to fetch/analyze:",
    min_value=1000, max_value=100000, value=20000, step=1000
)

# --- User selects folder to save CSVs
st.markdown("**Folder to save CSVs (original + filtered):**")
st.markdown("_Tip: Type/paste a path, default is `./data`._")
default_folder = Path.cwd() / "data"
output_folder = Path(st.text_input("Folder path:", value=str(st.session_state.get("output_folder", default_folder))))
output_folder.mkdir(parents=True, exist_ok=True)
if st.session_state.get("output_folder") != str(output_folder):
    st.session_state["output_folder"] = str(output_folder)
    latest = find_latest_sample(output_folder)
    set_current_csv(latest)

st.write(f"CSV files will be saved to: `{output_folder}`")

# Try to auto-discover latest sample
if get_current_csv_path() is None:
    latest = find_latest_sample(output_folder)
    if latest is not None:
        set_current_csv(latest)

# %% STEP 1: FETCH OR LOAD DATA
df = None
current_csv = get_current_csv_path()

if GAIA_AVAILABLE and st.button("Fetch Gaia Sample", key="fetch_btn"):
    st.write("Fetching Gaia sample...")
    base_name = "gaia_sample"
    extension = ".csv"

    # Determine next sample index
    sample_num = 0
    while (output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}").exists():
        sample_num += 1
    output_file = output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}"

    # Build ADQL query: choose astrophysical parameters columns known to exist in DR3
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
    v.best_class_name
FROM gaiadr3.vari_classifier_result AS v
JOIN gaiadr3.gaia_source AS g
    ON v.source_id = g.source_id
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON v.source_id = ap.source_id
WHERE v.best_class_name IS NOT NULL
"""
    try:
        job = Gaia.launch_job(query)
        df = job.get_results().to_pandas()
        df.to_csv(output_file, index=False)
        st.success(f"Gaia sample fetched and saved as {output_file.name}, rows: {len(df)}")

        # Persist metadata in session_state for reproducibility
        st.session_state["fetch_timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
        st.session_state["last_query"] = query.strip()
        st.session_state["gaia_release"] = "gaiadr3"
        st.session_state["sample_size"] = int(sample_size)

        # Persist this as the current dataset
        set_current_csv(output_file)

        # update current_csv pointer for immediate use below
        current_csv = get_current_csv_path()

    except Exception as e:
        st.error(f"Gaia query failed: {e}")
        st.warning("You can still proceed with mock data or load a local CSV.")

# Load current CSV if available (or fallback to mock)
current_csv = get_current_csv_path()
if current_csv and current_csv.exists():
    df = pd.read_csv(current_csv)
    st.write(f"Loaded CSV `{current_csv.name}`, rows: {len(df)}")
else:
    if df is None:
        st.warning("No Gaia data available. Using fallback mock data.")
        df = pd.DataFrame({
            "source_id": [1,2,3,4,5,6,7,8],
            "phot_g_mean_mag": [15,16,17,14,18,15,16,14],
            "phot_bp_mean_mag": [15.5,16.3,17.1,14.5,18.2,15.4,16.2,14.1],
            "phot_rp_mean_mag": [14.5,15.2,16.1,13.5,17.5,14.7,15.8,13.9],
            "parallax": [1.2,0.8,2.0,1.5,0.5,1.1,0.9,1.3],
            "parallax_error": [0.05]*8,
            "teff_gspphot": [5800, 4500, 6000, 5500, 5000, 5700, 4300, 5900],
            "lum_flame": [1.0, 50.0, 2.0, 0.5, 100.0, 160.0, 20.0, 0.8],
            "radius_flame": [1.0, 20.0, 1.2, 0.9, 25.0, 160.0, 2.0, 0.8],
            "ag_gspphot": [0.0]*8,
            "ebpminrp_gspphot": [0.0]*8,
            "best_class_name": ["SOLAR_LIKE","ECL","DSCT|GDOR|SXPHE","RS","SOLAR_LIKE","LPV","ECL","SOLAR_LIKE"]
        })

# %% SAFE COLUMN VALIDATION
expected_cols = ["phot_bp_mean_mag", "phot_rp_mean_mag", "phot_g_mean_mag"]
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan

# %% SAFE PARALLAX CHECK
expected_parallax_cols = ["parallax", "parallax_error", "phot_g_mean_mag"]
for c in expected_parallax_cols:
    if c not in df.columns:
        df[c] = np.nan

# %% STEP 2: FEATURE EXPANSION & PREPROCESSING
st.subheader("Feature Expansion")

# Map Gaia DR3 columns to consistent internal names (if present)
gaia_column_mapping = {
    'lum_val': 'lum_flame',
    'radius_val': 'radius_flame',
    'a_g_val': 'ag_gspphot',
    'e_bp_min_rp_val': 'ebpminrp_gspphot'
}
for internal_col, gaia_col in gaia_column_mapping.items():
    if gaia_col in df.columns:
        df[internal_col] = df[gaia_col]
    else:
        if internal_col not in df.columns:
            df[internal_col] = np.nan

# Compute color index and absolute magnitude
df['bp_rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']

def compute_abs_mag(plx, plx_err, G):
    # parallax in mas; convert to distance (pc) as d = 1000/plx
    if pd.isna(plx) or pd.isna(G) or plx <= 0:
        return np.nan
    # require decent S/N on parallax
    if plx_err and plx_err > 0 and plx / plx_err > 5:
        d_pc = 1000.0 / plx
        return G - 5 * np.log10(d_pc) + 5
    return np.nan

df['abs_mag_g'] = df.apply(lambda r: compute_abs_mag(r.get('parallax', np.nan), r.get('parallax_error', np.nan), r.get('phot_g_mean_mag', np.nan)), axis=1)

# Drop rows missing essential features
essential_cols = ['bp_rp', 'abs_mag_g', 'teff_gspphot']
df_clean = df.dropna(subset=essential_cols).copy()

# Encode classes
if 'best_class_name' in df_clean.columns:
    df_clean['class_label'] = df_clean['best_class_name'].astype('category').cat.codes
    class_mapping = dict(enumerate(df_clean['best_class_name'].astype('category').cat.categories))
else:
    df_clean['class_label'] = pd.Series(dtype=int)
    class_mapping = {}

st.write("Class counts (original):")
st.write(df_clean['best_class_name'].value_counts() if 'best_class_name' in df_clean.columns else "No class column")

# Balance classes (resample to median)
counts = df_clean['class_label'].value_counts() if 'class_label' in df_clean else pd.Series(dtype=int)
median_count = int(counts.median()) if len(counts) > 0 else 0
dfs = []
for lbl, group in df_clean.groupby('class_label'):
    if len(group) < median_count and median_count > 0:
        group_up = resample(group, replace=True, n_samples=median_count, random_state=42)
        dfs.append(group_up)
    else:
        dfs.append(group)
df_balanced = pd.concat(dfs) if dfs else df_clean
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
st.write(f"Balanced dataset: {len(df_balanced)} stars total")

# Derived features (simple BC approximation)
def bolometric_correction(teff):
    if pd.isna(teff):
        return 0.0
    return -0.8 + (5800.0 - teff) / 5000.0

df_balanced['M_bol'] = df_balanced['abs_mag_g'] + df_balanced['teff_gspphot'].apply(bolometric_correction)
df_balanced['lum_ratio'] = 10 ** ((4.74 - df_balanced['M_bol']) / 2.5)
df_balanced['teff_lum_ratio'] = df_balanced['teff_gspphot'] * df_balanced['lum_ratio']

# unify possible Gaia AP columns into internal names for feature matrix
for src in ['lum_flame', 'lum_val', 'luminosity']:
    if src in df_balanced.columns:
        df_balanced['lum_val'] = df_balanced[src]
        break
for src in ['radius_flame', 'radius_gspphot', 'radius_val']:
    if src in df_balanced.columns:
        df_balanced['radius_val'] = df_balanced[src]
        break
for src in ['ag_gspphot', 'a_g_val', 'ag_msc', 'ag_esphs']:
    if src in df_balanced.columns:
        df_balanced['a_g_val'] = df_balanced[src]
        break
for src in ['ebpminrp_gspphot', 'ebpminrp_esphs', 'e_bp_min_rp_val']:
    if src in df_balanced.columns:
        df_balanced['e_bp_min_rp_val'] = df_balanced[src]
        break

# Feature matrix candidate (will be used later for training)
feature_candidates = [
    'bp_rp', 'abs_mag_g', 'teff_gspphot',
    'lum_val', 'radius_val', 'a_g_val', 'e_bp_min_rp_val',
    'M_bol', 'lum_ratio', 'teff_lum_ratio'
]
X_full = df_balanced[[c for c in feature_candidates if c in df_balanced.columns]].copy()

# Fill missing numeric values with median
for f in X_full.columns:
    X_full[f] = X_full[f].fillna(X_full[f].median() if not X_full[f].dropna().empty else 0.0)

st.write("Feature matrix preview:")
st.dataframe(X_full.head(10))

# %% STEP 2.2: CORRELATION ANALYSIS
st.subheader("Correlation analysis")

numeric_df = df_balanced.select_dtypes(include=[np.number]).copy()
label_col = 'class_label'
numeric_features = numeric_df.drop(columns=[label_col]) if label_col in numeric_df.columns else numeric_df.copy()

if numeric_features.shape[1] < 1 or numeric_features.shape[0] < 2:
    st.warning("Not enough numeric data to compute correlations.")
else:
    corr = numeric_features.corr(method='pearson')
    corr_file = output_folder / f"{(current_csv.stem if current_csv else 'dataset')}_correlation_matrix.csv"
    corr.to_csv(corr_file)
    st.write(f"Saved correlation matrix to `{corr_file}`")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(min(14, 0.5*corr.shape[0]+3), min(10, 0.35*corr.shape[1]+3)))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.6}, ax=ax_corr)
    ax_corr.set_title("Pearson correlation matrix (numeric features)")
    plt.tight_layout()
    st.pyplot(fig_corr)

    # Feature vs class scoring
    if label_col not in df_balanced.columns or df_balanced[label_col].nunique() < 2:
        st.info("No or insufficient class labels present — skipping feature→class scoring.")
        scores_df = pd.DataFrame()
    else:
        X_scores = numeric_features.fillna(numeric_features.median())
        y_labels = df_balanced[label_col].values

        try:
            f_vals, p_vals = f_classif(X_scores, y_labels)
        except Exception:
            f_vals = np.full(X_scores.shape[1], np.nan)
            p_vals = np.full(X_scores.shape[1], np.nan)

        try:
            mi_vals = mutual_info_classif(X_scores, y_labels, discrete_features=False, random_state=42)
        except Exception:
            mi_vals = np.full(X_scores.shape[1], np.nan)

        def eta_squared(feature_series, labels):
            data = pd.DataFrame({"x": feature_series, "g": labels}).dropna()
            if data.empty:
                return np.nan
            grand_mean = data['x'].mean()
            ss_between = (data.groupby('g').size() * (data.groupby('g')['x'].mean() - grand_mean)**2).sum()
            ss_total = ((data['x'] - grand_mean)**2).sum()
            return float(ss_between / ss_total) if ss_total > 0 else np.nan

        eta2 = [eta_squared(X_scores[col], y_labels) for col in X_scores.columns]

        scores_df = pd.DataFrame({
            "feature": X_scores.columns,
            "f_score": f_vals,
            "f_pvalue": p_vals,
            "mutual_info": mi_vals,
            "eta_squared": eta2
        }).set_index("feature")

        top_f = scores_df['f_score'].sort_values(ascending=False).head(10)
        top_mi = scores_df['mutual_info'].sort_values(ascending=False).head(10)
        top_eta = scores_df['eta_squared'].sort_values(ascending=False).head(10)

        st.markdown("**Top features by ANOVA F-score (higher = better separation):**")
        st.dataframe(top_f.to_frame("f_score"))

        st.markdown("**Top features by mutual information (non-parametric):**")
        st.dataframe(top_mi.to_frame("mutual_info"))

        st.markdown("**Top features by eta-squared (effect size):**")
        st.dataframe(top_eta.to_frame("eta_squared"))

        # Save reproducible scores JSON
        serializable_scores = {feat: {k: (None if pd.isna(v) else (v.item() if isinstance(v, (np.floating, np.integer)) else v)) for k, v in row.items()} for feat, row in scores_df.iterrows()}
        out = {
            "dataset": current_csv.name if current_csv else "mock",
            "computed_at": dt.datetime.utcnow().isoformat() + "Z",
            "n_samples": int(len(df_balanced)),
            "correlation_csv": str(corr_file),
            "feature_scores": serializable_scores
        }
        scores_file = output_folder / f"{(current_csv.stem if current_csv else 'dataset')}_feature_scores.json"
        with open(scores_file, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=safe_json_default, ensure_ascii=False)
        st.success(f"Feature scores exported to `{scores_file}`")

        def top_features_summary(scores_df_local, n=5):
            s = []
            s.append("Top features summary:")
            s.append(f" - By F-score: {', '.join(list(scores_df_local['f_score'].sort_values(ascending=False).head(n).index))}")
            s.append(f" - By mutual information: {', '.join(list(scores_df_local['mutual_info'].sort_values(ascending=False).head(n).index))}")
            s.append(f" - By eta-squared: {', '.join(list(scores_df_local['eta_squared'].sort_values(ascending=False).head(n).index))}")
            return "\n".join(s)

        st.text(top_features_summary(scores_df, n=7))

# %% STEP 3: FEATURES & LABELS (prepare minimal X for training)
st.subheader("Prepare features & labels for model training")
features_min = ['bp_rp', 'abs_mag_g', 'teff_gspphot']
X = df_balanced[[c for c in features_min if c in df_balanced.columns]].copy()
y = df_balanced['class_label'] if 'class_label' in df_balanced else pd.Series(dtype=int)
X['teff_gspphot'] = X['teff_gspphot'].fillna(X['teff_gspphot'].median() if not X['teff_gspphot'].dropna().empty else 0.0)

st.write("Training feature preview:")
st.dataframe(X.head(8))

# %% STEP 4: SAFE TRAINING & GUI
st.subheader("Train classifier (safe defaults)")
clf = None
X_train = X_test = y_train = y_test = None

if y.empty:
    st.warning("No labels available — cannot train a classifier.")
else:
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    X_valid = X[y.isin(valid_classes)]
    y_valid = y[y.isin(valid_classes)]

    if len(y_valid) >= 2 and y_valid.nunique() >= 2:
        n_classes = y_valid.nunique()
        n_samples = len(y_valid)
        test_size = max(n_classes, int(0.2 * n_samples))
        if test_size >= n_samples:
            test_size = max(1, n_samples - 1)
        stratify_param = y_valid if test_size >= n_classes else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, stratify=stratify_param, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        st.success(f"Classifier trained on {len(X_train)} samples, test size {len(X_test)}")
    else:
        st.warning("Not enough valid data to train a classifier. Preview only available.")

# %% STEP 5: HYPERPARAMETER OPTIMIZATION (optional)
if clf is not None and X_train is not None and len(X_train) >= 2:
    cv_folds = min(5, len(X_train))
    if cv_folds >= 2:
        st.subheader("RandomForest Hyperparameter Optimization (GridSearchCV)")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        with st.spinner("Running GridSearchCV for RandomForest optimization..."):
            try:
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=42, class_weight='balanced'),
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                clf = grid_search.best_estimator_
                st.success(f"Best hyperparameters found: {grid_search.best_params_}")
                # Evaluate performance on test set
                y_pred_opt = clf.predict(X_test)
                acc_opt = accuracy_score(y_test, y_pred_opt)
                f1_macro_opt = f1_score(y_test, y_pred_opt, average="macro")
                f1_weighted_opt = f1_score(y_test, y_pred_opt, average="weighted")
                st.markdown(f"**Optimized performance (test set):** Accuracy: {acc_opt:.3f}  Macro F1: {f1_macro_opt:.3f}  Weighted F1: {f1_weighted_opt:.3f}")
            except Exception as e:
                st.error(f"Hyperparameter search failed: {e}")
    else:
        st.info("Not enough training samples for CV-based hyperparameter tuning.")

# %% STEP 6: EVALUATION & REPORT
if clf is not None and X_test is not None:
    y_pred = clf.predict(X_test)
    unique_classes = sorted(set(y_test) | set(y_pred))
    target_names = [class_mapping.get(i, f"class_{i}") for i in unique_classes]

    report_dict = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.style.format("{:.2f}", subset=['precision','recall','f1-score']))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    fig_cm, ax_cm = plt.subplots(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
else:
    st.info("No classifier trained yet — skipping evaluation.")

# %% STEP 7: PUBLICATION-QUALITY HR DIAGRAMS (3-panel)
st.subheader("HR Diagrams")

def first_existing_col(df_local, names):
    for n in names:
        if n in df_local.columns:
            return n
    return None

radius_col = first_existing_col(df_balanced, ["radius_flame","radius_gspphot","radius_val"])
lum_col = first_existing_col(df_balanced, ["lum_flame","lum_val","luminosity"])
bp_col = first_existing_col(df_balanced, ["phot_bp_mean_mag"])
rp_col = first_existing_col(df_balanced, ["phot_rp_mean_mag"])
g_col = first_existing_col(df_balanced, ["phot_g_mean_mag"])

if bp_col is None or rp_col is None or 'abs_mag_g' not in df_balanced.columns:
    st.error("Cannot draw HR diagram: missing BP/RP or absolute magnitude columns.")
else:
    df_plot = df_balanced.copy()
    df_plot['bp_rp'] = df_plot[bp_col] - df_plot[rp_col]
    if radius_col:
        df_plot['radius_val_plot'] = df_plot[radius_col].replace([np.inf, -np.inf], np.nan)
        df_plot['log_radius'] = np.log10(df_plot['radius_val_plot'].replace(0, np.nan))
    else:
        df_plot['log_radius'] = np.nan

    # classifier confidence sizes
    try:
        if clf is not None and hasattr(clf, "predict_proba"):
            clf_features = getattr(clf, "feature_names_in_", None)
            if clf_features is not None and set(clf_features).issubset(df_plot.columns):
                X_for_prob = df_plot[clf_features].copy().fillna(0)
            else:
                fallback = [c for c in ['bp_rp','abs_mag_g','teff_gspphot'] if c in df_plot.columns]
                X_for_prob = df_plot[fallback].copy().fillna(df_plot[fallback].median())
            probs = clf.predict_proba(X_for_prob)
            df_plot['class_conf'] = probs.max(axis=1)
        else:
            df_plot['class_conf'] = np.nan
    except Exception:
        df_plot['class_conf'] = np.nan

    x = df_plot['bp_rp'].replace([np.inf,-np.inf], np.nan).dropna()
    y = df_plot['abs_mag_g'].replace([np.inf,-np.inf], np.nan).dropna()
    x_min, x_max = np.percentile(x, [1,99])
    y_min, y_max = np.percentile(y, [1,99])
    dx = (x_max - x_min) * 0.05 if x_max != x_min else 0.5
    dy = (y_max - y_min) * 0.05 if y_max != y_min else 0.5

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True, gridspec_kw={"width_ratios":[1,1,0.9]})
    ax_class, ax_radius, ax_prob = axes

    # --------------------
    # Class-colored HR diagram with density contour
    # --------------------
    unique_codes = sorted(df_plot['class_label'].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {code: cmap(i % 20) for i, code in enumerate(unique_codes)}
    for class_code in unique_codes:
        subset = df_plot[df_plot['class_label'] == class_code]
        if subset.empty:
            continue
        ax_class.scatter(subset['bp_rp'], subset['abs_mag_g'], s=18, alpha=0.65,
                         label=class_mapping.get(class_code, f"class_{class_code}"),
                         color=color_map[class_code], edgecolors="none")

    # density contour
    try:
        import seaborn as sns
        sns.kdeplot(
            x=df_plot['bp_rp'], 
            y=df_plot['abs_mag_g'], 
            ax=ax_class, 
            levels=10,
            color="black",
            linewidths=1.0,
            alpha=0.6
        )
    except Exception:
        pass

    # optional hexbin background
    try:
        ax_class.hexbin(df_plot['bp_rp'], df_plot['abs_mag_g'], gridsize=160, cmap="Greys", mincnt=1, alpha=0.25)
    except Exception:
        pass

    ax_class.invert_yaxis()
    ax_class.set_xlim(x_min - dx, x_max + dx)
    ax_class.set_ylim(y_max + dy, y_min - dy)
    ax_class.set_xlabel("BP - RP (color index)")
    ax_class.set_ylabel("M_G (absolute magnitude)")
    ax_class.set_title("HR Diagram — colored by class")
    ax_class.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", frameon=False)

    # --------------------
    # Radius/Luminosity-colored HR diagram
    # --------------------
    if 'log_radius' in df_plot.columns and df_plot['log_radius'].notna().any():
        sc = ax_radius.scatter(df_plot['bp_rp'], df_plot['abs_mag_g'], c=df_plot['log_radius'], cmap="magma", s=25, alpha=0.85, edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax_radius, fraction=0.046, pad=0.04)
        cbar.set_label("log₁₀(R / R☉)")
    elif lum_col and lum_col in df_plot.columns and df_plot[lum_col].notna().any():
        df_plot['log_lum'] = np.log10(df_plot[lum_col].replace(0, np.nan))
        sc = ax_radius.scatter(df_plot['bp_rp'], df_plot['abs_mag_g'], c=df_plot['log_lum'], cmap="viridis", s=25, alpha=0.85, edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax_radius, fraction=0.046, pad=0.04)
        cbar.set_label("log₁₀(L / L☉)")
    else:
        ax_radius.scatter(df_plot['bp_rp'], df_plot['abs_mag_g'], s=20, alpha=0.6)

    # 1-sigma ellipses for giant classes
    try:
        if radius_col and 'radius_val_plot' in df_plot.columns:
            for class_code in unique_codes:
                subset = df_plot[df_plot['class_label'] == class_code]
                if subset.empty or subset['radius_val_plot'].dropna().empty:
                    continue
                med = subset['radius_val_plot'].median()
                if med >= 5.0 and subset[['bp_rp','abs_mag_g']].dropna().shape[0] >= 3:
                    vals = subset[['bp_rp','abs_mag_g']].dropna().values
                    cov = np.cov(vals, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
                    width, height = 2 * np.sqrt(eigvals)
                    mean_x, mean_y = np.mean(vals, axis=0)
                    from matplotlib.patches import Ellipse
                    ell = Ellipse((mean_x, mean_y), width=width*2, height=height*2, angle=angle,
                                  edgecolor=color_map.get(class_code,"k"), facecolor='none', lw=1.0, alpha=0.9)
                    ax_radius.add_patch(ell)
                    ax_radius.text(mean_x, mean_y, class_mapping.get(class_code, str(class_code)), fontsize=8)
    except Exception:
        pass

    ax_radius.invert_yaxis()
    ax_radius.set_xlim(x_min - dx, x_max + dx)
    ax_radius.set_ylim(y_max + dy, y_min - dy)
    ax_radius.set_xlabel("BP - RP (color index)")
    ax_radius.set_title("HR Diagram — colored by radius / luminosity")

    # --------------------
    # Probability-sized panel
    # --------------------
    if 'class_conf' in df_plot.columns and df_plot['class_conf'].notna().any():
        sizes = df_plot['class_conf'].fillna(df_plot['class_conf'].median())
        rng = sizes.max() - sizes.min()
        if rng is None or pd.isna(rng) or rng <= 0:
            sizes_scaled = np.full(len(sizes), 50.0)
        else:
            sizes_scaled = 10.0 + (sizes - sizes.min()) / rng * 160.0

        hb2 = ax_prob.hexbin(df_plot['bp_rp'], df_plot['abs_mag_g'], gridsize=120, cmap="Greys", mincnt=1, alpha=0.3)
        sc2 = ax_prob.scatter(df_plot['bp_rp'], df_plot['abs_mag_g'],
                              s=sizes_scaled, c=df_plot['class_conf'],
                              cmap="plasma", alpha=0.9, edgecolors='none')
        cbar2 = fig.colorbar(sc2, ax=ax_prob, fraction=0.046, pad=0.04)
        cbar2.set_label("Classifier confidence (max class prob)")
        ax_prob.set_title("HR Diagram — point size = classifier confidence")

        ax_prob.invert_yaxis()
        ax_prob.set_xlim(x_min - dx, x_max + dx)
        ax_prob.set_ylim(y_max + dy, y_min - dy)
        ax_prob.set_xlabel("BP - RP (color index)")

    plt.tight_layout()
    st.pyplot(fig)

    # Download and save HR diagram
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download HR diagram (PNG)", data=buf, file_name=f"{(current_csv.stem if current_csv else 'hr')}_hr_diagrams.png", mime="image/png")
    hr_diagram_path = output_folder / f"{(current_csv.stem if current_csv else 'hr')}_hr_diagrams.png"
    fig.savefig(hr_diagram_path, format="png", dpi=200, bbox_inches="tight")


# === RESEARCH LOG (Step 3.2) ===
log_file = output_folder / "research_log.md"
if not log_file.exists():
    with open(log_file, "w") as f:
        f.write("# DataNova Research Log\n\n")
with open(log_file, "a") as f:
    f.write(f"## Run on {dt.datetime.utcnow().isoformat()}Z\n")
    gaia_rel = st.session_state.get("gaia_release", "N/A")
    sample_size_meta = st.session_state.get("sample_size", "N/A")
    f.write(f"**Gaia release:** {gaia_rel}, sample size: {sample_size_meta}\n\n")
    query = st.session_state.get("last_query", "N/A")
    f.write(f"**Query:** \n```\n{query}\n```\n")
    fetch_ts = st.session_state.get("fetch_timestamp", None) or "N/A"
    f.write(f"**Fetch timestamp:** {fetch_ts}\n\n")
    try:
        acc = accuracy_score(y_test, y_pred)
        f.write("### Classifier performance\n")
        f.write(f"- Accuracy: {acc:.3f}\n")
        f.write(f"- Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}\n")
        f.write(f"- Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.3f}\n\n")
    except Exception:
        f.write("### Classifier performance: N/A (no valid classifier)\n\n")
    hr_file_name = f"{(current_csv.stem if current_csv else 'hr')}_hr_diagrams.png"
    f.write("### HR Diagram\n")
    f.write(f"![HR Diagram]({hr_file_name})\n")

# %% STEP 8: FILTERING (UI)
st.subheader("Filter stars")
dataset_version = st.session_state.get("dataset_version", 0)
filters_key_prefix = f"filters_v{dataset_version}_"
available_classes = list(df_clean['best_class_name'].unique()) if not df_clean.empty and 'best_class_name' in df_clean.columns else []
star_class = st.selectbox("Select class to filter:", options=[None] + available_classes, key=filters_key_prefix + "class")

min_default = float(df_clean['abs_mag_g'].min()) if not df_clean.empty else 0.0
max_default = float(df_clean['abs_mag_g'].max()) if not df_clean.empty else 0.0
min_mag = st.number_input("Minimum absolute magnitude (M_G):", value=min_default, key=filters_key_prefix + "min_mag")
max_mag = st.number_input("Maximum absolute magnitude (M_G):", value=max_default, key=filters_key_prefix + "max_mag")

def filter_stars_gui(df_, star_class=None, min_mag=None, max_mag=None):
    filtered = df_.copy()
    if star_class:
        filtered = filtered[filtered['best_class_name'] == star_class]
    if min_mag is not None:
        filtered = filtered[filtered['abs_mag_g'] >= min_mag]
    if max_mag is not None:
        filtered = filtered[filtered['abs_mag_g'] <= max_mag]
    cols = ['source_id','best_class_name','bp_rp','abs_mag_g','teff_gspphot']
    return filtered[cols] if all(c in filtered.columns for c in cols) else filtered

filtered_df = filter_stars_gui(df_clean, star_class, min_mag, max_mag)
st.write(f"Filtered stars: {len(filtered_df)}")
st.dataframe(filtered_df.head(20))

# === EXPORT with metadata (Step 3.1) ===
if st.button("Export CSVs", key="export_btn"):
    import json as _json

    stem = (current_csv.stem if current_csv else "gaia_mock")
    original_csv_path = output_folder / f"{stem}_original.csv"
    filtered_csv_path = output_folder / f"{stem}_filtered.csv"

    df_clean.to_csv(original_csv_path, index=False)
    filtered_df.to_csv(filtered_csv_path, index=False)

    # fetch timestamp: prefer session_state else file mtime else now
    if st.session_state.get("fetch_timestamp"):
        fetch_ts = st.session_state["fetch_timestamp"]
    elif current_csv and current_csv.exists():
        fetch_ts = dt.datetime.utcfromtimestamp(current_csv.stat().st_mtime).isoformat() + "Z"
    else:
        fetch_ts = dt.datetime.utcnow().isoformat() + "Z"

    last_query = st.session_state.get("last_query", None)
    gaia_release = st.session_state.get("gaia_release", "gaiadr3")
    sample_size_meta = int(st.session_state.get("sample_size", int(sample_size)))

    classification_features = None
    try:
        if clf is not None and hasattr(clf, "feature_names_in_"):
            classification_features = list(getattr(clf, "feature_names_in_"))
    except Exception:
        classification_features = None
    if not classification_features:
        if 'X' in locals() and hasattr(X, "columns"):
            classification_features = list(X.columns)
        else:
            classification_features = [c for c in ['bp_rp', 'abs_mag_g', 'teff_gspphot'] if c in df_clean.columns]

    columns_in_dataset = list(df_clean.columns)
    class_map = class_mapping if 'class_mapping' in locals() else {}

    metadata = {
        "dataset_stem": stem,
        "exported_at": dt.datetime.utcnow().isoformat() + "Z",
        "fetch_timestamp": fetch_ts,
        "query": last_query or "N/A",
        "gaia_release": gaia_release,
        "sample_size_requested": sample_size_meta,
        "n_rows_original": int(len(df_clean)),
        "n_rows_filtered": int(len(filtered_df)),
        "classification_features": classification_features,
        "columns_in_dataset": columns_in_dataset,
        "class_mapping": class_map,
        "notes": "Preprocessing: dropped NaNs on essential columns, resampled minority classes to median, computed M_bol/lum_ratio heuristically.",
        "random_seed": 42
    }

    metadata_file = output_folder / f"{stem}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as fh:
        _json.dump(metadata, fh, default=safe_json_default, indent=2, ensure_ascii=False)

    st.success(
        f"CSVs exported:\n- Original: {original_csv_path}\n- Filtered: {filtered_csv_path}\n- Metadata: {metadata_file}"
    )
