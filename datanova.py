
# %% DEPENDENCY CHECK
import importlib
import sys
required_packages = ["pandas","numpy","matplotlib","sklearn","streamlit","astroquery"]
missing = []

for pkg in required_packages:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    import streamlit as st
    st.error(
        f"The following required packages are missing:\n{', '.join(missing)}\n\n"
        "Please install them and restart the app."
    )
    sys.exit("Dependencies missing. Exiting.")

# %% IMPORTS
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
import streamlit as st

# Optional Gaia query
try:
    from astroquery.gaia import Gaia
    GAIA_AVAILABLE = True
except:
    GAIA_AVAILABLE = False

# %% STREAMLIT GUI
st.title("DataNova: Gaia Stellar Classification")
st.markdown("""
Fetch Gaia star samples, train a RandomForest classifier, filter stars, and export CSVs.  
**Note:** Larger samples give better classifier results.
""")

# --- User selects sample size
sample_size = st.number_input(
    "Number of stars to fetch/analyze:", 
    min_value=1000, max_value=100000, value=20000, step=1000
)

# --- User selects folder to save CSVs
st.markdown("**Folder to save CSVs (original + filtered):**")
st.markdown("_Tip: Type/paste a path, default is `./data`._")
output_folder = Path(st.text_input("Folder path:", value=str(Path.cwd() / "data")))
output_folder.mkdir(parents=True, exist_ok=True)
st.write(f"CSV files will be saved to: `{output_folder}`")

# %% STEP 1: FETCH OR LOAD DATA
SAMPLE_CSV = output_folder / "gaia_sample.csv"

if GAIA_AVAILABLE and st.button("Fetch Gaia Sample"):
    st.write("Fetching Gaia sample...")
    base_name = "gaia_sample"
    extension = ".csv"
    sample_num = 0
    while (output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}").exists():
        sample_num += 1
    output_file = output_folder / f"{base_name}{sample_num if sample_num > 0 else ''}{extension}"
    try:
        query = f"""
        SELECT TOP {sample_size}
            v.source_id,
            g.ra, g.dec,
            g.phot_g_mean_mag,
            g.phot_bp_mean_mag,
            g.phot_rp_mean_mag,
            g.parallax, g.parallax_error,
            g.teff_gspphot,
            v.best_class_name
        FROM gaiadr3.vari_classifier_result AS v
        JOIN gaiadr3.gaia_source AS g
            ON v.source_id = g.source_id
        WHERE v.best_class_name IS NOT NULL
        """
        job = Gaia.launch_job(query)
        df = job.get_results().to_pandas()
        df.to_csv(output_file, index=False)
        st.success(f"Gaia sample fetched and saved as {output_file.name}, rows: {len(df)}")
    except Exception as e:
        st.error(f"Gaia query failed: {e}")
        df = None

# Load CSV if exists
if SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
    st.write(f"Loaded CSV `{SAMPLE_CSV.name}`, rows: {len(df)}")
elif 'df' not in locals():
    st.warning("No Gaia data available. Using fallback mock data.")
    df = pd.DataFrame({
        "source_id": [1,2,3,4,5,6,7,8],
        "phot_g_mean_mag": [15,16,17,14,18,15,16,14],
        "phot_bp_mean_mag": [15.5,16.3,17.1,14.5,18.2,15.4,16.2,14.1],
        "phot_rp_mean_mag": [14.5,15.2,16.1,13.5,17.5,14.7,15.8,13.9],
        "parallax": [1.2,0.8,2.0,1.5,0.5,1.1,0.9,1.3],
        "parallax_error": [0.05]*8,
        "teff_gspphot": [5800, 4500, 6000, 5500, 5000, 5700, 4300, 5900],
        "best_class_name": ["SOLAR_LIKE","ECL","DSCT|GDOR|SXPHE","RS","SOLAR_LIKE","LPV","ECL","SOLAR_LIKE"]
    })

# %% STEP 2: PREPROCESSING
df['bp_rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']

def compute_abs_mag(row):
    plx, plx_err, G = row['parallax'], row['parallax_error'], row['phot_g_mean_mag']
    if pd.isna(plx) or pd.isna(G) or plx <= 0:
        return np.nan
    if plx / plx_err > 5:
        return G + 5 * np.log10(plx / 1000) + 5
    return np.nan

df['abs_mag_g'] = df.apply(compute_abs_mag, axis=1)
df_clean = df.dropna(subset=['bp_rp','abs_mag_g'])
df_clean['class_label'] = df_clean['best_class_name'].astype('category').cat.codes
class_mapping = dict(enumerate(df_clean['best_class_name'].astype('category').cat.categories))

st.subheader("Class counts (original):")
st.write(df_clean['best_class_name'].value_counts())

# %% STEP 2b: BALANCE CLASSES
counts = df_clean['class_label'].value_counts()
median_count = int(counts.median())
dfs = []
for lbl, group in df_clean.groupby('class_label'):
    if len(group) < median_count:
        group_up = resample(group, replace=True, n_samples=median_count, random_state=42)
        dfs.append(group_up)
    else:
        dfs.append(group)
df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
st.write(f"Balanced dataset: {len(df_balanced)} stars total")

# %% STEP 3: FEATURES & LABELS
features = ['bp_rp','abs_mag_g','teff_gspphot']
X = df_balanced[features].copy()
y = df_balanced['class_label']
X['teff_gspphot'] = X['teff_gspphot'].fillna(X['teff_gspphot'].median())

# %% STEP 4: SAFE TRAINING & GUI
clf = None
X_train = X_test = y_train = y_test = None
trainable = False

counts = y.value_counts()
valid_classes = counts[counts >= 2].index
X_valid = X[y.isin(valid_classes)]
y_valid = y[y.isin(valid_classes)]

if len(y_valid) >= 2 and y_valid.nunique() >= 2:
    trainable = True
    # Ensure test_size >= number of classes but < total samples
    n_classes = y_valid.nunique()
    n_samples = len(y_valid)
    test_size = max(n_classes, int(0.2 * n_samples))
    if test_size >= n_samples:
        test_size = n_samples - 1
    stratify_param = y_valid if test_size >= n_classes else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=test_size, stratify=stratify_param, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    st.success(f"Classifier trained on {len(X_train)} samples, test size {len(X_test)}")
else:
    st.warning("Not enough valid data to train a classifier. Preview only available.")

# %% STEP 6: EVALUATION (safe)
if clf is not None:
    y_pred = clf.predict(X_test)
    unique_classes = sorted(set(y_test) | set(y_pred))
    target_names = [class_mapping[i] for i in unique_classes]

    report_dict = classification_report(y_test, y_pred, target_names=target_names,
                                        zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df[['precision','recall','f1-score']] = report_df[['precision','recall','f1-score']].round(2)

    def color_scores(val):
        if isinstance(val, float):
            if val >= 0.9: return 'background-color:#c6efce'
            elif val >= 0.7: return 'background-color:#ffeb9c'
            else: return 'background-color:#f4c7c3'
        return ''

    st.subheader("Classification Report (Styled)")
    st.dataframe(report_df.style.applymap(color_scores, subset=['precision','recall','f1-score']))
else:
    st.info("No classifier trained yet — skipping classification report.")

# %% STEP 7: HR DIAGRAM (safe)
import matplotlib
matplotlib.use("Agg")
fig, ax = plt.subplots(figsize=(8,10))

if clf is not None:
    predicted_labels = clf.predict(X)
    for class_code, class_name in class_mapping.items():
        idx = predicted_labels == class_code
        ax.scatter(df_balanced['bp_rp'][idx], df_balanced['abs_mag_g'][idx],
                   label=class_name, s=20, alpha=0.7)
else:
    ax.scatter(df['bp_rp'], df['abs_mag_g'], color='gray', s=20, alpha=0.7)
    ax.set_title("HR Diagram (preview, classifier not trained)")

ax.invert_yaxis()
ax.set_xlabel('BP - RP (color index)')
ax.set_ylabel('M_G (absolute magnitude)')
ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# %% STEP 8: FILTERING
st.subheader("Filter stars")
star_class = st.selectbox(
    "Select class to filter:",
    options=[None]+list(df_clean['best_class_name'].unique())
)
min_mag = st.number_input(
    "Minimum absolute magnitude (M_G):",
    value=float(df_clean['abs_mag_g'].min())
)
max_mag = st.number_input(
    "Maximum absolute magnitude (M_G):",
    value=float(df_clean['abs_mag_g'].max())
)

def filter_stars_gui(df, star_class=None, min_mag=None, max_mag=None):
    filtered = df.copy()
    if star_class:
        filtered = filtered[filtered['best_class_name'] == star_class]
    if min_mag is not None:
        filtered = filtered[filtered['abs_mag_g'] >= min_mag]
    if max_mag is not None:
        filtered = filtered[filtered['abs_mag_g'] <= max_mag]
    return filtered[['source_id','best_class_name','bp_rp','abs_mag_g','teff_gspphot']]

filtered_df = filter_stars_gui(df_clean, star_class, min_mag, max_mag)
st.write(f"Filtered stars: {len(filtered_df)}")
st.dataframe(filtered_df.head(20))

# %% STEP 9: EXPORT
if st.button("Export CSVs"):
    original_csv_path = output_folder / "gaia_original.csv"
    filtered_csv_path = output_folder / "gaia_filtered.csv"
    df_clean.to_csv(original_csv_path, index=False)
    filtered_df.to_csv(filtered_csv_path, index=False)
    st.success(f"CSVs exported:\n- Original: {original_csv_path}\n- Filtered: {filtered_csv_path}")
