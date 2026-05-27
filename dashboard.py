import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spatial Transcriptomics Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title  { font-size:2rem; font-weight:700; color:#c0392b; margin-bottom:0.2rem; }
    .sub-title   { font-size:0.9rem; color:#777; margin-bottom:1rem; }
    .metric-card { background:#f8f9fa; border-radius:10px; padding:12px 16px;
                   border-left:4px solid #2980b9; margin-bottom:10px; }
    .metric-label{ font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:0.05em; }
    .metric-value{ font-size:1.5rem; font-weight:700; color:#2c2c2c; }
    .exhausted-badge { background:#fdecea; color:#c0392b; border-radius:20px;
                       padding:4px 16px; font-weight:700; font-size:1rem; display:inline-block; }
    .active-badge    { background:#e8f5ea; color:#1a7a34; border-radius:20px;
                       padding:4px 16px; font-weight:700; font-size:1rem; display:inline-block; }
    .section-header  { font-size:1.05rem; font-weight:600; color:#2c2c2c;
                       border-bottom:2px solid #eee; padding-bottom:4px; margin-bottom:10px; }
    .info-box  { background:#e8f1fb; border-radius:8px; padding:10px 14px;
                 font-size:0.87rem; color:#1a5276; margin-bottom:8px; }
    .warn-box  { background:#fef4e8; border-radius:8px; padding:10px 14px;
                 font-size:0.87rem; color:#7d4e00; margin-bottom:8px; }
    .success-box { background:#e8f5ea; border-radius:8px; padding:10px 14px;
                   font-size:0.87rem; color:#1a7a34; margin-bottom:8px; }
    .step-box  { background:#f4f4f4; border-radius:8px; padding:12px 16px;
                 font-size:0.87rem; margin-bottom:8px; border-left:4px solid #c0392b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    'adata':           None,   # loaded AnnData object
    'cell_obs':        None,   # computed cell dataframe
    'cell_type_name':  None,   # selected cell type label
    'score_name':      None,   # score column name
    'col_map':         {},     # column name mapping
    'data_path':       '',     # file path entered by user
    'data_loaded':     False,  # whether adata is loaded
    'pipeline_run':    False,  # whether pipeline has been run
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# LOAD SAVED CD8 MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load('dashboard_model.pkl')
    except Exception:
        return None

@st.cache_data
def load_cd8_reference():
    try:
        cells = pd.read_csv('dashboard_cells.csv', index_col=0)
        return cells
    except Exception:
        return None

rf_model  = load_model()
cd8_ref   = load_cd8_reference()


# ─────────────────────────────────────────────
# HELPER — check pipeline has been run
# ─────────────────────────────────────────────
def need_pipeline():
    if not st.session_state.pipeline_run:
        st.warning("Pipeline not run yet. Go to **⚙️ Run Pipeline** first.")
        if cd8_ref is not None:
            if st.button("Or load saved CD8+ T cell data (demo)"):
                st.session_state.cell_obs       = cd8_ref.copy()
                st.session_state.cell_type_name = "CD8+ T cells"
                st.session_state.score_name     = "exhaustion_score"
                st.session_state.pipeline_run   = True
                st.rerun()
        return True
    return False


# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────
def compute_spatial_features(data_obs, cell_mask, neighbour_mask,
                              cell_obs, col_map, radius=150, k=20):
    """
    Compute nearest neighbour distance, radius count and KNN fraction
    for any cell type using KDTree.
    """
    x_col      = col_map.get('x', 'x')
    y_col      = col_map.get('y', 'y')
    sample_col = col_map.get('sample', 'samples')
    cell_type_col = col_map.get('cell_type', 'cell.subtypes')

    cell_obs = cell_obs.copy()
    cell_obs["nearest_neighbour_dist"]     = np.nan
    cell_obs[f"neighbour_count_r{radius}"] = np.nan
    cell_obs[f"neighbour_frac_knn_{k}"]    = np.nan

    n_dists  = np.full(len(cell_obs), np.nan)
    r_counts = np.full(len(cell_obs), np.nan)
    knn_f    = np.full(len(cell_obs), np.nan)

    neighbour_labels = set(data_obs.loc[neighbour_mask, cell_type_col].unique())
    samples          = cell_obs[sample_col].unique()

    prog = st.progress(0, text="Computing spatial features...")

    for si, samp in enumerate(samples):
        prog.progress((si + 1) / len(samples),
                      text=f"Processing sample {si+1} of {len(samples)}: {samp}")

        cell_in_samp   = (cell_obs[sample_col] == samp).values
        cell_positions = np.where(cell_in_samp)[0]

        nb_in_samp = (data_obs[sample_col] == samp) & neighbour_mask
        nb_xy      = data_obs.loc[nb_in_samp, [x_col, y_col]].values.astype(float)

        if len(cell_positions) == 0 or len(nb_xy) == 0:
            continue

        cell_xy = cell_obs.iloc[cell_positions][[x_col, y_col]].values.astype(float)
        tree    = cKDTree(nb_xy)

        # Nearest distance
        dists, _ = tree.query(cell_xy, k=1)
        n_dists[cell_positions] = dists

        # Radius count
        counts = tree.query_ball_point(cell_xy, r=radius)
        r_counts[cell_positions] = [len(c) for c in counts]

        # KNN fraction
        all_in_samp = data_obs[sample_col] == samp
        all_xy      = data_obs.loc[all_in_samp, [x_col, y_col]].values.astype(float)
        all_types   = data_obs.loc[all_in_samp, cell_type_col].values

        if len(all_xy) > k + 1:
            tree_all = cKDTree(all_xy)
            _, idx   = tree_all.query(cell_xy, k=k+1)
            idx      = idx[:, 1:]
            for i, pos in enumerate(cell_positions):
                nn = sum(1 for j in idx[i] if all_types[j] in neighbour_labels)
                knn_f[cell_positions[i]] = nn / k

    prog.empty()

    cell_obs["nearest_neighbour_dist"]     = n_dists
    cell_obs[f"neighbour_count_r{radius}"] = r_counts
    cell_obs[f"neighbour_frac_knn_{k}"]    = knn_f

    return cell_obs


def compute_expression_score(adata, cell_mask, cell_obs,
                              markers, score_name="score"):
    """
    Extract marker expression values and compute z-scored composite score.
    Always extracts CD47 and THBS1 for use in Predict Cell State.
    """
    import scipy.sparse as sp
    cell_obs  = cell_obs.copy()

    # Always include CD47 and THBS1 for Predict Cell State, even if not in markers
    genes_to_extract = list(markers) + [g for g in ['CD47', 'THBS1'] if g not in markers]

    available = [m for m in genes_to_extract if m in adata.var_names]
    score_markers = [m for m in markers if m in adata.var_names]
    missing   = [m for m in markers if m not in adata.var_names]

    if missing:
        st.warning(f"These markers were not found in the dataset: {missing}")
    if not score_markers:
        st.error("No markers found. Check your gene names.")
        return cell_obs

    idx = [list(adata.var_names).index(m) for m in available]
    X = adata.X[np.where(cell_mask.values)[0], :][:, idx]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    df_m = pd.DataFrame(X, columns=available, index=cell_obs.index)
    for m in available:
        cell_obs[m] = df_m[m].values

    # Z-score each gene then average — only over the user-selected score markers
    score_cols = [m for m in score_markers if m in df_m.columns]
    df_z = (df_m[score_cols] - df_m[score_cols].mean()) / df_m[score_cols].std()
    cell_obs[score_name] = df_z.mean(axis=1).values

    return cell_obs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Spatial Transcriptomics")
    st.markdown("*General Purpose Analyser*")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Home",
        "📂 Load Data",
        "⚙️ Run Pipeline",
        "🔮 Predict Cell State",
        "🗺️ Spatial Plot",
        "📊 Feature Importance",
        "👥 Patient Clinical & Treatment Analysis",
        "📈 Exhaustion Threshold",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Status indicators
    st.markdown("**Status**")
    if st.session_state.data_loaded:
        adata = st.session_state.adata
        st.markdown(f"✅ Data loaded")
        st.markdown(f"- {adata.n_obs:,} cells")
        st.markdown(f"- {adata.n_vars:,} genes")
    else:
        st.markdown("❌ No data loaded")

    if st.session_state.pipeline_run:
        st.markdown(f"✅ Pipeline complete")
        st.markdown(f"- **{st.session_state.cell_type_name}**")
        st.markdown(f"- {len(st.session_state.cell_obs):,} cells")
    else:
        st.markdown("❌ Pipeline not run")

    if rf_model is not None:
        st.markdown("✅ ML model loaded")
    else:
        st.markdown("❌ ML model not found")


# ═══════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🔬 Spatial Transcriptomics Analyser</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A general purpose tool for analysing T cell spatial organisation '
                'in cancer spatial transcriptomics data</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-header">What does this app do?</div>', unsafe_allow_html=True)
    st.markdown("""
    This app takes any spatial transcriptomics dataset in `.h5ad` format and allows you to:

    - Compute spatial distances between any two cell types using KDTree
    - Calculate gene expression scores (e.g. exhaustion score) for any cell type
    - Visualise cell positions on the tissue map
    - Compare exhaustion patterns across clinical groups
    - Predict cell functional states using a trained machine learning model
    - Explore patient-level differences interactively
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">How to use this app — follow these steps in order</div>',
                unsafe_allow_html=True)

    steps = [
        ("1", "📂 Load Data",
         "Go to Load Data page → enter the path to your .h5ad file → map your column names → click Load"),
        ("2", "⚙️ Run Pipeline",
         "Go to Run Pipeline → select cell type and neighbour type → choose markers → click Run Pipeline"),
        ("3", "🗺️ Spatial Plot",
         "Explore T cell positions on the tissue coloured by exhaustion score or distance"),
        ("4", "📈 Exhaustion Threshold",
         "See at how many cancer neighbours exhaustion kicks in — split by clinical group"),
        ("5", "👥 Patient Explorer",
         "Compare patients side by side — boxplots by treatment group, scatter of patient-level patterns"),
        ("6", "🔮 Predict Cell State",
         "Enter feature values manually and predict whether a cell is Exhausted or Active"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""<div class="step-box">
            <strong>Step {num} — {title}</strong><br>
            <span style="color:#555;">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Compatible datasets</div>', unsafe_allow_html=True)
    st.markdown("""
    This app works with any `.h5ad` file that contains:
    - **Spatial coordinates** (x, y columns in `.obs`)
    - **Cell type annotations** (a column in `.obs`)
    - **Sample and patient identifiers** (columns in `.obs`)
    - **Gene expression matrix** (`.X`)

    Examples: MERSCOPE, Visium, MERFISH, seqFISH, Xenium datasets
    """)

    st.markdown('<div class="info-box">📌 This app was built for the Yeh et al. (2024) ovarian cancer '
                'spatial transcriptomics dataset but works with any compatible .h5ad file.</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 2 — LOAD DATA
# ═══════════════════════════════════════════
elif page == "📂 Load Data":
    st.markdown('<div class="main-title">📂 Load Data</div>', unsafe_allow_html=True)
    st.markdown("Enter the path to your .h5ad file and map your column names.")

    st.markdown('<div class="section-header">Step 1 — Enter file path</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Since spatial transcriptomics files are typically several GB, '
                'we read the file directly from your computer instead of uploading it through the browser.</div>',
                unsafe_allow_html=True)

    file_path = st.text_input(
        "Full path to your .h5ad file",
        value=st.session_state.data_path or
              r"C:\Users\user\Desktop\Spatial_transcriptomic\ST_Discovery.h5ad",
        help="Copy and paste the full file path from your computer"
    )

    st.markdown('<div class="section-header">Step 2 — Map your column names</div>',
                unsafe_allow_html=True)
    st.markdown("Tell the app which columns in your dataset correspond to each piece of information.")

    col1, col2, col3 = st.columns(3)
    with col1:
        cell_type_col = st.text_input("Cell type column",    value="cell.subtypes",
                                       help="Column in .obs that labels each cell's type")
        sample_col    = st.text_input("Sample column",       value="samples",
                                       help="Column that identifies each tissue sample")
    with col2:
        patient_col   = st.text_input("Patient column",      value="patients",
                                       help="Column that identifies each patient")
        treatment_col = st.text_input("Treatment column",    value="treatment_clean",
                                       help="Column with treatment group labels")
    with col3:
        x_col         = st.text_input("X coordinate column", value="x",
                                       help="Column with x spatial coordinates")
        y_col         = st.text_input("Y coordinate column", value="y",
                                       help="Column with y spatial coordinates")

    st.markdown('<div class="section-header">Step 3 — Load the file</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Step 2b — Clinical metadata file (optional)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">If your clinical information is in a separate file '
                '(e.g. supplementary Excel/CSV from the paper), enter the path here. '
                'The app will automatically merge it with your cell data using the sample ID column. '
                'Leave blank if clinical info is already inside your h5ad file.</div>',
                unsafe_allow_html=True)

    clinical_path = st.text_input(
        "Path to clinical metadata file (Excel or CSV)",
        value="",
        help="e.g. C:/Users/user/Desktop/Spatial_transcriptomic/41590_2024_1943_MOESM3_ESM.xlsx"
    )

    col_clin1, col_clin2, col_clin3 = st.columns(3)
    with col_clin1:
        clinical_sheet = st.text_input("Sheet name (Excel only)",
                                        value="Table 2b",
                                        help="Sheet name containing clinical data")
    with col_clin2:
        clinical_sample_col = st.text_input("Sample ID column in clinical file",
                                             value="profile",
                                             help="Column that matches your sample IDs")
    with col_clin3:
        clinical_header_row = st.number_input("Header row number",
                                               value=1, min_value=0, max_value=10,
                                               help="Row number of column headers (0=first row, 1=second row)")

    load_btn = st.button("📂 Load Data", type="primary")

    if load_btn:
        if not os.path.exists(file_path):
            st.error(f"File not found at: {file_path}\n\nPlease check the path and try again.")
        else:
            with st.spinner("Loading data... this may take a minute for large files..."):
                import anndata
                adata = anndata.read_h5ad(file_path)
                # Drop the heavy scaledata layer immediately to free memory
                if 'scaledata' in adata.layers:
                     del adata.layers['scaledata']

                # Validate columns exist
                missing_cols = []
                for col in [cell_type_col, sample_col, patient_col, x_col, y_col]:
                    if col not in adata.obs.columns:
                        missing_cols.append(col)

                if missing_cols:
                    st.error(f"These columns were not found in your dataset: {missing_cols}\n\n"
                             f"Available columns: {list(adata.obs.columns)}")
                else:
                    # Load and merge clinical file if provided
                    if clinical_path and os.path.exists(clinical_path):
                        try:
                            # Read clinical file
                            if clinical_path.endswith('.xlsx') or clinical_path.endswith('.xls'):
                                clin_df = pd.read_excel(
                                    clinical_path,
                                    sheet_name=clinical_sheet,
                                    header=int(clinical_header_row)
                                )
                            else:
                                clin_df = pd.read_csv(clinical_path)

                            # Clean column names
                            clin_df.columns = clin_df.columns.str.strip()

                            # Check sample column exists
                            if clinical_sample_col not in clin_df.columns:
                                st.warning(f"Sample column '{clinical_sample_col}' not found in clinical file. "
                                           f"Available columns: {list(clin_df.columns)}")
                            else:
                                # Set sample as index for merging
                                clin_df = clin_df.set_index(clinical_sample_col)

                                # Select useful clinical columns
                                clinical_cols = ['treatment', 'immunotherapy',
                                                 'sites_binary', 'stage', 'age',
                                                 'outcome', 'patients']
                                available_clin = [c for c in clinical_cols
                                                  if c in clin_df.columns]

                                # Clean up immunotherapy column
                                # (some values are drug names — convert to Yes/No)
                                if 'immunotherapy' in clin_df.columns:
                                    clin_df['immunotherapy_clean'] = clin_df['immunotherapy'].apply(
                                        lambda x: 'No' if str(x).strip().lower() in ['no', 'nan', '']
                                        else 'Yes'
                                    )
                                    available_clin.append('immunotherapy_clean')

                                # Add treatment_clean column
                                if 'treatment' in clin_df.columns:
                                    clin_df['treatment_clean'] = clin_df['treatment'].str.strip()
                                    available_clin.append('treatment_clean')

                                if 'sites_binary' in clin_df.columns:
                                    clin_df['site_clean'] = clin_df['sites_binary'].str.strip()
                                    available_clin.append('site_clean')

                                # Merge clinical data into adata.obs using sample column
                                for col in list(set(available_clin)):
                                    if col in clin_df.columns:
                                        adata.obs[col] = adata.obs[sample_col].map(clin_df[col])

                                st.success(f"✅ Clinical data merged — added columns: "
                                           f"{list(set(available_clin))}")

                        except Exception as e:
                            st.warning(f"Could not load clinical file: {e}")

                    elif clinical_path and not os.path.exists(clinical_path):
                        st.warning(f"Clinical file not found at: {clinical_path}")

                    st.session_state.adata       = adata
                    st.session_state.data_loaded = True
                    st.session_state.data_path   = file_path
                    st.session_state.col_map     = {
                        'cell_type': cell_type_col,
                        'sample':    sample_col,
                        'patient':   patient_col,
                        'treatment': treatment_col,
                        'x':         x_col,
                        'y':         y_col,
                    }
                    st.success(f"✅ Data loaded successfully!")

            if st.session_state.data_loaded:
                adata = st.session_state.adata

                # Show summary
                st.markdown('<div class="section-header">Dataset Summary</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total cells",   f"{adata.n_obs:,}")
                with c2:
                    st.metric("Total genes",   f"{adata.n_vars:,}")
                with c3:
                    st.metric("Samples",
                              f"{adata.obs[sample_col].nunique():,}")
                with c4:
                    st.metric("Patients",
                              f"{adata.obs[patient_col].nunique():,}")

                st.markdown('<div class="section-header">Cell Types Found</div>',
                            unsafe_allow_html=True)
                ct_counts = adata.obs[cell_type_col].value_counts().reset_index()
                ct_counts.columns = ['Cell Type', 'Count']
                st.dataframe(ct_counts, use_container_width=True, height=300)

                st.markdown('<div class="success-box">✅ Data loaded. '
                            'Go to <strong>⚙️ Run Pipeline</strong> to analyse a cell type.</div>',
                            unsafe_allow_html=True)

    # Show current status if already loaded
    if st.session_state.data_loaded and not load_btn:
        adata = st.session_state.adata
        col_map = st.session_state.col_map
        st.markdown('<div class="success-box">✅ Data already loaded from: '
                    f'{st.session_state.data_path}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total cells",  f"{adata.n_obs:,}")
        with c2: st.metric("Total genes",  f"{adata.n_vars:,}")
        with c3: st.metric("Samples",      f"{adata.obs[col_map['sample']].nunique():,}")
        with c4: st.metric("Patients",     f"{adata.obs[col_map['patient']].nunique():,}")


# ═══════════════════════════════════════════
# PAGE 3 — RUN PIPELINE
# ═══════════════════════════════════════════
elif page == "⚙️ Run Pipeline":
    st.markdown('<div class="main-title">⚙️ Run Spatial Pipeline</div>', unsafe_allow_html=True)
    st.markdown("Select a cell type and compute spatial features and expression scores.")

    if not st.session_state.data_loaded:
        st.warning("No data loaded yet. Go to **📂 Load Data** first.")
        st.stop()

    adata   = st.session_state.adata
    col_map = st.session_state.col_map
    cell_type_col = col_map['cell_type']

    # Get available cell types from the loaded data
    available_cell_types = sorted(adata.obs[cell_type_col].unique().tolist())
    available_genes      = list(adata.var_names)

    st.markdown('<div class="section-header">Step 1 — Select cell types</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cell_of_interest = st.multiselect(
            "Cell type of interest",
            options=available_cell_types,
            default=[t for t in available_cell_types
                     if 'CD8' in t or 'T.cell' in t][:2] or [available_cell_types[0]],
            help="Select one or more cell type labels that represent your cell of interest"
        )
        # Auto-generate display name from selected cell types
        auto_name = " + ".join(cell_of_interest) if cell_of_interest else "Selected cells"
        # Clean up the name to be more readable
        auto_name = auto_name.replace("CD8.T.cell_LC", "CD8+ T cells (LC)")                              .replace("CD8.T.cell",    "CD8+ T cells")                              .replace("CD4.T.cell_LC", "CD4+ T cells (LC)")                              .replace("CD4.T.cell",    "CD4+ T cells")                              .replace("CD4.T.cell.DN_LC", "CD4+ DN T cells (LC)")                              .replace("CD4.T.cell.DN",    "CD4+ DN T cells")                              .replace("Treg_LC",       "Tregs (LC)")                              .replace("Treg",          "Tregs")                              .replace("NK.cell_LC",    "NK cells (LC)")                              .replace("NK.cell",       "NK cells")                              .replace("T.B.cell_LC",   "T/B cells")                              .replace("T.cell.DP",     "DP T cells")                              .replace("B.cell_LC",     "B cells (LC)")                              .replace("B.cell",        "B cells")                              .replace("Monocyte_LC",   "Monocytes (LC)")                              .replace("Monocyte",      "Monocytes")                              .replace("Fibroblast_LC", "Fibroblasts (LC)")                              .replace("Fibroblast",    "Fibroblasts")                              .replace("Endothelial_LC","Endothelial (LC)")                              .replace("Endothelial",   "Endothelial")                              .replace("Mast.cell_LC",  "Mast cells (LC)")                              .replace("Mast.cell",     "Mast cells")                              .replace("Malignant_LC",  "Malignant (LC)")                              .replace("Malignant",     "Malignant")
        cell_type_name = st.text_input(
            "Display name for this cell type",
            value=auto_name,
            help="This is auto-generated from your selection above. You can edit it if needed."
        )
    with col2:
        neighbour_types = st.multiselect(
            "Neighbour cell type",
            options=available_cell_types,
            default=[t for t in available_cell_types if 'Malignant' in t][:2]
                    or [available_cell_types[0]],
            help="Select the cell type to measure distances TO"
        )

    st.markdown('<div class="section-header">Step 2 — Spatial parameters</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        radius = st.slider("Radius (µm)", 50, 300, 150, 10,
                           help="Size of circle to count neighbours within")
    with col4:
        k = st.slider("Number of neighbouring cells to consider (k)", 5, 50, 20, 5,
              help="Finds the k nearest cells of all types around each T cell "
                   "and calculates what fraction are cancer cells. "
                   "Higher k = broader neighbourhood. Recommended: 20")

    st.markdown('<div class="section-header">Step 3 — Expression score</div>',
                unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        score_name = st.text_input("Score name", value="exhaustion_score",
                                    help="Name for the composite expression score")
    with col6:
        # Suggest common exhaustion markers if available
        default_markers = [g for g in ['PDCD1','LAG3','HAVCR2','CTLA4']
                           if g in available_genes]
        # Search box for genes
        gene_search = st.text_input("Search genes (type to filter)",
                                     value="",
                                     help="Type a gene name to filter the list")
        if gene_search:
            filtered_genes = [g for g in available_genes
                              if gene_search.upper() in g.upper()][:100]
        else:
            # Show common markers + first 100 genes
            common = ['PDCD1','LAG3','HAVCR2','CTLA4','CD47','THBS1',
                      'TIGIT','CD274','CXCR3','CXCR4','CCR5','CCR7']
            common_in_data = [g for g in common if g in available_genes]
            filtered_genes = common_in_data + [g for g in available_genes
                             if g not in common_in_data][:100]
        selected_markers = st.multiselect(
            "Expression markers",
            options=filtered_genes,
            default=[m for m in default_markers if m in filtered_genes],
            help="Genes to include in the composite score. Use search box to find genes."
        )

    st.markdown("---")

    col_run, col_demo = st.columns([1, 2])
    with col_run:
        run_btn = st.button("▶ Run Pipeline", type="primary",
                            disabled=(not cell_of_interest or not neighbour_types))
    with col_demo:
        if cd8_ref is not None:
            if st.button("Load saved CD8+ T cell data (demo)"):
                st.session_state.cell_obs       = cd8_ref.copy()
                st.session_state.cell_type_name = "CD8+ T cells"
                st.session_state.score_name     = "exhaustion_score"
                st.session_state.pipeline_run   = True
                st.success(f"Loaded {len(cd8_ref):,} CD8+ T cells. Navigate to other pages.")

    if run_btn:
        if not cell_of_interest:
            st.error("Please select at least one cell type of interest.")
        elif not neighbour_types:
            st.error("Please select at least one neighbour cell type.")
        else:
            cell_mask = adata.obs[cell_type_col].isin(cell_of_interest)
            nb_mask   = adata.obs[cell_type_col].isin(neighbour_types)
            cell_obs  = adata.obs[cell_mask].copy()

            st.info(f"Found **{cell_mask.sum():,}** {cell_type_name} cells and "
                    f"**{nb_mask.sum():,}** neighbour cells.")

            if cell_mask.sum() == 0:
                st.error("No cells found for the selected cell type. Check your selection.")
                st.stop()

            # Compute spatial features
            st.markdown("**Computing spatial features...**")
            cell_obs = compute_spatial_features(
                adata.obs, cell_mask, nb_mask, cell_obs,
                col_map, radius=radius, k=k
            )

            # Compute expression score
            if selected_markers:
                st.markdown("**Computing expression scores...**")
                with st.spinner("Calculating z-scores..."):
                    cell_obs = compute_expression_score(
                        adata, cell_mask, cell_obs,
                        markers=selected_markers,
                        score_name=score_name
                    )

            # Save to session state
            st.session_state.cell_obs       = cell_obs
            st.session_state.cell_type_name = cell_type_name
            st.session_state.score_name     = score_name
            st.session_state.pipeline_run   = True

            st.success(f"✅ Pipeline complete — {len(cell_obs):,} {cell_type_name} cells processed.")

            # Quick summary
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Cells processed", f"{len(cell_obs):,}")
            with c2:
                d = cell_obs['nearest_neighbour_dist'].dropna()
                st.metric("Median nearest distance", f"{d.median():.1f} µm")
            with c3:
                if score_name in cell_obs.columns:
                    st.metric(f"Median {score_name}",
                              f"{cell_obs[score_name].median():.3f}")

            st.markdown('<div class="success-box">Navigate to other pages to explore results.</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 4 — PREDICT CELL STATE
# ═══════════════════════════════════════════
elif page == "🔮 Predict Cell State":
    st.markdown('<div class="main-title">🔮 Predict Cell State</div>', unsafe_allow_html=True)
    st.markdown("Enter feature values to predict whether a cell is Exhausted or Active.")

    if need_pipeline(): st.stop()
    if rf_model is None:
        st.error("ML model not found. Make sure dashboard_model.pkl is in the same folder.")
        st.stop()

    cell_obs  = st.session_state.cell_obs
    c_name    = st.session_state.cell_type_name
    score_col = st.session_state.score_name
    d_col     = 'nearest_neighbour_dist' if 'nearest_neighbour_dist' in cell_obs.columns \
                else 'nearest_tumour_dist'
    cnt_col   = [c for c in cell_obs.columns if 'count_r' in c]
    cnt_col   = cnt_col[0] if cnt_col else None
    knn_col   = [c for c in cell_obs.columns if 'frac_knn' in c]
    knn_col   = knn_col[0] if knn_col else None

    st.markdown(f'<div class="info-box">🧬 This tool predicts whether a T cell is likely '
                f'<strong>Exhausted</strong> (worn out from prolonged exposure to cancer cells) or '
                f'<strong>Active</strong> (still functioning normally), based on its physical location '
                f'in the tumour tissue and the activity of two key genes. '
                f'Uses a Random Forest model trained on CD8+ T cells (AUC = 0.5992). '
                f'Currently viewing: <strong>{c_name}</strong></div>',
                unsafe_allow_html=True)

    # ── Scenario presets ────────────────────────────────────────────────────
    d_med    = float(cell_obs[d_col].median())   if d_col   and d_col   in cell_obs.columns else 110.0
    cnt_med  = int(cell_obs[cnt_col].median())   if cnt_col and cnt_col in cell_obs.columns else 2
    knn_med  = float(cell_obs[knn_col].median()) if knn_col and knn_col in cell_obs.columns else 0.1

    def get_gene_mean(gene):
        if gene in cell_obs.columns:
            return float(cell_obs[gene].mean())
        if cd8_ref is not None and gene in cd8_ref.columns:
            return float(cd8_ref[gene].mean())
        return 0.5 if gene == 'CD47' else 0.1

    cd47_med  = get_gene_mean('CD47')
    thbs1_med = get_gene_mean('THBS1')

    scenarios = {
        "Dataset average (pipeline values)": {
            "dist": d_med, "count": cnt_med, "knn": knn_med,
            "cd47": cd47_med, "thbs1": thbs1_med
        },
        "Active cell — far from tumour": {
            "dist": 400.0, "count": 0, "knn": 0.0,
            "cd47": 0.0, "thbs1": 0.0
        },
        "Exhausted cell — deep in tumour": {
            "dist": 30.0, "count": 15, "knn": 0.75,
            "cd47": cd47_med * 2, "thbs1": thbs1_med * 2
        },
        "Borderline cell — at the threshold": {
            "dist": d_med, "count": 5, "knn": 0.25,
            "cd47": cd47_med, "thbs1": thbs1_med
        },
        "Custom — adjust sliders manually": None
    }

    scenario_choice = st.selectbox(
        "Simulate a scenario",
        options=list(scenarios.keys()),
        help="Pick a preset to auto-fill the sliders, or choose Custom to set values manually."
    )

    preset = scenarios[scenario_choice]
    s_dist  = float(preset["dist"])  if preset else d_med
    s_count = int(preset["count"])   if preset else cnt_med
    s_knn   = float(preset["knn"])   if preset else knn_med
    s_cd47  = min(5.0, float(preset["cd47"]))  if preset else cd47_med
    s_thbs1 = min(5.0, float(preset["thbs1"])) if preset else thbs1_med

    pipeline_key = st.session_state.get('cell_type_name', 'default')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Spatial Features**")

        nearest  = st.slider("Nearest Neighbour Distance (µm)", 0.0, 3335.0, s_dist, 1.0,
                             key=f"dist_{pipeline_key}_{scenario_choice}",
                             help="Distance from this T cell to the closest cancer cell in the tissue.")

        nb_count = st.slider("Cancer Cell Count within 150 µm radius", 0, 200, s_count,
                             key=f"count_{pipeline_key}_{scenario_choice}",
                             help="Number of cancer cells found within a 150 µm circle around this T cell.")

        knn_frac = st.slider("Tumour Neighbour Fraction (k=20)", 0.0, 1.0, s_knn, 0.05,
                             key=f"knn_{pipeline_key}_{scenario_choice}",
                             help="Fraction of the 20 nearest cells that are cancer cells.")

    with col2:
        st.markdown("**Gene Expression Features**")

        scenario_key = scenario_choice.split("—")[0].strip().replace(" ","_")

        cd47  = st.slider("CD47 Expression",  0.0, 5.0, s_cd47,  0.01,
                          key=f"cd47_{pipeline_key}_{scenario_key}",
                          help="Expression level of CD47, a protein on T cell surfaces that cancer cells exploit to suppress immune activity via the TSP-1/CD47 pathway.")

        thbs1 = st.slider("THBS1 Expression", 0.0, 5.0, s_thbs1, 0.01,
                          key=f"thbs1_{pipeline_key}_{scenario_key}",
                          help="Expression level of THBS1 (Thrombospondin-1), a protein secreted by cancer cells that acts as a barrier blocking T cells from reaching the tumour.")

    st.markdown("---")

    inp  = pd.DataFrame([[nearest, nb_count, knn_frac, cd47, thbs1]],
                         columns=['nearest_tumour_dist', 'tumour_count_r150',
                                  'tumour_frac_knn_20', 'CD47', 'THBS1'])
    pred = rf_model.predict(inp)[0]
    prob = rf_model.predict_proba(inp)[0]
    classes = list(rf_model.classes_)

    # Handle both string labels and numeric labels
    if 'Exhausted' in classes:
        ex_p = prob[classes.index('Exhausted')]
        ac_p = prob[classes.index('Active')]
        pred_label = pred
    else:
        # Classes are numeric — map to labels using cd8_ref
        ex_p = prob[1] if len(prob) > 1 else prob[0]
        ac_p = prob[0] if len(prob) > 1 else prob[0]
        pred_label = 'Exhausted' if pred == 1 else 'Active'
    pred = pred_label

    cp, cc, cg = st.columns([1, 1, 2])
    with cp:
        st.markdown("**Prediction**")
        if pred == 'Exhausted':
            st.markdown('<div class="exhausted-badge">😫 EXHAUSTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="active-badge">💪 ACTIVE</div>', unsafe_allow_html=True)
        st.caption("Random Forest · CD8+ model · AUC = 0.5992")
    with cc:
        st.markdown("**Confidence**")
        st.markdown(f"- Exhausted: **{ex_p:.1%}**")
        st.markdown(f"- Active:    **{ac_p:.1%}**")
    with cg:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=ex_p * 100,
            title={'text': "Probability of Exhaustion (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': '#c0392b'},
                'steps': [{'range': [0,  40], 'color': '#d4edda'},
                           {'range': [40, 60], 'color': '#fff3cd'},
                           {'range': [60,100], 'color': '#fdecea'}],
                'threshold': {'line': {'color': '#c0392b', 'width': 3},
                              'thickness': 0.75, 'value': 50}
            }
        ))
        fig_g.update_layout(height=230, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    # Context
    if d_col in cell_obs.columns:
        pct = (nearest > cell_obs[d_col]).mean() * 100
        direction = "further from" if pct > 50 else "closer to"
        st.info(f"This cell's distance ({nearest:.1f} µm) is higher than **{pct:.0f}%** "
                f"of {c_name} cells — {direction} the neighbour cells than average.")

    # ── Reliability warning for extreme values ──────────
    warnings_list = []

    # Check distance
    if cd8_ref is not None and 'nearest_tumour_dist' in cd8_ref.columns:
        d_p95 = cd8_ref['nearest_tumour_dist'].quantile(0.95)
        d_p05 = cd8_ref['nearest_tumour_dist'].quantile(0.05)
        if nearest > d_p95:
            warnings_list.append(
                f"Nearest Distance ({nearest:.0f} µm) is in the top 5% of training data "
                f"(above {d_p95:.0f} µm). The model has rarely seen cells this far away."
            )
        elif nearest < d_p05:
            warnings_list.append(
                f"Nearest Distance ({nearest:.0f} µm) is in the bottom 5% of training data "
                f"(below {d_p05:.0f} µm). The model has rarely seen cells this close."
            )

    # Check neighbour count
    if cd8_ref is not None and 'tumour_count_r150' in cd8_ref.columns:
        c_p95 = cd8_ref['tumour_count_r150'].quantile(0.95)
        if nb_count > c_p95:
            warnings_list.append(
                f"Neighbour Count ({nb_count}) is in the top 5% of training data "
                f"(above {c_p95:.0f}). Very few cells had this many neighbours."
            )

    # Check KNN fraction
    if cd8_ref is not None and 'tumour_frac_knn_20' in cd8_ref.columns:
        k_p95 = cd8_ref['tumour_frac_knn_20'].quantile(0.95)
        if knn_frac > k_p95:
            warnings_list.append(
                f"KNN Fraction ({knn_frac:.2f}) is in the top 5% of training data "
                f"(above {k_p95:.2f}). Very few cells had this high a fraction."
            )

    # Show warning if any extreme values found
    if warnings_list:
        st.markdown("---")
        warning_lines = ["One or more feature values are outside the reliable training range.",
                         "The model may not predict accurately for these extreme values:", ""]
        for w in warnings_list:
            warning_lines.append(f"- {w}")
        warning_lines.append("")
        warning_lines.append("Recommendation: Use feature values within the typical range of your dataset for more reliable predictions.")
        st.warning("\n".join(warning_lines))
    else:
        st.success("All feature values are within the reliable range of the training data. Prediction confidence is dependable.")


# ═══════════════════════════════════════════
# PAGE 5 — SPATIAL PLOT
# ═══════════════════════════════════════════
elif page == "🗺️ Spatial Plot":
    st.markdown('<div class="main-title">🗺️ Spatial Plot</div>', unsafe_allow_html=True)

    if need_pipeline(): st.stop()

    cell_obs  = st.session_state.cell_obs
    c_name    = st.session_state.cell_type_name
    s_col     = st.session_state.score_name
    col_map   = st.session_state.col_map
    sample_col = col_map.get('sample', 'samples')
    x_col      = col_map.get('x', 'x')
    y_col      = col_map.get('y', 'y')
    d_col      = 'nearest_neighbour_dist' if 'nearest_neighbour_dist' in cell_obs.columns \
                 else 'nearest_tumour_dist'

    c1, c2, c3 = st.columns(3)
    with c1:
        samp = st.selectbox("Select sample",
                            sorted(cell_obs[sample_col].unique()))
    with c2:
        colour_opts = []
        for opt in ['target', s_col, d_col, 'CD47', 'THBS1', 'PDCD1', 'LAG3']:
            if opt in cell_obs.columns:
                colour_opts.append(opt)
        colour_by = st.selectbox("Colour by", colour_opts)
    with c3:
        dot_size = st.slider("Dot size", 3, 15, 7)

    sdf = cell_obs[cell_obs[sample_col] == samp]

    # Option to show cancer cells as background
    show_cancer = st.checkbox("Show cancer cells in background", value=True)

    fig = go.Figure()

    # ── Layer 1: Cancer cells in background (grey dots) ──
    if show_cancer and st.session_state.adata is not None:
        adata     = st.session_state.adata
        col_map_  = st.session_state.col_map
        ct_col    = col_map_.get('cell_type', 'cell.subtypes')
        samp_col_ = col_map_.get('sample', 'samples')
        x_c       = col_map_.get('x', 'x')
        y_c       = col_map_.get('y', 'y')

        cancer_mask = (adata.obs[samp_col_] == samp) &                       adata.obs[ct_col].isin(['Malignant', 'Malignant_LC'])
        cancer_df   = adata.obs[cancer_mask]

        if len(cancer_df) > 0:
            fig.add_trace(go.Scatter(
                x=cancer_df[x_c], y=cancer_df[y_c],
                mode='markers',
                marker=dict(size=3, color='#e8c8c8', opacity=0.4),
                name=f'Cancer cells ({len(cancer_df):,})',
                hovertemplate='Cancer cell<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>'
            ))

    # ── Layer 2: T cells on top coloured by selected feature ──
    if colour_by == 'target' and 'target' in sdf.columns:
        for state, colour in [('Exhausted', '#c0392b'), ('Active', '#2980b9')]:
            sub = sdf[sdf['target'] == state]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode='markers',
                marker=dict(size=dot_size, color=colour, opacity=0.9),
                name=f'{state} ({len(sub):,})',
                hovertemplate=f'{state}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<extra></extra>'
            ))
    else:
        # Continuous colour scale
        colour_vals = sdf[colour_by] if colour_by in sdf.columns else None
        fig.add_trace(go.Scatter(
            x=sdf[x_col], y=sdf[y_col],
            mode='markers',
            marker=dict(
                size=dot_size,
                color=colour_vals,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title=colour_by),
                opacity=0.9
            ),
            name=f'{c_name} ({len(sdf):,})',
            hovertemplate=colour_by + '=%{marker.color:.3f}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{c_name} — {samp} ({len(sdf):,} cells)",
        height=540,
        margin=dict(t=50, b=20, r=180),
        yaxis=dict(autorange='reversed', title='Y position (µm)'),
        xaxis=dict(title='X position (µm)'),
        legend=dict(
            itemsizing='constant',
            bgcolor='rgba(30,30,60,0.85)',
            font=dict(color='white', size=11),
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            x=1.12,
            y=1.0,
            xanchor='left',
            yanchor='top'
        ),
        coloraxis_colorbar=dict(
            x=1.02,
            y=0.5,
            len=0.6,
            thickness=15,
        ),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Cells in sample", f"{len(sdf):,}")
    with c2:
        if d_col in sdf.columns:
            st.metric("Median distance", f"{sdf[d_col].median():.1f} µm")
    with c3:
        if s_col in sdf.columns:
            st.metric(f"Median {s_col}", f"{sdf[s_col].median():.3f}")


# ═══════════════════════════════════════════
# PAGE 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════
elif page == "📊 Feature Importance":
    st.markdown('<div class="main-title">📊 Feature Importance</div>', unsafe_allow_html=True)

    if rf_model is None:
        st.error("ML model not found. Make sure dashboard_model.pkl is in the same folder.")
        st.stop()

    FEATURES = ['nearest_tumour_dist', 'tumour_count_r150',
                'tumour_frac_knn_20', 'CD47', 'THBS1']
    FLABELS  = {
        'nearest_tumour_dist': 'Nearest Tumour Distance',
        'tumour_count_r150':   'Tumour Count (150 µm)',
        'tumour_frac_knn_20':  'KNN Tumour Fraction',
        'CD47':                'CD47 Expression',
        'THBS1':               'THBS1 Expression',
    }

    imp_df = pd.DataFrame({
        'Feature':    [FLABELS[f] for f in FEATURES],
        'Importance': rf_model.feature_importances_,
        'Type':       ['Spatial', 'Spatial', 'Spatial', 'Gene', 'Gene']
    }).sort_values('Importance', ascending=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(imp_df, x='Importance', y='Feature', color='Type',
                     color_discrete_map={'Spatial': '#2980b9', 'Gene': '#e67e22'},
                     orientation='h',
                     text=imp_df['Importance'].apply(lambda x: f'{x:.1%}'),
                     title='Random Forest Feature Importance (CD8+ T cell model)')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=360, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sp_imp = rf_model.feature_importances_[:3].sum()
        gn_imp = rf_model.feature_importances_[3:].sum()
        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><strong>📍 Spatial features</strong><br>'
                    f'{sp_imp:.1%} of predictive power</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box" style="border-left:4px solid #e67e22;background:#fef4e8;">'
                    f'<strong>🧬 Gene features</strong><br>'
                    f'{gn_imp:.1%} of predictive power</div>', unsafe_allow_html=True)
        st.markdown("Physical location in the tumour microenvironment is the primary driver of T cell exhaustion.")

    # Correlations
    if need_pipeline(): st.stop()
    cell_obs  = st.session_state.cell_obs
    score_col = st.session_state.score_name
    d_col     = 'nearest_neighbour_dist' if 'nearest_neighbour_dist' in cell_obs.columns \
                else 'nearest_tumour_dist'

    st.markdown('<div class="section-header">Spearman Correlation with Expression Score</div>',
                unsafe_allow_html=True)
    corr_features = [d_col] + [c for c in cell_obs.columns
                                if 'count_r' in c or 'frac_knn' in c]
    corrs = {}
    for f in corr_features:
        if f in cell_obs.columns and score_col in cell_obs.columns:
            r, _ = spearmanr(cell_obs[f].dropna(),
                             cell_obs.loc[cell_obs[f].notna(), score_col])
            corrs[f] = r

    if corrs:
        fig2 = px.bar(x=list(corrs.keys()), y=list(corrs.values()),
                      color=list(corrs.values()),
                      color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                      labels={'x': 'Feature', 'y': f'Spearman r with {score_col}'},
                      title=f'Feature Correlations with {score_col}')
        fig2.add_hline(y=0, line_dash='dash', line_color='grey')
        fig2.update_layout(height=300, margin=dict(t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE 7 — PATIENT EXPLORER
# ═══════════════════════════════════════════
elif page == "👥 Patient Clinical & Treatment Analysis":
    st.markdown('<div class="main-title">👥 Patient Clinical & Treatment Analysis</div>',
                unsafe_allow_html=True)

    if need_pipeline(): st.stop()

    cell_obs      = st.session_state.cell_obs
    c_name        = st.session_state.cell_type_name
    s_col         = st.session_state.score_name
    col_map       = st.session_state.col_map
    patient_col   = col_map.get('patient',   'patients')
    treatment_col = col_map.get('treatment', 'treatment_clean')
    d_col         = 'nearest_neighbour_dist' if 'nearest_neighbour_dist' in cell_obs.columns \
                    else 'nearest_tumour_dist'

    grp_cols = [patient_col]
    if treatment_col in cell_obs.columns:
        grp_cols.append(treatment_col)

    agg = {'n_cells': (s_col, 'count')}
    if s_col  in cell_obs.columns: agg['median_score']    = (s_col,  'median')
    if d_col  in cell_obs.columns: agg['median_distance'] = (d_col,  'median')
    if 'target' in cell_obs.columns:
        agg['pct_exhausted'] = ('target', lambda x: (x == 'Exhausted').mean() * 100)

    pat_df = cell_obs.groupby(grp_cols).agg(**agg).reset_index()

    tab1, tab2, tab3 = st.tabs(["📊 Patient Overview", "💊 Treatment Breakdown", "📦 Group Comparisons"])

    # ── TAB 1: Patient Overview ──────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            colour_col = treatment_col if treatment_col in pat_df.columns else None
            x_opts = [c for c in ['median_distance', 'pct_exhausted', 'n_cells'] if c in pat_df.columns]
            x_col_sel = st.selectbox("X axis", x_opts,
                                      format_func=lambda x: {
                                          'median_distance': 'Median Distance (µm)',
                                          'pct_exhausted':   '% Exhausted',
                                          'n_cells':         'Number of Cells'}.get(x, x))
        with c2:
            if colour_col and colour_col in pat_df.columns:
                colour_col = st.selectbox("Colour by", [c for c in grp_cols if c != patient_col])

        if 'median_score' in pat_df.columns and x_col_sel in pat_df.columns:
            colour_vals = pat_df[colour_col].dropna().unique() if colour_col in pat_df.columns else []
            colour_map  = {}
            palette     = ['#e74c3c','#2980b9','#27ae60','#f39c12','#8e44ad','#16a085','#d35400','#2c3e50']
            for i, val in enumerate(sorted(colour_vals)):
                colour_map[str(val)] = palette[i % len(palette)]

            fig = px.scatter(
                pat_df, x=x_col_sel, y='median_score',
                color=colour_col if colour_col in pat_df.columns else None,
                color_discrete_map=colour_map, size='n_cells', size_max=40,
                hover_data={patient_col: True, 'n_cells': True, 'median_score': ':.3f', 'median_distance': ':.1f'},
                title=f"Patient-Level Comparison — {c_name}",
                labels={x_col_sel: x_col_sel.replace('_',' ').title(),
                        'median_score': f'Median {s_col}', 'n_cells': 'Number of T Cells',
                        colour_col: colour_col.replace('_clean','').title() if colour_col else ''}
            )
            fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1.5,
                          annotation_text='Active | Exhausted boundary',
                          annotation_font_color='white', annotation_font_size=11)
            if x_col_sel == 'median_distance':
                fig.add_annotation(x=0.02, y=0.97, xref='paper', yref='paper',
                                   text='Close to tumour<br>+ Exhausted', showarrow=False,
                                   font=dict(size=10, color='#e74c3c'), align='left',
                                   bgcolor='rgba(0,0,0,0.3)', bordercolor='#e74c3c', borderwidth=1)
                fig.add_annotation(x=0.98, y=0.03, xref='paper', yref='paper',
                                   text='Far from tumour<br>+ Active', showarrow=False,
                                   font=dict(size=10, color='#2ecc71'), align='right',
                                   bgcolor='rgba(0,0,0,0.3)', bordercolor='#2ecc71', borderwidth=1)
            fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='white')))
            fig.update_layout(
                height=380, margin=dict(t=50, b=30, l=60, r=160),
                plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=False),
                legend=dict(bgcolor='rgba(30,30,60,0.8)', bordercolor='rgba(255,255,255,0.2)',
                            borderwidth=1, font=dict(size=12))
            )
            st.plotly_chart(fig, use_container_width=True)

            if colour_col in pat_df.columns and 'median_distance' in pat_df.columns:
                group_stats = {}
                for grp in sorted(pat_df[colour_col].dropna().unique()):
                    grp_data = pat_df[pat_df[colour_col] == grp]
                    if len(grp_data) > 0:
                        group_stats[grp] = {'n': len(grp_data),
                                            'dist': grp_data['median_distance'].median(),
                                            'score': grp_data['median_score'].median()}
                if len(group_stats) >= 2:
                    scores = {g: v['score'] for g, v in group_stats.items()}
                    most_exhausted = max(scores, key=scores.get)
                    most_active    = min(scores, key=scores.get)
                    for grp, stats in group_stats.items():
                        dot_colour   = colour_map.get(str(grp), '#2980b9')
                        other_scores = [v['score'] for g, v in group_stats.items() if g != grp]
                        avg_other    = sum(other_scores) / len(other_scores) if other_scores else 0
                        relative     = "more exhausted" if stats['score'] > avg_other else "less exhausted"
                        rel_colour   = "#e74c3c" if stats['score'] > avg_other else "#2ecc71"
                        marker       = " ← most exhausted" if grp == most_exhausted else \
                                       " ← most active"    if grp == most_active    else ""
                        st.markdown(
                            f'<div class="info-box" style="border-left:4px solid {dot_colour};">'
                            f'<strong>{grp}</strong> ({stats["n"]} patients) · '
                            f'Median distance = <strong>{stats["dist"]:.1f} µm</strong> · '
                            f'Median score = <strong>{stats["score"]:.3f}</strong> · '
                            f'<span style="color:{rel_colour};">{relative} than other groups</span>'
                            f'<strong>{marker}</strong></div>', unsafe_allow_html=True)
                    e = group_stats[most_exhausted]
                    a = group_stats[most_active]
                    st.markdown(
                        f'<div class="warn-box">📊 <strong>Key comparison:</strong> '
                        f'<strong>{most_exhausted}</strong> (score={e["score"]:.3f}, dist={e["dist"]:.1f} µm) '
                        f'vs <strong>{most_active}</strong> (score={a["score"]:.3f}, dist={a["dist"]:.1f} µm). '
                        f'Difference = {abs(e["score"]-a["score"]):.3f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Patient Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(pat_df.round(3), use_container_width=True, height=200)

    # ── TAB 2: Treatment Breakdown ───────────────────────────────────────────
    with tab2:
        treatment_type_cols   = ['treatment_clean','treatment','Treatment','treatment_type',
                                  'chemotherapy','immunotherapy_clean','immunotherapy']
        found_treatment_cols  = [c for c in treatment_type_cols if c in cell_obs.columns]
        clean_bases           = {c.replace('_clean','') for c in found_treatment_cols if c.endswith('_clean')}
        found_treatment_cols  = [c for c in found_treatment_cols if c not in clean_bases]

        if found_treatment_cols:
            patient_col_name = st.session_state.col_map.get('patient','patients')
            selected_tx_col  = st.selectbox(
                "Select treatment variable",
                found_treatment_cols,
                format_func=lambda c: c.replace('_clean','').replace('_',' ').title()
            )
            tx_counts = cell_obs.groupby(selected_tx_col).agg(
                Number_of_Patients=(patient_col_name, 'nunique'),
                Median_Exhaustion_Score=(s_col, 'median')
            ).reset_index()
            tx_counts.columns = [selected_tx_col, 'Number of Patients', 'Median Exhaustion Score']
            tx_counts = tx_counts.sort_values('Number of Patients', ascending=False)

            ta, tb_col = st.columns(2)
            with ta:
                fig_tx1 = px.bar(
                    tx_counts, x=selected_tx_col, y='Number of Patients',
                    color=selected_tx_col, text='Number of Patients',
                    title=f'Patients per {selected_tx_col.replace("_clean","").replace("_"," ").title()} Group',
                    color_discrete_sequence=['#2980b9','#e74c3c','#27ae60','#f39c12','#8e44ad','#16a085']
                )
                fig_tx1.update_traces(textposition='outside')
                fig_tx1.update_layout(
                    height=380, showlegend=False, xaxis_title='', yaxis_title='Number of Patients',
                    plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    margin=dict(b=40,t=50,l=40,r=20),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=-30, automargin=True),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_tx1, use_container_width=True)
            with tb_col:
                fig_tx2 = px.bar(
                    tx_counts, x=selected_tx_col, y='Median Exhaustion Score',
                    color='Median Exhaustion Score', color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title=f'Exhaustion Score by {selected_tx_col.replace("_clean","").replace("_"," ").title()}',
                    text=tx_counts['Median Exhaustion Score'].apply(lambda v: f'{v:.3f}')
                )
                fig_tx2.update_traces(textposition='outside')
                fig_tx2.add_hline(y=0, line_dash='dash', line_color='white', line_width=1.5,
                                  annotation_text='Active | Exhausted boundary',
                                  annotation_font_color='white', annotation_font_size=10)
                fig_tx2.update_layout(
                    height=380, showlegend=False, xaxis_title='', yaxis_title='Median Exhaustion Score',
                    plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    margin=dict(b=40,t=50,l=40,r=20),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=-30, automargin=True),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_tx2, use_container_width=True)

            # Specific drug names from raw immunotherapy column
            imm_col = next((c for c in ['immunotherapy'] if c in cell_obs.columns), None)
            if imm_col:
                st.markdown('<div class="section-header">Specific Treatment Received by Patients</div>',
                            unsafe_allow_html=True)
                drug_df     = cell_obs[[patient_col_name, imm_col]].drop_duplicates()
                drug_counts = drug_df.groupby(imm_col)[patient_col_name].nunique().reset_index()
                drug_counts.columns = ['Treatment Received','Number of Patients']
                drug_counts = drug_counts.sort_values('Number of Patients', ascending=False)

                tc, td = st.columns(2)
                with tc:
                    fig_drug = px.bar(
                        drug_counts, x='Treatment Received', y='Number of Patients',
                        color='Treatment Received', text='Number of Patients',
                        title='Patients per Specific Treatment Received',
                        color_discrete_sequence=['#2980b9','#e74c3c','#27ae60','#f39c12',
                                                  '#8e44ad','#16a085','#d35400','#2c3e50',
                                                  '#c0392b','#1abc9c']
                    )
                    fig_drug.update_traces(textposition='outside')
                    fig_drug.update_layout(
                        height=380, showlegend=False, xaxis_title='', yaxis_title='Number of Patients',
                        plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                        margin=dict(b=120,t=50,l=40,r=20),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=-40, automargin=True),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig_drug, use_container_width=True)
                with td:
                    patient_tx_table = drug_df.sort_values(imm_col).rename(columns={
                        patient_col_name: 'Patient', imm_col: 'Treatment Received'
                    })
                    st.markdown('<div class="section-header">Patient Treatment Record</div>',
                                unsafe_allow_html=True)
                    st.dataframe(patient_tx_table, use_container_width=True, hide_index=True, height=340)
        else:
            st.markdown('<div class="warn-box">Treatment type details not found. '
                        'Load clinical metadata on the Load Data page.</div>', unsafe_allow_html=True)

    # ── TAB 3: Group Comparisons ─────────────────────────────────────────────
    with tab3:
        if treatment_col in cell_obs.columns and d_col in cell_obs.columns:
            b1, b2 = st.columns(2)
            with b1:
                fig_b = px.box(cell_obs, x=treatment_col, y=d_col,
                               color=treatment_col, points=False,
                               title='Infiltration by Treatment Group',
                               labels={treatment_col: 'Treatment', d_col: 'Nearest Distance (µm)'})
                fig_b.update_layout(height=380, showlegend=False,
                                    plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig_b, use_container_width=True)
            with b2:
                if s_col in cell_obs.columns:
                    fig_b2 = px.box(cell_obs, x=treatment_col, y=s_col,
                                    color=treatment_col, points=False,
                                    title=f'{s_col} by Treatment Group',
                                    labels={treatment_col: 'Treatment', s_col: s_col})
                    fig_b2.update_layout(height=380, showlegend=False,
                                         plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)',
                                         font=dict(color='white'),
                                         xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                         yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
                    st.plotly_chart(fig_b2, use_container_width=True)
        else:
            st.markdown('<div class="warn-box">Run pipeline first to see group comparisons.</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 8 — EXHAUSTION THRESHOLD
# ═══════════════════════════════════════════
elif page == "📈 Exhaustion Threshold":
    st.markdown('<div class="main-title">📈 Exhaustion Threshold by Clinical Group</div>',
                unsafe_allow_html=True)
    st.markdown("At how many cancer neighbours does exhaustion kick in — and which treatment protects T cells best?")

    if need_pipeline(): st.stop()

    cell_obs      = st.session_state.cell_obs
    c_name        = st.session_state.cell_type_name
    s_col         = st.session_state.score_name
    col_map       = st.session_state.col_map
    treatment_col = col_map.get('treatment', 'treatment_clean')

    count_cols = [c for c in cell_obs.columns if 'neighbour_count_r' in c or 'tumour_count_r' in c]
    if not count_cols:
        st.warning("No radius count column found. Run the pipeline first.")
        st.stop()
    count_col = count_cols[0]
    radius_val = count_col.split('_r')[-1] if '_r' in count_col else '150'

    if s_col not in cell_obs.columns:
        st.warning(f"Score column '{s_col}' not found. Run pipeline with expression markers.")
        st.stop()

    bins   = [0, 1, 3, 5, 10, 20, 9999]
    labels = ['0', '1–2', '3–4', '5–9', '10–19', '20+']

    def get_threshold_df(df, group_col, group_val, label):
        sub = df[df[group_col] == group_val].copy() if group_val != 'All' else df.copy()
        sub = sub.dropna(subset=[count_col, s_col])
        sub['bin'] = pd.cut(sub[count_col], bins=bins, labels=labels, right=False)
        agg = sub.groupby('bin', observed=True)[s_col].mean().reset_index()
        agg.columns = ['Neighbours', 'Mean Score']
        agg['Group'] = label
        return agg

    # Build clinical column list
    exclude_cols    = ['samples', 'patients', count_col, s_col,
                       'x', 'y', 'orig.ident', 'ident', 'sites_binary',
                       'cell.types', 'cell.subtypes']
    common_clinical = ['treatment_clean', 'immunotherapy_clean', 'site_clean',
                       'treatment', 'immunotherapy', 'site', 'Treatment',
                       'clinical_group', 'stage']
    suggested    = [c for c in common_clinical if c in cell_obs.columns
                    and c not in exclude_cols and 2 <= cell_obs[c].nunique() <= 15]
    other_cols   = [c for c in cell_obs.columns if c not in suggested
                    and c not in exclude_cols
                    and 2 <= cell_obs[c].nunique() <= 10]
    ordered_cols = suggested + other_cols

    if not ordered_cols and cd8_ref is not None:
        for col in ['treatment_clean', 'immunotherapy_clean', 'site_clean']:
            if col in cd8_ref.columns and col not in cell_obs.columns:
                try:
                    cell_obs = cell_obs.copy()
                    cell_obs[col] = cd8_ref.loc[cell_obs.index, col]
                    if cell_obs[col].notna().sum() > 0:
                        ordered_cols.append(col)
                        st.session_state.cell_obs = cell_obs
                except:
                    pass

    if not ordered_cols:
        st.warning("No clinical grouping columns found.")
        st.stop()

    tab_clin, tab_drug = st.tabs(["🏥 Clinical Groups", "💊 Treatment Power Ranking"])

    # ── TAB 1: Clinical Groups ───────────────────────────────────────────────
    with tab_clin:
        r1, r2 = st.columns(2)
        with r1:
            selected_clinical_cols = st.multiselect(
                "Split by clinical column(s)", options=ordered_cols,
                default=ordered_cols[:2] if len(ordered_cols) >= 2 else ordered_cols,
                help="Each unique value becomes a separate group on the chart."
            )
        dfs = [get_threshold_df(cell_obs, None, 'All', f'All {c_name}')]
        for col in selected_clinical_cols:
            if col in cell_obs.columns:
                for grp in sorted(cell_obs[col].dropna().unique()):
                    label = f"{col.replace('_clean','').replace('_',' ').title()}: {grp}"
                    dfs.append(get_threshold_df(cell_obs, col, grp, label))
        threshold_df = pd.concat(dfs, ignore_index=True)
        all_groups   = threshold_df['Group'].unique().tolist()

        with r2:
            smart_default = [g for g in all_groups if any(x in g for x in ['All','Untreated','Treated'])]
            if not smart_default:
                smart_default = all_groups[:4]
            selected   = st.multiselect("Select groups to compare", options=all_groups, default=smart_default)
        chart_type = st.radio("Chart type", ["Line", "Bar"], horizontal=True)

        filtered = threshold_df[threshold_df['Group'].isin(selected)]
        if chart_type == "Line":
            fig_t = px.line(filtered, x='Neighbours', y='Mean Score', color='Group', markers=True,
                            title=f'Mean {s_col} by Neighbour Count — {c_name}',
                            labels={'Neighbours': f'Cancer Cells within {radius_val} µm', 'Mean Score': f'Mean {s_col}'})
        else:
            fig_t = px.bar(filtered, x='Neighbours', y='Mean Score', color='Group', barmode='group',
                           title=f'Mean {s_col} by Neighbour Count — {c_name}',
                           labels={'Neighbours': f'Cancer Cells within {radius_val} µm', 'Mean Score': f'Mean {s_col}'})
        fig_t.add_hline(y=0, line_dash='dot', line_color='grey', annotation_text='Zero line (threshold)')
        fig_t.update_layout(height=380, margin=dict(t=50, b=20),
                            plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig_t, use_container_width=True)

        # Summary table
        summary = []
        for grp in threshold_df['Group'].unique():
            grp_df = threshold_df[threshold_df['Group'] == grp]
            crossing = grp_df[grp_df['Mean Score'] > 0]
            threshold_bin = crossing.iloc[0]['Neighbours'] if len(crossing) > 0 else 'Does not cross zero'
            summary.append({'Clinical Group': grp, 'Threshold (first > 0)': threshold_bin,
                            'Max Score': round(grp_df['Mean Score'].max(), 3),
                            'Min Score': round(grp_df['Mean Score'].min(), 3)})
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True, height=220)
        st.markdown('<div class="info-box">💡 <strong>How to read:</strong> '
                    'The threshold is the first bin where mean score crosses zero. '
                    'Higher threshold = cell resists exhaustion longer = more robust immune response.</div>',
                    unsafe_allow_html=True)

    # ── TAB 2: Treatment Power Ranking ───────────────────────────────────────
    with tab_drug:
        imm_col = next((c for c in ['immunotherapy'] if c in cell_obs.columns), None)
        if not imm_col:
            st.markdown('<div class="warn-box">No specific treatment column found in dataset.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">💊 Compares each specific treatment received. '
                        'A <strong>higher threshold</strong> means T cells stay active even with more cancer neighbours — '
                        'indicating a more protective treatment.</div>', unsafe_allow_html=True)

            drug_vals = sorted(cell_obs[imm_col].dropna().unique())
            drug_dfs  = []
            for drug in drug_vals:
                if len(cell_obs[cell_obs[imm_col] == drug]) >= 50:
                    drug_dfs.append(get_threshold_df(cell_obs, imm_col, drug, drug))

            if not drug_dfs:
                st.markdown('<div class="warn-box">Not enough cells per treatment group '
                            '(minimum 50 cells required).</div>', unsafe_allow_html=True)
            else:
                drug_threshold_df = pd.concat(drug_dfs, ignore_index=True)

                # Build ranking first — used by both chart and table
                threshold_order = {'0': 0, '1–2': 1, '3–4': 2, '5–9': 3,
                                   '10–19': 4, '20+': 5, 'Does not cross zero': 6}
                ranking = []
                for drug in drug_vals:
                    grp_df = drug_threshold_df[drug_threshold_df['Group'] == drug]
                    if grp_df.empty:
                        continue
                    crossing = grp_df[grp_df['Mean Score'] > 0]
                    threshold_bin = crossing.iloc[0]['Neighbours'] if len(crossing) > 0 else 'Does not cross zero'
                    ranking.append({
                        'Treatment':      drug,
                        'Threshold':      threshold_bin,
                        'Threshold Rank': threshold_order.get(threshold_bin, 99),
                        'Min Score':      round(grp_df['Mean Score'].min(), 3),
                        'Max Score':      round(grp_df['Mean Score'].max(), 3),
                        'N Cells':        len(cell_obs[cell_obs[imm_col] == drug]),
                    })
                ranking_df = pd.DataFrame(ranking).sort_values('Threshold Rank', ascending=False).drop(columns='Threshold Rank')

                # Clean horizontal bar chart — one bar per treatment, height = threshold rank
                ranking_df['Threshold Rank Value'] = ranking_df['Threshold'].map(threshold_order)
                fig_drug_t = px.bar(
                    ranking_df.sort_values('Threshold Rank Value'),
                    y='Treatment',
                    x='Threshold Rank Value',
                    color='Threshold Rank Value',
                    color_continuous_scale='RdYlGn',
                    orientation='h',
                    text='Threshold',
                    title='Treatment Protection Level — higher bar = more protective',
                    hover_data={'Treatment': True, 'Threshold': True,
                                'Min Score': True, 'N Cells': True,
                                'Threshold Rank Value': False}
                )
                fig_drug_t.update_traces(textposition='outside')
                fig_drug_t.update_layout(
                    height=max(300, len(ranking_df) * 45),
                    margin=dict(t=50, b=20, l=10, r=100),
                    coloraxis_showscale=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(title='', automargin=True,
                               gridcolor='rgba(255,255,255,0.1)'),
                    plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                fig_drug_t.add_vline(x=0, line_dash='dot', line_color='grey')
                st.plotly_chart(fig_drug_t, use_container_width=True)

                # Ranking table + highlights side by side
                t1, t2 = st.columns([2, 1])
                with t1:
                    st.markdown('<div class="section-header">Power Ranking — most protective first</div>',
                                unsafe_allow_html=True)
                    display_df = ranking_df.drop(columns=['Threshold Rank Value'], errors='ignore')
                    st.dataframe(display_df, use_container_width=True, hide_index=True, height=260)
                with t2:
                    if not ranking_df.empty:
                        best  = ranking_df.iloc[0]
                        worst = ranking_df.iloc[-1]
                        st.markdown(
                            f'<div class="success-box">🏆 <strong>Most protective</strong><br>'
                            f'{best["Treatment"]}<br>'
                            f'Threshold: <strong>{best["Threshold"]}</strong><br>'
                            f'Min score: {best["Min Score"]}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="warn-box">⚠️ <strong>Least protective</strong><br>'
                            f'{worst["Treatment"]}<br>'
                            f'Threshold: <strong>{worst["Threshold"]}</strong><br>'
                            f'Min score: {worst["Min Score"]}</div>', unsafe_allow_html=True)