import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime
import io

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VietBank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS  – dark finance aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* ── root palette – clean light slate ── */
:root {
    --bg:        #f0f4f8;
    --surface:   #ffffff;
    --card:      #ffffff;
    --border:    #dde3ec;
    --accent:    #0ea47a;
    --accent2:   #3b82f6;
    --danger:    #dc2626;
    --warn:      #d97706;
    --text:      #1e293b;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'Inter', sans-serif;
}

/* ── global resets ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label { color: var(--text) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span { color: var(--text) !important; }

/* ── headings ── */
h1, h2, h3 { font-family: var(--font-head) !important; color: var(--text) !important; }

/* ── inputs ── */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="select"] div {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
input:focus, [data-baseweb="input"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(14,164,122,.15) !important;
}

/* ── labels ── */
label, .stSelectbox label, .stNumberInput label,
.stSlider label, .stDateInput label, .stRadio label {
    color: var(--text) !important;
    font-weight: 500 !important;
}

/* ── buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity .2s, box-shadow .2s;
    box-shadow: 0 2px 8px rgba(14,164,122,.25) !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    box-shadow: 0 4px 14px rgba(14,164,122,.35) !important;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: .8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 1.8rem !important;
    color: var(--text) !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}

/* ── tabs ── */
[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-baseweb="tab"] { color: var(--muted) !important; font-family: var(--font-head) !important; }
[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── alert boxes ── */
.stSuccess {
    background: #ecfdf5 !important;
    border-left: 4px solid var(--accent) !important;
    color: #065f46 !important;
}
.stError {
    background: #fef2f2 !important;
    border-left: 4px solid var(--danger) !important;
    color: #7f1d1d !important;
}
.stWarning {
    background: #fffbeb !important;
    border-left: 4px solid var(--warn) !important;
    color: #78350f !important;
}
.stInfo {
    background: #eff6ff !important;
    border-left: 4px solid var(--accent2) !important;
    color: #1e3a5f !important;
}

/* ── progress bar ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── selectbox arrow ── */
[data-baseweb="select"] svg { fill: var(--muted) !important; }

/* ── expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── number input buttons ── */
[data-testid="stNumberInput"] button {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Load models  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    pipeline  = joblib.load("./model/pipeline.pkl")
    model     = joblib.load("./model/best_model.pkl")
    threshold = joblib.load("./model/threshold.pkl")
    return pipeline, model, threshold

# ─────────────────────────────────────────────
#  Feature engineering  (mirrors EDA notebook)
# ─────────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2025-05-16")  # ≈ max_date + 1 day used in training

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── column renaming ──
    col_map = {
        "credit_sco": "credit_score",
        "occupation": "profession",
        "monthly_ir": "monthly_income",
        "tenure_ye":  "tenure_yearly",
    }
    df.rename(columns=col_map, inplace=True)

    # ── drop high-leakage / id-like columns ──
    drop_raw = [
        "id", "full_name", "gender", "address",
        "engagement_score", "loyalty_level", "risk_score",
        "risk_segment", "cluster_group", "last_transaction_month",
        "customer_segment",
    ]
    df.drop(columns=[c for c in drop_raw if c in df.columns], inplace=True)

    # ── profession mapping ──
    profession_map = {
        "Chủ Doanh nghiệp nhỏ":          "Small Business Owner",
        "Nội trợ/Sinh viên":              "Housewife/Student",
        "Giáo viên/Giảng viên":           "Teacher/Lecturer",
        "Kỹ sư/Chuyên viên IT":           "Engineer/IT Specialist",
        "Hưu trí":                        "Retired",
        "Nhân viên văn phòng/Công chức":  "Office Worker/Civil Servant",
        "Kinh doanh/Bán hàng":            "Business/Salesperson",
        "Quản lý/Lãnh đạo":               "Manager/Leader",
        "Kế toán/Tài chính":              "Accountant/Finance Worker",
        "Lao động phổ thông":             "General Laborer",
    }
    if "profession" in df.columns:
        df["profession"] = df["profession"].map(profession_map).fillna(df["profession"])

    # ── datetime conversion ──
    for col in ["last_active_date", "created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # ── feature engineering ──
    if "last_active_date" in df.columns:
        df["recency_days"] = (REFERENCE_DATE - df["last_active_date"]).dt.days

    if "created_date" in df.columns:
        df["account_age_days"]  = (REFERENCE_DATE - df["created_date"]).dt.days
        df["created_month"]     = df["created_date"].dt.month
        df["created_weekday"]   = df["created_date"].dt.weekday

        # ── account_age_groups (kept by pipeline) ──
        df["account_age_groups"] = pd.cut(
            df["account_age_days"],
            bins=[0, 180, 365, 730, 99999],
            labels=["<6m", "6-12m", "1-2y", "2y+"]
        ).astype(str)

    # ── encoding: all cols must be int except account_age_groups (matches training dtypes) ──
    if "active_member" in df.columns:
        df["active_member"] = df["active_member"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        ).astype(int)
    if "exit" in df.columns:
        df["exit"] = df["exit"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        ).astype(int)
    # In training, digital_behavior was int (mobile=1, offline=0), not a string
    if "digital_behavior" in df.columns:
        df["digital_behavior"] = df["digital_behavior"].map(
            {"mobile": 1, "offline": 0, 1: 1, 0: 0}
        ).astype(int)

    # ── final drop ──
    drop_final = [
        "last_active_date", "created_date",
        "origin_province", "profession",
        "account_age_days", "created_month", "created_weekday",  # dropped, NOT account_age_groups
    ]
    df.drop(columns=[c for c in drop_final if c in df.columns], inplace=True)

    return df


def predict(pipeline, model, threshold, df_raw: pd.DataFrame):
    df_feat   = engineer_features(df_raw)
    X         = df_feat.drop(columns=["exit"], errors="ignore")
    X_scaled  = pipeline.transform(X)
    proba     = model.predict_proba(X_scaled)[:, 1]
    label     = (proba >= threshold).astype(int)
    return proba, label


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 VietBank")
    st.markdown("### Churn Intelligence")
    st.markdown("---")
    mode = st.radio("Prediction mode", ["✏️  Manual Entry", "📂  CSV Upload"])
    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown("- Logistic Regression")
    st.markdown("- Threshold : `0.3`")
    st.markdown("- Recall (churn) : `0.948`")
    st.markdown("- ROC-AUC : `0.856`")
    st.markdown("---")
    st.caption("Synthetic Vietnam banking dataset · 80 000 records")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.4rem; margin-bottom:.2rem;'>
  Customer Churn <span style='color:#00c896'>Predictor</span>
</h1>
<p style='color:#64748b; margin-top:0; font-family:Inter,sans-serif;'>Vietnam Retail Banking · Logistic Regression</p>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Load models (show error if not found)
# ─────────────────────────────────────────────
try:
    pipeline, model, threshold = load_models()
    st.success("✅  Models loaded successfully", icon="✅")
except Exception as e:
    st.error(f"❌  Could not load models: {e}")
    st.info("Make sure `pipeline.pkl`, `best_model.pkl`, and `threshold.pkl` are inside `./model/`")
    st.stop()


# ═══════════════════════════════════════════════════════════
#  MODE 1: MANUAL ENTRY
# ═══════════════════════════════════════════════════════════
if mode == "✏️  Manual Entry":
    st.markdown("### 📝 Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔢 Financial**")
        credit_score   = st.number_input("Credit Score",   min_value=0,    max_value=850,  value=682,        step=1)
        balance        = st.number_input("Balance (VND)",  min_value=0,    max_value=int(1e9), value=30_000_000, step=100_000)
        monthly_income = st.number_input("Monthly Income (VND)", min_value=0,         max_value=220_000_000, value=23_000_000, step=500_000)

    with col2:
        st.markdown("**👤 Demographics**")
        age            = st.slider("Age",           min_value=18, max_value=90,  value=45)
        married        = st.selectbox("Marital Status", options=[0, 1, 2, 3],
                                      format_func=lambda x: {0:"Single",1:"Married",2:"Divorced",3:"Widowed"}[x])
        active_member  = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")
        digital_behavior = st.selectbox("Digital Behaviour", ["mobile", "offline"])

    with col3:
        st.markdown("**📅 Account**")
        tenure_yearly  = st.slider("Tenure (years)", min_value=0, max_value=10, value=2)
        nums_card      = st.slider("Number of Cards",    min_value=1, max_value=5, value=2)
        nums_service   = st.slider("Number of Services", min_value=1, max_value=8, value=3)
        last_active_date = st.date_input("Last Active Date", value=date(2025, 1, 15))
        created_date     = st.date_input("Account Created Date", value=date(2022, 6, 10))

    st.markdown("")
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

    if predict_btn:
        row = {
            "credit_sco":        credit_score,
            "age":               age,
            "balance":           balance,
            "monthly_ir":        monthly_income,
            "tenure_ye":         tenure_yearly,
            "married":           married,
            "nums_card":         nums_card,
            "nums_service":      nums_service,
            "active_member":     active_member,
            "digital_behavior":  digital_behavior,
            "last_active_date":  last_active_date.strftime("%d/%m/%Y"),
            "created_date":      created_date.strftime("%d/%m/%Y"),
        }
        df_input = pd.DataFrame([row])

        try:
            proba, label = predict(pipeline, model, threshold, df_input)
            p = float(proba[0])
            churned = bool(label[0])

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability",  f"{p*100:.1f}%")
            m2.metric("Threshold",          f"{threshold*100:.0f}%")
            m3.metric("Prediction",         "🔴 CHURN" if churned else "🟢 RETAIN")

            # Risk bar
            colour = "#ef4444" if p >= 0.5 else "#f59e0b" if p >= 0.3 else "#00c896"
            risk_label = "High" if p >= 0.5 else "Medium" if p >= 0.3 else "Low"
            st.markdown(f"""
            <div style='margin-top:1rem;'>
              <p style='color:#64748b; margin-bottom:.3rem;'>Risk Level: <b style='color:{colour}'>{risk_label}</b></p>
              <div style='background:#e2e8f0; border-radius:8px; height:14px; overflow:hidden;'>
                <div style='width:{p*100:.1f}%; background:{colour}; height:100%; border-radius:8px; transition:width .5s;'></div>
              </div>
              <p style='text-align:right; color:#64748b; font-size:.75rem;'>{p*100:.1f}% churn probability</p>
            </div>
            """, unsafe_allow_html=True)

            if churned:
                st.error("⚠️  This customer is predicted to **churn**. Consider proactive retention strategies.")
            else:
                st.success("✅  This customer is likely to **stay**. Keep engagement high!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════════
#  MODE 2: CSV UPLOAD
# ═══════════════════════════════════════════════════════════
else:
    st.markdown("### 📂 Batch Prediction via CSV")
    st.info("Upload a CSV matching the original dataset columns. Columns `exit`, `full_name`, `id`, `address` etc. are automatically handled.")

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**Loaded:** `{df_raw.shape[0]:,}` rows × `{df_raw.shape[1]}` columns")

            with st.expander("👀 Preview raw data"):
                st.dataframe(df_raw.head(10), use_container_width=True)

            run_btn = st.button("⚡ Run Batch Prediction", use_container_width=True)

            if run_btn:
                with st.spinner("Engineering features & predicting…"):
                    proba, label = predict(pipeline, model, threshold, df_raw.copy())

                results = df_raw.copy()
                results["churn_probability"] = np.round(proba * 100, 2)
                results["predicted_churn"]   = label
                results["risk_level"] = pd.cut(
                    proba,
                    bins=[-0.001, 0.3, 0.5, 1.001],
                    labels=["Low", "Medium", "High"]
                )

                # ── Summary metrics ──
                st.markdown("---")
                st.markdown("### 📊 Batch Summary")
                n_total   = len(results)
                n_churn   = int(label.sum())
                n_retain  = n_total - n_churn
                avg_risk  = proba.mean()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers",  f"{n_total:,}")
                c2.metric("Predicted Churn",  f"{n_churn:,}",  delta=f"{n_churn/n_total*100:.1f}%")
                c3.metric("Retained",         f"{n_retain:,}", delta=f"{n_retain/n_total*100:.1f}%")
                c4.metric("Avg Churn Prob",   f"{avg_risk*100:.1f}%")

                # ── Risk distribution ──
                st.markdown("#### Risk Distribution")
                risk_counts = results["risk_level"].value_counts().reindex(["High","Medium","Low"])
                rc1, rc2, rc3 = st.columns(3)
                colours = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#00c896"}
                for col, level in zip([rc1, rc2, rc3], ["High","Medium","Low"]):
                    count = int(risk_counts.get(level, 0))
                    pct   = count / n_total * 100 if n_total else 0
                    col.markdown(f"""
                    <div style='background:#ffffff; border:1px solid #dde3ec; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,.06);
                                padding:1rem; text-align:center;'>
                        <div style='font-size:1.8rem; font-family:Syne,sans-serif; color:{colours[level]};
                                    font-weight:800;'>{count:,}</div>
                        <div style='color:#64748b; font-size:.85rem;'>{level} Risk &nbsp;·&nbsp; {pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Results table ──
                st.markdown("---")
                st.markdown("#### 🗂 Prediction Results")

                display_cols = [c for c in ["full_name", "age", "credit_sco", "credit_score",
                                             "balance", "churn_probability", "predicted_churn", "risk_level"]
                                if c in results.columns]
                display_df = results[display_cols]

                # Only apply Styler gradient on small files (Streamlit cap ~262k cells)
                max_styled_rows = 262144 // max(len(display_cols), 1)
                if len(display_df) <= max_styled_rows:
                    render_df = display_df.style.background_gradient(
                        subset=["churn_probability"], cmap="RdYlGn_r"
                    )
                else:
                    render_df = display_df  # plain df for large uploads

                st.dataframe(render_df, use_container_width=True, height=400)

                # ── Download ──
                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download predictions as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.exception(e)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#334155; font-size:.75rem;'>"
    "VietBank Churn Intelligence · Synthetic data · Logistic Regression @ threshold 0.30"
    "</p>",
    unsafe_allow_html=True,
)