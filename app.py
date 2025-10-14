import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import RecategorizeSmoking

# New imports for advanced pages
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, confusion_matrix
import altair as alt

# 新增：校准与绘图/SHAP
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
# 移除顶层直接导入 shap，改为延迟导入以兼容 NumPy 2.0
import importlib
import matplotlib.pyplot as plt

# SHAP 已移除：删除延迟导入函数与重复导入

st.set_page_config(page_title="Diabetes Prediction (XGBoost)", page_icon="🩺", layout="centered")

DATA_DIR = Path('.')
MODEL_PATH = DATA_DIR / "final_xgboost_model.pkl"
META_PATH = DATA_DIR / "model_meta.yaml"
VALID_PATH = DATA_DIR / 'valid.csv'
TEST_PATH = DATA_DIR / 'test.csv'
DEFAULT_THRESHOLD = 0.41

@st.cache_resource(show_spinner=False)
def load_model(model_path_str: str, model_mtime_ns: int):
    # Cache depends on file path and its mtime to ensure refresh when model artifact updates
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

@st.cache_data(show_spinner=False)
def load_metadata(meta_path_str: str, meta_mtime_ns: int):
    p = Path(meta_path_str)
    if not p.exists():
        return {}
    with open(p, 'r') as f:
        return yaml.safe_load(f) or {}

model = None
model_load_error = None
meta = {}
try:
    mtime = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0
    model = load_model(str(MODEL_PATH), mtime)
    meta_mtime = META_PATH.stat().st_mtime_ns if META_PATH.exists() else 0
    meta = load_metadata(str(META_PATH), meta_mtime)
except Exception as e:
    model_load_error = str(e)

# Determine default threshold from metadata if available
meta_threshold = meta.get('threshold') if isinstance(meta, dict) else None
effective_default_threshold = float(meta_threshold) if meta_threshold is not None else DEFAULT_THRESHOLD

# Global UI styling: font stack, container width, cards, spacing
st.markdown(
    """
    <style>
    :root {
      --primary: #2A9D8F;
      --text: #1F2937;
      --bg: #F7FAFC;
      --card-bg: #FFFFFF;
    }
    html, body, [class*=\"css\"]  {
      font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, \"Noto Sans\", \"Liberation Sans\", sans-serif, \"Apple Color Emoji\",\"Segoe UI Emoji\";
      line-height: 1.6;
      color: var(--text);
    }
    .block-container { max-width: 1160px; padding-top: 1rem; }
    h1 { font-size: 2rem; line-height: 1.25; }
    h2, .stMarkdown h2 { font-size: 1.35rem; }
    /* paragraph margins: use defaults */
    .card { background: var(--card-bg); border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.06); padding: 16px; }
    .stAlert { border-radius: 10px; padding: 0.75rem 1rem; }
    /* removed alert markdown and text margin overrides to use defaults */
    .stAlert p, .stAlert span { line-height: 1.6; }
    /* Removed container-specific paragraph margin overrides; rely on global p spacing */
    .st-emotion-cache-12h5x7g p { word-break: break-word; }
    /* Buttons keep consistent internal spacing; no margin overrides applied */
    .stButton > button { line-height: 1.6; padding: 0.5rem 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Diabetes Prediction Dashboard")
st.caption("Binary classification dashboard for diabetes risk using XGBoost. Data source: cleaned_diabetes_dataset.csv; validation/test via valid.csv & test.csv.")

with st.sidebar:
    # st.divider()  # removed horizontal divider at top of sidebar

    st.header("Input")
    # Add a Predict button directly under the Input title (keep the original bottom button as well)
    predict_top_btn = st.button("Predict", key="predict_top")
    
    gender = st.radio("Gender", options=["Female", "Male"], index=0, horizontal=True)
    # female and male on the same line via horizontal=True
    age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    hypertension = st.selectbox("Hypertension", options=["No", "Yes"], index=0)
    heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"], index=0)
    smoking_history = st.selectbox(
        "Smoking History",
        options=["current_smoker", "non_smoker", "past_smoker", "unknown"],
        index=3,
    )
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1, format="%.1f")
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.6, step=0.1, format="%.1f")
    glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0, value=100.0, step=1.0, format="%.1f")

    threshold = st.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=effective_default_threshold, step=0.01,
                          help="Default threshold chosen by the training script (based on validation F1). You can adjust it.")

    st.markdown("<style>[data-testid=\"stSidebar\"] .stButton > button { width: 100%; }</style>", unsafe_allow_html=True)
    # Keep the original bottom Predict button
    predict_bottom_btn = st.button("Predict", key="predict_bottom")
    # Unified trigger variable used by the main page logic
    predict_btn = predict_top_btn or predict_bottom_btn


def build_feature_row():
    # 构建原始特征行，交由 Pipeline 内部完成 OneHot 编码与转换
    features = {
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history,
        "bmi": float(bmi),
        "HbA1c_level": float(hba1c),
        "blood_glucose_level": float(glucose),
    }
    return pd.DataFrame([features])

@st.cache_data(show_spinner=False)
def load_eval_data():
    if VALID_PATH.exists():
        valid_df = pd.read_csv(VALID_PATH)
    else:
        valid_df = None
    if TEST_PATH.exists():
        test_df = pd.read_csv(TEST_PATH)
    else:
        test_df = None
    return valid_df, test_df


def page_predict():
    if model_load_error:
        st.error(f"Model load failed: {model_load_error}")
        st.stop()

    # Show default prediction on first load
    run_default = st.session_state.get("show_default_prediction", True)

    # Top info banner when showing default results
    if run_default:
        st.info("This page shows a default prediction. Please adjust the parameters in the left sidebar and click the Predict button to run your prediction.")

    if predict_btn or run_default:
        X = build_feature_row()
        try:
            # Inference spinner
            with st.spinner("Running inference..."):
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0, 1]
                else:
                    if hasattr(model, "decision_function"):
                        score = float(model.decision_function(X))
                        proba = 1 / (1 + np.exp(-score))
                    else:
                        pred = int(model.predict(X)[0])
                        proba = float(pred)
            pred_label = 1 if proba >= threshold else 0

            st.subheader("Prediction")
            # removed card wrappers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Diabetes probability", value=f"{proba*100:.1f}%")
            with col2:
                st.metric(label="Classification Threshold", value=f"{threshold:.2f}")
            with col3:
                st.metric(label="Default Threshold (metadata)", value=f"{effective_default_threshold:.2f}")

            if pred_label == 1:
                st.error("Model prediction: High risk (please consult a doctor)")
            else:
                st.success("Model prediction: Low risk (maintain healthy habits)")

            with st.expander("View Input Features"):
                st.dataframe(build_feature_row())

            st.caption("Disclaimer: This application is for teaching and demonstration purposes only and does not replace professional medical advice.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        finally:
            # Only show default prediction once; subsequent runs require clicking Predict
            st.session_state["show_default_prediction"] = False
    else:
        # removed card wrappers
        st.info("Fill in the sidebar and click 'Predict' to run inference.")


def page_interpretability():
    st.header("Interpretability")
    valid_df, _ = load_eval_data()
    if valid_df is None:
        st.warning("Validation file valid.csv not found. Global interpretability analysis is unavailable.")
        return
    X_valid = valid_df[[
        'gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'
    ]]
    y_valid = valid_df['diabetes'] if 'diabetes' in valid_df.columns else None

    def extract_preprocess_and_clf(model):
        preprocess_obj, xgb_clf = None, None
        try:
            if hasattr(model, 'calibrated_classifiers_'):
                for cc in getattr(model, 'calibrated_classifiers_', []) or []:
                    cand = getattr(cc, 'base_estimator', None) or getattr(cc, 'estimator', None)
                    if cand is not None:
                        if hasattr(cand, 'named_steps'):
                            preprocess_obj = cand.named_steps.get('preprocess', preprocess_obj)
                            xgb_clf = cand.named_steps.get('clf', xgb_clf)
                        elif hasattr(cand, 'steps'):
                            steps_dict = dict(cand.steps)
                            preprocess_obj = steps_dict.get('preprocess', preprocess_obj)
                            xgb_clf = steps_dict.get('clf', xgb_clf)
                    if preprocess_obj is not None and xgb_clf is not None:
                        return preprocess_obj, xgb_clf
            base_est = getattr(model, 'base_estimator', None)
            if base_est is not None:
                if hasattr(base_est, 'named_steps'):
                    preprocess_obj = base_est.named_steps.get('preprocess', preprocess_obj)
                    xgb_clf = base_est.named_steps.get('clf', xgb_clf)
                elif hasattr(base_est, 'steps'):
                    steps_dict = dict(base_est.steps)
                    preprocess_obj = steps_dict.get('preprocess', preprocess_obj)
                    xgb_clf = steps_dict.get('clf', xgb_clf)
                if preprocess_obj is not None and xgb_clf is not None:
                    return preprocess_obj, xgb_clf
            if hasattr(model, 'named_steps'):
                preprocess_obj = model.named_steps.get('preprocess')
                xgb_clf = model.named_steps.get('clf')
                if preprocess_obj is not None and xgb_clf is not None:
                    return preprocess_obj, xgb_clf
            if hasattr(model, 'steps'):
                steps_dict = dict(model.steps)
                preprocess_obj = steps_dict.get('preprocess', preprocess_obj)
                xgb_clf = steps_dict.get('clf', xgb_clf)
                if preprocess_obj is not None and xgb_clf is not None:
                    return preprocess_obj, xgb_clf
            if hasattr(model, 'get_params'):
                params = model.get_params(deep=True)
                preprocess_obj = params.get('base_estimator__preprocess', preprocess_obj)
                xgb_clf = params.get('base_estimator__clf', xgb_clf)
                for k, v in params.items():
                    if preprocess_obj is None and isinstance(k, str) and k.endswith('__preprocess'):
                        preprocess_obj = v
                    if xgb_clf is None and isinstance(k, str) and k.endswith('__clf'):
                        xgb_clf = v
                    if hasattr(v, 'named_steps'):
                        preprocess_obj = v.named_steps.get('preprocess', preprocess_obj)
                        xgb_clf = v.named_steps.get('clf', xgb_clf)
                    elif hasattr(v, 'steps'):
                        sd = dict(v.steps)
                        preprocess_obj = sd.get('preprocess', preprocess_obj)
                        xgb_clf = sd.get('clf', xgb_clf)
                if preprocess_obj is not None and xgb_clf is not None:
                    return preprocess_obj, xgb_clf
        except Exception:
            pass
        return preprocess_obj, xgb_clf

    st.subheader("Global Explanation")
    # removed card wrappers
    try:
        preprocess_obj, xgb_clf = extract_preprocess_and_clf(model)
        if xgb_clf is not None:
            with st.spinner("Computing feature importance..."):
                enc = None
                try:
                    enc = preprocess_obj.named_steps.get('encoder') if hasattr(preprocess_obj, 'named_steps') else None
                except Exception:
                    enc = None
                feat_names = enc.get_feature_names_out() if enc is not None and hasattr(enc, 'get_feature_names_out') else None

                booster = xgb_clf.get_booster()
                imp = booster.get_score(importance_type='gain')
                if not imp:
                    imp = booster.get_score(importance_type='weight')
                items = []
                for k, v in imp.items():
                    name = k
                    if k.startswith('f'):
                        try:
                            idx = int(k[1:])
                            if feat_names is not None and idx < len(feat_names):
                                name = feat_names[idx]
                        except Exception:
                            name = k
                    items.append((name, float(v)))
                items.sort(key=lambda x: x[1], reverse=True)
                top_k = st.slider("Show top-K features", min_value=5, max_value=30, value=15)
                items = items[:top_k]
                if items:
                    fig, ax = plt.subplots(figsize=(9, 6))
                    names = [it[0] for it in items]
                    scores = [it[1] for it in items]
                    ax.barh(range(len(names)), scores, color='#4C78A8')
                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names)
                    ax.invert_yaxis()
                    ax.set_xlabel('Importance (gain/weight)')
                    ax.set_title('XGBoost Feature Importance')
                    st.pyplot(fig)
                else:
                    st.info("Unable to obtain XGBoost feature importance.")
        else:
            st.info("Unable to extract underlying classifier. Skipping quick importance.")
    except Exception as e_imp:
        st.error(f"Feature importance failed: {e_imp}")

    st.subheader("Permutation Importance")
    # removed card wrappers
    sample_size_pi = st.slider("Validation sample size", min_value=200, max_value=min(5000, len(valid_df)), value=min(1000, len(valid_df)))
    n_repeats = st.slider("Repeats (n_repeats)", min_value=1, max_value=10, value=3)
    if st.button("Compute permutation importance"):
        try:
            df_pi = valid_df.sample(n=min(sample_size_pi, len(valid_df)), random_state=42) if len(valid_df) > sample_size_pi else valid_df
            X_pi = df_pi[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
            y_pi = df_pi['diabetes'] if 'diabetes' in df_pi.columns else None
            if y_pi is None:
                st.info("Validation set missing target column 'diabetes'. Cannot compute permutation importance.")
            else:
                with st.spinner("Calculating permutation importance..."):
                    r = permutation_importance(model, X_pi, y_pi, n_repeats=n_repeats, random_state=42, scoring='roc_auc')
                imp_df = pd.DataFrame({
                    'feature': ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'],
                    'importance_mean': r.importances_mean,
                    'importance_std': r.importances_std,
                }).sort_values('importance_mean', ascending=False)
                st.dataframe(imp_df, use_container_width=True)
                chart = alt.Chart(imp_df).mark_bar().encode(
                    x=alt.X('importance_mean:Q', title='Importance'),
                    y=alt.Y('feature:N', sort=None, axis=alt.Axis(labelLimit=0)),
                    tooltip=[
                        alt.Tooltip('feature:N', title='Feature'),
                        alt.Tooltip('importance_mean:Q', title='Importance', format='.4f')
                    ]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")

    st.subheader("PDP/ICE")
    # removed card wrappers
    num_features = ['age','bmi','HbA1c_level','blood_glucose_level']
    pick_feat = st.selectbox("Select numerical feature for PDP/ICE", options=num_features)
    sample_size_pdp = st.slider("Validation sample size (PDP/ICE)", min_value=200, max_value=min(5000, len(valid_df)), value=min(1000, len(valid_df)))
    grid_points = st.slider("Grid points", min_value=8, max_value=25, value=12)
    ice_n = st.slider("ICE samples", min_value=3, max_value=20, value=6)
    if st.button("Compute PDP/ICE"):
        try:
            df_pdp = valid_df.sample(n=min(sample_size_pdp, len(valid_df)), random_state=42) if len(valid_df) > sample_size_pdp else valid_df
            X_pdp = df_pdp[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
            grid = np.linspace(float(X_valid[pick_feat].min()), float(X_valid[pick_feat].max()), grid_points)

            batch_list = []
            for g in grid:
                tmp = X_pdp.copy()
                tmp[pick_feat] = g
                tmp['_grid'] = g
                batch_list.append(tmp)
            batch = pd.concat(batch_list, axis=0)
            with st.spinner("Computing PDP..."):
                proba_batch = model.predict_proba(batch.drop(columns=['_grid']))[:, 1]
            batch['_proba'] = proba_batch
            pdp_df = batch.groupby('_grid', as_index=False)['_proba'].mean().rename(columns={'_proba': 'avg_proba'})
            st.line_chart(pdp_df.set_index('_grid'))

            ice_idx = np.random.choice(len(X_pdp), size=min(ice_n, len(X_pdp)), replace=False)
            ice_frames = []
            for sid in ice_idx:
                row = X_pdp.iloc[[sid]].copy()
                rows = []
                for g in grid:
                    rtmp = row.copy()
                    rtmp[pick_feat] = g
                    rtmp['_grid'] = g
                    rtmp['_id'] = f'individual_{sid}'
                    rows.append(rtmp)
                ice_frames.append(pd.concat(rows, axis=0))
            ice_batch = pd.concat(ice_frames, axis=0)
            with st.spinner("Computing ICE..."):
                ice_proba = model.predict_proba(ice_batch.drop(columns=['_grid','_id']))[:, 1]
            ice_batch['_proba'] = ice_proba
            ice_pivot = ice_batch.pivot(index='_grid', columns='_id', values='_proba').sort_index()
            st.line_chart(ice_pivot)
        except Exception as e:
            st.error(f"PDP/ICE computation failed: {e}")

    st.subheader("Calibration")
    # removed card wrappers
    try:
        if y_valid is None:
            st.info("Validation set missing target column 'diabetes'. Calibration evaluation is unavailable.")
        else:
            cal_sample = st.slider("Calibration sample size", min_value=200, max_value=min(5000, len(valid_df)), value=min(2000, len(valid_df)))
            df_cal = valid_df.sample(n=min(cal_sample, len(valid_df)), random_state=42) if len(valid_df) > cal_sample else valid_df
            X_cal = df_cal[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
            y_cal = df_cal['diabetes']
            with st.spinner("Computing calibration..."):
                proba_valid = model.predict_proba(X_cal)[:, 1]
                brier = brier_score_loss(y_cal, proba_valid)
                frac_pos, mean_pred = calibration_curve(y_cal, proba_valid, n_bins=10, strategy='uniform')
            st.metric(label="Brier Score", value=f"{brier:.4f}", help="Probability calibration error (lower is better, 0 is perfect)")
            cal_df = pd.DataFrame({
                'mean_predicted_value': mean_pred,
                'fraction_of_positives': frac_pos
            })
            diag_df = pd.DataFrame({'x':[0,1],'y':[0,1]})
            cal_chart = alt.layer(
                alt.Chart(cal_df).mark_line(point=True).encode(
                    x=alt.X('mean_predicted_value:Q', title='Mean predicted probability'),
                    y=alt.Y('fraction_of_positives:Q', title='Fraction of positives (observed)')
                ),
                alt.Chart(diag_df).mark_line(color='gray', strokeDash=[4,4]).encode(
                    x='x:Q', y='y:Q'
                )
            ).properties(height=300)
            st.altair_chart(cal_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Calibration computation failed: {e}")



def page_fairness():
    st.header("Fairness")
    valid_df, _ = load_eval_data()
    if valid_df is None:
        st.warning("Validation file valid.csv not found. Fairness evaluation is unavailable.")
        return
    X_valid = valid_df[[
        'gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'
    ]]
    y_valid = valid_df['diabetes']

    try:
        with st.spinner("Computing fairness metrics..."):
            proba = model.predict_proba(X_valid)[:, 1]
        thr = st.slider("Threshold for fairness evaluation", 0.0, 1.0, value=effective_default_threshold, step=0.01)
        preds = (proba >= thr).astype(int)

        def group_metrics(df, group_col):
            res = []
            for g, sub in df.groupby(group_col):
                y_true = sub['diabetes'].values
                y_pred = preds[sub.index]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tpr
                res.append({'group': str(g), 'TPR': tpr, 'FPR': fpr, 'Precision': prec, 'Recall': rec, 'count': len(sub)})
            return pd.DataFrame(res)

        st.subheader("By Gender")
        # removed residual card wrapper
        gm_gender = group_metrics(valid_df[['diabetes','gender']], 'gender')
        st.dataframe(gm_gender, use_container_width=True)
        gender_chart = (
            alt.Chart(gm_gender)
            .transform_fold(["TPR", "FPR"], as_=["metric", "value"])
            .mark_bar()
            .encode(
                x=alt.X("group:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("metric:N", legend=alt.Legend(title=None)),
            )
            .properties(height=300)
        )
        st.altair_chart(gender_chart, use_container_width=True)
        # removed residual card wrapper
        
        st.subheader("By Smoking History")
        # removed residual card wrapper
        gm_smoke = group_metrics(valid_df[['diabetes','smoking_history']], 'smoking_history')
        st.dataframe(gm_smoke, use_container_width=True)
        smoke_chart = (
            alt.Chart(gm_smoke)
            .transform_fold(["TPR", "FPR"], as_=["metric", "value"])
            .mark_bar()
            .encode(
                x=alt.X("group:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("metric:N", legend=alt.Legend(title=None)),
            )
            .properties(height=300)
        )
        st.altair_chart(smoke_chart, use_container_width=True)
        # removed residual card wrapper
        st.caption("Note: This page uses the validation set for grouped metrics to support preliminary fairness checks. Consider further intersectional analysis and statistical significance testing.")
    except Exception as e:
        st.error(f"Fairness evaluation failed: {e}")


# Router
tab1, tab2, tab3 = st.tabs(["Predict", "Interpretability", "Fairness"])
with tab1:
    page_predict()
with tab2:
    page_interpretability()
with tab3:
    page_fairness()