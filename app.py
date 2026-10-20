from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_heat_worker_data.csv"
MODEL_PATH = ROOT / "model" / "thermotwin_model.joblib"
TARGET = "simulated_heat_strain_label"
FEATURES = ["worker_type","age_years","ambient_temperature_c","relative_humidity_pct",
            "activity_intensity_1_to_5","heart_rate_bpm","personal_baseline_hr_bpm",
            "sleep_hours","hydration_index_0_to_100","heat_exposure_minutes","sensor_quality"]

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    cat = ["worker_type","sensor_quality"]
    num = [x for x in FEATURES if x not in cat]
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),
                             ("num","passthrough",num)])
    m = Pipeline([("pre",pre),("rf",RandomForestClassifier(n_estimators=160,max_depth=9,
                 min_samples_leaf=4,class_weight="balanced",random_state=42))])
    m.fit(X,y)
    return m

st.set_page_config(page_title="ThermoTwin-IN", page_icon="🌡️", layout="wide")
st.title("🌡️ ThermoTwin-IN")
st.subheader("Personal heat-strain digital twin for Indian outdoor workers")
st.warning("Educational simulation only. All data and labels are synthetic. This is not a medical device or a validated heat-illness predictor. Do not use it for real-world safety decisions.")

with st.sidebar:
    st.header("Virtual worker profile")
    worker = st.selectbox("Work context", ["delivery","construction","agriculture"])
    age = st.slider("Age (years)", 18, 65, 32)
    baseline = st.slider("Personal baseline heart rate (bpm)", 48, 100, 68)
    sleep = st.slider("Sleep last night (hours)", 2.5, 10.0, 6.5, .5)
    hydration = st.slider("Hydration index (simulated 0–100)", 0, 100, 65)
    st.header("Current conditions")
    temp = st.slider("Ambient temperature (°C)", 20.0, 48.0, 38.0, .5)
    humidity = st.slider("Relative humidity (%)", 15, 95, 60)
    activity = st.slider("Activity intensity (1 low – 5 very high)", 1, 5, 4)
    hr = st.slider("Current heart rate (bpm)", 45, 190, 112)
    exposure = st.slider("Heat exposure duration (minutes)", 5, 240, 75, 5)
    quality = st.selectbox("Sensor quality", ["good","delayed","noisy"])

row = {"worker_type":worker, "age_years":age, "ambient_temperature_c":temp,
       "relative_humidity_pct":humidity, "activity_intensity_1_to_5":activity,
       "heart_rate_bpm":hr, "personal_baseline_hr_bpm":baseline,
       "sleep_hours":sleep, "hydration_index_0_to_100":hydration,
       "heat_exposure_minutes":exposure, "sensor_quality":quality}
model = load_model()
risk = float(model.predict_proba(pd.DataFrame([row], columns=FEATURES))[0,1])
delta_hr = hr - baseline

col1, col2, col3 = st.columns(3)
col1.metric("Simulated strain-risk score", f"{risk*100:.0f}%")
col2.metric("Heart rate vs personal baseline", f"{delta_hr:+.0f} bpm")
col3.metric("Heat exposure", f"{exposure} min")
if quality != "good":
    st.warning("Sensor quality is marked as delayed/noisy. Treat the displayed score as lower-confidence; a real system should avoid confident predictions from unreliable readings.")
if risk >= .65:
    st.error("Higher simulated risk band — illustrative only")
elif risk >= .35:
    st.warning("Moderate simulated risk band — illustrative only")
else:
    st.success("Lower simulated risk band — illustrative only")
st.caption("Risk bands are demonstration thresholds, not occupational or medical cut-offs.")

st.markdown("## What-if scenario: rest in shade")
st.write("The next comparison is a simple scenario illustration, not a validated physiological forecast.")
rest_minutes = st.slider("Hypothetical rest duration (minutes)", 5, 45, 20, 5)
# Simple transparent illustrative scenario transforms, not medical claims.
rest_row = dict(row)
rest_row["ambient_temperature_c"] = max(20, temp - 4.0)
rest_row["activity_intensity_1_to_5"] = max(1, activity - 2)
rest_row["heat_exposure_minutes"] = max(0, exposure - rest_minutes)
rest_row["heart_rate_bpm"] = max(baseline, hr - min(18, rest_minutes*0.45))
rest_risk = float(model.predict_proba(pd.DataFrame([rest_row], columns=FEATURES))[0,1])
a,b = st.columns(2)
a.metric("Current scenario score", f"{risk*100:.0f}%")
b.metric("Hypothetical rest/shade score", f"{rest_risk*100:.0f}%", delta=f"{(rest_risk-risk)*100:+.0f} percentage points", delta_color="inverse")
fig, ax = plt.subplots()
ax.bar(["Current\nscenario","Hypothetical\nrest + shade"],[risk*100,rest_risk*100])
ax.set_ylabel("Model score (%)")
ax.set_ylim(0,100)
ax.set_title("Scenario comparison (synthetic model output)")
st.pyplot(fig, clear_figure=True)

st.markdown("## Why this is a digital twin")
x,y = st.columns(2)
x.info("**Historical/personal stream**\n\nWork context, age, personal baseline heart rate and sleep/hydration context.")
y.info("**Dynamic stream**\n\nTemperature, humidity, activity, current heart rate, exposure duration and sensor quality.")
st.markdown("## Limitations")
st.write("Training data and labels are generated synthetically. The rest/shade scenario changes inputs using illustrative assumptions. Neither the model score nor scenario difference is validated evidence of real heat-strain reduction. A deployable system needs validated heat metrics, approved data, external testing and occupational-health oversight.")
