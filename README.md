# ThermoTwin-IN
## A Personal Heat-Strain Digital Twin for Indian Outdoor Workers

**Participant:** Iswarya Suresh Kumar  
**Institution:** Stella Mary's College of Engineering  
**Challenge:** Happiest Health Digital Twin Challenge 2026  
**Prototype status:** Educational proof of concept; all data and outcomes are synthetic.

### 1. Problem
Outdoor workers can face high heat exposure while doing physically demanding work. A simple fixed threshold may not represent each person's normal readings or changing exposure. ThermoTwin-IN demonstrates a virtual worker profile that combines a personal baseline and historical context with changing environmental and wearable-style readings.

### 2. What makes this prototype distinctive
- **Personal-baseline comparison:** compares current heart rate with a simulated personal baseline.
- **Exposure-aware state:** combines temperature, humidity, activity intensity, exposure duration, sleep and hydration index.
- **What-if simulator:** compares the current scenario with a hypothetical shade/rest scenario.
- **Sensor trust indicator:** shows when readings are delayed/noisy and warns that confidence is reduced.
- **Indian outdoor-worker use case:** designed around simulated delivery, construction and agricultural work contexts.

### 3. Safety and scientific limits
This is not a medical device. The dataset and outcome labels are fully synthetic and are not validated clinical measurements. Risk bands and scenario deltas are educational simulations only; do not use them for diagnosis, treatment, fitness-for-work decisions or emergency response. Real-world deployment would require validated environmental/physiological models, appropriate wet-bulb globe temperature (WBGT) or equivalent measurements, ethically approved data, clinical/occupational-health review, external validation, privacy protection and worker consultation.

### 4. Tech stack
Python, Pandas, NumPy, scikit-learn, Streamlit, Matplotlib and Joblib.

### 5. Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### 6. Data and model
- Dataset: `data/synthetic_heat_worker_data.csv`
- Target: `simulated_heat_strain_label`
- Model: Random Forest classifier, trained/tested on synthetic labels.
- Metrics describe fit to this generated dataset only; they do not establish real-world performance.
- App can train the model automatically if no saved model exists.

### 7. Digital-twin workflow
1. Load a simulated historical worker profile and personal baseline.
2. Update environmental and wearable-style readings.
3. Compare current heart rate to the personal baseline.
4. Estimate a synthetic heat-strain risk score.
5. Compare current conditions with a hypothetical shade/rest scenario.
6. Show sensor-quality warnings and an explanatory dashboard.

### 8. Submission checklist
- [ ] Upload this folder to a public GitHub repository.
- [ ] Confirm the repository opens in an incognito/logged-out browser.
- [ ] Record a walkthrough of the actual working app and upload it to an accessible host.
- [ ] Add the demo video URL to this README.
- [ ] Review the registered Unstop page for exact required submission fields and video requirements.
- [ ] Submit the public GitHub URL and any required video URL through the official form before the deadline.

**GitHub URL:** Add after creating your repository.  
**Demo video URL:** Add after recording.

### 9. Future work
Replace synthetic rules with suitable ethically sourced data, use validated occupational heat metrics, model uncertainty, assess subgroup performance, and conduct external validation with occupational-health experts.
