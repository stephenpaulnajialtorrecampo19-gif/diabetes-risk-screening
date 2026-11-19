# Diabetes Risk Screening

A web application that uses a trained machine learning model to estimate whether a person's health and lifestyle answers look similar to people who reported diabetes. It is a screening aid, **not a medical diagnosis**.

**Course:** CPE ELEC 1 - CpE Elective 1 (Final Project)  
**Student:** TORRECAMPO

## 1. Problem

Many people have diabetes or are at high risk without knowing it because they are rarely screened. This app gives a quick self-check so a person can decide whether to get a proper blood sugar test. Intended users are adults who have not been screened, and community health workers who need a fast pre-screening aid.

## 2. How it works

1. The user fills in a short form (age, sex, height, weight, health conditions, lifestyle).
2. The app validates the answers and calculates BMI.
3. The trained model (loaded from `model/diabetes_model.joblib`) returns a risk score.
4. The app shows the score, a Higher risk / Lower risk result, suggestions and a disclaimer.

## 3. Dataset

CDC Diabetes Health Indicators, UCI Machine Learning Repository (dataset 891), based on the 2015 BRFSS health survey. See [DATASET.md](DATASET.md) for the full documentation (source, record count, features, target, preparation steps).

Download: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

## 4. Model and evaluation

- Task: binary classification (0 = no diabetes, 1 = prediabetes or diabetes).
- Models compared: Baseline (always predicts the majority class), Logistic Regression, Random Forest.
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.
- The best model by ROC-AUC is saved and used by the app.
- Results are saved in `results/` (`model_results.md`, `model_results.csv`, charts) and shown in the app under **About the model**.

Because only about 1 in 7 people in the data has diabetes, accuracy alone is misleading, so recall for the diabetes class is the metric that matters most for screening.

### Classification threshold
The model gives each person a score from 0 to 1. A score of 0.5 or higher is shown as **Higher risk**. Because this is a screening tool, missing a person who has diabetes is worse than an extra blood test, so recall is more important than precision. Notebook 02 compares thresholds from 0.2 to 0.8 (`results/threshold_analysis.csv` and `.png`) and the app shows the trade-off: a lower threshold finds more cases but flags many more people who do not have diabetes.

### How the scores should be interpreted
The classes were balanced during training, so the scores are **not calibrated probabilities**. Notebook 02 measures this with the Brier score and a calibration curve (`results/calibration_curve.png`). To make the score meaningful, the app shows how often people in each score range actually reported diabetes in the held-out test data (`results/score_ranges.csv`). For example, "about X% of people with a score of 60% to 80% reported diabetes". The app never presents the score as a clinical risk estimate.

## 5. Project structure

```
app.py                                        Streamlit user interface
predictor.py                                  Validation, input preparation and prediction logic
01_data_preparation_and_model_training.ipynb  Data preparation, training, evaluation, saving the model
02_threshold_and_score_reliability.ipynb      Threshold analysis and reliability of the risk scores
model/                                        diabetes_model.joblib and model_info.json (created by the notebook)
results/                                      Evaluation tables and charts (created by the notebook)
data/                                         Place the dataset CSV here (not uploaded to GitHub)
tests/test_predictor.py                       Unit tests
requirements.txt                              Python packages
```

## 6. Setup and usage

### Train the model (once)
1. Download the dataset and keep `diabetes_binary_health_indicators_BRFSS2015.csv`.
2. Put it in the `data/` folder (in Google Drive: `FINAL PROJECT TORRECAMPO/data/`).
3. Open `01_data_preparation_and_model_training.ipynb` in Google Colab and choose Runtime > Run all.
4. The notebook creates the `model/` and `results/` files and an updated `requirements.txt`.

### Analyze the threshold and score reliability (once, after training)
Run `02_threshold_and_score_reliability.ipynb` in Google Colab (Runtime > Run all). It adds the threshold table, score ranges and charts to `model/model_info.json` and `results/`, which the app then displays.

### Run the app on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
Then open the address shown in the terminal (usually http://localhost:8501).

### Run the tests
```
pip install pytest
python -m pytest tests -q
```

### Deploy (optional)
Push the repository to GitHub, then create an app on Streamlit Community Cloud, select the repository and set `app.py` as the main file.

## 7. Limitations and responsible use

- **Not a diagnosis and not clinically validated.** The app is a screening aid. Only a blood test (fasting glucose or HbA1c) from a health professional can confirm diabetes. The score must not be used as an established clinical risk estimate.
- **US data, not Philippine data.** The data are self-reported answers of US adults in 2015. Diet, healthcare access, body type and diabetes rates differ in the Philippines. For example, studies show that people of Asian descent can develop type 2 diabetes at lower BMI values than shown in US data, so the model may under-estimate risk for some Filipinos. The model should be validated with local data before any real use.
- **Self-reported inputs.** Height, weight, blood pressure, cholesterol and lifestyle answers are not measured by a professional.
- **Limited target variable.** A person is labeled 1 only if they reported being told they have diabetes or prediabetes. People with undiagnosed diabetes are labeled 0, so the model learns who reports a diagnosis, not who truly has the disease.
- **Scores are not probabilities.** See the section on how the scores should be interpreted above.
- **Survey data from 2015.** Patterns may have changed.

## 8. AI tools used

Claude (Anthropic) was used to help plan the project, write and review code, and draft documentation. The student ran the notebook, tested the application and is able to explain the code, data, model and results.
