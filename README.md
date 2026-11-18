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

## 5. Project structure

```
app.py                                        Streamlit user interface
predictor.py                                  Validation, input preparation and prediction logic
01_data_preparation_and_model_training.ipynb  Data preparation, training, evaluation, saving the model
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

## 7. Limitations

- The model learns from 2015 US survey answers. It may not fit every population.
- The risk score is not the probability of having diabetes. The classes were balanced during training, so the score is a relative indicator.
- The app does not replace a blood test or a doctor.

## 8. AI tools used

Claude (Anthropic) was used to help plan the project, write and review code, and draft documentation. The student ran the notebook, tested the application and is able to explain the code, data, model and results.
