import os
import pandas as pd
import streamlit as st
import predictor

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")


@st.cache_resource
def get_model():
    return predictor.load_model()


@st.cache_data
def get_info():
    return predictor.load_model_info()


# Simple styling: a calm header banner, a card look for the form and result numbers.
st.markdown(
    """
    <style>
    .app-header {
        background: linear-gradient(135deg, #17496B 0%, #1F6F78 100%);
        color: white;
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { color: white; margin: 0; padding: 0; font-size: 2rem; }
    .app-header p { color: #D6E6EA; margin: 0.4rem 0 0 0; }
    [data-testid="stForm"] {
        background: white;
        border: 1px solid #D5DFE5;
        border-radius: 14px;
        padding: 1.4rem;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #D5DFE5;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    </style>
    <div class="app-header">
        <h1>🩺 Diabetes Risk Screening</h1>
        <p>A screening aid built with machine learning. It is not a medical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = get_model()
    info = get_info()
except FileNotFoundError:
    st.error("The trained model files were not found. Run the training notebook first, then put the files in the 'model' folder.")
    st.stop()

tab_check, tab_help, tab_about = st.tabs(["Check my risk", "How to use", "About the model"])

with tab_check:
    st.write("Fill in the form and press **Check my risk**.")
    with st.form("risk_form"):
        st.subheader("About you")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=0, max_value=130, value=40, step=1)
        sex = c2.selectbox("Sex", ["Female", "Male"])
        c3, c4 = st.columns(2)
        height = c3.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=165.0, step=1.0)
        weight = c4.number_input("Weight (kg)", min_value=0.0, max_value=400.0, value=65.0, step=0.5)

        st.subheader("Health conditions")
        h1, h2, h3 = st.columns(3)
        high_bp = h1.radio("High blood pressure", ["No", "Yes"], horizontal=True)
        high_chol = h2.radio("High cholesterol", ["No", "Yes"], horizontal=True)
        diff_walking = h3.radio("Difficulty walking or climbing stairs", ["No", "Yes"], horizontal=True)
        h4, h5 = st.columns(2)
        stroke = h4.radio("Ever had a stroke", ["No", "Yes"], horizontal=True)
        heart = h5.radio("Heart disease or heart attack", ["No", "Yes"], horizontal=True)

        st.subheader("Lifestyle")
        l1, l2, l3 = st.columns(3)
        smoker = l1.radio("Smoked at least 100 cigarettes in your life", ["No", "Yes"], horizontal=True)
        phys_activity = l2.radio("Physical activity in the past 30 days", ["No", "Yes"], index=1, horizontal=True)
        heavy_alcohol = l3.radio("Heavy alcohol drinking", ["No", "Yes"], horizontal=True)
        l4, l5 = st.columns(2)
        fruits = l4.radio("Eat fruit every day", ["No", "Yes"], index=1, horizontal=True)
        veggies = l5.radio("Eat vegetables every day", ["No", "Yes"], index=1, horizontal=True)

        st.subheader("General well-being")
        gen_label = st.select_slider(
            "How would you rate your general health?",
            options=["Excellent", "Very good", "Good", "Fair", "Poor"],
            value="Good",
        )
        w1, w2 = st.columns(2)
        ment_days = w1.number_input("Days of poor mental health (last 30 days)", min_value=0, max_value=60, value=0, step=1)
        phys_days = w2.number_input("Days of poor physical health (last 30 days)", min_value=0, max_value=60, value=0, step=1)

        submitted = st.form_submit_button("Check my risk", type="primary")

    if submitted:
        yes_no = {"No": 0, "Yes": 1}
        general_health = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
        data = {
            "age": age,
            "sex": sex,
            "height_cm": height,
            "weight_kg": weight,
            "high_bp": yes_no[high_bp],
            "high_chol": yes_no[high_chol],
            "smoker": yes_no[smoker],
            "stroke": yes_no[stroke],
            "heart_disease": yes_no[heart],
            "phys_activity": yes_no[phys_activity],
            "fruits": yes_no[fruits],
            "veggies": yes_no[veggies],
            "heavy_alcohol": yes_no[heavy_alcohol],
            "diff_walking": yes_no[diff_walking],
            "gen_health": general_health[gen_label],
            "ment_health_days": ment_days,
            "phys_health_days": phys_days,
        }

        errors = predictor.validate_inputs(data)
        if len(errors) > 0:
            for message in errors:
                st.error(message)
        else:
            try:
                result = predictor.predict_risk(model, data, info)
            except Exception:
                st.error("Something went wrong while making the prediction. Please check your inputs and try again.")
                st.stop()

            st.divider()
            st.subheader("Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk score", str(result["percent"]) + "%")
            m2.metric("Result", result["label"])
            m3.metric("Your BMI", result["bmi"])
            st.progress(min(max(result["score"], 0.0), 1.0))

            if result["higher_risk"]:
                st.warning("Your answers look similar to people who reported diabetes. Please consider seeing a doctor for a blood sugar test.")
            else:
                st.success("Your answers look less similar to people who reported diabetes. Keep up healthy habits and get regular check-ups.")

            band = predictor.get_score_band(result["score"], info)
            if band is not None:
                st.info(
                    "**How to read this score:** in the test data, about "
                    + str(band["diabetes_rate"]) + "% of the " + format(band["count"], ",")
                    + " people with a score of " + band["label"]
                    + " reported having diabetes or prediabetes. The score is not a personal probability."
                )
            summary = predictor.threshold_summary(info)
            if summary is not None:
                st.caption(
                    "A score of " + str(int(info["threshold"] * 100)) + "% or higher is shown as Higher risk. "
                    "On the test data, this setting found " + str(round(summary["Recall"] * 100)) + "% of the people who reported diabetes, "
                    "but only " + str(round(summary["Precision"] * 100)) + "% of the people it flagged reported diabetes, so many flags are false alarms. "
                    "It is meant for screening, not diagnosis."
                )

            st.write("**Suggestions**")
            for tip in predictor.health_tips(data):
                st.write("- " + tip)

            st.caption("The risk score comes from a machine learning model trained on survey data. It is not a probability of having diabetes and it is not a diagnosis.")

with tab_help:
    st.subheader("How to use this app")
    st.markdown(
        """
1. Open the **Check my risk** tab.
2. Enter your age, sex, height and weight.
3. Answer the health and lifestyle questions honestly.
4. Press **Check my risk**.
5. Read the risk score, the result and the suggestions.

**Good to know**
- The model uses the answers you give. It does not store them.
- If something is out of range (for example a height of 20 cm), the app shows an error and asks you to fix it.
- This tool only screens. Only a blood test from a health professional can confirm diabetes.
"""
    )

with tab_about:
    st.subheader("About the model")
    st.write("**Dataset:** " + info["dataset"]["name"] + " (" + info["dataset"]["source"] + ")")
    st.write("**Records used:** " + format(info["dataset"]["records_after_cleaning"], ",") + " survey responses")
    st.write("**Selected model:** " + info["selected_model"])
    st.write("**Inputs used by the model:** " + ", ".join(info["features"]))

    st.write("**Model comparison on the test set**")
    results = pd.DataFrame(info["results"]).set_index("Model")
    st.dataframe(results.style.format("{:.3f}"))
    st.caption("The baseline always predicts 'no diabetes'. It looks accurate because most people in the data do not have diabetes, but it finds none of the people who do (recall 0).")

    if info.get("threshold_analysis"):
        st.write("**Choosing the threshold**")
        st.write("People with a score at or above the threshold are shown as Higher risk. A lower threshold finds more people with diabetes (higher recall) but flags more people who do not have it (lower precision). This app uses " + str(int(info["threshold"] * 100)) + "% as its threshold.")
        threshold_table = pd.DataFrame(info["threshold_analysis"]).set_index("Threshold")
        st.dataframe(threshold_table.style.format("{:.3f}"))

    if info.get("score_bands"):
        st.write("**What each score range means in the test data**")
        bands = pd.DataFrame(info["score_bands"])[["label", "count", "diabetes_rate"]]
        bands.columns = ["Score range", "People in test set", "Reported diabetes (%)"]
        st.dataframe(bands.set_index("Score range"))
        if info.get("brier_score") is not None:
            st.caption("Brier score of the raw scores: " + str(info["brier_score"]) + " (lower is better; always guessing the average gives " + str(info["brier_score_baseline"]) + "). The scores are not calibrated probabilities, so use the table above to interpret them.")

    for file_name, title in [("confusion_matrix.png", "Confusion matrix"), ("roc_curve.png", "ROC curve"), ("threshold_analysis.png", "Precision and recall at different thresholds"), ("calibration_curve.png", "Reliability of the scores (calibration curve)"), ("feature_importance.png", "Most important inputs")]:
        path = os.path.join(predictor.BASE_DIR, "results", file_name)
        if os.path.exists(path):
            st.image(path, caption=title)

    st.subheader("Limitations and responsible use")
    st.markdown(
        """
- **Not a diagnosis.** The model has not been clinically validated. Only a blood test (such as fasting glucose or HbA1c) from a health professional can confirm diabetes.
- **US data, not Philippine data.** The model learned from a 2015 survey of adults in the United States. Diet, healthcare access, body type and diabetes rates differ in the Philippines. For example, studies show that people of Asian descent can develop type 2 diabetes at lower BMI values, so the app may under-estimate risk for some Filipinos. It should be checked with local data before any real use.
- **Self-reported answers.** All inputs and the target come from what survey respondents said about themselves, not from medical records or lab tests.
- **The target is limited.** A person is labeled "diabetes" only if they reported being told they have diabetes or prediabetes. People who have diabetes but were never diagnosed are labeled as "no diabetes", so the model learns who reports a diagnosis, not who truly has the disease.
- **The score is not a probability.** The model was trained with balanced classes, so scores run higher than real rates. Use the score ranges table to see how often each range reported diabetes.
"""
    )

st.divider()
st.write("Tried the app? Tell us what you think.")
st.link_button(
    "Give feedback",
    "https://docs.google.com/forms/d/e/1FAIpQLSeeF1P4_MnmRjKddfx4l8AjiZwwQhmbDby_xB7JVb4cPcvYRg/viewform",
)
