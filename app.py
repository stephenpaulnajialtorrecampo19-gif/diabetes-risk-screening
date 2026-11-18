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


st.title("🩺 Diabetes Risk Screening")
st.caption("A screening aid built with machine learning. It is not a medical diagnosis.")

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

    for file_name, title in [("confusion_matrix.png", "Confusion matrix"), ("roc_curve.png", "ROC curve"), ("feature_importance.png", "Most important inputs")]:
        path = os.path.join(predictor.BASE_DIR, "results", file_name)
        if os.path.exists(path):
            st.image(path, caption=title)
