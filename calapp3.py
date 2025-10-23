import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings


model = joblib.load("C:/Users/Abhishek/OneDrive/Documents/DeploymentML/excel_only_calorie_predictor_model_v2.pkl")

# Dynamic MET values with adjusted Strength METs
def get_dynamic_met(workout_type, intensity, age, duration, experience_level):
    base_mets = {
        "Cardio": {"low": 4.0, "medium": 6.0, "high": 8.0},
        "Strength": {"low": 3.0, "medium": 5.0, "high": 6.5},  # Lowered for rest periods
        "HIIT": {"low": 6.0, "medium": 8.0, "high": 10.0},
        "Yoga": {"low": 2.5, "medium": 3.0, "high": 4.0}
    }
    age_factor = 1.0 if age < 30 else 0.95 if age < 50 else 0.9 if age < 70 else 0.85
    duration_factor = 1.0 if 0.5 <= duration <= 1.5 else 0.95 if duration < 0.5 else 0.85 + 0.05 * min(duration - 1.5, 3.5)
    experience_factor = 0.9 if experience_level == 1 else 1.0 if experience_level == 2 else 1.1
    intensity = min(intensity, 1.0)
    if workout_type == "Yoga" and intensity <= 0.55:
        return base_mets[workout_type]["low"] * age_factor * duration_factor * experience_factor
    elif intensity <= 0.5:
        return base_mets[workout_type]["low"] * age_factor * duration_factor * experience_factor
    elif intensity <= 0.7:
        return base_mets[workout_type]["medium"] * age_factor * duration_factor * experience_factor
    else:
        return base_mets[workout_type]["high"] * age_factor * duration_factor * experience_factor

# Constrain predictions
def constrain_prediction(row, prediction):
    weight = row["Weight (kg)"]
    duration = row["Session_Duration (hours)"]
    workout_type = row["Workout_Type"]
    intensity = row["Intensity"]
    age = row["Age"]
    experience_level = row["Experience_Level"]
    met = get_dynamic_met(workout_type, intensity, age, duration, experience_level)
    met_estimate = met * weight * duration
    constraint_factor = 0.05 if duration < 0.5 else 0.1
    max_allowed = met_estimate * (1 + constraint_factor)
    min_allowed = met_estimate * (1 - constraint_factor)
    return min(max(prediction, min_allowed), max_allowed), met, met_estimate

# Prediction function for calorie burn
def predict_calories(age, gender, weight_kg, height_m, max_bpm, avg_bpm, resting_bpm, duration_hours, workout_type, fat_percentage, water_intake, workout_frequency, experience_level):
    if duration_hours <= 0:
        raise ValueError("Duration must be positive")
    if workout_type not in ["Yoga", "Cardio", "Strength", "HIIT"]:
        raise ValueError("Invalid Workout Type")
    if experience_level not in [1, 2, 3]:
        raise ValueError("Experience Level must be 1 (Beginner), 2 (Intermediate), or 3 (Expert)")
    if resting_bpm >= avg_bpm:
        warnings.warn(f"Avg_BPM ({avg_bpm}) should be greater than Resting_BPM ({resting_bpm}). Setting Avg_BPM to Resting_BPM + 1.")
        avg_bpm = resting_bpm + 1
    if avg_bpm > max_bpm:
        warnings.warn(f"Avg_BPM ({avg_bpm}) exceeds Max_BPM ({max_bpm}). Capping Avg_BPM at Max_BPM.")
        avg_bpm = max_bpm
    intensity = (avg_bpm - resting_bpm) / (max_bpm - resting_bpm)
    bmi = weight_kg / (height_m ** 2)
    duration_adj_intensity = intensity * np.log1p(duration_hours) * (2 if duration_hours > 1 else 3)
    hrr_percentage = (avg_bpm - resting_bpm) / (220 - age - resting_bpm)
    # Define Gender Factor before using it
    gender_factor = 0 if gender == "Male" else 1
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Gender_Factor": [gender_factor],
        "Weight (kg)": [weight_kg],
        "Height (m)": [height_m],
        "Max_BPM": [max_bpm],
        "Avg_BPM": [avg_bpm],
        "Resting_BPM": [resting_bpm],
        "Session_Duration (hours)": [duration_hours],
        "Workout_Type": [workout_type],
        "Fat_Percentage": [fat_percentage],
        "Water_Intake (liters)": [water_intake],
        "Workout_Frequency (days/week)": [workout_frequency],
        "Experience_Level": [experience_level],
        "Intensity": [intensity],
        "BMI": [bmi],
        "Duration_Adjusted_Intensity": [duration_adj_intensity],
        "HRR_Percentage": [hrr_percentage]
    })
    
    raw_prediction = model.predict(input_data)[0]

   # ↓ Apply female calorie correction
    if gender.lower() == "female":
       raw_prediction *= 0.9

    constrained_prediction, met, met_estimate = constrain_prediction(input_data.iloc[0], raw_prediction)
    return raw_prediction, constrained_prediction, met, met_estimate
 

# Maintenance calories calculation using Mifflin-St Jeor Equation
def calculate_maintenance_calories(age, gender, weight_kg, height_m, workout_frequency):
    height_cm = height_m * 100
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    if workout_frequency <= 2:
        activity_factor = 1.2  # Sedentary
    elif workout_frequency <= 4:
        activity_factor = 1.375  # Lightly active
    elif workout_frequency <= 6:
        activity_factor = 1.55  # Moderately active
    else:
        activity_factor = 1.725  # Very active
    tdee = bmr * activity_factor
    return bmr, tdee

# Smoking status check
def check_smoking_status(avg_bpm, max_bpm, smoking_status):
    intensity = (avg_bpm - 70) / (max_bpm - 70)  # Assuming resting BPM = 70 for simplicity
    high_bpm = avg_bpm > 0.7 * max_bpm
    high_intensity = intensity > 0.7
    moderate_intensity = 0.5 <= intensity <= 0.7
    cautions = []
    if smoking_status == "Very Much":
        cautions.append("⚠️ **Heavy Smoker Warning**: Smoking heavily increases cardiovascular risks. Consult a doctor before engaging in any exercise, especially if your heart rate or intensity is high.")
    elif smoking_status == "Occasional":
        if high_bpm or high_intensity:
            cautions.append("⚠️ **Occasional Smoker Caution**: Your heart rate or exercise intensity is high. As an occasional smoker, consult a doctor before continuing intense workouts to ensure safety.")
        elif moderate_intensity:
            cautions.append("⚠️ **Occasional Smoker Note**: Moderate exercise intensity detected. Occasional smoking may still pose risks; consider consulting a doctor for personalized advice.")
    elif smoking_status == "Very Less":
        if high_bpm or high_intensity:
            cautions.append("⚠️ **Light Smoker Note**: High heart rate or intensity detected. Even light smoking can affect cardiovascular health. Monitor your condition and consult a doctor if you experience discomfort.")
    else:  # None
        if high_bpm or high_intensity:
            cautions.append("ℹ️ **Note**: High heart rate or intensity detected. Ensure you are cleared for intense exercise, especially if you have other health concerns.")
    return cautions

# Streamlit app
st.title("🏋️‍ Activity-based Calorie Burn Estimator")
st.markdown("Estimate calories burned during workouts, calculate maintenance calories, or check health cautions based on smoking status.")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Calorie Burn Estimator", "Maintenance Calories", "Smoking Status Check"])

# Tab 1: Calorie Burn Estimator
with tab1:
    st.header("Calorie Burn Estimator")
    st.info("Note: Strength workout calorie estimates account for rest periods, using lower MET values (e.g., 6.5 for high intensity) compared to continuous activities like Cardio.")
    
    with st.form("calorie_form"):
        st.subheader("Workout Details")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=20, help="Enter your age (10–100 years).", key="calorie_age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.", index=0, key="calorie_gender")
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=76.0, step=0.1, help="Enter your weight in kilograms (30–200 kg).", key="calorie_weight")
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.84, step=0.01, help="Enter your height in meters (1.0–2.5 m).", key="calorie_height")
            fat_percentage = st.number_input("Fat Percentage (%)", min_value=5.0, max_value=50.0, value=15.0, step=0.1, help="Enter your body fat percentage (5–50%).", key="calorie_fat")
        
        with col2:
            max_bpm = st.number_input("Max BPM", min_value=80, max_value=220, value=140, help="Enter your maximum heart rate (80–220 BPM).", key="calorie_max_bpm")
            avg_bpm = st.number_input("Average BPM", min_value=60, max_value=200, value=120, help="Enter your average heart rate during workout (60–200 BPM).", key="calorie_avg_bpm")
            resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=70, help="Enter your resting heart rate (40–100 BPM).", key="calorie_resting_bpm")
            duration = st.number_input("Session Duration (hours)", min_value=0.05, max_value=5.0, value=1.5, step=0.05, help="Enter workout duration in hours (0.05–5.0 hours).", key="calorie_duration")
            workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=5, help="Enter workouts per week (1–7 days).", key="calorie_frequency")
        
        workout_type = st.selectbox("Workout Type", ["Yoga", "Cardio", "Strength", "HIIT"], help="Select the type of workout.", index=2, key="calorie_workout")
        experience_level = st.selectbox("Experience Level", [1, 2, 3], format_func=lambda x: {1: "Beginner", 2: "Intermediate", 3: "Expert"}[x], help="Select your experience level: 1 (Beginner), 2 (Intermediate), 3 (Expert).", index=1, key="calorie_experience")
        water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, value=2.5, step=0.1, help="Enter daily water intake (0.5–5.0 liters).", key="calorie_water")
        submit_button = st.form_submit_button("Estimate Calories Burned")

    if submit_button:
        try:
            raw, constrained, met, met_estimate = predict_calories(
                age=age,
                gender=gender,
                weight_kg=weight,
                height_m=height,
                max_bpm=max_bpm,
                avg_bpm=avg_bpm,
                resting_bpm=resting_bpm,
                duration_hours=duration,
                workout_type=workout_type,
                fat_percentage=fat_percentage,
                water_intake=water_intake,
                workout_frequency=workout_frequency,
                experience_level=experience_level
            )
            st.success(f" **Estimated Calories Burned**")
            st.markdown(f"**Raw Prediction**: {raw:.2f} kcal")
            
            
            # Display input summary
            with st.expander("View Input Details"):
                st.write("**Input Summary**:")
                st.write(f"- Age: {age} years")
                st.write(f"- Gender: {gender}")
                st.write(f"- Weight: {weight} kg")
                st.write(f"- Height: {height} m")
                st.write(f"- Max BPM: {max_bpm}")
                st.write(f"- Average BPM: {avg_bpm}")
                st.write(f"- Resting BPM: {resting_bpm}")
                st.write(f"- Duration: {duration} hours")
                st.write(f"- Workout Type: {workout_type}")
                st.write(f"- Fat Percentage: {fat_percentage}%")
                st.write(f"- Water Intake: {water_intake} liters")
                st.write(f"- Workout Frequency: {workout_frequency} days/week")
                st.write(f"- Experience Level: { {1: 'Beginner', 2: 'Intermediate', 3: 'Expert'}[experience_level] }")
                st.write(f"- Intensity: {(avg_bpm - resting_bpm) / (max_bpm - resting_bpm):.3f}")
                st.write(f"- BMI: {weight / (height ** 2):.2f}")
        
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Tab 2: Maintenance Calories Calculator
with tab2:
    st.header("Maintenance Calories Calculator")
    st.markdown("Calculate your daily maintenance calories using the Mifflin-St Jeor Equation, based on your age, gender, weight, height, and workout frequency.")
    
    with st.form("maintenance_form"):
        st.subheader("Personal Details")
        col1, col2 = st.columns(2)
        
        with col1:
            maint_age = st.number_input("Age", min_value=10, max_value=100, value=20, help="Enter your age (10–100 years).", key="maint_age")
            maint_gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.", index=0, key="maint_gender")
            maint_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=76.0, step=0.1, help="Enter your weight in kilograms (30–200 kg).", key="maint_weight")
        
        with col2:
            maint_height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.84, step=0.01, help="Enter your height in meters (1.0–2.5 m).", key="maint_height")
            maint_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=5, help="Enter workouts per week (1–7 days).", key="maint_frequency")
        
        maint_submit = st.form_submit_button("Calculate Maintenance Calories")

    if maint_submit:
        try:
            bmr, tdee = calculate_maintenance_calories(
                age=maint_age,
                gender=maint_gender,
                weight_kg=maint_weight,
                height_m=maint_height,
                workout_frequency=maint_frequency
            )
            st.success(f" **Maintenance Calories**")
            st.markdown(f"**Basal Metabolic Rate (BMR)**: {bmr:.2f} kcal/day")
            st.markdown(f"**Total Daily Energy Expenditure (TDEE)**: {tdee:.2f} kcal/day")
            st.info("TDEE is your estimated daily calorie needs to maintain weight, based on your activity level. Adjust based on weight loss or gain goals.")
            
            # Display input summary
            with st.expander("View Input Details"):
                st.write("**Input Summary**:")
                st.write(f"- Age: {maint_age} years")
                st.write(f"- Gender: {maint_gender}")
                st.write(f"- Weight: {maint_weight} kg")
                st.write(f"- Height: {maint_height} m")
                st.write(f"- Workout Frequency: {maint_frequency} days/week")
                st.write(f"- BMI: {maint_weight / (maint_height ** 2):.2f}")
        
        except Exception as e:
            st.error(f"Calculation failed: {e}")

# Tab 3: Smoking Status Check
with tab3:
    st.header("Smoking Status Health Check")
    st.markdown("Check health cautions based on your smoking status and workout heart rate.")
    
    with st.form("smoking_form"):
        st.subheader("Health Details")
        col1, col2 = st.columns(2)
        
        with col1:
            smoke_avg_bpm = st.number_input("Average BPM", min_value=60, max_value=200, value=120, help="Enter your average heart rate during workout (60–200 BPM).", key="smoke_avg_bpm")
            smoke_max_bpm = st.number_input("Max BPM", min_value=80, max_value=220, value=140, help="Enter your maximum heart rate (80–220 BPM).", key="smoke_max_bpm")
        
        with col2:
            smoking_status = st.selectbox("Smoking Status", ["None", "Very Less", "Occasional", "Very Much"], help="Select your smoking frequency.", index=0, key="smoking_status")
        
        smoke_submit = st.form_submit_button("Check Health Cautions")

    if smoke_submit:
        try:
            cautions = check_smoking_status(
                avg_bpm=smoke_avg_bpm,
                max_bpm=smoke_max_bpm,
                smoking_status=smoking_status
            )
            if cautions:
                st.warning("⚠️ **Health Cautions**")
                for caution in cautions:
                    st.markdown(caution)
            else:
                st.success("✅ **No significant cautions**: Your heart rate and smoking status do not indicate immediate concerns for exercise. Always monitor your health and consult a doctor if needed.")
            
            # Display input summary
            with st.expander("View Input Details"):
                st.write("**Input Summary**:")
                st.write(f"- Average BPM: {smoke_avg_bpm}")
                st.write(f"- Max BPM: {smoke_max_bpm}")
                st.write(f"- Smoking Status: {smoking_status}")
                st.write(f"- Intensity: {(smoke_avg_bpm - 70) / (smoke_max_bpm - 70):.3f} (assuming resting BPM = 70)")
        
        except Exception as e:
            st.error(f"Check failed: {e}")

# Add footer
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost. Model trained on gym exercise data with MET-based constraints.")