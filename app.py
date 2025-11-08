import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google import genai
from dotenv import load_dotenv
import os
from src.constants import map_exercises, CATEGORICAL_COLS
from src.utils import (
    preprocess_for_xgb,
    preprocess_categorical,
    preprocess_target,
    dropna_func,
)


load_dotenv()
#api_key = os.getenv("GEMINI_API_KEY")
api_key=st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)



@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "models", "exercise_recommender.pkl")
    return joblib.load(
        model_path
    )


model = load_model()

# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="AI Fitness Recommender", page_icon="💪", layout="wide")

st.title("💪 Personalized Fitness Recommender")
st.markdown(
    "Get your top 10 personalized exercise recommendations based on your profile and fitness habits."
)

st.divider()

# ------------------------------------------------------------
# User Input Form
# ------------------------------------------------------------

with st.form("user_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 16, 70, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", 1.2, 2.2, 1.75, step=0.01)
        weight = st.number_input("Weight (kg)", 40, 150, 70, step=1)
        water_intake = st.slider("Water Intake (liters)", 0.5, 6.0, 2.5, step=0.1)

    with col2:
        experience = st.number_input("Years of Experience", 0, 50, 2, step=1)
        session_duration = st.slider(
            "Session Duration (hours)", 0.5, 3.0, 1.0, step=0.1
        )
        workout_freq = st.slider("Workout Frequency (days/week)", 1, 7, 4)
        daily_meals = st.slider("Daily Meals Frequency", 2, 8, 4)
        diet_type = st.selectbox(
            "Diet Type",
            ["Balanced", "Paleo", "Low-Carb", "Vegetarian", "Keto", "Vegan"],
        )

    with col3:
        body_part = st.selectbox(
            "Body Part",
            ["Abs", "Legs", "Arms", "Back", "Forearms", "Chest", "Shoulders"],
        )
        difficulty = st.selectbox(
            "Difficulty Level", ["Beginner", "Intermediate", "Advanced"]
        )
        equipment = st.selectbox(
            "Equipment Available", ["Step or Box", "Parallel Bars or Chair", "Bench"]
        )
        cooking_method = st.selectbox(
            "Preferred Cooking Method",
            ["Baked", "Steamed", "Raw", "Grilled", "Roasted", "Boiled"],
        )

    submitted = st.form_submit_button("🔍 Get My Top 10 Exercises")

    if submitted:
        user_input = pd.DataFrame(
            [
                {
                    "Age": float(age),
                    "Body Part": body_part,
                    "Daily meals frequency": float(daily_meals),
                    "Difficulty Level": difficulty,
                    "Equipment Needed": equipment,
                    "Experience_Level": float(experience),
                    "Gender": gender,
                    "Height (m)": float(height),
                    "Session_Duration (hours)": float(session_duration),
                    "Water_Intake (liters)": float(water_intake),
                    "Weight (kg)": float(weight),
                    "Workout_Frequency (days/week)": float(workout_freq),
                    "cooking_method": cooking_method,
                    "diet_type": diet_type,
                }
            ]
        )

        st.markdown("### 🧾 Your Profile Summary")
        st.dataframe(user_input, use_container_width=True)
        user_input = preprocess_for_xgb(user_input,cat_features=CATEGORICAL_COLS)
        top_10_response = model.predict_proba(user_input)
        top_10_indices = top_10_response[0].argsort()[-10:][::-1]
        ranked_exercises = [map_exercises[i] for i in top_10_indices]
        ranked_exercises = pd.DataFrame(
            {"Rank": range(1, 11), "Exercise": ranked_exercises}
        ).reset_index(drop=True)

        st.success("✅ Top 10 Exercises Recommended for You:")
        st.dataframe(ranked_exercises, width="content", hide_index=True)

        top_exercises = ranked_exercises["Exercise"].tolist()
        user_profile = (
            f"{age} years old {gender.lower()} with {experience} year of experience, "
            f"{workout_freq} workouts/week, prefers {diet_type} diet, "
            f"focus on {body_part} at {difficulty} level using {equipment}, height {height} m, weight {weight} kg, likes {cooking_method} meals and follows {daily_meals} meals/day."
        )
        st.session_state["user_profile"] = user_profile
        st.session_state["ranked_exercises"] = ranked_exercises


if "ranked_exercises" in st.session_state and "user_profile" in st.session_state:
    ranked_exercises = st.session_state["ranked_exercises"]
    user_profile = st.session_state["user_profile"]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Create My Exercise Plan"):
            with st.spinner("AI is generating your plan..."):
                prompt = f"""
                You are an expert fitness coach.
                Based on this user profile: {user_profile},
                and top exercises: {', '.join(ranked_exercises['Exercise'])},
                generate a personalized 3-day workout plan.
                Include exercises, sets, reps, rest time, and motivation tips.
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            st.session_state["exercise_plan"] = response.text  # ✅ separate key
            st.success("✅ Your AI-generated workout plan:")
            st.markdown(response.text)

    with col2:
        if st.button("🥗 Create Meal Plan"):
            with st.spinner("AI is generating your meal plan..."):
                prompt = f"""
                You are an expert nutritionist.
                Based on this user profile: {user_profile},
                generate a personalized 3-day meal plan.
                Include breakfast, lunch, dinner, snacks, and hydration tips.
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            st.session_state["meal_plan"] = response.text  # ✅ different key
            st.success("✅ Your AI-generated meal plan:")
            st.markdown(response.text)
