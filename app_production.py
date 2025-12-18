import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="AI Fitness Recommender", page_icon="💪", layout="wide")
st.title("💪 Personalized Fitness Recommender")

st.markdown(
    "Get your top 10 personalized exercise recommendations based on your profile and fitness habits."
)

st.divider()
inputs = {}
top_exercise = []
top_exercises_list = []
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
        inputs = {
            "Age": float(age),
            "Body_Part": body_part,
            "Daily_meals_frequency": float(daily_meals),
            "Difficulty_Level": difficulty,
            "Equipment_Needed": equipment,
            "Experience_Level": float(experience),
            "Gender": gender,
            "Height_m": float(height),
            "Session_Duration_hours": float(session_duration),
            "Water_Intake_liters": float(water_intake),
            "Weight_kg": float(weight),
            "Workout_Frequency_days_per_week": float(workout_freq),
            "cooking_method": cooking_method,
            "diet_type": diet_type,
        }
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/top_exercises/", json=inputs, timeout=15
            )
            resp.raise_for_status()
            top_exercise = resp.json()
            top_exercises_list = top_exercise.get("top_10_exercises", [])
            st.write(top_exercises_list)
            # persist for later buttons / reruns
            st.session_state["top_exercises_list"] = top_exercises_list
        except Exception as e:
            st.error(f"Failed to send error:{e}")

        st.session_state["user_profile"] = inputs
        st.session_state["ranked_exercises"] = top_exercise

if "ranked_exercises" in st.session_state and "user_profile" in st.session_state:
    ranked_exercises = st.session_state["ranked_exercises"]
    user_profile = st.session_state["user_profile"]
    # use persisted session values when available (prevents NameError)
    user_profile_for_plan = st.session_state.get("user_profile", inputs)
    top_exs_for_plan = st.session_state.get("top_exercises_list", top_exercises_list)

    exercise_plan_inputs = {
        "user_profile": user_profile_for_plan,
        "top_10_exercises": top_exs_for_plan if isinstance(top_exs_for_plan, list) else [],
    }

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Create My Exercise Plan"):
            with st.spinner("AI is generating your plan..."):
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/get_exercises_schedule/",
                        json=exercise_plan_inputs,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    exercise_response = resp.json()
                    schedule_text = exercise_response.get("exercise_schedule")
                    if schedule_text:
                        st.write(schedule_text)
                    else:
                        st.error(f"Missing 'exercise_schedule' in response: {resp.text}")

                except requests.exceptions.HTTPError as e:
                    # Surface server response body (helpful for 422 parse errors)
                    resp_obj = getattr(e, 'response', None)
                    body = None
                    try:
                        if resp_obj is not None:
                            body = resp_obj.text
                    except Exception:
                        body = None
                    st.error(f"HTTP error: {e}. Response body: {body}")
                except Exception as e:
                    st.error(f"Failed to send error:{e}")
    with col2:
        if st.button("📝 🥗 Create Meal Plan"):
            with st.spinner("AI is generating your meal plan..."):
                try:
                    meal_payload = st.session_state.get("user_profile", inputs)
                    resp = requests.post(
                        "http://127.0.0.1:8000/get_meal_schedule/",
                        json=meal_payload,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    meal_response = resp.json()
                    meal_text = meal_response.get("meal_schedule")
                    if meal_text:
                        st.write(meal_text)
                    else:
                        st.error(f"Missing 'meal_schedule' in response: {resp.text}")

                except requests.exceptions.HTTPError as e:
                    resp_obj = getattr(e, 'response', None)
                    body = None
                    try:
                        if resp_obj is not None:
                            body = resp_obj.text
                    except Exception:
                        body = None
                    st.error(f"HTTP error: {e}. Response body: {body}")
                except Exception as e:
                    st.error(f"Failed to send error:{e}")





