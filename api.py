from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
from src.utils import preprocess_for_xgb
from src.constants import map_exercises, CATEGORICAL_COLS
from google import genai
import os
from dotenv import load_dotenv
import pandas as pd

app = FastAPI()
client = load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


class FitnessInfo(BaseModel):
    Age: float
    Body_Part: str
    Daily_meals_frequency: float
    Difficulty_Level: str
    Equipment_Needed: str
    Experience_Level: float
    Gender: str
    Height_m: float
    Session_Duration_hours: float
    Water_Intake_liters: float
    Weight_kg: float
    Workout_Frequency_days_per_week: float
    cooking_method: str
    diet_type: str


class ExerciseResponse(BaseModel):
    user_profile:FitnessInfo
    top_10_exercises: list


@app.post("/top_exercises/")
def get_top_exercises(info: FitnessInfo):
    user_info = {
        "Age": info.Age,
        "Body Part": info.Body_Part,
        "Daily meals frequency": float(info.Daily_meals_frequency),
        "Difficulty Level": info.Difficulty_Level,
        "Equipment Needed": info.Equipment_Needed,
        "Experience_Level": float(info.Experience_Level),
        "Gender": info.Gender,
        "Height (m)": float(info.Height_m),
        "Session_Duration (hours)": float(info.Session_Duration_hours),
        "Water_Intake (liters)": float(info.Water_Intake_liters),
        "Weight (kg)": float(info.Weight_kg),
        "Workout_Frequency (days/week)": float(info.Workout_Frequency_days_per_week),
        "cooking_method": info.cooking_method,
        "diet_type": info.diet_type,
    }

    with open(
        "D:/Projects/Ai-ML-based-Personalized_Fitness_Recommender/models/exercise_recommender.pkl",
        "rb",
    ) as f:
        model = joblib.load(f)

    user_input = pd.DataFrame([user_info])
    user_input = preprocess_for_xgb(user_input, cat_features=CATEGORICAL_COLS)
    top_10_response = model.predict_proba(user_input)
    top_10_indices = top_10_response[0].argsort()[-10:][::-1]
    ranked_exercises = [map_exercises[i] for i in top_10_indices]
    return {"top_10_exercises": ranked_exercises}


@app.post("/get_exercises_schedule/")
async def get_exercise_response(request: Request):
    # read raw JSON and try to parse into the Pydantic model to provide clearer errors
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    try:
        info = ExerciseResponse.parse_obj(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail={"parse_error": str(e), "body": body})

    # use the parsed model
    user_profile = info.user_profile
    exercises_list = [str(x) for x in info.top_10_exercises]

    prompt = f"""
    You are an expert fitness coach.
    Based on this user profile: {user_profile},
    and top exercises: {", ".join(exercises_list)},
    generate a personalized 3-day workout plan.
    Include exercises, sets, reps, rest time, and motivation tips.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if text is None:
        try:
            text = response.output_text
        except Exception:
            text = str(response)
    return {"exercise_schedule": text}


@app.post("/get_meal_schedule/")
async def get_meal_response(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    try:
        info = FitnessInfo.parse_obj(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail={"parse_error": str(e), "body": body})

    user_info = {
        "Age": info.Age,
        "Body Part": info.Body_Part,
        "Daily meals frequency": float(info.Daily_meals_frequency),
        "Difficulty Level": info.Difficulty_Level,
        "Equipment Needed": info.Equipment_Needed,
        "Experience_Level": float(info.Experience_Level),
        "Gender": info.Gender,
        "Height (m)": float(info.Height_m),
        "Session_Duration (hours)": float(info.Session_Duration_hours),
        "Water_Intake (liters)": float(info.Water_Intake_liters),
        "Weight (kg)": float(info.Weight_kg),
        "Workout_Frequency (days/week)": float(info.Workout_Frequency_days_per_week),
        "cooking_method": info.cooking_method,
        "diet_type": info.diet_type,
    }
    prompt = f"""
    You are an expert nutritionist.
    Based on this user profile: {user_info},
    generate a personalized 3-day meal plan.
    Include breakfast, lunch, dinner, snacks, and hydration tips.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if text is None:
        try:
            text = response.output_text
        except Exception:
            text = str(response)
    return {"meal_schedule": text}
