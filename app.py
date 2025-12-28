
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from ml.food_classifier_training import (
    CLASS_NAMES_PATH,
    MODEL_OUTPUT_PATH,
    load_class_names,
    predict_top_k,
)

app = Flask(__name__)
# paths 
MODEL_PATH = Path(MODEL_OUTPUT_PATH)
CLASS_NAMES_FILE = Path(CLASS_NAMES_PATH)
FOOD_MODEL: tf.keras.Model | None = None
CLASS_NAMES: list[str] = []

#calorie estimates for common Indian food items 
FOOD_CALORIES = {
    "roti": 120,
    "rice": 200,
    "dal": 150,
    "sabzi": 130,
    "paneer": 180,
    "idli": 70,
    "dosa": 160,
    "curd": 100,
}

# simplified calories burned for workout intensity buckets.
WORKOUT_INTENSITY_BURN = {
    "low": 120,
    "medium": 220,
    "high": 340,
}

STEP_CALORIES_PER_STEP = 0.035  # 35 calories per 1,000 steps
BASE_MAINTENANCE = 2000  # average maintenece target

#estimate calorie intake
def estimate_calorie_intake(selected_foods: list[str]) -> int:
    return sum(FOOD_CALORIES.get(item, 0) for item in selected_foods)

#estimate calories burned
def estimate_calories_burned(step_count: int, workout_intensity: str) -> int:
    steps_burn = int(step_count * STEP_CALORIES_PER_STEP)
    workout_burn = WORKOUT_INTENSITY_BURN.get(workout_intensity, 0)
    return steps_burn + workout_burn

# estimate the step count 
def estimate_maintenance(step_count: int, workout_intensity: str) -> int:
    intensity_bonus = {"low": 0, "medium": 120, "high": 220}.get(workout_intensity, 0)
    steps_bonus = (step_count // 1000) * 25  # Reward overall active days.
    return BASE_MAINTENANCE + intensity_bonus + steps_bonus

#suggest bulk cut or maintain
def recommend_goal(net_intake: int, maintenance: int) -> str:
    buffer = 150  # Small buffer so tiny fluctuations stay in "maintain".
    if net_intake > maintenance + buffer:
        return "Bulk"
    if net_intake < maintenance - buffer:
        return "Cut"
    return "Maintain"

#to calculate bmi 
def calculate_bmi(height_cm: float, weight_kg: float) -> tuple[float | None, str | None]:
    """Return BMI and a coarse category if data is available."""
    if height_cm <= 0 or weight_kg <= 0:
        return None, None
    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return bmi, category


def ensure_model_loaded() -> None:
    if not MODEL_PATH.exists() or not CLASS_NAMES_FILE.exists():
        raise FileNotFoundError(
            "Model not found"
        )

    global FOOD_MODEL, CLASS_NAMES
    if FOOD_MODEL is None:
        FOOD_MODEL = tf.keras.models.load_model(MODEL_PATH)
    if not CLASS_NAMES:
        CLASS_NAMES = load_class_names(CLASS_NAMES_FILE)


@app.route("/")
def index():
    """Serve the prototype UI"""
    return render_template("index.html", food_options=FOOD_CALORIES)

#summary 
@app.post("/analyze")
def analyze():
    data = request.get_json(force=True)
    selected_foods = data.get("foods", [])
    workout_intensity = data.get("workoutIntensity", "low")
    step_count = int(data.get("steps", 0))
    height_cm = float(data.get("heightCm", 0))
    weight_kg = float(data.get("weightKg", 0))

    calorie_intake = estimate_calorie_intake(selected_foods)
    calories_burned = estimate_calories_burned(step_count, workout_intensity)
    maintenance_calories = estimate_maintenance(step_count, workout_intensity)
    net_intake = calorie_intake - calories_burned
    goal = recommend_goal(net_intake, maintenance_calories)
    bmi_value, bmi_category = calculate_bmi(height_cm, weight_kg)

    return jsonify(
        {
            "calorieIntake": calorie_intake,
            "caloriesBurned": calories_burned,
            "maintenanceCalories": maintenance_calories,
            "recommendedGoal": goal,
            "bmi": bmi_value,
            "bmiCategory": bmi_category,
        }
    )


@app.post("/predict-food")
def predict_food():
    """Accept an uploaded meal photo and return top-3 model predictions."""
    try:
        ensure_model_loaded()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "Please provide a meal photo."}), 400

    suffix = Path(file.filename).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        file.save(tmp_file.name)
        temp_path = Path(tmp_file.name)

    try:
        predictions = predict_top_k(
            MODEL_PATH,
            temp_path,
            CLASS_NAMES,
            k=3,
            model=FOOD_MODEL,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    finally:
        temp_path.unlink(missing_ok=True)

    return jsonify(
        {
            "predictions": [
                {"label": label, "probability": prob} for label, prob in predictions
            ]
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
