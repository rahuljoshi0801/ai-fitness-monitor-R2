
from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import tensorflow as tf
from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from storage import (
    create_user,
    fetch_recent_check_ins,
    find_user_by_email,
    find_user_by_id,
    init_db,
    record_check_in,
)
from ml.food_classifier_training import (
    CLASS_NAMES_PATH,
    MODEL_OUTPUT_PATH,
    load_class_names,
    predict_top_k,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
init_db()
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


def current_user() -> dict | None:
    user_id = session.get("user_id")
    if user_id is None:
        return None
    return find_user_by_id(user_id)


def login_user(user: dict) -> None:
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]


def logout_user() -> None:
    session.clear()


def generate_personalized_plan(
    history_entries: list[dict],
    latest_summary: dict | None,
) -> dict:
    steps_values = [
        int(entry.get("steps") or 0) for entry in history_entries if entry.get("steps")
    ]
    avg_steps = sum(steps_values) / len(steps_values) if steps_values else 5000
    latest_goal = (latest_summary or {}).get("recommendedGoal", "Maintain")
    latest_bmi = (latest_summary or {}).get("bmi")
    intensity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    for entry in history_entries:
        intensity = (entry.get("workoutIntensity") or "low").lower()
        if intensity in intensity_counts:
            intensity_counts[intensity] += 1

    dominant_intensity = max(intensity_counts, key=intensity_counts.get)
    steps_target = int(round(max(avg_steps + 1500, 6000), -2))

    focus_map = {
        "Bulk": "Lean muscle gain",
        "Cut": "Gentle fat loss",
        "Maintain": "Balanced maintenance",
    }
    focus = focus_map.get(latest_goal, "Balanced maintenance")

    if latest_bmi:
        if latest_bmi < 18.5:
            nutrition_tip = "Add calorie-dense snacks (peanuts, paneer, dals) to support healthy weight gain."
        elif latest_bmi < 25:
            nutrition_tip = "Stick with colourful plates and steady protein to maintain your groove."
        else:
            nutrition_tip = "Prioritise fibre-rich sabzis and lean proteins to keep you full while trimming."
    else:
        nutrition_tip = "Balance each plate with carbs, protein, and veg for steady energy."

    workout_tip = {
        "low": "Sprinkle shorter walks or light yoga through the day to build momentum.",
        "medium": "Great consistency—add one focused strength session to level up.",
        "high": "You’re crushing intensity. Mix in recovery mobility so your body stays fresh.",
    }.get(dominant_intensity, "Keep listening to your body and mix cardio with strength.")

    note = (
        f"Based on {len(history_entries)} recent check-ins."
        if history_entries
        else "Plan tailored from today’s inputs."
    )

    return {
        "focus": focus,
        "stepsTarget": steps_target,
        "workoutTip": workout_tip,
        "nutritionTip": nutrition_tip,
        "note": note,
    }


@app.route("/")
def index():
    """Serve the prototype UI"""
    user = current_user()
    history = []
    plan_preview = None
    if user:
        history = fetch_recent_check_ins(user["id"])
        latest_summary = history[0].get("summary") if history else None
        plan_preview = generate_personalized_plan(history, latest_summary)
    return render_template(
        "index.html",
        food_options=FOOD_CALORIES,
        user=user,
        history=history,
        plan=plan_preview,
    )

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

    response = {
        "calorieIntake": calorie_intake,
        "caloriesBurned": calories_burned,
        "maintenanceCalories": maintenance_calories,
        "recommendedGoal": goal,
        "bmi": bmi_value,
        "bmiCategory": bmi_category,
    }

    user = current_user()
    history_entries: list[dict] = []
    if user:
        record_check_in(user["id"], {**data, **response})
        history_entries = fetch_recent_check_ins(user["id"], limit=7)

    plan = generate_personalized_plan(history_entries, {**data, **response})
    response["personalPlan"] = plan
    if history_entries:
        response["history"] = history_entries

    return jsonify(response)


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
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return jsonify(
        {"predictions": [{"label": label, "probability": score} for label, score in predictions]}
    )


@app.post("/register")
def register():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    email = data.get("email", "").lower().strip()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400

    existing = find_user_by_email(email)
    if existing:
        return jsonify({"error": "Email already registered."}), 400

    password_hash = generate_password_hash(password)
    user = create_user(name, email, password_hash)
    login_user(user)
    return jsonify({"user": {"id": user["id"], "name": user["name"], "email": user["email"]}})


@app.post("/login")
def login():
    data = request.get_json(force=True)
    email = data.get("email", "").lower().strip()
    password = data.get("password", "")

    user = find_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):  # type: ignore[index]
        return jsonify({"error": "Invalid credentials."}), 400

    login_user(user)
    return jsonify({"user": {"id": user["id"], "name": user["name"], "email": user["email"]}})


@app.post("/logout")
def logout():
    logout_user()
    return jsonify({"ok": True})


@app.get("/history")
def history():
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    records = fetch_recent_check_ins(user["id"], limit=14)
    return jsonify({"history": records})


if __name__ == "__main__":
    app.run(debug=True)
