from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import csv
import io
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parent))

from tempfile import NamedTemporaryFile

import tensorflow as tf
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from calculators import get_calculator_configs, run_calculator
from ml.food_classifier_training import (
    CLASS_NAMES_PATH,
    MODEL_OUTPUT_PATH,
    load_class_names,
    predict_top_k,
)
from storage import (
    create_user,
    fetch_check_ins_paginated,
    fetch_check_ins_since,
    fetch_recent_check_ins,
    find_user_by_email,
    find_user_by_id,
    init_db,
    record_check_in,
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

PROTEIN_FOODS = {"paneer", "dal", "curd", "dosa", "idli"}

PORTION_CONFIG = {
    "roti": {
        "label": "Roti",
        "options": [
            {"label": "1", "value": "1", "multiplier": 1},
            {"label": "2", "value": "2", "multiplier": 2},
            {"label": "3", "value": "3", "multiplier": 3},
        ],
    },
    "rice": {
        "label": "Rice",
        "options": [
            {"label": "½ bowl", "value": "0_5", "multiplier": 0.5},
            {"label": "1 bowl", "value": "1", "multiplier": 1},
            {"label": "2 bowls", "value": "2", "multiplier": 2},
        ],
    },
    "paneer": {
        "label": "Paneer",
        "options": [
            {"label": "50g", "value": "50", "multiplier": 0.5},
            {"label": "100g", "value": "100", "multiplier": 1},
            {"label": "150g", "value": "150", "multiplier": 1.5},
        ],
    },
}

# curated exercise presets to replace the old intensity dropdown.
EXERCISE_LIBRARY = {
    "light_walk": {
        "label": "Light walk & mobility",
        "intensity": "low",
        "burn": 90,
        "default_sets": 1,
        "default_reps": 0,
        "default_duration": 25,
    },
    "yoga_flow": {
        "label": "Yoga / Pilates flow",
        "intensity": "low",
        "burn": 120,
        "default_sets": 1,
        "default_reps": 0,
        "default_duration": 30,
    },
    "mobility_reset": {
        "label": "Mobility reset & stretching",
        "intensity": "low",
        "burn": 110,
        "default_sets": 3,
        "default_reps": 8,
        "default_duration": 20,
    },
    "strength_circuit": {
        "label": "Strength circuit",
        "intensity": "medium",
        "burn": 220,
        "default_sets": 4,
        "default_reps": 12,
        "default_duration": 30,
    },
    "upper_body_push": {
        "label": "Upper body push day",
        "intensity": "medium",
        "burn": 240,
        "default_sets": 5,
        "default_reps": 10,
        "default_duration": 35,
    },
    "zone2_cardio": {
        "label": "Zone 2 run or cycle",
        "intensity": "medium",
        "burn": 260,
        "default_sets": 1,
        "default_reps": 0,
        "default_duration": 40,
    },
    "hiit_session": {
        "label": "HIIT + sprints",
        "intensity": "high",
        "burn": 340,
        "default_sets": 5,
        "default_reps": 10,
        "default_duration": 20,
    },
    "cycling_class": {
        "label": "Spin or cycling class",
        "intensity": "high",
        "burn": 360,
        "default_sets": 1,
        "default_reps": 0,
        "default_duration": 45,
    },
    "sports_day": {
        "label": "Team sport / long hike",
        "intensity": "high",
        "burn": 380,
        "default_sets": 2,
        "default_reps": 0,
        "default_duration": 60,
    },
    "boxing_rounds": {
        "label": "Boxing rounds + bag work",
        "intensity": "high",
        "burn": 400,
        "default_sets": 8,
        "default_reps": 3,
        "default_duration": 30,
    },
}
DEFAULT_EXERCISE_ID = "light_walk"
INTENSITY_BONUS = {"low": 0, "medium": 120, "high": 220}
INTENSITY_ORDER = {"low": 0, "medium": 1, "high": 2}

STEP_CALORIES_PER_STEP = 0.035  # 35 calories per 1,000 steps
BASE_MAINTENANCE = 2000  # average maintenece target
BASE_EXERCISE_DURATION = 30  # minutes represented by each preset burn
WEEKLY_REPORT_DAYS = 7
MONTHLY_AVG_DAYS = 30

#estimate calorie intake
def estimate_calorie_intake(
    selected_foods: list[str], food_portions: list[dict[str, Any]] | None = None
) -> int:
    portion_map: dict[str, float] = {}
    if food_portions:
        for portion in food_portions:
            food_id = portion.get("id")
            if not food_id or food_id not in FOOD_CALORIES:
                continue
            multiplier = portion.get("multiplier")
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError):
                multiplier_value = 0.0
            if multiplier_value <= 0:
                continue
            portion_map[food_id] = multiplier_value

    total = 0
    for item in selected_foods:
        calories = FOOD_CALORIES.get(item, 0)
        if calories == 0:
            continue
        multiplier_value = portion_map.pop(item, None)
        if multiplier_value:
            total += int(calories * multiplier_value)
        else:
            total += calories

    for food_id, multiplier_value in portion_map.items():
        base_calories = FOOD_CALORIES.get(food_id)
        if base_calories:
            total += int(base_calories * multiplier_value)

    return total


def _safe_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _safe_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


#lookup exercise helpers
def get_exercise(exercise_id: str) -> dict:
    return EXERCISE_LIBRARY.get(exercise_id, EXERCISE_LIBRARY[DEFAULT_EXERCISE_ID])


def normalize_exercise_entries(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return [{"id": DEFAULT_EXERCISE_ID}]

    if isinstance(raw_value, list):
        candidate_entries = raw_value
    else:
        candidate_entries = [raw_value]

    normalized: list[dict[str, Any]] = []
    for candidate in candidate_entries:
        if isinstance(candidate, str):
            exercise_id = candidate
            sets = reps = 0
            duration = BASE_EXERCISE_DURATION
        elif isinstance(candidate, dict):
            exercise_id = candidate.get("id") or candidate.get("exerciseId")
            sets = _safe_non_negative_int(candidate.get("sets"), 0)
            reps = _safe_non_negative_int(candidate.get("reps"), 0)
            duration = _safe_positive_float(
                candidate.get("duration") or candidate.get("durationMinutes") or candidate.get("minutes"),
                BASE_EXERCISE_DURATION,
            )
        else:
            continue

        if exercise_id not in EXERCISE_LIBRARY:
            continue

        meta = get_exercise(exercise_id)
        default_sets = meta.get("default_sets", 0)
        default_reps = meta.get("default_reps", 0)
        default_duration = meta.get("default_duration", BASE_EXERCISE_DURATION)

        sets = sets or default_sets
        reps = reps or default_reps
        duration = duration or default_duration

        normalized.append(
            {
                "id": exercise_id,
                "sets": sets,
                "reps": reps,
                "duration": duration,
            }
        )

    if not normalized:
        return [{"id": DEFAULT_EXERCISE_ID, "sets": 0, "reps": 0, "duration": BASE_EXERCISE_DURATION}]
    return normalized


def summarize_exercises(exercise_entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    total_burn = 0.0

    for entry in exercise_entries:
        exercise_meta = get_exercise(entry["id"]).copy()
        duration = entry.get("duration") or BASE_EXERCISE_DURATION
        duration = _safe_positive_float(duration, BASE_EXERCISE_DURATION)
        multiplier = max(duration / BASE_EXERCISE_DURATION, 0.25)
        burn = exercise_meta["burn"] * multiplier
        enriched_meta = {
            "id": entry["id"],
            "label": exercise_meta["label"],
            "intensity": exercise_meta["intensity"],
            "sets": entry.get("sets", 0),
            "reps": entry.get("reps", 0),
            "duration": round(duration, 1),
            "estimatedBurn": round(burn),
        }
        metas.append(enriched_meta)
        total_burn += burn

    if metas:
        primary = max(metas, key=lambda meta: INTENSITY_ORDER.get(meta["intensity"], 0))
    else:
        fallback = get_exercise(DEFAULT_EXERCISE_ID)
        primary = {
            "id": DEFAULT_EXERCISE_ID,
            "label": fallback["label"],
            "intensity": fallback["intensity"],
            "sets": 0,
            "reps": 0,
            "duration": BASE_EXERCISE_DURATION,
            "estimatedBurn": fallback["burn"],
        }
        metas.append(primary)
        total_burn = primary["estimatedBurn"]

    return metas, round(total_burn), primary


def estimate_calories_burned(step_count: int, exercise_entries: list[dict[str, Any]]) -> int:
    steps_burn = int(step_count * STEP_CALORIES_PER_STEP)
    _, workout_burn, _ = summarize_exercises(exercise_entries)
    return steps_burn + workout_burn

# estimate the step count 
def estimate_maintenance(step_count: int, exercise_entries: list[dict[str, Any]]) -> int:
    _, _, primary = summarize_exercises(exercise_entries)
    intensity_bonus = INTENSITY_BONUS.get(primary["intensity"], 0)
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


def enrich_history_entry(entry: dict) -> dict:
    summary_data = entry.get("summary") or {}
    raw_entries = (
        entry.get("exerciseEntries")
        or summary_data.get("exerciseEntries")
        or entry.get("exerciseDetails")
    )

    exercise_entries = None
    exercise_details = entry.get("exerciseDetails") or summary_data.get("exerciseDetails")

    if raw_entries:
        exercise_entries = normalize_exercise_entries(raw_entries)
        exercise_details, _, primary = summarize_exercises(exercise_entries)
        entry["exerciseEntries"] = exercise_entries
        entry["exerciseDetails"] = exercise_details
        entry["exerciseDisplay"] = " + ".join(detail["label"] for detail in exercise_details)
        entry["exerciseId"] = primary["id"]
        entry["exerciseIntensity"] = primary["intensity"]
        entry["exerciseLabel"] = entry["exerciseDisplay"] or primary["label"]
        return entry

    exercise_id = entry.get("exerciseId") or entry.get("workoutIntensity") or DEFAULT_EXERCISE_ID
    exercise_meta = get_exercise(exercise_id)
    entry["exerciseId"] = exercise_id
    entry["exerciseLabel"] = exercise_meta["label"]
    entry["exerciseIntensity"] = exercise_meta["intensity"]
    entry["exerciseDisplay"] = exercise_meta["label"]
    return entry


def enrich_history_entries(entries: list[dict]) -> list[dict]:
    return [enrich_history_entry(entry) for entry in entries]


@app.route("/")
def index():
    """Serve the main dashboard, requiring authentication."""
    user = current_user()
    if not user:
        return redirect(url_for("login_page"))

    history_entries = enrich_history_entries(
        fetch_recent_check_ins(user["id"], limit=7)
    )
    calculator_configs = get_calculator_configs()
    return render_template(
        "index.html",
        food_options=FOOD_CALORIES,
        user=user,
        history=history_entries,
        calculators=calculator_configs,
        exercise_options=EXERCISE_LIBRARY,
        portion_config=PORTION_CONFIG,
    )


@app.route("/login")
def login_page():
    """Serve standalone login screen."""
    if current_user():
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/register")
def register_page():
    if current_user():
        return redirect(url_for("index"))
    return render_template("register.html")


@app.post("/login")
def login_action():
    """Authenticate a user and start a session."""
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = find_user_by_email(email, include_sensitive=True)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 400

    clean_user = {k: v for k, v in user.items() if k != "password_hash"}
    login_user(clean_user)
    return jsonify({"ok": True, "user": clean_user})


@app.post("/logout")
def logout_action():
    logout_user()
    return jsonify({"ok": True})


@app.post("/register")
def register_action():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if len(name) < 2:
        return jsonify({"error": "Please provide your full name."}), 400
    if "@" not in email:
        return jsonify({"error": "Please enter a valid email."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    if find_user_by_email(email):
        return jsonify({"error": "An account with this email already exists."}), 400

    password_hash = generate_password_hash(password)
    user = create_user(name, email, password_hash)
    login_user(user)
    return jsonify({"ok": True, "user": user})


@app.get("/calculators")
def list_calculators():
    """Expose calculator metadata to the frontend."""
    return jsonify({"calculators": get_calculator_configs()})


@app.post("/calculators/<calc_id>")
def run_calculator_endpoint(calc_id: str):
    """Execute a calculator with the provided payload."""
    data = request.get_json(force=True) or {}
    try:
        result = run_calculator(calc_id, data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"result": result})


#summary 
@app.post("/analyze")
def analyze():
    data = request.get_json(force=True)
    selected_foods = data.get("foods", [])
    food_portions = data.get("foodPortions") or []
    raw_exercises = (
        data.get("exerciseEntries")
        or data.get("exerciseIds")
        or data.get("exerciseId")
        or data.get("workoutIntensity")
    )
    exercise_entries = normalize_exercise_entries(raw_exercises)
    exercise_details, exercise_burn, primary_exercise = summarize_exercises(
        exercise_entries
    )
    step_count = int(data.get("steps", 0))
    height_cm = float(data.get("heightCm", 0))
    weight_kg = float(data.get("weightKg", 0))

    calorie_intake = estimate_calorie_intake(selected_foods, food_portions)
    calories_burned = estimate_calories_burned(step_count, exercise_entries)
    maintenance_calories = estimate_maintenance(step_count, exercise_entries)
    net_intake = calorie_intake - calories_burned
    goal = recommend_goal(net_intake, maintenance_calories)
    bmi_value, bmi_category = calculate_bmi(height_cm, weight_kg)
    exercise_display = " + ".join(detail["label"] for detail in exercise_details)

    user = current_user()
    previous_steps: int | None = None
    if user:
        previous_entries = fetch_recent_check_ins(user["id"], limit=1)
        if previous_entries:
            previous_steps = previous_entries[0].get("steps")  # type: ignore[index]

    micro_coach_text = generate_micro_coach_text(
        selected_foods,
        step_count,
        calorie_intake,
        maintenance_calories,
        net_intake,
        goal,
        previous_steps=previous_steps,
    )

    response = {
        "calorieIntake": calorie_intake,
        "caloriesBurned": calories_burned,
        "maintenanceCalories": maintenance_calories,
        "recommendedGoal": goal,
        "bmi": bmi_value,
        "bmiCategory": bmi_category,
        "exerciseId": primary_exercise["id"],
        "exerciseLabel": exercise_display or primary_exercise["label"],
        "exerciseDisplay": exercise_display or primary_exercise["label"],
        "exerciseIntensity": primary_exercise["intensity"],
        "exerciseDetails": exercise_details,
        "microCoachText": micro_coach_text,
    }

    if user:
        record_payload = {
            **data,
            **response,
            "exerciseEntries": exercise_entries,
            "foodPortions": food_portions,
        }
        record_check_in(user["id"], record_payload)
        response["history"] = enrich_history_entries(
            fetch_recent_check_ins(user["id"], limit=7)
        )

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


@app.get("/history")
def history():
    """Return paginated, date-filtered history for the logged-in user."""
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    result = fetch_check_ins_paginated(
        user["id"], limit=limit, offset=offset, from_date=from_date, to_date=to_date
    )
    result["entries"] = enrich_history_entries(result["entries"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
