"""Utility functions for various fitness calculators."""
from __future__ import annotations

import math
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _require_positive(value: float, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required.")
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return value


def _require_nonempty(value: Any, label: str) -> Any:
    if value in (None, ""):
        raise ValueError(f"{label} is required.")
    return value


ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}


# ---------------------------------------------------------------------------
# Calculator implementations
# ---------------------------------------------------------------------------

def calculate_bmi(*, height_cm: float, weight_kg: float) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height") / 100
    weight = _require_positive(weight_kg, "Weight")
    bmi = weight / (height**2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 1), "category": category}


def calculate_bmr(*, height_cm: float, weight_kg: float, age: float, sex: str) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height")
    weight = _require_positive(weight_kg, "Weight")
    age_val = _require_positive(age, "Age")
    sex_value = _require_nonempty(sex, "Sex").lower()
    if sex_value not in {"male", "female"}:
        raise ValueError("Sex must be 'male' or 'female'.")
    # Mifflin-St Jeor
    bmr = 10 * weight + 6.25 * height - 5 * age_val + (5 if sex_value == "male" else -161)
    return {"bmr": round(bmr, 1)}


def calculate_calories(
    *,
    height_cm: float,
    weight_kg: float,
    age: float,
    sex: str,
    activity_level: str = "sedentary",
) -> dict[str, Any]:
    base = calculate_bmr(height_cm=height_cm, weight_kg=weight_kg, age=age, sex=sex)["bmr"]
    factor = ACTIVITY_FACTORS.get(activity_level, ACTIVITY_FACTORS["sedentary"])
    maintenance = base * factor
    return {"bmr": base, "maintenanceCalories": round(maintenance, 1)}


def calculate_body_fat(
    *,
    height_cm: float,
    neck_cm: float,
    waist_cm: float,
    sex: str,
    hip_cm: float | None = None,
) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height") / 2.54  # convert to inches
    neck = _require_positive(neck_cm, "Neck circumference") / 2.54
    waist = _require_positive(waist_cm, "Waist circumference") / 2.54
    sex_value = _require_nonempty(sex, "Sex").lower()
    if sex_value not in {"male", "female"}:
        raise ValueError("Sex must be 'male' or 'female'.")

    if sex_value == "male":
        body_fat = 86.010 * math.log10(waist - neck) - 70.041 * math.log10(height) + 36.76
    else:
        hip = _require_positive(hip_cm or 0, "Hip circumference") / 2.54
        body_fat = 163.205 * math.log10(waist + hip - neck) - 97.684 * math.log10(height) - 78.387
    return {"bodyFatPercent": round(body_fat, 1)}


def calculate_army_body_fat(**kwargs: Any) -> dict[str, Any]:
    return calculate_body_fat(**kwargs)


def calculate_ideal_weight(*, height_cm: float, sex: str) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height")
    sex_value = _require_nonempty(sex, "Sex").lower()
    base = 50 if sex_value == "male" else 45.5
    ideal = base + 0.9 * (height - 152.4)
    return {"idealWeightKg": round(ideal, 1)}


def calculate_pace(*, distance_km: float, duration_min: float) -> dict[str, Any]:
    distance = _require_positive(distance_km, "Distance")
    duration = _require_positive(duration_min, "Duration")
    pace_per_km = duration / distance
    return {
        "paceMinutesPerKm": round(pace_per_km, 2),
        "paceMinutesPerMile": round(pace_per_km * 1.60934, 2),
    }


def calculate_lean_body_mass(*, height_cm: float, weight_kg: float, sex: str) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height")
    weight = _require_positive(weight_kg, "Weight")
    sex_value = _require_nonempty(sex, "Sex").lower()
    if sex_value == "male":
        lbm = 0.407 * weight + 0.267 * height - 19.2
    else:
        lbm = 0.252 * weight + 0.473 * height - 48.3
    return {"leanBodyMassKg": round(lbm, 1)}


def calculate_healthy_weight(*, height_cm: float) -> dict[str, Any]:
    height = _require_positive(height_cm, "Height") / 100
    min_w = 18.5 * (height**2)
    max_w = 24.9 * (height**2)
    return {"healthyMinKg": round(min_w, 1), "healthyMaxKg": round(max_w, 1)}


def calculate_calories_burned(*, weight_kg: float, duration_min: float, met: float) -> dict[str, Any]:
    weight = _require_positive(weight_kg, "Weight")
    duration = _require_positive(duration_min, "Duration")
    met_value = _require_positive(met, "MET")
    calories = met_value * weight * (duration / 60)
    return {"caloriesBurned": round(calories, 1)}


def calculate_one_rep_max(*, lifted_weight_kg: float, reps: float) -> dict[str, Any]:
    weight = _require_positive(lifted_weight_kg, "Weight lifted")
    reps_value = _require_positive(reps, "Reps")
    orm = weight * (1 + reps_value / 30)
    return {"oneRepMaxKg": round(orm, 1)}


def calculate_target_heart_rate(*, age: float) -> dict[str, Any]:
    age_value = _require_positive(age, "Age")
    max_hr = 220 - age_value
    return {
        "maxHeartRate": round(max_hr),
        "moderateRange": [round(max_hr * 0.5), round(max_hr * 0.7)],
        "vigorousRange": [round(max_hr * 0.7), round(max_hr * 0.85)],
    }


body_fat_fields = [
    {"name": "height_cm", "label": "Height (cm)", "type": "number"},
    {"name": "neck_cm", "label": "Neck (cm)", "type": "number"},
    {"name": "waist_cm", "label": "Waist (cm)", "type": "number"},
    {
        "name": "hip_cm",
        "label": "Hip (cm)",
        "type": "number",
        "optional": True,
        "hint": "Required for women",
    },
    {
        "name": "sex",
        "label": "Sex",
        "type": "select",
        "options": ["male", "female"],
    },
]


CALCULATOR_DEFINITIONS: dict[str, dict[str, Any]] = {
    "bmi": {
        "label": "BMI Calculator",
        "description": "Body Mass Index from height and weight.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number", "min": 80},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number", "min": 25},
        ],
        "outputs": ["bmi", "category"],
        "compute": calculate_bmi,
    },
    "calorie": {
        "label": "Calorie Calculator",
        "description": "Daily calorie target based on activity level.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number", "min": 80},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number", "min": 25},
            {"name": "age", "label": "Age", "type": "number", "min": 10},
            {
                "name": "sex",
                "label": "Sex",
                "type": "select",
                "options": ["male", "female"],
            },
            {
                "name": "activity_level",
                "label": "Activity",
                "type": "select",
                "options": list(ACTIVITY_FACTORS.keys()),
            },
        ],
        "outputs": ["bmr", "maintenanceCalories"],
        "compute": calculate_calories,
    },
    "body_fat": {
        "label": "Body Fat Calculator",
        "description": "US Navy body-fat estimate.",
        "fields": body_fat_fields,
        "outputs": ["bodyFatPercent"],
        "compute": calculate_body_fat,
    },
    "bmr": {
        "label": "BMR Calculator",
        "description": "Mifflin-St Jeor basal metabolic rate.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number"},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number"},
            {"name": "age", "label": "Age", "type": "number"},
            {"name": "sex", "label": "Sex", "type": "select", "options": ["male", "female"]},
        ],
        "outputs": ["bmr"],
        "compute": calculate_bmr,
    },
    "ideal_weight": {
        "label": "Ideal Weight Calculator",
        "description": "Devine formula ideal weight.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number"},
            {"name": "sex", "label": "Sex", "type": "select", "options": ["male", "female"]},
        ],
        "outputs": ["idealWeightKg"],
        "compute": calculate_ideal_weight,
    },
    "pace": {
        "label": "Pace Calculator",
        "description": "Average pace per km and mile.",
        "fields": [
            {"name": "distance_km", "label": "Distance (km)", "type": "number", "min": 0.1},
            {"name": "duration_min", "label": "Duration (min)", "type": "number", "min": 1},
        ],
        "outputs": ["paceMinutesPerKm", "paceMinutesPerMile"],
        "compute": calculate_pace,
    },
    "army_body_fat": {
        "label": "Army Body Fat",
        "description": "Army standard body-fat check.",
        "fields": body_fat_fields,
        "outputs": ["bodyFatPercent"],
        "compute": calculate_army_body_fat,
    },
    "lean_body_mass": {
        "label": "Lean Body Mass",
        "description": "Boer formula estimate.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number"},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number"},
            {"name": "sex", "label": "Sex", "type": "select", "options": ["male", "female"]},
        ],
        "outputs": ["leanBodyMassKg"],
        "compute": calculate_lean_body_mass,
    },
    "healthy_weight": {
        "label": "Healthy Weight Range",
        "description": "BMI 18.5-24.9 window.",
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number"},
        ],
        "outputs": ["healthyMinKg", "healthyMaxKg"],
        "compute": calculate_healthy_weight,
    },
    "calories_burned": {
        "label": "Calories Burned",
        "description": "Estimate from MET, weight, and duration.",
        "fields": [
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number"},
            {"name": "duration_min", "label": "Duration (min)", "type": "number"},
            {"name": "met", "label": "MET", "type": "number", "hint": "e.g., 8 for running"},
        ],
        "outputs": ["caloriesBurned"],
        "compute": calculate_calories_burned,
    },
    "one_rep_max": {
        "label": "One Rep Max",
        "description": "Epley formula.",
        "fields": [
            {"name": "lifted_weight_kg", "label": "Weight lifted (kg)", "type": "number"},
            {"name": "reps", "label": "Reps", "type": "number"},
        ],
        "outputs": ["oneRepMaxKg"],
        "compute": calculate_one_rep_max,
    },
    "target_hr": {
        "label": "Target Heart Rate",
        "description": "Zones based on age.",
        "fields": [
            {"name": "age", "label": "Age", "type": "number"},
        ],
        "outputs": ["maxHeartRate", "moderateRange", "vigorousRange"],
        "compute": calculate_target_heart_rate,
    },
}

# Fix reference for army body fat fields (reuse body fat fields)
CALCULATOR_DEFINITIONS["army_body_fat"]["fields"] = [
    field.copy() for field in CALCULATOR_DEFINITIONS["body_fat"]["fields"]
]


def get_calculator_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for calc_id, definition in CALCULATOR_DEFINITIONS.items():
        configs.append(
            {
                "id": calc_id,
                "label": definition["label"],
                "description": definition["description"],
                "fields": definition["fields"],
                "outputs": definition["outputs"],
            }
        )
    return configs


def run_calculator(calc_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    definition = CALCULATOR_DEFINITIONS.get(calc_id)
    if not definition:
        raise ValueError(f"Unknown calculator '{calc_id}'.")
    compute: Callable[..., dict[str, Any]] = definition["compute"]

    kwargs: dict[str, Any] = {}
    for field in definition["fields"]:
        name = field["name"]
        value = inputs.get(name)
        if value in (None, ""):
            if field.get("optional"):
                continue
            raise ValueError(f"{field['label']} is required.")
        field_type = field.get("type", "text")
        try:
            if field_type == "number":
                kwargs[name] = float(value)
            else:
                kwargs[name] = value
        except ValueError as exc:
            raise ValueError(f"{field['label']} must be a number.") from exc

    return compute(**kwargs)
