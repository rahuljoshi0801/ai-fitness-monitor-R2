"""Simple SQLite helpers for user auth and daily check-in history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = PROJECT_ROOT / "instance"
DB_PATH = INSTANCE_DIR / "fitness.db"


def _connect() -> sqlite3.Connection:
    INSTANCE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS check_ins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                foods TEXT,
                steps INTEGER,
                workout_intensity TEXT,
                calorie_intake INTEGER,
                calories_burned INTEGER,
                maintenance_calories INTEGER,
                bmi REAL,
                bmi_category TEXT,
                recommended_goal TEXT,
                summary_json TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def _row_to_user(row: sqlite3.Row | None, include_sensitive: bool = False) -> dict[str, Any] | None:
    if row is None:
        return None
    user = {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "createdAt": row["created_at"],
    }
    if include_sensitive:
        user["password_hash"] = row["password_hash"]
    return user


def create_user(name: str, email: str, password_hash: str) -> dict[str, Any]:
    created_at = datetime.utcnow().isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, created_at),
        )
        user_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row)  # type: ignore[arg-type]


def find_user_by_email(email: str, include_sensitive: bool = False) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return _row_to_user(row, include_sensitive=include_sensitive)


def find_user_by_id(user_id: int, include_sensitive: bool = False) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row, include_sensitive=include_sensitive)


def record_check_in(user_id: int, payload: dict[str, Any]) -> None:
    created_at = datetime.utcnow().isoformat()
    foods = ", ".join(payload.get("foods", []))
    steps = int(payload.get("steps", 0))
<<<<<<< HEAD
    exercise_entries = payload.get("exerciseEntries") or payload.get("exercise_entries") or []
    primary_exercise_id = ""
    if isinstance(exercise_entries, list) and exercise_entries:
        first_entry = exercise_entries[0]
        if isinstance(first_entry, dict):
            primary_exercise_id = first_entry.get("id") or ""
        elif isinstance(first_entry, str):
            primary_exercise_id = first_entry

    workout_intensity = (
        primary_exercise_id
        or payload.get("exerciseId")
        or payload.get("workoutIntensity")
        or ""
    )
=======
    workout_intensity = payload.get("workoutIntensity", "")
>>>>>>> 78f879e262dae0458c71165928181f9fca795731

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO check_ins (
                user_id, created_at, foods, steps, workout_intensity,
                calorie_intake, calories_burned, maintenance_calories,
                bmi, bmi_category, recommended_goal, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                foods,
                steps,
                workout_intensity,
                payload.get("calorieIntake"),
                payload.get("caloriesBurned"),
                payload.get("maintenanceCalories"),
                payload.get("bmi"),
                payload.get("bmiCategory"),
                payload.get("recommendedGoal"),
                json.dumps(payload),
            ),
        )


def fetch_recent_check_ins(user_id: int, limit: int = 7) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM check_ins WHERE user_id = ? ORDER BY datetime(created_at) DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "id": row["id"],
            "createdAt": row["created_at"],
            "foods": row["foods"],
            "steps": row["steps"],
            "workoutIntensity": row["workout_intensity"],
<<<<<<< HEAD
            "exerciseId": row["workout_intensity"],
=======
>>>>>>> 78f879e262dae0458c71165928181f9fca795731
            "calorieIntake": row["calorie_intake"],
            "caloriesBurned": row["calories_burned"],
            "maintenanceCalories": row["maintenance_calories"],
            "bmi": row["bmi"],
            "bmiCategory": row["bmi_category"],
            "recommendedGoal": row["recommended_goal"],
        }
        summary_json = row["summary_json"]
<<<<<<< HEAD
        summary_data: dict[str, Any] | None = None
        if summary_json:
            try:
                summary_data = json.loads(summary_json)
            except json.JSONDecodeError:
                summary_data = None

        if summary_data:
            entry["summary"] = summary_data
            entry["exerciseEntries"] = summary_data.get("exerciseEntries")
            entry["exerciseDetails"] = summary_data.get("exerciseDetails")
            entry["exerciseDisplay"] = summary_data.get("exerciseDisplay") or summary_data.get("exerciseLabel")
            entry["exerciseIntensity"] = summary_data.get("exerciseIntensity") or entry.get("exerciseIntensity")
            micro_text = summary_data.get("microCoachText")
            if micro_text:
                entry["microCoachText"] = micro_text
        results.append(entry)
    return results


def fetch_check_ins_paginated(
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    """Return paginated, date-filtered check-ins for a user."""
    query = "SELECT * FROM check_ins WHERE user_id = ?"
    params: list[Any] = [user_id]

    if from_date:
        query += " AND date(created_at) >= date(?)"
        params.append(from_date)
    if to_date:
        query += " AND date(created_at) <= date(?)"
        params.append(to_date)

    query += " ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        # Determine if more entries exist beyond this page
        count_query = "SELECT COUNT(*) FROM check_ins WHERE user_id = ?"
        count_params = [user_id]
        if from_date:
            count_query += " AND date(created_at) >= date(?)"
            count_params.append(from_date)
        if to_date:
            count_query += " AND date(created_at) <= date(?)"
            count_params.append(to_date)
        total = conn.execute(count_query, count_params).fetchone()[0]
        has_more = (offset + limit) < total

    results: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "id": row["id"],
            "createdAt": row["created_at"],
            "foods": row["foods"],
            "steps": row["steps"],
            "workoutIntensity": row["workout_intensity"],
            "exerciseId": row["workout_intensity"],
            "calorieIntake": row["calorie_intake"],
            "caloriesBurned": row["calories_burned"],
            "maintenanceCalories": row["maintenance_calories"],
            "bmi": row["bmi"],
            "bmiCategory": row["bmi_category"],
            "recommendedGoal": row["recommended_goal"],
        }
        summary_json = row["summary_json"]
        summary_data: dict[str, Any] | None = None
        if summary_json:
            try:
                summary_data = json.loads(summary_json)
            except json.JSONDecodeError:
                summary_data = None

        if summary_data:
            entry["summary"] = summary_data
            entry["exerciseEntries"] = summary_data.get("exerciseEntries")
            entry["exerciseDetails"] = summary_data.get("exerciseDetails")
            entry["exerciseDisplay"] = summary_data.get("exerciseDisplay") or summary_data.get("exerciseLabel")
            entry["exerciseIntensity"] = summary_data.get("exerciseIntensity") or entry.get("exerciseIntensity")
            micro_text = summary_data.get("microCoachText")
            if micro_text:
                entry["microCoachText"] = micro_text
        results.append(entry)

    return {"entries": results, "hasMore": has_more, "total": total}


def fetch_check_ins_since(user_id: int, since: datetime) -> list[dict[str, Any]]:
    """Return all check-ins for a user since the provided timestamp."""
    cutoff = since.isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM check_ins
            WHERE user_id = ? AND datetime(created_at) >= datetime(?)
            ORDER BY datetime(created_at) DESC
            """,
            (user_id, cutoff),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "id": row["id"],
            "createdAt": row["created_at"],
            "foods": row["foods"],
            "steps": row["steps"],
            "workoutIntensity": row["workout_intensity"],
            "exerciseId": row["workout_intensity"],
            "calorieIntake": row["calorie_intake"],
            "caloriesBurned": row["calories_burned"],
            "maintenanceCalories": row["maintenance_calories"],
            "bmi": row["bmi"],
            "bmiCategory": row["bmi_category"],
            "recommendedGoal": row["recommended_goal"],
        }
        summary_json = row["summary_json"]
        summary_data: dict[str, Any] | None = None
        if summary_json:
            try:
                summary_data = json.loads(summary_json)
            except json.JSONDecodeError:
                summary_data = None

        if summary_data:
            entry["summary"] = summary_data
            entry["exerciseEntries"] = summary_data.get("exerciseEntries")
            entry["exerciseDetails"] = summary_data.get("exerciseDetails")
            entry["exerciseDisplay"] = summary_data.get("exerciseDisplay") or summary_data.get("exerciseLabel")
            entry["exerciseIntensity"] = summary_data.get("exerciseIntensity") or entry.get("exerciseIntensity")
            micro_text = summary_data.get("microCoachText")
            if micro_text:
                entry["microCoachText"] = micro_text
=======
        if summary_json:
            try:
                entry["summary"] = json.loads(summary_json)
            except json.JSONDecodeError:
                entry["summary"] = None
>>>>>>> 78f879e262dae0458c71165928181f9fca795731
        results.append(entry)
    return results
