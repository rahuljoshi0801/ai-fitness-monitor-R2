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
    workout_intensity = payload.get("workoutIntensity", "")

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
            "calorieIntake": row["calorie_intake"],
            "caloriesBurned": row["calories_burned"],
            "maintenanceCalories": row["maintenance_calories"],
            "bmi": row["bmi"],
            "bmiCategory": row["bmi_category"],
            "recommendedGoal": row["recommended_goal"],
        }
        summary_json = row["summary_json"]
        if summary_json:
            try:
                entry["summary"] = json.loads(summary_json)
            except json.JSONDecodeError:
                entry["summary"] = None
        results.append(entry)
    return results
