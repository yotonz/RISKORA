import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database.db")
LEGACY_USER_DB = os.path.join(_BASE, "users.json")


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_password):
    if not stored_password:
        return False
    if "$" not in stored_password:
        return stored_password == password
    salt, _ = stored_password.split("$", 1)
    return hash_password(password, salt) == stored_password


def create_tables():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT NOT NULL,
            applicant_email TEXT,
            phone TEXT,
            purpose TEXT,
            notes TEXT,
            age INTEGER NOT NULL,
            income REAL NOT NULL,
            loan REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            emi REAL NOT NULL,
            dti REAL NOT NULL,
            score INTEGER NOT NULL,
            risk TEXT NOT NULL,
            submitted_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def migrate_legacy_users():
    if not os.path.exists(LEGACY_USER_DB):
        return

    with open(LEGACY_USER_DB, "r", encoding="utf-8") as file:
        legacy_users = json.load(file)

    if not legacy_users:
        return

    for username, user_data in legacy_users.items():
        role = user_data.get("role", "user")
        password = user_data.get("password", "")
        add_user(username, password, role, already_hashed=False)


def init_db():
    create_tables()
    migrate_legacy_users()


def get_user(username):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def add_user(username, password, role="user", already_hashed=False):
    stored_password = password if already_hashed else hash_password(password)
    conn = connect()
    try:
        conn.execute(
            """
            INSERT INTO users (username, password, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, stored_password, role, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def validate_login(username, password):
    user = get_user(username)
    if not user:
        return None
    if verify_password(password, user["password"]):
        return user["role"]
    return None


def insert_application(application):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO applications (
            applicant_name, applicant_email, phone, purpose, notes,
            age, income, loan, credit_score, emi, dti, score, risk,
            submitted_by, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            application["applicant_name"],
            application.get("applicant_email", ""),
            application.get("phone", ""),
            application.get("purpose", ""),
            application.get("notes", ""),
            application["age"],
            application["income"],
            application["loan"],
            application["credit_score"],
            application["emi"],
            application["dti"],
            application["score"],
            application["risk"],
            application["submitted_by"],
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_applications():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applications ORDER BY created_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_user_applications(username):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM applications WHERE submitted_by = ? ORDER BY created_at DESC",
        (username,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def delete_application(application_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM applications WHERE id = ?", (application_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_application_stats():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM applications")
    total = cursor.fetchone()["total"]

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM applications
        WHERE risk = 'HIGH RISK'
        """
    )
    high_risk = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total FROM users")
    total_users = cursor.fetchone()["total"]
    conn.close()

    return {
        "total_applications": total,
        "high_risk_count": high_risk,
        "low_risk_count": total - high_risk,
        "total_users": total_users,
    }
