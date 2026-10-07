import sqlite3
from datetime import datetime

DATABASE = "database.db"


def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT
        )
    """)

    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT,
            confidence REAL,
            created_at TEXT,
            user_id INTEGER
        )
    """)

    # Check if created_at and user_id columns exist in predictions table
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN created_at TEXT")
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")

    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def create_user(username, email, password_hash):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%d %b %Y")

    try:
        cursor.execute(
            """
            INSERT INTO users(username, email, password, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, password_hash, date_str)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError as e:
        conn.close()
        err_msg = str(e)
        if "email" in err_msg:
            return None, "Email address is already registered."
        elif "username" in err_msg:
            return None, "Username is already taken."
        return None, "Registration failed. User already exists."


def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def insert_prediction(prediction, confidence, user_id=None):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%d %b %Y")

    cursor.execute(
        """
        INSERT INTO predictions(prediction, confidence, created_at, user_id)
        VALUES (?, ?, ?, ?)
        """,
        (prediction, confidence, date_str, user_id)
    )

    conn.commit()
    conn.close()


def get_predictions(user_id=None):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT id, prediction, confidence, COALESCE(created_at, strftime('%d %b %Y', 'now')) as created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT id, prediction, confidence, COALESCE(created_at, strftime('%d %b %Y', 'now')) as created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
        """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_prediction_stats(user_id=None):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    if user_id:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,))
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE (prediction LIKE '%Real%' OR prediction LIKE '%Genuine%') AND user_id = ?", (user_id,))
        real_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE (prediction LIKE '%Fake%' OR prediction LIKE '%Fraud%') AND user_id = ?", (user_id,))
        fake_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM predictions WHERE user_id = ?", (user_id,))
        avg_conf = cursor.fetchone()[0]
    else:
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE (prediction LIKE '%Real%' OR prediction LIKE '%Genuine%')")
        real_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE (prediction LIKE '%Fake%' OR prediction LIKE '%Fraud%')")
        fake_count = cursor.fetchone()[0]


        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = cursor.fetchone()[0]

    avg_conf = round(avg_conf, 1) if avg_conf is not None else 0.0
    conn.close()

    real_pct = round((real_count / total * 100), 1) if total > 0 else 0.0
    fake_pct = round((fake_count / total * 100), 1) if total > 0 else 0.0

    return {
        "total": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "avg_conf": avg_conf
    }



def clear_all_predictions(user_id=None):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    if user_id:
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    else:
        cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

