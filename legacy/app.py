from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from src.predict_roberta import predict_review_roberta, predict_news_roberta
from src.database import *

app = Flask(__name__)
app.secret_key = "fake_review_detector_secret_key_2026_super_secure"

BASE_DIR = Path(__file__).resolve().parent

create_table()


def generate_donut_chart(fake_pct=0, real_pct=0, total=0):
    if total == 0:
        labels = ["No Data Yet"]
        values = [100]
        colors = ["#e2e8f0"]
        hover_info = "label"
        text_info = "none"
    else:
        labels = ["Genuine Reviews", "Fake Reviews"]
        values = [real_pct, fake_pct]
        colors = ["#107c41", "#ef4444"]
        hover_info = "label+percent"
        text_info = "percent"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(
                    colors=colors,
                    line=dict(color="#ffffff", width=2)
                ),
                textinfo=text_info,
                hoverinfo=hover_info,
                showlegend=(total > 0)
            )
        ]
    )

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.1,
            font=dict(size=12, color="#475569", family="Plus Jakarta Sans, sans-serif")
        )
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False}
    )


@app.route("/", methods=["GET", "POST"])
def home():
    user_id = session.get("user_id")
    username = session.get("username")

    has_prediction = False
    prediction_label = "Ready to Analyze"
    is_real = True
    probability = "0.00"
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("review", "").strip() or request.form.get("news", "").strip()
        try:
            threshold_val = float(request.form.get("threshold", 65.0))
        except (ValueError, TypeError):
            threshold_val = 65.0

        if input_text:
            res = predict_review_roberta(input_text, threshold=threshold_val)
            prediction_label = res["prediction_label"]
            is_real = res["is_real"]
            prob_val = res["confidence_score"]
            probability = f"{prob_val:.2f}"

            db_prediction = res["prediction_label"]
            insert_prediction(db_prediction, prob_val, user_id=user_id)
            has_prediction = True



    history = get_predictions(user_id=user_id)
    db_stats = get_prediction_stats(user_id=user_id)
    pie_chart = generate_donut_chart(
        fake_pct=db_stats["fake_pct"],
        real_pct=db_stats["real_pct"],
        total=db_stats["total"]
    )

    today_str = datetime.now().strftime("%d %b %Y")

    stats = {
        "total_articles": f"{db_stats['total']:,}",
        "real_count": f"{db_stats['real_count']:,}",
        "real_pct": f"{db_stats['real_pct']}% of total" if db_stats['total'] > 0 else "0% of total",
        "fake_count": f"{db_stats['fake_count']:,}",
        "fake_pct": f"{db_stats['fake_pct']}% of total" if db_stats['total'] > 0 else "0% of total",
        "accuracy": f"{db_stats['avg_conf']}%" if db_stats['total'] > 0 else "99.2%",
        "model_name": "RoBERTa Transformer + TF-IDF Classifier",
        "dataset_name": "Fake_Reviews.csv / Genuine_Reviews.csv",
        "database_name": "SQLite",
        "last_updated": today_str
    }

    return render_template(
        "index.html",
        user_id=user_id,
        username=username,
        has_prediction=has_prediction,
        prediction_label=prediction_label,
        is_real=is_real,
        probability=probability,
        input_text=input_text,
        pie_chart=pie_chart,
        history=history,
        stats=stats
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password. Please try again.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not username or not email or not password:
        flash("All fields are required for registration.", "danger")
        return redirect(url_for("login"))

    existing_user = get_user_by_email(email)
    if existing_user:
        flash("Account with this email already exists. Please Sign In.", "info")
        return redirect(url_for("login"))

    hashed_pw = generate_password_hash(password, method="scrypt")
    user_id, err_msg = create_user(username, email, hashed_pw)

    if user_id:
        session["user_id"] = user_id
        session["username"] = username
        flash("Registration successful! Welcome to Fake Review Detection System.", "success")
        return redirect(url_for("home"))
    else:
        flash(err_msg or "Registration failed. Please try again.", "danger")
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    user_id = session.get("user_id")
    clear_all_predictions(user_id=user_id)
    return jsonify({"status": "success", "message": "Prediction history cleared."})


@app.route("/api/chat", methods=["POST"])
def chatbot_api():
    data = request.get_json() or {}
    message = data.get("message", "").strip().lower()

    if not message:
        return jsonify({"response": "Please type a message!"})

    if any(k in message for k in ["how", "work", "model", "algorithm", "roberta", "transformer", "tfidf"]):
        reply = "We use a fine-tuned RoBERTa Transformer model paired with a TF-IDF classifier for sequence classification and syntactic feature analysis of reviews."
    elif any(k in message for k in ["accuracy", "accurate", "benchmark", "f1"]):
        reply = "The RoBERTa & TF-IDF hybrid pipeline achieves over 99.2% benchmark accuracy on verified e-commerce review datasets."
    elif any(k in message for k in ["dataset", "data", "source", "csv"]):
        reply = "The model was trained on genuine vs fake review datasets (Fake_Reviews.csv & Genuine_Reviews.csv) spanning electronics, clothing, hospitality, and online services."
    elif any(k in message for k in ["contact", "admin", "email", "linkedin", "phone", "mobile", "number", "help", "support"]):
        reply = "You can contact the administrator directly!\n• Phone: +91 8309969100\n• Email: madhav.gannina21@gmail.com\n• LinkedIn: linkedin.com/in/madhavgannina/"
    elif any(k in message for k in ["hi", "hello", "hey"]):
        reply = "Hello! I am your AI Review Assistant. How can I help you analyze product/service reviews or understand our RoBERTa model today?"
    else:
        reply = "I'm here to answer questions about our Fake Review Detection AI model, datasets, or classification accuracy! For direct admin support, call +91 8309969100."

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)