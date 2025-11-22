import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from ml_utils import TextSelector, NumSelector

load_dotenv()

import pandas as pd
import joblib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from db import insert_service_request, fetch_all_requests

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load ML models
service_model = joblib.load("models/service_category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
duration_model = joblib.load("models/duration_model.pkl")
# Groq Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
print("GROQ_API_KEY:", GROQ_API_KEY)

def call_groq(issue_text, category, priority, hours):
    prompt = f"""
Issue: {issue_text}
Service Category: {category}
Priority: {priority}
Estimated Hours: {hours}

Explain the reasoning in 2–3 sentences.
Then provide next-step recommendations in bullet points.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an automotive workshop expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 350
    }

    print("\n===== GROQ REQUEST =====")
    print(payload)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )

        print("===== GROQ RAW RESPONSE =====")
        print(response.text)

        data = response.json()

        content = data["choices"][0]["message"]["content"]

# Split text into reasoning and recommendations
        if "Next-step recommendations:" in content:
            parts = content.split("Next-step recommendations:")
            reasoning = parts[0].strip()

    # extract bullet points
            rec_text = parts[1].strip()
            recommendations = "\n".join(
            line for line in rec_text.split("\n") if line.strip()
        )   

        else:
    # fallback if Groq changes formatting
            reasoning = content
            recommendations = "No recommendations found."

        return reasoning, recommendations
    except Exception as e:
        print("===== GROQ API ERROR =====")
        print("Error type:", type(e))
        print("Error message:", str(e))
        return "Groq request failed", "Groq request failed"
    
@app.route("/")
def home():
    return render_template("form.html")


@app.route("/submit", methods=["POST"])
def submit():
    # Gather input
    d = {k: request.form.get(k) for k in request.form}

    # Convert numbers
    d["car_age_years"] = int(d["car_age_years"])
    d["km_driven"] = int(d["km_driven"])
    d["last_service_months"] = int(d["last_service_months"])

    # ML Prediction
    X = pd.DataFrame([{
        "issue_text": d["issue_text"],
        "car_age_years": d["car_age_years"],
        "km_driven": d["km_driven"],
        "last_service_months": d["last_service_months"]
    }])

    category = service_model.predict(X)[0]
    priority = priority_model.predict(X)[0]
    hours = round(float(duration_model.predict(X)[0]))

    # Groq LLM
    reasoning, recommendation = call_groq(d["issue_text"], category, priority, hours)

    # Save DB
    payload = {
        **d,
        "predicted_category": category,
        "predicted_priority": priority,
        "predicted_hours": hours,
        "groq_reasoning": reasoning,
        "groq_recommendation": recommendation
    }

    insert_service_request(payload)

    return render_template(
        "success.html",
        category=category,
        priority=priority,
        hours=hours,
        reasoning=reasoning,
        recommendation=recommendation
    )


@app.route("/admin")
def admin():
    rows = fetch_all_requests()
    print(rows)
    df = pd.DataFrame(rows)

    # ___ charts ___
    os.makedirs("static/charts", exist_ok=True)

    if not df.empty:

        # Category Chart
        plt.figure(figsize=(10,10))
        plt.tight_layout()
        df["predicted_category"].value_counts().plot(kind="bar")
        plt.title("Requests per Category")
        plt.savefig(r"C:\Users\MEHAK\OneDrive\Documents\carAssist\app\static\charts\category.png")
        plt.close()

        # Priority Chart
        plt.figure(figsize=(10,10))
        plt.tight_layout()
        df["predicted_priority"].value_counts().plot(kind="bar")
        plt.title("Requests per Priority")
        plt.savefig(r"C:\Users\MEHAK\OneDrive\Documents\carAssist\app\static\charts\priority.png")
        plt.close()

    return render_template(
        "admin.html",
        total=len(rows),
        rows=rows
    )


if __name__ == "__main__":
    app.run(debug=True)
