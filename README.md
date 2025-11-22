# Car Workshop Service Intelligence
Full End-to-End AI System • ML Models • Groq LLM Reasoning • MySQL • Admin Dashboard

This project is an end-to-end AI-powered service management system for a car workshop.
It processes customer service requests, predicts service details using local ML models, generates expert reasoning using Groq LLM, stores all results in a MySQL database, and provides an admin dashboard with charts and analytics.

## Features

✔ Customer Request Form

✔ Local ML Models (Category, Priority, Duration)

✔ Groq LLM for reasoning + recommendations

✔ MySQL database integration

✔ Admin dashboard with live analytics

✔ Auto-updating bar charts (Matplotlib)

✔ Clean UI with responsive CSS

## How to Run the Project
### 1. Clone the Repository
	git clone https://github.com/01me01/car_assist.git
	cd car_assist

### 2. Create Virtual Environment
	python -m venv venv
	venv\Scripts\activate

### 3. Install Dependencies
	pip install -r requirements.txt

### 4. Set Up Environment Variables (.env)

Create a file named .env in the root:

	GROQ_API_KEY=your_groq_api_key_here
	DB_HOST=127.0.0.1
	DB_USER=root
	DB_PASSWORD=yourpassword
	DB_NAME=carworkshop

### 5. Create MySQL Database

Run in MySQL:

	CREATE DATABASE carworkshop;
	USE carworkshop;

	CREATE TABLE service_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200),
    phone VARCHAR(50),
    car_brand VARCHAR(100),
    car_model VARCHAR(100),
    car_age_years INT,
    km_driven INT,
    last_service_months INT,
    issue_text TEXT,
    predicted_category VARCHAR(100),
    predicted_priority VARCHAR(50),
    predicted_hours FLOAT,
    groq_reasoning TEXT,
    groq_recommendation TEXT,
    created_at DATETIME
	);

### 6. Run the Application
	python app.py

Open in browser:

	http://127.0.0.1:5000

## Dataset Generation

The project uses a synthetic dataset generated using:

	scripts/gen_data.py


Generated dataset is stored at:

	data/service_requests.csv

Run dataset generation:

	python scripts/gen_data.py

## Machine Learning Approach
### 1. Service Category Model

Logistic Regression / RandomForest

Predicts: Electrical, Engine, Mechanical, AC, etc.

### 2. Priority Level Model

RandomForestClassifier

Predicts urgency: Low / Medium / High

### 3. Duration Model

Linear Regression

Predicts estimated service hours

Model files:

    models/service_category_model.pkl
    models/priority_model.pkl
    models/duration_model.pkl

Groq LLM Usage
Code:
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You are an automotive expert."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=350,
    temperature=0.2 
    )

Groq returns:

Reasoning

Recommendations

## System Architecture:
<img width="621" height="118" alt="Carassist" src="https://github.com/user-attachments/assets/9182458c-2f40-446d-a206-cd98fb235f47" />

## Project Structure:

    car_assist/
    │── app
        │── static/
        │── templates/
        │── app.py
        │── db.py
        │── ml_utils.py   
    │── models/
    │── data/
    │── scripts/
    │── .env (ignored)
    │── .gitignore
    │── README.md

