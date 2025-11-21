import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def insert_service_request(data):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """INSERT INTO service_requests 
        (name, phone, car_brand, car_model, car_age_years, km_driven, last_service_months,
        issue_text, predicted_category, predicted_priority, predicted_hours,
        groq_reasoning, groq_recommendation, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    values = (
        data["name"],
        data["phone"],
        data["car_brand"],
        data["car_model"],
        data["car_age_years"],
        data["km_driven"],
        data["last_service_months"],
        data["issue_text"],
        data["predicted_category"],
        data["predicted_priority"],
        data["predicted_hours"],
        data["groq_reasoning"],
        data["groq_recommendation"],
        datetime.now()
    )

    cursor.execute(sql, values)
    conn.commit()

    cursor.close()
    conn.close()

def fetch_all_requests():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM service_requests ORDER BY created_at DESC")
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows
