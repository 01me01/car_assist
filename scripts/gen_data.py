# scripts/generate_data.py
import random
import pandas as pd
import numpy as np
import faker
from datetime import datetime

fake = faker.Faker()

N = 2000

brands = ['Toyota', 'Hyundai', 'Honda', 'Maruti', 'BMW', 'Mahindra', 'Kia']
models = {
    'Toyota':['Corolla','Camry'],
    'Hyundai':['i20','Creta'],
    'Honda':['City','Civic'],
    'Maruti':['Swift','Baleno'],
    'BMW':['3-Series','5-Series'],
    'Mahindra':['Thar','Bolero'],
    'Kia':['Seltos','Sonet']
}
service_categories = ['Engine', 'Electrical', 'Brakes', 'Oil Change', 'Tire', 'AC', 'General Check']
priority_levels = ['Low','Medium','High']
issue_text_samples = [
    'car not starting intermittently',
    'brakes making noise',
    'engine oil leak',
    'aircon not cooling',
    'battery dies quickly',
    'vibration while accelerating',
    'strange smell from cabin',
    'tyre puncture frequently',
    'squealing noise from belt'
]

rows = []
for i in range(N):
    brand = random.choice(brands)
    model = random.choice(models[brand])
    car_age = random.randint(0, 15)
    km_driven = int(max(1000, np.random.normal(40000, 25000)))
    last_service_months = random.randint(0, 36)
    issue_text = random.choice(issue_text_samples) + ('. ' + fake.sentence(nb_words=6) if random.random() < 0.6 else '')
    # simple heuristic to synthesize labels
    if 'oil' in issue_text or car_age>8:
        category = 'Oil Change' if 'oil' in issue_text else random.choice(['Engine','General Check'])
    elif 'brake' in issue_text:
        category = 'Brakes'
    elif 'aircon' in issue_text:
        category = 'AC'
    elif 'battery' in issue_text:
        category = 'Electrical'
    elif 'tyre' in issue_text or 'puncture' in issue_text:
        category = 'Tire'
    else:
        category = random.choice(service_categories)

    # priority heuristics
    if 'not starting' in issue_text or 'dies' in issue_text or 'vibration' in issue_text:
        priority = 'High'
    elif car_age > 10 or km_driven > 120000 or last_service_months>24:
        priority = 'Medium'
    else:
        priority = 'Low'

    # estimated hours heuristic
    base = {
        'Engine': 6, 'Electrical': 2.5, 'Brakes': 3.5, 'Oil Change': 1.5, 'Tire': 1.0, 'AC': 4, 'General Check': 2
    }.get(category, 2)
    estimated_hours = round(max(0.5, np.random.normal(base, base*0.3)), 1)

    rows.append({
        'name': fake.name(),
        'phone': fake.phone_number(),
        'car_brand': brand,
        'car_model': model,
        'car_age_years': car_age,
        'km_driven': km_driven,
        'last_service_months': last_service_months,
        'issue_text': issue_text,
        'service_category': category,
        'priority_level': priority,
        'estimated_hours': estimated_hours
    })

df = pd.DataFrame(rows)
df.to_csv('data/service_requests.csv', index=False)
print("Saved data/service_requests.csv with", len(df), "rows")
