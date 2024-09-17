'''
Image Upload and Processing: Allow the site engineer to upload images of parts.
Feature Extraction: Use a deep learning model to extract features from these images.
Feature Comparison: Compare the extracted features to those stored in the database.
Matching: Return whether a match is found.
'''

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import sqlite3
import os

app = Flask(__name__)

# Path to database
DATABASE = 'parts_database.db'

# Load pre-trained model for feature extraction
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_number TEXT UNIQUE NOT NULL,
                feature BLOB NOT NULL
            )
        ''')
    print("Database initialized.")

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features.flatten()

def store_part(part_number, features):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO parts (part_number, feature) VALUES (?, ?)', 
                       (part_number, features.tobytes()))
        conn.commit()

def get_part_features(part_number):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT feature FROM parts WHERE part_number = ?', (part_number,))
        result = cursor.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float32)
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    part_number = request.form.get('part_number')

    if not part_number:
        return jsonify({'error': 'No part number provided'}), 400

    image_path = f'/tmp/{file.filename}'
    file.save(image_path)
    
    uploaded_features = extract_features(image_path)
    
    stored_features = get_part_features(part_number)

    if stored_features is not None:
        similarity = cosine_similarity(uploaded_features, stored_features)
        if similarity > 0.9:  # threshold for matching
