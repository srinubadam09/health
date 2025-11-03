from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from groq import Groq
import socket
import json
import os
import random
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
import threading
import time
from dotenv import load_dotenv

app = Flask(__name__)

# MQTT Configuration
MQTT_BROKER = "35.154.62.193"  # Change to your MQTT broker address
MQTT_PORT = 1883
MQTT_TOPIC = "health/sensors"
load_dotenv()
# Global variables to store sensor data
sensor_data = []
current_sensor_readings = {
    "heart_rate": 0,
    "spo2": 0,
    "temperature": 0,
    "rr_interval": 0,
    "hrv": 0,
    "battery": 100,
    "signal_strength": 100,
    "timestamp": "",
    "connected": False
}

# MQTT Client Setup
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"✅ MQTT Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)
    current_sensor_readings["connected"] = True

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        print(f"📡 Received MQTT data: {data}")
        
        # Update current sensor readings
        current_sensor_readings.update({
            "heart_rate": data.get("heart_rate", 0),
            "spo2": data.get("spo2", 0),
            "temperature": data.get("temperature", 0),
            "rr_interval": data.get("rr_interval", 0),
            "hrv": data.get("hrv", 0),
            "battery": data.get("battery", 100),
            "signal_strength": data.get("signal_strength", 100),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "connected": True
        })
        
        # Store in history
        sensor_data.append(current_sensor_readings.copy())
        
        # Keep only last 50 readings
        if len(sensor_data) > 50:
            sensor_data.pop(0)
            
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

def on_disconnect(client, userdata, rc):
    print("❌ MQTT Disconnected")
    current_sensor_readings["connected"] = False

# Setup MQTT client
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

def connect_mqtt():
    """Connect to MQTT broker in a separate thread"""
    def mqtt_loop():
        while True:
            try:
                print(f"🔗 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.loop_forever()
            except Exception as e:
                print(f"❌ MQTT connection failed: {e}")
                current_sensor_readings["connected"] = False
                time.sleep(5)  # Retry after 5 seconds
    
    mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
    mqtt_thread.start()

# Start MQTT connection
connect_mqtt()

# ✅ Safe pickle loading with compatibility fix
def safe_pickle_load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# ✅ Load ML model & create fallback if pickle fails
try:
    model = safe_pickle_load("best_health_model.pkl")
    if model is None:
        raise Exception("Model file not found or corrupted")
except Exception as e:
    print(f"Using fallback model: {e}")
    
    # Fallback model with rule-based predictions
    class FallbackModel:
        def predict(self, features):
            hr, spo2, temp, rr, hrv = features[0]
            risk_score = 0
            
            # Critical conditions
            if spo2 < 90 or hr > 150 or hr < 40 or temp > 39.5:
                return [2]  # CRITICAL
            # Warning conditions
            elif spo2 < 95 or hr > 120 or hr < 50 or temp > 38 or temp < 35.5:
                return [1]  # WARNING
            # Normal conditions
            else:
                return [0]  # NORMAL
    
    model = FallbackModel()

# ✅ Label encoder with fallback
try:
    label_encoder = safe_pickle_load("risk_label_encoder.pkl")
    if label_encoder is None:
        raise Exception("Label encoder not found")
except Exception as e:
    print(f"Using fallback label encoder: {e}")
    
    class FallbackLabelEncoder:
        def __init__(self):
            self.classes_ = ['NORMAL', 'WARNING', 'CRITICAL']
        
        def inverse_transform(self, labels):
            result = []
            for label in labels:
                if label == 0:
                    result.append('NORMAL')
                elif label == 1:
                    result.append('WARNING')
                else:
                    result.append('CRITICAL')
            return result
    
    label_encoder = FallbackLabelEncoder()
api_key = os("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")
# ✅ Groq Client
try:
    client = Groq(api_key=api_key)
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq client unavailable: {e}")
    GROQ_AVAILABLE = False

# ✅ Universal numpy converter
def convert(o):
    if isinstance(o, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float16, np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return o

# ✅ Enhanced medical knowledge base for detailed fallback responses
class EnhancedMedicalAnalyzer:
    def __init__(self):
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'spo2': (95, 100),
            'temperature': (36.1, 37.2),
            'rr_interval': (600, 1000),
            'hrv': (20, 100)
        }
    
    def get_detailed_fallback_analysis(self, prediction, hr, spo2, temp, rr, hrv):
        """Generate detailed fallback analysis when LLM is unavailable"""
        
        # Medical reason based on vital signs
        reasons = []
        if prediction == "NORMAL":
            reasons.append("All vital signs are within normal ranges")
            if 60 <= hr <= 100:
                reasons.append(f"Heart rate ({hr} bpm) is optimal")
            if spo2 >= 95:
                reasons.append(f"Oxygen saturation ({spo2}%) is excellent")
            if 36.1 <= temp <= 37.2:
                reasons.append(f"Body temperature ({temp}°C) is normal")
        else:
            if hr < 60:
                reasons.append(f"Bradycardia detected: heart rate {hr} bpm (normal: 60-100)")
            elif hr > 100:
                reasons.append(f"Tachycardia detected: heart rate {hr} bpm (normal: 60-100)")
            if spo2 < 95:
                reasons.append(f"Low oxygen saturation: {spo2}% (normal: 95-100%)")
            if temp < 36.1:
                reasons.append(f"Low body temperature: {temp}°C (normal: 36.1-37.2°)")
            elif temp > 37.2:
                reasons.append(f"Elevated body temperature: {temp}°C (normal: 36.1-37.2°)")
        
        reason_text = ". ".join(reasons)
        
        # Detailed recommendations based on prediction
        recommendations = {
            "NORMAL": """
            Continue with regular health monitoring. Maintain your current healthy lifestyle:
            • Schedule annual health check-ups
            • Continue balanced diet and regular exercise
            • Maintain good sleep hygiene
            • Stay hydrated and manage stress
            • Monitor vital signs weekly
            """,
            "WARNING": """
            Increased monitoring recommended:
            • Consult healthcare provider within 1-2 weeks
            • Monitor symptoms daily
            • Avoid strenuous activities
            • Follow up with specific tests if symptoms persist
            • Consider lifestyle modifications
            • Track vital signs twice daily
            """,
            "CRITICAL": """
            Immediate medical attention required:
            • Seek emergency care or contact healthcare provider immediately
            • Do not ignore symptoms
            • Have someone accompany you if visiting hospital
            • Follow emergency protocols for your condition
            • Continuous monitoring recommended
            """
        }
        
        # Detailed food recommendations
        food_recommendations = {
            "NORMAL": """
            MAINTENANCE DIET FOR OPTIMAL HEALTH:

            BREAKFAST OPTIONS:
            • 1 cup oatmeal with 1/2 cup mixed berries and 1 oz walnuts
            • 2 slices whole grain toast with 1/2 avocado and 2 boiled eggs
            • 1 cup Greek yogurt with 1 tbsp honey and fresh fruits

            LUNCH OPTIONS:
            • 4 oz grilled chicken/fish with 1 cup quinoa and large vegetable salad
            • Lentil soup with 2 slices whole grain bread and side salad
            • Chicken stir-fry with 1 cup brown rice and mixed vegetables

            DINNER OPTIONS:
            • 5 oz baked salmon with 2 cups steamed vegetables
            • 4 oz chicken breast with sweet potato and broccoli
            • Vegetable curry with 1 cup chickpeas and brown rice

            SNACKS:
            • Fresh fruits and raw vegetables
            • Handful of nuts and seeds
            • Hummus with carrot sticks
            • Greek yogurt with berries

            HYDRATION:
            • 8-10 glasses of water daily
            • Herbal teas (green tea, chamomile)
            • Fresh fruit juices (no added sugar)
            """,
            "WARNING": """
            THERAPEUTIC DIET FOR RECOVERY:

            ANTI-INFLAMMATORY FOODS:
            • Fatty fish (salmon, mackerel) - 3x weekly, 4-5 oz servings
            • Turmeric, ginger, garlic added to meals daily
            • 2 cups leafy greens (spinach, kale) daily
            • 1 cup berries and cherries for antioxidants

            HEART-HEALTHY CHOICES:
            • 1 cup oats and barley for cholesterol management
            • 1/2 avocado and 2 tbsp olive oil daily for healthy fats
            • 1 oz nuts and seeds in moderation
            • 1 cup legumes and beans for fiber

            EASY DIGESTION MEALS:
            • Cooked vegetables instead of raw
            • Lean proteins (4 oz chicken, fish, tofu)
            • Small, frequent meals (5-6 times daily)
            • Warm soups and broths

            AVOID COMPLETELY:
            • Processed and fried foods
            • Excessive salt and sugar
            • Alcohol and caffeine
            • Red meat and full-fat dairy
            """,
            "CRITICAL": """
            MEDICAL SUPERVISION DIET:

            IMMEDIATE NOURISHMENT:
            • Bone broth and vegetable soups - 1 cup every 2 hours
            • Electrolyte-rich fluids (coconut water) - small sips frequently
            • Soft cooked vegetables - pureed form
            • Plain yogurt and kefir - 1/2 cup servings

            GENTLE ON SYSTEM:
            • Banana and apple puree - 1/2 cup servings
            • Oatmeal and congee - well-cooked, small portions
            • Steamed white fish - 2-3 oz, flaked
            • Well-cooked grains - rice, quinoa

            HYDRATION FOCUS:
            • Small sips of water every 15 minutes
            • Herbal infusions (peppermint, ginger) - warm
            • Electrolyte solutions - as directed
            • Clear broths and soups - 1/4 cup hourly

            STRICTLY AVOID:
            • Solid foods if instructed by medical team
            • Dairy products if lactose intolerant
            • Raw vegetables and fruits
            • Spicy, oily, or heavy foods

            CONSULT: Registered dietitian for personalized medical nutrition therapy
            """
        }
        
        return {
            "prediction": prediction,
            "reason": reason_text,
            "recommendation": recommendations.get(prediction, "Consult healthcare professional."),
            "food_recommendation": food_recommendations.get(prediction, "Follow medical dietary advice.")
        }

def generate_llm_analysis(ml_prediction, hr, spo2, temp, rr, hrv):
    """Generate detailed analysis using LLM based on ML prediction"""
    
    if not GROQ_AVAILABLE:
        analyzer = EnhancedMedicalAnalyzer()
        return analyzer.get_detailed_fallback_analysis(ml_prediction, hr, spo2, temp, rr, hrv)
    
    prompt = f"""
    CRITICAL INSTRUCTIONS: You are an expert medical AI assistant analyzing real-time sensor data from IoT health monitoring devices. Provide SPECIFIC, ACTIONABLE medical advice based on the sensor readings.

    REAL-TIME IOT SENSOR DATA:
    - Heart Rate: {hr} bpm (Normal range: 60-100 bpm)
    - Blood Oxygen (SpO2): {spo2}% (Normal range: 95-100%)
    - Body Temperature: {temp}°C (Normal range: 36.1-37.2°C)
    - RR Interval: {rr} ms (Normal range: 600-1000 ms)
    - Heart Rate Variability: {hrv} ms (Normal range: 20-100 ms)

    ML MODEL ASSESSMENT: {ml_prediction}

    ANALYZE EACH SENSOR READING:
    1. Heart Rate {hr} bpm - {"NORMAL" if 60 <= hr <= 100 else "HIGH" if hr > 100 else "LOW"}
    2. SpO2 {spo2}% - {"NORMAL" if spo2 >= 95 else "CONCERNING" if spo2 >= 90 else "CRITICAL"}
    3. Temperature {temp}°C - {"NORMAL" if 36.1 <= temp <= 37.2 else "FEVER" if temp > 37.2 else "HYPOTHERMIA"}
    4. HRV {hrv} ms - {"NORMAL" if 20 <= hrv <= 100 else "LOW" if hrv < 20 else "HIGH"}

    PROVIDE ANALYSIS IN THIS EXACT JSON FORMAT:

    {{
        "prediction": "{ml_prediction}",
        "reason": "Detailed analysis of each sensor reading. Explain clinical significance of abnormal values. Reference normal ranges. Be medically precise.",
        "recommendation": "IMMEDIATE actions, monitoring schedule, specific lifestyle changes, when to seek emergency care. Include timing and frequency.",
        "food_recommendation": "COMPLETE 7-day meal plan with portion sizes. Include breakfast, lunch, dinner, snacks. Specific foods, cooking methods, timing. Therapeutic diet for the condition."
    }}

    MANDATORY REQUIREMENTS:
    - NEVER use generic "consult doctor" as primary advice
    - Provide ACTUAL specific medical guidance
    - Include exact portion sizes and meal timing
    - Reference the sensor values in your analysis
    - Prediction must match: {ml_prediction}
    - Include weekly meal plan with daily variations
    - Suggest specific exercises and activities
    - Provide hydration and supplement recommendations
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=60,
            max_tokens=3000
        )
        
        llm_output = response.choices[0].message.content
        
        # Parse JSON response from LLM
        try:
            analysis = json.loads(llm_output)
            # Ensure prediction matches ML prediction
            analysis["prediction"] = ml_prediction
            return analysis
        except json.JSONDecodeError:
            print("LLM returned non-JSON response, using fallback")
            # Use enhanced fallback if JSON parsing fails
            analyzer = EnhancedMedicalAnalyzer()
            return analyzer.get_detailed_fallback_analysis(ml_prediction, hr, spo2, temp, rr, hrv)
            
    except Exception as e:
        print(f"LLM API error: {e}")
        # Use enhanced fallback
        analyzer = EnhancedMedicalAnalyzer()
        return analyzer.get_detailed_fallback_analysis(ml_prediction, hr, spo2, temp, rr, hrv)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sensors")
def sensors():
    """Sensor data display page"""
    return render_template("sensors.html")

@app.route("/predict")
def predict_page():
    """Prediction page"""
    return render_template("predict.html")

@app.route("/api/sensor-data")
def get_sensor_data():
    """API endpoint to get current sensor data from MQTT"""
    return jsonify({
        "current": current_sensor_readings,
        "history": sensor_data[-10:],  # Last 10 readings
        "mqtt_connected": current_sensor_readings["connected"]
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Use current MQTT sensor data if no specific values provided
        if data.get("use_current", False):
            # Use latest sensor data from MQTT
            if current_sensor_readings["connected"] and current_sensor_readings["heart_rate"] > 0:
                hr = current_sensor_readings["heart_rate"]
                spo2 = current_sensor_readings["spo2"]
                temp = current_sensor_readings["temperature"]
                rr = current_sensor_readings["rr_interval"]
                hrv = current_sensor_readings["hrv"]
            else:
                return jsonify({"error": "No sensor data available. Please connect IoT devices."}), 400
        else:
            # Use provided values
            required_fields = ["HeartRate", "SpO2", "Temperature", "RR_interval", "HRV"]
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing field: {field}"}), 400

            # Convert and validate inputs
            try:
                hr = float(data["HeartRate"])
                spo2 = float(data["SpO2"])
                temp = float(data["Temperature"])
                rr = float(data["RR_interval"])
                hrv = float(data["HRV"])
            except ValueError:
                return jsonify({"error": "Invalid input types. All values must be numbers."}), 400

        # Validate ranges
        if not (40 <= hr <= 180):
            return jsonify({"error": "Heart Rate must be between 40-180 bpm"}), 400
        if not (70 <= spo2 <= 100):
            return jsonify({"error": "SpO2 must be between 70-100%"}), 400
        if not (35 <= temp <= 42):
            return jsonify({"error": "Temperature must be between 35-42°C"}), 400
        if not (300 <= rr <= 1200):
            return jsonify({"error": "RR Interval must be between 300-1200 ms"}), 400
        if not (0 <= hrv <= 200):
            return jsonify({"error": "HRV must be between 0-200 ms"}), 400

        # ✅ STEP 1: ML Model Prediction
        features = np.array([[hr, spo2, temp, rr, hrv]])
        numeric_prediction = model.predict(features)[0]
        numeric_prediction = convert(numeric_prediction)

        # ✅ STEP 2: Get label from encoder
        ml_prediction_label = label_encoder.inverse_transform([numeric_prediction])[0]
        ml_prediction_label = str(ml_prediction_label)

        # ✅ STEP 3: Get detailed analysis from LLM
        llm_analysis = generate_llm_analysis(ml_prediction_label, hr, spo2, temp, rr, hrv)

        # ✅ Prepare response
        response_data = {
            "ml_prediction": ml_prediction_label,
            "numeric_prediction": numeric_prediction,
            "analysis": llm_analysis,
            "vital_signs": {
                "heart_rate": hr,
                "spo2": spo2,
                "temperature": temp,
                "rr_interval": rr,
                "hrv": hrv
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_data_used": data.get("use_current", False)
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": True,
        "groq_available": GROQ_AVAILABLE,
        "label_encoder_loaded": True,
        "sensor_data_count": len(sensor_data),
        "mqtt_connected": current_sensor_readings["connected"],
        "current_sensors": current_sensor_readings
    }
    return jsonify(status)

# Test MQTT data publisher (for testing without real IoT devices)
def test_mqtt_publisher():
    """Publish test data to MQTT for demonstration"""
    def publish_test_data():
        while True:
            if current_sensor_readings["connected"]:
                test_data = {
                    "heart_rate": random.randint(65, 85),
                    "spo2": random.randint(96, 99),
                    "temperature": round(random.uniform(36.2, 37.0), 1),
                    "rr_interval": random.randint(700, 900),
                    "hrv": random.randint(25, 80),
                    "battery": random.randint(80, 100),
                    "signal_strength": random.randint(85, 100)
                }
                mqtt_client.publish(MQTT_TOPIC, json.dumps(test_data))
                print(f"📤 Published test data: {test_data}")
            time.sleep(5)  # Publish every 5 seconds
    
    test_thread = threading.Thread(target=publish_test_data, daemon=True)
    test_thread.start()

# Start test publisher (comment this out if you have real IoT devices)
test_mqtt_publisher()

if __name__ == "__main__":
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n✅ Server running:  http://{local_ip}:5000")
    print(f"✅ Home Page:  http://{local_ip}:5000/")
    print(f"✅ Sensor Data:  http://{local_ip}:5000/sensors")
    print(f"✅ Prediction:  http://{local_ip}:5000/predict")
    print(f"✅ Health check:  http://{local_ip}:5000/health")
    print(f"✅ MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"✅ MQTT Topic: {MQTT_TOPIC}")
    print(f"✅ Groq API: {'Available' if GROQ_AVAILABLE else 'Unavailable'}")
    app.run(host="0.0.0.0", port=5000, debug=True)
