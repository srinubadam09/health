import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "35.154.62.193"
MQTT_PORT = 1883
MQTT_TOPIC = "health/sensors"

def simulate_iot_device(device_id):
    client = mqtt.Client(f"iot_device_{device_id}")
    
    def on_connect(client, userdata, flags, rc):
        print(f"Device {device_id} connected to MQTT broker")
    
    client.on_connect = on_connect
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        while True:
            # Simulate realistic health sensor data
            sensor_data = {
                "device_id": device_id,
                "heart_rate": random.randint(65, 85),
                "spo2": random.randint(96, 99),
                "temperature": round(random.uniform(36.2, 37.0), 1),
                "rr_interval": random.randint(700, 900),
                "hrv": random.randint(25, 80),
                "battery": random.randint(80, 100),
                "signal_strength": random.randint(85, 100),
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to MQTT
            client.publish(MQTT_TOPIC, json.dumps(sensor_data))
            print(f"Device {device_id} published: {sensor_data}")
            
            time.sleep(5)  # Send data every 5 seconds
            
    except Exception as e:
        print(f"Device {device_id} error: {e}")

if __name__ == "__main__":
    # Simulate 2 IoT devices
    import threading
    
    devices = ["patient_monitor_001", "wearable_sensor_002"]
    
    for device_id in devices:
        thread = threading.Thread(target=simulate_iot_device, args=(device_id,))
        thread.daemon = True
        thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping IoT simulators...")