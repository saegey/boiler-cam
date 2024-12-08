import sys
import json
import requests

# Home Assistant configuration
HA_URL = "http://192.168.2.91:8123/api/states"
HA_TOKEN = ENV["HA_TOKEN"]

def update_home_assistant(sensor_data):
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }

    for sensor, data in sensor_data.items():
        if sensor == "status":  # Skip non-sensor data
            continue
        entity_id = f"sensor.boiler_{sensor.replace(' ', '_').lower()}"
        payload = {
            "state": data["value"],
            "attributes": {
                "confidence": data.get("confidence", 1.0)
            }
        }
        response = requests.post(f"{HA_URL}/{entity_id}", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Failed to update {entity_id}: {response.status_code}, {response.text}")
        else:
            print(f"Updated {entity_id}: {data['value']}")

if len(sys.argv) != 2:
    print("Usage: python update_home_assistant.py <sensor_data.json>")
    sys.exit(1)

json_file = sys.argv[1]
try:
    with open(json_file, "r") as f:
        sensor_data = json.load(f)
    update_home_assistant(sensor_data)
except Exception as e:
    print(f"Error processing JSON file: {e}")
    sys.exit(1)
