#!/bin/bash
USER="saegey"
source /home/saegey/boiler-cam/venv/bin/activate
# Paths
PHOTO_DIR="/home/$USER/photos"
SCRIPT_DIR="/home/$USER"
LOG_FILE="/home/$USER/logfile.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/magic.py"
HA_UPDATE_SCRIPT="${SCRIPT_DIR}/update_home_assistant.py"

# Take a photo with timestamped filename
timestamp=$(date +%s)
image_file="${PHOTO_DIR}/image-${timestamp}.jpg"
libcamera-still -o "${image_file}"

# Log the photo capture
echo "$(date): Captured image: ${image_file}" >>"${LOG_FILE}"

# Run the Python script on the captured image
echo "$(date): Running Python script: ${PYTHON_SCRIPT}" >>"${LOG_FILE}"

python_output=$("${PYTHON_SCRIPT}" "${image_file}" 2>&1)
python_exit_code=$?

# Log the Python script output
echo "$(date): Python script output: ${python_output}" >>"${LOG_FILE}"

if [[ ${python_exit_code} -ne 0 ]]; then
	echo "$(date): Python script failed with exit code ${python_exit_code}" >>"${LOG_FILE}"
	exit 1
fi

# Parse the output JSON filename from the Python script log
json_file=$(echo "${python_output}" | grep -oP '(?<=System Sensor data written to ).*\.json')
if [[ -z "${json_file}" ]]; then
	echo "$(date): Failed to find JSON output file in Python script log" >>"${LOG_FILE}"
	exit 1
fi

# Run the Home Assistant update script with the JSON file
ha_output=$("${HA_UPDATE_SCRIPT}" "${json_file}" 2>&1)
ha_exit_code=$?

# Log the Home Assistant script output
echo "$(date): Home Assistant script output: ${ha_output}" >>"${LOG_FILE}"

if [[ ${ha_exit_code} -ne 0 ]]; then
	echo "$(date): Home Assistant update script failed with exit code ${ha_exit_code}" >>"${LOG_FILE}"
	exit 1
fi

echo "$(date): Successfully updated Home Assistant with data from ${json_file}" >>"${LOG_FILE}"
