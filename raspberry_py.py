import cv2
import numpy as np
import base64
import os
import threading
import time
import serial  # For Arduino communication
import socket  # For sending data to GUI
import json    # For formatting data to send
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# --- Load Environment Variables ---
load_dotenv()

# --- MongoDB Configuration ---
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://nyanprak:Samprakash3!@trash.utmo5ml.mongodb.net/?retryWrites=true&w=majority&appName=trash"
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "trash_management_db")
COLLECTION_NAME = "trash_cans"  # or you can use os.getenv("MONGODB_COLLECTION", "trash_cans")
TRASHCAN_ID = os.getenv("TRASHCAN_ID", "default_trashcan")

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGODB_URL)
    mongo_db = mongo_client[DATABASE_NAME]
    mongo_collection = mongo_db[COLLECTION_NAME]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo_collection = None

def store_item(data):
    if mongo_collection is None:
        print("MongoDB collection is not available. Skipping storage.")
        return
    try:
        result = mongo_collection.insert_one(data)
        print(f"Stored item in MongoDB with id: {result.inserted_id}")
    except Exception as e:
        print(f"Error storing item in MongoDB: {e}")

# --- OpenAI Client Configuration ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    print("Proceeding without OpenAI analysis capability.")
    client = None

# --- General Configuration ---
analysis_delay = 1  # Seconds the change must persist
process_every_n_frames = 3  # How often to check for change
cooldown_period = 5  # Seconds after analysis before next trigger possible
min_contour_area = 500  # Minimum pixel area for change detection

# --- Arduino Configuration ---
ARDUINO_PORT = "/dev/ttyACM0"  # Or "/dev/ttyUSB0", etc.
BAUD_RATE = 9600  # Must match the Arduino's Serial.begin rate

# --- GUI Communication Configuration ---
GUI_HOST = 'localhost'
GUI_PORT = 9999  # Make sure this matches the port used in gui.py's server

# --- Global Variables ---
is_processing = False
last_trigger_time = 0
change_first_seen_time = 0.0
ser = None  # Global serial connection object

# --- Background Subtractor ---
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

# --- Helper Functions ---
def encode_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def send_to_gui(data):
    """Connects to the GUI server and sends detection data."""
    global GUI_HOST, GUI_PORT
    try:
        # Serialize data to JSON string
        message = json.dumps(data)
        message_bytes = message.encode('utf-8')  # Encode string to bytes

        print(f"Attempting to send data to GUI at {GUI_HOST}:{GUI_PORT}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)  # Set a timeout for the connection attempt
            sock.connect((GUI_HOST, GUI_PORT))
            sock.sendall(message_bytes)
            print(f"Data sent successfully: {data}")
    except socket.timeout:
        print(f"Error: Connection to GUI server ({GUI_HOST}:{GUI_PORT}) timed out.")
    except socket.error as e:
        print(f"Error: Could not connect or send data to GUI server ({GUI_HOST}:{GUI_PORT}): {e}")
    except json.JSONDecodeError as e:
        print(f"Error encoding data to JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending data to GUI: {e}")

def ask_chatgpt(image_path):
    global client
    if not client:
        print("OpenAI client not available. Skipping analysis.")
        return None

    print(f"→ Analyzing image based on detected change: {image_path}")
    try:
        base64_image = encode_image(image_path)
        if not base64_image:
            return None  # Error during encoding

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a trashcan vision assistant.\n"
                                "Look at this image, which was captured because motion or change was detected.\n"
                                "Determine if the primary changing object is a common piece of trash or recycling.\n"
                                "If it is NOT trash/recycling (e.g., a person, hand entering/leaving, significant lighting change, empty view after object removed), respond ONLY with:\n"
                                "Classification: IGNORE\n"
                                "Smelly: NO\n"
                                "Smell Rating: 0\n"
                                "Volume Estimation: 0 cm^3\n"
                                "Item Name: IGNORE\n\n"
                                "If it IS a piece of trash/recycling:\n"
                                "1. Classify it as RECYCLING or TRASH.\n"
                                "2. Determine if it is typically smelly. Respond with YES or NO.\n"
                                "3. Rate how smelly it is likely to be on a scale of 1 to 10.\n"
                                "4. Estimate the volume of the object in cm cubed by identifying the object and finding the average volume of that kind of object. Respond with <number> cm^3.\n"
                                "5. Determine what the item is. Respond with the item name in ALL CAPS.\n\n"
                                "Respond in the following format EXACTLY:\n"
                                "Classification: <RECYCLING or TRASH or IGNORE>\n"
                                "Smelly: <YES or NO>\n"
                                "Smell Rating: <0 to 10>\n"
                                "Volume Estimation: <number> cm^3\n"
                                "Item Name: <ITEM NAME or IGNORE>"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
        )
        response_text = response.choices[0].message.content.strip()
        print("--- GPT-4 Turbo Raw Response ---")
        print(response_text)
        print("--------------------------")

        lines = response_text.splitlines()
        classification, is_smelly, smell_rating, volume_guess, item_name = "UNKNOWN", "UNKNOWN", -1, -1.0, "UNKNOWN"

        for line in lines:
            try:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "classification":
                    classification = value.upper()
                elif key == "smelly":
                    is_smelly = value.upper()
                elif key == "smell rating":
                    try:
                        smell_rating = int(value)
                    except ValueError:
                        smell_rating = -1
                elif key == "volume estimation":
                    try:
                        volume_str = value.split()[0].replace(',', '')  # Handle commas
                        volume_guess = float(volume_str)
                    except (ValueError, IndexError):
                        volume_guess = -1.0
                elif key == "item name":
                    item_name = value.upper()
            except ValueError:
                print(f"Warning: Could not parse line in GPT response: '{line}'")

        if classification == "IGNORE":
            print("→ GPT-4 Turbo determined the object/change should be ignored.")
            return None

        if classification not in ["RECYCLING", "TRASH"]:
            print(f"Warning: Received unexpected classification '{classification}'. Treating as UNKNOWN.")
            return None

        print("→ Final Classification:", classification)
        print("→ Smelly object?", is_smelly)
        print("→ Smell rating:", smell_rating)
        print("→ Estimated volume (cm^3):", volume_guess)
        print("→ Item:", item_name)

        return classification, is_smelly, smell_rating, volume_guess, item_name

    except Exception as e:
        print(f"An error occurred during OpenAI request or processing: {e}")
        return None

def initialize_serial():
    global ser
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino reset
        if ser.is_open:
            print("Arduino connected successfully.")
            return True
        else:
            print("Failed to open serial port.")
            ser = None
            return False
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        ser = None
        return False
    except Exception as e:
        print(f"An unexpected error occurred during serial initialization: {e}")
        ser = None
        return False

def process_capture(frame_to_process, image_path="item_capture.jpg"):
    """Saves frame, calls analysis, sends command to Arduino, stores result in MongoDB, and sends result to GUI."""
    global is_processing, last_trigger_time, ser
    print(f"Change detected for >{analysis_delay}s. Processing captured frame...")

    classification_result = None
    item_name_result = "UNKNOWN ITEM"  # Default name

    try:
        save_path = image_path
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        save_success = cv2.imwrite(save_path, frame_to_process)
        if not save_success:
            print(f"Error: Failed to save image to {save_path}")
            return

        print(f"Image captured and saved to {save_path}")

        result = ask_chatgpt(save_path)  # Returns tuple or None

        if result:
            classification, is_smelly, smell_rating, volume_guess, item_name = result
            classification_result = classification
            item_name_result = item_name if item_name != "UNKNOWN" else "DETECTED ITEM"

            if classification_result in ["TRASH", "RECYCLING"]:
                item_data = {
                    "trashcan_id": TRASHCAN_ID,
                    "timestamp": datetime.utcnow(),
                    "classification": classification_result,
                    "is_smelly": is_smelly,
                    "smell_rating": smell_rating,
                    "volume_estimation": volume_guess,
                    "item_name": item_name_result,
                    "image_path": save_path
                }
                store_item(item_data)

            if classification_result in ["TRASH", "RECYCLING"]:
                detection_data = {
                    "type": classification_result,
                    "name": item_name_result
                }
                send_to_gui(detection_data)
        else:
            print("Analysis resulted in IGNORE or an error. No command sent to Arduino or GUI.")

    except Exception as e:
        print(f"Error during frame processing or OpenAI analysis step: {e}")

    finally:
        if classification_result in ["TRASH", "RECYCLING"]:
            command = 'T\n' if classification_result == "TRASH" else 'R\n'
            if ser and ser.is_open:
                try:
                    print(f"Sending command '{command.strip()}' to Arduino...")
                    ser.write(command.encode('utf-8'))
                    print("Command sent.")
                except serial.SerialException as e:
                    print(f"Error writing to Arduino: {e}")
                except Exception as e:
                    print(f"Unexpected error during serial write: {e}")
            else:
                print("Cannot send command: Arduino serial port not available.")
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")

def live_feed_and_detect(image_path="item_capture.jpg"):
    """Main camera loop: captures, detects change, triggers processing."""
    global is_processing, last_trigger_time, change_first_seen_time, ser

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera with V4L2 backend, trying default...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if h == 0 or w == 0:
        print(f"Error: Could not get valid frame dimensions ({w}x{h}) from camera.")
        cap.release()
        if ser and ser.is_open:
            ser.close()
        return
    print(f"Camera opened. Resolution: {w}x{h} @ {fps:.2f} FPS")

    print("Allowing time for background learning...")
    initial_bg_frames = 30
    for i in range(initial_bg_frames):
        ret, frame = cap.read()
        if ret:
            _ = backSub.apply(frame)
        else:
            print(f"Warning: Failed to grab frame {i+1}/{initial_bg_frames} during background learning.")
        time.sleep(0.05)

    print(f"Starting monitoring. Analysis delay: {analysis_delay}s, Cooldown: {cooldown_period}s")

    frame_counter = 0
    status_text = "Monitoring"
    last_contour_boxes = []
    display_window_name = "Live Feed - Change Detection"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Camera might have disconnected.")
                time.sleep(1)
                break

            current_frame = frame.copy()
            display_frame = frame
            now = time.time()

            frame_counter += 1
            significant_change_detected_this_interval = False

            if frame_counter % process_every_n_frames == 0:
                fgMask = backSub.apply(current_frame)
                _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)

                contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                significant_change_detected = False
                current_contour_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_contour_area:
                        significant_change_detected = True
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        current_contour_boxes.append((x, y, x+cw, y+ch))
                if significant_change_detected:
                    last_contour_boxes = current_contour_boxes
                significant_change_detected_this_interval = significant_change_detected

                if significant_change_detected_this_interval:
                    if change_first_seen_time == 0.0:
                        print(f"Significant change detected. Starting {analysis_delay}s timer...")
                        change_first_seen_time = now
                        status_text = "Waiting (Change)..."
                    else:
                        elapsed_time = now - change_first_seen_time
                        status_text = f"Waiting (Change) {elapsed_time:.1f}s / {analysis_delay}s"
                        if elapsed_time >= analysis_delay:
                            if not is_processing and (now - last_trigger_time > cooldown_period):
                                status_text = "Analyzing..."
                                print(f"--- Change persisted >{analysis_delay}s. Triggering Analysis ---")
                                is_processing = True
                                change_first_seen_time = 0.0
                                frame_to_analyze = current_frame
                                threading.Thread(target=process_capture, args=(frame_to_analyze, image_path), daemon=True).start()
                            elif is_processing:
                                status_text = "Analyzing..."
                            elif (now - last_trigger_time <= cooldown_period):
                                remaining_cooldown = cooldown_period - (now - last_trigger_time)
                                status_text = f"Cooldown {remaining_cooldown:.1f}s"
                else:
                    if change_first_seen_time != 0.0:
                        print("Change disappeared before analysis delay.")
                        change_first_seen_time = 0.0
                    if is_processing:
                        status_text = "Analyzing..."
                    elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                        remaining_cooldown = cooldown_period - (now - last_trigger_time)
                        status_text = f"Cooldown {remaining_cooldown:.1f}s"
                    else:
                        status_text = "Monitoring"

            if change_first_seen_time != 0.0 or status_text == "Analyzing...":
                for (x1, y1, x2, y2) in last_contour_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(display_frame, f"Status: {status_text}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            try:
                cv2.imshow(display_window_name, display_frame)
            except cv2.error as e:
                print(f"Warning: Could not display window ({display_window_name}): {e}. Check if GUI environment is available.")
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed. Exiting live feed...")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("Cleanup complete.")

if __name__ == "__main__":
    print("Starting SmartBin Detection Script...")

    serial_initialized = initialize_serial()
    if not serial_initialized:
        print("Warning: Proceeding without Arduino communication.")

    live_feed_and_detect("item_capture.jpg")

    print("SmartBin Detection Script finished.")
