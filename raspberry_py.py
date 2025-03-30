import cv2
import numpy as np
import base64
import os
import threading
import time
import serial  # For Arduino communication
import socket  # For sending data to GUI
import json    # For formatting data to send
import traceback # For detailed error reporting
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# --- Load Environment Variables ---
load_dotenv()

# --- MongoDB Configuration ---
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://nyanprak:Samprakash3!@trash.utmo5ml.mongodb.net/?retryWrites=true&w=majority&appName=trash" # Replace with your actual URL or ensure it's in .env
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "trash_management_db")
COLLECTION_NAME = "trash_cans"  # or you can use os.getenv("MONGODB_COLLECTION", "trash_cans")
TRASHCAN_ID = os.getenv("TRASHCAN_ID", "default_trashcan") # Give your Pi a unique ID

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000) # Added timeout
    # The ismaster command is cheap and does not require auth.
    mongo_client.admin.command('ismaster')
    mongo_db = mongo_client[DATABASE_NAME]
    mongo_collection = mongo_db[COLLECTION_NAME]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo_client = None # Ensure client is None if connection fails
    mongo_db = None
    mongo_collection = None

def store_item(data):
    if mongo_collection is None:
        print("MongoDB collection is not available. Skipping storage.")
        return
    try:
        # Make sure timestamp is in a MongoDB-compatible format
        data['timestamp'] = datetime.utcnow()
        result = mongo_collection.insert_one(data)
        print(f"Stored item in MongoDB with id: {result.inserted_id}")
    except Exception as e:
        print(f"Error storing item in MongoDB: {e}")

# --- OpenAI Client Configuration ---
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        client = None
    else:
        client = OpenAI(api_key=openai_api_key)
        # Optional: Add a test call here if needed
        # client.models.list()
        print("OpenAI client initialized.")
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    print("Proceeding without OpenAI analysis capability.")
    client = None

# --- General Configuration ---
analysis_delay = 1.5  # Seconds the change must persist (increased slightly)
process_every_n_frames = 3  # How often to check for change
cooldown_period = 5  # Seconds after analysis before next trigger possible
min_contour_area = 800  # Minimum pixel area for change detection (adjust based on testing)

# --- Arduino Configuration ---
ARDUINO_PORT = "/dev/ttyACM0"  # Or "/dev/ttyUSB0", check with `ls /dev/tty*`
BAUD_RATE = 9600  # Must match the Arduino's Serial.begin rate

# --- GUI Communication Configuration ---
GUI_HOST = 'localhost' # Or the IP address of the machine running gui.py if different
GUI_PORT = 9999  # Make sure this matches the port used in gui.py's server

# --- Global Variables ---
is_processing = False
last_trigger_time = 0
change_first_seen_time = 0.0
ser = None  # Global serial connection object

# --- Background Subtractor ---
# Increased history, lower threshold might be better for indoor stable lighting
backSub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)

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
        # Check for common connection errors
        if e.errno == 111: # Connection refused
             print(f"Error: Connection refused by GUI server at {GUI_HOST}:{GUI_PORT}. Is gui.py running and listening?")
        else:
             print(f"Error: Could not connect or send data to GUI server ({GUI_HOST}:{GUI_PORT}): {e}")
    except json.JSONDecodeError as e:
        print(f"Error encoding data to JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending data to GUI: {e}")

def ask_chatgpt(image_path):
    global client
    if not client:
        print("OpenAI client not available. Skipping analysis.")
        # Return a default 'ignore' structure or None
        return None # Or maybe ("IGNORE", "NO", 0, 0.0, "IGNORE") depending on how process_capture handles None

    print(f"→ Analyzing image based on detected change: {image_path}")
    try:
        base64_image = encode_image(image_path)
        if not base64_image:
            return None  # Error during encoding

        response = client.chat.completions.create(
            model="gpt-4-turbo", # Or "gpt-4o" if available and preferred
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a trashcan vision assistant.\n"
                                "Look at this image, which was captured because motion or change was detected inside a trash bin view.\n"
                                "Determine if the primary changing object is a common piece of trash or recycling.\n"
                                "If it is NOT trash/recycling (e.g., a person's hand entering/leaving frame, significant lighting change, empty view after object removed, non-item object), respond ONLY with:\n"
                                "Classification: IGNORE\n"
                                "Smelly: NO\n"
                                "Smell Rating: 0\n"
                                "Volume Estimation: 0 cm^3\n"
                                "Item Name: IGNORE\n\n"
                                "If it IS a piece of trash/recycling:\n"
                                "1. Classify it as RECYCLING or TRASH based on common disposal rules (assume standard mixed recycling unless obvious).\n"
                                "2. Determine if it is typically smelly (e.g., food waste). Respond with YES or NO.\n"
                                "3. Rate how smelly it is likely to be on a scale of 0 (not smelly) to 10 (very smelly).\n"
                                "4. Estimate the volume of the object in cubic centimeters (cm^3). Identify the object and use an average volume. Respond with <number> cm^3.\n"
                                "5. Determine the specific name of the item (e.g., SODA CAN, APPLE CORE, PAPER CUP). Respond with the item name in ALL CAPS.\n\n"
                                "Respond in the following format EXACTLY, with each field on a new line:\n"
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
            max_tokens=150, # Slightly increased max_tokens
        )
        response_text = response.choices[0].message.content.strip()
        print("--- GPT Raw Response ---")
        print(response_text)
        print("--------------------------")

        # Initialize default values
        classification = "UNKNOWN"
        is_smelly = "NO"
        smell_rating = 0
        volume_guess = 0.0
        item_name = "UNKNOWN"

        parsed_data = {}
        lines = response_text.splitlines()
        expected_keys = ["classification", "smelly", "smell rating", "volume estimation", "item name"]

        for line in lines:
            try:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                parsed_data[key] = value
            except ValueError:
                print(f"Warning: Could not parse line in GPT response: '{line}'")
                continue # Skip this line

        # Safely extract values using the parsed dictionary
        classification = parsed_data.get("classification", "UNKNOWN").upper()
        is_smelly = parsed_data.get("smelly", "NO").upper()
        item_name = parsed_data.get("item name", "UNKNOWN").upper()

        try:
            smell_rating_str = parsed_data.get("smell rating", "0")
            smell_rating = int(smell_rating_str)
            if not (0 <= smell_rating <= 10):
                 print(f"Warning: Smell rating '{smell_rating}' out of range (0-10). Defaulting to 0.")
                 smell_rating = 0
        except (ValueError, TypeError):
            print(f"Warning: Could not parse smell rating '{parsed_data.get('smell rating', 'N/A')}'. Defaulting to 0.")
            smell_rating = 0

        try:
            volume_str_full = parsed_data.get("volume estimation", "0 cm^3")
            # Extract only the number part, handle commas
            volume_str = volume_str_full.split()[0].replace(',', '')
            volume_guess = float(volume_str)
            if volume_guess < 0:
                 print(f"Warning: Negative volume estimation '{volume_guess}'. Defaulting to 0.")
                 volume_guess = 0.0
        except (ValueError, IndexError, TypeError):
            print(f"Warning: Could not parse volume estimation '{parsed_data.get('volume estimation', 'N/A')}'. Defaulting to 0.0.")
            volume_guess = 0.0


        # --- Post-processing and Validation ---
        if classification == "IGNORE":
            print("→ GPT determined the object/change should be ignored.")
            return None # Return None for ignored items

        if classification not in ["RECYCLING", "TRASH"]:
            print(f"Warning: Received unexpected classification '{classification}'. Treating as UNKNOWN.")
            # Decide whether to ignore or treat as default trash/unknown
            return None # Or handle as needed

        # Ensure Item Name is not IGNORE if classification isn't IGNORE
        if item_name == "IGNORE" and classification != "IGNORE":
             print(f"Warning: Classification is '{classification}' but Item Name is 'IGNORE'. Setting Item Name to 'DETECTED ITEM'.")
             item_name = "DETECTED ITEM"

        print("→ Final Classification:", classification)
        print("→ Smelly object?", is_smelly)
        print("→ Smell rating:", smell_rating)
        print("→ Estimated volume (cm^3):", volume_guess)
        print("→ Item:", item_name)

        return classification, is_smelly, smell_rating, volume_guess, item_name

    except Exception as e:
        print(f"An error occurred during OpenAI request or processing: {e}")
        traceback.print_exc() # Print detailed traceback
        return None

def initialize_serial():
    global ser
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino reset after connection
        if ser.is_open:
            ser.flushInput() # Clear any lingering data from buffer
            print("Arduino connected successfully.")
            return True
        else:
            print("Failed to open serial port (is_open is False).")
            ser = None
            return False
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        print("Please check:")
        print(f"  - Is the Arduino plugged into the Raspberry Pi?")
        print(f"  - Is the port '{ARDUINO_PORT}' correct? (Try `ls /dev/tty*`)")
        print(f"  - Does the user running this script have permission? (Try adding user to 'dialout' group: `sudo usermod -a -G dialout $USER`)")
        ser = None
        return False
    except Exception as e:
        print(f"An unexpected error occurred during serial initialization: {e}")
        ser = None
        return False

def process_capture(frame_to_process, image_path="item_capture.jpg"):
    """Saves frame, calls analysis, sends command to Arduino, stores result in MongoDB, and sends result to GUI."""
    global is_processing, last_trigger_time, ser, mongo_collection, TRASHCAN_ID
    print(f"--- Processing captured frame ---")

    classification_result = None
    item_name_result = "UNKNOWN ITEM"  # Default name

    try:
        # --- Save Image ---
        save_path = os.path.abspath(image_path) # Use absolute path
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                print(f"Created directory: {save_dir}")
            except OSError as e:
                print(f"Error: Could not create directory {save_dir}: {e}")
                # Decide how to handle - maybe save to /tmp? For now, error out.
                is_processing = False # Reset flag
                return # Exit processing

        save_success = cv2.imwrite(save_path, frame_to_process)
        if not save_success:
            print(f"Error: Failed to save image to {save_path}")
            is_processing = False # Reset flag
            return # Exit processing
        print(f"Image captured and saved to {save_path}")

        # --- Analyze Image ---
        analysis_result = ask_chatgpt(save_path)  # Returns tuple or None

        if analysis_result:
            classification, is_smelly, smell_rating, volume_guess, item_name = analysis_result
            classification_result = classification # Store for command sending
            item_name_result = item_name if item_name not in ["UNKNOWN", "IGNORE"] else "DETECTED ITEM"

            # --- Store in MongoDB ---
            if classification_result in ["TRASH", "RECYCLING"] and mongo_collection is not None:
                 # Prepare data BEFORE calling store_item
                 item_data = {
                     "trashcan_id": TRASHCAN_ID,
                     # timestamp added inside store_item
                     "classification": classification_result,
                     "is_smelly": is_smelly == "YES", # Store as boolean
                     "smell_rating": smell_rating,
                     "volume_estimation_cm3": volume_guess, # Clearer field name
                     "item_name": item_name_result,
                     "image_path": save_path # Store path for reference
                 }
                 # Run storage in a separate thread to avoid blocking
                 storage_thread = threading.Thread(target=store_item, args=(item_data,), daemon=True)
                 storage_thread.start()

            # --- Send to GUI ---
            if classification_result in ["TRASH", "RECYCLING"]:
                detection_data = {
                    "type": classification_result,
                    "name": item_name_result
                }
                # Run GUI send in a separate thread
                gui_thread = threading.Thread(target=send_to_gui, args=(detection_data,), daemon=True)
                gui_thread.start()

        else:
            print("Analysis resulted in IGNORE or an error. No command sent to Arduino or data stored/sent.")
            classification_result = "IGNORE" # Explicitly set to ignore for clarity

    except Exception as e:
        print(f"Error during frame processing (saving/analysis/db/gui): {e}")
        traceback.print_exc()
        classification_result = "ERROR" # Indicate error occurred

    finally:
        # --- Send Command to Arduino ---
        command = None
        if classification_result == "TRASH":
            command = 'T\n'
        elif classification_result == "RECYCLING":
            command = 'R\n'
        # Only send command if it's Trash or Recycling
        if command and ser and ser.is_open:
            try:
                print(f"Sending command '{command.strip()}' to Arduino...")
                ser.write(command.encode('utf-8'))
                # Optional: Wait for a brief moment or read acknowledgment if Arduino sends one
                # time.sleep(0.1)
                # response = ser.readline().decode('utf-8').strip()
                # print(f"Arduino response: {response}")
                print("Command sent.")
            except serial.SerialException as e:
                print(f"Error writing to Arduino: {e}")
            except Exception as e:
                print(f"Unexpected error during serial write: {e}")
        elif command:
             print("Cannot send command: Arduino serial port not available or not open.")
        else:
             print("No command to send (result was IGNORE, ERROR, or None).")


        # --- Reset State ---
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")
        print("-" * 30) # Separator for clarity in logs


def live_feed_and_detect(image_path="item_capture.jpg"):
    """Main camera loop: captures, detects change, triggers processing."""
    global is_processing, last_trigger_time, change_first_seen_time, ser, backSub

    # --- Camera Initialization ---
    print("Initializing camera...")
    # Try common indices and backends
    camera_indices = [0, 1, 2] # Check 0 first, then others if needed
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY] # Try V4L2 first
    cap = None

    for index in camera_indices:
        for backend in backends:
            print(f"Trying camera index {index} with backend {backend}...")
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"Successfully opened camera {index} with backend {backend}")
                # Try setting properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Optional: Set FPS if needed, but often controlled by hardware/driver
                # cap.set(cv2.CAP_PROP_FPS, 15)
                time.sleep(0.5) # Allow settings to apply
                # Verify settings
                w_check = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_check = int(cap.get(cv2.CAP_PROP_HEIGHT))
                if w_check > 0 and h_check > 0:
                     print(f"Camera resolution set/verified: {w_check}x{h_check}")
                     break # Exit inner loop
                else:
                     print(f"Warning: Could not verify resolution for camera {index} backend {backend}. Got {w_check}x{h_check}. Releasing...")
                     cap.release()
                     cap = None
            else:
                 print(f"Failed to open camera {index} with backend {backend}")
        if cap and cap.isOpened():
             break # Exit outer loop if camera opened successfully

    if not cap or not cap.isOpened():
        print("Error: Cannot open any camera after trying multiple indices/backends.")
        if ser and ser.is_open:
            ser.close()
        return # Exit script

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Get actual FPS

    if h == 0 or w == 0:
        print(f"Error: Could not get valid frame dimensions ({w}x{h}) from camera after opening.")
        cap.release()
        if ser and ser.is_open:
            ser.close()
        return
    print(f"Camera ready. Resolution: {w}x{h} @ {fps:.2f} FPS (reported)")

    # --- Background Learning ---
    print("Allowing time for background learning...")
    initial_bg_frames = max(30, int(fps * 2)) # Learn for ~2 seconds or 30 frames
    for i in range(initial_bg_frames):
        ret, frame = cap.read()
        if ret:
            _ = backSub.apply(frame) # Feed frames to learn background
        else:
            print(f"Warning: Failed to grab frame {i+1}/{initial_bg_frames} during background learning.")
        time.sleep(1.0 / max(fps, 10.0)) # Sleep based on reported FPS, min 10fps rate

    print(f"Background learning complete. Starting monitoring.")
    print(f"Config: Analysis Delay={analysis_delay}s, Cooldown={cooldown_period}s, Min Area={min_contour_area}px")
    print("-" * 30)

    # --- Main Loop Variables ---
    frame_counter = 0
    status_text = "Monitoring"
    last_contour_boxes = []
    display_window_name = "SmartBin Live Feed"
    window_created = False # Flag to track if window was successfully created

    try:
        while True:
            # --- Read Frame ---
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Camera might have disconnected.")
                time.sleep(1) # Wait before trying again or breaking
                # Optional: Add logic to attempt camera re-initialization here
                break # Exit if camera fails persistently

            current_frame = frame.copy() # Keep original for analysis if needed
            display_frame = frame # Use the read frame for display modification
            now = time.time()

            # --- Change Detection Logic (Processed every N frames) ---
            frame_counter += 1
            significant_change_detected_this_interval = False

            if frame_counter % process_every_n_frames == 0:
                # 1. Apply Background Subtraction
                fgMask = backSub.apply(current_frame)

                # 2. Thresholding (Remove shadows and noise)
                # Adjusted threshold value based on backSub's varThreshold potentially
                _, fgMask_thresh = cv2.threshold(fgMask, 240, 255, cv2.THRESH_BINARY)

                # 3. Morphological Operations (Clean up the mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                # Opening removes small noise pixels
                fgMask_cleaned = cv2.morphologyEx(fgMask_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                # Closing fills small holes in objects
                fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

                # 4. Find Contours
                contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 5. Filter Contours by Area
                significant_change_detected = False
                current_contour_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_contour_area:
                        significant_change_detected = True
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        current_contour_boxes.append((x, y, x+cw, y+ch)) # Store bounding box

                # Update state based on detection this interval
                if significant_change_detected:
                    last_contour_boxes = current_contour_boxes # Keep track of where change was
                significant_change_detected_this_interval = significant_change_detected


                # --- State Machine for Triggering Analysis ---
                if significant_change_detected_this_interval:
                    if change_first_seen_time == 0.0: # First time seeing change
                        print(f"Significant change detected. Starting {analysis_delay}s timer...")
                        change_first_seen_time = now
                        status_text = "Waiting (Change)..."
                    else: # Change was already seen, check persistence
                        elapsed_time = now - change_first_seen_time
                        status_text = f"Waiting ({elapsed_time:.1f}s / {analysis_delay}s)"
                        # Check if delay met, not processing, and cooldown passed
                        if elapsed_time >= analysis_delay and not is_processing and (now - last_trigger_time > cooldown_period):
                            status_text = "Analyzing..."
                            print(f"--- Change persisted >{analysis_delay}s. Triggering Analysis ---")
                            is_processing = True
                            change_first_seen_time = 0.0 # Reset timer
                            # Capture the frame exactly when triggering analysis
                            frame_to_analyze = current_frame.copy()
                            # Start processing in a background thread
                            analysis_thread = threading.Thread(target=process_capture, args=(frame_to_analyze, image_path), daemon=True)
                            analysis_thread.start()
                        elif is_processing:
                            # Still processing previous item
                            status_text = "Analyzing..."
                        elif (now - last_trigger_time <= cooldown_period):
                            # Change persisted, but still in cooldown
                            remaining_cooldown = cooldown_period - (now - last_trigger_time)
                            status_text = f"Cooldown ({remaining_cooldown:.1f}s)"
                            # Keep change_first_seen_time active during cooldown if change persists
                else: # No significant change detected in this interval
                    if change_first_seen_time != 0.0:
                        # Change disappeared before timer expired
                        print("Change disappeared before analysis delay.")
                        change_first_seen_time = 0.0 # Reset timer
                    # Update status text based on current state (processing or cooldown)
                    if is_processing:
                        status_text = "Analyzing..."
                    elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                        remaining_cooldown = cooldown_period - (now - last_trigger_time)
                        status_text = f"Cooldown ({remaining_cooldown:.1f}s)"
                    else:
                        status_text = "Monitoring"


            # --- Draw Overlays on Display Frame ---
            # Draw bounding boxes if change is being tracked or analyzed
            if change_first_seen_time != 0.0 or status_text == "Analyzing...":
                for (x1, y1, x2, y2) in last_contour_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box

            # Draw status text
            cv2.putText(display_frame, f"Status: {status_text}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text


            # --- Display Frame ---
            try:
                cv2.imshow(display_window_name, display_frame)
                window_created = True # Mark that we've successfully shown the window at least once
            except cv2.error as e:
                 # Handle environments where display is not possible (like SSH without X forwarding)
                 # Check for specific error messages related to display connection
                 if "cannot be resolved" in str(e) or "display" in str(e).lower() or "NULL window" in str(e):
                     # Print only once or occasionally to avoid flooding console
                     if frame_counter % 100 == 1: # Print every ~100 frames
                         print(f"Warning: Could not display window ({display_window_name}): {e}. Running headlessly.", end='\r')
                     window_created = False # Ensure flag is false if display fails
                 else:
                     # For other cv2 errors, it might be more serious
                     print(f"Error during cv2.imshow: {e}")
                     traceback.print_exc()
                     break # Exit on unexpected cv2 errors


            # --- Handle User Input and Window Close ---
            key = cv2.waitKey(1) & 0xFF # Wait briefly for key press

            # 1. Check for 'q' key press
            if key == ord('q'):
                print("'q' pressed. Exiting live feed...")
                break

            # 2. Check if the window was closed via UI (the 'X' button)
            # Only check if the window was successfully created at some point
            if window_created:
                try:
                    # Check WND_PROP_VISIBLE. Returns >= 1.0 if visible/exists, < 1.0 (usually 0.0) if closed/hidden.
                    if cv2.getWindowProperty(display_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed via UI. Exiting...")
                        break
                except cv2.error:
                    # If getWindowProperty fails after window creation, assume it's gone.
                    print("Window seems to have been destroyed unexpectedly. Exiting...")
                    break

            # Small sleep to prevent 100% CPU usage if waitKey(1) returns immediately
            # Adjust based on desired responsiveness vs CPU load
            time.sleep(0.005) # 5ms sleep

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        print("Cleaning up resources...")
        if cap and cap.isOpened():
            cap.release()
            print("Camera released.")
        # Destroy window only if it might have been created
        if window_created:
             try:
                 cv2.destroyAllWindows()
                 print("OpenCV windows destroyed.")
             except cv2.error as e:
                 print(f"Note: Error destroying cv2 window (might be expected if display failed): {e}")

        if ser and ser.is_open:
            # Optional: Send a signal to Arduino that Pi is shutting down if needed
            # ser.write(b'STOP\n')
            ser.close()
            print("Serial port closed.")
        if mongo_client:
             mongo_client.close()
             print("MongoDB connection closed.")
        print("Cleanup complete. Exiting script.")

# --- Script Entry Point ---
if __name__ == "__main__":
    print("=" * 40)
    print(" Starting SmartBin Detection Script ")
    print("=" * 40)

    # Initialize Serial Connection
    serial_initialized = initialize_serial()
    if not serial_initialized:
        print("Warning: Proceeding without Arduino communication.")
        # Depending on requirements, you might want to exit here if Arduino is essential
        # exit(1)

    # Start the main detection loop
    # Pass the desired path for captured images
    live_feed_and_detect(image_path="/home/pi/smartbin_captures/item_capture.jpg") # Example path

    print("=" * 40)
    print(" SmartBin Detection Script Finished ")
    print("=" * 40)