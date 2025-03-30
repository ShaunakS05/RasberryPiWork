import cv2
import numpy as np # Still needed for OpenCV operations
import base64
import os
import threading
import time
import serial # For Arduino communication
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# --- DNN Model Configuration Removed ---
analysis_delay = 1.5  # Seconds the change must persist
process_every_n_frames = 3 # How often to check for change
cooldown_period = 5 # Seconds after analysis before next trigger possible
min_contour_area = 500 # <<< ADJUST: Minimum pixel area for change detection

# --- Arduino Configuration ---
# !!! CHANGE THIS TO YOUR ARDUINO'S PORT !!!
ARDUINO_PORT = "/dev/ttyACM0" # Or "/dev/ttyUSB0", etc.
BAUD_RATE = 9600 # Must match the Arduino's Serial.begin rate

# --- Global Variables ---
current_frame = None
is_processing = False
last_trigger_time = 0
change_first_seen_time = 0.0
ser = None # Global serial connection object

# --- DNN Model Loading Removed ---

# --- Background Subtractor ---
# history=500: Frames used for modeling background
# varThreshold=30: Sensitivity threshold (adjust if needed)
# detectShadows=True: Detect shadows (we filter them out)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

# --- Helper Functions ---

# Function to convert image file to base64 (Unchanged)
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Function to call the OpenAI API (Unchanged, relies on prompt for filtering)
def ask_chatgpt(image_path):
    global client
    if not client:
        print("OpenAI client not available. Skipping analysis.")
        return None

    print(f"→ Analyzing image based on detected change: {image_path}")
    try:
        base64_image = encode_image(image_path)
        if not base64_image:
            return None # Error during encoding

        response = client.chat.completions.create(
            # --- FIX: Update the model name here ---
            model="gpt-4-turbo",
            # ---------------------------------------
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
        print("--- GPT-4 Turbo Raw Response ---") # Updated print message
        print(response_text)
        print("--------------------------")

        lines = response_text.splitlines()
        classification, is_smelly, smell_rating, volume_guess, item_name = "UNKNOWN", "UNKNOWN", -1, -1.0, "UNKNOWN"

        for line in lines:
            try:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "classification": classification = value.upper()
                elif key == "smelly": is_smelly = value.upper()
                elif key == "smell rating":
                    try: smell_rating = int(value)
                    except ValueError: smell_rating = -1
                elif key == "volume estimation":
                    try:
                        volume_str = value.split()[0].replace(',', '') # Handle commas
                        volume_guess = float(volume_str)
                    except (ValueError, IndexError): volume_guess = -1.0
                elif key == "item name": item_name = value.upper()
            except ValueError:
                print(f"Warning: Could not parse line in GPT response: '{line}'")


        if classification == "IGNORE":
            print("→ GPT-4 Turbo determined the object/change should be ignored.")
            return None

        # Validate classification before returning
        if classification not in ["RECYCLING", "TRASH"]:
             print(f"Warning: Received unexpected classification '{classification}'. Treating as UNKNOWN.")
             # Decide how to handle - return None or maybe default to TRASH? Let's return None for now.
             return None


        print("→ Final Classification:", classification)
        print("→ Smelly object?", is_smelly)
        print("→ Smell rating:", smell_rating)
        print("→ Estimated volume (cm^3):", volume_guess)
        print("→ Item:", item_name)

        return classification, is_smelly, smell_rating, volume_guess, item_name

    except Exception as e:
        # Catch potential API errors specifically if possible
        # Example: from openai import APIError, RateLimitError etc.
        print(f"An error occurred during OpenAI request or processing: {e}")
        return None

# Function to Initialize Serial Connection (Unchanged)
def initialize_serial():
    global ser
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino reset
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

# Process Capture: Saves frame, calls GPT, sends command to Arduino (Unchanged from previous version)
def process_capture(frame_to_process, image_path="item_capture.jpg"):
    global is_processing, last_trigger_time, ser
    print(f"Change detected for >{analysis_delay}s. Processing captured frame...")
    classification_result = None
    try:
        save_path = image_path
        cv2.imwrite(save_path, frame_to_process)
        print(f"Image captured and saved to {save_path}")
        result = ask_chatgpt(save_path)
        if result:
            classification, _, _, _, _ = result
            classification_result = classification
        else:
            print("Analysis resulted in IGNORE or an error. No command sent to Arduino.")
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
                except serial.SerialException as e: print(f"Error writing to Arduino: {e}")
                except Exception as e: print(f"Unexpected error during serial write: {e}")
            else: print("Cannot send command: Arduino serial port not available.")
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")

# --- Main Detection Loop (DNN Ignore Check Removed) ---
def live_feed_and_detect(image_path="item_capture.jpg"):
    global current_frame, is_processing, last_trigger_time, change_first_seen_time, ser

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        print(f"Error: Could not get valid frame dimensions ({w}x{h}) from camera.")
        cap.release()
        if ser and ser.is_open: ser.close()
        return
    print(f"Camera resolution set to {w}x{h}")

    print("Camera opened. Allowing time for background learning...")
    initial_bg_frames = 30
    for i in range(initial_bg_frames):
        ret, frame = cap.read()
        if ret: _ = backSub.apply(frame)
        else: print(f"Warning: Failed to grab frame {i+1}/{initial_bg_frames} during background learning.")
        time.sleep(0.05)

    print(f"Starting monitoring. Will analyze significant changes present for >{analysis_delay}s...")

    frame_counter = 0
    status_text = "Monitoring"
    last_contour_boxes = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            current_frame = frame.copy()
            display_frame = frame
            now = time.time()

            frame_counter += 1
            # This flag now just reflects significant change detection result from the last processed frame
            significant_change_detected_in_last_processed_frame = False

            if frame_counter % process_every_n_frames == 0:
                # 1. Apply Background Subtraction
                fgMask = backSub.apply(current_frame)
                _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
                # 2. Clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
                # 3. Find Contours
                contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant_change_detected = False
                current_contour_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_contour_area:
                        significant_change_detected = True
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        current_contour_boxes.append((x, y, x+cw, y+ch))
                last_contour_boxes = current_contour_boxes
                significant_change_detected_in_last_processed_frame = significant_change_detected

                # --- 4. DNN Ignore Check Removed ---

                # --- Trigger Logic (Simplified condition) ---
                # Now based *only* on significant change detection
                if significant_change_detected_in_last_processed_frame:
                    if change_first_seen_time == 0.0:
                        print(f"Significant change detected. Starting {analysis_delay}s timer...")
                        change_first_seen_time = now
                        status_text = "Waiting (Change)..."
                    else:
                        elapsed_time = now - change_first_seen_time
                        status_text = f"Waiting (Change) {elapsed_time:.1f}s / {analysis_delay}s"
                        if elapsed_time >= analysis_delay:
                            if not is_processing and (now - last_trigger_time > cooldown_period):
                                status_text = "Analyzing (Change)..."
                                print(f"--- Change detected for >{analysis_delay}s. Triggering Analysis ---")
                                is_processing = True
                                change_first_seen_time = 0.0
                                frame_to_analyze = current_frame
                                threading.Thread(target=process_capture, args=(frame_to_analyze, image_path), daemon=True).start()
                            elif is_processing: status_text = "Waiting (Analysis in progress)"
                            elif (now - last_trigger_time <= cooldown_period):
                                remaining_cooldown = cooldown_period - (now - last_trigger_time)
                                status_text = f"Waiting (Cooldown {remaining_cooldown:.1f}s)"
                else:
                    # No significant change detected in this processed frame
                    if change_first_seen_time != 0.0:
                        print("Change disappeared before analysis delay.")
                        change_first_seen_time = 0.0
                    # Update status based on processing/cooldown
                    if is_processing: status_text = "Waiting (Analysis in progress)"
                    elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                        remaining_cooldown = cooldown_period - (now - last_trigger_time)
                        status_text = f"Waiting (Cooldown {remaining_cooldown:.1f}s)"
                    else: status_text = "Monitoring"

            # --- Drawing/Display Logic (Unchanged, draws contour boxes) ---
            if status_text.startswith("Waiting (Change)") or status_text.startswith("Analyzing (Change)"):
                for (x1, y1, x2, y2) in last_contour_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box
            if h > 0:
                cv2.putText(display_frame, f"Status: {status_text}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Live Feed - Change Detection", display_frame)

            # --- Handle Quit ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting live feed...")
                break

    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting...")
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("Cleanup complete.")


# --- Main Execution Block (DNN Check Removed) ---
if __name__ == "__main__":
    # Initialize Serial Connection
    serial_initialized = initialize_serial()
    if not serial_initialized:
         print("Warning: Proceeding without Arduino communication.")

    # Start the main detection loop
    live_feed_and_detect("item_capture.jpg") # Specify filename for saved captures