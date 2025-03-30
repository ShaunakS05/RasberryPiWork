import cv2
import numpy as np
import base64
import os
import threading
import time
import serial # For Arduino communication
import socket # For sending data to GUI
import json   # For formatting data to send
from openai import OpenAI
from dotenv import load_dotenv


# --- Configuration ---
load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    print("Proceeding without OpenAI analysis capability.")
    client = None # Ensure client is None if init fails

analysis_delay = 1  # Seconds the change must persist
process_every_n_frames = 3 # How often to check for change
cooldown_period = 5 # Seconds after analysis before next trigger possible
min_contour_area = 500 # <<< ADJUST: Minimum pixel area for change detection

# --- Arduino Configuration ---
# !!! CHANGE THIS TO YOUR ARDUINO'S PORT !!!
ARDUINO_PORT = "/dev/ttyACM0" # Or "/dev/ttyUSB0", etc.
BAUD_RATE = 9600 # Must match the Arduino's Serial.begin rate

# --- GUI Communication Configuration ---
# Host and port where the gui.py server is listening
# Use 'localhost' if running gui.py on the same RPi
# Use the actual IP address if gui.py is on another computer on the network
GUI_HOST = 'localhost'
GUI_PORT = 9999 # Make sure this matches the port used in gui.py's server

# --- Global Variables ---
is_processing = False
last_trigger_time = 0
change_first_seen_time = 0.0
ser = None # Global serial connection object

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
        message_bytes = message.encode('utf-8') # Encode string to bytes

        print(f"Attempting to send data to GUI at {GUI_HOST}:{GUI_PORT}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0) # Set a timeout for the connection attempt
            sock.connect((GUI_HOST, GUI_PORT))
            sock.sendall(message_bytes)
            print(f"Data sent successfully: {data}")
            # Optional: Receive acknowledgment from GUI? Not strictly necessary here.
            # ack = sock.recv(1024)
            # print(f"Received ack: {ack.decode('utf-8')}")
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
            return None # Error during encoding

        response = client.chat.completions.create(
            # --- FIX: Update the model name here ---
            model="gpt-4o",
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

def process_capture(frame_to_process, image_path="item_capture.jpg"):
    """Saves frame, calls analysis, sends command to Arduino (conditionally), and sends result to GUI."""
    global is_processing, last_trigger_time, ser
    print(f"Change detected for >{analysis_delay}s. Processing captured frame...")

    # Initialize variables that might be used in 'finally'
    classification_result = None
    item_name_result = "UNKNOWN ITEM" # Default name used for GUI and logic check
    item_name_from_openai = "UNKNOWN" # Store the raw name from OpenAI

    try:
        save_path = image_path
        # Ensure directory exists (if image_path includes a path)
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # Save the image
        save_success = cv2.imwrite(save_path, frame_to_process)
        if not save_success:
            print(f"Error: Failed to save image to {save_path}")
            is_processing = False
            last_trigger_time = time.time()
            return # Exit processing

        print(f"Image captured and saved to {save_path}")

        # Call OpenAI for analysis
        result = ask_chatgpt(save_path) # Returns tuple or None

        if result:
            # Unpack the full result tuple
            classification, is_smelly, smell_rating, volume_guess, item_name_from_openai = result
            classification_result = classification # Store the primary classification

            # Determine the name to use (primarily for GUI, but keep raw name too)
            item_name_result = item_name_from_openai if item_name_from_openai not in ["UNKNOWN", "IGNORE"] else "DETECTED ITEM"

            print(f"Debug: OpenAI Classification: '{classification_result}', Raw Item Name: '{item_name_from_openai}', Result Name Used: '{item_name_result}'")

            # --- Send result to GUI ---
            # GUI still gets sent only for TRASH/RECYCLING classifications
            if classification_result in ["TRASH", "RECYCLING"] and item_name_result != "CARDBOARD BOX" :
                detection_data = {
                    "type": classification_result,
                    "name": item_name_result
                }
                send_to_gui(detection_data)
            # --------------------------

        else:
            print("Analysis resulted in IGNORE or an error. Applying IGNORE logic for Arduino.")
            # Set classification specifically to IGNORE if analysis failed or returned None explicitly
            # This ensures the 'finally' block handles it correctly according to the new rule.
            classification_result = "IGNORE"
            item_name_result = "IGNORE" # Keep consistent


    except Exception as e:
        print(f"Error during frame processing or OpenAI analysis step: {e}")
        import traceback
        traceback.print_exc()
        classification_result = "ERROR" # Mark as error state
        item_name_result = "ERROR"

    finally:
        # --- Determine Arduino Command based on new rules ---

        send_command_to_arduino = False
        command_to_send = None

        # Rule 1: Handle TRASH classification
        if classification_result == "TRASH":
             # Optional: Keep cardboard check even for trash? Unlikely to be needed.
             # if item_name_result.upper() != "CARDBOARD BOX":
             send_command_to_arduino = True
             command_to_send = 'T\n'
             # else:
             #     print(f"Item classified as TRASH but name is '{item_name_result}'. Skipping Arduino command.")

        # Rule 2: Handle RECYCLING classification (with cardboard exception)
        elif classification_result == "RECYCLING":
            if item_name_result.upper() != "CARDBOARD BOX":
                send_command_to_arduino = True
                command_to_send = 'R\n'
            else:
                # This is the specific CARDBOARD BOX exception for RECYCLING
                print(f"Item identified as '{item_name_result}' (Recycling). Skipping Arduino command.")
                # command remains None, send_command flag remains False

        # Rule 3: Handle IGNORE classification -> Treat as TRASH for Arduino
        elif classification_result == "IGNORE":
            print(f"Classification is 'IGNORE'. Treating as TRASH for Arduino command.")
            send_command_to_arduino = True
            command_to_send = 'T\n'

        # Rule 4: Handle ERROR or other unexpected states
        else: # Covers ERROR, None (if somehow it gets here), or UNKNOWN
             print(f"Classification is '{classification_result}'. No command sent to Arduino.")


        # --- Execute Sending Logic ---
        if send_command_to_arduino and command_to_send:
            if ser and ser.is_open:
                try:
                    print(f"Sending command '{command_to_send.strip()}' to Arduino...")
                    ser.write(command_to_send.encode('utf-8'))
                    print("Command sent.")
                except serial.SerialException as e:
                    print(f"Error writing to Arduino: {e}")
                except Exception as e:
                    print(f"Unexpected error during serial write: {e}")
            else:
                # Conditions met, but serial port isn't ready
                print(f"Cannot send command '{command_to_send.strip()}': Arduino serial port not available.")
        elif command_to_send is None and classification_result == "RECYCLING" and item_name_result.upper() == "CARDBOARD BOX":
            # This case was handled above by printing the skip message, do nothing here.
            pass
        else:
            # Covers cases where send_command_to_arduino is False for other reasons (ERROR, etc.)
            # The message for these cases was printed in the rule checks above.
            pass
        # ----------------------------
        #
        # Reset processing flag and update trigger time regardless of command sending
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")
        print("-" * 30) # Add a separator line for log readability


def live_feed_and_detect(image_path="item_capture.jpg"):
    """Main camera loop: captures, detects change, triggers processing."""
    global is_processing, last_trigger_time, change_first_seen_time, ser

    # Attempt to open camera using Video4Linux backend explicitly if needed
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera with V4L2 backend, trying default...")
        cap = cv2.VideoCapture(0) # Try default backend
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

    # Set desired properties (best effort, camera might not support all)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 15) # Optionally try setting FPS

    time.sleep(0.5) # Allow camera to stabilize

    # Verify actual dimensions obtained
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Get actual FPS

    if h == 0 or w == 0:
        print(f"Error: Could not get valid frame dimensions ({w}x{h}) from camera.")
        cap.release()
        if ser and ser.is_open: ser.close()
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
                time.sleep(1) # Wait before retrying or exiting
                # Consider attempting to reopen the camera here or breaking the loop
                break # Exit loop if frame grab fails consistently

            current_frame = frame.copy() # Work on a copy
            display_frame = frame # Use original for display initially
            now = time.time()

            frame_counter += 1
            significant_change_detected_this_interval = False

            # Process only every N frames to save resources
            if frame_counter % process_every_n_frames == 0:
                # 1. Background Subtraction
                fgMask = backSub.apply(current_frame)
                # Experiment with threshold if needed (e.g., 128 or 200)
                _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

                # 2. Morphological Operations (Clean up mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                # Opening removes noise, Closing fills gaps
                fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)

                # 3. Find Contours
                contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 4. Filter Contours and Detect Significant Change
                significant_change_detected = False
                current_contour_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_contour_area:
                        significant_change_detected = True
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        current_contour_boxes.append((x, y, x+cw, y+ch))
                # Update last known boxes only if change was detected this interval
                if significant_change_detected:
                    last_contour_boxes = current_contour_boxes
                significant_change_detected_this_interval = significant_change_detected

                # --- Trigger Logic ---
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
                                status_text = "Analyzing..." # Updated status
                                print(f"--- Change persisted >{analysis_delay}s. Triggering Analysis ---")
                                is_processing = True
                                change_first_seen_time = 0.0 # Reset timer
                                frame_to_analyze = current_frame # Use the copy
                                # Start processing in a separate thread to avoid blocking camera feed
                                threading.Thread(target=process_capture, args=(frame_to_analyze, image_path), daemon=True).start()
                            elif is_processing: status_text = "Analyzing..." # Keep status while processing
                            elif (now - last_trigger_time <= cooldown_period):
                                remaining_cooldown = cooldown_period - (now - last_trigger_time)
                                status_text = f"Cooldown {remaining_cooldown:.1f}s" # Show cooldown
                else:
                    # No significant change detected in this interval
                    if change_first_seen_time != 0.0:
                        print("Change disappeared before analysis delay.")
                        change_first_seen_time = 0.0 # Reset timer
                    # Update status based on processing/cooldown if not monitoring
                    if is_processing: status_text = "Analyzing..."
                    elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                        remaining_cooldown = cooldown_period - (now - last_trigger_time)
                        status_text = f"Cooldown {remaining_cooldown:.1f}s"
                    else: status_text = "Monitoring"

            # --- Drawing/Display Logic ---
            # Draw boxes only if change is currently being timed or analyzed
            if change_first_seen_time != 0.0 or status_text == "Analyzing...":
                for (x1, y1, x2, y2) in last_contour_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box

            # Display status text
            cv2.putText(display_frame, f"Status: {status_text}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show the frame (optional, can be disabled for headless operation)
            try:
                cv2.imshow(display_window_name, display_frame)
            except cv2.error as e:
                 # Handle cases where display might fail (e.g., no graphical environment)
                 print(f"Warning: Could not display window ({display_window_name}): {e}. Check if GUI environment is available.")
                 # You might want to break or add a flag to stop trying to show the window after the first failure.
                 pass


            # --- Handle Quit ---
            key = cv2.waitKey(1) & 0xFF # waitKey is crucial for imshow to work
            if key == ord('q'):
                print("'q' pressed. Exiting live feed...")
                break

            # Small sleep to prevent 100% CPU usage, adjust if needed
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("Cleanup complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting SmartBin Detection Script...")

    # Initialize Serial Connection
    serial_initialized = initialize_serial()
    if not serial_initialized:
         print("Warning: Proceeding without Arduino communication.")

    # Start the main detection loop
    # Consider making image path configurable or timestamped
    live_feed_and_detect("item_capture.jpg")

    print("SmartBin Detection Script finished.")
