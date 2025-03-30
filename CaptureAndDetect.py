import cv2
import numpy as np
import base64
import os
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

# --- Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_proto = "MobileNetSSD_deploy.prototxt.txt"
model_weights = "MobileNetSSD_deploy.caffemodel"
confidence_threshold = 0.4
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
IGNORE_CLASSES = {"person"}
# --- New Configuration ---
analysis_delay = 1.5  # <<< Seconds the new object must be present before analysis
process_every_n_frames = 5 # Keep frame skipping
cooldown_period = 5 # Seconds after analysis before next trigger possible

# --- Global Variables ---
current_frame = None
is_processing = False # Flag for active OpenAI call
last_trigger_time = 0 # Timestamp of the last analysis start
new_object_first_seen_time = 0.0 # Timestamp when a new object candidate is first detected

# --- Load DNN Model ---
try:
    net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
    print("MobileNet SSD model loaded successfully.")
except cv2.error as e:
    print(f"Error loading DNN model: {e}")
    # ... (error handling)
    exit()

# --- (Keep encode_image, ask_chatgpt, process_capture functions as they are) ---

# << Functions encode_image, ask_chatgpt, process_capture go here - unchanged >>
# Function to convert image file to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Function to call the OpenAI API and classify the captured image
def ask_chatgpt(image_path):
    print(f"→ Analyzing image: {image_path}")
    try:
        imageDecoded = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4-turbo", # Or "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                # ... (Keep the detailed prompt as before) ...
                                "You are a trashcan vision assistant.\n"
                                "Look at this object and determine if it is a common piece of trash or recycling.\n"
                                "If it is NOT trash/recycling (e.g., a person, hand, background view), respond ONLY with:\n"
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
                                "url": f"data:image/jpeg;base64, {imageDecoded}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
        )

        response_text = response.choices[0].message.content.strip()
        # ... (Keep the response parsing logic as before) ...
        print("--- GPT-4V Raw Response ---")
        print(response_text)
        print("--------------------------")
        lines = response_text.splitlines()

        # Default values
        classification = "UNKNOWN"
        is_smelly = "UNKNOWN"
        smell_rating = -1
        volume_guess = -1.0
        item_name = "UNKNOWN"

        # Robust parsing
        for line in lines:
            if line.startswith("Classification:"):
                classification = line.split(":", 1)[1].strip().upper()
            elif line.startswith("Smelly:"):
                is_smelly = line.split(":", 1)[1].strip().upper()
            elif line.startswith("Smell Rating:"):
                try:
                    smell_rating = int(line.split(":", 1)[1].strip())
                except ValueError:
                    smell_rating = -1
            elif line.startswith("Volume Estimation:"):
                 try:
                    # Extract number, handling potential units like 'cm^3'
                    volume_str = line.split(":", 1)[1].strip().split()[0]
                    volume_guess = float(volume_str.replace(',', '')) # Handle potential commas
                 except (ValueError, IndexError):
                    volume_guess = -1.0
            elif line.startswith("Item Name:"):
                item_name = line.split(":", 1)[1].strip().upper()


        # --- Process the results ---
        if classification == "IGNORE":
            print("→ GPT-4V determined the object should be ignored.")
            return None # Return None to signal ignoring

        print("→ Final Classification:", classification)
        print("→ Smelly object?", is_smelly)
        print("→ Smell rating:", smell_rating)
        print("→ Estimated volume (cm^3):", volume_guess)
        print("→ Item:", item_name)

        # ... (Keep smell rating print logic as before) ...
        if smell_rating >= 7:
            print("Trash contains object that is very smelly")
        elif smell_rating >= 4:
            print("Trash contains object that is somewhat smelly")
        elif smell_rating >= 0 and is_smelly == "NO": # Check NO specifically
             print("Trash is fine for now (object not smelly)")
        elif smell_rating >= 0:
            print("Trash is fine for now") # Low smell rating but YES
        else:
            print("Smell rating could not be determined")


        return classification, is_smelly, smell_rating, volume_guess, item_name

    except Exception as e:
        print(f"An error occurred during OpenAI request or processing: {e}")
        return None # Indicate error/ignore

def process_capture(frame_to_process, image_path="item_capture.jpg"):
    global is_processing, last_trigger_time
    # This function remains largely the same, but the print message is updated
    print(f"Object present for >{analysis_delay}s. Processing captured frame...")
    try:
        save_path = image_path
        cv2.imwrite(save_path, frame_to_process)
        print(f"Image captured and saved to {save_path}")
        result = ask_chatgpt(save_path)
        if result:
            pass # Optional: Handle successful analysis result
        else:
            print("Analysis resulted in IGNORE or an error.")
    except Exception as e:
        print(f"Error during frame processing or saving: {e}")
    finally:
        # IMPORTANT: Reset processing flag and set last trigger time
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")
        # NOTE: new_object_first_seen_time is reset in the main loop's logic

# --- MODIFIED FUNCTION ---
def live_feed_and_detect(image_path="item_capture.jpg"):
    global current_frame, is_processing, last_trigger_time, new_object_first_seen_time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened. Allowing time for adjustments...")
    time.sleep(2.0)

    # --- Background Initialization (Unchanged) ---
    print("Capturing initial background view...")
    initial_object_classes = set()
    initial_frames_to_scan = 15
    initial_detections = Counter()
    for _ in range(initial_frames_to_scan):
        ret, frame = cap.read()
        if not ret: continue
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if idx < len(CLASSES) and CLASSES[idx] != "background" and CLASSES[idx] not in IGNORE_CLASSES:
                    initial_detections[CLASSES[idx]] += 1
        time.sleep(0.05)
    detection_threshold = initial_frames_to_scan // 2
    for obj_class, count in initial_detections.items():
        if count > detection_threshold:
            initial_object_classes.add(obj_class)
    if initial_object_classes:
        print(f"Initial background objects identified: {', '.join(initial_object_classes)}")
    else:
        print("Initial view appears empty or contains only ignored objects.")
    print(f"Starting continuous monitoring. Will analyze new objects present for >{analysis_delay}s...")
    # --- End Background Initialization ---

    frame_counter = 0
    last_detection_boxes = []
    status_text = "Monitoring" # Text to display on screen

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        current_frame = frame.copy()
        display_frame = frame
        now = time.time() # Get current time at the start of the loop

        frame_counter += 1

        # --- Process frame intermittently ---
        if frame_counter % process_every_n_frames == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            detected_new_object_this_frame = False
            ignore_object_detected = False
            current_detection_boxes = []
            new_object_label = "" # Store the label of the potential new object

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "Unknown"
                    color = (0, 255, 255) # Default Yellow

                    if idx < len(CLASSES):
                        detected_class = CLASSES[idx]
                        if detected_class == "background": continue

                        if detected_class in IGNORE_CLASSES:
                            ignore_object_detected = True
                            color = (0, 0, 255)
                            display_label = f"Ignoring: {detected_class}"
                        elif detected_class not in initial_object_classes:
                            # This is a potential new object candidate
                            detected_new_object_this_frame = True
                            new_object_label = detected_class # Store its label
                            color = (0, 255, 0)
                            display_label = f"NEW? {detected_class}" # Mark as potential new
                        else:
                            color = (255, 150, 0)
                            display_label = f"BG: {detected_class}"
                    else: continue

                    current_detection_boxes.append({
                        "box": (startX, startY, endX, endY),
                        "label": f"{display_label} ({confidence:.2f})",
                        "color": color
                    })

            # Update boxes for display immediately
            last_detection_boxes = current_detection_boxes

            # --- New Trigger Logic with Delay ---
            is_new_object_condition_met = detected_new_object_this_frame and not ignore_object_detected

            if is_new_object_condition_met:
                if new_object_first_seen_time == 0.0:
                    # First time seeing this new object candidate
                    print(f"New object candidate ({new_object_label}) detected. Starting {analysis_delay}s timer...")
                    new_object_first_seen_time = now
                    status_text = f"Waiting ({new_object_label})..."
                else:
                    # New object candidate still present, check if delay has passed
                    elapsed_time = now - new_object_first_seen_time
                    status_text = f"Waiting ({new_object_label}) {elapsed_time:.1f}s / {analysis_delay}s"
                    if elapsed_time >= analysis_delay:
                        # Delay passed! Check if ready to analyze
                        if not is_processing and (now - last_trigger_time > cooldown_period):
                            status_text = f"Analyzing ({new_object_label})..."
                            print(f"--- {new_object_label} detected for >{analysis_delay}s. Triggering Analysis ---")
                            is_processing = True # Set processing flag
                            # Reset timer *before* starting thread to prevent immediate re-trigger
                            new_object_first_seen_time = 0.0
                            frame_to_analyze = current_frame
                            threading.Thread(target=process_capture, args=(frame_to_analyze, image_path)).start()
                        elif is_processing:
                             status_text = "Waiting (Analysis in progress)"
                        elif (now - last_trigger_time <= cooldown_period):
                             status_text = f"Waiting (Cooldown {cooldown_period - (now - last_trigger_time):.1f}s)"


            else:
                # No new object detected in this frame (or an ignored one is present)
                if new_object_first_seen_time != 0.0:
                    # Reset the timer if the object disappears before analysis
                    print("New object candidate disappeared before analysis delay.")
                    new_object_first_seen_time = 0.0
                # Update status text based on whether processing or cooldown is active
                if is_processing:
                    status_text = "Waiting (Analysis in progress)"
                elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                     status_text = f"Waiting (Cooldown {cooldown_period - (now - last_trigger_time):.1f}s)"
                else:
                     status_text = "Monitoring"


        # --- Drawing Bounding Boxes and Status (on every frame) ---
        for detection in last_detection_boxes:
            (startX, startY, endX, endY) = detection["box"]
            label = detection["label"]
            color = detection["color"]
            cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(display_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display Status Text
        cv2.putText(display_frame, f"Status: {status_text}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Display the frame ---
        cv2.imshow("Live Feed - Object Detection", display_frame)

        # --- Handle Quit ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting live feed...")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

# --- Main Execution ---
if __name__ == "__main__":
    # ... (model file check) ...
    if not os.path.exists(model_proto) or not os.path.exists(model_weights):
        print(f"Error: Model files not found.")
        print(f"Please ensure '{model_proto}' and '{model_weights}' are in the script's directory.")
    else:
        live_feed_and_detect("item_capture.jpg")