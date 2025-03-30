import cv2
import numpy as np
import base64
import os
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv
# No need for Counter anymore for background init

# --- Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_proto = "MobileNetSSD_deploy.prototxt.txt" # Still needed for ignore check
model_weights = "MobileNetSSD_deploy.caffemodel" # Still needed for ignore check
confidence_threshold = 0.4 # For ignore check
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"] # Still needed for ignore check
IGNORE_CLASSES = {"person"} # Critical to prevent triggering on people
analysis_delay = 1.5  # Seconds the change must persist
process_every_n_frames = 3 # Can likely process more frames now (BG sub is faster)
cooldown_period = 5 # Seconds after analysis
min_contour_area = 500 # <<< ADJUST: Minimum pixel area to consider as significant change (tune this!)

# --- Global Variables ---
current_frame = None
is_processing = False
last_trigger_time = 0
change_first_seen_time = 0.0 # Timestamp when significant change is first detected

# --- Load DNN Model (Only needed for Ignore Check now) ---
try:
    net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
    print("MobileNet SSD model loaded (for ignore check).")
except cv2.error as e:
    print(f"Error loading DNN model: {e}")
    exit()

# --- Background Subtractor ---
# history=500: How many frames used for modeling background
# varThreshold=16: Threshold on the squared Mahalanobis distance to decide if pixel is foreground
# detectShadows=True: Detect and mark shadows (we'll filter them out)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True) # Tune varThreshold if needed

# --- (Keep encode_image, ask_chatgpt, process_capture functions as they are) ---

# << Functions encode_image, ask_chatgpt, process_capture go here - unchanged >>
# Function to convert image file to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Function to call the OpenAI API and classify the captured image
def ask_chatgpt(image_path):
    print(f"→ Analyzing image based on detected change: {image_path}")
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
                                # The prompt remains the same - asking GPT-4V to classify or ignore
                                "You are a trashcan vision assistant.\n"
                                "Look at this image, which was captured because motion or change was detected.\n"
                                "Determine if the primary changing object is a common piece of trash or recycling.\n"
                                "If it is NOT trash/recycling (e.g., a person, hand, significant lighting change, empty view after object removed), respond ONLY with:\n"
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
            print("→ GPT-4V determined the object/change should be ignored.")
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
    print(f"Change detected for >{analysis_delay}s. Processing captured frame...")
    try:
        save_path = image_path
        cv2.imwrite(save_path, frame_to_process)
        print(f"Image captured and saved to {save_path}")
        result = ask_chatgpt(save_path)
        # ... (rest of function unchanged)
        if result:
            pass
        else:
            print("Analysis resulted in IGNORE or an error.")
    except Exception as e:
        print(f"Error during frame processing or saving: {e}")
    finally:
        is_processing = False
        last_trigger_time = time.time()
        print(f"Processing finished. Cooldown active for {cooldown_period}s.")

# --- MODIFIED FUNCTION ---
def live_feed_and_detect(image_path="item_capture.jpg"):
    global current_frame, is_processing, last_trigger_time, change_first_seen_time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    (h, w) = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) # Get actual dimensions


    print("Camera opened. Allowing time for background learning...")
    # Give the background subtractor some initial frames to learn
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            backSub.apply(frame)
        time.sleep(0.05)

    print(f"Starting monitoring. Will analyze significant changes present for >{analysis_delay}s...")

    frame_counter = 0
    status_text = "Monitoring"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        current_frame = frame.copy() # Keep clean copy
        display_frame = frame # Frame to draw on
        now = time.time()

        frame_counter += 1

        # --- Process frame intermittently (BG Subtraction is faster, can use lower N) ---
        if frame_counter % process_every_n_frames == 0:
            # 1. Apply Background Subtraction
            fgMask = backSub.apply(current_frame)

            # Filter out shadows (value 127 in MOG2 default)
            fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1] # Keep only definite foreground

            # 2. Clean up mask (Morphological Operations)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Kernel for morphology
            # Remove small noise (erosion) then enlarge remaining areas (dilation)
            fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
            # Close gaps within objects
            fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)


            # 3. Find Contours (Areas of Change)
            contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            significant_change_detected = False
            contour_boxes = [] # Store boxes of significant contours
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    significant_change_detected = True
                    # Get bounding box for visualization
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    contour_boxes.append((x, y, x+cw, y+ch))
                    # No need to break, check all contours

            # --- Visualization (Optional: Show the mask) ---
            # cv2.imshow("Foreground Mask", fgMask_cleaned)
            # ------------------------------------------------

            # 4. Check for Ignored Objects (using DNN) *only if* change was detected
            ignore_object_detected_in_frame = False
            if significant_change_detected:
                # Preprocess frame for DNN
                blob = cv2.dnn.blobFromImage(cv2.resize(current_frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        idx = int(detections[0, 0, i, 1])
                        if idx < len(CLASSES) and CLASSES[idx] in IGNORE_CLASSES:
                            print(f"Ignoring change due to detected: {CLASSES[idx]}")
                            ignore_object_detected_in_frame = True
                            # Optional: Draw red box around ignored DNN object
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Ignoring: {CLASSES[idx]}", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            break # If we see one ignored object, stop DNN check


            # --- Trigger Logic based on Change Detection & Ignore Check ---
            is_valid_change_condition_met = significant_change_detected and not ignore_object_detected_in_frame

            if is_valid_change_condition_met:
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
                            change_first_seen_time = 0.0 # Reset timer before thread
                            frame_to_analyze = current_frame # Use the clean frame
                            threading.Thread(target=process_capture, args=(frame_to_analyze, image_path)).start()
                        elif is_processing:
                             status_text = "Waiting (Analysis in progress)"
                        elif (now - last_trigger_time <= cooldown_period):
                             status_text = f"Waiting (Cooldown {cooldown_period - (now - last_trigger_time):.1f}s)"

            else:
                # No valid change detected in this frame
                if change_first_seen_time != 0.0:
                    print("Change disappeared or ignored object detected before analysis delay.")
                    change_first_seen_time = 0.0 # Reset timer
                # Update status based on processing/cooldown
                if is_processing:
                    status_text = "Waiting (Analysis in progress)"
                elif (now - last_trigger_time <= cooldown_period) and last_trigger_time != 0:
                     status_text = f"Waiting (Cooldown {cooldown_period - (now - last_trigger_time):.1f}s)"
                else:
                     status_text = "Monitoring"


        # --- Drawing Bounding Boxes for Contours (on every frame) ---
        # Only draw if valid change was detected in the *last processed frame*
        # and the timer might be running or analysis starting
        if is_valid_change_condition_met or status_text.startswith("Waiting (Change)") or status_text.startswith("Analyzing (Change)"):
             for (x1, y1, x2, y2) in contour_boxes:
                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box for change


        # Display Status Text
        cv2.putText(display_frame, f"Status: {status_text}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Display the frame ---
        cv2.imshow("Live Feed - Change Detection", display_frame) # Renamed window

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
    # Only check for DNN model files now, as BG subtractor is built-in
    if not os.path.exists(model_proto) or not os.path.exists(model_weights):
        print(f"Error: DNN Model files for ignore check not found.")
        print(f"Please ensure '{model_proto}' and '{model_weights}' are in the script's directory.")
    else:
        live_feed_and_detect("item_capture.jpg")