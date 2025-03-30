import cv2
import numpy as np
import base64
import os
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Configuration ---
# Paths to the DNN model files (adjust if placed elsewhere)
model_proto = "MobileNetSSD_deploy.prototxt.txt"
model_weights = "MobileNetSSD_deploy.caffemodel"
# Confidence threshold for detection
confidence_threshold = 0.4 # Adjust as needed (0.2-0.5 is common)

# Object classes provided by MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# --- Define Target and Ignore Classes ---
# Objects we WANT to classify with OpenAI
TARGET_CLASSES = {"bottle", "cup", "can", "box"} # Add relevant classes if the model supports them (MobileNetSSD is limited)
# Note: MobileNetSSD might not have 'can', 'cup', 'box'. 'bottle' is present.
# You might need a different model for better trash detection or rely on GPT-4V
# identifying these from the image even if the local model doesn't explicitly label them.
# For now, we'll use 'bottle' as an example target. If *any* non-ignored object
# is detected, we can still send it to GPT-4V to decide. Let's refine this:
# Trigger if *any* object *other than* an ignored one is detected.
IGNORE_CLASSES = {"person"} # Objects we DON'T want to classify

# --- Global Variables ---
current_frame = None
is_processing = False # Flag to prevent multiple simultaneous OpenAI calls
object_present_previously = False # State flag for triggering
last_trigger_time = 0
cooldown_period = 5 # Seconds to wait after processing before allowing another trigger

# --- Load the DNN Model ---
try:
    net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
    print("MobileNet SSD model loaded successfully.")
except cv2.error as e:
    print(f"Error loading DNN model: {e}")
    print("Ensure the model files ('MobileNetSSD_deploy.prototxt.txt' and 'MobileNetSSD_deploy.caffemodel') are in the correct directory.")
    exit()

# --- Functions ---

def encode_image(image_path):
    # Function to convert image file to base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_chatgpt(image_path):
    # Function to call the OpenAI API and classify the captured image
    # (Same as your original function, just called from process_capture)
    print(f"→ Analyzing image: {image_path}")
    try:
        imageDecoded = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4-turbo", # Or "gpt-4-vision-preview" if turbo doesn't work well
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
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
            max_tokens=100, # Increased slightly for potentially longer item names/robustness
        )

        response_text = response.choices[0].message.content.strip()
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

        # Here you would add logic to ACT on the classification
        # e.g., control servo motors for sorting, update database, etc.

        return classification, is_smelly, smell_rating, volume_guess, item_name

    except Exception as e:
        print(f"An error occurred during OpenAI request or processing: {e}")
        return None # Indicate error/ignore

def process_capture(frame_to_process, image_path="item_capture.jpg"):
    # Saves the specific frame passed to it and processes it with ask_chatgpt
    global is_processing, last_trigger_time
    print("Processing captured frame...")
    try:
        cv2.imwrite(image_path, frame_to_process)
        print(f"Image captured and saved to {image_path}")
        # Call ask_chatgpt
        result = ask_chatgpt(image_path)
        if result:
            # Optional: do something with the results here if needed
            pass
        else:
            print("Analysis resulted in IGNORE or an error.")

    except Exception as e:
        print(f"Error during frame processing or saving: {e}")
    finally:
        # Ensure processing flag is reset and record time
        is_processing = False
        last_trigger_time = time.time()
        print("Processing finished. Ready for new detection.")


def live_feed_and_detect(image_path="item_capture.jpg"):
    global current_frame, is_processing, object_present_previously, last_trigger_time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # Use V4L2 backend
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Set desired resolution (lower is faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Optional: Lower frame rate if needed
    # cap.set(cv2.CAP_PROP_FPS, 15)

    print("Starting live feed and object detection...")
    print("Looking for non-ignored objects. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.5) # Wait a bit before retrying
            continue

        current_frame = frame.copy() # Keep a copy of the latest frame
        (h, w) = frame.shape[:2]

        # Preprocess the frame for the DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Set the blob as input to the network and perform inference
        net.setInput(blob)
        detections = net.forward()

        object_detected_this_frame = False
        ignore_object_detected = False
        detected_object_frame = None # Store the frame when a relevant object is detected

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])

                # Check if the detected class is in IGNORE_CLASSES
                if idx < len(CLASSES) and CLASSES[idx] in IGNORE_CLASSES:
                   ignore_object_detected = True
                   print(f"Ignoring detected: {CLASSES[idx]}")
                   # Draw box for ignored object (optional visualization)
                   box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                   (startX, startY, endX, endY) = box.astype("int")
                   cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2) # Red box
                   label = f"Ignoring: {CLASSES[idx]} ({confidence:.2f})"
                   cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                   # Don't break here, we want to see if *any* ignored object is present

                # Check if the detected class is *not* background and *not* ignored
                # This means it's a potential target object
                elif idx < len(CLASSES) and CLASSES[idx] != "background": # Removed specific TARGET_CLASSES check
                    object_detected_this_frame = True
                    detected_object_label = CLASSES[idx]
                    print(f"Potential target detected: {detected_object_label}")
                    # Draw box for potential target object (optional visualization)
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2) # Green box
                    label = f"Detected: {CLASSES[idx]} ({confidence:.2f})"
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # We'll trigger based on this detection *unless* an ignored object was also found

        # --- Triggering Logic ---
        # Check if a potential target was found AND no ignored object was detected
        trigger_condition = object_detected_this_frame and not ignore_object_detected

        # Get current time
        now = time.time()

        # If we detect a valid object, and we weren't already processing,
        # and the object wasn't present just before (rising edge),
        # and enough time has passed since the last trigger (cooldown)
        if trigger_condition and not is_processing and not object_present_previously and (now - last_trigger_time > cooldown_period) :
            print("--- Valid Object Detected! Triggering Analysis ---")
            is_processing = True # Set flag immediately
            object_present_previously = True # Mark object as present now
            # Use the frame where the detection occurred
            frame_to_analyze = current_frame # Use the clean frame copy
            # Start processing in a separate thread to keep the feed responsive
            threading.Thread(target=process_capture, args=(frame_to_analyze, image_path)).start()

        # If no valid object is detected currently, reset the 'present' flag
        if not trigger_condition:
            object_present_previously = False

        # --- Display the frame ---
        cv2.imshow("Live Feed - Object Detection", frame)

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
    # Check if model files exist before starting
    if not os.path.exists(model_proto) or not os.path.exists(model_weights):
        print(f"Error: Model files not found.")
        print(f"Please ensure '{model_proto}' and '{model_weights}' are in the script's directory.")
    else:
        live_feed_and_detect("item_capture.jpg") # Pass the desired filename