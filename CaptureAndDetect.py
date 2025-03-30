import cv2
import numpy as np
import base64
import os
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- (Keep all the previous imports and global variable definitions) ---
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
current_frame = None
is_processing = False
object_present_previously = False
last_trigger_time = 0
cooldown_period = 5
net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
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
        # Save the frame *before* potentially slow analysis
        save_path = image_path # Use the provided path directly
        cv2.imwrite(save_path, frame_to_process)
        print(f"Image captured and saved to {save_path}")

        # Call ask_chatgpt
        result = ask_chatgpt(save_path)
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


# --- MODIFIED FUNCTION ---
def live_feed_and_detect(image_path="item_capture.jpg"):
    global current_frame, is_processing, object_present_previously, last_trigger_time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Optional: Try setting FPS lower IF the camera supports it reliably
    # cap.set(cv2.CAP_PROP_FPS, 15)

    print("Starting live feed and object detection...")
    print("Looking for non-ignored objects. Press 'q' to quit.")

    # --- Frame Skipping Variables ---
    frame_counter = 0
    process_every_n_frames = 5  # <<< ADJUST THIS VALUE (Higher = less CPU, slower detection)
                                 # Start with 5, increase if still laggy (e.g., 10, 15)
                                 # Decrease if detection feels too slow (e.g., 3)

    last_detection_boxes = [] # Store last drawn boxes

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1) # Brief pause before retrying
            continue

        current_frame = frame.copy() # Keep a clean copy
        display_frame = frame # Frame to draw on and show

        frame_counter += 1

        # --- Only process every N frames ---
        if frame_counter % process_every_n_frames == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            object_detected_this_frame = False
            ignore_object_detected = False
            current_detection_boxes = [] # Boxes for *this* processed frame

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "Unknown"
                    color = (0, 255, 255) # Default Yellow

                    if idx < len(CLASSES):
                        label = CLASSES[idx]
                        if label in IGNORE_CLASSES:
                            ignore_object_detected = True
                            color = (0, 0, 255) # Red for ignored
                            display_label = f"Ignoring: {label} ({confidence:.2f})"
                            print(f"Ignoring detected: {label}") # Print only when processing
                        elif label != "background":
                            object_detected_this_frame = True
                            color = (0, 255, 0) # Green for potential target
                            display_label = f"Detected: {label} ({confidence:.2f})"
                            print(f"Potential target detected: {label}") # Print only when processing
                        else:
                            # Skip background detections entirely
                            continue
                    else:
                         # Skip detections with index out of bounds
                         continue

                    # Add box and label info to list for drawing
                    current_detection_boxes.append({
                        "box": (startX, startY, endX, endY),
                        "label": display_label,
                        "color": color
                    })

            # Update the boxes to display
            last_detection_boxes = current_detection_boxes

            # --- Triggering Logic (only checked when processing frames) ---
            trigger_condition = object_detected_this_frame and not ignore_object_detected
            now = time.time()

            if trigger_condition and not is_processing and not object_present_previously and (now - last_trigger_time > cooldown_period):
                print("--- Valid Object Detected! Triggering Analysis ---")
                is_processing = True
                object_present_previously = True
                frame_to_analyze = current_frame # Use the clean frame copy
                threading.Thread(target=process_capture, args=(frame_to_analyze, image_path)).start()

            if not trigger_condition:
                object_present_previously = False # Reset if no valid object found *in this processed frame*

        # --- Drawing Bounding Boxes (Draw last known boxes on *every* frame) ---
        # This makes the display feel more responsive even if detection is skipped
        for detection in last_detection_boxes:
            (startX, startY, endX, endY) = detection["box"]
            label = detection["label"]
            color = detection["color"]
            cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(display_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
    if not os.path.exists(model_proto) or not os.path.exists(model_weights):
        print(f"Error: Model files not found.")
        print(f"Please ensure '{model_proto}' and '{model_weights}' are in the script's directory.")
    else:
        live_feed_and_detect("item_capture.jpg")