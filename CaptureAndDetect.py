from openai import OpenAI
import cv2
import base64
import os
from dotenv import load_dotenv
import threading

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variable to hold the current frame
current_frame = None

# Function to capture and save the current frame, then process it with ask_chatgpt
def process_capture(image_path="item.jpg"):
    global current_frame
    if current_frame is not None:
        cv2.imwrite(image_path, current_frame)
        print(f"Image captured and saved to {image_path}")
        classification, is_smelly, smell_rating, volume_guess, item_name = ask_chatgpt(image_path)
        print("→ Final classification:", classification)
    else:
        print("No frame available to capture.")

# Function to convert image file to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Function to call the OpenAI API and classify the captured image
def ask_chatgpt(image_path):
    imageDecoded = encode_image(image_path)
    
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
                            "Look at this object and:\n"
                            "1. Classify it as RECYCLING or TRASH.\n"
                            "2. Determine if it is smelly. Respond with YES or NO.\n"
                            "3. Rate how smelly it is on a scale of 1 to 10.\n"
                            "Respond in the following format exactly:\n"
                            "Classification: <RECYCLING or TRASH>\n"
                            "Smelly: <YES or NO>\n"
                            "Smell Rating: <1 to 10>\n"
                            "4. Estimate the volume of the object in cm cubed by identifying the object and finding the average volume of that kind of object. Respond with <x cm^3>\n"
                            "5. Determine what the item is. This value should return a string in all caps."
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
        max_tokens=50,
    )
    
    response_text = response.choices[0].message.content.strip()
    lines = response_text.splitlines()

    try:
        classification = lines[0].split(":")[1].strip().upper()
    except:
        classification = "UNKNOWN"

    try:
        is_smelly = lines[1].split(":")[1].strip().upper()
    except:
        is_smelly = "UNKNOWN"

    try:
        smell_rating = int(lines[2].split(":")[1].strip())
    except:
        smell_rating = -1  # couldn't parse

    try:
        volume_line = lines[3]
        volume_str = volume_line.split(":")[1].strip().split()[0]
        volume_guess = float(volume_str)
    except:
        volume_guess = -1.0  # couldn't parse

    try:
        item_name = lines[4].split(":")[1].strip().upper()
    except:
        item_name = "UNKNOWN"

    print("→ Sort to:", classification)
    print("→ Smelly object?", is_smelly)
    print("→ Smell rating:", smell_rating)
    print("→ Estimated volume (cm^3):", volume_guess)
    print("→ Item:", item_name)

    if smell_rating >= 7:
        print("Trash contains object that is very smelly")
    elif smell_rating >= 4:
        print("Trash contains object that is somewhat smelly")
    elif smell_rating >= 0:
        print("Trash is fine for now")
    else:
        print("Smell rating could not be determined")

    return classification, is_smelly, smell_rating, volume_guess, item_name

# Function to display a live feed and capture images when 'c' is pressed
def live_feed_and_capture(image_path="item.jpg"):
    global current_frame
    # Open the camera with V4L2 backend
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set a lower resolution for improved performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Live feed started — press 'c' to capture, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_frame = frame.copy()  # Update the global frame
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Process capture in a separate thread to keep the feed live
            threading.Thread(target=process_capture, args=(image_path,)).start()
        elif key == ord('q'):
            print("Exiting live feed")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_feed_and_capture("item.jpg")
