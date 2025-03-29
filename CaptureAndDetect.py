from openai import OpenAI
import cv2
import base64
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

#Function to convert image to base64
def capture_image(path = "item.jpg"):
    cap = cv2.VideoCapture(0)
    print("Capturing image Nathan")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(path,frame)
        print(f"Image saved to {path}")
    cap.release()

#Function to convert image to base 64 because apis don't take binary code (jpg)
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

#classifyimage method
def ask_chatgpt(image_path):
    imageDecoded= encode_image(image_path)
    
    response = client.chat.completions.create(
        model = "gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a trashcan vision assistant.\n"
                            "Look at this object and:\n"
                            "1. Classify it as RECYCLING or TRASH.\n"
                            "2. Determine if it is smelly. Respond with YES or NO.\n"
                            "3. Rate how smelly it is on a scale of 1 to 10.\n"
                            "Respond in the following format exactly:\n"
                            "Classification: <RECYCLING or TRASH>\n"
                            "Smelly: <YES or NO>\n"
                            "Smell Rating: <1 to 10>\n"
                            "4. Estimate the volume of the obect in meters cubed. Respond with <x m^3>"
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64, {imageDecoded}"
                        }
                    }
                ]
            }
        ],
        max_tokens = 50,
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

    print("→ Sort to:", classification)
    print("→ Smelly object?", is_smelly)
    print("→ Smell rating:", smell_rating)
    print("→ Estimated volume (m^3):", volume_guess)

    if smell_rating >= 7:
        print("Trash contains object that is very smelly")
    elif smell_rating >= 4:
        print("Trash contains object that is somewhat smelly")
    elif smell_rating >= 0:
        print("Trash is fine for now")
    else:
        print("Smell rating could not be determined")

    return classification, is_smelly, smell_rating, volume_guess

if __name__ == "__main__":
    image_path = "item.jpg"
    capture_image(image_path)
    classification, is_smelly, smell_rating, volume_guess = ask_chatgpt(image_path)
    print("→ Final classification:", classification)
